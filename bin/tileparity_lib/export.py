"""Process 4 (Python side) - collect Output1, Output2a, Output2b and Output3 per sample.

Files are matched by their common prefix ``<sample>_mpp<scale>``. The R script
``tp_export_spe.R`` builds the SpatialExperiment from exactly the same files. The
objects hold the *extracted* tiles; the ``keep`` column marks those that pass every QC
threshold (identical sets with the default ``--min_tissue_extract``). With ``--cluster``,
the Leiden labels of the set ``keep`` (``<prefix>.clusters.tsv.gz``) are added to the tile
table as text columns ``cluster_<view>_leiden_k<k>`` (empty for tiles outside the set).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

from . import __version__, features, lineage, st
from .tiling import read_tiles

log = logging.getLogger(__name__)

# blocks in lbp.h5  ->  name in the exported objects (the channels of one input set are
# joined; different input sets stay separate alternatives). The blocks of uni2h.h5
# (UNI2H_RGB, UNI2H_RGB_HE) are exported as they are.
COMBINED_BLOCKS = {
    "LBP_RGB": ("LBP_R", "LBP_G", "LBP_B"),
    "LBP_RGB_HE": ("LBP_R_HE", "LBP_G_HE", "LBP_B_HE"),
    "LBP_HE": ("LBP_H", "LBP_E"),
    "LBP_CENTER_RGB": ("LBP_CENTER_R", "LBP_CENTER_G", "LBP_CENTER_B"),
    "LBP_CENTER_RGB_HE": ("LBP_CENTER_R_HE", "LBP_CENTER_G_HE", "LBP_CENTER_B_HE"),
    "LBP_CENTER_HE": ("LBP_CENTER_H", "LBP_CENTER_E"),
    "INTENSITY_HE": ("INTENSITY_HE",),
}
LINEAGE_STEPS = ("cluster", "uni2h", "lbp", "st", "prepare")


def discover(directory: str | Path) -> list[str]:
    return sorted(p.name[: -len(".tiles.tsv.gz")] for p in Path(directory).glob("*.tiles.tsv.gz"))


def lineage_files(directory: str | Path, prefix: str) -> list[Path]:
    """Lineage files of the leaf steps of a sample (each contains its own ancestors)."""
    d = Path(directory)
    found = [d / f"{prefix}.{s}.lineage.json" for s in LINEAGE_STEPS if (d / f"{prefix}.{s}.lineage.json").exists()]
    leaves = [f for f in found if not f.name.endswith(".prepare.lineage.json")]
    return leaves or found


def load_sample(directory: str | Path, prefix: str) -> dict:
    d = Path(directory)
    tiles = read_tiles(d / f"{prefix}.tiles.tsv.gz")
    tile_ids = tiles["tile_id"].to_numpy(dtype=object)
    clusters_path = d / f"{prefix}.clusters.tsv.gz"
    if clusters_path.exists():  # --cluster: Leiden labels of the set keep, as text
        from .cluster import export_columns

        labels = pd.read_csv(clusters_path, sep="\t", dtype=str, keep_default_na=False).set_index("tile_id")
        missing = set(tile_ids) - set(labels.index)
        if missing:
            raise ValueError(f"{clusters_path.name}: {len(missing)} extracted tiles have no cluster labels")
        for name, column in export_columns(labels.loc[tile_ids]).items():
            tiles[name] = column.to_numpy(dtype=object)
    prep = json.loads((d / f"{prefix}.prepare.json").read_text())
    blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    provenance: dict = {"prepare": prep}
    channels_path = d / f"{prefix}.channels.json"
    if channels_path.exists():  # input images of the extractors: bands, stain vectors, rgb_he QC
        provenance["channels"] = json.loads(channels_path.read_text())

    def aligned(path: Path, label: str) -> dict:
        ids, blks, attrs = features.read(str(path))
        missing = set(tile_ids) - set(ids)
        if missing:
            raise ValueError(f"{path.name}: {len(missing)} extracted tiles have no {label} features")
        idx = pd.Series(np.arange(len(ids)), index=ids).loc[tile_ids].to_numpy()
        provenance[label] = attrs
        return {name: (X[idx], cols) for name, (X, cols, _a) in blks.items()}

    uni_path = d / f"{prefix}.uni2h.h5"
    if uni_path.exists():
        blocks.update(aligned(uni_path, "uni2h"))
    lbp_path = d / f"{prefix}.lbp.h5"
    if lbp_path.exists():
        raw = aligned(lbp_path, "lbp")
        for out_name, parts in COMBINED_BLOCKS.items():
            if all(p in raw for p in parts):
                blocks[out_name] = (np.hstack([raw[p][0] for p in parts]), np.concatenate([raw[p][1] for p in parts]))

    counts_path = d / f"{prefix}.counts.h5"
    if counts_path.exists():
        Y, ctiles, gene_ids, gene_names = st.read_counts_h5(str(counts_path))
        idx = pd.Series(np.arange(len(ctiles)), index=ctiles).reindex(tile_ids)
        if idx.isna().any():
            raise ValueError(f"{counts_path.name}: {int(idx.isna().sum())} extracted tiles have no counts")
        Y = sparse.csc_matrix(Y)[:, idx.to_numpy(int)]
    else:
        Y, gene_ids, gene_names = sparse.csc_matrix((0, len(tile_ids)), dtype=np.int64), np.array([]), np.array([])

    thumb_path = d / f"{prefix}.thumbnail.png"
    thumb = np.asarray(Image.open(thumb_path).convert("RGB")) if thumb_path.exists() else None
    parents = lineage_files(d, prefix)
    merged = lineage.build("sample", "tileparity", prefix, parameters={}, inputs={}, outputs={}, parents=parents)
    return {"prefix": prefix, "tiles": tiles, "blocks": blocks, "counts": Y, "gene_ids": gene_ids,
            "gene_names": gene_names, "thumbnail": thumb, "provenance": provenance, "lineage_parents": parents,
            "lineage": merged}


def to_anndata(s: dict):
    import anndata as ad

    tiles = s["tiles"].copy()
    tiles.index = tiles["tile_id"].astype(str).to_numpy()
    var = pd.DataFrame({"gene_name": np.asarray(s["gene_names"], dtype=str)}, index=pd.Index(np.asarray(s["gene_ids"], dtype=str)))
    Y = sparse.csr_matrix(s["counts"].T)
    X = Y.astype(np.int32) if st.is_integer(Y) else Y.astype(np.float32)
    obsm = {f"X_{k.lower()}": np.asarray(v[0], dtype=np.float32) for k, v in s["blocks"].items()}
    obsm["spatial"] = tiles[["cx_fullres", "cy_fullres"]].to_numpy(float)
    prep = s["provenance"]["prepare"]
    sample = str(tiles["sample_id"].iloc[0]) if len(tiles) else prep["sample_id"]
    uns = {
        "tileparity": {
            "version": __version__,
            "columns": {f"X_{k.lower()}": list(map(str, v[1])) for k, v in s["blocks"].items()},
            "provenance": json.dumps(s["provenance"], default=str),
            "lineage": json.dumps(s["lineage"], default=str),
            "integer_counts": bool(st.is_integer(Y)),
        }
    }
    if s["thumbnail"] is not None:
        uns["spatial"] = {
            sample: {
                "images": {"lowres": s["thumbnail"]},
                "scalefactors": {
                    "tissue_lowres_scalef": float(prep["thumbnail"]["fullres_to_thumbnail"]),
                    "spot_diameter_fullres": float(prep["tile_px"] / prep["scale_factor"]),
                },
            }
        }
    for col in tiles.columns:
        if tiles[col].dtype == object:
            tiles[col] = tiles[col].fillna("").astype(str)
    return ad.AnnData(X=X, obs=tiles, var=var, obsm=obsm, uns=uns)


def combine_anndata(adatas: list):
    """Combine samples of one scale.

    Samples with expression data are combined on their shared genes. Image-only samples
    are left out of that object (they keep their per-sample file). If fewer than two
    samples have expression data, all samples are combined with image features only.
    """
    import anndata as ad

    adatas = [a for a in adatas if a.n_obs > 0]
    with_genes = [a for a in adatas if a.n_vars > 0]
    if len(with_genes) >= 2:
        if len(with_genes) < len(adatas):
            log.warning("image-only samples are not included in the combined object")
        adatas = with_genes
    elif with_genes:
        log.warning("fewer than two samples have expression data; the combined object keeps image features only")
        adatas = [a[:, []].copy() for a in adatas]
    if len(adatas) < 2:
        return None
    all_cols = list(dict.fromkeys(c for a in adatas for c in a.obs.columns))
    for a in adatas:
        for c in all_cols:
            if c not in a.obs.columns:
                a.obs[c] = ""
    keys = set.intersection(*(set(a.obsm.keys()) for a in adatas))
    for a in adatas:
        for k in list(a.obsm.keys()):
            if k not in keys:
                del a.obsm[k]
    if len({a.X.dtype for a in adatas}) > 1:  # integer and fractional counts: keep the fractions
        for a in adatas:
            a.X = a.X.astype(np.float32)
    combined = ad.concat(adatas, join="inner", merge="same", uns_merge="unique")
    combined.uns["tileparity"] = {"version": __version__,
                                  "samples": sorted(set(combined.obs["sample_id"].astype(str)))}
    return combined
