"""Process 4 (Python side) - collect Output1, Output2a, Output2b and Output3 per sample.

Files are matched by their common prefix ``<sample>_mpp<scale>``. The R script
``tp_export_spe.R`` builds the SpatialExperiment from exactly the same files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

from . import __version__, features, st
from .tiling import read_tiles

log = logging.getLogger(__name__)

# feature block in lbp.h5 / uni2h.h5  ->  name in the exported objects
LBP_COMBINED = {"LBP_HE": ("LBP_H", "LBP_E"), "LBP_GREY": ("LBP_GREY",), "INTENSITY_HE": ("INTENSITY_HE",)}


def discover(directory: str | Path) -> list[str]:
    return sorted(p.name[: -len(".tiles.tsv.gz")] for p in Path(directory).glob("*.tiles.tsv.gz"))


def load_sample(directory: str | Path, prefix: str) -> dict:
    d = Path(directory)
    tiles = read_tiles(d / f"{prefix}.tiles.tsv.gz")
    tile_ids = tiles["tile_id"].to_numpy(dtype=object)
    order = pd.Series(np.arange(len(tile_ids)), index=tile_ids)
    prep = json.loads((d / f"{prefix}.prepare.json").read_text())
    blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    provenance: dict = {"prepare": prep}

    def aligned(path: Path, label: str) -> dict:
        ids, blks, attrs = features.read(str(path))
        missing = set(tile_ids) - set(ids)
        if missing:
            raise ValueError(f"{path.name}: {len(missing)} kept tiles have no {label} features")
        idx = pd.Series(np.arange(len(ids)), index=ids).loc[tile_ids].to_numpy()
        provenance[label] = attrs
        return {name: (X[idx], cols) for name, (X, cols, _a) in blks.items()}

    uni_path = d / f"{prefix}.uni2h.h5"
    if uni_path.exists():
        blocks.update(aligned(uni_path, "uni2h"))
    lbp_path = d / f"{prefix}.lbp.h5"
    if lbp_path.exists():
        raw = aligned(lbp_path, "lbp")
        for out_name, parts in LBP_COMBINED.items():
            if all(p in raw for p in parts):
                blocks[out_name] = (np.hstack([raw[p][0] for p in parts]), np.concatenate([raw[p][1] for p in parts]))

    counts_path = d / f"{prefix}.counts.h5"
    if counts_path.exists():
        Y, ctiles, gene_ids, gene_names = st.read_counts_h5(str(counts_path))
        idx = pd.Series(np.arange(len(ctiles)), index=ctiles).reindex(tile_ids)
        if idx.isna().any():
            raise ValueError(f"{counts_path.name}: {int(idx.isna().sum())} kept tiles have no counts")
        Y = sparse.csc_matrix(Y)[:, idx.to_numpy(int)]
    else:
        Y, gene_ids, gene_names = sparse.csc_matrix((0, len(tile_ids)), dtype=np.int64), np.array([]), np.array([])

    thumb_path = d / f"{prefix}.thumbnail.png"
    thumb = np.asarray(Image.open(thumb_path).convert("RGB")) if thumb_path.exists() else None
    return {"prefix": prefix, "tiles": tiles, "blocks": blocks, "counts": Y, "gene_ids": gene_ids,
            "gene_names": gene_names, "thumbnail": thumb, "provenance": provenance}


def to_anndata(s: dict):
    import anndata as ad

    tiles = s["tiles"].copy()
    tiles.index = tiles["tile_id"].astype(str).to_numpy()
    var = pd.DataFrame({"gene_name": np.asarray(s["gene_names"], dtype=str)}, index=pd.Index(np.asarray(s["gene_ids"], dtype=str)))
    X = sparse.csr_matrix(s["counts"].T.astype(np.int32))
    obsm = {f"X_{k.lower()}": np.asarray(v[0], dtype=np.float32) for k, v in s["blocks"].items()}
    obsm["spatial"] = tiles[["cx_fullres", "cy_fullres"]].to_numpy(float)
    prep = s["provenance"]["prepare"]
    sample = str(tiles["sample_id"].iloc[0]) if len(tiles) else prep["sample_id"]
    uns = {
        "tileparity": {
            "version": __version__,
            "columns": {f"X_{k.lower()}": list(map(str, v[1])) for k, v in s["blocks"].items()},
            "provenance": json.dumps(s["provenance"], default=str),
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
    return ad.concat(adatas, join="inner", merge="same", uns_merge="unique")
