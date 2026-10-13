"""Readers for spatial-transcriptomics inputs.

Three input layouts are supported, all converted to the same internal form:

* ``spaceranger_hd`` - a Space Ranger Visium HD ``outs/`` folder, its
  ``binned_outputs/`` folder or one ``square_XXXum`` bin folder.
* ``spaceranger`` - a Space Ranger standard Visium ``outs/`` folder.
* ``hest`` - a HEST-1k ``st/<id>.h5ad`` file.

Internal form:

* positions: DataFrame with ``barcode``, ``in_tissue`` (0/1), ``x_fullres``
  (= ``pxl_col_in_fullres``) and ``y_fullres`` (= ``pxl_row_in_fullres``), in pixels
  of the full-resolution image that Space Ranger (or HEST) was given.
* counts: CSC sparse matrix of shape (genes x units) holding raw UMI counts,
  plus unit barcodes, gene ids and gene names.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

log = logging.getLogger(__name__)

FORMATS = ("spaceranger_hd", "spaceranger", "hest")
# Standard Visium geometry: 55 µm spots, 100 µm apart centre to centre.
VISIUM_SPOT_DIAMETER_UM = 55.0
VISIUM_SPOT_PITCH_UM = 100.0
# Space Ranger's "spot_diameter_fullres" is the diameter, in full-resolution pixels, of a
# *theoretical 65 µm spot* (10x Genomics, Space Ranger "Spatial outputs" documentation),
# not of the 55 µm capture spot.
SR_SPOT_DIAMETER_FULLRES_UM = 65.0
_SR_COLUMNS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]


@dataclass
class SpatialPaths:
    fmt: str
    root: Path
    positions: Path
    counts: Path
    scalefactors: Path | None = None
    unit_size_um: float | None = None  # HD bin side or Visium spot diameter
    notes: list[str] = field(default_factory=list)


def resolve(path: str | Path, fmt: str, hd_bin_um: int = 8) -> SpatialPaths:
    """Locate the positions, counts and scale-factor files inside an input path."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown spatial format {fmt!r}; expected one of {FORMATS}")
    root = Path(path)
    if fmt == "hest":
        if not root.is_file():
            raise FileNotFoundError(f"HEST input must be an .h5ad file, got {root}")
        return SpatialPaths(fmt, root, positions=root, counts=root, unit_size_um=VISIUM_SPOT_DIAMETER_UM)

    if fmt == "spaceranger_hd":
        bin_dir = f"square_{int(hd_bin_um):03d}um"
        candidates = [root / "binned_outputs" / bin_dir, root / bin_dir, root]
        base = next((c for c in candidates if (c / "filtered_feature_bc_matrix.h5").exists()), None)
        if base is None:
            raise FileNotFoundError(
                f"no filtered_feature_bc_matrix.h5 found for {bin_dir} under {root} "
                "(expected outs/binned_outputs/square_XXXum/)"
            )
        positions = base / "spatial" / "tissue_positions.parquet"
        if not positions.exists():
            raise FileNotFoundError(f"missing {positions}")
        sf = base / "spatial" / "scalefactors_json.json"
        unit = float(hd_bin_um)
        if sf.exists():
            unit = float(json.loads(sf.read_text()).get("bin_size_um", unit))
        return SpatialPaths(fmt, base, positions, base / "filtered_feature_bc_matrix.h5", sf if sf.exists() else None, unit)

    # standard Visium
    counts = root / "filtered_feature_bc_matrix.h5"
    if not counts.exists():
        raise FileNotFoundError(f"missing {counts} (expected a Space Ranger outs/ folder)")
    positions = root / "spatial" / "tissue_positions.csv"
    if not positions.exists():
        positions = root / "spatial" / "tissue_positions_list.csv"  # Space Ranger < 2.0
    if not positions.exists():
        raise FileNotFoundError(f"no tissue_positions(.csv|_list.csv) under {root / 'spatial'}")
    sf = root / "spatial" / "scalefactors_json.json"
    return SpatialPaths(fmt, root, positions, counts, sf if sf.exists() else None, VISIUM_SPOT_DIAMETER_UM)


def read_positions(sp: SpatialPaths) -> pd.DataFrame:
    """Unit positions in full-resolution image pixels."""
    if sp.fmt == "hest":
        import anndata

        adata = anndata.read_h5ad(sp.positions, backed="r")
        obs = adata.obs
        if {"pxl_col_in_fullres", "pxl_row_in_fullres"} <= set(obs.columns):
            x = obs["pxl_col_in_fullres"].to_numpy(float)
            y = obs["pxl_row_in_fullres"].to_numpy(float)
        else:
            xy = np.asarray(adata.obsm["spatial"], dtype=float)  # (x, y) = (col, row)
            x, y = xy[:, 0], xy[:, 1]
        in_tissue = obs["in_tissue"].to_numpy(int) if "in_tissue" in obs.columns else np.ones(len(obs), int)
        df = pd.DataFrame({"barcode": obs.index.astype(str), "in_tissue": in_tissue, "x_fullres": x, "y_fullres": y})
        adata.file.close()
        return df

    if sp.positions.suffix == ".parquet":
        raw = pd.read_parquet(sp.positions)
    else:
        with open(sp.positions) as fh:
            first = fh.readline()
        header = 0 if first.startswith("barcode") else None
        raw = pd.read_csv(sp.positions, header=header)
        if header is None:
            raw.columns = _SR_COLUMNS
    return pd.DataFrame(
        {
            "barcode": raw["barcode"].astype(str).to_numpy(),
            "in_tissue": raw["in_tissue"].astype(int).to_numpy(),
            "x_fullres": raw["pxl_col_in_fullres"].astype(float).to_numpy(),
            "y_fullres": raw["pxl_row_in_fullres"].astype(float).to_numpy(),
        }
    )


@dataclass
class Counts:
    matrix: sparse.csc_matrix  # genes x units
    barcodes: np.ndarray
    gene_ids: np.ndarray
    gene_names: np.ndarray


def _decode(arr) -> np.ndarray:
    return np.array([a.decode() if isinstance(a, bytes) else str(a) for a in arr], dtype=object)


def read_counts(sp: SpatialPaths) -> Counts:
    """Raw UMI counts as a (genes x units) sparse matrix."""
    if sp.fmt == "hest":
        import anndata

        adata = anndata.read_h5ad(sp.counts)
        x = adata.raw.X if adata.raw is not None else adata.X
        mat = sparse.csc_matrix(x.T)
        if mat.nnz and not np.allclose(mat.data, np.round(mat.data)):
            log.warning("HEST matrix is not integer-valued; it may be normalised rather than raw counts")
        var = adata.raw.var if adata.raw is not None else adata.var
        names = var.index.astype(str).to_numpy(dtype=object)
        ids = var["gene_ids"].astype(str).to_numpy(dtype=object) if "gene_ids" in var.columns else names
        return Counts(mat, adata.obs_names.astype(str).to_numpy(dtype=object), ids, names)

    with h5py.File(sp.counts, "r") as h5:
        m = h5["matrix"]
        shape = tuple(int(v) for v in m["shape"][:])
        mat = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        barcodes = _decode(m["barcodes"][:])
        ids = _decode(m["features"]["id"][:])
        names = _decode(m["features"]["name"][:])
        ftype = _decode(m["features"]["feature_type"][:]) if "feature_type" in m["features"] else None
    if ftype is not None:
        keep = ftype == "Gene Expression"
        if not keep.all():
            mat, ids, names = mat[keep, :], ids[keep], names[keep]
    return Counts(sparse.csc_matrix(mat), barcodes, ids, names)


def mpp_from_scalefactors(sp: SpatialPaths) -> tuple[float | None, str]:
    """Pixel size implied by Space Ranger's scale factors, and how it was obtained."""
    if sp.scalefactors is None:
        return None, "none"
    sf = json.loads(Path(sp.scalefactors).read_text())
    if sf.get("microns_per_pixel"):
        return float(sf["microns_per_pixel"]), "scalefactors:microns_per_pixel"
    diameter = sf.get("spot_diameter_fullres")
    if diameter:
        if sp.fmt == "spaceranger_hd" and sf.get("bin_size_um"):
            return float(sf["bin_size_um"]) / float(diameter), "scalefactors:bin_size_um/spot_diameter_fullres"
        if sp.fmt == "spaceranger":
            return SR_SPOT_DIAMETER_FULLRES_UM / float(diameter), "scalefactors:65um/spot_diameter_fullres"
    return None, "none"


def unit_pitch_um(sp: SpatialPaths) -> float | None:
    """Known centre-to-centre distance (µm) between neighbouring units, or None.

    100 µm for standard Visium spots, one bin size for Visium HD bins. Not used for HEST
    inputs: HEST-1k also contains older arrays with other spacings."""
    if sp.fmt == "spaceranger":
        return VISIUM_SPOT_PITCH_UM
    if sp.fmt == "spaceranger_hd":
        return sp.unit_size_um
    return None


def mpp_from_pitch(positions: pd.DataFrame, pitch_um: float, max_units: int = 50000) -> float | None:
    """Pixel size implied by the spacing of the units in the full-resolution image.

    The median distance from each unit to its nearest neighbour is the known pitch
    (``pitch_um``) expressed in pixels, so this does not depend on any metadata. Large
    Visium HD inputs are reduced to the ``max_units`` units nearest to the array centre,
    which keeps the lattice complete and the cost bounded."""
    from scipy.spatial import cKDTree

    xy = positions[["x_fullres", "y_fullres"]].to_numpy(float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 3:
        return None
    if len(xy) > max_units:
        d = np.abs(xy - np.median(xy, axis=0)).max(axis=1)  # square window around the centre
        xy = xy[d <= np.partition(d, max_units)[max_units]]
    dist, _ = cKDTree(xy).query(xy, k=2)
    nn = dist[:, 1]
    nn = nn[nn > 0]
    if len(nn) == 0:
        return None
    return float(pitch_um) / float(np.median(nn))
