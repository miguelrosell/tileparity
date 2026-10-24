"""Tile geometry.

Coordinates follow the usual image convention: x = column, y = row, pixel ``i``
covers the interval [i, i + 1). A tile is a square of ``tile_px`` pixels in the
*resampled* image (the image at the target µm/px), described by its top-left
corner (x0, y0).

* ``grid`` tiles form a regular lattice anchored at (0, 0) of the resampled image:
  tile (row, col) covers x in [col * tile_px, (col + 1) * tile_px) and
  y in [row * tile_px, (row + 1) * tile_px). This is exactly the lattice that
  fastLBP's own patch grid would use, so a per-tile LBP histogram and a per-tile
  UNI2-h embedding always describe the same pixels. Grid tiles never overlap.
* ``spot`` tiles are centred on standard Visium spots (optional; neighbouring spot
  tiles overlap when the tile is larger than the 100 µm spot spacing).

Spatial-transcriptomics units are related to grid tiles in two ways:

* Visium HD bins (8 µm squares) are assigned to the tile that contains their centre,
  so every bin is counted at most once (weight 1).
* Visium spots (55 µm discs) usually straddle tile borders. Each spot gets one weight
  per tile it touches: the fraction of the disc's area inside that tile
  (:func:`disc_tile_weights`). Process 3 then splits the spot's molecules between those
  tiles in these proportions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def scale_tag(mpp: float) -> str:
    """Text label for a target pixel size, e.g. 0.5 -> '0.5000'."""
    return f"{mpp:.4f}"


def grid_tiles(width: int, height: int, tile_px: int) -> pd.DataFrame:
    """All complete tiles of the lattice anchored at (0, 0)."""
    n_rows, n_cols = height // tile_px, width // tile_px
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    return pd.DataFrame(
        {
            "row": rows.astype(np.int64),
            "col": cols.astype(np.int64),
            "x0_px": (cols * tile_px).astype(np.int64),
            "y0_px": (rows * tile_px).astype(np.int64),
        }
    )


def spot_tiles(cx: np.ndarray, cy: np.ndarray, tile_px: int, width: int, height: int) -> pd.DataFrame:
    """Tiles centred (to the nearest pixel) on the given centres in resampled pixels."""
    x0 = np.rint(np.asarray(cx, float) - tile_px / 2).astype(np.int64)
    y0 = np.rint(np.asarray(cy, float) - tile_px / 2).astype(np.int64)
    inside = (x0 >= 0) & (y0 >= 0) & (x0 + tile_px <= width) & (y0 + tile_px <= height)
    return pd.DataFrame({"row": -1, "col": -1, "x0_px": x0, "y0_px": y0, "inside_image": inside})


def assign_to_grid(x: np.ndarray, y: np.ndarray, tile_px: int, n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid (row, col) of the tile containing each point; -1 when outside the grid."""
    col = np.floor(np.asarray(x, float) / tile_px).astype(np.int64)
    row = np.floor(np.asarray(y, float) / tile_px).astype(np.int64)
    outside = (col < 0) | (row < 0) | (col >= n_cols) | (row >= n_rows)
    col[outside] = -1
    row[outside] = -1
    return row, col


def _unit_disc_points(n: int = 80) -> np.ndarray:
    """Centres of an n x n grid of equal cells over [-1, 1]^2 that fall inside the unit
    disc; each point stands for the same small area of the disc."""
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    u, v = np.meshgrid(c, c)
    inside = u * u + v * v <= 1.0
    return np.stack([u[inside], v[inside]], axis=1)


def disc_tile_weights(cx: np.ndarray, cy: np.ndarray, radius: float, tile_px: int, n_rows: int, n_cols: int,
                      samples: int = 80, chunk: int = 2000) -> pd.DataFrame:
    """Fraction of each disc's area that falls in each grid tile.

    Discs have centres (cx, cy) and ``radius`` in resampled pixels. The disc is sampled on
    a regular grid of about (pi / 4) * samples^2 points (about 5000 for the default), so the
    fractions are accurate to a few tenths of a percent. Area outside the lattice (the
    incomplete last row/column of the image) is not assigned, so the weights of a disc sum
    to less than 1 there.

    Returns one row per (unit, tile) with columns ``unit`` (index into cx), ``row``, ``col``
    and ``weight``."""
    pts = _unit_disc_points(samples)
    n_pts = len(pts)
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)
    out = []
    for start in range(0, len(cx), chunk):
        sl = slice(start, min(start + chunk, len(cx)))
        px = cx[sl, None] + radius * pts[None, :, 0]
        py = cy[sl, None] + radius * pts[None, :, 1]
        col = np.floor(px / tile_px).astype(np.int64)
        row = np.floor(py / tile_px).astype(np.int64)
        unit = np.broadcast_to(np.arange(sl.start, sl.stop)[:, None], col.shape)
        ok = (col >= 0) & (row >= 0) & (col < n_cols) & (row < n_rows)
        key = (unit[ok] * n_rows + row[ok]) * n_cols + col[ok]
        keys, counts = np.unique(key, return_counts=True)
        out.append(pd.DataFrame({"unit": keys // (n_rows * n_cols), "row": (keys // n_cols) % n_rows,
                                 "col": keys % n_cols, "weight": counts / n_pts}))
    if not out:
        return pd.DataFrame({"unit": pd.Series(dtype=np.int64), "row": pd.Series(dtype=np.int64),
                             "col": pd.Series(dtype=np.int64), "weight": pd.Series(dtype=float)})
    return pd.concat(out, ignore_index=True)


def grid_tile_ids(sample_id: str, tag: str, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return np.array([f"{sample_id}_mpp{tag}_r{r:04d}_c{c:04d}" for r, c in zip(row, col)], dtype=object)


def spot_tile_ids(sample_id: str, tag: str, barcodes: np.ndarray) -> np.ndarray:
    return np.array([f"{sample_id}_mpp{tag}_{b}" for b in barcodes], dtype=object)


NUMERIC_COLUMNS = ("scale_mpp", "tile_px", "tile_um", "row", "col", "x0_px", "y0_px", "cx_px", "cy_px",
                   "cx_fullres", "cy_fullres", "cx_um", "cy_um", "tissue_fraction", "st_coverage",
                   "n_st_units", "st_units_equiv", "lapvar", "pen_fraction", "dark_fraction")
BOOL_COLUMNS = ("inside_image", "blur_flag", "pen_flag", "dark_flag", "extract", "keep")
INT_COLUMNS = ("tile_px", "row", "col", "x0_px", "y0_px", "n_st_units")


def read_tiles(path, subset: str | None = "extract") -> pd.DataFrame:
    """Read a tile table with explicit types: text columns (ids, metadata, qc_fail) stay
    text, empty numeric cells become NaN, boolean columns become bool.

    ``subset``: ``"extract"`` (default) returns the tiles whose features are computed,
    ``"keep"`` the tiles that pass every QC threshold (a subset of those), ``None`` every
    candidate tile. Tables without an ``extract`` column (tileparity < 0.2) use ``keep``."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for c in df.columns:
        if c in BOOL_COLUMNS:
            df[c] = df[c].str.lower().eq("true")
        elif c in NUMERIC_COLUMNS:
            df[c] = pd.to_numeric(df[c].replace("", np.nan))
            if c in INT_COLUMNS:
                df[c] = df[c].astype("int64")
    if subset is not None:
        if subset not in ("extract", "keep"):
            raise ValueError(f"subset must be 'extract', 'keep' or None, not {subset!r}")
        column = subset if subset in df.columns else "keep"
        df = df[df[column]].reset_index(drop=True)
    return df
