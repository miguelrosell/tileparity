"""Tile geometry.

Coordinates follow the usual image convention: x = column, y = row, pixel ``i``
covers the interval [i, i + 1). A tile is a square of ``tile_px`` pixels in the
*resampled* image (the image at the target µm/px), described by its top-left
corner (x0, y0).

* ``grid`` tiles form a regular lattice anchored at (0, 0) of the resampled image:
  tile (row, col) covers x in [col * tile_px, (col + 1) * tile_px) and
  y in [row * tile_px, (row + 1) * tile_px). This is exactly the lattice that
  fastLBP's own patch grid would use, so a per-tile LBP histogram and a per-tile
  UNI2-h embedding always describe the same pixels.
* ``spot`` tiles are centred on standard Visium spots.

Spatial-transcriptomics units (Visium HD bins or Visium spots) are assigned to the
tile that contains their centre, so every unit is counted at most once.
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


def grid_tile_ids(sample_id: str, tag: str, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return np.array([f"{sample_id}_mpp{tag}_r{r:04d}_c{c:04d}" for r, c in zip(row, col)], dtype=object)


def spot_tile_ids(sample_id: str, tag: str, barcodes: np.ndarray) -> np.ndarray:
    return np.array([f"{sample_id}_mpp{tag}_{b}" for b in barcodes], dtype=object)


NUMERIC_COLUMNS = ("scale_mpp", "tile_px", "tile_um", "row", "col", "x0_px", "y0_px", "cx_px", "cy_px",
                   "cx_fullres", "cy_fullres", "cx_um", "cy_um", "tissue_fraction", "st_coverage",
                   "n_st_units", "lapvar", "pen_fraction", "dark_fraction")
BOOL_COLUMNS = ("inside_image", "blur_flag", "pen_flag", "dark_flag", "keep")
INT_COLUMNS = ("tile_px", "row", "col", "x0_px", "y0_px", "n_st_units")


def read_tiles(path, keep_only: bool = True) -> pd.DataFrame:
    """Read a tile table with explicit types: text columns (ids, metadata, qc_fail) stay
    text, empty numeric cells become NaN, boolean columns become bool."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for c in df.columns:
        if c in BOOL_COLUMNS:
            df[c] = df[c].str.lower().eq("true")
        elif c in NUMERIC_COLUMNS:
            df[c] = pd.to_numeric(df[c].replace("", np.nan))
            if c in INT_COLUMNS:
                df[c] = df[c].astype("int64")
    if keep_only:
        df = df[df["keep"]].reset_index(drop=True)
    return df
