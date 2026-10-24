#!/usr/bin/env python3
"""How much Visium capture area does a grid tile hold?

Reproduces the numbers in docs/design.md ("Tiles"). Standard Visium spots are 55 um discs
on a hexagonal lattice with 100 um between centres. For grid tiles of 64, 112 and 224 um
(224 px at 0.2857, 0.5 and 1.0 um/px), this lays the lattice over the grid and reports,
for tiles fully inside the lattice: the spots' worth of area per tile (column
st_units_equiv; the expected value is the tile area divided by the area of one lattice
cell, 100^2 * sqrt(3) / 2 um^2), its minimum and 10th percentile, the share of tiles below
--min_spot_overlap 0.25 (not extracted), and the share of the tile's area that is measured.

    python scripts/visium_grid_coverage.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from tileparity_lib.tiling import disc_tile_weights  # noqa: E402

PITCH_UM, SPOT_UM, MIN_SPOT_OVERLAP = 100.0, 55.0, 0.25


def main() -> None:
    n = 70
    r, c = np.divmod(np.arange(n * n), n)
    xs = c * PITCH_UM + np.where(r % 2 == 1, PITCH_UM / 2, 0.0) + 200
    ys = r * PITCH_UM * np.sqrt(3) / 2 + 200
    cell_area = PITCH_UM**2 * np.sqrt(3) / 2
    for tile_um in (64, 112, 224):
        T = tile_um  # 1 px = 1 um here
        n_cols, n_rows = int((xs.max() + 300) // T), int((ys.max() + 300) // T)
        w = disc_tile_weights(xs, ys, SPOT_UM / 2, T, n_rows, n_cols)
        equiv = np.zeros((n_rows, n_cols))
        np.add.at(equiv, (w["row"].to_numpy(), w["col"].to_numpy()), w["weight"].to_numpy())
        inner = equiv[int(400 // T):int((ys.max() - 200) // T), int(400 // T):int((xs.max() - 200) // T)].ravel()
        measured = inner.mean() * np.pi * (SPOT_UM / 2) ** 2 / tile_um**2
        print(f"tile {tile_um:3d} um: spots' worth per tile mean {inner.mean():.2f} (expected {tile_um**2 / cell_area:.2f}), "
              f"min {inner.min():.2f}, 10th percentile {np.percentile(inner, 10):.2f}; "
              f"below {MIN_SPOT_OVERLAP}: {100 * np.mean(inner < MIN_SPOT_OVERLAP):.0f}% of tiles; "
              f"area measured {100 * measured:.0f}%")


if __name__ == "__main__":
    main()
