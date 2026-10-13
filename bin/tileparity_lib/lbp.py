"""Process 2b - LBP texture features per tile, computed with fastLBP's LBP kernel.

For every pixel c and every radius R (with P sampling points):

1. P points are placed on a circle of radius R around c; their grey values g_p are
   obtained by bilinear interpolation.
2. bit_p = 1 if g_p >= g_c, else 0.
3. U = number of 0/1 changes between consecutive bits. If U <= 2 (one arc of ones,
   a "uniform" pattern) the code is the number of ones (0..P); otherwise P + 1.
   There are P + 2 possible codes and the code does not change if the tissue rotates.

For every tile the codes of its tile_px x tile_px pixels are counted, giving one
histogram of P + 2 bins per radius and channel; histograms are concatenated.

The neighbours of pixels near the tile border are the real neighbouring pixels of the
slide (the tile is read with a margin of R_max + 1 px), never an artificial black
border. This module calls ``fastlbp.lbp.uniform_lbp_uint8_padded_absolute``, the same
kernel that ``fastlbp.run_chunked_fastlbp`` uses, with absolute image coordinates, so
the histograms equal those of fastLBP's whole-image run (verified in the tests).
Where the margin falls outside the slide, the slide is mirrored.

Default radii follow fastLBP's progression round(1.499 * 1.327^i) up to a quarter of
the tile (56 px for 224 px tiles): 1, 2, 3, 4, 5, 6, 8, 11, 14, 19, 25, 34, 45, with
P = ceil(2 * pi * R) points (fastLBP's default).

An intensity block (mean, standard deviation and 10th/50th/90th percentiles of the H
and E channels) is computed alongside, because LBP ignores absolute stain amounts.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .channels import CHANNELS

INTENSITY_STATS = ("mean", "sd", "p10", "p50", "p90")


def default_radii(max_radius: float, a: float = 1.499, b: float = 1.327) -> list[int]:
    radii: list[int] = []
    i = 0
    while True:
        r = round(a * b**i)
        if r > max_radius:
            return radii
        if r not in radii:
            radii.append(int(r))
        i += 1


def npoints_for(radius: float) -> int:
    return int(math.ceil(2 * math.pi * radius))


@dataclass
class LbpSpec:
    radii: list[int]
    npoints: list[int]
    channels: list[str]  # subset of CHANNELS, e.g. ["H", "E", "grey"]
    tile_px: int

    @property
    def pad(self) -> int:
        return int(math.ceil(max(self.radii))) + 1

    def columns(self, channel: str) -> list[str]:
        return [f"{channel}_r{r}_p{p}_u{u}" for r, p in zip(self.radii, self.npoints) for u in range(p + 2)]

    @property
    def n_features_per_channel(self) -> int:
        return sum(p + 2 for p in self.npoints)


def tile_histograms(crop: np.ndarray, spec: LbpSpec, abs_y: int, abs_x: int) -> np.ndarray:
    """Concatenated uniform-LBP histograms for one channel of one tile.

    ``crop`` is the tile with a margin of ``spec.pad`` pixels on every side (uint8, 2-D);
    (abs_y, abs_x) is the absolute position of the tile's top-left pixel in the slide.
    """
    from fastlbp.lbp import uniform_lbp_uint8_padded_absolute

    pad, T = spec.pad, spec.tile_px
    out = np.empty(spec.n_features_per_channel, dtype=np.uint32)
    offset = 0
    for r, p in zip(spec.radii, spec.npoints):
        m = int(math.ceil(r)) + 1  # margin needed by this radius
        sub = np.ascontiguousarray(crop[pad - m : pad + T + m, pad - m : pad + T + m])
        codes = uniform_lbp_uint8_padded_absolute(sub, int(p), float(r), int(abs_y), int(abs_x), [m, m, m, m])
        out[offset : offset + p + 2] = np.bincount(codes.ravel(), minlength=p + 2)
        offset += p + 2
    return out


def intensity_stats(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.float32).ravel()
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return np.array([v.mean(), v.std(), p10, p50, p90], dtype=np.float32)


# ---- parallel driver ---------------------------------------------------------------------
_worker_state: dict = {}


def _init_worker(channels_path: str, spec: LbpSpec) -> None:
    os.environ.setdefault("VIPS_CONCURRENCY", "1")
    import pyvips

    pyvips.cache_set_max(0)
    _worker_state["img"] = pyvips.Image.new_from_file(channels_path, access="random")
    _worker_state["spec"] = spec


def _process_batch(batch: list[tuple[int, int, int]]) -> list[tuple[int, dict]]:
    from .image import read_region

    img, spec = _worker_state["img"], _worker_state["spec"]
    pad, T = spec.pad, spec.tile_px
    results = []
    for idx, x0, y0 in batch:
        crop = read_region(img, x0 - pad, y0 - pad, T + 2 * pad, T + 2 * pad)  # (h, w, 3) bands H, E, grey
        feats = {}
        for c in spec.channels:
            band = CHANNELS.index(c)
            feats[c] = tile_histograms(crop[:, :, band], spec, y0, x0)
        inner = crop[pad : pad + T, pad : pad + T]
        feats["intensity"] = np.concatenate([intensity_stats(inner[:, :, 0]), intensity_stats(inner[:, :, 1])])
        results.append((idx, feats))
    return results


def extract(channels_path: str, tiles: pd.DataFrame, spec: LbpSpec, ncpus: int = 1, batch_size: int = 8) -> dict:
    """LBP histograms and intensity statistics for every row of ``tiles``.

    Returns a dict with one (n_tiles x n_features) array per channel plus "intensity".
    """
    n = len(tiles)
    out = {c: np.zeros((n, spec.n_features_per_channel), dtype=np.uint32) for c in spec.channels}
    out["intensity"] = np.zeros((n, 2 * len(INTENSITY_STATS)), dtype=np.float32)
    jobs = list(zip(range(n), tiles["x0_px"].astype(int), tiles["y0_px"].astype(int)))
    batches = [jobs[i : i + batch_size] for i in range(0, n, batch_size)]
    if not batches:
        return out
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, ncpus), initializer=_init_worker, initargs=(channels_path, spec)) as pool:
        for results in pool.imap_unordered(_process_batch, batches):
            for idx, feats in results:
                for key, value in feats.items():
                    out[key][idx] = value
    return out


def intensity_columns() -> list[str]:
    return [f"{c}_{s}" for c in ("H", "E") for s in INTENSITY_STATS]
