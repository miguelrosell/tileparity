"""Per-tile quality-control metrics.

Metrics are stored as columns of the tile table so that any filter can be changed
downstream without rerunning the pipeline. None of them takes part in the default
``keep`` decision (blurred tiles are dropped only with ``--drop_blurred``): it is
easier to discard flagged tiles later, e.g. after checking whether they drive a
clustering, than to recover dropped ones.

* ``lapvar``: variance of the Laplacian of the greyscale tile. The Laplacian is the
  sum of second derivatives and responds to edges; a sharp tile has many edges and a
  high variance, an out-of-focus tile a low one. A tile is flagged as blurred when
  log10(lapvar) is more than ``blur_mad`` robust standard deviations
  (1.4826 x median absolute deviation) below the slide's median.
* ``pen_fraction``: share of pixels with a green/cyan hue (70-200 degrees), saturation
  > 0.25 and value > 0.15, typical of marker pen and absent from H&E.
* ``dark_fraction``: share of pixels with value < 0.25 (folds, debris, black pen).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.color import rgb2hsv

GREY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601 luma


def tile_metrics(rgb: np.ndarray) -> tuple[float, float, float]:
    """(lapvar, pen_fraction, dark_fraction) for one RGB uint8 tile."""
    grey = rgb.astype(np.float32) @ GREY_WEIGHTS
    lapvar = float(ndimage.laplace(grey).var())
    hsv = rgb2hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    pen = (h >= 70 / 360) & (h <= 200 / 360) & (s > 0.25) & (v > 0.15)
    dark = v < 0.25
    return lapvar, float(pen.mean()), float(dark.mean())


def blur_flags(lapvar: np.ndarray, eligible: np.ndarray, blur_mad: float, min_tiles: int = 10) -> np.ndarray:
    """Robust low-sharpness outliers among ``eligible`` tiles."""
    flags = np.zeros(len(lapvar), dtype=bool)
    if blur_mad <= 0 or eligible.sum() < min_tiles:
        return flags
    logv = np.log10(np.asarray(lapvar, float) + 1e-6)
    ref = logv[eligible]
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med))
    if mad <= 0:
        return flags
    z = (logv - med) / mad
    flags[eligible] = z[eligible] < -blur_mad
    return flags
