"""Tissue detection on a low-resolution thumbnail.

Method (all parameters in micrometres, so they mean the same at any scan resolution):

1. Saturation S = (max(R,G,B) - min(R,G,B)) / max(R,G,B) for every thumbnail pixel.
   Glass is white or grey (R ~ G ~ B, S ~ 0); stained tissue is pink or purple (S high).
2. S is smoothed with a Gaussian (sigma = ``smooth_um``) to suppress speckle.
3. Pixels darker than ``min_value`` (V = max(R,G,B)/255) are ignored, so black scanner
   padding does not distort the threshold.
4. Otsu's method picks the threshold t that maximises the between-class variance
   w0 * w1 * (mu0 - mu1)^2 of the smoothed S values.
5. Morphological clean-up: closing with a disk of radius ``close_um``, filling holes
   smaller than ``fill_holes_um2`` and removing objects smaller than ``min_object_um2``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from skimage import filters


@dataclass
class MaskParams:
    mask_mpp: float = 4.0
    smooth_um: float = 8.0
    min_value: float = 0.10
    close_um: float = 16.0
    fill_holes_um2: float = 20000.0
    min_object_um2: float = 20000.0


def saturation_value(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HSV saturation and value of an RGB uint8 array, both in [0, 1]."""
    rgb = rgb.astype(np.float32) / 255.0
    vmax = rgb.max(axis=-1)
    vmin = rgb.min(axis=-1)
    sat = np.where(vmax > 0, (vmax - vmin) / np.maximum(vmax, 1e-12), 0.0)
    return sat.astype(np.float32), vmax.astype(np.float32)


def tissue_mask(thumb_rgb: np.ndarray, params: MaskParams) -> tuple[np.ndarray, dict]:
    """Boolean tissue mask of an RGB thumbnail taken at ``params.mask_mpp`` µm/px."""
    px = params.mask_mpp
    sat, val = saturation_value(thumb_rgb)
    sigma_px = params.smooth_um / px
    sat_s = ndimage.gaussian_filter(sat, sigma=sigma_px) if sigma_px > 0 else sat
    valid = val >= params.min_value
    if valid.sum() < 16:
        raise ValueError("thumbnail is almost entirely dark; cannot estimate a tissue threshold")
    threshold = float(filters.threshold_otsu(sat_s[valid]))
    mask = (sat_s >= threshold) & valid

    radius = int(round(params.close_um / px))
    if radius > 0:
        mask = binary_closing(mask, radius)
    hole_px = int(round(params.fill_holes_um2 / px**2))
    if hole_px > 0:
        mask = fill_small_holes(mask, hole_px)
    obj_px = int(round(params.min_object_um2 / px**2))
    if obj_px > 0:
        mask = remove_small_objects(mask, obj_px)

    info = {
        "otsu_threshold_saturation": threshold,
        "tissue_fraction_of_slide": float(mask.mean()),
        "params": asdict(params),
    }
    return mask.astype(bool), info


def disk(radius: int) -> np.ndarray:
    """Boolean disk-shaped structuring element of the given radius in pixels."""
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def binary_closing(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilation followed by erosion with a disk. The mask is padded first, so tissue
    touching the image border is not eroded away."""
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    closed = ndimage.binary_closing(padded, structure=disk(radius))
    return closed[radius:-radius, radius:-radius]


def fill_small_holes(mask: np.ndarray, max_area_px: int) -> np.ndarray:
    """Fill background regions enclosed by tissue that are smaller than ``max_area_px``.
    Background touching the image border is never filled."""
    labels, n = ndimage.label(~mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    fill = sizes < max_area_px
    fill[0] = False
    fill[border] = False
    return mask | fill[labels]


def remove_small_objects(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove connected tissue components smaller than ``min_area_px``."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area_px
    keep[0] = False
    return keep[labels]


def rect_fractions(
    mask: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    size: float,
    mask_scale: float,
) -> np.ndarray:
    """Fraction of mask pixels inside each square window.

    Windows are given in another pixel grid (here: the resampled image), with top-left
    corners (x0, y0) and side ``size``; ``mask_scale`` converts those coordinates to mask
    pixels. A mask pixel counts when its centre falls inside the window. Sums use an
    integral image, so the cost does not depend on the window size.
    """
    h, w = mask.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)

    def first_inside(coord: np.ndarray) -> np.ndarray:
        # mask pixel i covers [i, i+1) in mask units; its centre i + 0.5 is inside
        # [a, b) when i >= a - 0.5, i.e. i >= ceil(a - 0.5)
        return np.ceil(coord - 0.5).astype(np.int64)

    c0 = np.clip(first_inside(np.asarray(x0, float) * mask_scale), 0, w)
    c1 = np.clip(first_inside((np.asarray(x0, float) + size) * mask_scale), 0, w)
    r0 = np.clip(first_inside(np.asarray(y0, float) * mask_scale), 0, h)
    r1 = np.clip(first_inside((np.asarray(y0, float) + size) * mask_scale), 0, h)
    total = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    count = (r1 - r0) * (c1 - c0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return frac.astype(np.float32)
