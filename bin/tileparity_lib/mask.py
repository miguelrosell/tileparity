"""Tissue detection on a low-resolution copy of the slide.

Automatic methods (``method``), all with parameters in micrometres so that they mean the
same at any resolution:

1. A per-pixel "tissue score", higher for tissue than for glass:

   * ``saturation`` (default): S = (max(R,G,B) - min(R,G,B)) / max(R,G,B). Glass is white
     or grey (R ~ G ~ B, S ~ 0); stained tissue is pink or purple.
   * ``grey``: darkness 1 - (0.299 R + 0.587 G + 0.114 B) / 255. Glass is bright.
   * ``od``: mean optical density -log10(I / 255) over R, G and B, i.e. how much light the
     stain absorbs (Beer-Lambert). Macenko's stain normalisation uses OD < 0.15 for "background".

2. The score is smoothed with a Gaussian (sigma = ``smooth_um``) to suppress speckle.
3. Pixels darker than ``min_value`` (V = max(R,G,B)/255) are ignored: black scanner padding
   would otherwise look like tissue (grey, od) or distort the threshold.
4. Otsu's method picks the threshold t that maximises the between-class variance
   w0 * w1 * (mu0 - mu1)^2 of the smoothed scores; tissue = score >= t.
5. Morphological clean-up: closing with a disk of radius ``close_um``, filling holes smaller
   than ``fill_holes_um2`` and removing objects smaller than ``min_object_um2``.

An ``external`` mask (any image, e.g. exported from QuPath or ilastik, covering the whole
slide) can replace the automatic methods; it is only resized, never cleaned up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from skimage import filters

METHODS = ("saturation", "grey", "od")
GREY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class MaskParams:
    mask_mpp: float = 4.0
    smooth_um: float = 8.0
    min_value: float = 0.10
    close_um: float = 16.0
    fill_holes_um2: float = 20000.0
    min_object_um2: float = 20000.0
    method: str = "saturation"


def saturation_value(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HSV saturation and value of an RGB uint8 array, both in [0, 1]."""
    rgb = rgb.astype(np.float32) / 255.0
    vmax = rgb.max(axis=-1)
    vmin = rgb.min(axis=-1)
    sat = np.where(vmax > 0, (vmax - vmin) / np.maximum(vmax, 1e-12), 0.0)
    return sat.astype(np.float32), vmax.astype(np.float32)


def tissue_score(rgb: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """(score, value) for every pixel; tissue has the higher score."""
    rgb = rgb[:, :, :3]
    if method == "saturation":
        return saturation_value(rgb)
    value = rgb.max(axis=-1).astype(np.float32) / 255.0
    if method == "grey":
        return (1.0 - (rgb.astype(np.float32) @ GREY_WEIGHTS) / 255.0).astype(np.float32), value
    if method == "od":
        od = -np.log10(np.maximum(rgb.astype(np.float32), 1.0) / 255.0)
        return od.mean(axis=-1).astype(np.float32), value
    raise ValueError(f"unknown mask method {method!r}; choose from {METHODS}")


def tissue_mask(rgb: np.ndarray, params: MaskParams, method: str | None = None) -> tuple[np.ndarray, dict]:
    """Boolean tissue mask of an RGB image taken at ``params.mask_mpp`` µm/px."""
    method = method or params.method
    px = params.mask_mpp
    score, val = tissue_score(rgb, method)
    sigma_px = params.smooth_um / px
    score_s = ndimage.gaussian_filter(score, sigma=sigma_px) if sigma_px > 0 else score
    valid = val >= params.min_value
    if valid.sum() < 16:
        raise ValueError("image is almost entirely dark; cannot estimate a tissue threshold")
    threshold = float(filters.threshold_otsu(score_s[valid]))
    mask = (score_s >= threshold) & valid
    mask = clean_up(mask, params, px)
    info = {
        "method": method,
        "mpp": px,
        "otsu_threshold": threshold,
        "tissue_fraction_of_image": float(mask.mean()),
        "params": asdict(params) | {"method": method, "mask_mpp": px},
    }
    return mask.astype(bool), info


def clean_up(mask: np.ndarray, params: MaskParams, px: float) -> np.ndarray:
    radius = int(round(params.close_um / px))
    if radius > 0:
        mask = binary_closing(mask, radius)
    hole_px = int(round(params.fill_holes_um2 / px**2))
    if hole_px > 0:
        mask = fill_small_holes(mask, hole_px)
    obj_px = int(round(params.min_object_um2 / px**2))
    if obj_px > 0:
        mask = remove_small_objects(mask, obj_px)
    return mask


def disk(radius: int) -> np.ndarray:
    """Boolean disk-shaped structuring element of the given radius in pixels."""
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def binary_closing(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilation followed by erosion with a disk of ``radius`` pixels.

    Computed with Euclidean distance transforms, which gives exactly the result of
    :func:`scipy.ndimage.binary_closing` with :func:`disk` (a pixel is within the disk
    when the distance between pixel centres is <= radius) but in time independent of the
    radius, so it also works on high-resolution masks. The mask is padded first, so tissue
    touching the image border is not eroded away.
    """
    if radius <= 0:
        return mask.astype(bool)
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    if not padded.any():
        return mask.astype(bool)
    dilated = ndimage.distance_transform_edt(~padded) <= radius
    if dilated.all():
        closed = dilated
    else:
        closed = ndimage.distance_transform_edt(dilated) > radius
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


def window_fractions(mask: np.ndarray, x0: np.ndarray, y0: np.ndarray, size: float,
                     outside_as_background: bool = False) -> np.ndarray:
    """Fraction of mask pixels inside square windows given in mask pixels.

    A mask pixel counts when its centre falls inside the window [x0, x0 + size) x
    [y0, y0 + size). By default the fraction is taken over the part of the window that
    lies on the mask (a tile on the image edge may overhang a rounded thumbnail by a
    fraction of a pixel); with ``outside_as_background`` the rest counts as background
    (used when the mask covers only part of the image). Sums use an integral image, so
    the cost does not depend on the window size."""
    h, w = mask.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)

    def first_inside(coord: np.ndarray) -> np.ndarray:
        # mask pixel i covers [i, i+1); its centre i + 0.5 is inside [a, b) when i >= ceil(a - 0.5)
        return np.ceil(coord - 0.5).astype(np.int64)

    x0 = np.asarray(x0, float)
    y0 = np.asarray(y0, float)
    c0, c1 = first_inside(x0), first_inside(x0 + size)
    r0, r1 = first_inside(y0), first_inside(y0 + size)
    n_window = np.maximum(r1 - r0, 0) * np.maximum(c1 - c0, 0)
    c0, c1 = np.clip(c0, 0, w), np.clip(c1, 0, w)
    r0, r1 = np.clip(r0, 0, h), np.clip(r1, 0, h)
    total = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    n_pix = n_window if outside_as_background else (r1 - r0) * (c1 - c0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_pix > 0, total / np.maximum(n_pix, 1), 0.0)
    return frac.astype(np.float32)


def rect_fractions(mask: np.ndarray, x0: np.ndarray, y0: np.ndarray, size: float, mask_scale: float) -> np.ndarray:
    """Fraction of mask pixels inside square windows given in another pixel grid (here: the
    resampled image), with top-left corners (x0, y0) and side ``size``; ``mask_scale``
    converts those coordinates to mask pixels."""
    return window_fractions(mask, np.asarray(x0, float) * mask_scale, np.asarray(y0, float) * mask_scale,
                            size * mask_scale)


def load_external(path, out_width: int, out_height: int, native_width: int, native_height: int,
                  aspect_tolerance: float = 0.02) -> tuple[np.ndarray, dict]:
    """A user-provided mask image resized (nearest neighbour) to ``out_width`` x ``out_height``.

    The mask must cover the whole slide (same aspect ratio as the full-resolution image),
    at any resolution. Non-zero pixels are tissue for 0/1 masks; pixels >= 128 for 8-bit."""
    import pyvips

    img = pyvips.Image.new_from_file(str(path), access="sequential")
    if img.bands > 1:
        img = img.extract_band(0)
    ratio, native_ratio = img.width / img.height, native_width / native_height
    if abs(ratio - native_ratio) / native_ratio > aspect_tolerance:
        raise ValueError(
            f"external mask {path}: aspect ratio {img.width}x{img.height} does not match the image "
            f"({native_width}x{native_height}); the mask must cover the whole slide"
        )
    small = img.resize(out_width / img.width, vscale=out_height / img.height, kernel="nearest")
    arr = small.numpy()
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    out = np.zeros((out_height, out_width), dtype=arr.dtype)
    h, w = min(out_height, arr.shape[0]), min(out_width, arr.shape[1])
    out[:h, :w] = arr[:h, :w]
    mask = out > 0 if out.max() <= 1 else out >= 128
    info = {"method": "external", "source": str(path), "source_size": [img.width, img.height],
            "tissue_fraction_of_image": float(mask.mean())}
    return mask, info


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of two masks: 2 |a and b| / (|a| + |b|); 1 = identical."""
    a, b = a.astype(bool), b.astype(bool)
    total = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / total) if total else 1.0


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-weighted resize of a boolean mask (a pixel is tissue when >= half of it is)."""
    from PIL import Image

    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.resize((width, height), Image.BOX)) >= 128


def lookup(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask value at points given in mask pixels (False outside the mask)."""
    xi = np.floor(np.asarray(x, float)).astype(np.int64)
    yi = np.floor(np.asarray(y, float)).astype(np.int64)
    ok = (xi >= 0) & (yi >= 0) & (xi < mask.shape[1]) & (yi < mask.shape[0])
    out = np.zeros(len(xi), bool)
    out[ok] = mask[yi[ok], xi[ok]]
    return out
