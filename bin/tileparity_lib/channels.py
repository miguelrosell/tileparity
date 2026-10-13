"""Process 1b - LBP adapter: single-channel images computed pixel by pixel from RGB.

LBP compares one number with another, so it needs a single-channel image. All three
channels are computed from the *same* resampled RGB pixels as the UNI2-h input, with
no mixing of neighbouring pixels:

* H and E - colour deconvolution (Ruifrok & Johnston 2001), exactly as
  :func:`skimage.color.rgb2hed`:

  1. rgb = max(RGB / 255, 1e-6)
  2. optical density od = log(rgb) / log(1e-6)            (Beer-Lambert: od is linear in dye amount)
  3. stains = max(od @ hed_from_rgb, 0)                    (3 x 3 linear unmixing)
  4. 8-bit value = round(255 * min(stain / stain_max, 1)), with a *fixed* stain_max shared
     by all samples, so that values are comparable between slides.

* grey - ITU-R BT.601 luma 0.299 R + 0.587 G + 0.114 B.

The computation is expressed as libvips operations, so it streams over the slide.
:func:`rgb_to_channels_numpy` is the reference implementation used in the tests.
"""

from __future__ import annotations

import math

import numpy as np
import pyvips
from skimage.color import hed_from_rgb

CHANNELS = ("H", "E", "grey")
GREY_WEIGHTS = (0.299, 0.587, 0.114)


def rgb_to_channels_vips(rgb: pyvips.Image, h_max: float, e_max: float) -> pyvips.Image:
    """3-band uchar image with bands (H, E, grey)."""
    x = rgb.cast("double") / 255.0
    x = (x < 1e-6).ifthenelse(1e-6, x)
    od = x.log() / math.log(1e-6)
    # recomb: out_j = sum_i M[j][i] * in_i ; skimage computes od (row vector) @ hed_from_rgb,
    # i.e. out_j = sum_i od_i * hed_from_rgb[i][j], so M = hed_from_rgb.T
    stains = od.recomb(pyvips.Image.new_from_array(hed_from_rgb.T.tolist()))
    stains = (stains < 0).ifthenelse(0, stains)
    h = _to_uint8(stains.extract_band(0), h_max)
    e = _to_uint8(stains.extract_band(1), e_max)
    grey = rgb.cast("double").recomb(pyvips.Image.new_from_array([list(GREY_WEIGHTS)]))
    grey = _round_clip(grey)
    return h.bandjoin([e, grey]).copy(interpretation="multiband")


def _to_uint8(band: pyvips.Image, vmax: float) -> pyvips.Image:
    return _round_clip(band * (255.0 / vmax))


def _round_clip(band: pyvips.Image) -> pyvips.Image:
    band = (band > 255).ifthenelse(255, band)
    return (band + 0.5).floor().cast("uchar")


def rgb_to_channels_numpy(rgb: np.ndarray, h_max: float, e_max: float) -> np.ndarray:
    """Reference implementation (float64) of :func:`rgb_to_channels_vips`."""
    x = np.maximum(rgb.astype(np.float64) / 255.0, 1e-6)
    od = np.log(x) / np.log(1e-6)
    stains = np.maximum(od @ hed_from_rgb, 0)
    h = np.floor(np.minimum(stains[..., 0] * 255.0 / h_max, 255) + 0.5)
    e = np.floor(np.minimum(stains[..., 1] * 255.0 / e_max, 255) + 0.5)
    grey = np.floor(np.minimum(rgb.astype(np.float64) @ np.array(GREY_WEIGHTS), 255) + 0.5)
    return np.stack([h, e, grey], axis=-1).astype(np.uint8)


def clipping_fractions(rgb: pyvips.Image, h_max: float, e_max: float) -> dict:
    """Share of pixels whose H or E value exceeds the fixed range (saturates at 255)."""
    x = rgb.cast("double") / 255.0
    x = (x < 1e-6).ifthenelse(1e-6, x)
    od = x.log() / math.log(1e-6)
    stains = od.recomb(pyvips.Image.new_from_array(hed_from_rgb.T.tolist()))
    h_over = (stains.extract_band(0) > h_max).ifthenelse(1.0, 0.0).avg()
    e_over = (stains.extract_band(1) > e_max).ifthenelse(1.0, 0.0).avg()
    return {"H_fraction_clipped": float(h_over), "E_fraction_clipped": float(e_over)}
