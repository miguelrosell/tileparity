"""Process 1b - LBP adapter: single-channel images computed pixel by pixel from RGB.

LBP compares one number with another, so it needs a single-channel image. Every channel
is computed from the *same* resampled RGB pixels that UNI2-h reads, with no mixing of
neighbouring pixels. LBP is run on the channels of the selected *channel sets*
(``--lbp_channels``), which are alternatives to compare, never to be concatenated:

* ``he``   - H and E by colour deconvolution (Ruifrok & Johnston 2001), exactly as
  :func:`skimage.color.rgb2hed`:

  1. rgb = max(RGB / 255, 1e-6)
  2. optical density od = log(rgb) / log(1e-6)            (Beer-Lambert: od is linear in dye amount)
  3. stains = max(od @ hed_from_rgb, 0)                    (3 x 3 linear unmixing)
  4. 8-bit value = round(255 * min(stain / stain_max, 1)), with a *fixed* stain_max shared
     by all samples, so that values are comparable between slides.

* ``grey`` - ITU-R BT.601 luma 0.299 R + 0.587 G + 0.114 B (the classic input of LBP).
* ``rgb``  - the R, G and B values themselves, i.e. exactly UNI2-h's input, one channel at
  a time (what fastLBP does with an RGB image).

H and E are always written, because the intensity block (mean, SD and percentiles of H
and E per tile) is computed from them. The computation is expressed as libvips
operations, so it streams over the slide; :func:`rgb_to_channels_numpy` is the reference
implementation used in the tests.
"""

from __future__ import annotations

import math

import numpy as np
import pyvips
from skimage.color import hed_from_rgb

CHANNEL_SETS = {"he": ("H", "E"), "grey": ("grey",), "rgb": ("R", "G", "B")}
CHANNELS = ("H", "E", "grey", "R", "G", "B")
GREY_WEIGHTS = (0.299, 0.587, 0.114)


def parse_sets(text: str) -> list[str]:
    """Validated, de-duplicated channel sets in their canonical order (he, grey, rgb)."""
    sets = [s.strip().lower() for s in str(text).split(",") if s.strip()]
    unknown = sorted(set(sets) - set(CHANNEL_SETS))
    if not sets or unknown:
        raise ValueError(f"channel sets {unknown or sets!r}: choose from {', '.join(CHANNEL_SETS)} (comma-separated)")
    return [s for s in CHANNEL_SETS if s in sets]


def band_order(sets: list[str]) -> list[str]:
    """Bands of the channels image for the given sets; H and E always come first."""
    bands = ["H", "E"]
    for s in parse_sets(",".join(sets)):
        bands += [c for c in CHANNEL_SETS[s] if c not in bands]
    return bands


def lbp_channels(sets: list[str]) -> list[str]:
    """Channels that LBP is computed on, in band order."""
    wanted = {c for s in parse_sets(",".join(sets)) for c in CHANNEL_SETS[s]}
    return [c for c in band_order(sets) if c in wanted]


def rgb_to_channels_vips(rgb: pyvips.Image, h_max: float, e_max: float, bands: list[str] | None = None) -> pyvips.Image:
    """uchar image with the requested bands (default: H, E, grey)."""
    bands = bands or ["H", "E", "grey"]
    x = rgb.cast("double") / 255.0
    x = (x < 1e-6).ifthenelse(1e-6, x)
    od = x.log() / math.log(1e-6)
    # recomb: out_j = sum_i M[j][i] * in_i ; skimage computes od (row vector) @ hed_from_rgb,
    # i.e. out_j = sum_i od_i * hed_from_rgb[i][j], so M = hed_from_rgb.T
    stains = od.recomb(pyvips.Image.new_from_array(hed_from_rgb.T.tolist()))
    stains = (stains < 0).ifthenelse(0, stains)
    made = {
        "H": lambda: _to_uint8(stains.extract_band(0), h_max),
        "E": lambda: _to_uint8(stains.extract_band(1), e_max),
        "grey": lambda: _round_clip(rgb.cast("double").recomb(pyvips.Image.new_from_array([list(GREY_WEIGHTS)]))),
        "R": lambda: rgb.extract_band(0).cast("uchar"),
        "G": lambda: rgb.extract_band(1).cast("uchar"),
        "B": lambda: rgb.extract_band(2).cast("uchar"),
    }
    out = [made[b]() for b in bands]
    return out[0].bandjoin(out[1:]).copy(interpretation="multiband") if len(out) > 1 else out[0]


def _to_uint8(band: pyvips.Image, vmax: float) -> pyvips.Image:
    return _round_clip(band * (255.0 / vmax))


def _round_clip(band: pyvips.Image) -> pyvips.Image:
    band = (band > 255).ifthenelse(255, band)
    return (band + 0.5).floor().cast("uchar")


def rgb_to_channels_numpy(rgb: np.ndarray, h_max: float, e_max: float, bands: list[str] | None = None) -> np.ndarray:
    """Reference implementation (float64) of :func:`rgb_to_channels_vips`."""
    bands = bands or ["H", "E", "grey"]
    x = np.maximum(rgb.astype(np.float64) / 255.0, 1e-6)
    od = np.log(x) / np.log(1e-6)
    stains = np.maximum(od @ hed_from_rgb, 0)
    made = {
        "H": np.floor(np.minimum(stains[..., 0] * 255.0 / h_max, 255) + 0.5),
        "E": np.floor(np.minimum(stains[..., 1] * 255.0 / e_max, 255) + 0.5),
        "grey": np.floor(np.minimum(rgb.astype(np.float64) @ np.array(GREY_WEIGHTS), 255) + 0.5),
        "R": rgb[..., 0].astype(np.float64),
        "G": rgb[..., 1].astype(np.float64),
        "B": rgb[..., 2].astype(np.float64),
    }
    return np.stack([made[b] for b in bands], axis=-1).astype(np.uint8)


def clipping_fractions(rgb: pyvips.Image, h_max: float, e_max: float) -> dict:
    """Share of pixels whose H or E value exceeds the fixed range (saturates at 255)."""
    x = rgb.cast("double") / 255.0
    x = (x < 1e-6).ifthenelse(1e-6, x)
    od = x.log() / math.log(1e-6)
    stains = od.recomb(pyvips.Image.new_from_array(hed_from_rgb.T.tolist()))
    h_over = (stains.extract_band(0) > h_max).ifthenelse(1.0, 0.0).avg()
    e_over = (stains.extract_band(1) > e_max).ifthenelse(1.0, 0.0).avg()
    return {"H_fraction_clipped": float(h_over), "E_fraction_clipped": float(e_over)}
