"""Process 1b - the input images of both extractors, computed pixel by pixel from the
resampled RGB image of Process 1 (no mixing of neighbouring pixels).

Input sets. fastLBP (``--lbp_channels``) and UNI2-h (``--uni2h_inputs``) use the same
names, and in ``rgb`` and ``rgb_he`` both extractors see exactly the same pixels:

* ``rgb``    - the resampled RGB image as it is. UNI2-h reads the image; fastLBP computes
  one LBP per channel, R, G and B.
* ``rgb_he`` - the same image rebuilt from its haematoxylin and eosin only: the colour that
  is neither H nor E is removed. UNI2-h reads it as an RGB image; fastLBP computes one LBP
  per channel of it (``R_he``, ``G_he``, ``B_he``).
* ``he``     - fastLBP only, optional: the amounts of H and of E as two single-channel
  images (stain-specific textures: nuclei; cytoplasm and stroma).

How H and E are separated (colour deconvolution, Ruifrok & Johnston 2001):

1. optical density, per pixel and channel: od = log(max(RGB / 255, 1e-6)) / log(1e-6)
   (Beer-Lambert: od is proportional to the amount of dye; 0 = no light absorbed);
2. every pixel's od (a point in 3-D) is written as od = a_H * H + a_E * E + a_R * R, where
   H and E are the colours of the two stains (unit vectors in od space) and R = H x E is
   the direction perpendicular to both;
3. ``rgb_he``: od_he = a_H * H + a_E * E, i.e. od with its component along R removed (the
   closest colour made of H and E only), and RGB_he = round(255 * min(1e-6 ** od_he, 1));
4. ``he``: 8-bit value = round(255 * min(max(a, 0) / a_max, 1)), with a fixed a_max shared
   by all samples, so that values are comparable between slides.

The stain vectors H and E (``--stain_vectors``):

* ``slide`` (default): estimated on each slide from the pixels of its extracted tiles, with
  the method of Macenko et al. (2009): keep the stained pixels (length of the natural-log
  od vector >= 0.15), find the plane through the origin that holds most of their optical
  density (the two largest singular vectors), and take as H and E the directions at the
  1st and 99th percentile of the angle in that plane (H: the one that absorbs more red).
  Stain colours vary between laboratories and scanners, so fixed vectors would put part of
  the real H and E into the "removed" component. (Common implementations of Macenko keep
  pixels with od >= 0.15 in every channel; that drops pale eosin-stained stroma, which
  absorbs little red, and left a larger residual on real H&E, so the length is used.)
  ``rgb_he`` depends only on the plane; the separate H and E directions matter for ``he``
  and the intensity block.
* ``standard``: Ruifrok and Johnston's published H and E vectors (those of
  ``skimage.color.rgb2hed``), the same for every slide. Also used, with a note in
  ``channels.json``, when a slide has too few stained pixels for the estimate or the
  estimate is not a plausible pair of stain colours.

H and E (8 bits) are always written, because the intensity block of fastLBP (mean, SD and
percentiles of H and E per tile) is computed from them. The computation is expressed as
libvips operations, so it streams over the slide; :func:`rgb_to_channels_numpy` is the
reference implementation used in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyvips

CHANNEL_SETS = {"rgb": ("R", "G", "B"), "rgb_he": ("R_he", "G_he", "B_he"), "he": ("H", "E")}
LBP_SETS = ("rgb", "rgb_he", "he")
UNI2H_INPUTS = ("rgb", "rgb_he")
CHANNELS = ("H", "E", "R", "G", "B", "R_he", "G_he", "B_he")
REMOVED = {"grey": "the grey channel set was removed in tileparity 0.3 (use rgb and rgb_he)"}
STAIN_VECTORS = ("slide", "standard")
LOG_1E6 = math.log(1e-6)
TISSUE_OD = 0.15 / -LOG_1E6  # stained pixel: length of its natural-log od vector >= 0.15 (Macenko's beta), in these units
MIN_ANGLE_HE = 15.0  # degrees: H and E closer than this are poorly separated (noted in channels.json)
# Ruifrok & Johnston (2001) H and E colours, as rows 0 and 1 of skimage.color.rgb_from_hed
# (written out so that the UNI2-h environment does not need scikit-image)
RUIFROK_H = (0.65, 0.70, 0.29)
RUIFROK_E = (0.07, 0.99, 0.11)


def parse_sets(text: str, allowed: tuple = LBP_SETS, allow_empty: bool = False) -> list[str]:
    """Validated, de-duplicated input sets in their canonical order (rgb, rgb_he, he)."""
    sets = [s.strip().lower() for s in str(text or "").split(",") if s.strip() and s.strip().lower() != "none"]
    for s in sets:
        if s in REMOVED:
            raise ValueError(f"'{s}': {REMOVED[s]}")
    unknown = sorted(set(sets) - set(allowed))
    if unknown or (not sets and not allow_empty):
        raise ValueError(f"input sets {unknown or sets!r}: choose from {', '.join(allowed)} (comma-separated)")
    return [s for s in CHANNEL_SETS if s in sets]


def band_order(lbp_sets: list[str], uni2h_inputs: list[str] = ()) -> list[str]:
    """Bands of the channels image: H and E always (intensity block), the R, G, B copies when
    fastLBP uses ``rgb``, and the H&E-only RGB image when fastLBP or UNI2-h uses ``rgb_he``.
    (UNI2-h's ``rgb`` input is the resampled image itself.)"""
    lbp_sets = parse_sets(",".join(lbp_sets), LBP_SETS, allow_empty=True)
    uni2h_inputs = parse_sets(",".join(uni2h_inputs), UNI2H_INPUTS, allow_empty=True)
    bands = ["H", "E"]
    if "rgb" in lbp_sets:
        bands += list(CHANNEL_SETS["rgb"])
    if "rgb_he" in lbp_sets or "rgb_he" in uni2h_inputs:
        bands += list(CHANNEL_SETS["rgb_he"])
    return bands


def lbp_channels(sets: list[str]) -> list[str]:
    """Channels that LBP is computed on, in the canonical order of the sets."""
    return [c for s in parse_sets(",".join(sets), LBP_SETS) for c in CHANNEL_SETS[s]]


# ---- stain vectors ---------------------------------------------------------------------------
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


@dataclass
class StainBasis:
    """H and E colours (unit vectors in optical-density space) and how they were obtained."""

    H: np.ndarray
    E: np.ndarray
    method: str
    n_pixels: int = 0
    plane_energy: float | None = None  # share of the stained pixels' squared optical density in the H-E plane
    note: str = ""

    @property
    def R(self) -> np.ndarray:
        return _unit(np.cross(self.H, self.E))

    @property
    def matrix(self) -> np.ndarray:
        """Rows H, E, R: od = [a_H, a_E, a_R] @ matrix."""
        return np.vstack([self.H, self.E, self.R])

    @property
    def unmix(self) -> np.ndarray:
        """od @ unmix = [a_H, a_E, a_R]."""
        return np.linalg.inv(self.matrix)

    def to_dict(self) -> dict:
        return {"method": self.method, "H": [round(float(x), 5) for x in self.H],
                "E": [round(float(x), 5) for x in self.E], "R": [round(float(x), 5) for x in self.R],
                "angle_H_E_degrees": round(float(np.degrees(np.arccos(np.clip(self.H @ self.E, -1, 1)))), 2),
                "angle_to_standard_degrees": {"H": _angle(self.H, standard_basis().H),
                                              "E": _angle(self.E, standard_basis().E)},
                "n_pixels": int(self.n_pixels), "plane_energy_share": self.plane_energy, "note": self.note}


def _angle(a, b) -> float:
    return round(float(np.degrees(np.arccos(np.clip(float(_unit(a) @ _unit(b)), -1, 1)))), 2)


def standard_basis(note: str = "") -> StainBasis:
    """Ruifrok and Johnston's H and E vectors, as in skimage.color.rgb2hed."""
    return StainBasis(_unit(RUIFROK_H), _unit(RUIFROK_E),
                      "standard (Ruifrok & Johnston 2001, as in skimage.color.rgb2hed)", note=note)


def estimate_stain_basis(pixels: np.ndarray, alpha: float = 1.0, min_pixels: int = 10000) -> StainBasis:
    """H and E vectors of one slide from a sample of its tissue pixels (N x 3, uint8), after
    Macenko et al. (2009); falls back to :func:`standard_basis` when the estimate is not
    reliable (too few tissue pixels, or vectors that are not plausible stain colours)."""
    od = optical_density(np.asarray(pixels).reshape(-1, 3))
    od = od[np.linalg.norm(od, axis=1) >= TISSUE_OD]
    if len(od) < min_pixels:
        return standard_basis(note=f"standard vectors used: only {len(od)} stained pixels (< {min_pixels})")
    _u, s, vt = np.linalg.svd(od, full_matrices=False)  # plane through the origin
    plane = vt[:2].T.copy()  # 3 x 2
    for k in range(2):
        if plane[0, k] < 0:
            plane[:, k] *= -1
    coords = od @ plane
    phi = np.arctan2(coords[:, 1], coords[:, 0])
    lo, hi = np.percentile(phi, [alpha, 100 - alpha])
    v_lo = _unit(plane @ [math.cos(lo), math.sin(lo)])
    v_hi = _unit(plane @ [math.cos(hi), math.sin(hi)])
    H, E = (v_lo, v_hi) if v_lo[0] > v_hi[0] else (v_hi, v_lo)
    energy = float((s[0] ** 2 + s[1] ** 2) / (s**2).sum())
    if (H < -0.05).any() or (E < -0.05).any() or _angle(H, E) < 5:
        return standard_basis(note=f"standard vectors used: the slide estimate H={H.round(3).tolist()}, "
                                   f"E={E.round(3).tolist()} is not a plausible pair of stain colours")
    H, E = _unit(np.clip(H, 0, None)), _unit(np.clip(E, 0, None))  # tiny negative parts are estimation noise
    note = ""
    if _angle(H, E) < MIN_ANGLE_HE:
        note = (f"H and E are only {_angle(H, E):.0f} degrees apart: the slide may hold little of one stain, so the "
                "H and E amounts (he channels, intensity block) are poorly separated; rgb_he is not affected")
    return StainBasis(H, E, "slide (Macenko et al. 2009: SVD of the stained pixels' optical densities, "
                            "1st/99th angle percentiles)", n_pixels=len(od), plane_energy=round(energy, 5), note=note)


# ---- channel images ----------------------------------------------------------------------------
def _od_vips(rgb: pyvips.Image) -> pyvips.Image:
    x = rgb.cast("double") / 255.0
    x = (x < 1e-6).ifthenelse(1e-6, x)
    return x.log() / LOG_1E6


def _amounts_vips(rgb: pyvips.Image, basis: StainBasis) -> pyvips.Image:
    # recomb: out_j = sum_i M[j][i] * in_i ; we want a_j = sum_i od_i * unmix[i][j], so M = unmix.T
    return _od_vips(rgb).recomb(pyvips.Image.new_from_array(basis.unmix.T.tolist()))


def _rgb_he_vips(amounts: pyvips.Image, basis: StainBasis) -> pyvips.Image:
    he = amounts.extract_band(0, n=2)
    od_he = he.recomb(pyvips.Image.new_from_array([[basis.H[j], basis.E[j]] for j in range(3)]))
    x = (od_he * LOG_1E6).exp()
    x = (x > 1).ifthenelse(1, x)
    return (x * 255.0 + 0.5).floor().cast("uchar")


def rgb_to_channels_vips(rgb: pyvips.Image, h_max: float, e_max: float, bands: list[str] | None = None,
                         basis: StainBasis | None = None) -> pyvips.Image:
    """uchar image with the requested bands (default: H and E)."""
    bands = bands or ["H", "E"]
    basis = basis or standard_basis()
    amounts = _amounts_vips(rgb, basis)
    rgb_he = _rgb_he_vips(amounts, basis) if any(b.endswith("_he") for b in bands) else None
    made = {
        "H": lambda: _to_uint8(amounts.extract_band(0), h_max),
        "E": lambda: _to_uint8(amounts.extract_band(1), e_max),
        "R": lambda: rgb.extract_band(0).cast("uchar"),
        "G": lambda: rgb.extract_band(1).cast("uchar"),
        "B": lambda: rgb.extract_band(2).cast("uchar"),
        "R_he": lambda: rgb_he.extract_band(0),
        "G_he": lambda: rgb_he.extract_band(1),
        "B_he": lambda: rgb_he.extract_band(2),
    }
    out = [made[b]() for b in bands]
    return out[0].bandjoin(out[1:]).copy(interpretation="multiband") if len(out) > 1 else out[0]


def _to_uint8(band: pyvips.Image, vmax: float) -> pyvips.Image:
    band = (band < 0).ifthenelse(0, band) * (255.0 / vmax)
    band = (band > 255).ifthenelse(255, band)
    return (band + 0.5).floor().cast("uchar")


def stain_amounts(rgb: np.ndarray, basis: StainBasis) -> np.ndarray:
    """[a_H, a_E, a_R] per pixel (float64)."""
    return optical_density(rgb[..., :3]) @ basis.unmix


def rgb_he_numpy(rgb: np.ndarray, basis: StainBasis | None = None) -> np.ndarray:
    """Reference for ``rgb_he``: od without its component along R, back to 8-bit RGB."""
    basis = basis or standard_basis()
    a = stain_amounts(rgb, basis)
    od_he = a[..., :1] * basis.H + a[..., 1:2] * basis.E
    return np.floor(np.clip(np.exp(LOG_1E6 * od_he), 0, 1) * 255.0 + 0.5).astype(np.uint8)


def rgb_to_channels_numpy(rgb: np.ndarray, h_max: float, e_max: float, bands: list[str] | None = None,
                          basis: StainBasis | None = None) -> np.ndarray:
    """Reference implementation (float64) of :func:`rgb_to_channels_vips`."""
    bands = bands or ["H", "E"]
    basis = basis or standard_basis()
    a = np.maximum(stain_amounts(rgb, basis), 0)
    rgb_he = rgb_he_numpy(rgb, basis) if any(b.endswith("_he") for b in bands) else None
    made = {
        "H": lambda: np.floor(np.minimum(a[..., 0] * 255.0 / h_max, 255) + 0.5),
        "E": lambda: np.floor(np.minimum(a[..., 1] * 255.0 / e_max, 255) + 0.5),
        "R": lambda: rgb[..., 0],
        "G": lambda: rgb[..., 1],
        "B": lambda: rgb[..., 2],
        "R_he": lambda: rgb_he[..., 0],
        "G_he": lambda: rgb_he[..., 1],
        "B_he": lambda: rgb_he[..., 2],
    }
    return np.stack([made[b]() for b in bands], axis=-1).astype(np.uint8)


# ---- what rgb_he removes ------------------------------------------------------------------------
def optical_density(rgb: np.ndarray) -> np.ndarray:
    """od = log(max(RGB / 255, 1e-6)) / log(1e-6), per channel (0 = no light absorbed)."""
    return np.log(np.maximum(np.asarray(rgb, dtype=np.float64) / 255.0, 1e-6)) / LOG_1E6


def residual_share(rgb: np.ndarray, rgb_he: np.ndarray) -> float:
    """Share of the optical density of ``rgb`` that the H&E-only image does not reproduce:
    sum |od - od_he| / sum |od| over pixels and channels (0 = fully explained by H and E)."""
    od, od_he = optical_density(rgb[..., :3]), optical_density(rgb_he[..., :3])
    total = np.abs(od).sum()
    return float(np.abs(od - od_he).sum() / total) if total > 0 else 0.0


def clipping_fractions(rgb: pyvips.Image, h_max: float, e_max: float, basis: StainBasis | None = None) -> dict:
    """Share of pixels whose H or E amount exceeds the fixed range of the ``he`` channels."""
    amounts = _amounts_vips(rgb, basis or standard_basis())
    h_over = (amounts.extract_band(0) > h_max).ifthenelse(1.0, 0.0).avg()
    e_over = (amounts.extract_band(1) > e_max).ifthenelse(1.0, 0.0).avg()
    return {"H_fraction_clipped": float(h_over), "E_fraction_clipped": float(e_over)}
