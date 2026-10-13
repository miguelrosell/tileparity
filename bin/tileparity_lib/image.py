"""Whole-slide image access built on libvips (pyvips).

libvips streams images in small regions, so resampling and writing a multi-gigabyte
slide needs memory proportional to the region size, not to the slide size. This is
what lets the same pipeline run on a laptop and on a large server.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pyvips

# Aperio SVS descriptions contain "MPP = 0.2527"; OME-XML contains PhysicalSizeX="0.2627".
_APERIO_MPP = re.compile(r"\bMPP\s*=\s*([0-9]*\.?[0-9]+)")
_OME_PHYSICAL_SIZE_X = re.compile(r'PhysicalSizeX="([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"')
_OME_PHYSICAL_SIZE_X_UNIT = re.compile(r'PhysicalSizeXUnit="([^"]+)"')
_UNIT_TO_UM = {"µm": 1.0, "um": 1.0, "micron": 1.0, "nm": 1e-3, "mm": 1e3, "cm": 1e4, "m": 1e6}


def open_rgb(path: str | Path, access: str = "random") -> pyvips.Image:
    """Open a brightfield image as 8-bit, 3-band RGB.

    Page 0 (the full-resolution level) is used for pyramidal TIFFs. An alpha band is
    dropped and 16-bit data are scaled to 8 bits, because both feature extractors
    expect 8-bit RGB.
    """
    img = pyvips.Image.new_from_file(str(path), access=access)
    if img.bands in (2, 4):  # grey + alpha or RGB + alpha
        img = img.extract_band(0, n=img.bands - 1)
    if img.bands == 1:
        raise ValueError(f"{path}: single-band image; tileparity expects RGB brightfield H&E")
    if img.bands != 3:
        raise ValueError(f"{path}: {img.bands} bands; tileparity expects RGB brightfield H&E")
    if img.interpretation not in ("srgb", "rgb", "multiband", "b-w"):
        img = img.colourspace("srgb")
    if img.format == "ushort":
        img = (img / 257.0).cast("uchar")
    elif img.format != "uchar":
        # Floating point or other integer formats: assume values already in 0..255.
        img = img.cast("uchar")
    return img


def mpp_from_metadata(img: pyvips.Image) -> float | None:
    """Pixel size (µm/px) stored in the file, or None when it is absent or implausible."""
    fields = set(img.get_fields())
    for field in ("image-description", "openslide.comment"):
        if field in fields:
            desc = str(img.get(field))
            m = _OME_PHYSICAL_SIZE_X.search(desc)
            if m:
                unit = _OME_PHYSICAL_SIZE_X_UNIT.search(desc)
                factor = _UNIT_TO_UM.get(unit.group(1) if unit else "µm", 1.0)
                value = float(m.group(1)) * factor
                if 0.01 < value < 100:
                    return value
            m = _APERIO_MPP.search(desc)
            if m and 0.01 < float(m.group(1)) < 100:
                return float(m.group(1))
    if "openslide.mpp-x" in fields:
        value = float(img.get("openslide.mpp-x"))
        if 0.01 < value < 100:
            return value
    # libvips stores xres in pixels per millimetre. It defaults to 1.0 when the file has
    # no resolution tags, which would mean 1000 µm/px: treat that as missing.
    if "resolution-unit" in fields and img.xres > 1.0:
        value = 1000.0 / img.xres
        if 0.01 < value < 100:
            return value
    return None


def resample(img: pyvips.Image, scale: float, kernel: str = "lanczos3") -> pyvips.Image:
    """Resize by ``scale`` (< 1 shrinks). Identity when scale is 1."""
    if math.isclose(scale, 1.0, rel_tol=0, abs_tol=1e-9):
        return img
    return img.resize(scale, kernel=kernel)


def write_tiled_tiff(
    img: pyvips.Image,
    path: str | Path,
    mpp: float | None = None,
    compression: str = "deflate",
    tile: int = 512,
) -> None:
    """Write a tiled BigTIFF that supports fast random access to any region.

    ``compression`` should be lossless ("deflate", "lzw", "zstd" or "none") so that the
    pipeline does not add compression artefacts on top of the scanner's own.
    """
    kwargs = dict(tile=True, tile_width=tile, tile_height=tile, bigtiff=True, compression=compression)
    if compression in ("deflate", "lzw", "zstd"):
        kwargs["predictor"] = "horizontal"
    if mpp:
        px_per_mm = 1000.0 / mpp
        img = img.copy(xres=px_per_mm, yres=px_per_mm)
        kwargs["resunit"] = "cm"
    img.write_to_file(str(path), **kwargs)


def read_region(img: pyvips.Image, x: int, y: int, width: int, height: int, pad_mode: str = "reflect") -> np.ndarray:
    """Read ``img[y:y+height, x:x+width]`` as a numpy array.

    Parts of the window outside the image are filled by mirroring the image
    (``pad_mode`` as in :func:`numpy.pad`), never with black, so that texture
    operators do not see an artificial dark border.
    """
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, img.width), min(y + height, img.height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"region ({x},{y},{width},{height}) lies outside the {img.width}x{img.height} image")
    arr = img.crop(x0, y0, x1 - x0, y1 - y0).numpy()
    if arr.ndim == 2:
        arr = arr[:, :, None]
    pads = ((y0 - y, y + height - y1), (x0 - x, x + width - x1), (0, 0))
    if any(p for pair in pads for p in pair):
        arr = np.pad(arr, pads, mode=pad_mode)
    return arr


def to_numpy(img: pyvips.Image) -> np.ndarray:
    """Whole (small) image as numpy array with a trailing channel axis."""
    arr = img.numpy()
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr
