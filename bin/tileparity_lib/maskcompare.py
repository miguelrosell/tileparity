"""Comparison of tissue masks (``--mask_compare``).

For one slide, the mask that Process 1 uses is compared with

* the other automatic methods (saturation, grey, optical density) at the same
  resolution (``--mask_mpp``, 4 µm/px by default), and
* the same automatic method computed at finer resolutions (``--mask_compare_mpp``,
  1 and 0.5 µm/px by default), read from the full-resolution image.

For every mask the comparison reports the share of the image called tissue, the Dice
overlap with the mask in use, how many tiles would pass the tissue threshold and how
many change their decision, and - for spatial samples - how well the mask agrees with
Space Ranger's own in-tissue call for every spot/bin. Fine resolutions of a large slide
are computed inside the bounding box of the tissue (plus a margin) when the whole image
would exceed ``--mask_compare_max_mpx`` megapixels.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from PIL import Image

from . import figures, image, mask as maskmod

BOTH = (0, 158, 115)
ONLY_ALT = (204, 121, 167)
ONLY_USED = (230, 159, 0)
MARGIN_UM = 500.0


def _difference_overlay(thumb: np.ndarray, used: np.ndarray, alt: np.ndarray) -> Image.Image:
    base = thumb[:, :, :3].astype(np.float32).copy()
    for sel, colour in ((used & alt, BOTH), (alt & ~used, ONLY_ALT), (used & ~alt, ONLY_USED)):
        base[sel] = 0.5 * base[sel] + 0.5 * np.array(colour, np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _agreement(m: np.ndarray, x: np.ndarray, y: np.ndarray, in_tissue: np.ndarray) -> tuple[float | None, float | None]:
    inside = maskmod.lookup(m, x, y)
    sens = float(inside[in_tissue].mean()) if in_tissue.any() else None
    spec = float((~inside[~in_tissue]).mean()) if (~in_tissue).any() else None
    return sens, spec


def compare(*, native_path, mpp_native: float, native_size: tuple[int, int], thumb: np.ndarray, sx: float, sy: float,
            scale: float, mpp_target: float, used_mask: np.ndarray, used_info: dict, params: maskmod.MaskParams,
            compare_mpp: list[float], max_mpx: float, lattice: pd.DataFrame, tile_px: int, min_tissue: float,
            positions: pd.DataFrame | None, upsample_tolerance: float = 0.02) -> tuple[pd.DataFrame, list[dict]]:
    """Masks to compare, with their metrics. Returns (table, cells); each cell holds the
    mask resampled to the thumbnail grid for the panel."""
    native_w, native_h = native_size
    thumb_h, thumb_w = used_mask.shape
    mask_scale = (sx + sy) / 2.0
    x0 = lattice["x0_px"].to_numpy(float)
    y0 = lattice["y0_px"].to_numpy(float)
    used_frac = maskmod.rect_fractions(used_mask, x0, y0, tile_px, mask_scale)
    used_pass = used_frac >= min_tissue
    if positions is not None:
        ux_t = positions["x_fullres"].to_numpy(float) * scale * sx
        uy_t = positions["y_fullres"].to_numpy(float) * scale * sy
        in_tissue = positions["in_tissue"].to_numpy() == 1

    rows, cells = [], []

    def add(label, method, mpp, thumb_mask, frac, info, note, sens_spec):
        passes = frac >= min_tissue if frac is not None else None
        sens, spec = sens_spec
        rows.append({
            "mask": label, "method": method, "mpp": mpp, "used": label == "used",
            "otsu_threshold": info.get("otsu_threshold") if info else None,
            "tissue_fraction_of_image": float(thumb_mask.mean()),
            "dice_vs_used": maskmod.dice(thumb_mask, used_mask),
            "n_tiles_pass": int(passes.sum()) if passes is not None else None,
            "n_tiles_gained": int((passes & ~used_pass).sum()) if passes is not None else None,
            "n_tiles_lost": int((~passes & used_pass).sum()) if passes is not None else None,
            "spaceranger_in_tissue_inside_mask": sens,
            "spaceranger_background_outside_mask": spec,
            "note": note,
        })
        cells.append({"label": label, "method": method, "mpp": mpp, "mask": thumb_mask})

    # 1. the mask in use
    sr = _agreement(used_mask, ux_t, uy_t, in_tissue) if positions is not None else (None, None)
    add("used", used_info.get("method"), used_info.get("mpp", params.mask_mpp), used_mask, used_frac, used_info,
        "mask used for the tiles", sr)

    # 2. the other automatic methods on the same thumbnail
    for method in maskmod.METHODS:
        if method == used_info.get("method"):
            continue
        m, info = maskmod.tissue_mask(thumb, params, method=method)
        frac = maskmod.rect_fractions(m, x0, y0, tile_px, mask_scale)
        sr = _agreement(m, ux_t, uy_t, in_tissue) if positions is not None else (None, None)
        add(f"{method} @ {params.mask_mpp:g} um/px", method, params.mask_mpp, m, frac, info, "", sr)

    # 3. the automatic method at finer resolutions, from the full-resolution image
    method = params.method
    native = None
    for mpp_c in compare_mpp:
        label = f"{method} @ {mpp_c:g} um/px"
        if mpp_c < mpp_native * (1 - upsample_tolerance):
            rows.append({"mask": label, "method": method, "mpp": mpp_c, "used": False,
                         "note": f"skipped: finer than the scan ({mpp_native:.4f} um/px)"})
            continue
        if abs(mpp_c - params.mask_mpp) < 1e-9:
            continue
        f = mpp_native / mpp_c  # native px -> comparison px
        W_c, H_c = int(round(native_w * f)), int(round(native_h * f))
        region = (0, 0, W_c, H_c)
        note = "whole image"
        if W_c * H_c > max_mpx * 1e6:
            ys, xs = np.nonzero(used_mask)
            if len(xs) == 0:
                rows.append({"mask": label, "method": method, "mpp": mpp_c, "used": False,
                             "note": "skipped: no tissue in the used mask"})
                continue
            k = params.mask_mpp / mpp_c  # thumbnail px -> comparison px
            margin = MARGIN_UM / mpp_c
            rx0 = int(max(np.floor(xs.min() * k - margin), 0))
            ry0 = int(max(np.floor(ys.min() * k - margin), 0))
            rx1 = int(min(np.ceil((xs.max() + 1) * k + margin), W_c))
            ry1 = int(min(np.ceil((ys.max() + 1) * k + margin), H_c))
            region = (rx0, ry0, rx1 - rx0, ry1 - ry0)
            note = f"inside the tissue bounding box + {MARGIN_UM:g} um (threshold from that region)"
            if region[2] * region[3] > max_mpx * 1e6:
                rows.append({"mask": label, "method": method, "mpp": mpp_c, "used": False,
                             "note": f"skipped: {region[2] * region[3] / 1e6:.0f} Mpx even inside the tissue box "
                                     f"(> --mask_compare_max_mpx {max_mpx:g})"})
                continue
        if native is None:
            native = image.open_rgb(native_path, access="random")
        resized = image.resample(native, f, "linear")
        rx0, ry0, rw, rh = region
        rw, rh = min(rw, resized.width - rx0), min(rh, resized.height - ry0)
        rgb = image.to_numpy(resized.crop(rx0, ry0, rw, rh))[:, :, :3]
        m, info = maskmod.tissue_mask(rgb, replace(params, mask_mpp=mpp_c), method=method)
        del rgb
        # tissue fraction of every tile, in this mask's own pixels
        s = mpp_target / mpp_c  # resampled (target) px -> comparison px
        frac = maskmod.window_fractions(m, x0 * s - rx0, y0 * s - ry0, tile_px * s, outside_as_background=True)
        # on the thumbnail grid for the picture and the Dice overlap
        k = mpp_c / params.mask_mpp
        tx0, ty0 = int(round(rx0 * k)), int(round(ry0 * k))
        tw, th = max(1, int(round(rw * k))), max(1, int(round(rh * k)))
        small = maskmod.resize_mask(m, tw, th)
        thumb_mask = np.zeros((thumb_h, thumb_w), bool)
        thumb_mask[ty0:ty0 + th, tx0:tx0 + tw] = small[: max(0, min(th, thumb_h - ty0)), : max(0, min(tw, thumb_w - tx0))]
        if positions is not None:
            sr = _agreement(m, positions["x_fullres"].to_numpy(float) * f - rx0,
                            positions["y_fullres"].to_numpy(float) * f - ry0, in_tissue)
        else:
            sr = (None, None)
        info["region_px"] = list(region)
        add(label, method, mpp_c, thumb_mask, frac, info, note, sr)
        del m
    return pd.DataFrame(rows), cells


def panel(thumb: np.ndarray, table: pd.DataFrame, cells: list[dict], min_tissue: float, caption: dict) -> Image.Image:
    used = cells[0]["mask"]
    imgs, labels = [], []
    by_label = table.set_index("mask")
    for c in cells:
        r = by_label.loc[c["label"]]
        if c["label"] == "used":
            imgs.append(figures.mask_overlay(thumb, used, colour=BOTH))
            title = f"USED: {c['method']} @ {c['mpp']:g} um/px"
        else:
            imgs.append(_difference_overlay(thumb, used, c["mask"]))
            title = c["label"]
        text = (f"{title}. Tissue {100 * r['tissue_fraction_of_image']:.1f}% of the image; Dice with used "
                f"{r['dice_vs_used']:.3f}; tiles with >= {min_tissue:.0%} tissue: {r['n_tiles_pass']}"
                f" (+{r['n_tiles_gained']} / -{r['n_tiles_lost']} vs used).")
        if r.get("spaceranger_in_tissue_inside_mask") is not None and not pd.isna(r.get("spaceranger_in_tissue_inside_mask")):
            text += f" Space Ranger in-tissue units inside the mask: {100 * r['spaceranger_in_tissue_inside_mask']:.1f}%"
            if r.get("spaceranger_background_outside_mask") is not None and not pd.isna(r.get("spaceranger_background_outside_mask")):
                text += f"; background units outside: {100 * r['spaceranger_background_outside_mask']:.1f}%"
            text += "."
        if r.get("note") and c["label"] != "used":
            text += f" ({r['note']})"
        labels.append(text)
    skipped = table[table["note"].astype(str).str.startswith("skipped")]
    cell_w = 300 if thumb.shape[1] < 300 else min(420, thumb.shape[1])
    grid_img = figures.grid(imgs, labels, ncols=3 if len(imgs) > 4 else 2, cell_width=cell_w)
    parts = [grid_img, figures.legend([(BOTH, "tissue in both masks"), (ONLY_ALT, "tissue only in this mask"),
                                       (ONLY_USED, "tissue only in the used mask")], width=grid_img.width)]
    if len(skipped):
        parts.append(figures.caption_strip(grid_img.width, [("Not computed", "; ".join(
            f"{r['mask']}: {r['note']}" for _, r in skipped.iterrows()))]))
    return figures.with_caption(figures.stack(parts), **caption)
