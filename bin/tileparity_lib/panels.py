"""QC figures of Process 1: tiles over the thumbnail, and the tissue-threshold panel.

Every figure has its explanation (what it shows, how to read it, caveats, where it
comes from) printed underneath; see :mod:`tileparity_lib.figures`.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import figures

KEPT = (0, 158, 115)
DROPPED = (213, 94, 0)
UNITS = (0, 110, 255)
THRESHOLD_COLOURS = [(230, 159, 0), (86, 180, 233), (0, 114, 178), (0, 0, 0)]


def canvas(thumb: np.ndarray, min_width: int = 720) -> tuple[Image.Image, float]:
    """Thumbnail as a PIL image, enlarged (nearest neighbour) if it is narrower than
    ``min_width`` so that tile outlines stay visible; returns (image, enlargement)."""
    img = Image.fromarray(thumb[:, :, :3].astype(np.uint8))
    factor = 1.0
    if img.width < min_width:
        factor = float(np.ceil(min_width / img.width))
        img = img.resize((int(img.width * factor), int(img.height * factor)), Image.NEAREST)
    return img, factor


def units_layer(shape, positions, scale, sx, sy, unit_radius_thumb_px: float) -> np.ndarray:
    """Boolean layer of the in-tissue spatial units (discs for spots, points for bins)."""
    layer = np.zeros(shape, bool)
    if positions is None:
        return layer
    units = positions[positions["in_tissue"] == 1]
    ux = units["x_fullres"].to_numpy() * scale * sx
    uy = units["y_fullres"].to_numpy() * scale * sy
    ok = (ux >= 0) & (uy >= 0) & (ux < shape[1]) & (uy < shape[0])
    ux, uy = ux[ok], uy[ok]
    r = unit_radius_thumb_px
    if r <= 1:
        layer[np.floor(uy).astype(int), np.floor(ux).astype(int)] = True
        return layer
    ri = int(np.ceil(r))
    yy, xx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    for x, y in zip(ux, uy):
        xi, yi = int(np.floor(x)), int(np.floor(y))
        dx, dy = xx + xi + 0.5 - x, yy + yi + 0.5 - y
        d = dx * dx + dy * dy <= r * r
        y0, y1 = max(yi - ri, 0), min(yi + ri + 1, shape[0])
        x0, x1 = max(xi - ri, 0), min(xi + ri + 1, shape[1])
        layer[y0:y1, x0:x1] |= d[y0 - (yi - ri):y1 - (yi - ri), x0 - (xi - ri):x1 - (xi - ri)]
    return layer


def overlay(thumb, tiles, positions, scale, sx, sy, unit_radius_thumb_px, min_tissue, caption: dict) -> Image.Image:
    """Thumbnail with the spatial units (blue tint) and the tiles: kept (green) and those
    with enough tissue that were dropped for another reason (orange)."""
    base = thumb[:, :, :3].astype(np.float32).copy()
    layer = units_layer(base.shape[:2], positions, scale, sx, sy, unit_radius_thumb_px)
    base[layer] = 0.6 * base[layer] + 0.4 * np.array(UNITS, np.float32)
    img, f = canvas(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    width = 2 if f >= 2 else 1
    for t in tiles.itertuples(index=False):
        if not t.keep and t.tissue_fraction < min_tissue:
            continue
        x0, y0 = t.x0_px * sx * f, t.y0_px * sy * f
        x1, y1 = (t.x0_px + t.tile_px) * sx * f, (t.y0_px + t.tile_px) * sy * f
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=KEPT if t.keep else DROPPED, width=width)
    items = [(KEPT, "kept tile"), (DROPPED, "dropped although it has enough tissue")]
    if positions is not None:
        items.insert(0, (UNITS, "in-tissue spatial units (Space Ranger)"))
    fig = figures.stack([img, figures.legend(items, width=img.width)])
    return figures.with_caption(fig, **caption)


def thresholds(thumb, tiles, sx, sy, values: list[float], caption: dict) -> tuple[Image.Image, dict]:
    """Tiles coloured by the highest tissue-fraction threshold they reach."""
    values = sorted(values)
    img, f = canvas(thumb)
    draw = ImageDraw.Draw(img)
    width = 3 if f >= 2 else 2
    frac = tiles["tissue_fraction"].to_numpy(float)
    level = np.full(len(tiles), -1)
    for k, v in enumerate(values):
        level[frac >= v] = k
    order = np.argsort(level)  # draw higher levels last (on top)
    for i in order:
        if level[i] < 0:
            continue
        t = tiles.iloc[i]
        x0, y0 = t["x0_px"] * sx * f, t["y0_px"] * sy * f
        x1, y1 = (t["x0_px"] + t["tile_px"]) * sx * f, (t["y0_px"] + t["tile_px"]) * sy * f
        draw.rectangle([x0 + 1, y0 + 1, x1 - 2, y1 - 2], outline=THRESHOLD_COLOURS[level[i] % 4], width=width)
    counts = {f"{v:g}": int((frac >= v).sum()) for v in values}
    items = []
    for k, v in enumerate(values):
        hi = values[k + 1] if k + 1 < len(values) else None
        in_band = (frac >= v) & (frac < hi) if hi is not None else (frac >= v)
        band = f"{v:.0%} to {hi:.0%}" if hi is not None else f">= {v:.0%}"
        items.append((THRESHOLD_COLOURS[k % 4], f"tissue {band}: {int(in_band.sum())} tiles"))
    width = max(img.width, 600)
    totals = "; ".join(f">= {v:.0%}: {counts[f'{v:g}']} tiles" for v in values)
    fig = figures.stack([img, figures.legend(items, width=width),
                         figures.caption_strip(width, [("Tiles kept by each threshold", totals)])])
    return figures.with_caption(fig, **caption), counts
