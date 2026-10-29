"""Figures of the optional clustering step (``--cluster``), drawn with Pillow like every other
figure of the pipeline, each with its four-part explanation underneath:

* ``<prefix>.cluster_agreement_<set>.png``: adjusted Rand index (ARI) between every pair of
  views at the reference k (Leiden and k-means), and the ARI of every view with the
  reference view (EXPRESSION when present) for every k, with the spatial null;
* ``<prefix>.cluster_maps_<set>.png``: the clusters of every view on the slide, with the
  colours of the matching clusters of the reference view;
* ``<prefix>.cluster_contingency_keep.png``: how the clusters of every view are distributed
  over the clusters of the reference view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import cluster, figures
from .figures import BG, INK, MUTED, plain, text_width

# one colour per view, the same in every figure
VIEW_COLOURS = {"UNI2H_RGB": (0, 114, 178), "UNI2H_RGB_HE": (86, 180, 233), "LBP_RGB": (213, 94, 0),
                "LBP_RGB_HE": (230, 159, 0), "LBP_HE": (204, 121, 167), "INTENSITY_HE": (0, 158, 115),
                "EXPRESSION": (0, 0, 0)}
# cluster colours: strong colours first, then their light versions (Tableau 20 without grey)
CLUSTER_COLOURS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
                   (227, 119, 194), (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
                   (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148), (247, 182, 210),
                   (219, 219, 141), (158, 218, 229), (90, 90, 90), (190, 190, 190)]
SCALE = [(247, 251, 255), (198, 219, 239), (107, 174, 214), (33, 113, 181), (8, 48, 107)]  # 0 .. 1
DIAGONAL = (232, 232, 232)


def cluster_colour(i: int) -> tuple[int, int, int]:
    return CLUSTER_COLOURS[int(i) % len(CLUSTER_COLOURS)]


def scale_colour(v: float) -> tuple[int, int, int]:
    """Sequential blue scale from 0 (white) to 1 (dark blue); values below 0 are white."""
    if not np.isfinite(v):
        return (255, 255, 255)
    t = float(np.clip(v, 0, 1)) * (len(SCALE) - 1)
    i = min(int(t), len(SCALE) - 2)
    f = t - i
    return tuple(int(round(SCALE[i][c] + f * (SCALE[i + 1][c] - SCALE[i][c]))) for c in range(3))


def vertical_text(text: str, fnt, fill=INK) -> Image.Image:
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    box = probe.textbbox((0, 0), plain(text), font=fnt)
    img = Image.new("RGB", (box[2] - box[0] + 4, box[3] - box[1] + 4), BG)
    ImageDraw.Draw(img).text((2 - box[0], 2 - box[1]), plain(text), font=fnt, fill=fill)
    return img.rotate(90, expand=True)


def _centred(draw, x, y, w, h, text, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x + (w - (box[2] - box[0])) / 2 - box[0], y + (h - (box[3] - box[1])) / 2 - box[1]), text,
              font=fnt, fill=fill)


# ---- ARI heatmap -----------------------------------------------------------------------------------------
def heatmap(values: np.ndarray, labels: list[str], title: str, notes=None, diagonal=None,
            cell: int = 62) -> Image.Image:
    """Symmetric matrix of ARI values (blue scale 0 to 1, number in each cell, a small note
    under it), with an optional different quantity on the diagonal (grey cells)."""
    n = len(labels)
    f_lab, f_val, f_note, f_title = figures.font(12), figures.font(14), figures.font(10), figures.font(14)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lw = int(max(text_width(probe, plain(s), f_lab) for s in labels)) + 12
    top = 28 + lw
    W, H = lw + n * cell + 12, top + n * cell + 8
    img = Image.new("RGB", (max(W, int(text_width(probe, plain(title), f_title)) + 16), H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((6, 4), plain(title), font=f_title, fill=INK)
    for j, s in enumerate(labels):
        v = vertical_text(s, f_lab)
        img.paste(v, (lw + j * cell + (cell - v.width) // 2, top - v.height - 4))
    for i, s in enumerate(labels):
        draw.text((lw - 6 - text_width(draw, plain(s), f_lab), top + i * cell + cell / 2 - 8), plain(s), font=f_lab,
                  fill=INK)
    for i in range(n):
        for j in range(n):
            x, y = lw + j * cell, top + i * cell
            v = (diagonal[i] if diagonal is not None else np.nan) if i == j else values[i, j]
            fill = DIAGONAL if i == j else scale_colour(v)
            draw.rectangle([x, y, x + cell - 3, y + cell - 3], fill=fill)
            if not np.isfinite(v):
                continue
            ink = (255, 255, 255) if (i != j and v > 0.55) else INK
            note = notes[i][j] if notes is not None else ""
            if note:
                _centred(draw, x, y + 4, cell - 3, (cell - 3) * 0.55, f"{v:.2f}", f_val, ink)
                _centred(draw, x, y + (cell - 3) * 0.55, cell - 3, (cell - 3) * 0.4, plain(note), f_note,
                         ink if ink != INK else MUTED)
            else:
                _centred(draw, x, y, cell - 3, cell - 3, f"{v:.2f}", f_val, ink)
    return img


def colour_bar(width: int = 260) -> Image.Image:
    f = figures.font(11)
    img = Image.new("RGB", (width + 120, 34), BG)
    draw = ImageDraw.Draw(img)
    for x in range(width):
        draw.line([(60 + x, 4), (60 + x, 16)], fill=scale_colour(x / (width - 1)))
    draw.rectangle([60, 4, 60 + width - 1, 16], outline=MUTED)
    draw.text((8, 3), "ARI", font=f, fill=INK)
    for v in (0, 0.25, 0.5, 0.75, 1):
        x = 60 + v * (width - 1)
        draw.text((x - 8, 19), f"{v:g}", font=f, fill=MUTED)
    return img


# ---- the agreement figure ----------------------------------------------------------------------------------
def agreement_figure(table: pd.DataFrame, info: dict, set_name: str, methods: list[str], caption: dict,
                     prefix: str) -> Image.Image | None:
    views = info["views_clustered"]
    if len(views) < 2:
        return None
    k_ref, ref = info["k_ref"], info["reference_view"]
    t = table[table["set"] == set_name]
    heat, charts = [], []
    for m in methods:
        tm = t[(t["method"] == m) & (t["k"] == k_ref)]
        M = np.full((len(views), len(views)), np.nan)
        notes = [["" for _ in views] for _ in views]
        for r in tm.itertuples(index=False):
            i, j = views.index(r.view_a), views.index(r.view_b)
            M[i, j] = M[j, i] = r.ari
            if np.isfinite(r.null_ari_p95):
                notes[i][j] = notes[j][i] = f"null {r.null_ari_p95:.2f}"
        diag = None
        if len(methods) == 2:
            d = t[(t["method"] == cluster.METHOD_PAIR) & (t["k"] == k_ref)].set_index("view_a")["ari"]
            diag = [float(d.get(v, np.nan)) for v in views]
        name = "Leiden" if m == "leiden" else "k-means"
        heat.append(heatmap(M, views, f"{name}, k = {k_ref}: ARI between views", notes, diag))
        series = []
        for v in views:
            if v == ref:
                continue
            a, b = sorted((v, ref), key=views.index)
            rows = t[(t["method"] == m) & (t["view_a"] == a) & (t["view_b"] == b)].sort_values("k")
            rows = rows[(rows["k_a"] == rows["k"]) & (rows["k_b"] == rows["k"])]  # same k in both partitions
            s = {"name": v, "x": rows["k"].to_numpy(float), "y": rows["ari"].to_numpy(float),
                 "colour": VIEW_COLOURS.get(v, figures.SERIES[len(series) % len(figures.SERIES)])}
            if rows["null_ari_p95"].notna().all() and len(rows):
                s["lo"], s["hi"] = rows["null_ari_p05"].to_numpy(float), rows["null_ari_p95"].to_numpy(float)
            series.append(s)
        width = max(heat[-1].width, 460)
        if any(len(s["x"]) for s in series):
            # a fixed scale, comparable between slides; slightly negative null values are drawn at 0
            lo = min([0.0] + [float(np.nanmin(np.r_[s["y"], s.get("lo", s["y"])])) for s in series if len(s["x"])])
            y0 = 0.0 if lo >= -0.05 else -0.25
            for s in series:
                if "lo" in s:
                    s["lo"] = np.maximum(s["lo"], y0)
            chart = figures.line_chart(series, "k (number of clusters)", f"ARI with {ref}", size=(width, 330),
                                       y_range=(y0, 1.0), x_ticks=info["k"])
        else:
            chart = Image.new("RGB", (width, 60), BG)
            ImageDraw.Draw(chart).text((10, 20), "no k reached by both clusterings", font=figures.font(13), fill=MUTED)
        charts.append(figures.stack([figures.title_strip(width, f"{name}: ARI of every view with {ref}, by k", 14),
                                     chart]))
    row1, row2 = _side_by_side(heat), _side_by_side(charts)
    items = [(VIEW_COLOURS.get(v, figures.SERIES[0]), v) for v in views if v != ref]
    if info["null"]["moves"]:
        items.append(((200, 200, 200), "shaded: 5-95 % of the spatial null"))
    body = figures.stack([row1, colour_bar(), row2, figures.legend(items, width=max(row1.width, row2.width))])
    n = info["n_tiles"]
    return figures.with_caption(body, title=f"Agreement between feature views - {prefix}, set '{set_name}' ({n} tiles)",
                                **caption)


def _side_by_side(images: list[Image.Image], pad: int = 16) -> Image.Image:
    W = sum(i.width for i in images) + pad * (len(images) + 1)
    H = max(i.height for i in images) + 2 * pad // 2
    out = Image.new("RGB", (W, H), BG)
    x = pad
    for i in images:
        out.paste(i, (x, pad // 2))
        x += i.width + pad
    return out


# ---- maps ----------------------------------------------------------------------------------------------------
def _thumbnail_frame(sample: dict, tiles: pd.DataFrame, margin: float = 0.03):
    """(faded thumbnail crop or white canvas, sx, sy, x offset, y offset) covering the tiles."""
    thumb = sample.get("thumbnail")
    meta = (sample.get("provenance") or {}).get("prepare", {}).get("thumbnail", {}) or {}
    x0 = tiles["x0_px"].to_numpy(float)
    y0 = tiles["y0_px"].to_numpy(float)
    T = tiles["tile_px"].to_numpy(float)
    if thumb is not None and "resampled_to_thumbnail_x" in meta:
        sx, sy = float(meta["resampled_to_thumbnail_x"]), float(meta["resampled_to_thumbnail_y"])
    else:
        thumb = None
        extent = max(float((x0 + T).max()), float((y0 + T).max()), 1.0)
        sx = sy = min(1.0, 1200.0 / extent)
    left, top = x0.min() * sx, y0.min() * sy
    right, bottom = (x0 + T).max() * sx, (y0 + T).max() * sy
    mx, my = margin * (right - left), margin * (bottom - top)
    left, top = int(max(0, np.floor(left - mx))), int(max(0, np.floor(top - my)))
    right, bottom = int(np.ceil(right + mx)), int(np.ceil(bottom + my))
    if thumb is not None:
        right, bottom = min(right, thumb.shape[1]), min(bottom, thumb.shape[0])
        crop = thumb[top:bottom, left:right, :3].astype(np.float32)
        grey = crop @ np.array([0.299, 0.587, 0.114], np.float32)
        faded = 255 - 0.3 * (255 - grey)
        base = Image.fromarray(np.repeat(faded[:, :, None], 3, axis=2).astype(np.uint8))
    else:
        base = Image.new("RGB", (max(right - left, 1), max(bottom - top, 1)), (250, 250, 250))
    return base, sx, sy, left, top


def cluster_map(sample: dict, tiles: pd.DataFrame, colours: list[tuple], width: int) -> Image.Image:
    base, sx, sy, ox, oy = _thumbnail_frame(sample, tiles)
    draw = ImageDraw.Draw(base)
    for (x0, y0, T), c in zip(tiles[["x0_px", "y0_px", "tile_px"]].to_numpy(float), colours):
        draw.rectangle([x0 * sx - ox, y0 * sy - oy, (x0 + T) * sx - ox - 1, (y0 + T) * sy - oy - 1], fill=c)
    h = max(1, int(round(base.height * width / max(base.width, 1))))
    return base.resize((width, h), Image.NEAREST if width >= base.width else Image.BOX)


def maps_figure(sample: dict, result: dict, set_name: str, caption: dict, prefix: str,
                cell_width: int = 330) -> Image.Image | None:
    info = result["sets"][set_name]
    views, ref, k = info["views_clustered"], info["reference_view"], info["k_ref"]
    if not views:
        return None
    method = "leiden" if any(key[1] == "leiden" for key in result["labels"][set_name]) else "kmeans"
    labels = result["labels"][set_name]
    tiles = sample["tiles"][result["masks"][set_name]]
    ref_lab = labels[(ref, method, k)]
    cells, texts, used = [], [], set()
    for v in views:
        lab = labels[(v, method, k)]
        if v == ref:
            colour_of = {int(c): int(c) for c in np.unique(lab)}
            text = f"{v} (reference): {cluster.n_clusters(lab)} clusters"
        else:
            colour_of = cluster.match_to_reference(ref_lab, lab)
            text = f"{v}: {cluster.n_clusters(lab)} clusters, ARI {cluster.ari(lab, ref_lab):.2f} with {ref}"
        used.update(colour_of.values())
        cells.append(cluster_map(sample, tiles, [cluster_colour(colour_of[int(c)]) for c in lab], cell_width))
        texts.append(text)
    grid = figures.grid(cells, texts, ncols=min(4, len(cells)), cell_width=cell_width)
    sizes = np.bincount(ref_lab)
    items = [(cluster_colour(c), f"{ref} cluster {c} ({sizes[c]} tile{'' if sizes[c] == 1 else 's'})")
             for c in range(len(sizes))]
    items += [(cluster_colour(c), "cluster with no partner in the reference") for c in sorted(used) if c >= len(sizes)]
    body = figures.stack([grid, figures.legend(items, width=grid.width)])
    name = "Leiden" if method == "leiden" else "k-means"
    return figures.with_caption(body, title=(f"Clusters of every view ({name}, k = {k}) - {prefix}, set '{set_name}' "
                                             f"({info['n_tiles']} tiles)"), **caption)


# ---- contingency ---------------------------------------------------------------------------------------------------
def contingency_panel(lab: np.ndarray, ref_lab: np.ndarray, view: str, ref: str, cell: int = 34) -> Image.Image:
    """Rows: the view's clusters (ordered by the reference cluster they are matched to);
    columns: the reference clusters; cell: percentage of the row's tiles."""
    M = cluster.contingency(lab, ref_lab)
    mapping = cluster.match_to_reference(ref_lab, lab)
    order = sorted(range(M.shape[0]), key=lambda i: (mapping[i], i))
    R = M[order] / np.maximum(M[order].sum(axis=1, keepdims=True), 1)
    f_lab, f_val, f_title = figures.font(11), figures.font(11), figures.font(13)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    row_text = [f"{i} ({M[i].sum()})" for i in order]
    lw = int(max(text_width(probe, s, f_lab) for s in row_text)) + 24
    title = f"{view} vs {ref}: ARI {cluster.ari(lab, ref_lab):.2f}"
    top = 46
    W = max(lw + M.shape[1] * cell + 10, int(text_width(probe, title, f_title)) + 12)
    H = top + len(order) * cell + 22
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), plain(title), font=f_title, fill=INK)
    draw.text((4, 20), plain(f"rows: {view} clusters (tiles); columns: {ref} clusters"), font=figures.font(10),
              fill=MUTED)
    for j in range(M.shape[1]):
        x = lw + j * cell
        draw.rectangle([x + cell // 2 - 6, top - 12, x + cell // 2 + 5, top - 3], fill=cluster_colour(j))
        _centred(draw, x, H - 20, cell, 16, str(j), f_lab, INK)
    for r, i in enumerate(order):
        y = top + r * cell
        draw.rectangle([4, y + cell // 2 - 6, 15, y + cell // 2 + 5], fill=cluster_colour(mapping[i]))
        draw.text((20, y + cell / 2 - 7), row_text[r], font=f_lab, fill=INK)
        for j in range(M.shape[1]):
            v = R[r, j]
            x = lw + j * cell
            draw.rectangle([x, y, x + cell - 2, y + cell - 2], fill=scale_colour(v))
            if v >= 0.01:
                _centred(draw, x, y, cell - 2, cell - 2, f"{100 * v:.0f}", f_val, (255, 255, 255) if v > 0.55 else INK)
    return img


def contingency_figure(result: dict, set_name: str, caption: dict, prefix: str) -> Image.Image | None:
    info = result["sets"][set_name]
    views, ref, k = info["views_clustered"], info["reference_view"], info["k_ref"]
    others = [v for v in views if v != ref]
    if not others:
        return None
    method = "leiden" if any(key[1] == "leiden" for key in result["labels"][set_name]) else "kmeans"
    labels = result["labels"][set_name]
    panels = [contingency_panel(labels[(v, method, k)], labels[(ref, method, k)], v, ref) for v in others]
    ncols = min(3, len(panels))
    rows = [_side_by_side(panels[i:i + ncols]) for i in range(0, len(panels), ncols)]
    name = "Leiden" if method == "leiden" else "k-means"
    title = f"Clusters of every view against {ref} ({name}, k = {k}) - {prefix}, set '{set_name}'"
    return figures.with_caption(figures.stack(rows), title=title, **caption)
