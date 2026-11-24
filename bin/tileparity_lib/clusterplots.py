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
                "LBP_RGB_HE": (230, 159, 0), "LBP_HE": (204, 121, 167), "LBP_H": (117, 77, 160),
                "LBP_E": (160, 30, 80), "INTENSITY_HE": (0, 158, 115), "EXPRESSION": (0, 0, 0)}
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


def colour_bar(width: int = 260, label: str = "ARI", top: float = 1.0) -> Image.Image:
    """The blue scale from 0 to ``top`` (values above ``top`` take the darkest colour)."""
    f = figures.font(11)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x0 = max(60, int(text_width(probe, plain(label), f)) + 18)
    img = Image.new("RGB", (width + x0 + 60, 34), BG)
    draw = ImageDraw.Draw(img)
    for x in range(width):
        draw.line([(x0 + x, 4), (x0 + x, 16)], fill=scale_colour(x / (width - 1)))
    draw.rectangle([x0, 4, x0 + width - 1, 16], outline=MUTED)
    draw.text((8, 3), plain(label), font=f, fill=INK)
    for v in (0, 0.25, 0.5, 0.75, 1):
        x = x0 + v * (width - 1)
        t = f"{v * top:g}" + ("+" if v == 1 and top != 1 else "")
        draw.text((x - 8, 19), t, font=f, fill=MUTED)
    return img


# ---- the agreement figure ----------------------------------------------------------------------------------
def _matrix(t: pd.DataFrame, views: list[str], value: str) -> np.ndarray:
    M = np.full((len(views), len(views)), np.nan)
    for r in t.itertuples(index=False):
        if r.view_a in views and r.view_b in views:
            i, j = views.index(r.view_a), views.index(r.view_b)
            M[i, j] = M[j, i] = getattr(r, value)
    return M


def agreement_figure(table: pd.DataFrame, info: dict, set_name: str, methods: list[str], caption: dict,
                     prefix: str) -> Image.Image | None:
    """Per method: ARI between every pair of views at the k shown (with its spatial null), the
    ARI averaged over every k (with the corrected ARI), and the ARI of every view with the
    reference view by k (with the null band and the k shown marked)."""
    views = info["views_clustered"]
    if len(views) < 2:
        return None
    ref = info["reference_view"]
    k_show = info.get("k_show") or {m: info["k_ref"] for m in methods}
    auto = str(info.get("k_show_rule", "")).startswith("auto")
    t = table[table["set"] == set_name]
    same = t[(t["k_a"] == t["k"]) & (t["k_b"] == t["k"])]
    heat, means, charts = [], [], []
    for m in methods:
        k = k_show[m]
        name = "Leiden" if m == "leiden" else "k-means"
        tm = t[(t["method"] == m) & (t["k"] == k)]
        M = _matrix(tm, views, "ari")
        notes = [["" for _ in views] for _ in views]
        for r in tm.itertuples(index=False):
            if r.view_a in views and r.view_b in views and np.isfinite(r.null_ari_p95):
                i, j = views.index(r.view_a), views.index(r.view_b)
                notes[i][j] = notes[j][i] = (f"null {r.null_ari_p95:.2f}" if r.ari > r.null_ari_p95
                                             else f"<= null {r.null_ari_p95:.2f}")
        diag = None
        if len(methods) == 2:
            d = t[(t["method"] == cluster.METHOD_PAIR) & (t["k"] == k) & t["view_a"].isin(views)]
            d = d.set_index("view_a")["ari"]
            diag = [float(d.get(v, np.nan)) for v in views]
        why = " (best agreement with " + str(ref) + ")" if auto else ""
        heat.append(heatmap(M, views, f"{name}, k = {k}{why}: ARI between views", notes, diag))
        sm = same[same["method"] == m]
        g = sm.groupby(["view_a", "view_b"], sort=False)
        mean = g["ari"].mean().reset_index()
        corr = g["ari_corrected"].mean().reset_index() if "ari_corrected" in sm else None
        Mm = _matrix(mean, views, "ari")
        notes_m = [["" for _ in views] for _ in views]
        if corr is not None:
            for r in corr.itertuples(index=False):
                if r.view_a in views and r.view_b in views and np.isfinite(r.ari_corrected):
                    i, j = views.index(r.view_a), views.index(r.view_b)
                    notes_m[i][j] = notes_m[j][i] = f"corr. {r.ari_corrected:.2f}"
        diag_m = None
        if len(methods) == 2:
            d = t[(t["method"] == cluster.METHOD_PAIR) & t["view_a"].isin(views)].groupby("view_a")["ari"].mean()
            diag_m = [float(d.get(v, np.nan)) for v in views]
        ks = ", ".join(map(str, info["k"]))
        means.append(heatmap(Mm, views, f"{name}, mean over k = {ks}", notes_m, diag_m))
        series = []
        for v in views:
            if v == ref:
                continue
            a, b = sorted((v, ref), key=views.index)
            rows = sm[(sm["view_a"] == a) & (sm["view_b"] == b)].sort_values("k")
            s = {"name": v, "x": rows["k"].to_numpy(float), "y": rows["ari"].to_numpy(float),
                 "colour": VIEW_COLOURS.get(v, figures.SERIES[len(series) % len(figures.SERIES)])}
            if rows["null_ari_p95"].notna().all() and len(rows):
                s["lo"], s["hi"] = rows["null_ari_p05"].to_numpy(float), rows["null_ari_p95"].to_numpy(float)
            series.append(s)
        width = max(heat[-1].width, means[-1].width, 460)
        if any(len(s["x"]) for s in series):
            # a fixed scale, comparable between slides; slightly negative null values are drawn at 0
            lo = min([0.0] + [float(np.nanmin(np.r_[s["y"], s.get("lo", s["y"])])) for s in series if len(s["x"])])
            y0 = 0.0 if lo >= -0.05 else -0.25
            for s in series:
                if "lo" in s:
                    s["lo"] = np.maximum(s["lo"], y0)
            chart = figures.line_chart(series, "k (number of clusters)", f"ARI with {ref}", size=(width, 330),
                                       y_range=(y0, 1.0), x_ticks=info["k"], v_lines=[(k, f"k shown = {k}")])
        else:
            chart = Image.new("RGB", (width, 60), BG)
            ImageDraw.Draw(chart).text((10, 20), "no k reached by both clusterings", font=figures.font(13), fill=MUTED)
        charts.append(figures.stack([figures.title_strip(width, f"{name}: ARI of every view with {ref}, by k", 14),
                                     chart]))
    row1, row2, row3 = _side_by_side(heat), _side_by_side(means), _side_by_side(charts)
    items = [(VIEW_COLOURS.get(v, figures.SERIES[0]), v) for v in views if v != ref]
    if info["null"]["moves"]:
        items.append(((200, 200, 200), "shaded: 5-95 % of the spatial null"))
    width = max(row1.width, row2.width, row3.width)
    body = figures.stack([row1, row2, colour_bar(), row3, figures.legend(items, width=width)])
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
    views, ref = info["views_clustered"], info["reference_view"]
    if not views:
        return None
    method = "leiden" if any(key[1] == "leiden" for key in result["labels"][set_name]) else "kmeans"
    k = (info.get("k_show") or {}).get(method, info["k_ref"])
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
    views, ref = info["views_clustered"], info["reference_view"]
    others = [v for v in views if v != ref]
    if not others:
        return None
    method = "leiden" if any(key[1] == "leiden" for key in result["labels"][set_name]) else "kmeans"
    k = (info.get("k_show") or {}).get(method, info["k_ref"])
    labels = result["labels"][set_name]
    panels = [contingency_panel(labels[(v, method, k)], labels[(ref, method, k)], v, ref) for v in others]
    ncols = min(3, len(panels))
    rows = [_side_by_side(panels[i:i + ncols]) for i in range(0, len(panels), ncols)]
    name = "Leiden" if method == "leiden" else "k-means"
    title = f"Clusters of every view against {ref} ({name}, k = {k}) - {prefix}, set '{set_name}'"
    return figures.with_caption(figures.stack(rows), title=title, **caption)


# ---- the LBP recipes (--cluster_lbp_recipes) ------------------------------------------------------------------
RECIPE_COLOURS = {"hellinger": (0, 114, 178), "scanpy": (230, 159, 0), "zscore": (0, 158, 115)}


def recipes_figure(table: pd.DataFrame, info: dict, set_name: str, methods: list[str], caption: dict,
                   prefix: str) -> Image.Image | None:
    """For every LBP view: ARI with the reference view by k, one line per preparation of
    the LBP histograms; one row of panels per clustering method."""
    alternatives = info.get("recipe_views_clustered") or []
    ref = info["reference_view"]
    bases = [v for v in info["views_clustered"] if v.startswith("LBP")
             and any(cluster.split_variant(a)[0] == v for a in alternatives)]
    if not bases or ref is None:
        return None
    t = table[table["set"] == set_name]
    order = list(info["views_clustered"]) + list(alternatives)
    recipes = [r for r in cluster.LBP_RECIPES if r == "hellinger" or any(a.endswith("." + r) for a in alternatives)]
    cells, labels = [], []
    for m in methods:
        name = "Leiden" if m == "leiden" else "k-means"
        for base in bases:
            series = []
            for r in recipes:
                v = cluster.variant(base, r)
                if v not in order:
                    continue
                a, b = sorted((v, ref), key=order.index)
                rows = t[(t["method"] == m) & (t["view_a"] == a) & (t["view_b"] == b)].sort_values("k")
                rows = rows[(rows["k_a"] == rows["k"]) & (rows["k_b"] == rows["k"])]
                series.append({"name": r, "x": rows["k"].to_numpy(float), "y": rows["ari"].to_numpy(float),
                               "colour": RECIPE_COLOURS[r]})
            if not any(len(s["x"]) for s in series):
                continue
            lo = min([0.0] + [float(np.nanmin(s["y"])) for s in series if len(s["x"])])
            cells.append(figures.line_chart(series, "k (number of clusters)", f"ARI with {ref}", size=(380, 280),
                                            y_range=(0.0 if lo >= -0.05 else -0.25, 1.0), x_ticks=info["k"]))
            labels.append(f"{base} - {name}")
    if not cells:
        return None
    grid = figures.grid(cells, labels, ncols=len(bases), cell_width=380)
    items = [(RECIPE_COLOURS[r], f"{r}: {cluster.RECIPE_SHORT[r]}") for r in recipes]
    body = figures.stack([grid, figures.legend(items, width=grid.width)])
    return figures.with_caption(body, title=(f"Three preparations of the LBP histograms - {prefix}, set '{set_name}' "
                                             f"({info['n_tiles']} tiles)"), min_width=1100, **caption)


# ---- redundancy between views (--redundancy) -------------------------------------------------------------------
def redundancy_figure(red: dict, caption: dict, prefix: str, set_name: str) -> Image.Image | None:
    """Left: R^2 of every view (columns) predicted from every other view (rows). Right: R^2
    of each principal component of the target, for the pairs on identical pixels."""
    table, pcs, info = red.get("table"), red.get("pcs"), red["info"]
    if table is None or table.empty:
        return None
    views = info["views"]
    M = np.full((len(views), len(views)), np.nan)
    notes = [["" for _ in views] for _ in views]
    for r in table.itertuples(index=False):
        i, j = views.index(r.predictor), views.index(r.target)
        M[i, j] = r.r2
        if np.isfinite(getattr(r, "r2_ci_low", np.nan)):
            notes[i][j] = f"{r.r2_ci_low:.2f}-{r.r2_ci_high:.2f}"
    heat = heatmap(M, views, "R2 of the column view predicted from the row view (95 % interval)", notes=notes, cell=64)
    series = []
    for a, b in info.get("map_pairs", []):
        rows = pcs[(pcs["predictor"] == a) & (pcs["target"] == b)].sort_values("pc")
        # out-of-sample R2 has no lower bound; values below -0.25 are drawn at -0.25
        series.append({"name": f"{b} from {a}", "x": rows["pc"].to_numpy(float),
                       "y": np.maximum(rows["r2"].to_numpy(float), -0.25),
                       "colour": VIEW_COLOURS.get(b, figures.SERIES[len(series) % len(figures.SERIES)])})
    parts = [heat]
    if series:
        n = int(max(s["x"].max() for s in series if len(s["x"])))
        ticks = [t for t in (1, 5, 10, 20, 30, 40, 50, 75, 100) if t <= n]
        chart = figures.line_chart(series, "principal component of the predicted view", "R2", size=(560, 360),
                                   y_range=(-0.25, 1.0), x_ticks=ticks)
        legend = figures.legend([(s["colour"], s["name"]) for s in series], width=560)
        parts.append(figures.stack([figures.title_strip(560, "R2 of each component, UNI2-h and LBP on the same pixels",
                                                        14), chart, legend]))
    body = figures.stack([_side_by_side(parts), colour_bar(label="R2")])
    return figures.with_caption(body, title=(f"How much of each view another view predicts - {prefix}, set "
                                             f"'{set_name}' ({info['n_tiles']} tiles)"), min_width=1100, **caption)


def redundancy_maps_figure(sample: dict, tiles: pd.DataFrame, red: dict, caption: dict, prefix: str,
                           cell_width: int = 330, top: float = 2.0) -> Image.Image | None:
    """The H&E thumbnail and, for every drawn pair, the share of the target view that the
    predictor does not predict, per tile (blue scale from 0 to ``top`` times the average)."""
    per_tile = red.get("per_tile") or {}
    if not per_tile:
        return None
    cells, texts = [], []
    thumb = sample.get("thumbnail")
    if thumb is not None:
        base, _, _, ox, oy = _thumbnail_frame(sample, tiles)
        crop = Image.fromarray(np.asarray(thumb)[oy:oy + base.height, ox:ox + base.width, :3])
        h = max(1, int(round(crop.height * cell_width / max(crop.width, 1))))
        cells.append(crop.resize((cell_width, h), Image.LANCZOS))
        texts.append("H&E (thumbnail)")
    for (a, b), u in per_tile.items():
        colours = [scale_colour(v / top) if np.isfinite(v) else (255, 255, 255) for v in u]
        cells.append(cluster_map(sample, tiles, colours, cell_width))
        r2 = 1 - float(np.nanmean(u))
        texts.append(f"{b} not predicted from {a} (R2 {r2:.2f})")
    grid = figures.grid(cells, texts, ncols=min(5, len(cells)), cell_width=cell_width)
    body = figures.stack([grid, colour_bar(label="unexplained", top=top)])
    return figures.with_caption(body, title=f"Where each view has what the other one does not predict - {prefix}",
                                min_width=1100, **caption)


# ---- F1 per domain -----------------------------------------------------------------------------------------------
def rect_heatmap(values: np.ndarray, row_labels: list[str], col_labels: list[str], title: str,
                 cell: int = 54) -> Image.Image:
    """Matrix of values between 0 and 1 (blue scale, number in each cell)."""
    f_lab, f_val, f_title = figures.font(12), figures.font(13), figures.font(14)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lw = int(max(text_width(probe, plain(s), f_lab) for s in row_labels)) + 14
    th = int(max(text_width(probe, plain(s), f_lab) for s in col_labels)) + 10
    top = 28 + th
    W = max(lw + len(col_labels) * cell + 12, int(text_width(probe, plain(title), f_title)) + 16)
    H = top + len(row_labels) * cell + 8
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((6, 4), plain(title), font=f_title, fill=INK)
    for j, s in enumerate(col_labels):
        v = vertical_text(s, f_lab)
        img.paste(v, (lw + j * cell + (cell - v.width) // 2, top - v.height - 4))
    for i, s in enumerate(row_labels):
        draw.text((lw - 6 - text_width(draw, plain(s), f_lab), top + i * cell + cell / 2 - 8), plain(s), font=f_lab,
                  fill=INK)
        for j in range(len(col_labels)):
            v = values[i, j]
            x, y = lw + j * cell, top + i * cell
            draw.rectangle([x, y, x + cell - 3, y + cell - 3], fill=scale_colour(v))
            if np.isfinite(v):
                _centred(draw, x, y, cell - 3, cell - 3, f"{v:.2f}", f_val, (255, 255, 255) if v > 0.55 else INK)
    return img


def f1_figure(f1: pd.DataFrame, info: dict, set_name: str, methods: list[str], caption: dict,
              prefix: str) -> Image.Image | None:
    """Per method: F1 of every domain of the reference view for every other view at the k
    shown, and the macro-F1 (mean over domains) of every view by k."""
    if f1.empty:
        return None
    ref = info["reference_view"]
    views = [v for v in info["views_clustered"] if v != ref]
    t = f1[(f1["set"] == set_name) & f1["view"].isin(views)]
    if t.empty:
        return None
    heat, charts = [], []
    for m in methods:
        k = (info.get("k_show") or {}).get(m, info["k_ref"])
        name = "Leiden" if m == "leiden" else "k-means"
        tk = t[(t["method"] == m) & (t["k"] == k)]
        if tk.empty:
            continue
        domains = sorted(tk["domain"].unique())
        sizes = tk.groupby("domain")["n_domain"].first()
        M = np.full((len(domains), len(views)), np.nan)
        for r in tk.itertuples(index=False):
            M[domains.index(r.domain), views.index(r.view)] = r.f1
        rows = [f"{ref} {d} ({int(sizes[d])})" for d in domains]
        heat.append(rect_heatmap(M, rows, views, f"{name}, k = {k}: F1 of every {ref} domain"))
        mac = cluster.macro_f1(t[t["method"] == m])
        series = []
        for v in views:
            mv = mac[mac["view"] == v].sort_values("k")
            if len(mv):
                series.append({"name": v, "x": mv["k"].to_numpy(float), "y": mv["macro_f1"].to_numpy(float),
                               "colour": VIEW_COLOURS.get(v, figures.SERIES[len(series) % len(figures.SERIES)])})
        width = max(heat[-1].width, 460)
        chart = figures.line_chart(series, "k (number of clusters)", "macro-F1 (mean over domains)", size=(width, 300),
                                   y_range=(0.0, 1.0), x_ticks=info["k"], v_lines=[(k, f"k shown = {k}")])
        charts.append(figures.stack([figures.title_strip(width, f"{name}: macro-F1 of every view, by k", 14), chart]))
    if not heat:
        return None
    items = [(VIEW_COLOURS.get(v, figures.SERIES[0]), v) for v in views]
    row1, row2 = _side_by_side(heat), _side_by_side(charts)
    body = figures.stack([row1, colour_bar(label="F1"), row2, figures.legend(items, width=max(row1.width, row2.width))])
    return figures.with_caption(body, title=f"Which {ref} domains each view finds (F1) - {prefix}, set '{set_name}'",
                                **caption)


# ---- stability under subsampling -------------------------------------------------------------------------------------
def stability_figure(summary: pd.DataFrame, info: dict, set_name: str, methods: list[str], caption: dict,
                     prefix: str) -> Image.Image | None:
    """Per method: the ARI of every view with the reference view on subsamples (mean, band =
    mean +- standard deviation), and the ARI of every view's subsample clustering with its
    clustering of all tiles."""
    if summary.empty:
        return None
    ref = info["reference_view"]
    views = info["views_clustered"]
    t = summary[summary["set"] == set_name]
    panels = []
    for m in methods:
        name = "Leiden" if m == "leiden" else "k-means"
        k = (info.get("k_show") or {}).get(m, info["k_ref"])
        tm = t[t["method"] == m]
        out = []
        for col, label, keep in (("ari_with_reference", f"ARI with {ref} (subsamples)", [v for v in views if v != ref]),
                                 ("ari_with_full", "ARI with the clustering of all tiles", views)):
            series = []
            for v in keep:
                r = tm[tm["view"] == v].sort_values("k")
                if not len(r) or r[f"{col}_mean"].isna().all():
                    continue
                y = r[f"{col}_mean"].to_numpy(float)
                sd = np.nan_to_num(r[f"{col}_sd"].to_numpy(float))
                series.append({"name": v, "x": r["k"].to_numpy(float), "y": y, "lo": np.clip(y - sd, -0.25, 1),
                               "hi": np.clip(y + sd, -0.25, 1),
                               "colour": VIEW_COLOURS.get(v, figures.SERIES[len(series) % len(figures.SERIES)])})
            if series:
                chart = figures.line_chart(series, "k (number of clusters)", label, size=(520, 300),
                                           y_range=(0.0, 1.0), x_ticks=info["k"], v_lines=[(k, f"k shown = {k}")])
                out.append(figures.stack([figures.title_strip(520, f"{name}: {label}", 14), chart]))
        if out:
            panels.append(_side_by_side(out))
    if not panels:
        return None
    items = [(VIEW_COLOURS.get(v, figures.SERIES[0]), v) for v in views]
    body = figures.stack(panels + [figures.legend(items, width=max(p.width for p in panels))])
    return figures.with_caption(body, title=f"How stable the clusterings are - {prefix}, set '{set_name}'", **caption)


# ---- relevance (--relevance) -----------------------------------------------------------------------------------------
def relevance_figure(rel: dict, caption: dict, prefix: str, set_name: str = "keep") -> Image.Image | None:
    """a: R^2 of the expression's components from every image view (95 % interval, spatial
    null). b: for pairs of views, what each adds to the other and what they share."""
    table, part = rel.get("table"), rel.get("partition")
    if table is None or table.empty:
        return None
    views = table[table["kind"] == "view"]
    rows = []
    for r in views.itertuples(index=False):
        frac = getattr(r, "fraction", np.nan)
        text = f"{r.r2:.2f}" + (f" ({frac:.0%} of the reachable)" if np.isfinite(frac) else "")
        rows.append({"label": r.predictor, "segments": [(r.r2, VIEW_COLOURS.get(r.predictor, figures.SERIES[0]))],
                     "lo": r.r2_ci_low, "hi": r.r2_ci_high, "marker": r.r2_null_p95, "text": text})
    lo = min(0.0, float(np.nanmin(views["r2"])) - 0.05)
    a = figures.hbar_chart(rows, "R2 of the expression's components", x_range=(max(lo, -0.25), 1.0), width=700)
    top = [figures.stack([figures.title_strip(700, "a. Expression predicted from every image view", 14), a])]
    pcs = [c for c in table.columns if c.startswith("r2_pc")]
    series = []
    for r in views.itertuples(index=False):
        y = np.array([getattr(r, c) for c in pcs], float)
        if np.isfinite(y).any():
            series.append({"name": r.predictor, "x": np.arange(1, len(pcs) + 1, dtype=float), "y": y,
                           "colour": VIEW_COLOURS.get(r.predictor, figures.SERIES[len(series) % len(figures.SERIES)])})
    if series:
        lo_pc = float(np.nanmin(np.concatenate([s["y"] for s in series])))
        chart = figures.line_chart(series, "expression component", "R2 of the component", size=(420, 300),
                                   y_range=(0.0 if lo_pc >= 0 else -0.25, 1.0), x_ticks=list(range(1, len(pcs) + 1)))
        items = [(s["colour"], s["name"]) for s in series]
        top.append(figures.stack([figures.title_strip(420, "c. Per component (PC1 = largest)", 14), chart,
                                  figures.legend(items, width=420, size=11)]))
    parts = [_side_by_side(top)]
    if part is not None and len(part):
        rows_b = []
        for r in part.itertuples(index=False):
            ca = VIEW_COLOURS.get(r.view_a, figures.SERIES[0])
            cb = VIEW_COLOURS.get(r.view_b, figures.SERIES[1])
            rows_b.append({"label": f"{r.view_a} + {r.view_b}",
                           "segments": [(max(r.unique_a, 0.0), ca), (max(r.shared, 0.0), (190, 190, 190)),
                                        (max(r.unique_b, 0.0), cb)],
                           "text": f"only A {r.unique_a:.2f}, both {r.shared:.2f}, only B {r.unique_b:.2f}"})
        b = figures.hbar_chart(rows_b, "R2 of the expression's components, split", x_range=(0.0, 1.0), width=980)
        items = [((190, 190, 190), "predicted by both A and B"), ((120, 120, 120), "only A: left colour; only B: right colour")]
        parts.append(figures.stack([figures.title_strip(980, "b. What each view adds to the other (A + B)", 14), b,
                                    figures.legend(items, width=980)]))
    body = figures.stack(parts)
    return figures.with_caption(body, title=(f"How much of the expression the image predicts - {prefix}, set "
                                             f"'{set_name}' ({rel['info']['n_tiles']} tiles)"), min_width=1000,
                                **caption)




GENE_CATEGORY_COLOURS = {"both": (117, 77, 160), "only_y": (213, 94, 0), "only_x": (0, 114, 178),
                         "neither": (205, 205, 205)}


def relevance_genes_figure(rel: dict, caption: dict, prefix: str, x_view: str = "UNI2H_RGB", y_view: str = "LBP_RGB",
                           n_labels: int = 12) -> Image.Image | None:
    """a: every gene's R^2 from two views, coloured by which views beat the per-gene spatial
    null (both / only one / neither). b: the genes the second view predicts best, with the
    first view's R^2 marked."""
    from .relevance import gene_category

    g = rel.get("genes")
    if g is None or f"r2_{x_view}" not in g or f"r2_{y_view}" not in g:
        return None
    x, y = g[f"r2_{x_view}"].to_numpy(float), g[f"r2_{y_view}"].to_numpy(float)
    qy = g.get(f"q_{y_view}", pd.Series(np.nan, index=g.index)).to_numpy(float)
    qx = g.get(f"q_{x_view}", pd.Series(np.nan, index=g.index)).to_numpy(float)
    cat = gene_category(qx, qy)
    colours = [GENE_CATEGORY_COLOURS[c] for c in cat]
    counts = {c: int((cat == c).sum()) for c in GENE_CATEGORY_COLOURS}
    names = g["gene_name"].astype(str).to_numpy()
    top_y = list(np.argsort(-np.nan_to_num(y, nan=-9))[:n_labels])
    adds = g.get(f"adds_{y_view}_to_{x_view}")
    top_add = list(np.argsort(-np.nan_to_num(adds.to_numpy(float), nan=-9))[:n_labels // 2]) if adds is not None else []
    labels = [(int(i), names[i]) for i in dict.fromkeys(top_y + top_add)]
    hi = float(np.nanmax(np.r_[x, y, 0.1]))
    lo = max(-0.2, float(np.nanmin(np.r_[x, y, 0.0])))
    a = figures.scatter_chart(x, y, colours, f"R2 of the gene from {x_view}", f"R2 of the gene from {y_view}",
                              size=(620, 560), x_range=(lo, hi + 0.05), y_range=(lo, hi + 0.05), labels=labels)
    items = [(GENE_CATEGORY_COLOURS["both"], f"beats the null from both ({counts['both']})"),
             (GENE_CATEGORY_COLOURS["only_y"], f"only from {y_view} ({counts['only_y']})"),
             (GENE_CATEGORY_COLOURS["only_x"], f"only from {x_view} ({counts['only_x']})"),
             (GENE_CATEGORY_COLOURS["neither"], f"from neither ({counts['neither']})")]
    left = figures.stack([figures.title_strip(620, f"a. Every gene: from {x_view} and from {y_view}", 14), a,
                          figures.legend(items, width=620)])
    rows = []
    for i in top_y[:15]:
        t = f"{x_view} {x[i]:.2f}"
        if "r2_INTENSITY_HE" in g:
            t += f" | INTENSITY_HE {g['r2_INTENSITY_HE'].iloc[i]:.2f}"
        if "reliability" in g and np.isfinite(g["reliability"].iloc[i]):
            t += f" | reliability {g['reliability'].iloc[i]:.2f}"
        rows.append({"label": names[i], "segments": [(y[i], GENE_CATEGORY_COLOURS[cat[i]])], "marker": x[i], "text": t})
    top_x = float(np.ceil(max(hi, 0.2) * 10) / 10)
    b = figures.hbar_chart(rows, f"R2 from {y_view} (black tick: from {x_view})", x_range=(min(0.0, lo), top_x),
                           width=860)
    right = figures.stack([figures.title_strip(640, f"b. Genes best predicted from {y_view}", 14), b])
    body = _side_by_side([left, right])
    return figures.with_caption(body, title=f"Which genes the image predicts - {prefix}, set 'keep' ({len(g)} genes)",
                                min_width=1200, **caption)


PARTITION_COLOURS = {"only_a": (0, 114, 178), "shared": (150, 150, 150), "only_b": (213, 94, 0)}


def gene_partition_figure(gp: pd.DataFrame, caption: dict, prefix: str, view_a: str = "UNI2H_RGB",
                          view_b: str = "LBP_RGB", n_top: int = 25) -> Image.Image | None:
    """a: for the genes best predicted from both views together, R^2(A+B) split into only A,
    shared and only B. b: the same parts summed over the genes that beat the spatial null
    from at least one view."""
    if gp is None or gp.empty:
        return None
    t = gp[(gp["view_a"] == view_a) & (gp["view_b"] == view_b)].reset_index(drop=True)
    if t.empty:
        return None
    ca = VIEW_COLOURS.get(view_a, PARTITION_COLOURS["only_a"])
    cb = VIEW_COLOURS.get(view_b, PARTITION_COLOURS["only_b"])
    order = np.argsort(-np.nan_to_num(t["r2_ab"].to_numpy(float), nan=-9))[:n_top]
    rows = []
    for i in order:
        r = t.iloc[i]
        rel = r["reliability"] if np.isfinite(r.get("reliability", np.nan)) else np.nan
        text = f"A {r['only_a']:.2f} | both {r['shared']:.2f} | B {r['only_b']:.2f}"
        if np.isfinite(rel):
            text += f" | reliability {rel:.2f}"
        rows.append({"label": str(r["gene_name"]),
                     "segments": [(max(r["only_a"], 0.0), ca), (max(r["shared"], 0.0), PARTITION_COLOURS["shared"]),
                                  (max(r["only_b"], 0.0), cb)],
                     "marker": rel, "text": text})
    top_x = float(np.ceil(max(float(np.nanmax(t["r2_ab"])), 0.2) * 10) / 10)
    a = figures.hbar_chart(rows, f"R2 of the gene from {view_a} + {view_b}, split (black tick: reliability)",
                           x_range=(0.0, min(1.0, max(top_x, 0.2))), width=980)
    sig = (t["q_a"] < 0.05) | (t["q_b"] < 0.05)
    s = t[sig] if sig.any() else t
    pos = s[["only_a", "shared", "only_b"]].clip(lower=0).sum()
    total = float(pos.sum())
    shares = (pos / total) if total > 0 else pos * np.nan
    b_rows = [{"label": f"{len(s)} genes", "segments": [(shares["only_a"], ca),
                                                         (shares["shared"], PARTITION_COLOURS["shared"]),
                                                         (shares["only_b"], cb)],
               "text": f"only {view_a} {shares['only_a']:.0%} | both {shares['shared']:.0%} | only {view_b} "
                       f"{shares['only_b']:.0%}"}]
    b = figures.hbar_chart(b_rows, "share of the summed R2 (parts below 0 counted as 0)", x_range=(0.0, 1.0), width=980)
    items = [(ca, f"only {view_a}: R2(A+B) - R2(B)"), (PARTITION_COLOURS["shared"], "both: R2(A) + R2(B) - R2(A+B)"),
             (cb, f"only {view_b}: R2(A+B) - R2(A)")]
    which = "that beat the spatial null from at least one view" if sig.any() else "(none beats the null: all genes)"
    body = figures.stack([figures.title_strip(980, f"a. The {len(order)} genes best predicted from both together", 14), a,
                          figures.title_strip(980, f"b. Summed over the genes {which}", 14), b,
                          figures.legend(items, width=980)])
    return figures.with_caption(body, title=f"What each view predicts of every gene - {prefix}, set 'keep'", min_width=1000,
                                **caption)


def markers_figure(markers: pd.DataFrame, f1: pd.DataFrame, info: dict, methods: list[str], caption: dict,
                   prefix: str, n_genes: int = 8) -> Image.Image | None:
    """Per method at the k shown: one row per expression domain with its size, the F1 of
    every image view for it and its best marker genes (AUROC)."""
    if markers is None or markers.empty:
        return None
    ref = info.get("reference_view", "EXPRESSION")
    views = [v for v in info.get("views_clustered", []) if v != ref]
    f_head, f_cell, f_small = figures.font(13), figures.font(13), figures.font(12)
    panels = []
    for m in methods:
        k = (info.get("k_show") or {}).get(m, info.get("k_ref"))
        t = markers[(markers["method"] == m) & (markers["k"] == k)]
        if t.empty:
            continue
        fk = f1[(f1["method"] == m) & (f1["k"] == k) & (f1["set"] == "keep")] if f1 is not None and len(f1) else f1
        domains = sorted(t["domain"].unique())
        cell, row_h, lw = 56, 30, 150
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        texts = {}
        for d in domains:
            g = t[t["domain"] == d].sort_values("rank").head(n_genes)
            texts[d] = ", ".join(f"{r.gene_name} ({r.auroc:.2f})" for r in g.itertuples(index=False))
        tw = int(max([text_width(probe, plain(s), f_small) for s in texts.values()] + [200])) + 20
        head_h = int(max([text_width(probe, plain(v), f_small) for v in views] + [40])) + 12
        W = lw + len(views) * cell + tw + 20
        H = 30 + head_h + len(domains) * row_h + 10
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        name = "Leiden" if m == "leiden" else "k-means"
        draw.text((6, 4), plain(f"{name}, k = {k}: the {ref} domains, how well each view finds them, their markers"),
                  font=f_head, fill=INK)
        top = 30 + head_h
        for j, v in enumerate(views):
            lab = vertical_text(v, f_small, VIEW_COLOURS.get(v, INK))
            img.paste(lab, (lw + j * cell + (cell - lab.width) // 2, top - lab.height - 4))
        draw.text((lw + len(views) * cell + 10, top - 20), "best markers (AUROC)", font=f_small, fill=INK)
        for i, d in enumerate(domains):
            y = top + i * row_h
            n_in = int(t[t["domain"] == d]["n_in"].iloc[0])
            draw.rectangle([6, y + 8, 20, y + 22], fill=cluster_colour(d))
            draw.text((26, y + 7), plain(f"domain {d} ({n_in})"), font=f_cell, fill=INK)
            for j, v in enumerate(views):
                val = np.nan
                if fk is not None and len(fk):
                    hit = fk[(fk["view"] == v) & (fk["domain"] == d)]
                    if len(hit):
                        val = float(hit["f1"].iloc[0])
                x = lw + j * cell
                draw.rectangle([x, y + 2, x + cell - 3, y + row_h - 3], fill=scale_colour(val))
                if np.isfinite(val):
                    _centred(draw, x, y + 2, cell - 3, row_h - 5, f"{val:.2f}", f_small,
                             (255, 255, 255) if val > 0.55 else INK)
            draw.text((lw + len(views) * cell + 10, y + 8), plain(texts[d]), font=f_small, fill=INK)
        panels.append(img)
    if not panels:
        return None
    body = figures.stack(panels + [colour_bar(label="F1")])
    return figures.with_caption(body, title=f"What the expression domains are, and which views find them - {prefix}",
                                min_width=1000, **caption)


def diverging_colour(v: float, top: float) -> tuple[int, int, int]:
    """Blue (negative) - white (0) - red (positive), saturated at +-top."""
    if not np.isfinite(v):
        return (240, 240, 240)
    t = float(np.clip(v / top, -1, 1))
    end = (178, 24, 43) if t > 0 else (33, 102, 172)
    return tuple(int(round(255 + abs(t) * (end[c] - 255))) for c in range(3))


def texture_genes_figure(table: pd.DataFrame, summary: pd.DataFrame, caption: dict, prefix: str, n_genes: int = 30,
                         level: float = 0.05, title: str | None = None, count_label: str | None = None,
                         dot_label: str | None = None) -> Image.Image | None:
    """a: partial Spearman correlation of the genes most associated with texture (rows) with
    every texture feature (columns: type, then radius); a dot where q < level. b: per type
    and radius, the number of genes with q < level."""
    if table is None or table.empty:
        return None
    feats = list(dict.fromkeys(table["feature"]))
    ft = table.drop_duplicates("feature").set_index("feature").loc[feats]
    sig = table[table["q"] < level]
    pool = sig if len(sig) else table
    best = pool.assign(a=pool["rho"].abs()).sort_values("a", ascending=False).drop_duplicates("gene_id")
    genes = best["gene_id"].head(n_genes).tolist()
    if not genes:
        return None
    name_of = dict(zip(table["gene_id"], table["gene_name"]))
    sub = table[table["gene_id"].isin(genes)]
    rho = sub.pivot_table(index="gene_id", columns="feature", values="rho").reindex(index=genes, columns=feats)
    qv = sub.pivot_table(index="gene_id", columns="feature", values="q").reindex(index=genes, columns=feats)
    # order genes by their profile (largest correlation with spot features first, then edge...)
    order = np.argsort(-np.nan_to_num(rho.to_numpy(float)).mean(axis=1))
    rho, qv = rho.iloc[order], qv.iloc[order]
    top = max(0.1, float(np.nanpercentile(np.abs(rho.to_numpy(float)), 98)))
    cw, ch = 16, 16
    f_lab, f_small, f_title = figures.font(12), figures.font(11), figures.font(14)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lw = int(max(text_width(probe, plain(str(name_of[g])), f_lab) for g in rho.index)) + 14
    head = 60
    title_a = "a. Partial Spearman correlation of each gene with each texture feature"
    W = max(lw + len(feats) * cw + 20, int(text_width(probe, plain(title_a), f_title)) + 20, lw + 560)
    H = 30 + head + len(rho) * ch + 40
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((6, 4), plain(title_a), font=f_title, fill=INK)
    y0 = 30 + head
    types = list(dict.fromkeys(ft["type"]))
    for t in types:
        cols = [i for i, f in enumerate(feats) if ft.loc[f, "type"] == t]
        x_a, x_b = lw + cols[0] * cw, lw + (cols[-1] + 1) * cw
        draw.line([(x_a + 2, 30 + 18), (x_b - 2, 30 + 18)], fill=INK, width=2)
        draw.text((x_a + 2, 30), plain(t.replace("_", " ")), font=f_small, fill=INK)
        for i in cols:
            r = int(ft.loc[feats[i], "radius_px"])
            lab = vertical_text(str(r), f_small, MUTED)
            img.paste(lab, (lw + i * cw + (cw - lab.width) // 2, y0 - lab.height - 3))
    for a, g in enumerate(rho.index):
        y = y0 + a * ch
        draw.text((lw - 6 - text_width(draw, plain(str(name_of[g])), f_lab), y + 1), plain(str(name_of[g])), font=f_lab,
                  fill=INK)
        for i, f in enumerate(feats):
            v = rho.iloc[a, i]
            x = lw + i * cw
            draw.rectangle([x, y, x + cw - 2, y + ch - 2], fill=diverging_colour(v, top))
            if np.isfinite(qv.iloc[a, i]) and qv.iloc[a, i] < level:
                draw.ellipse([x + cw / 2 - 3, y + ch / 2 - 3, x + cw / 2 + 2, y + ch / 2 + 2], fill=INK)
    yb = y0 + len(rho) * ch + 8
    for i in range(160):
        v = -top + 2 * top * i / 159
        draw.line([(lw + i, yb), (lw + i, yb + 10)], fill=diverging_colour(v, top))
    draw.text((lw - 40, yb - 2), f"{-top:.2f}", font=f_small, fill=MUTED)
    draw.text((lw + 166, yb - 2), f"{top:.2f}   rho (columns: radius in px); dot: {dot_label or f'q < {level:g}'}",
              font=f_small, fill=MUTED)
    panels = [img]
    if summary is not None and len(summary):
        series = []
        colours = {"spot_bright": (230, 159, 0), "spot_dark_or_flat": (0, 0, 0), "edge": (0, 114, 178),
                   "irregular": (204, 121, 167)}
        for t in types:
            s = summary[summary["type"] == t].sort_values("radius_px")
            series.append({"name": t, "x": s["radius_px"].to_numpy(float), "y": s["n_sig"].to_numpy(float),
                           "colour": colours.get(t, figures.SERIES[len(series) % len(figures.SERIES)])})
        ymax = float(np.ceil(max(5, summary["n_sig"].max()) / 5) * 5)
        chart = figures.line_chart(series, "LBP radius (px, log scale)", count_label or f"genes with q < {level:g}",
                                   size=(560, 320), y_range=(0.0, ymax), log_x=True,
                                   x_ticks=sorted(summary["radius_px"].unique().tolist()))
        items = [(s["colour"], s["name"].replace("_", " ")) for s in series]
        panels.append(figures.stack([figures.title_strip(560, "b. How many genes go with each feature", 14), chart,
                                     figures.legend(items, width=560)]))
    body = _side_by_side(panels)
    if title is None:
        title = f"Which texture goes with which gene - {prefix}, set 'keep' ({int(table['n_tiles'].iloc[0])} tiles)"
    return figures.with_caption(body, title=title, min_width=1100, **caption)
