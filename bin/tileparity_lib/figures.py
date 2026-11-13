"""Figures with their explanation attached.

Every PNG figure the pipeline writes carries, below the picture, a strip of English
text with four parts: what it shows, how to read it, caveats, and where it comes from
(the chain of processes and the original inputs, from the lineage). The text is part
of the image, so it travels with the figure when it is copied into a report.

Only Pillow is used (no plotting library), so the figures are plain but the module
works in every container of the pipeline.
"""

from __future__ import annotations

import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (255, 255, 255)
INK = (25, 25, 25)
MUTED = (90, 90, 90)
GRID = (225, 225, 225)
# colour-blind-friendly series colours (Okabe-Ito)
SERIES = [(0, 114, 178), (230, 159, 0), (0, 158, 115), (204, 121, 167), (86, 180, 233), (213, 94, 0),
          (240, 228, 66), (0, 0, 0)]


def font(size: int = 14) -> ImageFont.ImageFont:
    """Pillow's built-in scalable font (FreeType) when available, else its bitmap font."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


# characters missing from Pillow's built-in font
_PLAIN = {"µ": "u", "≥": ">=", "≤": "<=", "–": "-", "—": "-", "’": "'", "×": "x", "→": "->"}


def plain(text: str) -> str:
    for a, b in _PLAIN.items():
        text = text.replace(a, b)
    return text


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt) -> float:
    try:
        return draw.textlength(text, font=fnt)
    except AttributeError:  # very old Pillow
        return fnt.getsize(text)[0]


def _wrap(draw, text: str, fnt, width: int) -> list[str]:
    words = text.split()
    lines, line = [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if text_width(draw, trial, fnt) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def caption_strip(width: int, sections: list[tuple[str, str]], size: int = 14, margin: int = 12) -> Image.Image:
    """White strip with ``label: text`` paragraphs wrapped to ``width`` pixels."""
    fnt, bold = font(size), font(size)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    line_h = int(size * 1.35)
    blocks = []
    for label, text in sections:
        if not text:
            continue
        head = f"{plain(label)}: "
        lines = _wrap(probe, head + " ".join(plain(str(text)).split()), fnt, width - 2 * margin)
        blocks.append((head, lines))
    height = margin * 2 + sum(len(lines) * line_h + 4 for _, lines in blocks)
    strip = Image.new("RGB", (width, max(height, 1)), BG)
    draw = ImageDraw.Draw(strip)
    draw.line([(0, 0), (width, 0)], fill=GRID, width=1)
    y = margin
    for head, lines in blocks:
        for i, line in enumerate(lines):
            if i == 0 and line.startswith(head):
                draw.text((margin, y), head, fill=INK, font=bold)
                draw.text((margin + text_width(draw, head, bold), y), line[len(head):], fill=MUTED, font=fnt)
            else:
                draw.text((margin, y), line, fill=MUTED, font=fnt)
            y += line_h
        y += 4
    return strip


def with_caption(img: Image.Image, what: str, how: str, caveats: str, source: str,
                 min_width: int = 960, title: str | None = None) -> Image.Image:
    """``img`` with an optional title above and the four-part explanation below."""
    img = img.convert("RGB")
    width = max(img.width, min_width)
    parts = []
    if title:
        parts.append(title_strip(width, title))
    body = Image.new("RGB", (width, img.height), BG)
    body.paste(img, ((width - img.width) // 2, 0))
    parts.append(body)
    parts.append(caption_strip(width, [("What it shows", what), ("How to read it", how),
                                       ("Caveats", caveats), ("Where it comes from", source)]))
    return stack(parts)


def title_strip(width: int, text: str, size: int = 18) -> Image.Image:
    fnt = font(size)
    strip = Image.new("RGB", (width, int(size * 2.0)), BG)
    ImageDraw.Draw(strip).text((12, int(size * 0.45)), plain(text), fill=INK, font=fnt)
    return strip


def stack(images: list[Image.Image]) -> Image.Image:
    width = max(i.width for i in images)
    out = Image.new("RGB", (width, sum(i.height for i in images)), BG)
    y = 0
    for i in images:
        out.paste(i, (0, y))
        y += i.height
    return out


def grid(cells: list[Image.Image], labels: list[str], ncols: int, cell_width: int, label_size: int = 13,
         pad: int = 8) -> Image.Image:
    """Images resized to ``cell_width`` and laid out in rows, each with a label above."""
    fnt = font(label_size)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    resized = []
    for c in cells:
        h = max(1, int(round(c.height * cell_width / max(c.width, 1))))
        resized.append(c.convert("RGB").resize((cell_width, h), Image.NEAREST))
    wrapped = [_wrap(probe, plain(lab), fnt, cell_width) for lab in labels]
    label_h = int(label_size * 1.35) * max((len(w) for w in wrapped), default=1) + 4
    rows = [resized[i:i + ncols] for i in range(0, len(resized), ncols)]
    row_heights = [label_h + max(r.height for r in row) for row in rows]
    W = ncols * cell_width + (ncols + 1) * pad
    H = sum(row_heights) + (len(rows) + 1) * pad
    out = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(out)
    y = pad
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            x = pad + c * (cell_width + pad)
            for k, line in enumerate(wrapped[r * ncols + c]):
                draw.text((x, y + k * int(label_size * 1.35)), line, fill=INK, font=fnt)
            out.paste(im, (x, y + label_h))
        y += row_heights[r] + pad
    return out


def legend(items: list[tuple[tuple[int, int, int], str]], size: int = 13, swatch: int = 14,
           width: int | None = None) -> Image.Image:
    """Horizontal legend of coloured squares (wraps when ``width`` is given)."""
    fnt = font(size)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    items = [(c, plain(t)) for c, t in items]
    widths = [swatch + 6 + int(text_width(probe, t, fnt)) + 18 for _, t in items]
    width = width or sum(widths) + 12
    lines, line, x = [], [], 12
    for item, w in zip(items, widths):
        if line and x + w > width:
            lines.append(line)
            line, x = [], 12
        line.append((item, x))
        x += w
    if line:
        lines.append(line)
    row_h = swatch + 10
    out = Image.new("RGB", (width, row_h * len(lines) + 8), BG)
    draw = ImageDraw.Draw(out)
    for r, line in enumerate(lines):
        y = 4 + r * row_h
        for (colour, text), x in line:
            draw.rectangle([x, y, x + swatch, y + swatch], fill=colour, outline=INK)
            draw.text((x + swatch + 6, y), text, fill=INK, font=fnt)
    return out


def line_chart(series: list[dict], x_label: str, y_label: str, size: tuple[int, int] = (900, 460),
               y_range: tuple[float, float] | None = None, x_ticks: list[float] | None = None,
               log_x: bool = False, v_lines: list[tuple[float, str]] | None = None) -> Image.Image:
    """Simple line chart. Each series: {"name", "x", "y", optional "lo"/"hi" band, optional
    "colour" (default: the next colour of SERIES)}. ``v_lines``: (x, label) vertical dashed
    markers."""
    W, H = size
    left, right, top, bottom = 70, 20, 30, 60
    fnt, small = font(13), font(12)
    x_label, y_label = plain(x_label), plain(y_label)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    xs = np.concatenate([np.asarray(s["x"], float) for s in series]) if series else np.array([0.0, 1.0])
    ys = np.concatenate([np.asarray(s.get("hi", s["y"]), float) for s in series]) if series else np.array([0.0, 1.0])
    tx = (lambda v: np.log10(v)) if log_x else (lambda v: np.asarray(v, float))
    if x_ticks is not None and len(x_ticks):  # the axis covers every tick, with or without data there
        xs = np.concatenate([xs, np.asarray(x_ticks, float)])
    x0, x1 = float(np.nanmin(tx(xs))), float(np.nanmax(tx(xs)))
    if x1 == x0:
        x1 = x0 + 1
    y0, y1 = y_range if y_range else (0.0, float(np.nanmax(ys)) if np.isfinite(np.nanmax(ys)) else 1.0)
    if y1 == y0:
        y1 = y0 + 1

    def px(x, y):
        X = left + (tx(x) - x0) / (x1 - x0) * (W - left - right)
        Y = top + (1 - (np.asarray(y, float) - y0) / (y1 - y0)) * (H - top - bottom)
        return X, Y

    for k in range(6):  # horizontal grid and y labels
        yv = y0 + k * (y1 - y0) / 5
        _, Y = px(np.array([10 ** x0 if log_x else x0]), np.array([yv]))
        draw.line([(left, float(Y[0])), (W - right, float(Y[0]))], fill=GRID)
        draw.text((8, float(Y[0]) - 7), f"{yv:g}", fill=MUTED, font=small)
    for xv in (x_ticks if x_ticks is not None else np.unique(xs)):
        X, _ = px(np.array([xv]), np.array([y0]))
        draw.line([(float(X[0]), H - bottom), (float(X[0]), H - bottom + 5)], fill=INK)
        label = f"{xv:g}"
        draw.text((float(X[0]) - text_width(draw, label, small) / 2, H - bottom + 8), label, fill=MUTED, font=small)
    draw.line([(left, top), (left, H - bottom), (W - right, H - bottom)], fill=INK)
    draw.text((left + (W - left - right) / 2 - text_width(draw, x_label, fnt) / 2, H - 24), x_label, fill=INK, font=fnt)
    draw.text((8, 2), y_label, fill=INK, font=fnt)
    for xv, label in (v_lines or []):
        X, _ = px(np.array([xv]), np.array([y0]))
        xx = float(X[0])
        for yy in range(top, H - bottom, 8):
            draw.line([(xx, yy), (xx, min(yy + 4, H - bottom))], fill=MUTED, width=1)
        if label:
            draw.text((xx + 4, H - bottom - 18), plain(label), fill=MUTED, font=small)
    for i, s in enumerate(series):
        colour = tuple(s.get("colour", SERIES[i % len(SERIES)]))
        x = np.asarray(s["x"], float)
        if x.size == 0:
            continue
        if "lo" in s and "hi" in s:
            Xl, Yl = px(x, s["lo"])
            Xh, Yh = px(x, s["hi"])
            band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(band).polygon(list(zip(Xl, Yl)) + list(zip(Xh[::-1], Yh[::-1])), fill=colour + (50,))
            img.paste(band, (0, 0), band)
            draw = ImageDraw.Draw(img)
        X, Y = px(x, s["y"])
        draw.line(list(zip(X, Y)), fill=colour, width=3)
        for a, b in zip(X, Y):
            draw.ellipse([a - 3, b - 3, a + 3, b + 3], fill=colour)
    return img


def mask_overlay(thumb: np.ndarray, mask: np.ndarray, colour=(0, 158, 115), alpha: float = 0.45,
                 outline: bool = True) -> Image.Image:
    """Thumbnail with the tissue mask tinted (and its border drawn)."""
    base = thumb[:, :, :3].astype(np.float32)
    m = mask.astype(bool)
    if m.shape != base.shape[:2]:
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((base.shape[1], base.shape[0]),
                                                                       Image.NEAREST)) > 0
    out = base.copy()
    out[m] = (1 - alpha) * out[m] + alpha * np.array(colour, np.float32)
    if outline:
        p = np.pad(m, 1, mode="edge")  # the image border is not a tissue border
        inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        out[m & ~inner] = (0, 0, 0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def wrap_text(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width))


def hbar_chart(rows: list[dict], x_label: str, x_range: tuple[float, float] = (0.0, 1.0), width: int = 640,
               row_h: int = 28) -> Image.Image:
    """Horizontal bars. Each row: {"label", "segments": [(value, colour), ...] stacked from 0,
    optional "lo"/"hi" (whisker), optional "marker" (a black tick, e.g. a null level),
    optional "text" (written after the bar)}. Negative single values are drawn leftwards."""
    fnt, small = font(13), font(12)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lw = int(max([text_width(probe, plain(r["label"]), fnt) for r in rows] + [40])) + 14
    right = int(max([text_width(probe, plain(r.get("text", "")), small) for r in rows] + [40])) + 24
    top, bottom = 10, 46
    H = top + len(rows) * row_h + bottom
    img = Image.new("RGB", (width, H), BG)
    draw = ImageDraw.Draw(img)
    x0, x1 = x_range

    def X(v):
        return lw + (float(np.clip(v, x0, x1)) - x0) / (x1 - x0) * (width - lw - right)

    for t in np.linspace(x0, x1, 6):
        draw.line([(X(t), top), (X(t), H - bottom)], fill=GRID)
        lab = f"{t:.2g}"
        draw.text((X(t) - text_width(draw, lab, small) / 2, H - bottom + 6), lab, fill=MUTED, font=small)
    if x0 < 0 < x1:
        draw.line([(X(0), top), (X(0), H - bottom)], fill=INK)
    for i, r in enumerate(rows):
        y = top + i * row_h
        draw.text((lw - 8 - text_width(draw, plain(r["label"]), fnt), y + row_h / 2 - 8), plain(r["label"]), font=fnt,
                  fill=INK)
        start = 0.0
        for value, colour in r.get("segments", []):
            if not np.isfinite(value):
                continue
            a, b = (start, start + value) if value >= 0 else (start + value, start)
            draw.rectangle([X(a), y + 5, X(b), y + row_h - 6], fill=tuple(colour))
            start = start + value if value >= 0 else start
        if np.isfinite(r.get("lo", np.nan)) and np.isfinite(r.get("hi", np.nan)):
            ym = y + row_h / 2
            draw.line([(X(r["lo"]), ym), (X(r["hi"]), ym)], fill=INK, width=2)
            for v in (r["lo"], r["hi"]):
                draw.line([(X(v), ym - 5), (X(v), ym + 5)], fill=INK, width=2)
        if np.isfinite(r.get("marker", np.nan)):
            draw.line([(X(r["marker"]), y + 2), (X(r["marker"]), y + row_h - 3)], fill=(0, 0, 0), width=3)
        if r.get("text"):
            end = max([start] + [r.get("hi", -np.inf) if np.isfinite(r.get("hi", np.nan)) else -np.inf])
            draw.text((X(end) + 6, y + row_h / 2 - 8), plain(r["text"]), font=small, fill=MUTED)
    draw.text((lw + (width - lw - right) / 2 - text_width(draw, plain(x_label), fnt) / 2, H - 24), plain(x_label),
              fill=INK, font=fnt)
    return img


def scatter_chart(x: np.ndarray, y: np.ndarray, colours, x_label: str, y_label: str, size: tuple[int, int] = (620, 560),
                  x_range: tuple[float, float] | None = None, y_range: tuple[float, float] | None = None,
                  labels: list[tuple[int, str]] | None = None, diagonal: bool = True, radius: int = 3) -> Image.Image:
    """Scatter plot of points (x, y) with one colour per point (or one for all), an optional
    y = x line and text labels next to some points (index, text)."""
    W, H = size
    left, right, top, bottom = 70, 20, 30, 60
    fnt, small = font(13), font(11)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x0, x1 = x_range or (float(np.nanmin(x[ok])) if ok.any() else 0.0, float(np.nanmax(x[ok])) if ok.any() else 1.0)
    y0, y1 = y_range or (float(np.nanmin(y[ok])) if ok.any() else 0.0, float(np.nanmax(y[ok])) if ok.any() else 1.0)
    x1, y1 = (x1 if x1 > x0 else x0 + 1), (y1 if y1 > y0 else y0 + 1)

    def P(a, b):
        return (left + (np.clip(a, x0, x1) - x0) / (x1 - x0) * (W - left - right),
                top + (1 - (np.clip(b, y0, y1) - y0) / (y1 - y0)) * (H - top - bottom))

    for t in np.linspace(0, 1, 6):
        xv, yv = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        X, _ = P(xv, y0)
        _, Y = P(x0, yv)
        draw.line([(X, top), (X, H - bottom)], fill=GRID)
        draw.line([(left, Y), (W - right, Y)], fill=GRID)
        draw.text((X - 10, H - bottom + 6), f"{xv:.2g}", fill=MUTED, font=small)
        draw.text((8, Y - 7), f"{yv:.2g}", fill=MUTED, font=small)
    if diagonal:
        lo, hi = max(x0, y0), min(x1, y1)
        if hi > lo:
            draw.line([P(lo, lo), P(hi, hi)], fill=MUTED, width=1)
    draw.line([(left, top), (left, H - bottom), (W - right, H - bottom)], fill=INK)
    cols = colours if isinstance(colours, (list, np.ndarray)) and len(colours) == len(x) else [colours] * len(x)
    order = np.argsort([sum(c) for c in cols])[::-1]  # light points first, strong colours on top
    for i in order:
        if ok[i]:
            a, b = P(x[i], y[i])
            draw.ellipse([a - radius, b - radius, a + radius, b + radius], fill=tuple(cols[i]))
    for i, text in (labels or []):
        if ok[i]:
            a, b = P(x[i], y[i])
            draw.text((a + 5, b - 7), plain(text), fill=INK, font=small)
    draw.text((left + (W - left - right) / 2 - text_width(draw, plain(x_label), fnt) / 2, H - 24), plain(x_label),
              fill=INK, font=fnt)
    draw.text((8, 4), plain(y_label), fill=INK, font=fnt)
    return img


def strip_chart(categories: list[str], points: list[list[tuple]], y_label: str, y_range: tuple[float, float] = (0, 1),
                width: int | None = None, height: int = 380, lines: list[list[tuple[int, float, int, float]]] | None = None,
                cat_colours: list | None = None) -> Image.Image:
    """Values of several samples per category: ``points[i]`` = [(value, colour), ...] for
    category i, drawn as jittered dots with the mean as a black bar. ``lines``: optional
    segments [(i_from, y_from, i_to, y_to), ...] joining paired values (drawn grey)."""
    n = len(categories)
    fnt, small = font(13), font(12)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lab_h = int(max([text_width(probe, plain(c), small) for c in categories] + [20])) + 12
    W = width or max(420, 90 + n * 70)
    left, right, top, bottom = 70, 20, 30, lab_h + 34
    img = Image.new("RGB", (W, height), BG)
    draw = ImageDraw.Draw(img)
    y0, y1 = y_range
    step = (W - left - right) / max(n, 1)

    def X(i):
        return left + (i + 0.5) * step

    def Y(v):
        return top + (1 - (float(np.clip(v, y0, y1)) - y0) / (y1 - y0)) * (height - top - bottom)

    step_t = 0.2 if (y1 - y0) > 0.6 else 0.1
    for t in np.arange(np.ceil(y0 / step_t) * step_t, y1 + 1e-9, step_t):
        draw.line([(left, Y(t)), (W - right, Y(t))], fill=GRID)
        draw.text((8, Y(t) - 7), f"{t:.1f}", fill=MUTED, font=small)
    if y0 < 0 < y1:
        draw.line([(left, Y(0)), (W - right, Y(0))], fill=MUTED)
    draw.line([(left, top), (left, height - bottom), (W - right, height - bottom)], fill=INK)
    for seg in (lines or []):
        for i0, v0, i1, v1 in seg:
            draw.line([(X(i0), Y(v0)), (X(i1), Y(v1))], fill=(170, 170, 170), width=1)
    rng = np.random.default_rng(0)
    for i, pts in enumerate(points):
        vals = [v for v, _c in pts if np.isfinite(v)]
        for v, c in pts:
            if not np.isfinite(v):
                continue
            x = X(i) + float(rng.uniform(-0.18, 0.18)) * step
            draw.ellipse([x - 4, Y(v) - 4, x + 4, Y(v) + 4], fill=tuple(c), outline=(60, 60, 60))
        if vals:
            m = float(np.mean(vals))
            draw.line([(X(i) - 0.3 * step, Y(m)), (X(i) + 0.3 * step, Y(m))], fill=(0, 0, 0), width=3)
        lab = vertical_text_img(categories[i], small, (cat_colours[i] if cat_colours else INK))
        img.paste(lab, (int(X(i) - lab.width / 2), height - bottom + 6))
    draw.text((8, 4), plain(y_label), fill=INK, font=fnt)
    return img


def vertical_text_img(text: str, fnt, fill=INK) -> Image.Image:
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    box = probe.textbbox((0, 0), plain(text), font=fnt)
    img = Image.new("RGB", (box[2] - box[0] + 4, box[3] - box[1] + 4), BG)
    ImageDraw.Draw(img).text((2 - box[0], 2 - box[1]), plain(text), font=fnt, fill=tuple(fill))
    return img.rotate(90, expand=True)
