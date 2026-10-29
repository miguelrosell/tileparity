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
               log_x: bool = False) -> Image.Image:
    """Simple line chart. Each series: {"name", "x", "y", optional "lo"/"hi" band, optional
    "colour" (default: the next colour of SERIES)}."""
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
