#!/usr/bin/env python3
"""Optional (with --cluster, on by default: --cluster_gallery): galleries of the tiles that
show what each view groups. For every view of the set keep, one figure with, for every
cluster at the k shown (Leiden when computed, else k-means), the tiles closest to the
cluster's centre in the view's principal components, and for the first principal
components the tiles with the highest and the lowest scores. The tiles are cut from the
image the extractors read (Process 1's resampled image, input set rgb).

Inputs: <prefix>.resampled.tif, <prefix>.tiles.tsv.gz and <prefix>.cluster_representatives.tsv
(written by tp_cluster.py)."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from tileparity_lib import cluster, clusterplots, figures, image, lineage  # noqa: E402
from tileparity_lib.tiling import read_tiles  # noqa: E402

log = logging.getLogger("tp_gallery")


def crop(img, row, size: int) -> Image.Image:
    """The tile (x0_px, y0_px, tile_px) of the resampled image, as a ``size`` px square."""
    x0, y0, T = int(row.x0_px), int(row.y0_px), int(row.tile_px)
    x0, y0 = max(0, x0), max(0, y0)
    w, h = min(T, img.width - x0), min(T, img.height - y0)
    if w <= 0 or h <= 0:
        return Image.new("RGB", (size, size), (240, 240, 240))
    region = img.crop(x0, y0, w, h)
    arr = np.ndarray(buffer=region.write_to_memory(), dtype=np.uint8, shape=(h, w, region.bands))
    tile = Image.fromarray(arr[:, :, :3])
    canvas = Image.new("RGB", (T, T), (240, 240, 240))
    canvas.paste(tile, (0, 0))
    return canvas.resize((size, size), Image.LANCZOS)


def strip(label: str, tiles: list[Image.Image], swatch, size: int, label_w: int) -> Image.Image:
    pad = 4
    out = Image.new("RGB", (label_w + len(tiles) * (size + pad) + pad, size + 2 * pad), figures.BG)
    draw = ImageDraw.Draw(out)
    if swatch is not None:
        draw.rectangle([6, pad + 6, 18, pad + 18], fill=swatch)
    f = figures.font(12)
    lines = figures.wrap_text(label, 26).split("\n")
    for i, line in enumerate(lines[:4]):
        draw.text((24 if swatch is not None else 6, pad + 4 + 16 * i), figures.plain(line), font=f, fill=figures.INK)
    for j, t in enumerate(tiles):
        out.paste(t, (label_w + pad + j * (size + pad), pad))
    return out


def gallery(img, tiles: pd.DataFrame, reps: pd.DataFrame, view: str, prefix: str, size: int, source: str,
            reference: str | None) -> Image.Image | None:
    r = reps[reps["view"] == view]
    if r.empty:
        return None
    cl = r[r["kind"] == "cluster"]
    method = "leiden" if (cl["method"] == "leiden").any() else "kmeans"
    cl = cl[cl["method"] == method]
    label_w = 210
    rows = []
    k = int(cl["k"].iloc[0]) if len(cl) else None
    for c, g in cl.groupby("group", sort=True):
        g = g.sort_values("rank")
        crops = [crop(img, tiles.loc[t], size) for t in g["tile_id"] if t in tiles.index]
        colour = clusterplots.cluster_colour(int(g["colour"].iloc[0])) if pd.notna(g["colour"].iloc[0]) else None
        rows.append(strip(f"cluster {int(c)} ({int(g['group_size'].iloc[0])} tiles)", crops, colour, size, label_w))
    pcs = r[r["kind"].isin(["pc_high", "pc_low"])]
    for (pc, kind), g in sorted(pcs.groupby(["group", "kind"]), key=lambda x: (x[0][0], x[0][1] != "pc_high")):
        g = g.sort_values("rank")
        crops = [crop(img, tiles.loc[t], size) for t in g["tile_id"] if t in tiles.index]
        side = "highest" if kind == "pc_high" else "lowest"
        rows.append(strip(f"PC{int(pc)}, {side} scores", crops, None, size, label_w))
    if not rows:
        return None
    width = max(x.width for x in rows)
    parts = []
    if len(cl):
        parts.append(figures.title_strip(width, f"Clusters ({'Leiden' if method == 'leiden' else 'k-means'}, k = {k}): "
                                         "the tiles closest to each cluster's centre", 14))
    n_cluster_rows = cl["group"].nunique() if len(cl) else 0
    parts += rows[:n_cluster_rows]
    if len(pcs):
        parts.append(figures.title_strip(width, "Principal components: the tiles at both ends", 14))
        parts += rows[n_cluster_rows:]
    body = figures.stack(parts)
    n_per = int(cl["rank"].max()) if len(cl) else int(r["rank"].max())
    colours = (f"Colour squares: the colour of the cluster in cluster_maps_keep.png (the {reference} cluster it is "
               "matched to). " if reference and view != reference else "")
    caption = {
        "what": (f"What the view {view} groups, on the slide's own tiles (set 'keep'): for every cluster at the k shown in "
                 f"the clustering figures, the {n_per} tiles "
                 "closest to the cluster's centre in the view's principal components (its most typical tiles), and for "
                 "the first principal components the tiles with the highest and the lowest scores. " + colours +
                 f"{view} = {cluster.view_description(view)}."),
        "how": ("Look for what the tiles of one row share and what changes between rows: cell types, density, glands, "
                "stroma, muscle, mucus, blood, or artefacts (blur, folds, tissue edges, stain). A row of mixed-looking "
                "tiles is a cluster defined by something the eye does not see directly (for EXPRESSION: genes). The two "
                "ends of a component show what that direction of the view separates."),
        "caveats": ("The most typical tiles are not the whole cluster: look at the maps for where it lies. Tiles are "
                    "shown as the extractors read them (Process 1's image at the analysis scale), resized for display. "
                    "Which tiles: cluster_representatives.tsv."),
        "source": source,
    }
    return figures.with_caption(body, title=f"Tiles that show what {view} groups - {prefix}, set 'keep'", **caption)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--size", type=int, default=96, help="displayed size of each tile (px)")
    lineage.add_cli(ap)
    args = ap.parse_args(argv)
    process = args.process or "TILEPARITY_GALLERY"
    d = Path(args.input_dir)
    prefix = args.prefix
    reps_path = d / f"{prefix}.cluster_representatives.tsv"
    reps = pd.read_csv(reps_path, sep="\t", dtype={"tile_id": str})
    tiles = read_tiles(d / f"{prefix}.tiles.tsv.gz", subset=None).set_index("tile_id", drop=False)
    img = image.open_rgb(d / f"{prefix}.resampled.tif")
    parents = [p for p in (d / f"{prefix}.prepare.lineage.json", d / f"{prefix}.cluster.lineage.json") if p.exists()]
    summary_path = d / f"{prefix}.cluster.json"
    reference = None
    if summary_path.exists():
        import json

        reference = (json.loads(summary_path.read_text()).get("sets", {}).get("keep", {}) or {}).get("reference_view")
    source = lineage.source_text(process, {}, parents=[str(p) for p in parents], context=f"{prefix}, set 'keep'")
    outputs = {}
    for view in dict.fromkeys(reps["view"]):
        fig = gallery(img, tiles, reps, view, prefix, args.size, source, reference)
        if fig is not None:
            path = f"{prefix}.gallery_{view}.png"
            fig.save(path)
            outputs[f"gallery_{view}"] = path
            log.info("%s: %s", prefix, path)
    lineage.write(f"{prefix}.gallery.lineage.json", lineage.build(
        "gallery", process, prefix, parameters={"size": args.size},
        inputs={"image": d / f"{prefix}.resampled.tif", "tiles": d / f"{prefix}.tiles.tsv.gz",
                "representatives": reps_path},
        outputs=outputs, parents=parents, packages=["pyvips", "pillow", "pandas"]))


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
