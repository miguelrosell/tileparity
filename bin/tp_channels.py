#!/usr/bin/env python3
"""Process 1b - the input images of both extractors: H and E (always), the R, G, B copies for
fastLBP's ``rgb`` set and the image rebuilt from H and E only (``rgb_he``, for fastLBP and
UNI2-h), with a QC figure and table of what ``rgb_he`` changes."""

import argparse
import json
import logging
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import channels, figures, image, lineage  # noqa: E402
from tileparity_lib.tiling import read_tiles  # noqa: E402

SAMPLE_TILES = 500  # tiles sampled for the stain-vector estimate
SAMPLE_PIXELS = 1_000_000  # pixels sampled in total (spread evenly over the sampled tiles)
OVERVIEW_PX = 900  # longest side of the overview thumbnails
CHANGE_FULL = 40  # change (RGB levels) drawn in full colour in the QC figure
CHANGED = 10  # a pixel counts as changed when R, G or B moves by more than this
QC_COLUMNS = ["tile_id", "residual_share", "mean_abs_change", "changed_fraction"]


def sample_pixels(img, tiles: pd.DataFrame, T: int, seed: int) -> np.ndarray:
    """Random pixels of a random subset of the extracted tiles (fixed seed)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tiles))[:SAMPLE_TILES]
    per_tile = int(np.ceil(SAMPLE_PIXELS / max(len(idx), 1)))
    out = []
    for i in idx:
        px = image.read_region(img, int(tiles["x0_px"].iloc[i]), int(tiles["y0_px"].iloc[i]), T, T)[:, :, :3]
        px = px.reshape(-1, 3)
        out.append(px[rng.choice(len(px), size=min(per_tile, len(px)), replace=False)])
    return np.vstack(out) if out else np.zeros((0, 3), np.uint8)


def thumbnail(img, bands: list[int]) -> np.ndarray:
    img = img.extract_band(bands[0]).bandjoin([img.extract_band(b) for b in bands[1:]])
    scale = min(1.0, OVERVIEW_PX / max(img.width, img.height))
    small = img.resize(scale, kernel="linear") if scale < 1 else img
    return image.to_numpy(small.cast("uchar"))[:, :, :3]


def change_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """White (unchanged) to dark orange (R, G or B moved by CHANGE_FULL levels or more)."""
    d = np.abs(a.astype(int) - b.astype(int)).max(axis=-1)
    t = np.clip(d / CHANGE_FULL, 0, 1)[..., None]
    return (255 * (1 - t) + np.array(figures.SERIES[5]) * t).astype(np.uint8)


def tile_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(int) - b.astype(int))
    return {"residual_share": channels.residual_share(a, b), "mean_abs_change": float(diff.mean()),
            "changed_fraction": float((diff.max(axis=-1) > CHANGED).mean())}


def grid_metrics(orig, rebuilt, tiles: pd.DataFrame, T: int) -> pd.DataFrame:
    """:func:`tile_metrics` of every tile in one streaming pass over the slide (block means of
    per-pixel maps), for tiles on the grid anchored at (0, 0) and inside the image."""
    W, H = (orig.width // T) * T, (orig.height // T) * T
    a, b = orig.crop(0, 0, W, H), rebuilt.crop(0, 0, W, H)
    od_a, od_b = channels._od_vips(a), channels._od_vips(b)
    res = (od_a - od_b).abs()
    d = (a.cast("int") - b.cast("int")).abs()
    d_max = (d[0] > d[1]).ifthenelse(d[0], d[1])
    d_max = (d_max > d[2]).ifthenelse(d_max, d[2])
    maps = (res[0] + res[1] + res[2]).bandjoin([od_a[0] + od_a[1] + od_a[2], (d[0] + d[1] + d[2]) / 3.0,
                                                (d_max > CHANGED).ifthenelse(1.0, 0.0)]).cast("double")
    blocks = image.to_numpy(maps.shrink(T, T))  # mean of every T x T block
    v = blocks[(tiles["y0_px"] // T).to_numpy(int), (tiles["x0_px"] // T).to_numpy(int)]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(v[:, 1] > 0, v[:, 0] / v[:, 1], 0.0)
    return pd.DataFrame({"tile_id": tiles["tile_id"].to_numpy(), "residual_share": share,
                         "mean_abs_change": v[:, 2], "changed_fraction": v[:, 3]}, columns=QC_COLUMNS)


def on_grid(tiles: pd.DataFrame, T: int, width: int, height: int) -> bool:
    x0, y0 = tiles["x0_px"].to_numpy(int), tiles["y0_px"].to_numpy(int)
    return bool(len(tiles) and (x0 % T == 0).all() and (y0 % T == 0).all() and (x0 >= 0).all() and (y0 >= 0).all()
                and (x0 + T <= width).all() and (y0 + T <= height).all())


def nearest_rank(values: pd.Series, q: float) -> int:
    """Index of the tile at quantile ``q`` (nearest rank: the value shown is a real tile's)."""
    order = values.sort_values(kind="stable")
    return int(order.index[min(len(order) - 1, max(0, int(np.ceil(q * len(order))) - 1))])


def qc(args, bands: list[str], tiles: pd.DataFrame, T: int, basis: channels.StainBasis, source: str):
    """Per-tile change made by rgb_he, and the QC figure."""
    from PIL import Image

    import pyvips

    orig = image.open_rgb(args.image, access="random")
    chan = pyvips.Image.new_from_file(f"{args.prefix}.channels.tif", access="random")
    he_bands = [bands.index(b) for b in channels.CHANNEL_SETS["rgb_he"]]

    def pair(t):
        a = image.read_region(orig, int(t.x0_px), int(t.y0_px), T, T)[:, :, :3]
        b = image.read_region(chan, int(t.x0_px), int(t.y0_px), T, T)[:, :, he_bands]
        return a, b

    if on_grid(tiles, T, orig.width, orig.height):
        rebuilt = chan.extract_band(he_bands[0]).bandjoin([chan.extract_band(k) for k in he_bands[1:]])
        table = grid_metrics(orig, rebuilt, tiles, T)
    else:  # e.g. tiles centred on Visium spots
        rows = [{"tile_id": t.tile_id, **tile_metrics(*pair(t))} for t in tiles.itertuples(index=False)]
        table = pd.DataFrame(rows, columns=QC_COLUMNS)

    thumb, thumb_he = thumbnail(orig, [0, 1, 2]), thumbnail(chan, he_bands)
    cells = [Image.fromarray(thumb), Image.fromarray(thumb_he), Image.fromarray(change_image(thumb, thumb_he))]
    labels = ["Slide (thumbnail)", "Rebuilt from H and E only (rgb_he)", "Change"]
    summary = {"n_tiles": int(len(table))}
    if len(table):
        s = table["residual_share"]
        picks = {"median": nearest_rank(s, 0.5), "90th percentile": nearest_rank(s, 0.9), "largest": int(s.idxmax())}
        summary.update(residual_share_median=float(s[picks["median"]]), residual_share_p90=float(s[picks["90th percentile"]]),
                       residual_share_max=float(s.max()),
                       mean_abs_change_median=float(table["mean_abs_change"].median()),
                       changed_fraction_median=float(table["changed_fraction"].median()))
        shown = []
        for label, k in picks.items():
            if k in shown:
                continue
            shown.append(k)
            a, b = pair(tiles.iloc[k])
            cells += [Image.fromarray(a), Image.fromarray(b), Image.fromarray(change_image(a, b))]
            labels += [f"Tile with the {label} share ({100 * s[k]:.1f} %)", "Rebuilt from H and E only", "Change"]
    panel = figures.grid(cells, labels, ncols=3, cell_width=300)
    bd = basis.to_dict()
    pct = (lambda key: f"{100 * summary[key]:.1f} %") if len(table) else (lambda key: "n/a")
    vectors = (f"H = {[round(x, 3) for x in bd['H']]} and E = {[round(x, 3) for x in bd['E']]} (optical density of R, "
               f"G, B), {bd['angle_to_standard_degrees']['H']:.0f} and {bd['angle_to_standard_degrees']['E']:.0f} "
               "degrees from the standard vectors")
    changed = (f"; in the median tile R, G and B move by {summary['mean_abs_change_median']:.1f} levels on average and "
               f"{100 * summary['changed_fraction_median']:.0f} % of the pixels by more than {CHANGED}") if len(table) else ""
    fig = figures.with_caption(
        panel, title=f"{Path(args.prefix).name} - input set rgb_he: the slide rebuilt from haematoxylin and eosin only",
        what=("The image that UNI2-h and fastLBP receive as input set rgb_he: every pixel's colour replaced by the "
              "closest colour made of haematoxylin and eosin only. Top: the whole slide, its H&E-only version and how "
              "much each pixel changed. Below: the extracted tiles with the median, 90th-percentile and largest share "
              "of optical density removed (a tile that is both is shown once), original next to rebuilt."),
        how=(f"rgb_he should look like the slide. Change: white = unchanged, darkest orange = R, G or B moved by "
             f"{CHANGE_FULL} levels (of 255) or more. Share of the optical density removed: {pct('residual_share_median')} "
             f"in the median tile, {pct('residual_share_p90')} at the 90th percentile and {pct('residual_share_max')} at "
             f"most{changed} (per tile in rgb_he_qc.tsv). Change concentrated in one place points to colour that is "
             "neither H nor E: pen marks, pigment, folds, blood, dust; an even change everywhere, to a tint of the "
             "scanner or the glass. Comparing the extractors on rgb and rgb_he shows how much they rely on that colour."),
        caveats=(f"H and E: {bd['method']}; {vectors}. Removed = the part of each pixel's optical density "
                 "perpendicular to the plane of H and E (it depends on the plane only, not on the two directions in "
                 "it). Near-white pixels have little optical density, so a slight tint of the glass weighs much in "
                 "their share. The overview is drawn at thumbnail resolution, the tiles at the resolution the "
                 "extractors use." + (f" Note: {bd['note']}." if bd["note"] else "")),
        source=source)
    return table, fig, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="resampled RGB image from Process 1")
    ap.add_argument("--tiles", required=True, help="tile table from Process 1 (stain estimate and QC)")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--lbp-sets", default="", help=f"fastLBP's input sets, subset of {','.join(channels.LBP_SETS)}")
    ap.add_argument("--uni2h-inputs", default="", help=f"UNI2-h's input sets, subset of {','.join(channels.UNI2H_INPUTS)}")
    ap.add_argument("--stain-vectors", default="slide", choices=channels.STAIN_VECTORS)
    ap.add_argument("--h-max", type=float, default=0.30, help="haematoxylin amount mapped to 255 (H channel)")
    ap.add_argument("--e-max", type=float, default=0.15, help="eosin amount mapped to 255 (E channel)")
    ap.add_argument("--compression", default="deflate", choices=["deflate", "lzw", "zstd", "none"])
    lineage.add_cli(ap)
    args = ap.parse_args(argv)

    try:
        lbp_sets = channels.parse_sets(args.lbp_sets, channels.LBP_SETS, allow_empty=True)
        uni2h_inputs = channels.parse_sets(args.uni2h_inputs, channels.UNI2H_INPUTS, allow_empty=True)
    except ValueError as err:
        ap.error(str(err))
    name = Path(args.prefix).name
    bands = channels.band_order(lbp_sets, uni2h_inputs)
    tiles = read_tiles(args.tiles)
    T = int(tiles["tile_px"].iloc[0]) if len(tiles) else 224

    if args.stain_vectors == "slide":
        sampled = sample_pixels(image.open_rgb(args.image, access="random"), tiles, T, seed=zlib.crc32(name.encode()))
        basis = channels.estimate_stain_basis(sampled)
    else:
        basis = channels.standard_basis()
    if basis.note:
        logging.warning("%s: %s", name, basis.note)
    logging.info("%s: bands %s; H %s, E %s (%s)", name, bands, basis.H.round(3), basis.E.round(3), basis.method)

    rgb = image.open_rgb(args.image, access="sequential")
    out_path = f"{args.prefix}.channels.tif"
    image.write_tiled_tiff(channels.rgb_to_channels_vips(rgb, args.h_max, args.e_max, bands, basis), out_path,
                           mpp=image.mpp_from_metadata(rgb), compression=args.compression)

    report = {
        "lbp_sets": lbp_sets,
        "uni2h_inputs": uni2h_inputs,
        "bands": bands,
        "lbp_channels": channels.lbp_channels(lbp_sets) if lbp_sets else [],
        "stain_vectors": basis.to_dict(),
        "h_max": args.h_max,
        "e_max": args.e_max,
        "deconvolution": "optical density = [a_H, a_E, a_R] @ [H; E; R], R = H x E (Ruifrok & Johnston 2001); "
                         "rgb_he = the image of a_H * H + a_E * E",
        **channels.clipping_fractions(image.open_rgb(args.image, access="sequential"), args.h_max, args.e_max, basis),
    }
    outputs = {"channels": out_path, "report": f"{args.prefix}.channels.json"}
    if "R_he" in bands:
        source = lineage.source_text(args.process or "TILEPARITY_CHANNELS", {"image": args.image},
                                     parents=args.lineage_parent, context=f"extracted tiles of {name}")
        table, fig, summary = qc(args, bands, tiles, T, basis, source)
        table.to_csv(f"{args.prefix}.rgb_he_qc.tsv", sep="\t", index=False)
        fig.save(f"{args.prefix}.rgb_he_qc.png")
        report["rgb_he"] = summary
        outputs.update(qc_figure=f"{args.prefix}.rgb_he_qc.png", qc_table=f"{args.prefix}.rgb_he_qc.tsv")
    Path(f"{args.prefix}.channels.json").write_text(json.dumps(report, indent=2))
    if lbp_sets:  # the intensity block of fastLBP reads H and E
        for key in ("H_fraction_clipped", "E_fraction_clipped"):
            if report[key] > 0.01:
                logging.warning("%s = %.3f: consider raising --he_h_max/--he_e_max", key, report[key])
    lineage.write(f"{args.prefix}.channels.lineage.json", lineage.build(
        "channels", args.process or "TILEPARITY_CHANNELS", name, parameters=vars(args) | {"stain_basis": basis.to_dict()},
        inputs={"image": args.image, "tiles": args.tiles}, outputs=outputs,
        parents=args.lineage_parent, packages=["pyvips", "numpy", "pillow"]))


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
