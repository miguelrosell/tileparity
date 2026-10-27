#!/usr/bin/env python3
"""Process 2b - LBP histograms (and H/E intensity statistics) for every extracted tile."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import channels, features, figures, lbp, lineage  # noqa: E402
from tileparity_lib.tiling import read_tiles  # noqa: E402


def int_list(text):
    return [int(v) for v in text.split(",") if v.strip()] if text else None


def qc_figure(fracs: dict, spec: lbp.LbpSpec, n_tiles: int, source: str, title: str):
    series, rows = [], []
    for c, f in fracs.items():
        if len(f) == 0:
            continue
        mean, p10, p50, p90 = (100 * f.mean(axis=0), 100 * np.percentile(f, 10, axis=0),
                               100 * np.percentile(f, 50, axis=0), 100 * np.percentile(f, 90, axis=0))
        series.append({"name": c, "x": spec.radii, "y": mean, "lo": p10, "hi": p90})
        for k, (r, p) in enumerate(zip(spec.radii, spec.npoints)):
            rows.append({"channel": c, "radius": r, "npoints": p, "mean_pct_irregular": mean[k],
                         "p10": p10[k], "p50": p50[k], "p90": p90[k]})
    chart = figures.line_chart(series, x_label="radius R (px)", y_label="% of the tile's pixels with an irregular pattern",
                               y_range=(0, 100), x_ticks=spec.radii, log_x=len(spec.radii) > 1)
    leg = figures.legend([(figures.SERIES[i % len(figures.SERIES)], s["name"]) for i, s in enumerate(series)],
                         width=chart.width)
    img = figures.with_caption(figures.stack([chart, leg]), title=title,
        what=(f"For each LBP channel, the share of a tile's pixels whose pattern at radius R is irregular (more than two "
              f"0/1 changes round the circle, code P + 1): mean over the {n_tiles} extracted tiles (line) and 10th-90th "
              "percentile across tiles (band)."),
        how=("The histogram describes texture through the regular patterns. A radius at which nearly every pixel is "
             "irregular puts almost all of its P + 2 bins' mass into a single bin and adds little information to the "
             "vector; such radii are candidates for removal (--lbp_radii) or for fewer points (--lbp_npoints)."),
        caveats=("Large circles have many points and are irregular more often by construction. Flat areas (e.g. glass, "
                 "almost constant in every channel) give regular codes, noisy areas irregular ones, so the share also "
                 "depends on the tissue fraction and the scanner noise. Channels: R, G, B of the slide (rgb), of the "
                 "slide rebuilt from H and E only (R_he, G_he, B_he) and the H and E amounts (H, E) when chosen."),
        source=source)
    return img, pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels-image", required=True, help="channels image from Process 1b")
    ap.add_argument("--channels-json", default=None,
                    help="report of Process 1b, with the band order of the channels image (default: the order "
                         "Process 1b writes for --channel-sets alone)")
    ap.add_argument("--tiles", required=True, help="tile table from Process 1")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--radii", default=None, help="comma-separated radii in px (default: fastLBP progression)")
    ap.add_argument("--npoints", default=None, help="comma-separated points per radius (default: ceil(2*pi*R))")
    ap.add_argument("--max-radius", type=float, default=None, help="default: tile_px / 4")
    ap.add_argument("--channel-sets", default="rgb,rgb_he", help=f"comma-separated subset of {','.join(channels.LBP_SETS)}")
    ap.add_argument("--center", action="store_true", help="also write the LBP_CENTER_* blocks (central pixels only)")
    ap.add_argument("--ncpus", type=int, default=1)
    lineage.add_cli(ap)
    args = ap.parse_args(argv)

    tiles = read_tiles(args.tiles)
    tile_px = int(tiles["tile_px"].iloc[0]) if len(tiles) else 224
    radii = int_list(args.radii) or lbp.default_radii(args.max_radius or tile_px / 4)
    npoints = int_list(args.npoints) or [lbp.npoints_for(r) for r in radii]
    if len(npoints) != len(radii):
        ap.error("--npoints must have one value per radius")
    try:
        sets = channels.parse_sets(args.channel_sets, channels.LBP_SETS)
    except ValueError as err:
        ap.error(str(err))
    chans = channels.lbp_channels(sets)
    report = json.loads(Path(args.channels_json).read_text()) if args.channels_json else {}
    bands = report.get("bands") or channels.band_order(sets)
    missing = [c for c in [*chans, "H", "E"] if c not in bands]
    if missing:
        ap.error(f"the channels image has bands {bands}, not {missing}: Process 1b ran with other input sets")
    spec = lbp.LbpSpec(radii=radii, npoints=npoints, channels=chans, tile_px=tile_px, bands=bands, center=args.center)
    logging.info("%d tiles, radii %s, channels %s, %d features per channel", len(tiles), radii, chans,
                 spec.n_features_per_channel)

    out = lbp.extract(args.channels_image, tiles, spec, ncpus=args.ncpus)
    blocks = {lbp.block_name(c): (out[c], spec.columns(c)) for c in chans}
    if args.center:
        blocks.update({lbp.block_name(c, center=True): (out[f"center_{c}"], [f"center_{x}" for x in spec.columns(c)])
                       for c in chans})
    blocks["INTENSITY_HE"] = (out["intensity"], lbp.intensity_columns())
    lbp_attrs = {"radii": radii, "npoints": npoints, "method": "uniform (fastLBP / scikit-image 'uniform')",
                 "kernel": "fastlbp.lbp.uniform_lbp_uint8_padded_absolute", "tile_px": tile_px,
                 "channel_sets": sets, "margin": "real neighbouring pixels; mirrored outside the slide",
                 "values": "raw counts of the tile's pixels per code (each radius sums to tile_px^2)"}
    if report.get("stain_vectors"):
        lbp_attrs["stain_vectors"] = report["stain_vectors"]
    center_attrs = {**lbp_attrs, "values": "mean one-hot code vector of the tile's central pixels (2 x 2 for an even "
                                           "tile size); ablation of the histogram"}
    name = Path(args.prefix).name
    source = lineage.source_text(args.process or "FASTLBP_EXTRACT", {"channels": args.channels_image},
                                 parents=args.lineage_parent, context=f"tiles of {name}")
    fig, qc_table = qc_figure({c: lbp.irregular_fractions(out[c], spec) for c in chans}, spec, len(tiles), source,
                              title=f"{name} - LBP: share of irregular patterns per radius")
    fig.save(f"{args.prefix}.lbp_qc.png")
    qc_table.to_csv(f"{args.prefix}.lbp_qc.tsv", sep="\t", index=False)
    features.write(f"{args.prefix}.lbp.h5", tiles["tile_id"].to_numpy(dtype=object), blocks,
                   attrs={"extractor": "fastLBP", **lbp_attrs},
                   block_attrs={b: (center_attrs if b.startswith("LBP_CENTER_") else lbp_attrs)
                                for b in blocks if b.startswith("LBP_")})
    lineage.write(f"{args.prefix}.lbp.lineage.json", lineage.build(
        "lbp", args.process or "FASTLBP_EXTRACT", name, parameters=vars(args),
        inputs={"channels": args.channels_image, "channels_report": args.channels_json, "tiles": args.tiles},
        outputs={"features": f"{args.prefix}.lbp.h5", "qc_figure": f"{args.prefix}.lbp_qc.png",
                 "qc_table": f"{args.prefix}.lbp_qc.tsv"},
        parents=args.lineage_parent, packages=["fastlbp", "numpy", "pyvips", "h5py"]))


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
