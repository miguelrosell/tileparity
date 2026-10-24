#!/usr/bin/env python3
"""Process 1b - LBP adapter: write the single-channel images (H, E and, per channel set, grey / R, G, B)."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import channels, image, lineage  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="resampled RGB image from Process 1")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--channel-sets", default="he,grey", help=f"comma-separated subset of {','.join(channels.CHANNEL_SETS)}")
    ap.add_argument("--h-max", type=float, default=0.30, help="haematoxylin value mapped to 255")
    ap.add_argument("--e-max", type=float, default=0.15, help="eosin value mapped to 255")
    ap.add_argument("--compression", default="deflate", choices=["deflate", "lzw", "zstd", "none"])
    lineage.add_cli(ap)
    args = ap.parse_args(argv)

    try:
        sets = channels.parse_sets(args.channel_sets)
    except ValueError as err:
        ap.error(str(err))
    bands = channels.band_order(sets)
    rgb = image.open_rgb(args.image, access="sequential")
    mpp = image.mpp_from_metadata(rgb)
    out = channels.rgb_to_channels_vips(rgb, args.h_max, args.e_max, bands)
    image.write_tiled_tiff(out, f"{args.prefix}.channels.tif", mpp=mpp, compression=args.compression)

    rgb = image.open_rgb(args.image, access="sequential")
    report = {
        "channel_sets": sets,
        "bands": bands,
        "lbp_channels": channels.lbp_channels(sets),
        "h_max": args.h_max,
        "e_max": args.e_max,
        "grey_weights": list(channels.GREY_WEIGHTS),
        "deconvolution": "skimage.color.rgb2hed (Ruifrok & Johnston 2001)",
        **channels.clipping_fractions(rgb, args.h_max, args.e_max),
    }
    Path(f"{args.prefix}.channels.json").write_text(json.dumps(report, indent=2))
    for key in ("H_fraction_clipped", "E_fraction_clipped"):
        if report[key] > 0.01:
            logging.warning("%s = %.3f: consider raising --h-max/--e-max", key, report[key])
    lineage.write(f"{args.prefix}.lbpchannels.lineage.json", lineage.build(
        "lbpchannels", args.process or "TILEPARITY_LBPCHANNELS", Path(args.prefix).name, parameters=vars(args),
        inputs={"image": args.image}, outputs={"channels": f"{args.prefix}.channels.tif",
                                              "report": f"{args.prefix}.channels.json"},
        parents=args.lineage_parent, packages=["pyvips", "scikit-image", "numpy"]))


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
