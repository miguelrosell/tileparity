#!/usr/bin/env python3
"""Process 2b - LBP histograms (and H/E intensity statistics) for every kept tile."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import features, lbp  # noqa: E402
from tileparity_lib.channels import CHANNELS  # noqa: E402
from tileparity_lib.tiling import read_tiles  # noqa: E402


def int_list(text):
    return [int(v) for v in text.split(",") if v.strip()] if text else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels-image", required=True, help="H/E/grey image from Process 1b")
    ap.add_argument("--tiles", required=True, help="tile table from Process 1")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--radii", default=None, help="comma-separated radii in px (default: fastLBP progression)")
    ap.add_argument("--npoints", default=None, help="comma-separated points per radius (default: ceil(2*pi*R))")
    ap.add_argument("--max-radius", type=float, default=None, help="default: tile_px / 4")
    ap.add_argument("--channels", default="H,E,grey", help=f"subset of {','.join(CHANNELS)}")
    ap.add_argument("--ncpus", type=int, default=1)
    args = ap.parse_args(argv)

    tiles = read_tiles(args.tiles)
    tile_px = int(tiles["tile_px"].iloc[0]) if len(tiles) else 224
    radii = int_list(args.radii) or lbp.default_radii(args.max_radius or tile_px / 4)
    npoints = int_list(args.npoints) or [lbp.npoints_for(r) for r in radii]
    if len(npoints) != len(radii):
        ap.error("--npoints must have one value per radius")
    chans = [c.strip() for c in args.channels.split(",") if c.strip()]
    unknown = set(chans) - set(CHANNELS)
    if unknown:
        ap.error(f"unknown channels {sorted(unknown)}; choose from {CHANNELS}")
    spec = lbp.LbpSpec(radii=radii, npoints=npoints, channels=chans, tile_px=tile_px)
    logging.info("%d tiles, radii %s, channels %s, %d features per channel", len(tiles), radii, chans,
                 spec.n_features_per_channel)

    out = lbp.extract(args.channels_image, tiles, spec, ncpus=args.ncpus)
    blocks = {f"LBP_{c.upper()}": (out[c], spec.columns(c)) for c in chans}
    blocks["INTENSITY_HE"] = (out["intensity"], lbp.intensity_columns())
    lbp_attrs = {"radii": radii, "npoints": npoints, "method": "uniform (fastLBP / scikit-image 'uniform')",
                 "kernel": "fastlbp.lbp.uniform_lbp_uint8_padded_absolute", "tile_px": tile_px,
                 "margin": "real neighbouring pixels; mirrored outside the slide"}
    features.write(f"{args.prefix}.lbp.h5", tiles["tile_id"].to_numpy(dtype=object), blocks,
                   attrs={"extractor": "fastLBP", **lbp_attrs},
                   block_attrs={name: lbp_attrs for name in blocks if name.startswith("LBP_")})


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
