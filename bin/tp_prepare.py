#!/usr/bin/env python3
"""Process 1 - common preparation: resample, tissue mask, tiles, QC, unit->tile assignment."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import mask, prepare  # noqa: E402


def parse_meta(items):
    meta = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if not key:
            raise SystemExit(f"--meta expects key=value, got {item!r}")
        meta[key] = value
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--mpp-target", required=True, type=float, help="target pixel size (µm/px)")
    ap.add_argument("--prefix", required=True, help="output file prefix")
    ap.add_argument("--mpp-native", type=float, default=None, help="native pixel size; overrides file metadata")
    ap.add_argument("--spatial", type=Path, default=None)
    ap.add_argument("--spatial-format", choices=["spaceranger_hd", "spaceranger", "hest"], default=None)
    ap.add_argument("--tiling", choices=["auto", "grid", "spot"], default="auto")
    ap.add_argument("--hd-bin-um", type=int, default=8)
    ap.add_argument("--tile-px", type=int, default=224)
    ap.add_argument("--min-tissue", type=float, default=0.5)
    ap.add_argument("--min-st-coverage", type=float, default=0.5)
    ap.add_argument("--upsample-tolerance", type=float, default=0.02)
    ap.add_argument("--resample-kernel", default="lanczos3",
                    choices=["nearest", "linear", "cubic", "mitchell", "lanczos2", "lanczos3"])
    ap.add_argument("--compression", default="deflate", choices=["deflate", "lzw", "zstd", "none"])
    ap.add_argument("--blur-mad", type=float, default=3.0)
    ap.add_argument("--drop-blurred", action="store_true", help="drop blur-flagged tiles (default: keep and flag)")
    ap.add_argument("--mask-mpp", type=float, default=4.0)
    ap.add_argument("--mask-smooth-um", type=float, default=8.0)
    ap.add_argument("--mask-min-value", type=float, default=0.10)
    ap.add_argument("--mask-close-um", type=float, default=16.0)
    ap.add_argument("--mask-fill-holes-um2", type=float, default=20000.0)
    ap.add_argument("--mask-min-object-um2", type=float, default=20000.0)
    ap.add_argument("--meta", action="append", help="sample metadata column, key=value (repeatable)")
    args = ap.parse_args(argv)

    if args.spatial is not None and args.spatial_format is None:
        ap.error("--spatial needs --spatial-format")

    params = prepare.PrepareParams(
        sample_id=args.sample,
        image=args.image,
        mpp_target=args.mpp_target,
        out_prefix=args.prefix,
        mpp_native=args.mpp_native,
        spatial=args.spatial,
        spatial_format=args.spatial_format,
        tiling=args.tiling,
        hd_bin_um=args.hd_bin_um,
        tile_px=args.tile_px,
        min_tissue=args.min_tissue,
        min_st_coverage=args.min_st_coverage,
        upsample_tolerance=args.upsample_tolerance,
        resample_kernel=args.resample_kernel,
        compression=args.compression,
        blur_mad=args.blur_mad,
        drop_blurred=args.drop_blurred,
        mask=mask.MaskParams(
            mask_mpp=args.mask_mpp,
            smooth_um=args.mask_smooth_um,
            min_value=args.mask_min_value,
            close_um=args.mask_close_um,
            fill_holes_um2=args.mask_fill_holes_um2,
            min_object_um2=args.mask_min_object_um2,
        ),
        metadata=parse_meta(args.meta),
    )
    report = prepare.run(params)
    if report["status"] == "ok":
        t = report["tiles"]
        logging.info("%s: %d candidate tiles, %d kept", args.sample, t["n_candidates"], t["n_keep"])


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
