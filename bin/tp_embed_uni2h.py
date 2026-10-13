#!/usr/bin/env python3
"""Processes 1a + 2a - UNI2-h embeddings (1536-d class token) for every kept tile."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import features, uni2h  # noqa: E402
from tileparity_lib.tiling import read_tiles  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="resampled RGB image from Process 1")
    ap.add_argument("--tiles", required=True, help="tile table from Process 1")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--weights", default=None, help="UNI2-h pytorch_model.bin (otherwise downloaded from the hub)")
    ap.add_argument("--test-mode", action="store_true", help="1-block random model, for pipeline tests only")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args(argv)

    tiles = read_tiles(args.tiles)
    if len(tiles) and int(tiles["tile_px"].iloc[0]) != 224:
        ap.error("UNI2-h expects 224 px tiles (--tile_px 224)")

    model, source = uni2h.build_model(args.weights, args.test_mode)
    emb, info = uni2h.embed(args.image, tiles, model, device=args.device, precision=args.precision,
                            batch_size=args.batch_size, num_workers=args.num_workers)
    attrs = {"extractor": "UNI2-h", "model_source": source, "embedding": "class token",
             "normalisation": f"x/255, ImageNet mean {uni2h.IMAGENET_MEAN} std {uni2h.IMAGENET_STD}",
             "resize": "none (tiles are cut at 224 px)", **info}
    if args.weights:
        attrs["weights_sha256"] = uni2h.sha256(args.weights)
    if args.test_mode:
        logging.warning("TEST MODE: embeddings come from a random 1-block network, not UNI2-h")
    columns = [f"uni2h_{i:04d}" for i in range(uni2h.EMBED_DIM)]
    features.write(f"{args.prefix}.uni2h.h5", tiles["tile_id"].to_numpy(dtype=object), {"UNI2H": (emb, columns)},
                   attrs=attrs, block_attrs={"UNI2H": attrs})


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
