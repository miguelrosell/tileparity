#!/usr/bin/env python3
"""Processes 1a + 2a - UNI2-h embeddings (1536-d class token) for every extracted tile, one
block per input set: UNI2H_RGB (the resampled image) and UNI2H_RGB_HE (the same image rebuilt
from haematoxylin and eosin only, from the channels image of Process 1b)."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import channels, features, lineage, uni2h  # noqa: E402
from tileparity_lib.tiling import read_tiles  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="resampled RGB image from Process 1 (input set rgb)")
    ap.add_argument("--tiles", required=True, help="tile table from Process 1")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--inputs", default="rgb", help=f"comma-separated subset of {','.join(channels.UNI2H_INPUTS)}")
    ap.add_argument("--channels-image", default=None, help="channels image from Process 1b (input set rgb_he)")
    ap.add_argument("--channels-json", default=None, help="report of Process 1b, with the band order of the channels image")
    ap.add_argument("--weights", default=None, help="UNI2-h pytorch_model.bin (otherwise downloaded from the hub)")
    ap.add_argument("--test-mode", action="store_true", help="1-block random model, for pipeline tests only")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    lineage.add_cli(ap)
    args = ap.parse_args(argv)

    try:
        inputs = channels.parse_sets(args.inputs, channels.UNI2H_INPUTS)
    except ValueError as err:
        ap.error(str(err))
    sources = {"rgb": (args.image, None)}
    report = {}
    if "rgb_he" in inputs:
        if not (args.channels_image and args.channels_json):
            ap.error("input set rgb_he needs --channels-image and --channels-json (Process 1b)")
        report = json.loads(Path(args.channels_json).read_text())
        bands = report.get("bands", [])
        missing = [b for b in channels.CHANNEL_SETS["rgb_he"] if b not in bands]
        if missing:
            ap.error(f"the channels image has bands {bands}, not {missing}: Process 1b ran without rgb_he")
        sources["rgb_he"] = (args.channels_image, [bands.index(b) for b in channels.CHANNEL_SETS["rgb_he"]])

    tiles = read_tiles(args.tiles)
    if len(tiles) and int(tiles["tile_px"].iloc[0]) != 224:
        ap.error("UNI2-h expects 224 px tiles (--tile_px 224)")

    model, source = uni2h.build_model(args.weights, args.test_mode)
    attrs = {"extractor": "UNI2-h", "model_source": source, "embedding": "class token", "inputs": inputs,
             "normalisation": f"x/255, ImageNet mean {uni2h.IMAGENET_MEAN} std {uni2h.IMAGENET_STD}",
             "resize": "none (tiles are cut at 224 px)"}
    if args.weights:
        attrs["weights_sha256"] = uni2h.sha256(args.weights)
    if args.test_mode:
        logging.warning("TEST MODE: embeddings come from a random 1-block network, not UNI2-h")
    blocks, block_attrs = {}, {}
    for s in inputs:
        path, bands = sources[s]
        logging.info("UNI2-h on input set %s: %d tiles of %s", s, len(tiles), Path(path).name)
        emb, info = uni2h.embed(path, tiles, model, device=args.device, precision=args.precision,
                                batch_size=args.batch_size, num_workers=args.num_workers, bands=bands)
        name = uni2h.block_name(s)
        blocks[name] = (emb, uni2h.columns(s))
        block_attrs[name] = {**attrs, **info, "input": s, "input_description": uni2h.INPUT_DESCRIPTION[s]}
        if s == "rgb_he" and report.get("stain_vectors"):
            block_attrs[name]["stain_vectors"] = report["stain_vectors"]
        attrs.update(info)
    features.write(f"{args.prefix}.uni2h.h5", tiles["tile_id"].to_numpy(dtype=object), blocks,
                   attrs=attrs, block_attrs=block_attrs)
    params = vars(args) | {"model_source": source, "weights_sha256": attrs.get("weights_sha256")}
    lineage.write(f"{args.prefix}.uni2h.lineage.json", lineage.build(
        "uni2h", args.process or "UNI2H_EMBED", Path(args.prefix).name, parameters=params,
        inputs={"image": args.image, "tiles": args.tiles,
                "channels": args.channels_image if "rgb_he" in inputs else None,
                "channels_report": args.channels_json if "rgb_he" in inputs else None},
        outputs={"features": f"{args.prefix}.uni2h.h5"},
        parents=args.lineage_parent, packages=["torch", "timm", "pyvips", "h5py", "numpy"]))


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
