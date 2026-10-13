#!/usr/bin/env python3
"""Download HEST-1k samples (image + expression + metadata) and write a samplesheet.

    export HF_TOKEN=hf_xxx      # Hugging Face token with access to MahmoodLab/hest
    python scripts/hest_download.py --ids ZEN36 ZEN37 ... --outdir /data/hest \
        --samplesheet samplesheet_cohort_b.csv --cohort B

Cohort B of the project (Valdeolivas et al. 2024, 14 sections) is --ids ZEN36 ... ZEN49.
The pixel size written to the samplesheet comes from HEST's metadata
(pixel_size_um_estimated, i.e. derived from the 100 µm spot spacing).
"""

import argparse
import csv
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--samplesheet", required=True)
    ap.add_argument("--cohort", default="")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    out = Path(args.outdir)
    patterns = [f"{folder}/{i}[_.]*" for i in args.ids for folder in ("wsis", "st", "metadata", "tissue_seg", "thumbnails")]
    snapshot_download("MahmoodLab/hest", repo_type="dataset", allow_patterns=patterns, local_dir=str(out),
                      token=os.environ.get("HF_TOKEN") or None)

    rows = []
    for i in args.ids:
        meta = json.loads((out / "metadata" / f"{i}.json").read_text())
        image = next((out / "wsis").glob(f"{i}.*"))
        # HEST estimates the pixel size from the spot spacing in its own image, i.e. in the
        # pixel grid of the coordinates; the value embedded in the file is the fallback.
        mpp = meta.get("pixel_size_um_estimated") or meta.get("pixel_size_um_embedded")
        rows.append({
            "sample": i,
            "image": str(image.resolve()),
            "spatial": str((out / "st" / f"{i}.h5ad").resolve()),
            "spatial_format": "hest",
            "tiling": "spot",
            "mpp": mpp,
            "patient": meta.get("patient") or "",
            "condition": meta.get("disease_state") or "",
            "cohort": args.cohort,
            "subseries": meta.get("subseries") or "",
        })
    with open(args.samplesheet, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.samplesheet} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
