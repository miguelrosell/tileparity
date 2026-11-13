#!/usr/bin/env python3
"""Download HEST-1k samples (image + expression + metadata) and write a samplesheet.

    export HF_TOKEN=hf_xxx      # Hugging Face token; access to MahmoodLab/hest is granted automatically
    python scripts/hest_download.py --ids ZEN36 ZEN37 ... --outdir /data/hest \
        --samplesheet samplesheet_cohort_b.csv --cohort B

or, with a table of samples and their metadata (one row per HEST id):

    python scripts/hest_download.py --from-tsv assets/cohort_c_skin_hest.tsv --outdir /data/hest \
        --samplesheet samplesheet_skin.csv --cohort C

With --from-tsv, the columns patient, condition, disease, lesion and replicate of the table
are written to the samplesheet (they become sample metadata: the cohort summary groups by
condition and pairs by patient). Rows with single_biopsy = False (sections that carry
biopsies of two patients) and rows whose pixel size is coarser than --max-mpp are left out,
with a message.

Cohort B of the project (Valdeolivas et al. 2024, 14 sections) is --ids ZEN36 ... ZEN49.
Cohort C (Schäbitz et al. 2022, inflammatory skin diseases) is assets/cohort_c_skin_hest.tsv.
The pixel size written to the samplesheet comes from HEST's metadata
(pixel_size_um_estimated, i.e. derived from the 100 µm spot spacing).
"""

import argparse
import csv
import json
import os
from pathlib import Path

TSV_COLUMNS = ("patient", "condition", "disease", "lesion", "replicate")


def read_table(path: str, max_mpp: float) -> tuple[list[str], dict]:
    """HEST ids and their metadata from a table with an 'id' column."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    ids, meta = [], {}
    for r in rows:
        if str(r.get("single_biopsy", "True")).strip().lower() in ("false", "0", "no"):
            print(f"skip {r['id']}: the section carries biopsies of more than one patient ({r.get('patient', '')})")
            continue
        try:
            mpp = float(r.get("mpp") or "nan")
        except ValueError:
            mpp = float("nan")
        if mpp == mpp and mpp > max_mpp:
            print(f"skip {r['id']}: pixel size {mpp:.3f} um/px is coarser than --max-mpp {max_mpp}")
            continue
        ids.append(r["id"])
        meta[r["id"]] = {k: r.get(k, "") for k in TSV_COLUMNS if k in r}
    return ids, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids", nargs="+")
    src.add_argument("--from-tsv", help="table with an 'id' column (and optionally patient, condition, ...)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--samplesheet", required=True)
    ap.add_argument("--cohort", default="")
    ap.add_argument("--max-mpp", type=float, default=0.6,
                    help="with --from-tsv: leave out sections coarser than this (um/px); default 0.6")
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded and stop")
    args = ap.parse_args()

    if args.from_tsv:
        ids, table_meta = read_table(args.from_tsv, args.max_mpp)
    else:
        ids, table_meta = args.ids, {}
    print(f"{len(ids)} samples: {' '.join(ids)}")
    if args.dry_run or not ids:
        return

    from huggingface_hub import snapshot_download

    out = Path(args.outdir)
    patterns = [f"{folder}/{i}[_.]*" for i in ids for folder in ("wsis", "st", "metadata", "tissue_seg", "thumbnails")]
    snapshot_download("MahmoodLab/hest", repo_type="dataset", allow_patterns=patterns, local_dir=str(out),
                      token=os.environ.get("HF_TOKEN") or None)

    rows = []
    for i in ids:
        meta = json.loads((out / "metadata" / f"{i}.json").read_text())
        image = next((out / "wsis").glob(f"{i}.*"))
        # HEST estimates the pixel size from the spot spacing in its own image, i.e. in the
        # pixel grid of the coordinates; the value embedded in the file is the fallback.
        mpp = meta.get("pixel_size_um_estimated") or meta.get("pixel_size_um_embedded")
        row = {
            "sample": i,
            "image": str(image.resolve()),
            "spatial": str((out / "st" / f"{i}.h5ad").resolve()),
            "spatial_format": "hest",
            # empty = auto: the pipeline's --visium_tiling decides (grid by default since
            # tileparity 0.2.0); write "spot" here to get spot-centred tiles for a sample
            "tiling": "",
            "mpp": mpp,
            "patient": meta.get("patient") or "",
            "condition": meta.get("disease_state") or "",
            "cohort": args.cohort,
            "subseries": meta.get("subseries") or "",
        }
        for k, v in table_meta.get(i, {}).items():  # the table's metadata wins over HEST's
            if v not in (None, ""):
                row[k] = v
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.samplesheet, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.samplesheet} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
