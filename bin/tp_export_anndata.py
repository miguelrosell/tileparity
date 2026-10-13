#!/usr/bin/env python3
"""Process 4 (Python) - one AnnData per sample and one combined AnnData per scale."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import export  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--combined-name", required=True, help="file name stem of the combined object")
    ap.add_argument("--no-combined", action="store_true")
    args = ap.parse_args(argv)

    prefixes = export.discover(args.input_dir)
    if not prefixes:
        raise SystemExit("no *.tiles.tsv.gz files found")
    adatas = []
    for prefix in prefixes:
        adata = export.to_anndata(export.load_sample(args.input_dir, prefix))
        adata.write_h5ad(f"{prefix}.h5ad", compression="gzip")
        logging.info("%s: %d tiles x %d genes, obsm %s", prefix, adata.n_obs, adata.n_vars, list(adata.obsm))
        adatas.append(adata)
    if not args.no_combined and len(adatas) > 1:
        combined = export.combine_anndata(adatas)
        if combined is not None:
            combined.write_h5ad(f"{args.combined_name}.h5ad", compression="gzip")


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
