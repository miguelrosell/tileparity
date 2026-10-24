#!/usr/bin/env python3
"""Process 4 (Python) - one AnnData per sample and one combined AnnData per scale."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import export, lineage  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--combined-name", required=True, help="file name stem of the combined object")
    ap.add_argument("--no-combined", action="store_true")
    lineage.add_cli(ap)
    args = ap.parse_args(argv)
    process = args.process or "TILEPARITY_EXPORTANNDATA"

    prefixes = export.discover(args.input_dir)
    if not prefixes:
        raise SystemExit("no *.tiles.tsv.gz files found")
    adatas, leaf_lineages, parents_by_sample = [], [], {}
    for prefix in prefixes:
        sample = export.load_sample(args.input_dir, prefix)
        parents_by_sample[str(sample["provenance"]["prepare"]["sample_id"])] = sample["lineage_parents"]
        adata = export.to_anndata(sample)
        adata.write_h5ad(f"{prefix}.h5ad", compression="gzip")
        rec = lineage.build("anndata", process, prefix, parameters=vars(args), inputs={},
                            outputs={"anndata": f"{prefix}.h5ad"}, parents=sample["lineage_parents"],
                            packages=["anndata", "numpy", "scipy", "pandas", "h5py"])
        lineage.write(f"{prefix}.anndata.lineage.json", rec)
        leaf_lineages.append(f"{prefix}.anndata.lineage.json")
        logging.info("%s: %d tiles x %d genes, obsm %s", prefix, adata.n_obs, adata.n_vars, list(adata.obsm))
        adatas.append(adata)
    if not args.no_combined and len(adatas) > 1:
        combined = export.combine_anndata(adatas)
        if combined is not None:
            # the object carries the lineage of every sample it contains, like the per-sample objects
            included = combined.uns["tileparity"]["samples"]
            merged = lineage.build("combined", "tileparity", args.combined_name, parameters={}, inputs={}, outputs={},
                                   parents=[p for sid in included for p in parents_by_sample.get(sid, [])])
            combined.uns["tileparity"]["lineage"] = json.dumps(merged, default=str)
            combined.write_h5ad(f"{args.combined_name}.h5ad", compression="gzip")
            lineage.write(f"{args.combined_name}.anndata.lineage.json", lineage.build(
                "anndata_combined", process, args.combined_name, parameters=vars(args), inputs={},
                outputs={"anndata": f"{args.combined_name}.h5ad"}, parents=leaf_lineages,
                packages=["anndata", "numpy", "scipy", "pandas", "h5py"]))


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
