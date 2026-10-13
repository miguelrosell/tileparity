#!/usr/bin/env python3
"""Process 3 - expression per tile: Y = X A (sum of unit counts within each kept tile)."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import spatial, st  # noqa: E402
from tileparity_lib.tiling import read_tiles  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--spatial", default=None)
    ap.add_argument("--spatial-format", choices=list(spatial.FORMATS), default=None)
    ap.add_argument("--hd-bin-um", type=int, default=8)
    args = ap.parse_args(argv)

    tiles = read_tiles(args.tiles)
    tile_ids = tiles["tile_id"].to_numpy(dtype=object)
    report = {"n_tiles": int(len(tile_ids))}

    if args.spatial:
        sp = spatial.resolve(args.spatial, args.spatial_format, args.hd_bin_um)
        counts = spatial.read_counts(sp)
        assignment = pd.read_csv(args.assignment, sep="\t", dtype=str)
        Y, summary, rep = st.aggregate(counts, assignment, tile_ids)
        gene_ids, gene_names = counts.gene_ids, counts.gene_names
        report.update(rep, n_genes=int(Y.shape[0]), source=str(sp.counts))
    else:
        Y = sparse.csc_matrix((0, len(tile_ids)), dtype=np.int64)
        gene_ids = gene_names = np.array([], dtype=object)
        summary = pd.DataFrame({"tile_id": tile_ids, "n_units_summed": 0, "total_counts": 0, "n_genes_detected": 0})
        report.update(n_genes=0, source=None)

    st.write_counts_h5(f"{args.prefix}.counts.h5", Y, tile_ids, gene_ids, gene_names)
    summary.to_csv(f"{args.prefix}.counts_summary.tsv", sep="\t", index=False)
    Path(f"{args.prefix}.aggregate.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
