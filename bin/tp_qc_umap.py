#!/usr/bin/env python3
"""Optional QC (--qc_umap): interactive 3-D UMAP of every feature view of one scale."""

import argparse
import gzip
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import export, lineage, qcumap  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--name", required=True, help="output stem, e.g. combined_mpp0.5000")
    ap.add_argument("--seed", type=int, default=0)
    lineage.add_cli(ap)
    args = ap.parse_args(argv)
    process = args.process or "TILEPARITY_QCUMAP"

    prefixes = export.discover(args.input_dir)
    if not prefixes:
        raise SystemExit("no *.tiles.tsv.gz files found")
    samples = [export.load_sample(args.input_dir, p) for p in prefixes]
    data = qcumap.views(samples)
    obs = data["obs"]
    parents = [str(f) for s in samples for f in s["lineage_parents"]]
    source = lineage.source_text(process, {}, parents=parents, context=f"{len(prefixes)} sample(s), {args.name}")

    sections, rows = [], []
    first = True
    for view, (X, description, vobs) in data["views"].items():
        coords, how = qcumap.embed3d(X, seed=args.seed)
        if coords is None:
            sections.append((view, "", f"{description}. Not computed: {how}."))
            continue
        fig = qcumap.figure(coords, vobs, title=f"{view} - {args.name}")
        sections.append((view, qcumap.to_html(fig, first), f"{description}; {how}; {len(vobs)} tiles."))
        first = False
        rows.append(pd.DataFrame({"tile_id": vobs["tile_id"].to_numpy(), "view": view, "umap1": coords[:, 0],
                                  "umap2": coords[:, 1], "umap3": coords[:, 2]}))
    caption = {
        "What it shows": ("For every feature view of this scale (UNI2-h and LBP on each input set: rgb = the slide, "
                          "rgb_he = the slide rebuilt from H and E only, he = the H and E amounts; expression), a 3-D "
                          f"UMAP of the {len(obs)} extracted tiles of {len(prefixes)} sample(s). Use the menu to colour "
                          "the points by sample, keep, tissue fraction, blur flag or total counts; drag to rotate."),
        "How to read it": ("Points close together have similar features. Samples that form separate islands in a view "
                           "while the tissue types are shared suggest a batch effect in that view; isolated groups "
                           "that coincide with blur_flag, keep = False or low tissue fraction point to technical "
                           "artefacts rather than biology. A batch effect seen with rgb but not with rgb_he points to "
                           "colour that is neither H nor E (scanner tint, pigment, pen)."),
        "Caveats": ("UMAP preserves local neighbourhoods but distorts distances: the size of a group and the gap "
                    "between groups mean nothing. With few tiles the layout is unstable. This is a quick QC look; "
                    "quantitative comparisons belong to the analysis (variance budget)."),
        "Where it comes from": source,
    }
    out_html = f"{args.name}.qc_umap.html"
    Path(out_html).write_text(qcumap.page(sections, f"3-D UMAP QC - {args.name}", caption))
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["tile_id", "view", "umap1", "umap2",
                                                                                  "umap3"])
    with gzip.open(f"{args.name}.qc_umap.tsv.gz", "wt") as fh:
        table.to_csv(fh, sep="\t", index=False)
    lineage.write(f"{args.name}.qc_umap.lineage.json", lineage.build(
        "qc_umap", process, args.name, parameters=vars(args), inputs={},
        outputs={"html": out_html, "coordinates": f"{args.name}.qc_umap.tsv.gz"}, parents=parents,
        packages=["umap-learn", "numba", "scikit-learn", "plotly", "numpy"]))
    logging.info("%s: %d views, %d tiles", args.name, len(rows), len(obs))


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
