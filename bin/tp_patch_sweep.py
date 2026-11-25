#!/usr/bin/env python3
"""Optional (--patch_sweep, Visium HD only): what LBP knows about the expression as a function
of the patch size, from LBP histograms counted once per 8 µm cell and summed into patches of
16 to 224 µm. See tileparity_lib/patchsweep.py for the method."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tileparity_lib import channels, figures, lbp, lineage, patchsweep, spatial  # noqa: E402
from tileparity_lib.tiling import read_tiles  # noqa: E402

log = logging.getLogger("tp_patch_sweep")


def gene_panel(svl: pd.DataFrame, args) -> "figures.Image.Image | None":
    """Scatter: per gene, share of the reachable with small patches against with large ones."""
    if svl is None or svl.empty:
        return None
    x, y = svl["fraction_large"].to_numpy(float), svl["fraction_small"].to_numpy(float)
    above = y > x
    colours = [(213, 94, 0) if a else (0, 114, 178) for a in above]
    lab = [(int(i), str(svl["gene_name"].iloc[i])) for i in np.argsort(-svl["difference"].to_numpy(float))[:8]]
    lo = float(min(0.0, np.nanmin(np.r_[x, y])))
    hi = float(max(1.0, np.nanmax(np.r_[x, y])))
    chart = figures.scatter_chart(x, y, colours, f"share of the reachable, patches >= {args.large_um:g} um (mean)",
                                  f"share of the reachable, patches <= {args.small_um:g} um (mean)", size=(560, 480),
                                  x_range=(lo, hi), y_range=(lo, hi), labels=lab)
    items = [((213, 94, 0), f"better with small patches ({int(above.sum())})"),
             ((0, 114, 178), f"better with large patches ({int((~above).sum())})")]
    return figures.stack([figures.title_strip(560, "d. Gene by gene: small against large patches", 14), chart,
                          figures.legend(items, width=560)])


def figure(table: pd.DataFrame, radius: pd.DataFrame, spot: dict, args, prefix: str, source: str, n_tiles: int,
           channel_set: str, svl: pd.DataFrame | None = None):
    t = table.dropna(subset=["r2_full"]).sort_values("size_um")
    if t.empty:
        return None
    x = t["size_um"].to_numpy(float)
    series = [{"name": "R2 of the expression from LBP", "x": x, "y": t["r2_full"].to_numpy(float),
               "lo": t["r2_full_ci_low"].to_numpy(float), "hi": t["r2_full_ci_high"].to_numpy(float),
               "colour": (213, 94, 0)},
              {"name": "half B from LBP", "x": x, "y": t["r2_half"].to_numpy(float), "colour": (230, 159, 0)},
              {"name": "ceiling of half B (share that is not sampling noise)", "x": x,
               "y": t["r2_ceiling"].to_numpy(float), "colour": (90, 90, 90)}]
    v = [(spot["sweet_spot_um"], f"sweet spot {spot['sweet_spot_um']:g} um")] if spot else []
    ticks = [s for s in (16, 32, 48, 64, 96, 112, 128, 160, 192, 224) if x.min() <= s <= x.max()]
    a = figures.line_chart(series, "patch side (um)", "R2", size=(620, 360), y_range=(min(0.0, float(np.nanmin(
        t[["r2_full", "r2_half", "r2_ceiling"]].to_numpy(float)))), 1.0), x_ticks=ticks, v_lines=v)
    fr = t.dropna(subset=["fraction"])
    b_series = [{"name": "share of the reachable, expression components", "x": fr["size_um"].to_numpy(float),
                 "y": fr["fraction"].to_numpy(float),
                 "lo": (fr["fraction"] - fr["fraction_se"].fillna(0)).to_numpy(float),
                 "hi": (fr["fraction"] + fr["fraction_se"].fillna(0)).to_numpy(float), "colour": (0, 114, 178)}]
    if "fraction_genes" in t and t["fraction_genes"].notna().any():
        g = t.dropna(subset=["fraction_genes"])
        b_series.append({"name": f"share of the reachable, {int(g['n_genes'].iloc[0])} genes one by one (same genes at "
                                 "every size)", "x": g["size_um"].to_numpy(float), "y": g["fraction_genes"].to_numpy(float),
                         "colour": (213, 94, 0)})
    for k, colour in zip(args.k, ((0, 158, 115), (204, 121, 167), (117, 77, 160))):
        col = f"ari_k{k}"
        if col in t:
            b_series.append({"name": f"ARI LBP vs expression, k-means k = {k}", "x": x, "y": t[col].to_numpy(float),
                             "colour": colour})
    b = figures.line_chart(b_series, "patch side (um)", "share of the reachable / ARI", size=(620, 360),
                           y_range=(0.0, 1.0), x_ticks=ticks, v_lines=v)
    left = figures.stack([figures.title_strip(620, "a. What LBP predicts of the expression, by patch size", 14), a,
                          figures.legend([(s["colour"], s["name"]) for s in series], width=620)])
    right = figures.stack([figures.title_strip(620, "b. Share of the reachable (R2 from LBP / ceiling) and ARI", 14), b,
                           figures.legend([(s["colour"], s["name"]) for s in b_series], width=620)])
    parts = [figures.stack([_row([left, right])])]
    d = gene_panel(svl, args)
    if radius is not None and len(radius):
        sizes = sorted(radius["size_um"].unique())
        radii = sorted(radius["radius_px"].unique())
        M = np.full((len(radii), len(sizes)), np.nan)
        for r in radius.itertuples(index=False):
            M[radii.index(r.radius_px), sizes.index(r.size_um)] = r.fraction
        from tileparity_lib import clusterplots

        heat = clusterplots.rect_heatmap(M, [f"radius {int(r)} px" for r in radii], [f"{s:g}" for s in sizes],
                                         "c. Share of the reachable with one LBP radius alone (columns: patch side, um)",
                                         cell=40)
        c_panel = figures.stack([heat, clusterplots.colour_bar(label="share")])
        parts.append(_row([c_panel, d]) if d is not None else c_panel)
    elif d is not None:
        parts.append(d)
    body = figures.stack(parts)
    caption = {
        "what": (f"How much of the expression LBP predicts as the patch grows, on the {n_tiles} kept tiles of this slide "
                 f"cut into cells of {args.cell_um:g} um: LBP codes (input set {channel_set}, radii up to "
                 f"{args.max_radius} px, a radius used only when it is at most a quarter of the patch) counted per cell, "
                 "summed into non-overlapping square patches; expression (Visium HD bins) summed the same way. a: R2 of "
                 "the expression's principal components predicted from LBP's (ridge regression, spatial blocks of "
                 f"{args.block_um:g} um; band: 95 % block-bootstrap interval); and, with every patch's molecules split at "
                 "random in two halves (two independent samples of the same tissue), R2 of half B from LBP and half B's "
                 "ceiling: the variance the two halves share along half B's components over half B's variance, i.e. the "
                 "share of half B that is not sampling noise and the most any predictor can reach with that many "
                 "molecules. b: the share of the reachable, R2(half B from LBP) / ceiling (band: +- standard error over "
                 f"the folds); the same for the {args.genes} genes most variable at {args.gene_size_um:g} um, the same "
                 "genes at every size, each predicted on its own (sum of their R2 / sum of their ceilings; a gene's "
                 "ceiling is its correlation between the halves); and the ARI between k-means clusters of LBP and of the "
                 "expression. c: the share of the reachable with each radius alone. d: per gene, the share of the "
                 f"reachable with small patches (mean over sizes up to {args.small_um:g} um) against large ones (sizes from "
                 f"{args.large_um:g} um), for the genes whose ceiling is at least 0.1 at every small size."),
        "how": ("The share of the reachable is comparable between sizes: it removes the loss of molecules in small "
                "patches. Its peak is the patch size at which texture says most about the expression; the sweet spot "
                "(dashed) is the smallest size within one standard error of the peak. The gene curve asks the same of "
                "individual genes rather than of the expression's main axes. In d, genes above the diagonal are "
                "predicted relatively better with small patches (less mixing of cells), below it with large ones."),
        "caveats": ("Expression bins are given to the cell containing their centre (bins and cells are not aligned). "
                    "Large patches are few, so their values are less precise; very small patches hold few molecules, so "
                    "a gene's share there is noisy (confirm a gene above the diagonal on other slides). The radii are the "
                    "same at every size (up to the maximum), so a larger patch does not see larger structures, only more "
                    "pixels and more cells. Linear prediction. Values: patch_sweep.tsv, patch_sweep_radius.tsv, "
                    "patch_sweep_genes.tsv.gz, patch_sweep_genes_small_vs_large.tsv."),
        "source": source,
    }
    return figures.with_caption(body, title=f"Patch size of fastLBP - {prefix}", min_width=1200, **caption)


def _row(images):
    from tileparity_lib.clusterplots import _side_by_side

    return _side_by_side(images)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels-image", required=True)
    ap.add_argument("--channels-json", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--prepare-json", required=True)
    ap.add_argument("--spatial", required=True, help="Visium HD binned_outputs (or outs/) folder")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--sizes-um", default="16:224:8")
    ap.add_argument("--cell-um", type=float, default=8.0)
    ap.add_argument("--channel-set", default="rgb")
    ap.add_argument("--max-radius", type=int, default=8)
    ap.add_argument("--k", default="4,8")
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--n-hvg", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-um", type=float, default=1000.0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--max-patches", type=int, default=60000)
    ap.add_argument("--genes", type=int, default=500, help="genes followed one by one at every size (0: none)")
    ap.add_argument("--gene-size-um", type=float, default=112.0,
                    help="patch side at which the genes are chosen (the most variable there)")
    ap.add_argument("--small-um", type=float, default=32.0, help="largest 'small' patch for the gene comparison")
    ap.add_argument("--large-um", type=float, default=176.0, help="smallest 'large' patch for the gene comparison")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ncpus", type=int, default=1)
    lineage.add_cli(ap)
    args = ap.parse_args(argv)
    args.k = [int(x) for x in str(args.k).split(",") if x.strip()]
    process = args.process or "TILEPARITY_PATCH_SWEEP"
    prefix = args.prefix
    prep = json.loads(Path(args.prepare_json).read_text())
    mpp, scale = float(prep["mpp_target"]), float(prep["scale_factor"])
    tiles = read_tiles(args.tiles, subset="keep")
    T = int(tiles["tile_px"].iloc[0])
    cell_px = args.cell_um / mpp
    out = {"table": f"{prefix}.patch_sweep.tsv", "radius": f"{prefix}.patch_sweep_radius.tsv",
           "genes": f"{prefix}.patch_sweep_genes.tsv.gz", "genes_small_vs_large": f"{prefix}.patch_sweep_genes_small_vs_large.tsv",
           "summary": f"{prefix}.patch_sweep.json"}
    summary = {"prefix": prefix, "mpp": mpp, "cell_um": args.cell_um, "parameters": {k: v for k, v in vars(args).items()
                                                                                      if k not in ("lineage_parent",)}}

    def finish(status: str, table=None, radius=None, spot=None, genes=None, svl=None):
        summary["status"] = status
        (table if table is not None else pd.DataFrame(columns=patchsweep.TABLE_COLUMNS)).to_csv(
            out["table"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
        (radius if radius is not None else pd.DataFrame(columns=["size_um", "radius_px", "r2_half", "fraction"])).to_csv(
            out["radius"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
        (genes if genes is not None else pd.DataFrame(columns=patchsweep.GENE_COLUMNS)).to_csv(
            out["genes"], sep="\t", index=False, na_rep="NA", float_format="%.6g", compression="gzip")
        (svl if svl is not None else patchsweep.small_vs_large(None)).to_csv(
            out["genes_small_vs_large"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
        if spot:
            summary["sweet_spot"] = spot
        Path(out["summary"]).write_text(json.dumps(lineage._jsonable(summary), indent=2))
        lineage.write(f"{prefix}.patch_sweep.lineage.json", lineage.build(
            "patch_sweep", process, prefix, parameters=summary["parameters"],
            inputs={"channels": args.channels_image, "tiles": args.tiles, "spatial": args.spatial}, outputs=out,
            parents=[p for p in (args.lineage_parent or [])], packages=["fastlbp", "scikit-learn", "numpy", "scipy"]))
        log.info("%s: %s", prefix, status)

    if abs(cell_px - round(cell_px)) > 1e-6 or T % round(cell_px):
        return finish(f"not computed: a cell of {args.cell_um:g} um is {cell_px:g} px at {mpp:g} um/px, not a whole "
                      f"divisor of the {T} px tile")
    cell_px = int(round(cell_px))
    if args.cell_um not in (2, 8, 16):
        return finish(f"not computed: Visium HD bins are 2, 8 or 16 um, not {args.cell_um:g}")
    try:
        sp = spatial.resolve(args.spatial, "spaceranger_hd", int(args.cell_um))
    except Exception as exc:  # not Visium HD
        return finish(f"not computed: no Visium HD bins of {args.cell_um:g} um ({exc})")
    report = json.loads(Path(args.channels_json).read_text())
    bands = report.get("bands") or channels.band_order([args.channel_set])
    chans = list(channels.CHANNEL_SETS[args.channel_set])
    if any(c not in bands for c in chans):
        return finish(f"not computed: the channels image has no bands {chans} (Process 1b ran without "
                      f"'{args.channel_set}')")
    radii = [r for r in lbp.default_radii(T / 4) if r <= args.max_radius]
    spec = lbp.LbpSpec(radii=radii, npoints=[lbp.npoints_for(r) for r in radii], channels=chans, tile_px=T, bands=bands)
    log.info("%s: LBP per %d px cell on %d tiles (%s, radii %s)", prefix, cell_px, len(tiles), chans, radii)
    Hc, cr, cc = patchsweep.cell_histograms(args.channels_image, tiles, spec, cell_px, ncpus=args.ncpus)
    pos = spatial.read_positions(sp)
    counts = spatial.read_counts(sp)
    pos = pos.set_index("barcode").reindex(counts.barcodes)
    ok = pos["in_tissue"].fillna(0).to_numpy(int) == 1
    X = counts.matrix[:, np.flatnonzero(ok)]
    origin = (int(tiles["x0_px"].min()), int(tiles["y0_px"].min()))
    Yc = patchsweep.cell_counts(pos["x_fullres"].to_numpy(float)[ok] * scale, pos["y_fullres"].to_numpy(float)[ok] * scale,
                                X, cr, cc, origin, cell_px)
    log.info("%s: %d cells, %d molecules in them", prefix, len(cr), int(Yc.sum()))
    n_ref = max(2, int(round(args.gene_size_um / args.cell_um)))
    gene_idx = patchsweep.gene_list(Yc, cr, cc, n_ref, args.genes)
    gene_ids = np.asarray(counts.gene_ids, dtype=object)[gene_idx] if len(gene_idx) else None
    gene_names = np.asarray(counts.gene_names, dtype=object)[gene_idx] if len(gene_idx) else None
    summary["genes"] = {"n": int(len(gene_idx)), "chosen_at_um": n_ref * args.cell_um,
                        "rule": "most variable genes of the patches of that size; the same genes at every size",
                        "min_ceiling_for_a_gene_fraction": patchsweep.MIN_GENE_CEILING}
    log.info("%s: %d genes followed one by one (chosen at %g um)", prefix, len(gene_idx), n_ref * args.cell_um)
    rows, rad, gene_rows = [], [], []
    for s in patchsweep.parse_sizes(args.sizes_um, args.cell_um):
        n = int(round(s / args.cell_um))
        row, rr, gr = patchsweep.one_size(Hc, Yc, cr, cc, n, cell_px, mpp, spec, args.max_radius, args.n_pcs, args.n_hvg,
                                          args.folds, args.block_um, args.seed, args.k, args.max_patches, args.bootstrap,
                                          gene_idx, gene_ids, gene_names)
        rows.append(row)
        rad += rr
        gene_rows += gr
        log.info("%s: %g um, %d patches, R2 %.3f, share %.3f", prefix, s, row.get("n_patches", 0),
                 row.get("r2_full", np.nan), row.get("fraction", np.nan))
    table = pd.DataFrame(rows)
    for c in patchsweep.TABLE_COLUMNS:
        if c not in table:
            table[c] = np.nan
    table = table[patchsweep.TABLE_COLUMNS + [c for c in table.columns if c not in patchsweep.TABLE_COLUMNS]]
    radius = pd.DataFrame(rad)
    genes = pd.DataFrame(gene_rows, columns=patchsweep.GENE_COLUMNS)
    svl = patchsweep.small_vs_large(genes, args.small_um, args.large_um)
    spot = patchsweep.sweet_spot(table)
    if len(svl):
        summary["genes"].update({"compared": int(len(svl)), "better_small": int((svl["difference"] > 0).sum()),
                                 "small_um_max": args.small_um, "large_um_min": args.large_um})
    source = lineage.source_text(process, {}, parents=[str(p) for p in (args.lineage_parent or [])],
                                 context=f"{prefix}, {len(tiles)} kept tiles, cells of {args.cell_um:g} um, seed {args.seed}")
    fig = figure(table, radius, spot, args, prefix, source, len(tiles), args.channel_set, svl)
    if fig is not None:
        fig.save(f"{prefix}.patch_sweep.png")
        out["figure"] = f"{prefix}.patch_sweep.png"
    finish("computed", table, radius, spot, genes, svl)


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
