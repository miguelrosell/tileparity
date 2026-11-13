#!/usr/bin/env python3
"""The cohort summary of one scale (with --cluster; --cohort_summary, on by default): every
slide's clustering, redundancy and relevance results gathered into tables, figures and one
HTML page, with comparisons between groups of slides when the samplesheet defines them.
See tileparity_lib/cohort.py."""

import argparse
import base64
import html
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from tileparity_lib import cluster, clusterplots, cohort, figures, lineage  # noqa: E402

log = logging.getLogger("tp_cohort")
GROUP_COLOURS = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167), (230, 159, 0), (86, 180, 233),
                 (117, 77, 160), (160, 30, 80), (90, 90, 90)]


def colours_by(meta: pd.DataFrame, samples: list[str], group: str) -> tuple[dict, list]:
    """{sample: colour} by the group column (one colour if absent), and legend items."""
    if group in meta and meta[group].astype(str).str.len().gt(0).any():
        levels = list(dict.fromkeys(meta[group].astype(str)))
        col = {g: GROUP_COLOURS[i % len(GROUP_COLOURS)] for i, g in enumerate(levels)}
        g_of = dict(zip(meta["sample"], meta[group].astype(str)))
        return {s: col.get(g_of.get(s, ""), (120, 120, 120)) for s in samples}, [(c, f"{group} {g}") for g, c in col.items()]
    return {s: (0, 114, 178) for s in samples}, [((0, 114, 178), "slide")]


def views_order(key: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(key["view"]))
    return [v for v in cluster.VIEWS if v in present] + [v for v in present if v not in cluster.VIEWS]


def by_view_figure(key: pd.DataFrame, metric: str, label: str, colour_of: dict, y_range) -> Image.Image | None:
    if metric not in key or key[metric].isna().all():
        return None
    views = views_order(key)
    points = [[(float(r[metric]), colour_of.get(r["sample"], (90, 90, 90))) for _, r in key[key["view"] == v].iterrows()]
              for v in views]
    return figures.strip_chart(views, points, label, y_range=y_range,
                               cat_colours=[clusterplots.VIEW_COLOURS.get(v, figures.INK) for v in views])


def ari_by_k_figure(long: pd.DataFrame, method: str, ref: str) -> Image.Image | None:
    t = long[(long["analysis"] == "agreement") & (long["metric"] == "ari") & (long["method"] == method) &
             (long["set"] == "keep") & ((long["a"] == ref) | (long["b"] == ref))]
    if t.empty:
        return None
    t = t.assign(view=np.where(t["a"] == ref, t["b"], t["a"]))
    t = t[~t["view"].str.contains(r"\.", regex=True)]
    series = []
    for v, g in t.groupby("view", sort=False):
        s = g.groupby("k")["value"].agg(["mean", "min", "max"]).reset_index().sort_values("k")
        series.append({"name": v, "x": s["k"].to_numpy(float), "y": s["mean"].to_numpy(float),
                       "lo": s["min"].to_numpy(float), "hi": s["max"].to_numpy(float),
                       "colour": clusterplots.VIEW_COLOURS.get(v, figures.SERIES[len(series) % len(figures.SERIES)])})
    ks = sorted(t["k"].unique())
    return figures.line_chart(series, "k (number of clusters)", f"ARI with {ref}: mean (band: lowest to highest slide)",
                              size=(620, 360), y_range=(min(0.0, float(t["value"].min())), 1.0), x_ticks=ks)


def redundancy_figure(long: pd.DataFrame) -> Image.Image | None:
    t = long[(long["analysis"] == "redundancy") & (long["metric"] == "r2")]
    if t.empty:
        return None
    views = [v for v in cluster.VIEWS if v in set(t["a"]) | set(t["b"])]
    med = t.groupby(["a", "b"])["value"].median()
    M = np.full((len(views), len(views)), np.nan)
    for (a, b), v in med.items():
        if a in views and b in views:
            M[views.index(a), views.index(b)] = v
    n = t["sample"].nunique()
    return clusterplots.heatmap(M, views, f"Median over {n} slides: R2 of the column view from the row view", cell=60)


def png_data(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--prefix", required=True, help="cohort_mpp<scale>")
    ap.add_argument("--samples", default=None, help="samples.tsv: sample and its metadata columns")
    ap.add_argument("--group", default="condition", help="metadata column whose groups are compared")
    ap.add_argument("--pair", default="patient", help="metadata column of the unit (sections of one patient averaged)")
    ap.add_argument("--results-link", default="../..", help="path from the report to the results folder")
    lineage.add_cli(ap)
    args = ap.parse_args(argv)
    process = args.process or "TILEPARITY_COHORT"
    d = Path(args.input_dir)
    prefixes = cohort.discover(d)
    if not prefixes:
        raise SystemExit(f"no <prefix>.cluster.json in {d}")
    slides = [cohort.load(d, p) for p in prefixes]
    samples = [cohort.sample_name(s) for s in slides]
    meta = pd.read_csv(args.samples, sep="\t", dtype=str, keep_default_na=False) if args.samples else \
        pd.DataFrame({"sample": samples})
    meta = meta[meta["sample"].isin(samples)].drop_duplicates("sample")
    scale = prefixes[0].rsplit("_mpp", 1)[-1]
    out = args.prefix

    long = cohort.long_table(slides).merge(meta, on="sample", how="left")
    key = cohort.key_table(slides)
    key_m = key.merge(meta, on="sample", how="left")
    files = {"long": f"{out}.long.tsv.gz", "key": f"{out}.key.tsv"}
    long.to_csv(files["long"], sep="\t", index=False, na_rep="NA", float_format="%.6g", compression="gzip")
    key_m.to_csv(files["key"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    comp = cohort.compare_groups(key, meta, args.group, args.pair) if args.group in meta else pd.DataFrame()
    if len(comp):
        files["groups"] = f"{out}.groups.tsv"
        comp.to_csv(files["groups"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    genes = [s["genes"].assign(sample=cohort.sample_name(s)) for s in slides if s.get("genes") is not None]
    if genes:
        g = pd.concat(genes, ignore_index=True)
        r2cols = [c for c in g.columns if c.startswith("r2_") and "null" not in c]
        gm = g.groupby(["gene_id", "gene_name"])[r2cols].agg(["mean", "count"])
        gm.columns = [f"{a}_{b}" for a, b in gm.columns]
        files["genes"] = f"{out}.genes.tsv.gz"
        gm.reset_index().sort_values(gm.columns[0], ascending=False).to_csv(
            files["genes"], sep="\t", index=False, na_rep="NA", float_format="%.6g", compression="gzip")

    colour_of, group_items = colours_by(meta, samples, args.group)
    ref = next((s["info"].get("sets", {}).get("keep", {}).get("reference_view") for s in slides
                if s["info"].get("sets", {}).get("keep", {}).get("reference_view")), "EXPRESSION")
    source = lineage.source_text(process, {}, parents=[str(d / f"{p}.cluster.lineage.json") for p in prefixes
                                                      if (d / f"{p}.cluster.lineage.json").exists()],
                                 context=f"{len(slides)} slides at {scale} um/px")
    n = len(slides)
    figs = {}
    parts = []
    for metric in ("ari_corrected_mean_k_kmeans", "ari_corrected_mean_k_leiden"):
        f = by_view_figure(key, metric, cohort.KEY_METRICS[metric], colour_of, (-0.1, 1.0))
        if f is not None:
            parts.append(figures.stack([figures.title_strip(f.width, cohort.KEY_METRICS[metric], 14), f]))
    if parts:
        body = figures.stack([clusterplots._side_by_side(parts), figures.legend(group_items, width=900)])
        figs["agreement"] = figures.with_caption(
            body, title=f"Agreement of every image view with {ref}, {n} slides ({scale} um/px)",
            what=(f"For every image view (columns) and slide (dots), its agreement with {ref}: the corrected ARI ((ARI "
                  "- spatial null mean) / (1 - null mean)) of their clusterings, averaged over every k, with k-means "
                  "(left) and Leiden (right); black bar: mean over the slides. Colours: the groups of the samplesheet "
                  f"column '{args.group}'."),
            how=("Views whose dots lie high in every slide agree with the expression consistently; a spread of dots "
                 "means the agreement depends on the slide. 0 is the agreement of unrelated maps with the same "
                 "patchiness."),
            caveats=("Averages over k are steadier than single clusterings (see each slide's cluster_stability_keep.png). "
                     "Each dot is a section; sections of the same patient are not independent."),
            source=source, min_width=1100)
    for m in ("kmeans", "leiden"):
        f = ari_by_k_figure(long, m, ref)
        if f is not None:
            parts.append((m, f))
    charts = [figures.stack([figures.title_strip(620, f"{'k-means' if m == 'kmeans' else 'Leiden'}", 14), f])
              for m, f in [p for p in parts if isinstance(p, tuple)]]
    if charts:
        views = [v for v in cluster.VIEWS if v != ref and v in set(key["view"])]
        body = figures.stack([clusterplots._side_by_side(charts),
                              figures.legend([(clusterplots.VIEW_COLOURS.get(v, figures.INK), v) for v in views],
                                             width=1300)])
        figs["ari_by_k"] = figures.with_caption(
            body, title=f"Agreement with {ref} by number of clusters, {n} slides ({scale} um/px)",
            what=(f"ARI of every image view's clusters with those of {ref}, for every k: the line is the mean over the "
                  "slides, the band goes from the lowest to the highest slide."),
            how=("The k where the lines peak is where the views share their regions best with the expression; a "
                 "narrow band means slides agree."),
            caveats="Raw ARI (not corrected for the spatial null). Per slide and k: long.tsv.gz.", source=source,
            min_width=1100)
    f = by_view_figure(key, "macro_f1_mean_k_kmeans", cohort.KEY_METRICS["macro_f1_mean_k_kmeans"], colour_of, (0, 1))
    if f is not None:
        figs["f1"] = figures.with_caption(
            figures.stack([f, figures.legend(group_items, width=f.width)]),
            title=f"Which {ref} domains each view finds (macro-F1), {n} slides",
            what=(f"For every image view and slide, the macro-F1 of its k-means clusters against the {ref} domains (each "
                  "domain matched one-to-one to a cluster; F1 averaged over domains), averaged over k. Black bar: "
                  "mean over slides."),
            how="1: the view finds every expression domain exactly; higher dots: the view finds more of the tissues.",
            caveats="Domains are expression clusters, not annotated tissues. Per domain: each slide's cluster_f1.tsv.",
            source=source)
    f = redundancy_figure(long)
    if f is not None:
        figs["redundancy"] = figures.with_caption(
            figures.stack([f, clusterplots.colour_bar(label="R2")]),
            title=f"How much of each view another view predicts, median over {n} slides",
            what=("Median over the slides of the cross-validated R2 of the column view's principal components predicted "
                  "from the row view's (ridge regression, spatial blocks), the share of the column view's variance "
                  "that the row view predicts."),
            how=("Asymmetric values: the view that predicts the other well but is predicted badly holds information the "
                 "other lacks."),
            caveats="Per slide, with intervals and nulls: long.tsv.gz and each slide's redundancy.tsv.", source=source)
    f = by_view_figure(key, "relevance_r2", cohort.KEY_METRICS["relevance_r2"], colour_of, (-0.1, 1.0))
    if f is not None:
        figs["relevance"] = figures.with_caption(
            figures.stack([f, figures.legend(group_items, width=f.width)]),
            title=f"How much of the expression each image view predicts, {n} slides",
            what=("For every image view and slide, the cross-validated R2 of the expression's principal components "
                  "predicted from the view's (ridge regression, spatial blocks). Black bar: mean over slides."),
            how="Higher: the view predicts more of the expression. Compare the views within each slide.",
            caveats="Linear prediction; what each view adds to another: long.tsv.gz (relevance_partition).",
            source=source)
    if len(comp):
        pl = cohort.patient_level(key, meta, args.group, args.pair)
        metric = "ari_corrected_mean_k_kmeans" if "ari_corrected_mean_k_kmeans" in pl else None
        metric = "relevance_r2" if "relevance_r2" in pl and pl["relevance_r2"].notna().any() else metric
        if metric:
            views = views_order(key)
            groups = list(dict.fromkeys(pl[args.group].astype(str)))
            cats, points, lines = [], [], []
            for v in views:
                for g in groups:
                    t = pl[(pl["view"] == v) & (pl[args.group].astype(str) == g)]
                    cats.append(f"{v} | {g}")
                    points.append([(float(x), GROUP_COLOURS[groups.index(g) % len(GROUP_COLOURS)]) for x in t[metric]])
                if len(groups) == 2:
                    a = pl[(pl["view"] == v) & (pl[args.group].astype(str) == groups[0])].set_index("unit")[metric]
                    b = pl[(pl["view"] == v) & (pl[args.group].astype(str) == groups[1])].set_index("unit")[metric]
                    i0 = len(cats) - 2
                    lines.append([(i0, float(a[u]), i0 + 1, float(b[u])) for u in a.index.intersection(b.index)
                                  if np.isfinite(a[u]) and np.isfinite(b[u])])
            chart = figures.strip_chart(cats, points, cohort.KEY_METRICS[metric], y_range=(-0.1, 1.0), lines=lines,
                                        width=max(700, 60 * len(cats)))
            tests = comp[comp["metric"] == metric]
            text = "; ".join(f"{r.view}: {r.test or 'no test'}" + (f" p = {r.p_value:.3g}" if np.isfinite(r.p_value) else "")
                             for r in tests.itertuples(index=False))
            figs["groups"] = figures.with_caption(
                figures.stack([chart, figures.legend([(GROUP_COLOURS[i % len(GROUP_COLOURS)], f"{args.group} {g}")
                                                      for i, g in enumerate(groups)], width=chart.width)]),
                title=f"{cohort.KEY_METRICS[metric]} by '{args.group}', unit: {args.pair}",
                what=(f"For every image view, {cohort.KEY_METRICS[metric]} in each group of the samplesheet column "
                      f"'{args.group}'; dots are {args.pair}s (sections of one {args.pair} and group averaged), grey "
                      f"lines join the two groups of the same {args.pair}. Tests: {text}."),
                how=(f"Lines that rise or fall consistently: the measure differs between the groups within the same "
                     f"{args.pair}. The test treats each {args.pair} as one observation."),
                caveats=("Few patients: read the tests as descriptive, not as confirmed differences. Every metric and "
                         "view: groups.tsv."),
                source=source, min_width=1000)
    sweeps = []
    for f in sorted(d.glob("*.patch_sweep.tsv")):
        t = pd.read_csv(f, sep="\t")
        if len(t) and t["fraction"].notna().any():
            sweeps.append(t.assign(sample=f.name[: -len(".patch_sweep.tsv")].rsplit("_mpp", 1)[0]))
    if sweeps:
        from tileparity_lib import patchsweep

        sw = pd.concat(sweeps, ignore_index=True).merge(meta, on="sample", how="left")
        files["patch_sweep"] = f"{out}.patch_sweep.tsv"
        sw.to_csv(files["patch_sweep"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
        mean = sw.groupby("size_um").agg(fraction=("fraction", "mean"), fraction_sd=("fraction", "std"),
                                         n=("fraction", "count")).reset_index()
        mean["fraction_se"] = mean["fraction_sd"] / np.sqrt(mean["n"].clip(lower=1))
        spot = patchsweep.sweet_spot(mean) if len(sweeps) > 1 else {}
        series = []
        for i, (smp, g) in enumerate(sw.groupby("sample", sort=False)):
            g = g.dropna(subset=["fraction"]).sort_values("size_um")
            series.append({"name": smp, "x": g["size_um"].to_numpy(float), "y": g["fraction"].to_numpy(float),
                           "colour": colour_of.get(smp, GROUP_COLOURS[i % len(GROUP_COLOURS)])})
        series.append({"name": "mean of the slides", "x": mean["size_um"].to_numpy(float),
                       "y": mean["fraction"].to_numpy(float), "colour": (0, 0, 0)})
        xs = sorted(sw["size_um"].unique())
        chart = figures.line_chart(series, "patch side (um)", "share of the reachable (R2 from LBP / ceiling)",
                                   size=(900, 420), y_range=(0.0, 1.0),
                                   x_ticks=[x for x in (16, 32, 48, 64, 96, 112, 128, 160, 192, 224) if xs[0] <= x <= xs[-1]],
                                   v_lines=[(spot["sweet_spot_um"], f"sweet spot {spot['sweet_spot_um']:g} um")] if spot else None)
        figs["patch_sweep"] = figures.with_caption(
            figures.stack([chart, figures.legend([(x["colour"], x["name"]) for x in series], width=900)]),
            title=f"Patch size of fastLBP across {len(sweeps)} Visium HD slides",
            what=("For every Visium HD slide (lines) and their mean (black), the share of the reachable: R2 of the "
                  "expression (half of every patch's molecules) predicted from LBP, divided by the R2 reached from the "
                  "other half of the molecules, for patches of every size (LBP counted per 8 um cell and summed). "
                  + (f"Sweet spot of the mean: {spot['sweet_spot_um']:g} um (the smallest size within one standard "
                     "error, over the slides, of the largest mean)." if spot else "")),
            how=("Where the curves peak, texture says most about the expression relative to what the molecules allow; a "
                 "common peak across slides is a size to use for LBP."),
            caveats="Few slides; see each slide's patch_sweep.png for the intervals and the radii.", source=source)
    for name, img in figs.items():
        path = f"{out}.{name}.png"
        img.save(path)
        files[f"figure_{name}"] = path

    # one page with everything, the figures embedded, links to every slide's figures
    rows = []
    for p, s in zip(prefixes, samples):
        links = " ".join(f'<a href="{args.results_link}/{s}/mpp{scale}/cluster/{p}.{n_}.png">{n_.replace("_keep", "")}</a>'
                         for n_ in ("cluster_agreement_keep", "cluster_maps_keep", "cluster_f1_keep",
                                    "cluster_stability_keep", "redundancy_keep", "relevance_keep", "relevance_genes_keep"))
        gal = f'<a href="{args.results_link}/{s}/mpp{scale}/gallery/">galleries</a>'
        m = meta[meta["sample"] == s]
        md = ", ".join(f"{c}: {html.escape(str(m[c].iloc[0]))}" for c in m.columns if c != "sample" and len(m) and m[c].iloc[0])
        rows.append(f"<tr><td>{html.escape(s)}</td><td>{md}</td><td>{links} {gal}</td></tr>")
    imgs = "".join(f'<h2>{html.escape(n_)}</h2><img alt="{n_}" src="data:image/png;base64,{png_data(im)}">'
                   for n_, im in figs.items())
    keytab = key_m.round(3).to_html(index=False, na_rep="", border=0)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>tileparity cohort {scale} um/px</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1400px}}img{{max-width:100%;border:1px solid #ddd}}
table{{border-collapse:collapse;font-size:13px}}td,th{{padding:4px 8px;border-bottom:1px solid #ddd;text-align:left}}</style>
</head><body><h1>tileparity cohort summary, {n} slides at {scale} um/px</h1>
<p>Every figure has its explanation underneath. Tables: {', '.join(Path(v).name for k_, v in files.items() if not k_.startswith('figure'))}.</p>
<h2>Slides</h2><table><tr><th>slide</th><th>metadata</th><th>figures</th></tr>{''.join(rows)}</table>
{imgs}<h2>Key numbers per slide and view</h2>{keytab}</body></html>"""
    files["report"] = f"{out}.report.html"
    Path(files["report"]).write_text(page)
    lineage.write(f"{out}.cohort.lineage.json", lineage.build(
        "cohort", process, out, parameters={"group": args.group, "pair": args.pair, "slides": samples},
        inputs={f"{p}.cluster.json": d / f"{p}.cluster.json" for p in prefixes}, outputs=files,
        parents=[d / f"{p}.cluster.lineage.json" for p in prefixes if (d / f"{p}.cluster.lineage.json").exists()],
        packages=["pandas", "scipy", "pillow"]))
    log.info("%s: %d slides, %d figures", out, n, len(figs))


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
