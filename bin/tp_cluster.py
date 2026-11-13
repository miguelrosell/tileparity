#!/usr/bin/env python3
"""Optional (--cluster): every feature view of one slide clustered on its own (Leiden and
k-means at the same numbers of clusters), and how much the views agree (ARI, NMI and a
spatial null). See tileparity_lib/cluster.py for the method. With --lbp-recipes, the LBP
views are also prepared in two other ways and compared; with --redundancy, how much of each
image view the others predict (tileparity_lib/redundancy.py)."""

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import cluster, clusterplots, export, lineage, redundancy, relevance  # noqa: E402

log = logging.getLogger("tp_cluster")


def names(text: str, allowed, what: str) -> list[str]:
    items = [x.strip() for x in str(text).split(",") if x.strip()]
    if items == ["all"]:
        return list(allowed)
    bad = [x for x in items if x not in allowed]
    if bad or not items:
        raise SystemExit(f"--{what}: unknown {', '.join(bad) or 'empty list'}; choose from {', '.join(allowed)}")
    return [x for x in allowed if x in items]


def integers(text: str) -> list[int]:
    try:
        out = sorted({int(x) for x in str(text).split(",") if x.strip()})
    except ValueError:
        raise SystemExit(f"--k: not a list of whole numbers: {text}") from None
    if not out or min(out) < 2:
        raise SystemExit(f"--k: every number of clusters must be at least 2: {text}")
    return out


def views_text(info: dict) -> str:
    """What every clustered view of a tile set is, for the captions."""
    done = [v for v in cluster.VIEWS if info.get("views", {}).get(v, {}).get("status") == "clustered"]
    return "Views: " + "; ".join(f"{v} = {cluster.VIEW_DESCRIPTIONS[v]}" for v in done) + "." if done else ""


def captions(info: dict, set_name: str, args, ref: str, source: str, mask_mpp) -> dict:
    """The four-part explanations of the figures of one tile set."""
    n = info["n_tiles"]
    ks = ", ".join(map(str, info["k"]))
    k_show = info.get("k_show") or {m: info["k_ref"] for m in args.methods}
    shown = "; ".join(f"{'Leiden' if m == 'leiden' else 'k-means'} k = {k}" for m, k in k_show.items())
    auto = str(info.get("k_show_rule", "")).startswith("auto")
    rule = (f"The k shown ({shown}) is chosen per method as the one at which the image views agree best with {ref}: "
            "the highest mean corrected ARI (see below) of the image views with " + ref + ", the smaller k on ties."
            if auto else f"The k shown ({shown}) was fixed with --cluster_k_ref.")
    null = info["null"]
    if null["moves"]:
        null_how = (f"The null moves one map over the other on the tile grid with wrap-around ({null['moves']} "
                    "moves of at least 20 % of the grid, with at least 30 % of the tiles still over tiles): both maps "
                    "keep their patchiness but lose their alignment.")
    else:
        null_how = "No spatial null here (" + null.get("note", "not computed").replace("not computed: ", "") + ")."
    both = len(args.methods) == 2
    methods = "Leiden and k-means" if both else ("Leiden" if args.methods == ["leiden"] else "k-means")
    top = "Top row: adjusted Rand index (ARI) of every pair of views at the k shown"
    if null["moves"]:
        top += ("; the small number under each value is the 95th percentile of its spatial null ('<= null' when the "
                "ARI does not exceed it)")
    if both:
        top += ", and the grey diagonal is the ARI between the Leiden and the k-means clusterings of the same view"
    middle = f"Middle row: the ARI averaged over every k ({ks})"
    if null["moves"]:
        middle += (", with the corrected ARI averaged the same way underneath (corr.: (ARI - null mean) / (1 - null "
                   "mean), the share of the way from the agreement of two unrelated maps with the same patchiness, 0, "
                   "to identical partitions, 1)")
    bottom = f"Bottom: ARI of every view with {ref} for every k, the k shown marked"
    if null["moves"]:
        bottom += "; the shaded band is the 5-95 % range of the spatial null"
    how = ("ARI is 1 when two clusterings group the tiles identically and about 0 when they agree no more than "
           "random labels would; values between 0.2 and 0.5 mean that the views share some of their regions.")
    if null["moves"]:
        how += (" An ARI above its null (the band, or the small number) is more agreement than two unrelated maps "
                "with the same patchiness would show.")
    if both:
        how += (" If Leiden and k-means give the same picture, the conclusion does not depend on the method; a low "
                "grey diagonal means that view has no clear clusters at that k.")
    agreement = {
        "what": (f"How much the clusterings of the feature views of this slide agree, on the {n} tiles of the set "
                 f"'{set_name}': {info['definition']}. Every view was clustered on its own: PCA (at most "
                 f"{args.n_pcs} components), then Leiden on the graph of the {args.n_neighbors} nearest neighbours, "
                 "with the resolution searched until the partition has exactly k clusters, and k-means with the same "
                 f"k ({methods} here). {top}. {middle}. {bottom}. {rule} {views_text(info)}"),
        "how": how,
        "caveats": ("The clustering ignores where the tiles are (no spatial smoothing) and uses one random seed; a "
                    "single clustering can change with a small change of the input (cluster_stability_keep.png), so "
                    "the averages over k are the steadier numbers. Choosing the k with the best agreement favours "
                    "agreement: compare views with the averages. ARI depends on k and on the cluster sizes, so views "
                    "are only compared at the same k: points where Leiden could not reach exactly k are left out. "
                    + null_how + " Every pair, every k, NMI, the null and the corrected ARI are in "
                    "cluster_agreement.tsv; averages and the values at the k shown in cluster_summary.tsv."),
        "source": source,
    }
    name = "Leiden" if "leiden" in args.methods else "k-means"
    k_ref = k_show.get("leiden" if "leiden" in args.methods else "kmeans")
    thumb = f"{mask_mpp:g} um/px thumbnail" if mask_mpp else "slide thumbnail"
    maps = {
        "what": (f"The {n} tiles of the set '{set_name}' coloured by their cluster ({name}, k = {k_ref}) in every "
                 f"view, each view clustered on its own, over the faded {thumb}. Colours follow the clusters of {ref}: "
                 f"the clusters of every other view take the colour of the {ref} cluster they share most tiles with "
                 "(one-to-one matching), so the same colour in two maps marks matched clusters. " + views_text(info)),
        "how": ("Where two maps show the same colours in the same places, the two views divide the tissue in the same "
                "way. A region that only one view separates is something that view sees and the others do not. A "
                "salt-and-pepper map without coherent regions groups tiles by something that changes from tile to "
                "tile (focus, stain intensity, cutting artefacts) rather than by tissue regions."),
        "caveats": (f"Matching is one-to-one: when a view splits one {ref} cluster in two, one half keeps the colour "
                    "and the other takes the colour of another cluster. Cluster numbers are per view and per slide "
                    "(cluster 3 of one view has nothing to do with cluster 3 of another view, or of another slide). "
                    "Tiles outside the set are not drawn."),
        "source": source,
    }
    contingency = {
        "what": (f"For every view, where the tiles of each of its clusters ({name}, k = {k_ref}; rows, with their "
                 f"number of tiles) fall among the clusters of {ref} (columns): each cell is the percentage of the "
                 f"row's tiles in that {ref} cluster. Rows are ordered by the {ref} cluster they are matched to, so "
                 f"matched pairs lie on the diagonal; the colour squares are those of cluster_maps_{set_name}.png."),
        "how": (f"A dark diagonal with little elsewhere: the view reproduces the clusters of {ref}. A row spread over "
                f"several columns: a cluster that mixes several {ref} clusters. A column covered by several rows: a "
                f"{ref} cluster that the view splits."),
        "caveats": ("Percentages are per row, so a small cluster counts as much as a large one (see the tile "
                    "counts). Cluster numbers are per view and per slide."),
        "source": source,
    }
    out = {"agreement": agreement, "maps": maps, "contingency": contingency}
    out["f1"] = {
        "what": (f"Which clusters ('domains') of {ref} each view finds, on the {n} tiles of the set '{set_name}'. Every "
                 f"{ref} domain is matched one-to-one to the cluster of the view (same method and k) that maximises the "
                 "summed F1 (Hungarian algorithm); precision = shared tiles / cluster size, recall = shared tiles / "
                 "domain size, F1 = 2 x shared / (domain size + cluster size), their harmonic mean. Top: F1 of every "
                 f"domain (rows, with their number of tiles) for every view at the k shown. Bottom: macro-F1, the mean "
                 f"F1 over the domains, for every k. {views_text(info)}"),
        "how": ("1: the view has a cluster that is exactly that domain; 0: no cluster overlaps it. Unlike the ARI, "
                "which the large regions dominate, F1 tells which tissues each view finds, and macro-F1 weighs every "
                "domain the same."),
        "caveats": ("Domains are clusters of expression, not annotated tissues: name them from the maps, the H&E and "
                    "their genes. Small domains give unsteady F1. Every domain, view, method and k: cluster_f1.tsv."),
        "source": source,
    }
    out["stability"] = {
        "what": (f"How much the clusterings change when the input changes a little, set '{set_name}': {args.stability_reps} "
                 f"times, a random {args.stability_fraction:.0%} of the tiles were clustered again (same principal "
                 "components, a new nearest-neighbour graph, another seed), with every method and k. Left: ARI of every "
                 f"view with {ref} clustered on the same subsample. Right: ARI of every view's subsample clustering with "
                 "its own clustering of all tiles. Lines: mean over the subsamples; band: mean +- standard deviation. "
                 + views_text(info)),
        "how": (f"A narrow band: the agreement with {ref} at that k does not depend on which tiles happen to be "
                "clustered. A wide band, or a low right-hand value: that clustering is fragile, and a single value of "
                "it (in the heatmaps above) should not be read alone."),
        "caveats": ("Subsampling mimics small changes of the input (tile borders, a slightly different section); it "
                    "does not test other slides. Per subsample: cluster_stability.tsv."),
        "source": source,
    }
    if info.get("recipe_views_clustered"):
        recipes = info.get("lbp_recipes", {})
        listed = "; ".join(f"{r} = {d}" + (f" ({args.lbp_hvf:.0%} of the bins kept)" if r == "scanpy" else "")
                           for r, d in recipes.items())
        out["recipes"] = {
            "what": (f"For every LBP view of the set '{set_name}' ({n} tiles), the ARI of its clusters with {ref} for "
                     f"k = {ks}, with the LBP histograms prepared in {len(recipes)} ways before the same PCA and "
                     f"clustering ({methods}; one row of panels per method). Preparations: {listed}. "
                     f"{views_text(info)}"),
            "how": ("Lines that lie together: what LBP shows does not depend on how its histograms are prepared. A "
                    "preparation whose line lies clearly above the others at most k and with both methods keeps more "
                    f"of what {ref} sees."),
            "caveats": ("Choosing a preparation by its agreement with expression on these slides tunes LBP to the "
                        "reference: confirm on other slides before adopting one. Each alternative has choices of its "
                        "own (log1p, the share of bins kept, the clip at 10). Every pair, k and null are in "
                        "cluster_agreement.tsv (views named <view>.<recipe>)."),
            "source": source,
        }
    return out


def redundancy_captions(red: dict, args, source: str) -> dict:
    info = red["info"]
    block = f"{info['block']:g} {'um' if info['block_unit'] == 'um' else 'tiles'}"
    views = "Views: " + "; ".join(f"{v} = {cluster.view_description(v)}" for v in info["views"]) + "."
    matrix = {
        "what": (f"For the {info['n_tiles']} tiles of the set 'keep', how much of each view (columns) another view "
                 f"(rows) predicts: the predicted view's principal components (those of the clustering, at most "
                 f"{args.n_pcs}) are predicted from the other view's with ridge regression, every tile by a model "
                 f"fitted on other parts of the slide (spatial cross-validation: {info['n_folds']} folds of square "
                 f"blocks of {block}). R2 = the share of the predicted view's variance that the other view predicts; "
                 "under each value its 95 % interval (block bootstrap: the spatial blocks resampled with replacement). "
                 "Right: R2 of each component of the predicted view, for UNI2-h and LBP on the same pixels (values "
                 "below -0.25, components that the other view predicts worse than their mean, are drawn at -0.25). "
                 + views),
        "how": ("Rows are 'from', columns 'predicted'. High R2 both ways: the two views carry largely the same "
                "information. Asymmetric values: the view that predicts the other well but is predicted badly holds "
                "information the other lacks (its own part is 1 - R2). In the curves, high R2 on the first "
                "components and low on later ones: the views share their main structure and differ in finer detail."),
        "caveats": ("R2 refers to the variance the components keep (target_pc_variance in redundancy.tsv) and to "
                    "linear prediction: a view may be predictable in a non-linear way. The spatial blocks limit, but "
                    "do not remove, the help of spatial autocorrelation; the spatial null of every pair (the predicting "
                    "map moved over the tile grid, so that it keeps its structure but no longer describes the same "
                    "tissue: r2_null_mean, r2_null_p95, p_null in redundancy.tsv) is the R2 reached by chance. No "
                    "classical p-value is given: neighbouring tiles are not independent. What one view has and the "
                    "other lacks is not necessarily biology (noise, staining, focus): the maps show where it is. Every "
                    "pair: redundancy.tsv; every component: redundancy_pcs.tsv.gz."),
        "source": source,
    }
    maps = {
        "what": ("For UNI2-h and LBP on the same pixels, both ways: the part of the predicted view that the other view "
                 "does not predict, per tile (squared prediction error relative to the average variation of the "
                 "predicted view, so that its mean over the tiles is 1 - R2), over the faded thumbnail. " + views),
        "how": ("Dark tiles hold what one view has and the other cannot predict. Coherent dark regions: tissue that "
                "only one view describes. Scattered dark tiles or dark edges: likely artefacts (blur, folds, stain, "
                "tissue borders), worth checking on the H&E."),
        "caveats": ("Out-of-sample predictions with spatial blocks; values above 2 (twice the average) take the "
                    "darkest colour. Per-tile values: redundancy_tiles.tsv.gz."),
        "source": source,
    }
    return {"matrix": matrix, "maps": maps}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--prefix", default=None, help="<sample>_mpp<scale> (default: the only sample in --input-dir)")
    ap.add_argument("--views", default="all", help=f"comma list of {', '.join(cluster.VIEWS)}, or all")
    ap.add_argument("--sets", default="keep,extended")
    ap.add_argument("--methods", default="leiden,kmeans")
    ap.add_argument("--k", default=",".join(map(str, cluster.DEFAULT_K)), help="numbers of clusters")
    ap.add_argument("--k-ref", default="auto",
                    help="k of the heatmaps, maps and contingency tables: auto (the k at which the image views agree "
                         "best with EXPRESSION, per method) or a number")
    ap.add_argument("--stability-reps", type=int, default=5, help="subsamples of the stability check (0: none)")
    ap.add_argument("--stability-fraction", type=float, default=0.9, help="share of the tiles in every subsample")
    ap.add_argument("--representatives", type=int, default=8,
                    help="tiles per cluster and per component extreme listed for the galleries (0: none)")
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--min-counts-ratio", type=float, default=0.1,
                    help="set 'extended': minimum molecules per spatial unit, as a fraction of the keep tiles' median")
    ap.add_argument("--null-shifts", type=int, default=100, help="moves of the spatial null (0: no null)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lbp-recipes", default="hellinger",
                    help=f"preparations of the LBP views, comma list of {', '.join(cluster.LBP_RECIPES)}; hellinger is "
                         "always used for the views themselves, the others give views <view>.<recipe>")
    ap.add_argument("--lbp-hvf", type=float, default=0.3, help="recipe scanpy: share of the LBP bins kept")
    ap.add_argument("--redundancy", action="store_true", help="how much of each image view the others predict")
    ap.add_argument("--redundancy-views", default="images",
                    help="views of the redundancy analysis: images (every clustered view except EXPRESSION), or a "
                         "comma list of views")
    ap.add_argument("--redundancy-folds", type=int, default=5)
    ap.add_argument("--redundancy-block-um", type=float, default=1000.0, help="side of the spatial blocks (um)")
    ap.add_argument("--redundancy-null-shifts", type=int, default=20,
                    help="moves of the spatial null of R2 (redundancy and relevance; 0: none)")
    ap.add_argument("--redundancy-bootstrap", type=int, default=200,
                    help="block-bootstrap resamples for the 95 % interval of R2 (0: none)")
    ap.add_argument("--relevance", action="store_true",
                    help="how much of the expression (components and genes) each image view predicts")
    ap.add_argument("--relevance-gene-views", default=",".join(relevance.DEFAULT_GENE_VIEWS),
                    help="image views that predict every gene (comma list; 'none' for no gene-level analysis)")
    ap.add_argument("--relevance-gene-null-shifts", type=int, default=20,
                    help="moves of the per-gene spatial null (0: none)")
    lineage.add_cli(ap)
    args = ap.parse_args(argv)
    process = args.process or "TILEPARITY_CLUSTER"
    args.views = names(args.views, cluster.VIEWS, "views")
    args.sets = names(args.sets, cluster.SETS, "sets")
    args.methods = names(args.methods, cluster.METHODS, "methods")
    args.k = integers(args.k)
    if str(args.k_ref).lower() != "auto":
        try:
            args.k_ref = int(args.k_ref)
        except ValueError:
            raise SystemExit(f"--k-ref: 'auto' or a whole number, not {args.k_ref}") from None
    else:
        args.k_ref = "auto"
    if args.stability_reps < 0 or not 0 < args.stability_fraction < 1:
        raise SystemExit("--stability-reps must be >= 0 and --stability-fraction between 0 and 1")
    args.lbp_recipes = names(args.lbp_recipes, cluster.LBP_RECIPES, "lbp-recipes")
    if not 0 < args.lbp_hvf <= 1:
        raise SystemExit(f"--lbp-hvf: a share between 0 and 1, not {args.lbp_hvf}")
    if args.redundancy_views != "images":
        args.redundancy_views = names(args.redundancy_views, cluster.VIEWS, "redundancy-views")
    args.relevance_gene_views = ([] if str(args.relevance_gene_views).lower() == "none" else
                                 names(args.relevance_gene_views, cluster.VIEWS, "relevance-gene-views"))
    if args.redundancy_folds < 2 or args.redundancy_block_um <= 0:
        raise SystemExit("--redundancy-folds must be at least 2 and --redundancy-block-um positive")

    prefixes = export.discover(args.input_dir)
    if args.prefix is None:
        if len(prefixes) != 1:
            raise SystemExit(f"--prefix is needed: {len(prefixes)} samples in {args.input_dir}")
        args.prefix = prefixes[0]
    elif args.prefix not in prefixes:
        raise SystemExit(f"no {args.prefix}.tiles.tsv.gz in {args.input_dir}")
    prefix = args.prefix
    sample = export.load_sample(args.input_dir, prefix)
    result = cluster.run(sample, views=args.views, ks=args.k, k_ref=args.k_ref, methods=args.methods, sets=args.sets,
                         n_pcs=args.n_pcs, n_neighbors=args.n_neighbors, n_hvg=args.n_hvg,
                         min_counts_ratio=args.min_counts_ratio, null_shifts=args.null_shifts, seed=args.seed,
                         lbp_recipes=args.lbp_recipes, lbp_hvf=args.lbp_hvf, log=log)

    outputs = {"labels": f"{prefix}.clusters.tsv.gz", "agreement": f"{prefix}.cluster_agreement.tsv",
               "summary": f"{prefix}.cluster.json"}
    with gzip.open(outputs["labels"], "wt") as fh:
        cluster.labels_table(sample, result, args.views).to_csv(fh, sep="\t", index=False, na_rep="")
    result["agreement"].to_csv(outputs["agreement"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    outputs["cluster_summary"] = f"{prefix}.cluster_summary.tsv"
    cluster.summary_table(result["agreement"], result["sets"]).to_csv(outputs["cluster_summary"], sep="\t", index=False,
                                                                    na_rep="NA", float_format="%.6g")
    f1 = cluster.f1_table(result)
    outputs["cluster_f1"] = f"{prefix}.cluster_f1.tsv"
    f1.to_csv(outputs["cluster_f1"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    stab = cluster.stability(result, "keep", args.methods, reps=args.stability_reps, fraction=args.stability_fraction,
                             n_neighbors=args.n_neighbors, seed=args.seed, log=log)
    if len(stab):
        outputs["cluster_stability"] = f"{prefix}.cluster_stability.tsv"
        stab.to_csv(outputs["cluster_stability"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    reps = cluster.representatives(sample, result, "keep", n=args.representatives)
    if len(reps):
        outputs["cluster_representatives"] = f"{prefix}.cluster_representatives.tsv"
        reps.to_csv(outputs["cluster_representatives"], sep="\t", index=False, na_rep="NA", float_format="%.6g")

    parents = [str(p) for p in sample["lineage_parents"]]
    prep = sample["provenance"].get("prepare", {})
    mask_mpp = (prep.get("thumbnail") or {}).get("mpp")
    for s in args.sets:
        info = result["sets"][s]
        if info.get("status") != "clustered":
            continue
        ref = info["reference_view"]
        source = lineage.source_text(process, {}, parents=parents,
                                     context=f"{prefix}, set '{s}' ({info['n_tiles']} tiles), seed {args.seed}")
        text = captions(info, s, args, ref, source, mask_mpp)
        made = {
            f"agreement_{s}": clusterplots.agreement_figure(result["agreement"], info, s, args.methods,
                                                            text["agreement"], prefix),
            f"maps_{s}": clusterplots.maps_figure(sample, result, s, text["maps"], prefix),
        }
        if s == "keep":
            made["contingency_keep"] = clusterplots.contingency_figure(result, s, text["contingency"], prefix)
            made["stability_keep"] = clusterplots.stability_figure(cluster.stability_summary(stab), info, s,
                                                                   args.methods, text["stability"], prefix)
        made[f"f1_{s}"] = clusterplots.f1_figure(f1, info, s, args.methods, text["f1"], prefix)
        if "recipes" in text:
            made[f"lbp_recipes_{s}"] = clusterplots.recipes_figure(result["agreement"], info, s, args.methods,
                                                                   text["recipes"], prefix)
        for name, img in made.items():
            if img is not None:
                path = f"{prefix}.cluster_{name}.png"
                img.save(path)
                outputs[f"figure_{name}"] = path

    red_info = None
    if args.redundancy:
        red_info = run_redundancy(sample, result, args, prefix, parents, process, outputs)

    rel_info = None
    if args.relevance:
        rel_info = run_relevance(sample, result, args, prefix, parents, process, outputs)

    summary = {
        "prefix": prefix,
        "sample_id": str(sample["tiles"]["sample_id"].iloc[0]) if len(sample["tiles"]) else prep.get("sample_id"),
        "scale_mpp": prep.get("mpp_target"),
        "parameters": {k: v for k, v in vars(args).items() if k not in ("input_dir", "lineage_parent", "process")},
        "sets": result["sets"],
        "view_descriptions": {v: cluster.view_description(v)
                              for v in dict.fromkeys(v for i in result["sets"].values() for v in (i.get("views") or {}))},
        "outputs": {k: Path(v).name for k, v in outputs.items() if k != "summary"},
        "notes": ["cluster numbers are per view and per slide; 0 is the largest cluster",
                  "columns of the labels table: <set>_<view>_<method>_k<k>; empty outside the set",
                  "agreement rows with method leiden_vs_kmeans compare the two methods on the same view",
                  "views <view>.<recipe>: an LBP view prepared with an alternative recipe (--lbp-recipes)",
                  "ari_corrected = (ari - null_ari_mean) / (1 - null_ari_mean); k_show: the k of the figures, per "
                  "method (sets.<set>.k_show, k_choice and k_show_rule)"],
    }
    if red_info is not None:
        summary["redundancy"] = red_info
    if rel_info is not None:
        summary["relevance"] = rel_info
    Path(outputs["summary"]).write_text(json.dumps(lineage._jsonable(summary), indent=2))
    inputs = {role: Path(args.input_dir) / f"{prefix}.{suffix}" for role, suffix in
              (("tiles", "tiles.tsv.gz"), ("uni2h", "uni2h.h5"), ("lbp", "lbp.h5"), ("counts", "counts.h5"))}
    lineage.write(f"{prefix}.cluster.lineage.json", lineage.build(
        "cluster", process, prefix, parameters=summary["parameters"],
        inputs={k: v for k, v in inputs.items() if v.exists()}, outputs=outputs, parents=parents,
        packages=["scikit-learn", "igraph", "umap-learn", "numpy", "scipy", "pandas"]))
    for s, info in result["sets"].items():
        log.info("%s, set %s: %s", prefix, s, info.get("status"))


def run_redundancy(sample: dict, result: dict, args, prefix: str, parents, process: str, outputs: dict) -> dict:
    """The redundancy analysis of the set keep: tables, figures and what it did."""
    s = "keep"
    info = result["sets"].get(s, {})
    if info.get("status") != "clustered":
        return {"status": f"not computed: the set '{s}' was not clustered"}
    clustered = info["views_clustered"]
    views = ([v for v in clustered if v != "EXPRESSION"] if args.redundancy_views == "images"
             else [v for v in args.redundancy_views if v in clustered])
    tiles = sample["tiles"][result["masks"][s]]
    mpp = (sample["provenance"].get("prepare") or {}).get("mpp_target")
    red = redundancy.run(result["scores"][s], result["pc_variance"][s], tiles, views, s, mpp,
                         n_folds=args.redundancy_folds, block_um=args.redundancy_block_um, seed=args.seed,
                         null_shifts=args.redundancy_null_shifts, n_boot=args.redundancy_bootstrap)
    if red["info"]["status"] != "computed":
        log.info("redundancy: %s", red["info"]["status"])
        return red["info"]
    files = {"redundancy": f"{prefix}.redundancy.tsv", "redundancy_pcs": f"{prefix}.redundancy_pcs.tsv.gz",
             "redundancy_tiles": f"{prefix}.redundancy_tiles.tsv.gz"}
    red["table"].to_csv(files["redundancy"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    with gzip.open(files["redundancy_pcs"], "wt") as fh:
        red["pcs"].to_csv(fh, sep="\t", index=False, na_rep="NA", float_format="%.6g")
    with gzip.open(files["redundancy_tiles"], "wt") as fh:
        red["tiles"].to_csv(fh, sep="\t", index=False, na_rep="NA", float_format="%.6g")
    outputs.update(files)
    source = lineage.source_text(process, {}, parents=parents,
                                 context=f"{prefix}, set '{s}' ({len(tiles)} tiles), seed {args.seed}")
    text = redundancy_captions(red, args, source)
    for name, img in (("redundancy_keep", clusterplots.redundancy_figure(red, text["matrix"], prefix, s)),
                      ("redundancy_maps_keep", clusterplots.redundancy_maps_figure(sample, tiles, red, text["maps"],
                                                                                   prefix))):
        if img is not None:
            path = f"{prefix}.{name}.png"
            img.save(path)
            outputs[f"figure_{name}"] = path
    log.info("redundancy: %d pairs of %d views", len(red["table"]), len(views))
    return red["info"]


def relevance_captions(rel: dict, args, source: str) -> dict:
    info = rel["info"]
    block = f"{info['block']:g} {'um' if info['block_unit'] == 'um' else 'tiles'}"
    views = "Views: " + "; ".join(f"{v} = {cluster.view_description(v)}" for v in info["views"]) + "."
    model = (f"ridge regression with spatial cross-validation ({info['n_folds']} folds of square blocks of {block}): "
             "every tile is predicted by a model fitted on other parts of the slide")
    main = {
        "what": (f"How much of the expression of the {info['n_tiles']} tiles of the set 'keep' each image view predicts. "
                 f"a: the expression's principal components (those of the clustering, at most {args.n_pcs}) predicted "
                 f"from each view's components, {model}; R2 = the share of the expression's variance that the view "
                 "predicts; whiskers: 95 % interval (block bootstrap); black tick: 95th percentile of the spatial null "
                 "(the view's map moved over the tile grid, so that it no longer describes the same tissue). b: for "
                 "pairs of views A and B, the R2 of A and B together split into what only A predicts (R2(A+B) - "
                 "R2(B)), what only B predicts (R2(A+B) - R2(A)) and what both predict (the rest). " + views),
        "how": ("A bar well beyond its black tick: the view predicts expression better than an unrelated map with the "
                "same structure would. In b, a long 'only A' part: A sees expression that B misses. With LBP + "
                "INTENSITY_HE, 'only LBP' is what texture adds to the mere amount of stain."),
        "caveats": ("Linear prediction between principal components; R2 refers to the variance the expression's "
                    "components keep. Out-of-sample R2 can be slightly negative, and so can a 'unique' part. "
                    "Neighbouring tiles are not independent, so no classical p-value is given. Every predictor: "
                    "relevance.tsv; the split: relevance_partition.tsv."),
        "source": source,
    }
    genes = {
        "what": (f"Gene by gene: each of the expression's most variable genes (counts per 10 000, log1p, z-scores, as in "
                 f"the clustering) predicted from the components of an image view ({model}; a penalty per gene). a: "
                 "every gene's R2 from two views (diagonal: equal R2); coloured when the gene beats the per-gene "
                 "spatial null (z = (R2 - null mean) / null sd, one-sided normal p-value, Benjamini-Hochberg q < 0.05), "
                 "labelled: the genes best predicted from the second view and those it adds most to the first. b: the "
                 "genes best predicted from the second view, with the first view's R2 (black tick). " + views),
        "how": ("Genes far above the diagonal are predicted from texture (LBP) better than from UNI2-h; genes on the "
                "diagonal are seen equally by both. Genes of abundant, visually distinct cell types (stroma, muscle, "
                "epithelium) are expected among the best predicted; a gene predicted from LBP beyond INTENSITY_HE "
                "(column adds_LBP_RGB_to_INTENSITY_HE) relates to texture rather than to the amount of stain."),
        "caveats": ("A gene predicted from the image is associated with what the tissue looks like there, not "
                    "necessarily expressed by the cells that create the texture. The null has few moves, so the "
                    "p-values are approximate and assume a normal null; genes are correlated with each other. Use the "
                    "ranking, and confirm a gene on other slides. Every gene: relevance_genes.tsv.gz."),
        "source": source,
    }
    return {"main": main, "genes": genes}


def run_relevance(sample: dict, result: dict, args, prefix: str, parents, process: str, outputs: dict) -> dict:
    """The relevance analysis of the set keep: tables, figures and what it did."""
    s = "keep"
    info = result["sets"].get(s, {})
    if info.get("status") != "clustered":
        return {"status": f"not computed: the set '{s}' was not clustered"}
    clustered = info["views_clustered"]
    views = [v for v in clustered if v != "EXPRESSION"]
    mask = result["masks"][s]
    tiles = sample["tiles"][mask]
    mpp = (sample["provenance"].get("prepare") or {}).get("mpp_target")
    genes = None
    if args.relevance_gene_views and sample["counts"].shape[0] > 0:
        from scipy import sparse

        Z, idx = relevance.expression_genes(sparse.csc_matrix(sample["counts"])[:, np.flatnonzero(mask)], args.n_hvg)
        genes = (Z, np.asarray(sample["gene_ids"])[idx], np.asarray(sample["gene_names"])[idx])
    rel = relevance.run(result["scores"][s], result["pc_variance"][s], tiles, views, s, mpp, genes=genes,
                        gene_views=args.relevance_gene_views, n_folds=args.redundancy_folds,
                        block_um=args.redundancy_block_um, seed=args.seed, null_shifts=args.redundancy_null_shifts,
                        n_boot=args.redundancy_bootstrap, gene_null_shifts=args.relevance_gene_null_shifts, log=log)
    if rel["info"]["status"] != "computed":
        log.info("relevance: %s", rel["info"]["status"])
        return rel["info"]
    files = {"relevance": f"{prefix}.relevance.tsv", "relevance_partition": f"{prefix}.relevance_partition.tsv"}
    rel["table"].to_csv(files["relevance"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    rel["partition"].to_csv(files["relevance_partition"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    if "genes" in rel:
        files["relevance_genes"] = f"{prefix}.relevance_genes.tsv.gz"
        with gzip.open(files["relevance_genes"], "wt") as fh:
            rel["genes"].to_csv(fh, sep="\t", index=False, na_rep="NA", float_format="%.6g")
    outputs.update(files)
    source = lineage.source_text(process, {}, parents=parents,
                                 context=f"{prefix}, set '{s}' ({len(tiles)} tiles), seed {args.seed}")
    text = relevance_captions(rel, args, source)
    made = {"relevance_keep": clusterplots.relevance_figure(rel, text["main"], prefix, s)}
    if "genes" in rel:
        made["relevance_genes_keep"] = clusterplots.relevance_genes_figure(rel, text["genes"], prefix)
    for name, img in made.items():
        if img is not None:
            path = f"{prefix}.{name}.png"
            img.save(path)
            outputs[f"figure_{name}"] = path
    return rel["info"]


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
