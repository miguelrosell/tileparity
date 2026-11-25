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


def thin_note(args) -> str:
    """Words for the captions when the counts were thinned (--expression-thin-to)."""
    info = getattr(args, "thin_info", None) or {}
    if not info.get("applied"):
        return ""
    return (f", counts thinned to a median of {info['target_molecules_per_tile']:g} molecules per tile "
            f"(every molecule kept with p = {info['probability']:.3g})")


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
                    "tissue; each null map a mosaic of a few moves covering (almost) every tile, fitted on the same "
                    "tiles and folds as the real map: r2_null_mean, r2_null_p95, p_null in redundancy.tsv) is the R2 "
                    "reached by chance. No "
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
    ap.add_argument("--relevance-gene-null-shifts", type=int, default=199,
                    help="moves of the per-gene spatial null (0: none); the smallest p-value is 1 / (1 + moves)")
    ap.add_argument("--markers", action="store_true", help="marker genes of every expression domain")
    ap.add_argument("--markers-top", type=int, default=25, help="markers kept per domain")
    ap.add_argument("--texture-genes", action="store_true",
                    help="partial Spearman correlation of 52 named LBP texture features with the most variable genes")
    ap.add_argument("--texture-genes-view", default="LBP_RGB", help="LBP view of the texture features")
    ap.add_argument("--texture-genes-channel", default="mean",
                    help="'mean' (mean over the view's channels) or one channel of the view (e.g. R)")
    ap.add_argument("--texture-genes-n", type=int, default=500, help="most variable genes tested")
    ap.add_argument("--expression-thin-to", type=float, default=0,
                    help="thin the counts so that the keep tiles' median is about this many molecules per tile "
                         "(same probability for every molecule; 0: off)")
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
    if args.expression_thin_to < 0 or args.markers_top < 1 or args.texture_genes_n < 1:
        raise SystemExit("--expression-thin-to must be >= 0, --markers-top and --texture-genes-n >= 1")

    prefixes = export.discover(args.input_dir)
    if args.prefix is None:
        if len(prefixes) != 1:
            raise SystemExit(f"--prefix is needed: {len(prefixes)} samples in {args.input_dir}")
        args.prefix = prefixes[0]
    elif args.prefix not in prefixes:
        raise SystemExit(f"no {args.prefix}.tiles.tsv.gz in {args.input_dir}")
    prefix = args.prefix
    sample = export.load_sample(args.input_dir, prefix)
    thin_info = None
    if args.expression_thin_to > 0:
        thin_info = cluster.thin_counts(sample, args.expression_thin_to, args.seed)
        args.thin_info = thin_info
        log.info("expression thinning: %s", thin_info)
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
                                     context=f"{prefix}, set '{s}' ({info['n_tiles']} tiles), seed {args.seed}{thin_note(args)}")
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

    markers_info = None
    if args.markers:
        markers_info = run_markers(sample, result, f1, args, prefix, parents, process, outputs)

    texture_info = None
    if args.texture_genes:
        texture_info = run_texture_genes(sample, result, args, prefix, parents, process, outputs)

    summary = {
        "prefix": prefix,
        "sample_id": str(sample["tiles"]["sample_id"].iloc[0]) if len(sample["tiles"]) else prep.get("sample_id"),
        "scale_mpp": prep.get("mpp_target"),
        "parameters": {k: v for k, v in vars(args).items()
                       if k not in ("input_dir", "lineage_parent", "process", "thin_info")},
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
    if markers_info is not None:
        summary["markers"] = markers_info
    if texture_info is not None:
        summary["texture_genes"] = texture_info
    if thin_info is not None:
        summary["expression_thinning"] = thin_info
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
                                 context=f"{prefix}, set '{s}' ({len(tiles)} tiles), seed {args.seed}{thin_note(args)}")
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


Q_TEXT = relevance.Q_EXPLAINED


def relevance_captions(rel: dict, args, source: str) -> dict:
    info = rel["info"]
    block = f"{info['block']:g} {'um' if info['block_unit'] == 'um' else 'tiles'}"
    views = "Views: " + "; ".join(f"{v} = {cluster.view_description(v)}" for v in info["views"]) + "."
    model = (f"ridge regression with spatial cross-validation ({info['n_folds']} folds of square blocks of {block}): "
             "every tile is predicted by a model fitted on other parts of the slide")
    ginfo = info.get("genes") or {}
    n_null = ginfo.get("null_moves", 0)
    ceiling = (info.get("split_halves") or {}).get("r2_ceiling")
    null_text = ("the spatial null: the view's map moved over the tile grid, so that it keeps its structure but no "
                 "longer describes the same tissue; each null map is a mosaic of a few moves that covers (almost) every "
                 "tile and is fitted on the same tiles and folds as the real map")
    main = {
        "what": (f"How much of the expression of the {info['n_tiles']} tiles of the set 'keep' each image view predicts. "
                 f"a: the expression's principal components (those of the clustering, at most {args.n_pcs}) predicted "
                 f"from each view's components, {model}; R2 = the share of the expression's variance that the view "
                 f"predicts; whiskers: 95 % interval (block bootstrap); black tick: 95th percentile of {null_text}. In "
                 "brackets: the share of the reachable. Every molecule of a tile goes at random to half A or half B (two "
                 "independent samples of the same tissue); the ceiling is the share of half B's variance that the two "
                 "halves share, i.e. that is not sampling noise"
                 + (f" (= {ceiling:.2f} here)" if ceiling is not None and np.isfinite(ceiling) else "") + "; share of "
                 "the reachable = R2 of half B from the view / ceiling. b: for pairs of views A and B, the expression's "
                 "variance (0 to 1) split by a joint model (one regression on the components of both views): only A = "
                 "R2(joint) - R2(B), shared (what both predict) = R2(A) + R2(B) - R2(joint), only B = R2(joint) - "
                 "R2(A), and not explained = 1 - R2(joint); only A + shared + only B = R2(joint). c: R2 of each of the "
                 "first five expression components on its own. " + views),
        "how": ("A bar well beyond its black tick: the view predicts expression better than an unrelated map with the "
                "same structure would. A low R2 with a high share of the reachable: the expression itself is noisy "
                "(few molecules per tile), not the image blind; the share of the reachable does not depend on the depth. "
                "In b, a long 'only A' part: A sees expression that B misses. With LBP + INTENSITY_HE, 'only LBP' is "
                "what texture adds to the mere amount of stain. In c, a view that predicts PC1 but not the later "
                "components sees only the main axis of the tissue."),
        "caveats": ("Linear prediction between principal components; R2 refers to the variance the expression's "
                    "components keep. Out-of-sample R2 can be slightly negative, and so can a part in b. The ceiling is "
                    "an estimate (it can be off by sampling error). Neighbouring tiles are not independent, so no "
                    "classical p-value is given. Every predictor: relevance.tsv; the split: relevance_partition.tsv."),
        "source": source,
    }
    genes = {
        "what": (f"Gene by gene: each of the expression's most variable genes (counts per 10 000, log1p, z-scores, as in "
                 f"the clustering) predicted from the components of an image view ({model}; a penalty per gene chosen "
                 "by generalised cross-validation inside the training folds). a: every gene's R2 from two views "
                 "(diagonal: equal R2), coloured by which views beat the per-gene spatial null (" + null_text +
                 f"; {n_null} null maps, each fitted exactly like the real map; p = (1 + null maps reaching the real R2) / "
                 "(1 + null maps); q < 0.05). " + Q_TEXT + " Labelled: the genes best predicted from the second view and "
                 "those it adds most to the first. b: the genes best predicted from the second view, with the first "
                 "view's R2 (black tick) and the gene's reliability (split-half correlation stepped up to full depth: "
                 "roughly the largest R2 its noise allows). " + views),
        "how": ("Purple: both views predict the gene beyond chance; orange: only LBP; blue: only UNI2-h; grey: neither. "
                "Genes far above the diagonal are predicted from texture (LBP) better than from UNI2-h. Genes of "
                "abundant, visually distinct cell types (stroma, muscle, epithelium) are expected among the best "
                "predicted; a gene predicted from LBP beyond INTENSITY_HE (column adds_LBP_RGB_to_INTENSITY_HE) "
                "relates to texture rather than to the amount of stain."),
        "caveats": ("Beating the null means a real association, not a large one: with thousands of tiles, an R2 of a few "
                    "thousandths can be significant (for example a gene whose measured value follows the number of "
                    "cells, which the image sees). A gene predicted from the image is associated with what the tissue "
                    "looks like there, not necessarily expressed by the cells that create the texture. With "
                    f"{n_null} null maps the smallest possible p is {1 / (1 + n_null) if n_null else float('nan'):.3g}. "
                    "Genes are correlated with each other. Confirm a gene on other slides. Every gene: "
                    "relevance_genes.tsv.gz."),
        "source": source,
    }
    partition = {
        "what": ("Gene by gene, each gene's variance (z-scores: 1 per gene) split by what UNI2-h and LBP predict: the "
                 "joint model is one regression on the components of both views; only UNI2-h = R2(joint) - R2(LBP), "
                 "shared (what both predict) = R2(UNI2-h) + R2(LBP) - R2(joint), only LBP = R2(joint) - R2(UNI2-h), "
                 "and not explained = 1 - R2(joint) (light grey); only UNI2-h + shared + only LBP = R2(joint). a: the "
                 "genes best predicted by the joint model; black tick: the gene's reliability (split-half correlation "
                 "stepped up to full depth, an estimate of the largest R2 its sampling noise allows). b: the parts "
                 "averaged over all genes and over the well-measured genes (reliability >= 0.5); black tick: their "
                 "mean reliability."),
        "how": ("A gene with a long 'only UNI2-h' part is predicted by what UNI2-h sees beyond LBP texture; a long "
                "'shared' part: the gene follows texture that UNI2-h also encodes; 'only LBP': texture UNI2-h does not "
                "represent linearly. Between the end of the coloured parts and the tick: expression the image misses "
                "although the molecules would allow it; beyond the tick: sampling noise."),
        "caveats": ("Linear models on principal components: 'only UNI2-h' is what UNI2-h predicts beyond the LBP "
                    "histograms, not beyond texture in general. Out-of-sample R2 is noisy, so small parts (and "
                    "negative ones, drawn as 0) mean little. The reliability assumes the full depth is the sum of the "
                    "halves, an approximation on the log scale. No test of the parts themselves. Every gene: "
                    "relevance_gene_partition.tsv.gz."),
        "source": source,
    }
    return {"main": main, "genes": genes, "partition": partition}


def run_relevance(sample: dict, result: dict, args, prefix: str, parents, process: str, outputs: dict) -> dict:
    """The relevance analysis of the set keep: tables, figures and what it did."""
    from scipy import sparse

    s = "keep"
    info = result["sets"].get(s, {})
    if info.get("status") != "clustered":
        return {"status": f"not computed: the set '{s}' was not clustered"}
    clustered = info["views_clustered"]
    views = [v for v in clustered if v != "EXPRESSION"]
    mask = result["masks"][s]
    tiles = sample["tiles"][mask]
    mpp = (sample["provenance"].get("prepare") or {}).get("mpp_target")
    genes, counts, idx = None, None, None
    if sample["counts"].shape[0] > 0:
        Yset = sparse.csc_matrix(sample["counts"])[:, np.flatnonzero(mask)]
        counts = sparse.csr_matrix(Yset.T)
        if args.relevance_gene_views:
            Z, idx = relevance.expression_genes(Yset, args.n_hvg)
            genes = (Z, np.asarray(sample["gene_ids"])[idx], np.asarray(sample["gene_names"])[idx])
    rel = relevance.run(result["scores"][s], result["pc_variance"][s], tiles, views, s, mpp, genes=genes,
                        gene_views=args.relevance_gene_views, n_folds=args.redundancy_folds,
                        block_um=args.redundancy_block_um, seed=args.seed, null_shifts=args.redundancy_null_shifts,
                        n_boot=args.redundancy_bootstrap, gene_null_shifts=args.relevance_gene_null_shifts,
                        counts=counts, gene_index=idx, n_hvg=args.n_hvg, log=log)
    if rel["info"]["status"] != "computed":
        log.info("relevance: %s", rel["info"]["status"])
        return rel["info"]
    files = {"relevance": f"{prefix}.relevance.tsv", "relevance_partition": f"{prefix}.relevance_partition.tsv"}
    rel["table"].to_csv(files["relevance"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    rel["partition"].to_csv(files["relevance_partition"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    for key, name in (("genes", "relevance_genes"), ("gene_partition", "relevance_gene_partition")):
        if key in rel:
            files[name] = f"{prefix}.{name}.tsv.gz"
            with gzip.open(files[name], "wt") as fh:
                rel[key].to_csv(fh, sep="\t", index=False, na_rep="NA", float_format="%.6g")
    outputs.update(files)
    source = lineage.source_text(process, {}, parents=parents,
                                 context=f"{prefix}, set '{s}' ({len(tiles)} tiles), seed {args.seed}{thin_note(args)}")
    text = relevance_captions(rel, args, source)
    made = {"relevance_keep": clusterplots.relevance_figure(rel, text["main"], prefix, s)}
    if "genes" in rel:
        made["relevance_genes_keep"] = clusterplots.relevance_genes_figure(rel, text["genes"], prefix)
    if "gene_partition" in rel:
        made["relevance_gene_partition_keep"] = clusterplots.gene_partition_figure(rel["gene_partition"],
                                                                                   text["partition"], prefix)
    for name, img in made.items():
        if img is not None:
            path = f"{prefix}.{name}.png"
            img.save(path)
            outputs[f"figure_{name}"] = path
    return rel["info"]


def markers_captions(args, source: str, info: dict) -> dict:
    ref = info.get("reference_view", "EXPRESSION")
    return {
        "what": (f"The domains of {ref} (its clusters, at the k shown, per method) described by their marker genes. "
                 "Per domain: its colour and number of tiles; the F1 of every image view for it (how well the view's "
                 "matched cluster covers the domain: 2 x shared tiles / (domain + cluster)); and its best markers, "
                 "ranked by AUROC (the probability that a random tile of the domain has more of the gene than a random "
                 "tile outside it; log1p counts per 10 000), among the genes with log2 fold change > 0, detected in at "
                 "least 10 % of the domain's tiles and in at least 5 % of the set's tiles."),
        "how": ("Read a row as a sentence: this domain is where these genes are high, and these views find it (dark "
                "cells) or miss it (white cells). A domain that UNI2-h finds and LBP misses is a candidate for what "
                "UNI2-h sees beyond texture; its markers say what it is in biology. AUROC 0.5: no difference, 1: "
                "every tile of the domain above every other tile."),
        "caveats": ("Markers describe the domains; they are not tests (the tiles of a domain are neighbours, and the "
                    "domains were found from these same genes). A tile holds many cells, so a marker can come from "
                    "any of them. Every domain, method and k: markers.tsv.gz; the F1 of every view: cluster_f1.tsv."),
        "source": source,
    }


def run_markers(sample: dict, result: dict, f1, args, prefix: str, parents, process: str, outputs: dict) -> dict:
    """Marker genes of the expression domains of the set keep (every method and k)."""
    from scipy import sparse

    from tileparity_lib import markers

    s = "keep"
    info = result["sets"].get(s, {})
    if info.get("status") != "clustered" or sample["counts"].shape[0] == 0:
        return {"status": "not computed: no clustered expression in the set 'keep'"}
    mask = result["masks"][s]
    parts = markers.expression_partitions(result, s)
    if not parts:
        return {"status": "not computed: EXPRESSION was not clustered"}
    counts = sparse.csr_matrix(sparse.csc_matrix(sample["counts"])[:, np.flatnonzero(mask)].T)
    table = markers.domain_markers(counts, sample["gene_ids"], sample["gene_names"], parts, s, n_top=args.markers_top)
    files = {"markers": f"{prefix}.markers.tsv.gz"}
    with gzip.open(files["markers"], "wt") as fh:
        table.to_csv(fh, sep="\t", index=False, na_rep="NA", float_format="%.6g")
    dom = markers.with_f1(table, f1)
    if len(dom):
        files["markers_domains"] = f"{prefix}.markers_domains.tsv"
        dom.to_csv(files["markers_domains"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    outputs.update(files)
    source = lineage.source_text(process, {}, parents=parents,
                                 context=f"{prefix}, set '{s}' ({int(mask.sum())} tiles), seed {args.seed}"
                                         f"{thin_note(args)}")
    img = clusterplots.markers_figure(table, f1, info, args.methods, markers_captions(args, source, info), prefix)
    if img is not None:
        path = f"{prefix}.cluster_markers_keep.png"
        img.save(path)
        outputs["figure_markers_keep"] = path
    log.info("markers: %d rows for %d partitions", len(table), len(parts))
    return {"status": "computed", "partitions": len(parts), "rows": int(len(table)), "top": args.markers_top}


def texture_captions(args, source: str, n_genes: int, channels: str) -> dict:
    return {
        "what": (f"For every named texture feature of {args.texture_genes_view} (channels: {channels}) and every one of "
                 f"the {n_genes} most variable genes, the partial Spearman correlation over the tiles of the set 'keep' "
                 "with the tile's depth removed (ranks of the feature and of the gene, log1p counts per 10 000, each "
                 "regressed on the rank of log10 molecules per tile; correlation of the residuals). A feature is the "
                 "fraction of the tile's pixels of one LBP type at one radius: spot bright (code 0: every neighbour "
                 "darker), spot dark or flat (code P: no neighbour darker), edge (codes 1 .. P - 1: the brighter "
                 "neighbours form one arc), irregular (code P + 1). a: the genes most associated with texture "
                 "(largest |rho| among q < 0.05) against the 52 features; colour: rho, saturated at the 98th "
                 "percentile of |rho| among the genes shown (written under the panel), so that small differences show; "
                 "rho itself ranges from -1 to 1; dot: q < 0.05. b: per feature, how many genes reach q < 0.05. "
                 + Q_TEXT),
        "how": ("Red: more of this texture where the gene is high; blue: less. A column with many dots is a texture "
                "that follows the expression; a row, a gene that follows texture. In b, the radius where a type peaks "
                "is the scale (in pixels; x 0.5 um at 0.5 um per px) of the structure that goes with the genes, e.g. "
                "dark spots of nucleus size."),
        "caveats": ("Neighbouring tiles are alike, so the p-values use the modified t-test (Clifford, Richardson & "
                    "Hemon 1989; Dutilleul 1993): the number of tiles is replaced by an effective number from the "
                    "spatial correlograms of the two variables (column n_eff). Benjamini-Hochberg q over all features "
                    "and genes. Features of neighbouring radii and of different types are correlated with each other "
                    "(the four types of one radius add up to 1). A correlation is an association: cell density, tissue "
                    "type and stain all move texture and genes together. Every pair: texture_genes.tsv.gz; per "
                    "feature: texture_genes_summary.tsv."),
        "source": source,
    }


def run_texture_genes(sample: dict, result: dict, args, prefix: str, parents, process: str, outputs: dict) -> dict:
    """Named LBP texture features against the most variable genes, set keep."""
    from scipy import sparse

    from tileparity_lib import texture_genes

    s = "keep"
    info = result["sets"].get(s, {})
    if info.get("status") != "clustered" or sample["counts"].shape[0] == 0:
        return {"status": "not computed: no clustered expression in the set 'keep'"}
    if args.texture_genes_view not in sample["blocks"]:
        return {"status": f"not computed: no view {args.texture_genes_view}"}
    mask = result["masks"][s]
    tiles = sample["tiles"][mask]
    if "row" not in tiles or (tiles["row"].astype(int) < 0).any():
        return {"status": "not computed: the tiles are not on a grid"}
    X, cols = sample["blocks"][args.texture_genes_view]
    F, fnames, ftable = texture_genes.features(X[mask], cols, args.texture_genes_channel)
    Yset = sparse.csc_matrix(sample["counts"])[:, np.flatnonzero(mask)]
    from tileparity_lib.cluster import highly_variable_genes

    N = relevance.normalise(Yset.T)
    idx = highly_variable_genes(N, args.texture_genes_n)
    depth = np.log10(np.asarray(Yset.sum(axis=0)).ravel() + 1.0)
    table = texture_genes.run(F, fnames, ftable, N[:, idx].toarray(), np.asarray(sample["gene_ids"])[idx],
                              np.asarray(sample["gene_names"])[idx], depth, tiles["row"], tiles["col"])
    summ = texture_genes.summary(table)
    files = {"texture_genes": f"{prefix}.texture_genes.tsv.gz", "texture_genes_summary": f"{prefix}.texture_genes_summary.tsv"}
    with gzip.open(files["texture_genes"], "wt") as fh:
        table.to_csv(fh, sep="\t", index=False, na_rep="NA", float_format="%.6g")
    summ.to_csv(files["texture_genes_summary"], sep="\t", index=False, na_rep="NA", float_format="%.6g")
    outputs.update(files)
    channels = ftable["channels"].iloc[0] if len(ftable) else ""
    source = lineage.source_text(process, {}, parents=parents,
                                 context=f"{prefix}, set '{s}' ({int(mask.sum())} tiles), seed {args.seed}"
                                         f"{thin_note(args)}")
    img = clusterplots.texture_genes_figure(table, summ, texture_captions(args, source, len(idx), channels), prefix)
    if img is not None:
        path = f"{prefix}.texture_genes_keep.png"
        img.save(path)
        outputs["figure_texture_genes_keep"] = path
    n_sig = int((table["q"] < 0.05).sum()) if len(table) else 0
    log.info("texture-genes: %d features x %d genes, %d pairs with q < 0.05", F.shape[1], len(idx), n_sig)
    return {"status": "computed", "view": args.texture_genes_view, "channels": channels, "n_features": int(F.shape[1]),
            "n_genes": int(len(idx)), "n_pairs_q05": n_sig,
            "median_n_eff": float(table["n_eff"].median()) if len(table) else None, "n_tiles": int(mask.sum())}


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
