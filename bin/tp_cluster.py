#!/usr/bin/env python3
"""Optional (--cluster): every feature view of one slide clustered on its own (Leiden and
k-means at the same numbers of clusters), and how much the views agree (ARI, NMI and a
spatial null). See tileparity_lib/cluster.py for the method."""

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tileparity_lib import cluster, clusterplots, export, lineage  # noqa: E402

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


def captions(info: dict, set_name: str, args, ref: str, source: str, mask_mpp) -> dict:
    """The four-part explanations of the figures of one tile set."""
    n, k_ref = info["n_tiles"], info["k_ref"]
    ks = ", ".join(map(str, info["k"]))
    null = info["null"]
    if null["moves"]:
        null_how = (f"The null moves one map over the other on the tile grid with wrap-around ({null['moves']} "
                    "moves of at least 20 % of the grid, with at least 30 % of the tiles still over tiles): both maps "
                    "keep their patchiness but lose their alignment.")
    else:
        null_how = "No spatial null here (" + null.get("note", "not computed").replace("not computed: ", "") + ")."
    both = len(args.methods) == 2
    methods = "Leiden and k-means" if both else ("Leiden" if args.methods == ["leiden"] else "k-means")
    top = f"Top: adjusted Rand index (ARI) of every pair of views at k = {k_ref}"
    if null["moves"]:
        top += "; the small number under each value is the 95th percentile of its spatial null"
    if both:
        top += ", and the grey diagonal is the ARI between the Leiden and the k-means clusterings of the same view"
    bottom = f"Bottom: ARI of every view with {ref} for k = {ks}"
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
                 f"k ({methods} here). {top}. {bottom}."),
        "how": how,
        "caveats": ("The clustering ignores where the tiles are (no spatial smoothing) and uses one random seed. ARI "
                    "depends on k and on the cluster sizes, so views are only compared at the same k: points where "
                    "Leiden could not reach exactly k are left out of the curves. " + null_how +
                    " Every pair, every k, NMI and the null are in cluster_agreement.tsv."),
        "source": source,
    }
    name = "Leiden" if "leiden" in args.methods else "k-means"
    thumb = f"{mask_mpp:g} um/px thumbnail" if mask_mpp else "slide thumbnail"
    maps = {
        "what": (f"The {n} tiles of the set '{set_name}' coloured by their cluster ({name}, k = {k_ref}) in every "
                 f"view, each view clustered on its own, over the faded {thumb}. Colours follow the clusters of {ref}: "
                 f"the clusters of every other view take the colour of the {ref} cluster they share most tiles with "
                 "(one-to-one matching), so the same colour in two maps marks matched clusters."),
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
    return {"agreement": agreement, "maps": maps, "contingency": contingency}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--prefix", default=None, help="<sample>_mpp<scale> (default: the only sample in --input-dir)")
    ap.add_argument("--views", default="all", help=f"comma list of {', '.join(cluster.VIEWS)}, or all")
    ap.add_argument("--sets", default="keep,extended")
    ap.add_argument("--methods", default="leiden,kmeans")
    ap.add_argument("--k", default=",".join(map(str, cluster.DEFAULT_K)), help="numbers of clusters")
    ap.add_argument("--k-ref", type=int, default=8, help="k of the heatmaps, maps and contingency tables")
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--min-counts-ratio", type=float, default=0.1,
                    help="set 'extended': minimum molecules per spatial unit, as a fraction of the keep tiles' median")
    ap.add_argument("--null-shifts", type=int, default=100, help="moves of the spatial null (0: no null)")
    ap.add_argument("--seed", type=int, default=0)
    lineage.add_cli(ap)
    args = ap.parse_args(argv)
    process = args.process or "TILEPARITY_CLUSTER"
    args.views = names(args.views, cluster.VIEWS, "views")
    args.sets = names(args.sets, cluster.SETS, "sets")
    args.methods = names(args.methods, cluster.METHODS, "methods")
    args.k = integers(args.k)

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
                         min_counts_ratio=args.min_counts_ratio, null_shifts=args.null_shifts, seed=args.seed, log=log)

    outputs = {"labels": f"{prefix}.clusters.tsv.gz", "agreement": f"{prefix}.cluster_agreement.tsv",
               "summary": f"{prefix}.cluster.json"}
    with gzip.open(outputs["labels"], "wt") as fh:
        cluster.labels_table(sample, result, args.views).to_csv(fh, sep="\t", index=False, na_rep="")
    result["agreement"].to_csv(outputs["agreement"], sep="\t", index=False, na_rep="NA", float_format="%.6g")

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
        for name, img in made.items():
            if img is not None:
                path = f"{prefix}.cluster_{name}.png"
                img.save(path)
                outputs[f"figure_{name}"] = path

    summary = {
        "prefix": prefix,
        "sample_id": str(sample["tiles"]["sample_id"].iloc[0]) if len(sample["tiles"]) else prep.get("sample_id"),
        "scale_mpp": prep.get("mpp_target"),
        "parameters": {k: v for k, v in vars(args).items() if k not in ("input_dir", "lineage_parent", "process")},
        "sets": result["sets"],
        "outputs": {k: Path(v).name for k, v in outputs.items() if k != "summary"},
        "notes": ["cluster numbers are per view and per slide; 0 is the largest cluster",
                  "columns of the labels table: <set>_<view>_<method>_k<k>; empty outside the set",
                  "agreement rows with method leiden_vs_kmeans compare the two methods on the same view"],
    }
    Path(outputs["summary"]).write_text(json.dumps(lineage._jsonable(summary), indent=2))
    inputs = {role: Path(args.input_dir) / f"{prefix}.{suffix}" for role, suffix in
              (("tiles", "tiles.tsv.gz"), ("uni2h", "uni2h.h5"), ("lbp", "lbp.h5"), ("counts", "counts.h5"))}
    lineage.write(f"{prefix}.cluster.lineage.json", lineage.build(
        "cluster", process, prefix, parameters=summary["parameters"],
        inputs={k: v for k, v in inputs.items() if v.exists()}, outputs=outputs, parents=parents,
        packages=["scikit-learn", "igraph", "umap-learn", "numpy", "scipy", "pandas"]))
    for s, info in result["sets"].items():
        log.info("%s, set %s: %s", prefix, s, info.get("status"))


if __name__ == "__main__":
    from tileparity_lib import setup_logging

    setup_logging()
    main()
