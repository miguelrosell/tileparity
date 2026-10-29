"""Optional clustering of every feature view of one slide, and how much the views agree
(``--cluster``).

Every slide (and scale) is clustered on its own: clustering several slides together would
mostly separate the slides (batch effect), which is not the question here. Two sets of
tiles are clustered:

* ``keep``: the default analysis set (column ``keep`` of the tile table: tissue >=
  ``--min_tissue`` and the spatial checks);
* ``extended``, a sensitivity analysis: every extracted tile (tissue >=
  ``--min_tissue_extract``) with at least ``--cluster_min_counts`` times the slide's median
  number of molecules per spatial unit (Visium HD bin or Visium spot) of the ``keep`` tiles.
  It adds tiles with less tissue that still carry RNA and removes nearly empty tiles (glass,
  blood lakes). Samples without expression use every extracted tile. With the default
  ``--min_tissue_extract`` (= ``--min_tissue``) it differs from ``keep`` only by the counts
  floor; a set with exactly the tiles of ``keep`` is not clustered twice.

Views (those computed): UNI2H_RGB, UNI2H_RGB_HE, LBP_RGB, LBP_RGB_HE, LBP_HE, INTENSITY_HE
and EXPRESSION. Preparation, then PCA (``--cluster_n_pcs`` components):

* UNI2-h and INTENSITY_HE: every column standardised (z-score over the set's tiles);
* LBP: proportions, square root (Hellinger) and the same weight for every channel and
  radius (:func:`tileparity_lib.qcumap.lbp_recipe`, as in the 3-D UMAP QC);
* EXPRESSION: Scanpy's standard recipe - counts per 10 000, log1p, the ``--cluster_n_hvg``
  most variable genes (dispersion normalised within 20 bins of mean expression, Scanpy's
  'seurat' flavour), z-scores clipped at 10.

Clustering of the principal components (``--cluster_methods``):

* ``leiden``: Leiden community detection (modularity with a resolution parameter) on the
  graph of the ``--cluster_n_neighbors`` nearest neighbours, weighted as in UMAP and
  Scanpy (fuzzy simplicial set). Leiden has no "number of clusters" parameter: the
  resolution is searched by bisection until the partition has exactly k clusters, so that
  all views are compared at the same k (the ARI depends on the number of clusters);
* ``kmeans``: k-means with k clusters (10 starts), a check that does not use a graph.

Agreement between two partitions of the same tiles:

* ARI (adjusted Rand index): over all pairs of tiles, how often the two partitions agree
  (same cluster in both, or different clusters in both), rescaled so that random
  partitions score about 0 and identical ones 1;
* NMI (normalised mutual information): how much knowing one partition tells about the
  other, from 0 to 1;
* a spatial null: two maps made of large patches agree partly by chance. One map is moved
  over the other on the tile grid, with wrap-around (torus translation), so that both keep
  their patches but lose their alignment; the ARI of the moved maps is the chance level
  (only for tiles on the grid).
"""

from __future__ import annotations

import math
import random
import warnings

import numpy as np
import pandas as pd

from . import qcumap

VIEWS = ("UNI2H_RGB", "UNI2H_RGB_HE", "LBP_RGB", "LBP_RGB_HE", "LBP_HE", "INTENSITY_HE", "EXPRESSION")
METHODS = ("leiden", "kmeans")
SETS = ("keep", "extended")
DEFAULT_K = (2, 3, 4, 6, 8, 10, 12, 15)
MIN_TILES = 6  # smaller sets are not clustered; a k is used only when the set has at least 2k tiles
LABEL_COLUMN = "{set}_{view}_{method}_k{k}"  # columns of <prefix>.clusters.tsv.gz
EXPORT_PREFIX = "cluster_"  # exported columns: cluster_<view>_leiden_k<k> (set keep)
METHOD_PAIR = "leiden_vs_kmeans"  # agreement rows comparing the two methods on one view


# ---- tile sets -------------------------------------------------------------------------------------
def counts_per_unit(sample: dict) -> np.ndarray | None:
    """Molecules per spatial unit (Visium HD bin, or Visium spot equivalent) of every tile;
    None for samples without expression."""
    from scipy import sparse

    if sample["counts"].shape[0] == 0:
        return None
    totals = np.asarray(sparse.csc_matrix(sample["counts"]).sum(axis=0)).ravel().astype(float)
    units = pd.to_numeric(sample["tiles"]["st_units_equiv"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(units > 0, totals / units, np.nan)


def tile_sets(sample: dict, min_counts_ratio: float) -> tuple[dict, dict]:
    """Boolean masks over the sample's extracted tiles for the ``keep`` and ``extended``
    sets, and how they were made."""
    tiles = sample["tiles"]
    keep = tiles["keep"].to_numpy(bool)
    per_unit = counts_per_unit(sample)
    info = {"keep": {"definition": ("the tiles with keep = True in the tile table (at least --min_tissue tissue, "
                                    "spatial checks passed)")}}
    if per_unit is None:
        extended = np.ones(len(tiles), bool)
        info["extended"] = {"definition": "every extracted tile (no expression data for a counts floor)"}
    else:
        median = float(np.nanmedian(per_unit[keep])) if keep.any() else float("nan")
        floor = min_counts_ratio * median
        extended = np.nan_to_num(per_unit, nan=0.0) >= floor if np.isfinite(floor) else np.zeros(len(tiles), bool)
        info["extended"] = {"definition": (f"the extracted tiles with at least {min_counts_ratio:g} x the median "
                                           "number of molecules per spatial unit of the keep tiles"),
                            "median_counts_per_unit": median if np.isfinite(median) else None,
                            "floor_counts_per_unit": floor if np.isfinite(floor) else None,
                            "keep_tiles_below_floor": int((keep & ~extended).sum()),
                            "tiles_added_to_keep": int((extended & ~keep).sum())}
    info["keep"]["n_tiles"] = int(keep.sum())
    info["extended"]["n_tiles"] = int(extended.sum())
    return {"keep": keep, "extended": extended}, info


# ---- views -------------------------------------------------------------------------------------------
def zscore(X: np.ndarray, clip: float | None = None) -> np.ndarray:
    """Columns centred and divided by their standard deviation (ddof = 1, as Scanpy's
    ``scale``); constant columns become 0."""
    X = np.asarray(X, float)
    sd = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.ones(X.shape[1])
    sd[~(sd > 0)] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    return np.clip(Z, -clip, clip) if clip else Z


def highly_variable_genes(logN, n_top: int) -> np.ndarray:
    """Indices (sorted) of the ``n_top`` most variable genes of log1p-normalised counts
    (tiles x genes), as Scanpy's ``highly_variable_genes(flavor='seurat')``: the dispersion
    (variance / mean of the normalised counts, log) is standardised within 20 bins of mean
    expression; a gene alone in its bin gets 1."""
    from scipy import sparse

    X = sparse.csr_matrix(logN, dtype=float, copy=True)
    X.data = np.expm1(X.data)
    n = X.shape[0]
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    var = (mean_sq - mean**2) * (n / max(n - 1, 1))
    mean[mean == 0] = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = var / mean
        disp[disp == 0] = np.nan
        disp = np.log(disp)
    df = pd.DataFrame({"disp": disp, "bin": pd.cut(np.log1p(mean), 20)})
    stats = df.groupby("bin", observed=True)["disp"].agg(avg="mean", dev="std")
    single = stats["dev"].isna()
    stats.loc[single, "dev"] = stats.loc[single, "avg"]
    stats.loc[single, "avg"] = 0.0
    per_gene = stats.reindex(df["bin"]).to_numpy(float)
    norm = (df["disp"].to_numpy(float) - per_gene[:, 0]) / per_gene[:, 1]
    norm[~np.isfinite(norm)] = -np.inf
    n_top = min(int(n_top), int(np.isfinite(norm).sum()))
    return np.sort(np.argsort(-norm, kind="stable")[:n_top])


def expression_features(Y_genes_x_tiles, n_hvg: int) -> tuple[np.ndarray, int]:
    """Tiles x genes matrix of the most variable genes: counts per 10 000, log1p, z-scores
    clipped at 10. Returns (matrix, number of genes detected in these tiles)."""
    from scipy import sparse

    Y = sparse.csr_matrix(sparse.csc_matrix(Y_genes_x_tiles).T, dtype=float)
    totals = np.asarray(Y.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    N = sparse.csr_matrix(sparse.diags(1e4 / totals) @ Y)
    N.data = np.log1p(N.data)
    detected = int((np.asarray((Y > 0).sum(axis=0)).ravel() > 0).sum())
    idx = highly_variable_genes(N, n_hvg)
    return zscore(N[:, idx].toarray(), clip=10.0), detected


def view_matrices(sample: dict, sel: np.ndarray, views, n_hvg: int) -> dict:
    """{view: (prepared tiles x features matrix of the selected tiles, description)} for the
    requested views that exist in the sample."""
    from scipy import sparse

    out = {}
    tiles = sample["tiles"]
    tile_px = int(tiles["tile_px"].iloc[0]) if len(tiles) else 224
    for v in views:
        if v == "EXPRESSION":
            if sample["counts"].shape[0] == 0:
                continue
            X, detected = expression_features(sparse.csc_matrix(sample["counts"])[:, np.flatnonzero(sel)], n_hvg)
            out[v] = (X, f"the {X.shape[1]} most variable of {detected} detected genes: counts per 10 000, log1p, "
                         "z-scores clipped at 10")
        elif v in sample["blocks"]:
            X, cols = sample["blocks"][v]
            X = np.asarray(X)[sel]
            if v.startswith("LBP"):
                out[v] = (qcumap.lbp_recipe(X, cols, tile_px),
                          f"{X.shape[1]} LBP bins: proportions, Hellinger, equal weight per channel and radius")
            else:
                out[v] = (zscore(X), f"{X.shape[1]} columns, z-scores")
    return out


def pca(X: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Principal components (tiles x k, k <= n_components) and their explained variance
    ratios."""
    from sklearn.decomposition import PCA

    X = np.asarray(X, float)
    X = X[:, X.std(axis=0) > 0] if X.shape[1] else X
    k = int(min(n_components, X.shape[0] - 1, X.shape[1]))
    if k < 1:
        return np.zeros((X.shape[0], 0)), np.zeros(0)
    model = PCA(n_components=k, random_state=seed)
    return model.fit_transform(X), model.explained_variance_ratio_


# ---- clustering ----------------------------------------------------------------------------------------
def n_clusters(labels: np.ndarray) -> int:
    return int(len(np.unique(labels)))


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Clusters renumbered 0, 1, ... from the largest to the smallest (ties: original order)."""
    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)
    order = values[np.lexsort((values, -counts))]
    lookup = {v: i for i, v in enumerate(order.tolist())}
    return np.array([lookup[v] for v in labels.tolist()], dtype=int)


def knn_graph(P: np.ndarray, n_neighbors: int, seed: int):
    """Weighted, undirected graph of the nearest neighbours of every tile in PC space: exact
    Euclidean neighbours (the tile itself counts as the first), weighted with UMAP's fuzzy
    simplicial set, which is what Scanpy's ``neighbors`` computes (Scanpy switches to
    approximate neighbours for large data sets; here they are always exact)."""
    import igraph
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors
    from umap.umap_ import fuzzy_simplicial_set

    P = np.asarray(P, float)
    n = P.shape[0]
    k = int(max(2, min(n_neighbors, n)))
    # the k - 1 nearest other tiles, then the tile itself first at distance exactly 0 (a
    # rounding error there would become UMAP's distance to the nearest neighbour)
    dist, idx = NearestNeighbors(n_neighbors=k - 1).fit(P).kneighbors()
    idx = np.hstack([np.arange(n)[:, None], idx])
    dist = np.hstack([np.zeros((n, 1)), dist])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        G, _, _ = fuzzy_simplicial_set(sparse.coo_matrix((n, 1)), k, np.random.RandomState(seed), "euclidean",
                                       knn_indices=idx, knn_dists=dist)
    G = sparse.triu(sparse.coo_matrix(G), k=1).tocoo()
    graph = igraph.Graph(n=n, edges=list(zip(G.row.tolist(), G.col.tolist())), directed=False)
    graph.es["weight"] = G.data.astype(float).tolist()
    return graph


def leiden(graph, resolution: float, seed: int) -> np.ndarray:
    random.seed(seed)  # python-igraph draws its random numbers from Python's random module
    part = graph.community_leiden(objective_function="modularity", weights="weight", resolution=resolution,
                                  n_iterations=-1)
    return relabel_by_size(np.array(part.membership))


def leiden_k(graph, k: int, seed: int, cache: dict | None = None, lo: float = 1e-3, hi: float = 100.0,
             max_iter: int = 40, tol: float = 1e-4) -> tuple[np.ndarray, float, int]:
    """Leiden partition with k clusters, by bisection of the resolution on a log scale.

    ``cache`` (resolution -> labels) can be shared between calls on the same graph, so that
    the search for one k starts from what the searches for other k found. Returns (labels,
    resolution, number of clusters reached); when no resolution gives exactly k (the number
    of clusters can jump, and it never falls below the number of connected components of
    the graph), the closest partition found."""
    cache = {} if cache is None else cache

    def at(r: float) -> np.ndarray:
        if r not in cache:
            cache[r] = leiden(graph, r, seed)
        return cache[r]

    at(lo)
    for r, lab in cache.items():
        if n_clusters(lab) == k:
            return lab, r, k
    if n_clusters(cache[lo]) > k:  # more than k clusters even at the lowest resolution (disconnected graph)
        return cache[lo], lo, n_clusters(cache[lo])
    a, b = math.log10(lo), math.log10(hi)
    below = [math.log10(r) for r, lab in cache.items() if n_clusters(lab) < k]
    above = [math.log10(r) for r, lab in cache.items() if n_clusters(lab) > k]
    if below and above and max(below) < min(above):  # narrow the bracket with what is known
        a, b = max(below), min(above)
    for _ in range(max_iter):
        if b - a < tol:
            break
        m = (a + b) / 2
        n = n_clusters(at(10**m))
        if n == k:
            return cache[10**m], 10**m, k
        a, b = (m, b) if n < k else (a, m)
    r = min(cache, key=lambda r: (abs(n_clusters(cache[r]) - k), r))
    return cache[r], r, n_clusters(cache[r])


def kmeans(P: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")  # fewer distinct points than k: fewer clusters, recorded as k reached
        return relabel_by_size(KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(P))


# ---- agreement -----------------------------------------------------------------------------------------
def _codes(x) -> np.ndarray:
    """Labels as small non-negative integers (cluster labels already are)."""
    x = np.asarray(x).ravel()
    if x.dtype.kind in "iu" and x.size and x.min() >= 0 and x.max() <= 4 * x.size:
        return x.astype(np.int64, copy=False)
    return np.unique(x, return_inverse=True)[1].ravel()


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index, the same formula as scikit-learn's ``adjusted_rand_score`` (from
    the pair confusion matrix) but fast enough for the spatial null."""
    a, b = _codes(a), _codes(b)
    n = float(a.size)
    if n == 0:
        return 1.0
    kb = int(b.max()) + 1
    M = np.bincount(a * kb + b, minlength=(int(a.max()) + 1) * kb).astype(float).reshape(-1, kb)
    S = float((M**2).sum())
    rows = float((M.sum(axis=1) ** 2).sum())
    cols = float((M.sum(axis=0) ** 2).sum())
    tp, fp, fn = S - n, cols - S, rows - S
    tn = n * n - fp - fn - S
    if fn == 0 and fp == 0:
        return 1.0
    return float(2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)))


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(a, b))


def on_grid(tiles: pd.DataFrame) -> bool:
    return bool(len(tiles) and (tiles["row"].astype(int) >= 0).all() and (tiles["col"].astype(int) >= 0).all())


def torus_pairs(rows, cols, n: int, seed: int, min_overlap: float = 0.3,
                min_shift: float = 0.2) -> list[tuple[np.ndarray, np.ndarray]]:
    """Moves of a map over the tile grid for the spatial null. Each item (i, j) pairs tile i
    of one map with tile j of the other map moved by (dy, dx) rows and columns with
    wrap-around, for up to ``n`` distinct moves of at least ``min_shift`` of the grid's
    extent (shorter way round, rows and columns as fractions of the grid) that leave at
    least ``min_overlap`` of the tiles over tiles."""
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    if len(rows) == 0:
        return []
    R, C = rows - rows.min(), cols - cols.min()
    H, W = int(R.max()) + 1, int(C.max()) + 1
    index = np.full((H, W), -1)
    index[R, C] = np.arange(len(rows))
    rng = np.random.default_rng(seed)
    out = []
    for s in rng.permutation(H * W):
        if len(out) >= n:
            break
        dy, dx = divmod(int(s), W)
        if math.hypot(min(dy, H - dy) / H, min(dx, W - dx) / W) < min_shift:
            continue
        moved = np.roll(index, (dy, dx), axis=(0, 1))
        m = (index >= 0) & (moved >= 0)
        if m.sum() < min_overlap * len(rows):
            continue
        out.append((index[m], moved[m]))
    return out


def null_ari(a: np.ndarray, b: np.ndarray, pairs) -> np.ndarray:
    return np.array([ari(a[i], b[j]) for i, j in pairs], float)


def contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Number of tiles in cluster i of ``a`` and cluster j of ``b``."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    M = np.zeros((int(a.max()) + 1, int(b.max()) + 1), int)
    np.add.at(M, (a, b), 1)
    return M


def match_to_reference(ref: np.ndarray, labels: np.ndarray) -> dict:
    """{cluster of ``labels``: cluster of ``ref`` it is matched to}: one-to-one matching that
    maximises the number of shared tiles (Hungarian algorithm). Clusters left over (when
    ``labels`` has more clusters than ``ref``) get new numbers after those of ``ref``."""
    from scipy.optimize import linear_sum_assignment

    M = contingency(labels, ref)
    r, c = linear_sum_assignment(-M)
    mapping = {int(i): int(j) for i, j in zip(r, c)}
    nxt = M.shape[1]
    for i in range(M.shape[0]):
        if i not in mapping:
            mapping[i] = nxt
            nxt += 1
    return mapping


def reference_view(views: list[str]) -> str | None:
    """The view other views are compared with in the curves, maps and contingency tables:
    EXPRESSION when computed, else UNI2H_RGB, else the first view."""
    for v in ("EXPRESSION", "UNI2H_RGB"):
        if v in views:
            return v
    return views[0] if views else None


# ---- one slide ----------------------------------------------------------------------------------------
AGREEMENT_COLUMNS = ["set", "method", "k", "view_a", "view_b", "n_tiles", "k_a", "k_b", "ari", "nmi",
                     "null_moves", "null_ari_mean", "null_ari_p05", "null_ari_p50", "null_ari_p95", "p_null"]


def run(sample: dict, views=VIEWS, ks=DEFAULT_K, k_ref: int = 8, methods=METHODS, sets=SETS, n_pcs: int = 50,
        n_neighbors: int = 15, n_hvg: int = 2000, min_counts_ratio: float = 0.1, null_shifts: int = 100,
        seed: int = 0, log=None) -> dict:
    """Cluster every view of one sample in every tile set and measure the agreement.

    Returns {"sets": {set: info}, "masks": {set: bool over the sample's tiles},
    "labels": {set: {(view, method, k): labels over the set's tiles}}, "agreement":
    DataFrame}. ``info`` records the definition and size of the set, the k used, the
    reference view and, per view, the features, PCs and Leiden resolutions."""
    masks, set_info = tile_sets(sample, min_counts_ratio)
    tiles = sample["tiles"]
    ks = sorted({int(k) for k in ks if int(k) >= 2})
    methods = [m for m in METHODS if m in methods]
    out = {"sets": {}, "masks": {}, "labels": {}}
    rows = []
    for s in sets:
        sel = masks[s]
        info = dict(set_info[s])
        n = int(sel.sum())
        same = [t for t in out["labels"] if np.array_equal(out["masks"][t], sel)]
        usable = [k for k in ks if 2 * k <= n]
        if same:
            info["status"] = f"not clustered: the same {n} tiles as the set '{same[0]}'"
        elif n < MIN_TILES or not usable:
            info["status"] = f"not clustered: {n} tiles (at least {max(MIN_TILES, 2 * min(ks, default=2))} needed)"
        if "status" in info:
            out["sets"][s] = info
            out["masks"][s] = sel
            if log:
                log.info("%s: %s", s, info["status"])
            continue
        info["k"] = usable
        info["k_ref"] = k_ref if k_ref in usable else max([k for k in usable if k <= k_ref] or [min(usable)])
        info["views"] = {}
        labels: dict = {}
        for v, (X, description) in view_matrices(sample, sel, views, n_hvg).items():
            P, evr = pca(X, n_pcs, seed)
            vinfo = {"features": description, "n_features": int(X.shape[1]), "n_pcs": int(P.shape[1]),
                     "pc_variance_explained": round(float(evr.sum()), 4)}
            if P.shape[1] == 0:
                vinfo["status"] = "not clustered: no variable features"
                info["views"][v] = vinfo
                continue
            if "leiden" in methods:
                graph = knn_graph(P, n_neighbors, seed)
                cache: dict = {}
                vinfo["leiden"] = {}
                for k in usable:
                    lab, r, reached = leiden_k(graph, k, seed, cache=cache)
                    labels[(v, "leiden", k)] = lab
                    vinfo["leiden"][str(k)] = {"resolution": float(f"{r:.6g}"), "k_reached": reached}
                vinfo["leiden_runs"] = len(cache)
            if "kmeans" in methods:
                vinfo["kmeans"] = {}
                for k in usable:
                    labels[(v, "kmeans", k)] = lab = kmeans(P, k, seed)
                    vinfo["kmeans"][str(k)] = {"k_reached": n_clusters(lab)}
            vinfo["status"] = "clustered"
            info["views"][v] = vinfo
            if log:
                log.info("%s: %s clustered (%d tiles, %d PCs)", s, v, n, P.shape[1])
        done = [v for v in views if info["views"].get(v, {}).get("status") == "clustered"]
        info["views_clustered"] = done
        info["reference_view"] = reference_view(done)
        sub = tiles[sel]
        pairs = torus_pairs(sub["row"], sub["col"], null_shifts, seed) if on_grid(sub) and null_shifts > 0 else []
        info["null"] = ({"moves": len(pairs), "min_shift": 0.2, "min_overlap": 0.3} if pairs else
                        {"moves": 0, "note": "not computed: tiles not on a grid (spot tiles), or no possible move"})
        for m in methods:
            for k in usable:
                for i, a in enumerate(done):
                    for b in done[i + 1:]:
                        rows.append(_row(s, m, k, a, b, labels[(a, m, k)], labels[(b, m, k)], pairs))
        if len(methods) == 2:
            for k in usable:
                for v in done:
                    rows.append(_row(s, METHOD_PAIR, k, v, v, labels[(v, "leiden", k)], labels[(v, "kmeans", k)], []))
        info["status"] = "clustered"
        out["sets"][s] = info
        out["labels"][s] = labels
        out["masks"][s] = sel
    out["agreement"] = pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)
    return out


def _row(s, method, k, a, b, la, lb, pairs) -> dict:
    row = {"set": s, "method": method, "k": k, "view_a": a, "view_b": b, "n_tiles": int(len(la)),
           "k_a": n_clusters(la), "k_b": n_clusters(lb), "ari": ari(la, lb), "nmi": nmi(la, lb),
           "null_moves": len(pairs)}
    if pairs:
        null = null_ari(la, lb, pairs)
        row.update(null_ari_mean=float(null.mean()), null_ari_p05=float(np.percentile(null, 5)),
                   null_ari_p50=float(np.percentile(null, 50)), null_ari_p95=float(np.percentile(null, 95)),
                   p_null=float((1 + (null >= row["ari"]).sum()) / (1 + len(null))))
    return row


def labels_table(sample: dict, result: dict, views=VIEWS) -> pd.DataFrame:
    """One row per extracted tile: tile_id, in_<set> and the label of every clustering
    (``<set>_<view>_<method>_k<k>``; empty for tiles outside that set)."""
    tiles = sample["tiles"]
    cols = {"tile_id": tiles["tile_id"].astype(str).to_numpy()}
    for s in SETS:
        if s in result["masks"]:
            cols[f"in_{s}"] = result["masks"][s]
    order = {v: i for i, v in enumerate(views)}
    for s, labs in result["labels"].items():
        idx = np.flatnonzero(result["masks"][s])
        for (v, m, k) in sorted(labs, key=lambda t: (order.get(t[0], 99), METHODS.index(t[1]), t[2])):
            col = pd.array([pd.NA] * len(tiles), dtype="Int64")
            col[idx] = labs[(v, m, k)]
            cols[LABEL_COLUMN.format(set=s, view=v, method=m, k=k)] = col
    return pd.DataFrame(cols)


def export_columns(table: pd.DataFrame) -> dict:
    """Columns of a clusters table that the exports add to the tile table: the Leiden labels
    of the set ``keep``, named ``cluster_<view>_leiden_k<k>``."""
    out = {}
    for c in table.columns:
        if c.startswith("keep_") and "_leiden_k" in c:
            out[EXPORT_PREFIX + c[len("keep_"):]] = table[c]
    return out
