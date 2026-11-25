"""Optional (``--relevance``, with ``--cluster``): how much of the expression each image view
predicts, what each view adds to the others, and which genes the image predicts.

The redundancy analysis (:mod:`tileparity_lib.redundancy`) asks what one image view knows
about another. Relevance asks what the image knows about the expression, with the same
machinery (ridge regression, spatial block cross-validation, block-bootstrap interval,
spatial null), on the set ``keep`` of one slide:

* Expression as a whole: the principal components of EXPRESSION (those of the clustering)
  predicted from the principal components of every image view. R^2 = the share of the
  expression's variance (as far as its components keep it) that the view predicts; also
  per component for the first five (``r2_pc1`` .. ``r2_pc5``).
* Spatial null (view level): the view's map moved over the tile grid, as in the redundancy.
  Every null realization is a mosaic of a few moves that gives (almost) every tile a partner,
  so that the moved map is fitted on the same tiles and folds as the real one (a fit on fewer
  tiles has a lower R^2 even by chance); ``p_null`` = (1 + realizations with R^2 >= the real
  one) / (1 + realizations).
* Noise ceiling (split halves): every molecule of a tile goes at random to half A or half B
  (binomial thinning, p = 0.5); given the tile's biology the two halves are independent
  samples of it. Half B is prepared like the expression (its most variable genes, z-scores,
  principal components) and half A on the same genes; half A is projected on half B's
  component axes. Over the tiles, cov(A, B) along an axis is the variance both halves share
  (the biology) and var(B) the total, so ``r2_ceiling`` = sum cov / sum var over the
  components is the share of half B's variance that is not sampling noise: the largest R^2
  any predictor that does not see half B's own molecules can reach (estimate; it can be off
  by sampling error). ``r2_half`` is the R^2 of half B's components from the view and
  ``fraction`` = r2_half / r2_ceiling the share of the reachable that the view reaches. Under
  this model it does not depend on the depth (more noise lowers both alike).
* What each view adds (variance partitioning, for pairs of views A, B): R^2 from A, from B
  and from the joint model (A's and B's components side by side, one regression). Then
  only A = R^2(joint) - R^2(B), only B = R^2(joint) - R^2(A),
  shared = R^2(A) + R^2(B) - R^2(joint) (what both predict: the overlap),
  not explained = 1 - R^2(joint); only A + shared + only B = R^2(joint).
  The default pairs are UNI2-h and LBP on the same pixels (what texture adds to the
  foundation model and the reverse) and LBP with INTENSITY_HE (what texture adds to the
  mere amount of stain).
* Gene by gene: every one of the expression's most variable genes (counts per 10 000,
  log1p, z-scores, as in the clustering) predicted from each of a few image views, with a
  ridge penalty per gene chosen by generalised cross-validation inside the training folds
  (Golub, Heath & Wahba 1979). Spatial null per gene: ``gene_null_shifts`` realizations of
  the moved view as above, each fitted with exactly the rule of the real fit on the same
  tiles and folds; p = (1 + realizations with R^2 >= the gene's real R^2) / (1 +
  realizations), so the smallest possible p is 1 / (1 + realizations); Benjamini-Hochberg
  q-values over the genes. In simulations (irregular tissue, 50 smooth predictors, white or
  smooth noise genes) this keeps its level (2-4 % below 0.05) at every overlap, where moves
  of a single shift exceeded it when they kept few tiles (up to 10 % at 34 % overlap).
* Per gene, split-half correlation r (log1p counts per 10 000 of half A and half B over the
  tiles): the share of the gene's variance at half depth that is signal; stepped up to full
  depth with the Spearman-Brown formula, ``reliability`` = 2 r / (1 + r): roughly the
  largest R^2 any predictor could reach for the gene (the formula assumes the full depth is
  the sum of the halves; on the log scale this is an approximation).
* Per gene, partition for the pairs (UNI2H_RGB, LBP_RGB) and (INTENSITY_HE, LBP_RGB), as
  above (``gene_partition``).

R^2 is out-of-sample, so it can be negative when a view predicts nothing. Genes are not
independent of each other: use the q-values to rank genes, and confirm a gene on other
slides. "Only A" is what A predicts linearly that B does not, not what B cannot see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import redundancy as R

N_PC_COLUMNS = 5
TABLE_COLUMNS = (["set", "predictor", "kind", "n_tiles", "n_folds", "n_pcs_predictor", "n_pcs_expression",
                  "expression_pc_variance", "r2"] + [f"r2_pc{i}" for i in range(1, N_PC_COLUMNS + 1)] +
                 ["r2_ci_low", "r2_ci_high", "r2_fold_sd", "null_moves", "r2_null_mean", "r2_null_p95", "p_null",
                  "r2_half", "r2_ceiling", "fraction"])
PARTITION_COLUMNS = ["set", "view_a", "view_b", "r2_a", "r2_b", "r2_joint", "only_a", "only_b", "shared", "not_explained"]
GENE_PARTITION_COLUMNS = ["gene_id", "gene_name", "view_a", "view_b", "r2_a", "r2_b", "r2_joint", "only_a", "only_b",
                          "shared", "not_explained", "q_a", "q_b", "split_half_r", "reliability"]
Q_EXPLAINED = ("q is the Benjamini-Hochberg adjusted p-value: with thousands of tests, some p-values are small by "
               "chance alone; if every result with q < 0.05 is kept, at most about 5 % of them are expected to be false "
               "(a property of the set, not of each result). It is computed by ranking the p-values from smallest to "
               "largest and multiplying the p-value at rank i of m by m / i (then never letting q decrease down the "
               "list).")
WELL_MEASURED = 0.5  # reliability above which a gene counts as well measured (at least half of it is signal)
DEFAULT_PAIRS = (("UNI2H_RGB", "LBP_RGB"), ("UNI2H_RGB_HE", "LBP_RGB_HE"), ("INTENSITY_HE", "LBP_RGB"),
                 ("UNI2H_RGB", "INTENSITY_HE"))
DEFAULT_GENE_VIEWS = ("UNI2H_RGB", "LBP_RGB", "INTENSITY_HE")
GENE_PAIRS = (("UNI2H_RGB", "LBP_RGB"), ("INTENSITY_HE", "LBP_RGB"))
JOINT = "+"  # name of a joint predictor: "<view a>+<view b>"


def normalise(Y_tiles_x_genes):
    """log1p of counts per 10 000 (sparse, tiles x genes)."""
    from scipy import sparse

    Y = sparse.csr_matrix(Y_tiles_x_genes, dtype=float)
    totals = np.asarray(Y.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    N = sparse.csr_matrix(sparse.diags(1e4 / totals) @ Y)
    N.data = np.log1p(N.data)
    return N


def expression_genes(Y_genes_x_tiles, n_hvg: int) -> tuple[np.ndarray, np.ndarray]:
    """(tiles x genes z-scores, indices of the genes) of the ``n_hvg`` most variable genes,
    prepared as for the clustering (counts per 10 000, log1p, z-scores clipped at 10)."""
    from scipy import sparse

    from .cluster import highly_variable_genes, zscore

    N = normalise(sparse.csc_matrix(Y_genes_x_tiles).T)
    idx = highly_variable_genes(N, n_hvg)
    return zscore(N[:, idx].toarray(), clip=10.0), idx


def split_halves(Y_tiles_x_genes, seed: int):
    """Every molecule to half A or half B at random (binomial thinning, p = 0.5)."""
    from scipy import sparse

    rng = np.random.default_rng(seed)
    A = sparse.csr_matrix(Y_tiles_x_genes, dtype=float, copy=True)
    A.data = rng.binomial(np.round(A.data).astype(np.int64), 0.5).astype(float)
    B = sparse.csr_matrix(Y_tiles_x_genes, dtype=float) - A
    A.eliminate_zeros()
    B.eliminate_zeros()
    return A, B


def zscore_genes(Y_tiles_x_genes, idx) -> np.ndarray:
    """z-scores (clipped at 10) of the genes ``idx`` prepared as for the clustering (counts per
    10 000 of all genes, log1p)."""
    from .cluster import zscore

    return zscore(normalise(Y_tiles_x_genes)[:, idx].toarray(), clip=10.0)


def split_half_components(ZA: np.ndarray, ZB: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, ...]:
    """Half B's principal components and how much of each is signal.

    ``ZA``, ``ZB``: tiles x the same genes, each half prepared alike. PCA of half B gives its
    components EB (tiles x c) and their axes; half A is projected on the same axes. Half A's
    sampling noise is independent of half B's (and of the axes, which come from half B), so
    over the tiles cov(A_c, B_c) estimates the variance of component c that both halves share
    and var(B_c) its total variance. Returns (EB, shared variance per component, total
    variance per component); ceiling = shared.sum() / total.sum()."""
    from sklearn.decomposition import PCA

    ZA, ZB = np.asarray(ZA, float), np.asarray(ZB, float)
    ok = ZB.std(axis=0) > 0
    ZA, ZB = ZA[:, ok], ZB[:, ok]
    k = int(min(n_components, ZB.shape[0] - 1, ZB.shape[1]))
    if k < 1:
        return np.zeros((ZB.shape[0], 0)), np.zeros(0), np.zeros(0)
    model = PCA(n_components=k, random_state=seed).fit(ZB)
    EB = model.transform(ZB)
    EB -= EB.mean(axis=0)
    PA = (ZA - ZA.mean(axis=0)) @ model.components_.T
    return EB, (PA * EB).mean(axis=0), (EB**2).mean(axis=0)


def ceiling_of(shared: np.ndarray, total: np.ndarray) -> float:
    """Share of the total variance that both halves share (NaN when there is none)."""
    t = float(np.sum(total))
    return float(np.sum(shared)) / t if t > 0 else float("nan")


def gene_reliability(A, B, idx) -> tuple[np.ndarray, np.ndarray]:
    """(split-half r, Spearman-Brown reliability 2r / (1 + r)) of the genes ``idx`` from two
    halves (tiles x genes counts)."""
    a = normalise(A)[:, idx].toarray()
    b = normalise(B)[:, idx].toarray()
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    den = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))
    r = np.where(den > 0, (a * b).sum(axis=0) / np.where(den > 0, den, 1.0), np.nan)
    rel = np.where(r > -1, 2 * r / (1 + r), np.nan)
    return r, rel


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN stays NaN)."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def ridge_gcv_predict(X: np.ndarray, Y: np.ndarray, folds: np.ndarray, alphas=R.ALPHAS) -> np.ndarray:
    """Out-of-fold predictions of every column of ``Y`` from ``X``: ridge regression with an
    intercept, one penalty per column of Y chosen among ``alphas`` by generalised
    cross-validation (GCV) inside the training folds. Through the SVD of the training X the
    residual sum of squares of every column and penalty costs O(components x columns), so
    thousands of genes and hundreds of null moves stay cheap."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    alphas = np.asarray(alphas, float)
    pred = np.zeros(Y.shape, float)
    for f in np.unique(folds):
        test = folds == f
        Xtr, Ytr = X[~test], Y[~test]
        n = len(Xtr)
        xm, ym = Xtr.mean(axis=0), Ytr.mean(axis=0)
        U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
        keep = s > (s[0] * 1e-10 if s.size else 0)
        U, s, Vt = U[:, keep], s[keep], Vt[keep]
        Yc = Ytr - ym
        UtY = U.T @ Yc  # components x columns
        total = (Yc**2).sum(axis=0)
        D = s[None, :] ** 2 / (s[None, :] ** 2 + alphas[:, None])  # penalties x components
        rss = total[None, :] - (2 * D - D**2) @ (UtY**2)
        dof = 1.0 + D.sum(axis=1)
        gcv = rss / np.maximum(1 - dof / n, 1e-12)[:, None] ** 2
        pick = alphas[np.argmin(gcv, axis=0)]
        coef = Vt.T @ ((s[:, None] / (s[:, None] ** 2 + pick[None, :])) * UtY)
        pred[test] = (X[test] - xm) @ coef + ym
    return pred


def _r2_columns(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return R.r2_scores(Y, pred)[1]


def gene_null(X: np.ndarray, G: np.ndarray, folds: np.ndarray, partners) -> np.ndarray:
    """realizations x genes null R^2: the genes of the tiles with a partner predicted from the
    view's tiles laid under them (``partners``: one array per realization, see
    :func:`tileparity_lib.cluster.patched_partners`), same folds and exactly the rule of the
    real fit."""
    out = []
    for partner in partners:
        ok = np.flatnonzero(partner >= 0)
        f = folds[ok]
        if len(np.unique(f)) < 2:
            continue
        out.append(_r2_columns(G[ok], ridge_gcv_predict(X[partner[ok]], G[ok], f)))
    return np.vstack(out) if out else np.empty((0, G.shape[1]))


def empirical_p(real: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided empirical p-value per column: (1 + #null >= real) / (1 + #null)."""
    if len(null) == 0:
        return np.full(real.shape, np.nan)
    return (1.0 + (null >= real[None, :]).sum(axis=0)) / (1.0 + len(null))


def run(scores: dict, variance: dict, tiles: pd.DataFrame, views: list[str], set_name: str, mpp: float | None,
        genes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, gene_views=DEFAULT_GENE_VIEWS,
        pairs=DEFAULT_PAIRS, n_folds: int = 5, block_um: float = 1000.0, seed: int = 0, null_shifts: int = 20,
        n_boot: int = 200, gene_null_shifts: int = 199, counts=None, gene_index=None, n_hvg: int = 2000,
        log=None) -> dict:
    """``scores``: {view: tiles x components} including EXPRESSION; ``genes``: (tiles x genes
    z-scores, gene ids, gene names) or None; ``counts``: the set's tiles x genes counts (for
    the split halves; None skips them) and ``gene_index`` the columns of ``counts`` that
    ``genes`` holds. Returns {"table", "partition", "genes", "gene_partition", "info"} or
    {"info"} with the reason it could not run."""
    info = {"n_tiles": int(len(tiles)), "n_folds_requested": int(n_folds)}
    if "EXPRESSION" not in scores or scores["EXPRESSION"].shape[1] == 0:
        info["status"] = "not computed: no expression in this set"
        return {"info": info}
    views = [v for v in views if v in scores and v != "EXPRESSION" and scores[v].shape[1] > 0]
    if not views:
        info["status"] = "not computed: no image view"
        return {"info": info}
    x, y, unit = R.tile_centres_um(tiles, mpp)
    block = block_um if unit == "um" else max(1.0, round(block_um / 112.0))
    folds = R.spatial_folds(x, y, block, n_folds, seed)
    info.update({"views": views, "block": float(block), "block_unit": unit})
    if folds is None:
        info["status"] = f"not computed: the tiles fit in fewer than 2 blocks of {block:g} {unit}"
        return {"info": info}
    info["n_folds"] = int(folds.max()) + 1
    blocks = R.spatial_blocks(x, y, block)
    partners, info["null"] = R.null_partners(tiles, max(null_shifts, gene_null_shifts), seed)
    E = scores["EXPRESSION"]

    # split halves for the noise ceiling
    EB, ceiling, halves = None, np.nan, None
    if counts is not None and counts.shape[0] == len(tiles):
        halves = split_halves(counts, seed)
        ZB, idx_b = expression_genes(halves[1].T, n_hvg)
        ZA = zscore_genes(halves[0], idx_b)
        EB, shared_c, total_c = split_half_components(ZA, ZB, E.shape[1], seed)
        ceiling = ceiling_of(shared_c, total_c)
        if EB.shape[1] == 0:
            EB = None
        info["split_halves"] = {
            "r2_ceiling": float(ceiling), "seed": int(seed), "n_components": int(len(total_c)),
            "ceiling_pc1_5": [float(x) for x in (shared_c[:5] / np.where(total_c[:5] > 0, total_c[:5], np.nan))],
            "molecules_median_half": float(np.median(np.asarray(halves[1].sum(axis=1)).ravel()))}

    def fit(X):
        pred, _ = R.cross_predict(X, E, folds)
        return pred, R.r2_scores(E, pred)

    rows, r2_of = [], {}
    predictors = [(v, scores[v], "view") for v in views]
    joint_pairs = [(a, b) for a, b in pairs if a in views and b in views]
    predictors += [(f"{a}{JOINT}{b}", np.hstack([scores[a], scores[b]]), "joint") for a, b in joint_pairs]
    for name, X, kind in predictors:
        pred, (total, per) = fit(X)
        r2_of[name] = total
        fr = R.fold_r2(E, pred, folds)
        boot = R.block_bootstrap_r2(E, pred, blocks, n_boot, seed)
        null = (R.null_r2(X, E, folds, partners[:null_shifts]) if (partners and kind == "view" and null_shifts > 0)
                else np.array([], float))
        row = {"set": set_name, "predictor": name, "kind": kind, "n_tiles": int(len(tiles)),
               "n_folds": info["n_folds"], "n_pcs_predictor": int(X.shape[1]), "n_pcs_expression": int(E.shape[1]),
               "expression_pc_variance": float(variance.get("EXPRESSION", np.nan)), "r2": total,
               "r2_ci_low": float(np.nanpercentile(boot, 2.5)) if boot.size else np.nan,
               "r2_ci_high": float(np.nanpercentile(boot, 97.5)) if boot.size else np.nan,
               "r2_fold_sd": float(np.nanstd(fr, ddof=1)) if len(fr) > 1 else np.nan,
               "null_moves": int(null.size), "r2_null_mean": float(null.mean()) if null.size else np.nan,
               "r2_null_p95": float(np.percentile(null, 95)) if null.size else np.nan,
               "p_null": float((1 + (null >= total).sum()) / (1 + null.size)) if null.size else np.nan,
               "r2_ceiling": ceiling}
        for i in range(N_PC_COLUMNS):
            row[f"r2_pc{i + 1}"] = float(per[i]) if per.size > i else np.nan
        if EB is not None:
            ph, _ = R.cross_predict(X, EB, folds)
            row["r2_half"] = R.r2_scores(EB, ph)[0]
            row["fraction"] = row["r2_half"] / ceiling if ceiling > 0 else np.nan
        rows.append(row)
        if log:
            log.info("relevance: expression from %s, R2 %.3f", name, total)
    part = []
    for a, b in joint_pairs:
        ra, rb, rab = r2_of[a], r2_of[b], r2_of[f"{a}{JOINT}{b}"]
        part.append({"set": set_name, "view_a": a, "view_b": b, "r2_a": ra, "r2_b": rb, "r2_joint": rab,
                     "only_a": rab - rb, "only_b": rab - ra, "shared": ra + rb - rab, "not_explained": 1 - rab})
    out = {"table": pd.DataFrame(rows, columns=TABLE_COLUMNS), "partition": pd.DataFrame(part, columns=PARTITION_COLUMNS),
           "info": info}

    if genes is not None and genes[0].shape[1] > 0:
        G, gene_ids, gene_names = genes
        gtab = pd.DataFrame({"gene_id": np.asarray(gene_ids, dtype=object), "gene_name": np.asarray(gene_names, dtype=object)})
        if halves is not None and gene_index is not None:
            gtab["split_half_r"], gtab["reliability"] = gene_reliability(halves[0], halves[1], gene_index)
        gviews = [v for v in gene_views if v in views]
        n_null = 0
        for v in gviews:
            r2 = _r2_columns(G, ridge_gcv_predict(scores[v], G, folds))
            gtab[f"r2_{v}"] = r2
            if partners and gene_null_shifts > 0:
                N = gene_null(scores[v], G, folds, partners[:gene_null_shifts])
                n_null = max(n_null, len(N))
                if len(N) >= 2:
                    mu, sd = N.mean(axis=0), N.std(axis=0, ddof=1)
                    p = empirical_p(r2, N)
                    gtab[f"r2_null_mean_{v}"], gtab[f"r2_null_sd_{v}"] = mu, sd
                    gtab[f"r2_null_p95_{v}"] = np.percentile(N, 95, axis=0)
                    gtab[f"z_{v}"] = (r2 - mu) / np.where(sd > 0, sd, np.nan)
                    gtab[f"p_{v}"], gtab[f"q_{v}"] = p, _bh(p)
            if log:
                log.info("relevance: %d genes from %s (median R2 %.3f)", G.shape[1], v, float(np.nanmedian(r2)))
        gparts = []
        for a, b in GENE_PAIRS:
            if a in gviews and b in gviews:
                rab = _r2_columns(G, ridge_gcv_predict(np.hstack([scores[a], scores[b]]), G, folds))
                gtab[f"r2_{a}{JOINT}{b}"] = rab
                gtab[f"adds_{b}_to_{a}"] = rab - gtab[f"r2_{a}"]
                gtab[f"adds_{a}_to_{b}"] = rab - gtab[f"r2_{b}"]
                ra, rb = gtab[f"r2_{a}"].to_numpy(float), gtab[f"r2_{b}"].to_numpy(float)
                gparts.append(pd.DataFrame({
                    "gene_id": gtab["gene_id"], "gene_name": gtab["gene_name"], "view_a": a, "view_b": b,
                    "r2_a": ra, "r2_b": rb, "r2_joint": rab, "only_a": rab - rb, "only_b": rab - ra,
                    "shared": ra + rb - rab, "not_explained": 1 - rab,
                    "q_a": gtab.get(f"q_{a}", np.nan), "q_b": gtab.get(f"q_{b}", np.nan),
                    "split_half_r": gtab.get("split_half_r", np.nan), "reliability": gtab.get("reliability", np.nan)}))
        out["genes"] = gtab
        out["gene_partition"] = (pd.concat(gparts, ignore_index=True)[GENE_PARTITION_COLUMNS] if gparts
                                 else pd.DataFrame(columns=GENE_PARTITION_COLUMNS))
        info["genes"] = {"n": int(G.shape[1]), "views": gviews, "null_moves": int(n_null),
                         "smallest_p": (1.0 / (1 + n_null)) if n_null else None,
                         "null": "moved view as a mosaic of moves covering (almost) every tile",
                         "penalty": "one per gene, generalised cross-validation inside the training folds"}
    info["status"] = "computed"
    return out


def gene_category(q_x: np.ndarray, q_y: np.ndarray, level: float = 0.05) -> np.ndarray:
    """'both' / 'only_x' / 'only_y' / 'neither': which of two views beats the spatial null."""
    sx, sy = np.asarray(q_x, float) < level, np.asarray(q_y, float) < level
    return np.where(sx & sy, "both", np.where(sx, "only_x", np.where(sy, "only_y", "neither")))
