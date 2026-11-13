"""Optional (``--relevance``, with ``--cluster``): how much of the expression each image view
predicts, what each view adds to the others, and which genes the image predicts.

The redundancy analysis (:mod:`tileparity_lib.redundancy`) asks what one image view knows
about another. Relevance asks what the image knows about the expression, with the same
machinery (ridge regression, spatial block cross-validation, block-bootstrap interval,
spatial null), on the set ``keep`` of one slide:

* Expression as a whole: the principal components of EXPRESSION (those of the clustering)
  predicted from the principal components of every image view. R^2 = the share of the
  expression's variance (as far as its components keep it) that the view predicts.
* What each view adds (variance partitioning, for pairs of views A, B): R^2 from A, from B
  and from A and B together (their components side by side). Then
  unique to A = R^2(A+B) - R^2(B), unique to B = R^2(A+B) - R^2(A),
  shared = R^2(A) + R^2(B) - R^2(A+B).
  The default pairs are UNI2-h and LBP on the same pixels (what texture adds to the
  foundation model and the reverse) and LBP with INTENSITY_HE (what texture adds to the
  mere amount of stain).
* Gene by gene: every one of the expression's most variable genes (counts per 10 000,
  log1p, z-scores, as in the clustering) predicted from each of a few image views (a ridge
  penalty per gene), with a spatial null per gene (the view's map moved over the tile
  grid): ``r2_null_mean``, ``r2_null_sd``, z = (R^2 - null mean) / null sd, an approximate
  one-sided p-value from z (normal), and Benjamini-Hochberg q-values over the genes. The
  joint fits give, per gene, what LBP adds to UNI2-h and what LBP adds to INTENSITY_HE.

R^2 is out-of-sample, so it can be negative when a view predicts nothing. Genes are not
independent of each other, and the z-based p-values assume a roughly normal null: use them
to rank genes, and confirm a gene on other slides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import redundancy as R

TABLE_COLUMNS = ["set", "predictor", "kind", "n_tiles", "n_folds", "n_pcs_predictor", "n_pcs_expression",
                 "expression_pc_variance", "r2", "r2_pc1", "r2_ci_low", "r2_ci_high", "r2_fold_sd", "null_moves",
                 "r2_null_mean", "r2_null_p95", "p_null"]
PARTITION_COLUMNS = ["set", "view_a", "view_b", "r2_a", "r2_b", "r2_ab", "unique_a", "unique_b", "shared"]
DEFAULT_PAIRS = (("UNI2H_RGB", "LBP_RGB"), ("UNI2H_RGB_HE", "LBP_RGB_HE"), ("INTENSITY_HE", "LBP_RGB"),
                 ("UNI2H_RGB", "INTENSITY_HE"))
DEFAULT_GENE_VIEWS = ("UNI2H_RGB", "LBP_RGB", "INTENSITY_HE")
JOINT = "+"  # name of a joint predictor: "<view a>+<view b>"


def expression_genes(Y_genes_x_tiles, n_hvg: int) -> tuple[np.ndarray, np.ndarray]:
    """(tiles x genes z-scores, indices of the genes) of the ``n_hvg`` most variable genes,
    prepared as for the clustering (counts per 10 000, log1p, z-scores clipped at 10)."""
    from scipy import sparse

    from .cluster import highly_variable_genes, zscore

    Y = sparse.csr_matrix(sparse.csc_matrix(Y_genes_x_tiles).T, dtype=float)
    totals = np.asarray(Y.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    N = sparse.csr_matrix(sparse.diags(1e4 / totals) @ Y)
    N.data = np.log1p(N.data)
    idx = highly_variable_genes(N, n_hvg)
    return zscore(N[:, idx].toarray(), clip=10.0), idx


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


def _predict_genes(X: np.ndarray, G: np.ndarray, folds: np.ndarray, per_gene_alpha: bool) -> np.ndarray:
    from sklearn.linear_model import RidgeCV

    pred = np.zeros_like(G)
    for f in np.unique(folds):
        test = folds == f
        model = RidgeCV(alphas=R.ALPHAS, alpha_per_target=per_gene_alpha).fit(X[~test], G[~test])
        pred[test] = model.predict(X[test])
    return pred


def _r2_columns(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return R.r2_scores(Y, pred)[1]


def run(scores: dict, variance: dict, tiles: pd.DataFrame, views: list[str], set_name: str, mpp: float | None,
        genes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, gene_views=DEFAULT_GENE_VIEWS,
        pairs=DEFAULT_PAIRS, n_folds: int = 5, block_um: float = 1000.0, seed: int = 0, null_shifts: int = 20,
        n_boot: int = 200, gene_null_shifts: int = 20, log=None) -> dict:
    """``scores``: {view: tiles x components} including EXPRESSION; ``genes``: (tiles x genes
    z-scores, gene ids, gene names) or None. Returns {"table", "partition", "genes", "info"}
    or {"info"} with the reason it could not run."""
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
    moves = []
    if null_shifts > 0 and "row" in tiles and (tiles["row"].astype(int) >= 0).all():
        from .cluster import torus_pairs

        moves = torus_pairs(tiles["row"], tiles["col"], max(null_shifts, gene_null_shifts), seed)
    info["null"] = {"moves": len(moves)} if moves else {"moves": 0, "note": "not computed: tiles not on a grid"}
    E = scores["EXPRESSION"]

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
        null = (R.null_r2(X, E, folds, moves[:null_shifts]) if (moves and kind == "view" and null_shifts > 0)
                else np.array([], float))
        rows.append({"set": set_name, "predictor": name, "kind": kind, "n_tiles": int(len(tiles)),
                     "n_folds": info["n_folds"], "n_pcs_predictor": int(X.shape[1]), "n_pcs_expression": int(E.shape[1]),
                     "expression_pc_variance": float(variance.get("EXPRESSION", np.nan)), "r2": total,
                     "r2_pc1": float(per[0]) if per.size else np.nan,
                     "r2_ci_low": float(np.nanpercentile(boot, 2.5)) if boot.size else np.nan,
                     "r2_ci_high": float(np.nanpercentile(boot, 97.5)) if boot.size else np.nan,
                     "r2_fold_sd": float(np.nanstd(fr, ddof=1)) if len(fr) > 1 else np.nan,
                     "null_moves": int(null.size), "r2_null_mean": float(null.mean()) if null.size else np.nan,
                     "r2_null_p95": float(np.percentile(null, 95)) if null.size else np.nan,
                     "p_null": float((1 + (null >= total).sum()) / (1 + null.size)) if null.size else np.nan})
        if log:
            log.info("relevance: expression from %s, R2 %.3f", name, total)
    part = []
    for a, b in joint_pairs:
        ra, rb, rab = r2_of[a], r2_of[b], r2_of[f"{a}{JOINT}{b}"]
        part.append({"set": set_name, "view_a": a, "view_b": b, "r2_a": ra, "r2_b": rb, "r2_ab": rab,
                     "unique_a": rab - rb, "unique_b": rab - ra, "shared": ra + rb - rab})
    out = {"table": pd.DataFrame(rows, columns=TABLE_COLUMNS), "partition": pd.DataFrame(part, columns=PARTITION_COLUMNS),
           "info": info}

    if genes is not None and genes[0].shape[1] > 0:
        G, gene_ids, gene_names = genes
        gtab = pd.DataFrame({"gene_id": np.asarray(gene_ids, dtype=object), "gene_name": np.asarray(gene_names, dtype=object)})
        gviews = [v for v in gene_views if v in views]
        for v in gviews:
            pred = _predict_genes(scores[v], G, folds, per_gene_alpha=True)
            r2 = _r2_columns(G, pred)
            gtab[f"r2_{v}"] = r2
            if moves and gene_null_shifts > 0:
                nulls = []
                for i, j in moves[:gene_null_shifts]:
                    f = folds[i]
                    if len(np.unique(f)) < 2:
                        continue
                    nulls.append(_r2_columns(G[i], _predict_genes(scores[v][j], G[i], f, per_gene_alpha=False)))
                N = np.vstack(nulls) if nulls else np.empty((0, G.shape[1]))
                if len(N) >= 2:
                    from scipy.stats import norm

                    mu, sd = N.mean(axis=0), N.std(axis=0, ddof=1)
                    z = (r2 - mu) / np.where(sd > 0, sd, np.nan)
                    p = norm.sf(z)
                    gtab[f"r2_null_mean_{v}"], gtab[f"r2_null_sd_{v}"] = mu, sd
                    gtab[f"z_{v}"], gtab[f"p_{v}"], gtab[f"q_{v}"] = z, p, _bh(p)
            if log:
                log.info("relevance: %d genes from %s (median R2 %.3f)", G.shape[1], v, float(np.nanmedian(r2)))
        for a, b in (("UNI2H_RGB", "LBP_RGB"), ("INTENSITY_HE", "LBP_RGB")):
            if a in gviews and b in gviews:
                pred = _predict_genes(np.hstack([scores[a], scores[b]]), G, folds, per_gene_alpha=True)
                rab = _r2_columns(G, pred)
                gtab[f"r2_{a}{JOINT}{b}"] = rab
                gtab[f"adds_{b}_to_{a}"] = rab - gtab[f"r2_{a}"]
                gtab[f"adds_{a}_to_{b}"] = rab - gtab[f"r2_{b}"]
        out["genes"] = gtab
        info["genes"] = {"n": int(G.shape[1]), "views": gviews, "null_moves": int(min(len(moves), gene_null_shifts))}
    info["status"] = "computed"
    return out
