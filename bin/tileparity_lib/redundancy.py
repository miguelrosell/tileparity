"""Optional (``--redundancy``, with ``--cluster``): how much of each image view another view
predicts, as a measure of redundant and view-specific information.

For one tile set (``keep``) of one slide, every view is represented by the principal
components the clustering uses (same preparation, same PCA). For every ordered pair of
views (A, B), B's components are predicted from A's with ridge regression in a spatial
cross-validation, and

    R^2(B from A) = 1 - sum of squared prediction errors / sum of squares of B's components

is the share of B's variance (as far as its components keep it) that A predicts. It is not
symmetric, and the asymmetry is the point: if UNI2-h predicts most of LBP but LBP predicts
little of UNI2-h, LBP's information is largely contained in UNI2-h's, and what UNI2-h adds
is the part LBP cannot predict.

* Spatial cross-validation: the tissue is cut into square blocks (``block_um`` wide); every
  block goes to one of ``n_folds`` folds (folds balanced by number of tiles), and every tile
  is predicted by a model fitted on the other folds only. Neighbouring tiles look alike, so
  predicting a tile from a model that saw its neighbours would overstate what A knows
  about B.
* Ridge regression (scikit-learn ``RidgeCV``): the penalty is chosen on the training folds
  by efficient leave-one-out cross-validation, one value for all of B's components.
* Per component: the R^2 of each of B's components, to see whether what A predicts is B's
  main structure (its first components) or a minor part.
* Per tile: the part of B that A does not predict, relative to B's average variation,
  ``sum_j (b_ij - prediction_ij)^2 / mean_i sum_j b_ij^2``. Its mean over the tiles is
  1 - R^2; a tile at 3 has three times the average variation of B left unexplained. Maps of
  it show where on the slide the view-specific information lies (tissue, or artefacts).

R^2 is out-of-sample, so it can be slightly negative when A predicts nothing of B.

How much to trust an R^2 (classical p-values do not apply: neighbouring tiles are not
independent observations):

* Interval: the spatial blocks are resampled with replacement (block bootstrap of the
  out-of-fold predictions, ``n_boot`` times) and R^2 recomputed; the 2.5 and 97.5
  percentiles give a 95 % interval. The spread of R^2 between the folds is also reported.
* Spatial null: A's map is moved over B's on the tile grid with wrap-around (the moves of
  the clustering's null, :func:`tileparity_lib.cluster.torus_pairs`), so that A keeps its
  spatial structure but no longer describes the same tissue as B, and B is predicted again
  from the moved A with the same cross-validation. The R^2 of the moved maps is the level
  reached by chance; ``p_null`` = (1 + moves with R^2 >= the real one) / (1 + moves). Only
  for tiles on the grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHAS = np.logspace(-2, 5, 15)  # ridge penalties tried on the training folds
TABLE_COLUMNS = ["set", "predictor", "target", "n_tiles", "n_folds", "block_um", "n_pcs_predictor", "n_pcs_target",
                 "target_pc_variance", "r2", "r2_pc1", "r2_pcs1_5", "alpha_median", "r2_fold_min", "r2_fold_max",
                 "r2_fold_sd", "r2_ci_low", "r2_ci_high", "null_moves", "r2_null_mean", "r2_null_p95", "p_null"]


def tile_centres_um(tiles: pd.DataFrame, mpp: float | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Centres of the tiles in micrometres (in tile widths when the scale is unknown)."""
    x0 = tiles["x0_px"].to_numpy(float)
    y0 = tiles["y0_px"].to_numpy(float)
    T = tiles["tile_px"].to_numpy(float)
    if mpp and np.isfinite(mpp) and mpp > 0:
        return (x0 + T / 2) * mpp, (y0 + T / 2) * mpp, "um"
    return (x0 + T / 2) / T, (y0 + T / 2) / T, "tiles"


def spatial_blocks(x: np.ndarray, y: np.ndarray, block: float) -> np.ndarray:
    """Block (0 .. n - 1) of every tile: square blocks of side ``block`` (same unit as x, y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    bx = np.floor((x - x.min()) / block).astype(int)
    by = np.floor((y - y.min()) / block).astype(int)
    return pd.factorize(pd.Series(list(zip(bx.tolist(), by.tolist()))))[0]


def spatial_folds(x: np.ndarray, y: np.ndarray, block: float, n_folds: int, seed: int) -> np.ndarray | None:
    """Fold (0 .. n_folds - 1) of every tile: square blocks of side ``block`` (same unit as x
    and y), each block in one fold, blocks given in random order to the fold with the fewest
    tiles so far. None when there are fewer than 2 blocks."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 0:
        return None
    block_id = spatial_blocks(x, y, block)
    n_blocks = int(block_id.max()) + 1
    k = int(min(n_folds, n_blocks))
    if k < 2:
        return None
    sizes = np.bincount(block_id)
    load = np.zeros(k)
    fold_of_block = np.empty(n_blocks, int)
    for b in np.random.default_rng(seed).permutation(n_blocks):
        f = int(np.argmin(load))
        fold_of_block[b] = f
        load[f] += sizes[b]
    return fold_of_block[block_id]


def cross_predict(X: np.ndarray, Y: np.ndarray, folds: np.ndarray, alphas=ALPHAS) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold ridge predictions of Y (tiles x q) from X (tiles x p), and the penalty
    chosen in every fold."""
    from sklearn.linear_model import RidgeCV

    X, Y = np.asarray(X, float), np.asarray(Y, float)
    pred = np.zeros_like(Y)
    chosen = []
    for f in np.unique(folds):
        test = folds == f
        model = RidgeCV(alphas=alphas).fit(X[~test], Y[~test])
        pred[test] = model.predict(X[test])
        chosen.append(float(model.alpha_))
    return pred, chosen


def r2_scores(Y: np.ndarray, pred: np.ndarray) -> tuple[float, np.ndarray]:
    """(R^2 over all of Y's columns together, R^2 of every column); Y is centred, so its
    sum of squares is its variation around the mean."""
    Y = np.asarray(Y, float)
    Yc = Y - Y.mean(axis=0)
    ss_res = ((Y - pred) ** 2).sum(axis=0)
    ss_tot = (Yc**2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)
    total = float(1 - ss_res.sum() / ss_tot.sum()) if ss_tot.sum() > 0 else float("nan")
    return total, per


def fold_r2(Y: np.ndarray, pred: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """R^2 of every fold's tiles (sums of squares around the mean of all tiles)."""
    Y = np.asarray(Y, float)
    mu = Y.mean(axis=0)
    out = []
    for f in np.unique(folds):
        m = folds == f
        tot = ((Y[m] - mu) ** 2).sum()
        out.append(1 - ((Y[m] - pred[m]) ** 2).sum() / tot if tot > 0 else np.nan)
    return np.array(out, float)


def block_bootstrap_r2(Y: np.ndarray, pred: np.ndarray, blocks: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """R^2 of ``n_boot`` resamples of the spatial blocks (with replacement)."""
    Y = np.asarray(Y, float)
    mu = Y.mean(axis=0)
    res = ((Y - pred) ** 2).sum(axis=1)
    tot = ((Y - mu) ** 2).sum(axis=1)
    ids = np.unique(blocks)
    if n_boot <= 0 or len(ids) < 2:
        return np.array([], float)
    res_b = np.bincount(np.searchsorted(ids, blocks), weights=res, minlength=len(ids))
    tot_b = np.bincount(np.searchsorted(ids, blocks), weights=tot, minlength=len(ids))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    T = tot_b[draws].sum(axis=1)
    return np.where(T > 0, 1 - res_b[draws].sum(axis=1) / np.where(T > 0, T, 1), np.nan)


def null_r2(X: np.ndarray, Y: np.ndarray, folds: np.ndarray, moves, alphas=ALPHAS) -> np.ndarray:
    """R^2 of Y predicted from X moved over the tile grid: for every move (i, j), tile i of Y
    is paired with tile j of X; same folds as the real prediction (those of the Y tiles)."""
    out = []
    for i, j in moves:
        f = folds[i]
        if len(np.unique(f)) < 2:
            continue
        pred, _ = cross_predict(X[j], Y[i], f, alphas)
        out.append(r2_scores(Y[i], pred)[0])
    return np.array(out, float)


def unexplained_per_tile(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Squared prediction error of every tile relative to the average squared distance of a
    tile from B's mean (mean over the tiles = 1 - R^2)."""
    Y = np.asarray(Y, float)
    Yc = Y - Y.mean(axis=0)
    scale = float((Yc**2).sum(axis=1).mean())
    return ((Y - pred) ** 2).sum(axis=1) / scale if scale > 0 else np.full(Y.shape[0], np.nan)


def default_map_pairs(views: list[str]) -> list[tuple[str, str]]:
    """(predictor, target) pairs drawn on the slide: UNI2-h and LBP on identical pixels, both
    ways."""
    out = []
    for u, l in (("UNI2H_RGB", "LBP_RGB"), ("UNI2H_RGB_HE", "LBP_RGB_HE")):
        if u in views and l in views:
            out += [(u, l), (l, u)]
    return out


def run(scores: dict, variance: dict, tiles: pd.DataFrame, views: list[str], set_name: str, mpp: float | None,
        n_folds: int = 5, block_um: float = 1000.0, seed: int = 0, map_pairs=None, null_shifts: int = 20,
        n_boot: int = 200) -> dict:
    """Cross-prediction between every ordered pair of ``views`` of one tile set.

    ``scores``: {view: tiles x components}; ``variance``: {view: share of the view's
    variance its components keep}; ``tiles``: the set's rows of the tile table. Returns
    {"table": one row per pair, "pcs": one row per pair and target component, "tiles":
    per-tile unexplained share of the drawn pairs, "folds", "info"} or {"info"} with a
    reason when it could not run."""
    views = [v for v in views if v in scores and scores[v].shape[1] > 0]
    x, y, unit = tile_centres_um(tiles, mpp)
    block = block_um if unit == "um" else max(1.0, round(block_um / 112.0))
    folds = spatial_folds(x, y, block, n_folds, seed)
    info = {"views": views, "n_tiles": int(len(tiles)), "block": float(block), "block_unit": unit,
            "n_folds_requested": int(n_folds)}
    if len(views) < 2:
        info["status"] = "not computed: fewer than 2 views"
        return {"info": info}
    if folds is None:
        info["status"] = f"not computed: the tiles fit in fewer than 2 blocks of {block:g} {unit}"
        return {"info": info}
    info["n_folds"] = int(folds.max()) + 1
    info["fold_sizes"] = np.bincount(folds).tolist()
    blocks = spatial_blocks(x, y, block)
    moves = []
    if null_shifts > 0 and "row" in tiles and (tiles["row"].astype(int) >= 0).all():
        from .cluster import torus_pairs

        moves = torus_pairs(tiles["row"], tiles["col"], null_shifts, seed)
    info["null"] = ({"moves": len(moves), "min_shift": 0.2, "min_overlap": 0.3} if moves else
                    {"moves": 0, "note": "not computed: tiles not on a grid, or no possible move"})
    info["bootstrap"] = {"resamples": int(n_boot), "unit": "spatial block", "n_blocks": int(blocks.max()) + 1}
    pairs = [(a, b) for a in views for b in views if a != b]
    drawn = set(map_pairs if map_pairs is not None else default_map_pairs(views))
    rows, pc_rows, per_tile = [], [], {}
    for a, b in pairs:
        pred, alphas = cross_predict(scores[a], scores[b], folds)
        total, per = r2_scores(scores[b], pred)
        var = (scores[b] - scores[b].mean(axis=0)).var(axis=0)
        share = var / var.sum() if var.sum() > 0 else np.full(var.size, np.nan)
        top5 = slice(0, min(5, per.size))
        r2_top5 = float(np.nansum(per[top5] * share[top5]) / np.nansum(share[top5])) if per.size else float("nan")
        fr = fold_r2(scores[b], pred, folds)
        boot = block_bootstrap_r2(scores[b], pred, blocks, n_boot, seed)
        null = null_r2(scores[a], scores[b], folds, moves) if moves else np.array([], float)
        rows.append({"set": set_name, "predictor": a, "target": b, "n_tiles": int(len(tiles)),
                     "n_folds": info["n_folds"], "block_um": float(block_um) if unit == "um" else float("nan"),
                     "n_pcs_predictor": int(scores[a].shape[1]), "n_pcs_target": int(scores[b].shape[1]),
                     "target_pc_variance": float(variance.get(b, float("nan"))), "r2": total,
                     "r2_pc1": float(per[0]) if per.size else float("nan"), "r2_pcs1_5": r2_top5,
                     "alpha_median": float(np.median(alphas)),
                     "r2_fold_min": float(np.nanmin(fr)), "r2_fold_max": float(np.nanmax(fr)),
                     "r2_fold_sd": float(np.nanstd(fr, ddof=1)) if len(fr) > 1 else float("nan"),
                     "r2_ci_low": float(np.nanpercentile(boot, 2.5)) if boot.size else float("nan"),
                     "r2_ci_high": float(np.nanpercentile(boot, 97.5)) if boot.size else float("nan"),
                     "null_moves": int(null.size),
                     "r2_null_mean": float(null.mean()) if null.size else float("nan"),
                     "r2_null_p95": float(np.percentile(null, 95)) if null.size else float("nan"),
                     "p_null": float((1 + (null >= total).sum()) / (1 + null.size)) if null.size else float("nan")})
        for j in range(per.size):
            pc_rows.append({"set": set_name, "predictor": a, "target": b, "pc": j + 1,
                            "variance_share": float(share[j]), "r2": float(per[j])})
        if (a, b) in drawn:
            per_tile[(a, b)] = unexplained_per_tile(scores[b], pred)
    info["status"] = "computed"
    info["map_pairs"] = [list(p) for p in per_tile]
    tile_table = pd.DataFrame({"tile_id": tiles["tile_id"].astype(str).to_numpy(), "fold": folds})
    for (a, b), u in per_tile.items():
        tile_table[f"unexplained_{b}_from_{a}"] = u
    return {"table": pd.DataFrame(rows, columns=TABLE_COLUMNS), "pcs": pd.DataFrame(pc_rows), "tiles": tile_table,
            "per_tile": per_tile, "folds": folds, "info": info}
