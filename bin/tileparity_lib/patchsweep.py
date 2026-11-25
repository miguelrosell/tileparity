"""Optional (``--patch_sweep``, Visium HD only): how the patch size of fastLBP changes what
LBP knows about the expression, measured on a near-continuous range of sizes without
running fastLBP again for every size.

1. Cells. Every tile of the set ``keep`` is cut into square cells of ``cell_um`` (default
   8 µm = 16 px at 0.5 µm/px, the Visium HD bin). LBP codes are computed once per tile,
   with the tile's real neighbourhood (as in Process 2b), and counted per cell: one
   histogram per cell, channel and radius. Every 8 µm bin of expression goes to the cell
   that contains its centre.
2. Patches. A patch of side s = n x ``cell_um`` is a square of n x n cells on the cell grid
   (patches do not overlap; only patches whose cells all lie in kept tiles are used). Its
   LBP histogram is the sum of its cells' histograms (histograms add up) and its expression
   the sum of its cells' molecules, so every size comes from the same computation.
3. What LBP knows about the expression at size s: the expression's principal components
   (counts per 10 000, log1p, most variable genes, z-scores, PCA) predicted from LBP's
   (proportions, square root, equal block weights, PCA) with ridge regression and spatial
   block cross-validation (as ``--relevance``). Radii larger than a quarter of the patch
   are left out; every radius is also used alone.
4. Noise ceiling. Small patches hold few molecules, so their expression is noisy and any
   predictor does worse. Every patch's molecules are split at random into two halves
   (binomial thinning): given the tissue, two independent samples of it. Half B is prepared
   like the expression, half A on the same genes and projected on half B's component axes;
   the variance the halves share along those axes, over half B's variance, is the share of
   half B that is not sampling noise (``r2_ceiling``: the most any predictor that does not
   see half B's molecules can reach; see :func:`tileparity_lib.relevance.split_half_components`).
   ``fraction`` = R^2(half B from LBP) / r2_ceiling is the share of the reachable that LBP
   reaches; it does not depend on the number of molecules, so it is comparable between sizes.
   (Until 0.7.0 the ceiling was the R^2 of half B predicted from half A, which counts half
   A's noise against the ceiling and inflated the fraction where molecules are few.)
5. Gene by gene (0.7.1): one fixed list of genes, the most variable at a reference size
   (``--gene-size-um``, the tile of the main analysis), at every size: each gene of half B
   predicted from LBP (ridge, one penalty per gene) and its own ceiling, the correlation of
   the gene between the halves. ``fraction_genes`` = sum of the genes' R^2 / sum of their
   ceilings (the share of the genes' reachable variance that LBP predicts); per gene, its
   R^2, ceiling and fraction (only where the ceiling is at least 0.1).
6. Agreement: ARI between k-means clusters of LBP and of the expression at every size.

The sweet spot reported is the smallest size whose ``fraction`` is within one standard
error (over the folds) of the largest ``fraction``.
"""

from __future__ import annotations

import math
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import sparse

from . import lbp as L
from . import redundancy as R

TABLE_COLUMNS = ["size_um", "cells_per_side", "patch_px", "n_patches", "n_radii", "max_radius_px", "molecules_median",
                 "r2_full", "r2_full_ci_low", "r2_full_ci_high", "r2_half", "r2_ceiling", "fraction", "fraction_se",
                 "n_folds", "n_genes", "r2_genes_mean", "ceiling_genes_mean", "fraction_genes"]
GENE_COLUMNS = ["size_um", "gene_id", "gene_name", "r2_half", "ceiling", "fraction"]
MIN_GENE_CEILING = 0.1  # a gene's fraction is reported only where its ceiling reaches this


def parse_sizes(text: str, cell_um: float) -> list[int]:
    """Patch sides in µm from 'start:end:step' or a comma list, kept when a multiple of the
    cell and at least two cells."""
    text = str(text).strip()
    if ":" in text:
        a, b, c = (int(float(x)) for x in text.split(":"))
        sizes = list(range(a, b + 1, c))
    else:
        sizes = [int(float(x)) for x in text.split(",") if x.strip()]
    return sorted({s for s in sizes if s >= 2 * cell_um and abs(s / cell_um - round(s / cell_um)) < 1e-6})


# ---- 1. LBP histograms per cell -------------------------------------------------------------------------------
def _cells_batch(batch: list[tuple[int, int, int]]) -> list[tuple[int, np.ndarray]]:
    from fastlbp.lbp import uniform_lbp_uint8_padded_absolute

    from .image import read_region

    img, spec = L._worker_state["img"], L._worker_state["spec"]
    cell = L._worker_state["cell_px"]
    pad, T = spec.pad, spec.tile_px
    k = T // cell
    out = []
    for idx, x0, y0 in batch:
        crop = read_region(img, x0 - pad, y0 - pad, T + 2 * pad, T + 2 * pad)
        parts = []
        for c in spec.channels:
            band = crop[:, :, spec.bands.index(c)]
            for r, p in zip(spec.radii, spec.npoints):
                m = int(math.ceil(r)) + 1
                sub = np.ascontiguousarray(band[pad - m: pad + T + m, pad - m: pad + T + m])
                codes = uniform_lbp_uint8_padded_absolute(sub, int(p), float(r), int(y0), int(x0), [m, m, m, m])
                cells = codes.reshape(k, cell, k, cell).transpose(0, 2, 1, 3).reshape(k * k, cell * cell).astype(np.int64)
                flat = (cells + (np.arange(k * k)[:, None] * (p + 2))).ravel()
                parts.append(np.bincount(flat, minlength=k * k * (p + 2)).reshape(k * k, p + 2).astype(np.uint16))
        out.append((idx, np.hstack(parts)))
    return out


def cell_histograms(channels_path: str, tiles: pd.DataFrame, spec: "L.LbpSpec", cell_px: int, ncpus: int = 1,
                    batch_size: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cells x bins uint16 histograms, cell row, cell column) for every tile of ``tiles``;
    cells are numbered tile by tile, row-major inside a tile; rows and columns count cells
    from the top-left tile of the set."""
    T = spec.tile_px
    if T % cell_px:
        raise ValueError(f"the tile ({T} px) is not a whole number of cells of {cell_px} px")
    k = T // cell_px
    n = len(tiles)
    nb = len(spec.channels) * spec.n_features_per_channel
    H = np.zeros((n * k * k, nb), np.uint16)
    x0, y0 = tiles["x0_px"].astype(int).to_numpy(), tiles["y0_px"].astype(int).to_numpy()
    jobs = list(zip(range(n), x0, y0))
    batches = [jobs[i: i + batch_size] for i in range(0, n, batch_size)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, ncpus), initializer=_init, initargs=(channels_path, spec, cell_px)) as pool:
        for results in pool.imap_unordered(_cells_batch, batches):
            for idx, h in results:
                H[idx * k * k:(idx + 1) * k * k] = h
    ii, jj = np.divmod(np.arange(k * k), k)
    cr = (((y0 - y0.min()) // cell_px)[:, None] + ii[None, :]).ravel()
    cc = (((x0 - x0.min()) // cell_px)[:, None] + jj[None, :]).ravel()
    return H, cr, cc


def _init(channels_path: str, spec, cell_px: int) -> None:
    L._init_worker(channels_path, spec)
    L._worker_state["cell_px"] = cell_px


# ---- 2. expression per cell --------------------------------------------------------------------------------------
def cell_counts(x_px: np.ndarray, y_px: np.ndarray, counts_genes_x_units, cr: np.ndarray, cc: np.ndarray,
                origin: tuple[int, int], cell_px: int) -> sparse.csr_matrix:
    """Cells x genes molecule counts: every unit (bin) goes to the cell containing its centre
    (x, y in pixels of the resampled image); units outside the cells are dropped."""
    gr = np.floor((np.asarray(y_px) - origin[1]) / cell_px).astype(np.int64)
    gc = np.floor((np.asarray(x_px) - origin[0]) / cell_px).astype(np.int64)
    lookup = pd.Series(np.arange(len(cr)), index=pd.MultiIndex.from_arrays([cr, cc]))
    key = pd.MultiIndex.from_arrays([gr, gc])
    pos = lookup.reindex(key).to_numpy()
    ok = np.isfinite(pos)
    A = sparse.csr_matrix((np.ones(int(ok.sum())), (pos[ok].astype(int), np.flatnonzero(ok))),
                          shape=(len(cr), len(gr)))
    return sparse.csr_matrix(A @ sparse.csc_matrix(counts_genes_x_units).T)


# ---- 3-5. one patch size ------------------------------------------------------------------------------------------------
def patches(cr: np.ndarray, cc: np.ndarray, n: int) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """(patches x cells indicator of the complete n x n patches, patch row, patch column)."""
    pr, pc = cr // n, cc // n
    key = pd.Series(list(zip(pr.tolist(), pc.tolist())))
    codes, uniques = pd.factorize(key)
    full = np.bincount(codes) == n * n
    keep = full[codes]
    remap = -np.ones(len(uniques), int)
    remap[np.flatnonzero(full)] = np.arange(int(full.sum()))
    rows = remap[codes[keep]]
    M = sparse.csr_matrix((np.ones(int(keep.sum())), (rows, np.flatnonzero(keep))), shape=(int(full.sum()), len(cr)))
    up = np.array([uniques[i] for i in np.flatnonzero(full)]).reshape(-1, 2) if full.any() else np.zeros((0, 2), int)
    return M, up[:, 0], up[:, 1]


def lbp_features(Hp: np.ndarray, spec: "L.LbpSpec", radii_keep: list[int]) -> tuple[np.ndarray, dict]:
    """Patch histograms -> proportions, square root, equal block weights (per channel and
    radius); also {radius: columns} to use one radius alone."""
    blocks, offset, by_radius = [], 0, {}
    col = 0
    for _c in spec.channels:
        for r, p in zip(spec.radii, spec.npoints):
            if r in radii_keep:
                X = Hp[:, offset: offset + p + 2].astype(np.float64)
                tot = X.sum(axis=1, keepdims=True)
                S = np.sqrt(X / np.where(tot > 0, tot, 1))
                S -= S.mean(axis=0)
                lam = np.linalg.eigvalsh(np.cov(S.T, bias=True))[-1] if S.shape[0] > 1 else 1.0
                S /= math.sqrt(lam) if lam > 0 else 1.0
                blocks.append(S)
                by_radius.setdefault(r, []).extend(range(col, col + p + 2))
                col += p + 2
            offset += p + 2
    return (np.hstack(blocks) if blocks else np.zeros((len(Hp), 0))), by_radius


def _pcs(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    from .cluster import pca

    return pca(X, n, seed)[0]


def gene_list(Yc: sparse.csr_matrix, cr, cc, n_ref: int, n_genes: int) -> np.ndarray:
    """Indices of the ``n_genes`` most variable genes of the patches of ``n_ref`` cells a side
    (the list used at every size)."""
    from .cluster import highly_variable_genes
    from .relevance import normalise

    M, _, _ = patches(cr, cc, n_ref)
    if M.shape[0] < 20 or n_genes <= 0:
        return np.zeros(0, int)
    return np.asarray(highly_variable_genes(normalise(M @ Yc), n_genes), int)


def genes_at_size(A: sparse.csr_matrix, B: sparse.csr_matrix, idx: np.ndarray, P_lbp: np.ndarray,
                  folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R^2 of every gene of half B from LBP, its ceiling = shared / total variance of the gene
    in the two halves); NaN for a gene without variance in half B."""
    from .relevance import _r2_columns, ridge_gcv_predict, zscore_genes

    zA, zB = zscore_genes(A, idx), zscore_genes(B, idx)
    zA -= zA.mean(axis=0)
    zB -= zB.mean(axis=0)
    tot = (zB**2).sum(axis=0)
    ceiling = np.where(tot > 0, (zA * zB).sum(axis=0) / np.where(tot > 0, tot, 1.0), np.nan)
    r2 = _r2_columns(zB, ridge_gcv_predict(P_lbp, zB, folds))
    return r2, ceiling


def one_size(Hc: np.ndarray, Yc: sparse.csr_matrix, cr, cc, n: int, cell_px: int, mpp: float, spec, max_radius: int,
             n_pcs: int, n_hvg: int, n_folds: int, block_um: float, seed: int, ks, max_patches: int,
             n_boot: int, gene_idx=None, gene_ids=None, gene_names=None) -> tuple[dict, list[dict], list[dict]]:
    from .cluster import ari, kmeans
    from .relevance import ceiling_of, expression_genes, split_half_components, zscore_genes

    M, pr, pc = patches(cr, cc, n)
    patch_px = n * cell_px
    radii = [r for r in spec.radii if r <= max_radius and r <= patch_px / 4]
    row = {"size_um": round(n * cell_px * mpp, 6), "cells_per_side": n, "patch_px": patch_px,
           "n_patches": int(M.shape[0]), "n_radii": len(radii), "max_radius_px": max(radii) if radii else np.nan}
    rng = np.random.default_rng(seed + n)
    if M.shape[0] > max_patches:
        pick = np.sort(rng.choice(M.shape[0], max_patches, replace=False))
        M, pr, pc = M[pick], pr[pick], pc[pick]
    if M.shape[0] < 20 or not radii:
        return row, [], []
    Hp = np.asarray(M @ Hc.astype(np.float32))
    Yp = sparse.csr_matrix(M @ Yc)
    Yp.data = np.round(Yp.data)
    row["molecules_median"] = float(np.median(np.asarray(Yp.sum(axis=1)).ravel()))
    X, by_radius = lbp_features(Hp, spec, radii)
    P_lbp = _pcs(X, n_pcs, seed)
    Z, _idx = expression_genes(Yp.T, n_hvg)
    E = _pcs(Z.astype(np.float32), n_pcs, seed)
    x_um = (pc + 0.5) * patch_px * mpp
    y_um = (pr + 0.5) * patch_px * mpp
    folds = R.spatial_folds(x_um, y_um, block_um, n_folds, seed)
    if folds is None:
        return row, [], []
    blocks = R.spatial_blocks(x_um, y_um, block_um)
    row["n_folds"] = int(folds.max()) + 1
    pred, _ = R.cross_predict(P_lbp, E, folds)
    row["r2_full"] = R.r2_scores(E, pred)[0]
    boot = R.block_bootstrap_r2(E, pred, blocks, n_boot, seed)
    row["r2_full_ci_low"] = float(np.nanpercentile(boot, 2.5)) if boot.size else np.nan
    row["r2_full_ci_high"] = float(np.nanpercentile(boot, 97.5)) if boot.size else np.nan
    # split halves: half B predicted from LBP; its ceiling from the variance the halves share
    A = Yp.copy()
    A.data = rng.binomial(A.data.astype(np.int64), 0.5).astype(float)
    B = Yp - A
    A.eliminate_zeros()
    B.eliminate_zeros()
    ZB, idx_b = expression_genes(sparse.csr_matrix(B).T, n_hvg)
    ZA = zscore_genes(A, idx_b)
    EB, shared, total = split_half_components(ZA, ZB, n_pcs, seed)
    ceiling = ceiling_of(shared, total)
    pl, _ = R.cross_predict(P_lbp, EB, folds)
    row["r2_half"] = R.r2_scores(EB, pl)[0]
    row["r2_ceiling"] = ceiling
    ok_c = np.isfinite(ceiling) and ceiling > 0
    row["fraction"] = row["r2_half"] / ceiling if ok_c else np.nan
    frac = R.fold_r2(EB, pl, folds) / ceiling if ok_c else np.array([np.nan])
    row["fraction_se"] = (float(np.nanstd(frac, ddof=1) / math.sqrt(np.isfinite(frac).sum()))
                          if np.isfinite(frac).sum() > 1 else np.nan)
    for k in ks:
        if M.shape[0] >= 2 * k:
            row[f"ari_k{k}"] = ari(kmeans(P_lbp, k, seed), kmeans(E, k, seed))
    rad_rows = []
    for r, cols in by_radius.items():
        Pr = _pcs(X[:, cols], min(n_pcs, len(cols)), seed)
        pr_, _ = R.cross_predict(Pr, EB, folds)
        r2r = R.r2_scores(EB, pr_)[0]
        rad_rows.append({"size_um": row["size_um"], "radius_px": r, "r2_half": r2r,
                         "fraction": r2r / ceiling if ok_c else np.nan})
    gene_rows = []
    if gene_idx is not None and len(gene_idx):
        r2g, ceil_g = genes_at_size(A, B, gene_idx, P_lbp, folds)
        ok = np.isfinite(ceil_g)
        r2_sum, ceil_sum = float(np.nansum(np.where(ok, r2g, 0.0))), float(np.nansum(np.where(ok, ceil_g, 0.0)))
        row.update({"n_genes": int(len(gene_idx)), "r2_genes_mean": r2_sum / len(gene_idx),
                    "ceiling_genes_mean": ceil_sum / len(gene_idx),
                    "fraction_genes": r2_sum / ceil_sum if ceil_sum > 0 else np.nan})
        for g in range(len(gene_idx)):
            c = ceil_g[g]
            gene_rows.append({"size_um": row["size_um"], "gene_id": gene_ids[g] if gene_ids is not None else gene_idx[g],
                              "gene_name": gene_names[g] if gene_names is not None else "", "r2_half": r2g[g],
                              "ceiling": c, "fraction": r2g[g] / c if np.isfinite(c) and c >= MIN_GENE_CEILING else np.nan})
    return row, rad_rows, gene_rows


def small_vs_large(genes: pd.DataFrame, small_max_um: float = 32.0, large_min_um: float = 176.0) -> pd.DataFrame:
    """Per gene: mean fraction over the small sizes (<= ``small_max_um``) and over the large
    ones (>= ``large_min_um``), for genes whose ceiling reaches the minimum at every small
    size; ``difference`` = small - large (positive: LBP predicts the gene better, relative to
    what its molecules allow, with small patches)."""
    if genes is None or genes.empty:
        return pd.DataFrame(columns=["gene_id", "gene_name", "fraction_small", "fraction_large", "difference",
                                     "ceiling_small"])
    small = genes[genes["size_um"] <= small_max_um]
    large = genes[genes["size_um"] >= large_min_um]
    if small.empty or large.empty:
        return pd.DataFrame(columns=["gene_id", "gene_name", "fraction_small", "fraction_large", "difference",
                                     "ceiling_small"])
    measured = small.groupby("gene_id")["fraction"].apply(lambda f: f.notna().all())
    s = small.groupby(["gene_id", "gene_name"]).agg(fraction_small=("fraction", "mean"), ceiling_small=("ceiling", "mean"))
    lg = large.groupby(["gene_id", "gene_name"]).agg(fraction_large=("fraction", "mean"))
    out = s.join(lg).reset_index()
    out = out[out["gene_id"].map(measured).fillna(False).astype(bool)]
    out["difference"] = out["fraction_small"] - out["fraction_large"]
    return out.sort_values("difference", ascending=False)[["gene_id", "gene_name", "fraction_small", "fraction_large",
                                                            "difference", "ceiling_small"]]


def sweet_spot(table: pd.DataFrame) -> dict:
    """The smallest size whose fraction is within one standard error of the largest."""
    t = table.dropna(subset=["fraction"])
    if t.empty:
        return {}
    best = t.loc[t["fraction"].idxmax()]
    se = float(best["fraction_se"]) if np.isfinite(best.get("fraction_se", np.nan)) else 0.0
    ok = t[t["fraction"] >= best["fraction"] - se].sort_values("size_um")
    pick = ok.iloc[0]
    return {"best_size_um": float(best["size_um"]), "best_fraction": float(best["fraction"]),
            "sweet_spot_um": float(pick["size_um"]), "sweet_spot_fraction": float(pick["fraction"]),
            "rule": "smallest size with fraction >= the largest fraction - its standard error over the folds"}
