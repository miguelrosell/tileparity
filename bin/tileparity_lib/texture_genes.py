"""Optional (``--texture-genes``, with ``--cluster``): which named texture features go with
which genes, tile by tile.

Texture features. In an LBP histogram (one per channel and radius R, P points on the
circle, see :mod:`tileparity_lib.lbp`) every pixel has one code: the number of
neighbours at least as bright as the pixel when they form one unbroken arc, else P + 1.
The codes fall into four types:

* ``spot_bright`` (code 0): every neighbour is darker than the pixel; a bright dot of
  about R pixels (e.g. a gap or lumen smaller than the circle).
* ``spot_dark_or_flat`` (code P): no neighbour is darker; a dark dot (e.g. a nucleus of
  about the circle's size) or a flat area.
* ``edge`` (codes 1 .. P - 1): the brighter neighbours form one arc; a border, line end or
  corner at that scale.
* ``irregular`` (code P + 1): brighter and darker neighbours alternate; fine texture
  that is not a spot or an edge at that scale.

A feature is the fraction of a tile's pixels of one type at one radius: 4 types x 13 radii
= 52 features, from the channels of one LBP view (by default their mean; or one channel).

Association. For every feature and every one of the ``n_genes`` most variable genes
(log1p counts per 10 000): the partial Spearman correlation over the tiles of the set
``keep`` with the tile's depth (log10 molecules) removed: features, genes and depth are
turned into ranks, the ranks of the feature and of the gene are regressed on the rank of
depth (with an intercept), and the residuals are correlated (Pearson).

Test. Neighbouring tiles are alike, so the tiles are not independent and a classical test
of the correlation would be far too optimistic. The modified t-test of Clifford,
Richardson & Hemon (1989; Dutilleul 1993) replaces the number of tiles n with an effective
number n_eff = 1 + n^2 / tr(R_x R_y), where R_x and R_y are the spatial correlation
matrices of the two residuals estimated from their correlograms (distance classes one tile
wide, on the tile grid). With r the correlation, t = r sqrt((n_eff - 2) / (1 - r^2)) on
n_eff - 2 degrees of freedom (two-sided). Benjamini-Hochberg q-values over every feature
and gene of the slide.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TYPES = ("spot_bright", "spot_dark_or_flat", "edge", "irregular")
COLUMN = re.compile(r"^(?P<channel>.+)_r(?P<r>\d+)_p(?P<p>\d+)_u(?P<u>\d+)$")
COLUMNS = ["feature", "type", "radius_px", "gene_id", "gene_name", "rho", "n_tiles", "n_eff", "p", "q"]


def features(X: np.ndarray, cols, channel: str = "mean") -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """(tiles x features fractions, feature names, table of the features) of one LBP view
    (histogram columns ``<channel>_r<R>_p<P>_u<code>``). ``channel``: 'mean' (mean of the
    fractions over the view's channels) or one channel name."""
    parsed = [COLUMN.match(str(c)) for c in cols]
    if not parsed or any(m is None for m in parsed):
        raise ValueError("not LBP histogram columns (<channel>_r<R>_p<P>_u<code>)")
    meta = pd.DataFrame([{"channel": m["channel"], "r": int(m["r"]), "p": int(m["p"]), "u": int(m["u"])} for m in parsed])
    channels = list(dict.fromkeys(meta["channel"]))
    use = channels if channel == "mean" else [channel]
    missing = [c for c in use if c not in channels]
    if missing:
        raise ValueError(f"channel {missing[0]} not in the view (channels: {', '.join(channels)})")
    X = np.asarray(X, float)
    radii = sorted(meta["r"].unique())
    out, names, rows = [], [], []
    for t in TYPES:
        for r in radii:
            fr = []
            for c in use:
                sel = (meta["channel"] == c) & (meta["r"] == r)
                if not sel.any():
                    continue
                p = int(meta.loc[sel, "p"].iloc[0])
                u = meta.loc[sel, "u"].to_numpy()
                block = X[:, sel.to_numpy()]
                total = block.sum(axis=1)
                total[total == 0] = 1.0
                pick = {"spot_bright": u == 0, "spot_dark_or_flat": u == p, "edge": (u >= 1) & (u <= p - 1),
                        "irregular": u == p + 1}[t]
                fr.append(block[:, pick].sum(axis=1) / total)
            if fr:
                out.append(np.mean(fr, axis=0))
                names.append(f"{t}_r{r}")
                rows.append({"feature": names[-1], "type": t, "radius_px": int(r), "channels": "+".join(use)})
    return np.column_stack(out) if out else np.zeros((len(X), 0)), names, pd.DataFrame(rows)


def _ranks(X: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(X, axis=0, method="average")


def residual_ranks(V: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Ranks of every column of ``V`` with the rank of ``covariate`` regressed out (OLS with
    an intercept)."""
    Rv = _ranks(np.asarray(V, float))
    c = _ranks(np.asarray(covariate, float).reshape(-1, 1)).ravel()
    A = np.column_stack([np.ones_like(c), c])
    coef, *_ = np.linalg.lstsq(A, Rv, rcond=None)
    return Rv - A @ coef


def correlograms(V: np.ndarray, rows, cols) -> tuple[np.ndarray, np.ndarray]:
    """Spatial correlograms on the tile grid: (variables x classes correlations, pairs per
    class). Class 0 = the tile itself (correlation 1, n pairs); class k = ordered pairs of
    tiles at a distance in (k - 1, k] tile sides. Computed for all lags at once with FFTs of
    the grid (no wrap-around)."""
    V = np.asarray(V, float)
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    r, c = rows - rows.min(), cols - cols.min()
    H, W = int(r.max()) + 1, int(c.max()) + 1
    Hp, Wp = 2 * H, 2 * W
    dy = np.fft.fftfreq(Hp, 1 / Hp)
    dx = np.fft.fftfreq(Wp, 1 / Wp)
    dist = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
    cls = np.ceil(dist).astype(int).ravel()
    K = int(cls.max()) + 1

    def acf(values: np.ndarray) -> np.ndarray:  # values: n x m -> m x lags
        g = np.zeros((values.shape[1], Hp, Wp))
        g[:, r, c] = values.T
        f = np.fft.rfft2(g)
        return np.fft.irfft2(f * np.conj(f), s=(Hp, Wp)).reshape(values.shape[1], -1)

    pairs_lag = np.rint(acf(np.ones((len(r), 1)))[0])
    pairs = np.bincount(cls, weights=pairs_lag, minlength=K)
    Vc = V - V.mean(axis=0)
    var = (Vc**2).mean(axis=0)
    var[var == 0] = np.nan
    out = np.zeros((V.shape[1], K))
    for start in range(0, V.shape[1], 64):
        A = acf(Vc[:, start:start + 64])
        S = np.vstack([np.bincount(cls, weights=a, minlength=K) for a in A])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:start + 64] = S / np.where(pairs > 0, pairs, np.nan)[None, :] / var[start:start + 64, None]
    out[:, pairs == 0] = 0.0
    out[:, 0] = 1.0
    return np.nan_to_num(out), pairs


def effective_n(rho_x: np.ndarray, rho_y: np.ndarray, pairs: np.ndarray, n: int) -> np.ndarray:
    """n_eff = 1 + n^2 / tr(R_x R_y) for every pair of variables (x rows, y rows), with
    tr(R_x R_y) = sum over classes of pairs x rho_x x rho_y; clipped to [3, n]."""
    tr = (rho_x * pairs[None, :]) @ rho_y.T
    with np.errstate(divide="ignore", invalid="ignore"):
        ne = 1.0 + n**2 / tr
    return np.clip(np.where(np.isfinite(ne) & (tr > 0), ne, n), 3.0, float(n))


def run(F: np.ndarray, feature_names: list[str], feature_table: pd.DataFrame, G: np.ndarray, gene_ids, gene_names,
        depth: np.ndarray, rows, cols) -> pd.DataFrame:
    """Partial Spearman correlation and modified t-test for every feature and gene (tiles
    in rows of ``F``, ``G``)."""
    from scipy.stats import t as tdist

    from .relevance import _bh

    n = len(F)
    if n < 10 or F.shape[1] == 0 or G.shape[1] == 0:
        return pd.DataFrame(columns=COLUMNS)
    Fr = residual_ranks(F, depth)
    Gr = residual_ranks(G, depth)
    Fn = np.linalg.norm(Fr, axis=0)
    Gn = np.linalg.norm(Gr, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (Fr.T @ Gr) / np.outer(np.where(Fn > 0, Fn, np.nan), np.where(Gn > 0, Gn, np.nan))
    rho = np.clip(rho, -1.0, 1.0)
    cf, pairs = correlograms(Fr, rows, cols)
    cg, _ = correlograms(Gr, rows, cols)
    ne = effective_n(cf, cg, pairs, n)
    df = ne - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        tval = rho * np.sqrt(df / np.maximum(1 - rho**2, 1e-12))
    p = 2 * tdist.sf(np.abs(tval), df)
    q = _bh(p.ravel()).reshape(p.shape)
    ft = feature_table.set_index("feature")
    nf, ng = F.shape[1], G.shape[1]
    return pd.DataFrame({
        "feature": np.repeat(np.asarray(feature_names, dtype=object), ng),
        "type": np.repeat(ft.loc[feature_names, "type"].to_numpy(dtype=object), ng),
        "radius_px": np.repeat(ft.loc[feature_names, "radius_px"].to_numpy(int), ng),
        "gene_id": np.tile(np.asarray(gene_ids, dtype=object), nf),
        "gene_name": np.tile(np.asarray(gene_names, dtype=object), nf),
        "rho": rho.ravel(), "n_tiles": n, "n_eff": ne.ravel(), "p": p.ravel(), "q": q.ravel()})[COLUMNS]


def summary(table: pd.DataFrame, level: float = 0.05) -> pd.DataFrame:
    """Per feature: genes with q < level, positive and negative, and the strongest gene."""
    if table.empty:
        return pd.DataFrame(columns=["feature", "type", "radius_px", "n_genes", "n_sig", "n_sig_pos", "n_sig_neg",
                                     "top_gene", "top_rho", "median_n_eff"])
    rows = []
    for (f, t, r), g in table.groupby(["feature", "type", "radius_px"], sort=False):
        sig = g["q"] < level
        top = g.loc[g["rho"].abs().idxmax()]
        rows.append({"feature": f, "type": t, "radius_px": int(r), "n_genes": len(g), "n_sig": int(sig.sum()),
                     "n_sig_pos": int((sig & (g["rho"] > 0)).sum()), "n_sig_neg": int((sig & (g["rho"] < 0)).sum()),
                     "top_gene": top["gene_name"], "top_rho": float(top["rho"]), "median_n_eff": float(g["n_eff"].median())})
    return pd.DataFrame(rows)
