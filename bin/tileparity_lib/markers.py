"""Optional (``--markers``, with ``--cluster``): the genes that mark every expression domain.

The expression domains are the clusters of the view EXPRESSION (one partition per method
and k). For every domain d and every gene detected in at least ``min_detected`` of the
set's tiles:

* AUROC: the probability that a random tile of d has more of the gene (log1p counts per
  10 000) than a random tile outside d, ties counted as one half. Computed from ranks
  (Mann-Whitney U / (n_in n_out)). 0.5: no difference; 1: every tile of d above every
  other tile.
* log2FC = log2((mean in d + 0.1) / (mean outside d + 0.1)), means of counts per 10 000.
* pct_in, pct_out: share of tiles with at least one molecule of the gene, in d and outside.

Markers are the genes with log2FC > 0 and pct_in >= ``min_pct_in``, ranked by AUROC; the
``n_top`` best of every domain are kept. These are descriptions of the domains, not tests:
the tiles of a domain are neighbours of each other and the domains were found from these
same genes, so no p-value is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["set", "method", "k", "domain", "n_in", "n_out", "rank", "gene_id", "gene_name", "auroc", "log2fc", "pct_in",
           "pct_out", "mean_in", "mean_out"]
PSEUDO = 0.1


def _ranks(X: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(X, axis=0, method="average")


def domain_markers(counts_tiles_x_genes, gene_ids, gene_names, partitions: dict, set_name: str = "keep",
                   n_top: int = 25, min_detected: float = 0.05, min_pct_in: float = 0.1, chunk: int = 2000) -> pd.DataFrame:
    """``partitions``: {(method, k): domain labels over the set's tiles}. Returns the
    markers table (``COLUMNS``)."""
    from scipy import sparse

    from .relevance import normalise

    Y = sparse.csr_matrix(counts_tiles_x_genes, dtype=float)
    n = Y.shape[0]
    if n == 0 or Y.shape[1] == 0 or not partitions:
        return pd.DataFrame(columns=COLUMNS)
    detected = np.asarray((Y > 0).mean(axis=0)).ravel()
    genes = np.flatnonzero(detected >= min_detected)
    if genes.size == 0:
        return pd.DataFrame(columns=COLUMNS)
    totals = np.asarray(Y.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    cp10k = sparse.csr_matrix(sparse.diags(1e4 / totals) @ Y[:, genes])
    logN = normalise(Y[:, genes])
    pos = sparse.csr_matrix(Y[:, genes] > 0, dtype=float)
    keys = list(partitions)
    onehots = {}
    for key in keys:
        lab = np.asarray(partitions[key], int)
        onehots[key] = (lab, np.unique(lab))
    # per partition, per domain: rank sums, means and detection, gene chunk by gene chunk
    acc = {key: {"rsum": [], "mean": [], "pct": []} for key in keys}
    for start in range(0, genes.size, chunk):
        sl = slice(start, min(start + chunk, genes.size))
        Rk = _ranks(logN[:, sl].toarray())
        C = cp10k[:, sl]
        P = pos[:, sl]
        for key in keys:
            lab, doms = onehots[key]
            M = sparse.csr_matrix((np.ones(n), (lab, np.arange(n))), shape=(int(lab.max()) + 1, n))
            acc[key]["rsum"].append(M @ Rk)
            acc[key]["mean"].append(np.asarray((M @ C).todense()))
            acc[key]["pct"].append(np.asarray((M @ P).todense()))
    tot_mean = np.asarray(cp10k.sum(axis=0)).ravel()
    tot_pct = np.asarray(pos.sum(axis=0)).ravel()
    gid = np.asarray(gene_ids, dtype=object)[genes]
    gname = np.asarray(gene_names, dtype=object)[genes]
    rows = []
    for key in keys:
        method, k = key
        lab, doms = onehots[key]
        rsum = np.hstack(acc[key]["rsum"])
        ssum = np.hstack(acc[key]["mean"])
        psum = np.hstack(acc[key]["pct"])
        for d in doms:
            n_in = int((lab == d).sum())
            n_out = n - n_in
            if n_in == 0 or n_out == 0:
                continue
            auc = (rsum[d] - n_in * (n_in + 1) / 2) / (n_in * n_out)
            m_in = ssum[d] / n_in
            m_out = (tot_mean - ssum[d]) / n_out
            p_in = psum[d] / n_in
            p_out = (tot_pct - psum[d]) / n_out
            lfc = np.log2((m_in + PSEUDO) / (m_out + PSEUDO))
            ok = np.flatnonzero((lfc > 0) & (p_in >= min_pct_in))
            order = ok[np.lexsort((-lfc[ok], -auc[ok]))][:n_top]
            for r, g in enumerate(order, start=1):
                rows.append({"set": set_name, "method": method, "k": int(k), "domain": int(d), "n_in": n_in,
                             "n_out": n_out, "rank": r, "gene_id": gid[g], "gene_name": gname[g], "auroc": float(auc[g]),
                             "log2fc": float(lfc[g]), "pct_in": float(p_in[g]), "pct_out": float(p_out[g]),
                             "mean_in": float(m_in[g]), "mean_out": float(m_out[g])})
    return pd.DataFrame(rows, columns=COLUMNS)


def expression_partitions(result: dict, set_name: str = "keep", view: str = "EXPRESSION") -> dict:
    """{(method, k): labels} of the expression domains of a set."""
    labels = result.get("labels", {}).get(set_name, {})
    return {(m, k): lab for (v, m, k), lab in labels.items() if v == view}


def with_f1(markers: pd.DataFrame, f1: pd.DataFrame) -> pd.DataFrame:
    """Per domain: its size and the F1 of every image view (wide), for the domains of the
    markers table."""
    if markers.empty or f1 is None or f1.empty:
        return pd.DataFrame()
    dom = markers.drop_duplicates(["set", "method", "k", "domain"])[["set", "method", "k", "domain", "n_in"]]
    wide = f1.pivot_table(index=["set", "method", "k", "domain"], columns="view", values="f1").reset_index()
    wide.columns = [c if c in ("set", "method", "k", "domain") else f"f1_{c}" for c in wide.columns]
    return dom.merge(wide, on=["set", "method", "k", "domain"], how="left")
