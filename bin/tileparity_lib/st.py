"""Process 3 - expression per tile.

With X (genes x units) the raw UMI counts of Space Ranger's units (Visium HD bins or
Visium spots) and A (units x tiles) the assignment matrix written by Process 1, where
A[b, t] is the share of unit b that lies in tile t:

* Visium HD bins and spot-centred tiles: A[b, t] = 1 for the tile holding the unit's
  centre (at most one 1 per row), and

      Y = X A,    Y[g, t] = sum over units b in tile t of X[g, b]

  i.e. the number of molecules of gene g captured over the tile's area.

* Visium spots on the grid: a spot that straddles tile borders has one weight per tile
  it touches (the share of its area inside the tile; the weights of a spot sum to 1, or
  less where part of it falls on tiles that are not used). Two ways to split:

  - ``multinomial`` (default): each molecule of the spot is given to one tile at random,
    tile t with probability A[b, t] (and dropped with probability 1 - sum_t A[b, t]).
    Counts stay integers, every molecule is counted at most once, so tiles do not share
    information; if the spot's counts are Poisson, the parts are independent Poisson
    counts (count splitting). The random seed is fixed and recorded.
  - ``fraction``: Y = X A with fractional weights, i.e. the expected value of the split.
    Counts become non-integers and a spot's molecules are spread over several tiles.

  Both assume that molecules are spread evenly over the spot's area.
"""

from __future__ import annotations

import logging
import zlib

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .spatial import Counts

log = logging.getLogger(__name__)
SPLITS = ("multinomial", "fraction")


def aggregate(counts: Counts, assignment: pd.DataFrame, tile_ids: np.ndarray, split: str = "multinomial",
              seed: int = 1, stream: str = "") -> tuple[sparse.csc_matrix, pd.DataFrame, dict]:
    """Expression per tile; returns (genes x tiles matrix, per-tile summary, report)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    col_of_barcode = pd.Series(np.arange(len(counts.barcodes)), index=pd.Index(counts.barcodes))
    col_of_tile = pd.Series(np.arange(len(tile_ids)), index=pd.Index(tile_ids))
    a = assignment[assignment["tile_id"].isin(col_of_tile.index)].copy()
    a["weight"] = pd.to_numeric(a["weight"]) if "weight" in a.columns else 1.0
    found = a["barcode"].isin(col_of_barcode.index)
    report = {"n_assignments": int(len(a)), "n_assignments_found_in_matrix": int(found.sum())}
    if len(a) and found.mean() < 0.95:
        log.warning("only %.1f%% of assigned units are present in the count matrix", 100 * found.mean())
    a = a[found]
    rows = col_of_barcode.loc[a["barcode"]].to_numpy()
    cols = col_of_tile.loc[a["tile_id"]].to_numpy()
    w = a["weight"].to_numpy(float)
    X = sparse.csc_matrix(counts.matrix)
    unit_total = np.asarray(X.sum(axis=0)).ravel()
    if len(rows) and np.any(np.bincount(rows, weights=w, minlength=X.shape[1]) > 1 + 1e-6):
        raise AssertionError("a spatial unit is assigned more than once (its weights sum to more than 1)")
    whole = bool(len(w) == 0 or (np.allclose(w, 1.0) and pd.Series(rows).is_unique))
    if split == "multinomial" and not whole and not is_integer(X):
        log.warning("the count matrix is not integer-valued: splitting spots by fraction instead of multinomially")
        split = "fraction"
    if whole:
        A = sparse.csc_matrix((np.ones(len(a), dtype=np.int64), (rows, cols)), shape=(X.shape[1], len(tile_ids)))
        Y = sparse.csc_matrix(X.astype(np.int64) @ A)
        report["split"] = "none (each unit in one tile)"
    elif split == "fraction":
        A = sparse.csc_matrix((w, (rows, cols)), shape=(X.shape[1], len(tile_ids)))
        Y = sparse.csc_matrix(X.astype(np.float64) @ A)
        report["split"] = "fraction"
    else:
        Y = _multinomial(X, rows, cols, w, len(tile_ids), np.random.default_rng([seed, zlib.crc32(stream.encode())]))
        report.update(split="multinomial", seed=seed, stream=stream)
    units = np.unique(rows)
    in_total = float(unit_total[units].sum())
    out_total = float(Y.sum())
    report.update(n_units=int(len(units)), counts_of_assigned_units=in_total, counts_in_tiles=out_total,
                  share_of_counts_in_tiles=out_total / in_total if in_total else None)
    n_units = np.bincount(cols, minlength=len(tile_ids))
    units_equiv = np.bincount(cols, weights=w, minlength=len(tile_ids))
    total = np.asarray(Y.sum(axis=0)).ravel()
    summary = pd.DataFrame(
        {
            "tile_id": tile_ids,
            "n_units_summed": n_units.astype(int),
            "units_equiv": units_equiv,
            "total_counts": total if not np.issubdtype(Y.dtype, np.integer) else total.astype(np.int64),
            "n_genes_detected": np.asarray((Y > 0).sum(axis=0)).ravel().astype(int),
        }
    )
    return Y, summary, report


def _multinomial(X: sparse.csc_matrix, rows: np.ndarray, cols: np.ndarray, w: np.ndarray, n_tiles: int,
                 rng: np.random.Generator) -> sparse.csc_matrix:
    """Split every unit's integer counts between its tiles (multinomial, weights ``w``)."""
    order = np.lexsort((cols, rows))
    rows, cols, w = rows[order], cols[order], w[order]
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    data = X.data.astype(np.int64)
    out_g, out_t, out_n = [], [], []
    for s, e in zip(starts, ends):
        b = rows[s]
        lo, hi = X.indptr[b], X.indptr[b + 1]
        if lo == hi:
            continue
        genes = X.indices[lo:hi]
        remaining = data[lo:hi].copy()
        mass = 1.0  # probability mass not yet given out (includes the dropped share)
        weights = w[s:e]
        complete = abs(weights.sum() - 1.0) < 1e-9
        for k in range(e - s):
            if complete and k == e - s - 1:
                n = remaining.copy()  # last tile takes the rest: exact multinomial
            else:
                p = min(max(weights[k] / mass, 0.0), 1.0) if mass > 0 else 0.0
                n = rng.binomial(remaining, p)
            remaining = remaining - n
            mass -= weights[k]
            nz = n > 0
            if nz.any():
                out_g.append(genes[nz])
                out_t.append(np.full(int(nz.sum()), cols[s + k]))
                out_n.append(n[nz])
    if not out_g:
        return sparse.csc_matrix((X.shape[0], n_tiles), dtype=np.int64)
    Y = sparse.coo_matrix((np.concatenate(out_n), (np.concatenate(out_g), np.concatenate(out_t))),
                          shape=(X.shape[0], n_tiles), dtype=np.int64)
    return sparse.csc_matrix(Y)


def is_integer(Y: sparse.spmatrix) -> bool:
    data = sparse.csc_matrix(Y).data
    return bool(np.issubdtype(data.dtype, np.integer) or np.allclose(data, np.round(data)))


def write_counts_h5(path: str, Y: sparse.csc_matrix, tile_ids, gene_ids, gene_names, attrs: dict | None = None) -> None:
    """10x-style HDF5 (CSC, genes x tiles); readable from Python (h5py) and R (rhdf5).
    Integer counts are stored as int32; fractional counts (``--visium_split fraction``) as float32."""
    Y = sparse.csc_matrix(Y)
    integer = is_integer(Y)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        g = h5.create_group("matrix")
        g.attrs["n_genes"] = int(Y.shape[0])
        g.attrs["n_tiles"] = int(Y.shape[1])
        g.attrs["integer_counts"] = int(integer)
        for k, v in (attrs or {}).items():
            g.attrs[k] = v if isinstance(v, (int, float, str)) else str(v)
        data = np.round(Y.data).astype(np.int32) if integer else Y.data.astype(np.float32)
        g.create_dataset("data", data=data, compression="gzip")
        g.create_dataset("indices", data=Y.indices.astype(np.int64), compression="gzip")
        g.create_dataset("indptr", data=Y.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(Y.shape, dtype=np.int64))
        g.create_dataset("barcodes", data=np.asarray(tile_ids, dtype=object), dtype=str_dt)
        f = g.create_group("features")
        f.create_dataset("id", data=np.asarray(gene_ids, dtype=object), dtype=str_dt)
        f.create_dataset("name", data=np.asarray(gene_names, dtype=object), dtype=str_dt)


def read_counts_h5(path: str) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        g = h5["matrix"]
        shape = tuple(int(v) for v in g["shape"][:])
        tiles = np.array([s.decode() if isinstance(s, bytes) else s for s in g["barcodes"][:]], dtype=object)
        if shape[0] == 0:
            return sparse.csc_matrix(shape, dtype=np.int64), tiles, np.array([], dtype=object), np.array([], dtype=object)
        Y = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        ids = np.array([s.decode() if isinstance(s, bytes) else s for s in g["features"]["id"][:]], dtype=object)
        names = np.array([s.decode() if isinstance(s, bytes) else s for s in g["features"]["name"][:]], dtype=object)
    return Y, tiles, ids, names
