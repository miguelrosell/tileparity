"""Process 3 - expression per tile.

With X (genes x units) the raw UMI counts of Space Ranger's units (Visium HD bins or
Visium spots) and A (units x tiles) the assignment matrix written by Process 1
(A[b, t] = 1 if unit b's centre lies in tile t, else 0; at most one 1 per row):

    Y = X A,    Y[g, t] = sum over units b in tile t of X[g, b]

Y[g, t] is the number of molecules of gene g captured over the area of tile t, i.e.
what one spot with exactly the tile's footprint would have measured. Sums of Poisson
counts are Poisson counts, so Y can be modelled and normalised like any count matrix.
For spot-centred tiles each tile has exactly one unit and Y is the spot's profile.
"""

from __future__ import annotations

import logging

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .spatial import Counts

log = logging.getLogger(__name__)


def aggregate(counts: Counts, assignment: pd.DataFrame, tile_ids: np.ndarray) -> tuple[sparse.csc_matrix, pd.DataFrame, dict]:
    """Sum unit counts per tile; returns (genes x tiles matrix, per-tile summary, report)."""
    col_of_barcode = pd.Series(np.arange(len(counts.barcodes)), index=pd.Index(counts.barcodes))
    col_of_tile = pd.Series(np.arange(len(tile_ids)), index=pd.Index(tile_ids))
    a = assignment[assignment["tile_id"].isin(col_of_tile.index)]
    found = a["barcode"].isin(col_of_barcode.index)
    report = {"n_assigned_units": int(len(a)), "n_units_found_in_matrix": int(found.sum())}
    if len(a) and found.mean() < 0.95:
        log.warning("only %.1f%% of assigned units are present in the count matrix", 100 * found.mean())
    a = a[found]
    rows = col_of_barcode.loc[a["barcode"]].to_numpy()
    cols = col_of_tile.loc[a["tile_id"]].to_numpy()
    A = sparse.csc_matrix((np.ones(len(a), dtype=np.int64), (rows, cols)), shape=(len(counts.barcodes), len(tile_ids)))
    if A.nnz and A.sum(axis=1).max() > 1:
        raise AssertionError("a spatial unit is assigned to more than one tile")
    Y = sparse.csc_matrix(counts.matrix.astype(np.int64) @ A)
    summary = pd.DataFrame(
        {
            "tile_id": tile_ids,
            "n_units_summed": np.asarray(A.sum(axis=0)).ravel().astype(int),
            "total_counts": np.asarray(Y.sum(axis=0)).ravel().astype(np.int64),
            "n_genes_detected": np.asarray((Y > 0).sum(axis=0)).ravel().astype(int),
        }
    )
    return Y, summary, report


def write_counts_h5(path: str, Y: sparse.csc_matrix, tile_ids, gene_ids, gene_names) -> None:
    """10x-style HDF5 (CSC, genes x tiles); readable from Python (h5py) and R (rhdf5)."""
    Y = sparse.csc_matrix(Y)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        g = h5.create_group("matrix")
        g.attrs["n_genes"] = int(Y.shape[0])
        g.attrs["n_tiles"] = int(Y.shape[1])
        g.create_dataset("data", data=Y.data.astype(np.int32), compression="gzip")
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
