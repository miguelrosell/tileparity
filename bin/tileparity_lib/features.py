"""HDF5 container for per-tile feature blocks (Output2a and Output2b).

Layout::

    /tile_id                    (n,)      string
    /blocks/<NAME>/X            (n, d)    numeric   (row i <-> tile_id[i])
    /blocks/<NAME>/columns      (d,)      string
    attributes on / and on each block: provenance (model, radii, ...)

h5py writes arrays in row-major order; rhdf5 in R returns them transposed (d x n).
"""

from __future__ import annotations

import json

import h5py
import numpy as np


def write(path: str, tile_ids, blocks: dict, attrs: dict | None = None, block_attrs: dict | None = None) -> None:
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tile_id", data=np.asarray(tile_ids, dtype=object), dtype=str_dt)
        for k, v in (attrs or {}).items():
            h5.attrs[k] = _attr(v)
        grp = h5.create_group("blocks")
        for name, (X, columns) in blocks.items():
            b = grp.create_group(name)
            X = np.asarray(X)
            if X.ndim != 2 or X.shape[0] != len(tile_ids) or X.shape[1] != len(columns):
                raise ValueError(f"block {name}: shape {X.shape} does not match {len(tile_ids)} tiles x {len(columns)} columns")
            b.create_dataset("X", data=X, compression="gzip", chunks=True if X.size else None)
            b.create_dataset("columns", data=np.asarray(columns, dtype=object), dtype=str_dt)
            for k, v in (block_attrs or {}).get(name, {}).items():
                b.attrs[k] = _attr(v)


def _attr(v):
    """HDF5-safe attribute value: plain str, int or float; anything else as JSON text."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        return str.__str__(v)  # also turns str subclasses (e.g. torch's version) into plain str
    if isinstance(v, (int, float, np.integer, np.floating)):
        return v.item() if hasattr(v, "item") else v
    return json.dumps(v, default=str)


def read(path: str) -> tuple[np.ndarray, dict, dict]:
    def dec(a):
        return np.array([s.decode() if isinstance(s, bytes) else s for s in a], dtype=object)

    with h5py.File(path, "r") as h5:
        tile_ids = dec(h5["tile_id"][:])
        attrs = {k: _maybe_json(v) for k, v in h5.attrs.items()}
        blocks = {}
        for name, b in h5["blocks"].items():
            blocks[name] = (b["X"][:], dec(b["columns"][:]), {k: _maybe_json(v) for k, v in b.attrs.items()})
    return tile_ids, blocks, attrs


def _maybe_json(v):
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v
