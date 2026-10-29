"""Reference results of Scanpy for tests/python/test_cluster.py (the clustering step follows
Scanpy's standard recipe and must reproduce it).

Regenerate (needs scanpy, which the pipeline itself does not use):

    pip install scanpy==1.12.5 igraph==1.0.0
    python tests/data/make_scanpy_reference.py

Writes tests/data/scanpy_reference.npz with:

* ``counts`` (tiles x genes) and, from Scanpy's normalize_total(1e4) + log1p +
  highly_variable_genes(flavor='seurat', n_top_genes=50) + scale(max_value=10):
  ``hvg`` (indices of the selected genes) and ``scaled`` (the scaled matrix of those genes);
* ``pcs`` (points) and, from neighbors(n_neighbors=10) on them: the connectivities as CSR
  (``conn_data``, ``conn_indices``, ``conn_indptr``) and the Leiden labels at resolution 1
  (``leiden``; flavor igraph, n_iterations -1, random_state 0).
"""

from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

rng = np.random.default_rng(42)
n_tiles, n_genes = 150, 200
domain = rng.integers(0, 4, n_tiles)
profiles = np.vstack([rng.gamma(0.8, 2.0, n_genes) * rng.lognormal(0, 0.8, n_genes) for _ in range(4)])
counts = rng.poisson(profiles[domain] * rng.lognormal(0, 0.4, n_tiles)[:, None]).astype(np.uint16)
counts[:, :5] = 0  # undetected genes

a = ad.AnnData(sparse.csr_matrix(counts.astype(np.float64)))
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=50)
hvg = np.flatnonzero(a.var["highly_variable"].to_numpy())
b = a[:, hvg].copy()
sc.pp.scale(b, max_value=10)

n_points = 100
pcs = rng.normal(0, 1, (4, 6))[rng.integers(0, 4, n_points)] + rng.normal(0, 0.9, (n_points, 6))
g = ad.AnnData(np.zeros((n_points, 1)))
g.obsm["X_pca"] = pcs
sc.pp.neighbors(g, n_neighbors=10, use_rep="X_pca")
sc.tl.leiden(g, flavor="igraph", n_iterations=-1, resolution=1.0, random_state=0)
conn = g.obsp["connectivities"].tocsr()

out = Path(__file__).resolve().parent / "scanpy_reference.npz"
np.savez_compressed(out, counts=counts, hvg=hvg, scaled=np.asarray(b.X, np.float64), pcs=pcs,
                    conn_data=conn.data.astype(np.float64), conn_indices=conn.indices, conn_indptr=conn.indptr,
                    leiden=g.obs["leiden"].astype(int).to_numpy())
print(f"wrote {out} ({out.stat().st_size} bytes), scanpy {version('scanpy')}")
