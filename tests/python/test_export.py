import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from tileparity_lib import export


def _adata(sample, n_tiles, genes):
    obs = pd.DataFrame({"sample_id": sample, "tile_id": [f"{sample}_{i}" for i in range(n_tiles)]},
                       index=[f"{sample}_{i}" for i in range(n_tiles)])
    X = sparse.csr_matrix(np.ones((n_tiles, len(genes)), dtype=np.int32))
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    a.obsm["X_uni2h"] = np.zeros((n_tiles, 4), np.float32)
    return a


def test_image_only_samples_are_left_out_of_the_combined_object():
    combined = export.combine_anndata([_adata("a", 3, ["g1", "g2", "g3"]), _adata("b", 2, ["g2", "g3", "g4"]),
                                       _adata("img", 4, [])])
    assert combined.n_obs == 5 and list(combined.var_names) == ["g2", "g3"]
    assert set(combined.obs["sample_id"]) == {"a", "b"}


def test_single_expression_sample_gives_image_only_combination():
    combined = export.combine_anndata([_adata("a", 3, ["g1"]), _adata("img", 2, [])])
    assert combined.n_obs == 5 and combined.n_vars == 0
    assert combined.obsm["X_uni2h"].shape == (5, 4)
