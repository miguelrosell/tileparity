"""The optional 3-D UMAP QC and the LBP preparation recipe it uses."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tileparity_lib import qcumap


def test_lbp_recipe_gives_every_radius_block_the_same_weight():
    rng = np.random.default_rng(0)
    T = 10
    cols, blocks = [], []
    for r, p in ((1, 7), (2, 13)):
        cols += [f"H_r{r}_p{p}_u{u}" for u in range(p + 2)]
        counts = rng.multinomial(T * T, rng.dirichlet(np.ones(p + 2)) * (1 if r == 1 else 0.1) + 1e-3, size=40)
        blocks.append(counts)
    X = np.hstack(blocks)
    Z = qcumap.lbp_recipe(X, cols, T)
    first, second = Z[:, :9], Z[:, 9:]
    for B in (first, second):
        top = np.linalg.svd(B - B.mean(axis=0), compute_uv=False)[0] ** 2 / (B.shape[0] - 1)
        assert top == pytest.approx(1.0)


def test_expression_matrix_normalises_counts():
    Y = sparse.csc_matrix(np.array([[10, 0], [0, 20], [10, 20]]))  # 3 genes x 2 tiles
    E = qcumap.expression_matrix(Y)
    assert E.shape == (2, 3)
    np.testing.assert_allclose(E[0], np.log1p([5000, 0, 5000]))


def _sample(name, n_tiles, genes, rng):
    tiles = pd.DataFrame({"tile_id": [f"{name}_{i}" for i in range(n_tiles)], "sample_id": name, "tile_px": 10,
                          "cx_um": 0.0, "cy_um": 0.0, "tissue_fraction": 1.0, "keep": True, "blur_flag": False})
    counts = sparse.csc_matrix(rng.poisson(3, (len(genes), n_tiles)))
    return {"tiles": tiles, "blocks": {"UNI2H_RGB": (rng.normal(size=(n_tiles, 4)), [f"f{i}" for i in range(4)]),
                                       "UNI2H_RGB_HE": (rng.normal(size=(n_tiles, 4)), [f"g{i}" for i in range(4)])},
            "counts": counts, "gene_ids": genes}


def test_expression_view_leaves_out_image_only_samples():
    rng = np.random.default_rng(2)
    samples = [_sample("a", 5, ["g1", "g2", "g3"], rng), _sample("b", 4, ["g2", "g3", "g4"], rng),
               _sample("img", 6, [], rng)]
    data = qcumap.views(samples)
    assert len(data["obs"]) == 15
    X, note, obs = data["views"]["UNI2H_RGB"]
    assert X.shape == (15, 4) and len(obs) == 15
    assert "rebuilt from H and E only" in data["views"]["UNI2H_RGB_HE"][1]
    E, note, obs = data["views"]["EXPRESSION"]
    assert E.shape == (9, 2)  # the 9 tiles of a and b, on their shared genes g2 and g3
    assert set(obs["sample_id"]) == {"a", "b"} and "1 image-only sample(s) left out" in note
    assert obs["total_counts"].notna().all()
    assert data["obs"].loc[data["obs"]["sample_id"] == "img", "total_counts"].isna().all()


def test_embedding_and_page():
    pytest.importorskip("umap")
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (30, 8)), rng.normal(5, 1, (30, 8))])
    coords, how = qcumap.embed3d(X)
    assert coords.shape == (60, 3) and "UMAP" in how
    assert qcumap.embed3d(X[:5])[0] is None
    obs = pd.DataFrame({"tile_id": [f"t{i}" for i in range(60)], "sample_id": ["a"] * 30 + ["b"] * 30,
                        "cx_um": 0.0, "cy_um": 0.0, "tissue_fraction": 1.0, "keep": True, "blur_flag": False})
    fig = qcumap.figure(coords, obs, "test")
    html = qcumap.page([("UNI2H_RGB", qcumap.to_html(fig, True), "note")], "title",
                       {"What it shows": "w", "How to read it": "h", "Caveats": "c", "Where it comes from": "s"})
    assert "Where it comes from" in html and "plotly" in html.lower()


def test_lbp_columns_of_channels_with_underscores_stay_apart():
    cols = [f"{c}_r1_p7_u{u}" for c in ("R", "R_he", "center_B_he") for u in range(9)]
    keys = [qcumap.COLUMN_RE.match(c).group("ch") for c in cols]
    assert keys == ["R"] * 9 + ["R_he"] * 9 + ["B_he"] * 9
