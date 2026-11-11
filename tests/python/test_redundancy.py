"""The redundancy analysis (--redundancy): spatial folds, cross-validated ridge prediction
between views, R2 and the per-tile unexplained share."""

import numpy as np
import pandas as pd
import pytest

from tileparity_lib import redundancy

pytest.importorskip("sklearn")


def _grid_tiles(H=20, W=20, T=224):
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    r, c = r.ravel(), c.ravel()
    return pd.DataFrame({"tile_id": [f"t{a}_{b}" for a, b in zip(r, c)], "x0_px": c * T, "y0_px": r * T,
                         "tile_px": T, "row": r, "col": c})


def test_spatial_folds_keep_blocks_together_and_balance_tiles():
    tiles = _grid_tiles()
    x, y, unit = redundancy.tile_centres_um(tiles, 0.5)  # 112 um tiles
    assert unit == "um" and np.isclose(x.min(), 56.0)
    folds = redundancy.spatial_folds(x, y, 560.0, 5, seed=0)  # blocks of 5 x 5 tiles: 16 blocks
    blocks = (tiles["row"] // 5).astype(str) + "_" + (tiles["col"] // 5).astype(str)
    assert pd.Series(folds).groupby(blocks.to_numpy()).nunique().max() == 1  # a block is in one fold
    sizes = np.bincount(folds)
    assert len(sizes) == 5 and sizes.max() - sizes.min() <= 25
    # no scale: blocks in tile widths
    x, y, unit = redundancy.tile_centres_um(tiles, None)
    assert unit == "tiles" and np.isclose(x.max(), 19.5)
    # everything in one block: no folds
    assert redundancy.spatial_folds(x, y, 1000.0, 5, seed=0) is None


def test_prediction_is_high_for_a_linear_function_and_null_for_noise():
    rng = np.random.default_rng(0)
    tiles = _grid_tiles()
    n = len(tiles)
    X = rng.normal(size=(n, 10))
    Y = X @ rng.normal(size=(10, 6)) + 0.1 * rng.normal(size=(n, 6))
    folds = redundancy.spatial_folds(*redundancy.tile_centres_um(tiles, 0.5)[:2], 560.0, 5, 0)
    pred, alphas = redundancy.cross_predict(X, Y, folds)
    total, per = redundancy.r2_scores(Y, pred)
    assert total > 0.95 and (per > 0.9).all() and len(alphas) == 5
    noise = rng.normal(size=(n, 6))
    total, _ = redundancy.r2_scores(noise, redundancy.cross_predict(X, noise, folds)[0])
    assert -0.1 < total < 0.05
    # the mean of the per-tile unexplained share is 1 - R2
    u = redundancy.unexplained_per_tile(Y, pred)
    assert np.isclose(u.mean(), 1 - redundancy.r2_scores(Y, pred)[0])


def test_run_measures_containment_both_ways():
    """B is a part of A: A predicts B, B predicts only part of A."""
    rng = np.random.default_rng(1)
    tiles = _grid_tiles()
    n = len(tiles)
    A = rng.normal(size=(n, 12))
    B = A[:, :3] @ rng.normal(size=(3, 3)) + 0.05 * rng.normal(size=(n, 3))
    C = rng.normal(size=(n, 5))
    scores = {"UNI2H_RGB": A, "LBP_RGB": B, "INTENSITY_HE": C}
    out = redundancy.run(scores, {"UNI2H_RGB": 0.9, "LBP_RGB": 0.8, "INTENSITY_HE": 1.0}, tiles,
                         ["UNI2H_RGB", "LBP_RGB", "INTENSITY_HE"], "keep", 0.5, n_folds=4, block_um=560.0)
    t = out["table"].set_index(["predictor", "target"])
    assert len(t) == 6 and list(out["table"].columns) == redundancy.TABLE_COLUMNS
    assert t.loc[("UNI2H_RGB", "LBP_RGB"), "r2"] > 0.95
    assert 0.1 < t.loc[("LBP_RGB", "UNI2H_RGB"), "r2"] < 0.4  # 3 of 12 equal-variance dimensions
    assert t.loc[("UNI2H_RGB", "INTENSITY_HE"), "r2"] < 0.05
    assert t.loc[("UNI2H_RGB", "LBP_RGB"), "target_pc_variance"] == 0.8
    pcs = out["pcs"]
    assert len(pcs[(pcs["predictor"] == "UNI2H_RGB") & (pcs["target"] == "LBP_RGB")]) == 3
    # the pairs drawn on the slide: UNI2-h and LBP on the same pixels, both ways
    assert out["info"]["map_pairs"] == [["UNI2H_RGB", "LBP_RGB"], ["LBP_RGB", "UNI2H_RGB"]]
    assert {"tile_id", "fold", "unexplained_LBP_RGB_from_UNI2H_RGB",
            "unexplained_UNI2H_RGB_from_LBP_RGB"} <= set(out["tiles"].columns)
    assert out["info"]["n_folds"] == 4 and sum(out["info"]["fold_sizes"]) == n


def test_run_reports_why_it_did_not_run():
    tiles = _grid_tiles(3, 3)
    one = redundancy.run({"LBP_RGB": np.ones((9, 2))}, {}, tiles, ["LBP_RGB"], "keep", 0.5)
    assert one["info"]["status"].startswith("not computed: fewer than 2 views")
    small = redundancy.run({"A": np.ones((9, 2)), "B": np.ones((9, 2))}, {}, tiles, ["A", "B"], "keep", 0.5,
                           block_um=5000.0)
    assert "fewer than 2 blocks" in small["info"]["status"]
