import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tileparity_lib import mask, st, tiling
from tileparity_lib.spatial import Counts


def test_grid_is_anchored_at_origin_and_complete():
    t = tiling.grid_tiles(width=500, height=230, tile_px=224)
    assert len(t) == 2  # 2 columns x 1 row; partial tiles are dropped
    assert list(t["x0_px"]) == [0, 224] and list(t["y0_px"]) == [0, 0]


def test_assignment_puts_each_point_in_exactly_one_tile():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-50, 700, 5000), rng.uniform(-50, 500, 5000)
    row, col = tiling.assign_to_grid(x, y, 224, n_rows=2, n_cols=3)
    inside = (x >= 0) & (y >= 0) & (x < 672) & (y < 448)
    assert np.array_equal(row >= 0, inside)
    assert np.all(col[inside] == np.floor(x[inside] / 224))
    # boundaries belong to the tile on their right/bottom
    r, c = tiling.assign_to_grid(np.array([223.999, 224.0]), np.array([0.0, 0.0]), 224, 2, 3)
    assert list(c) == [0, 1]


def test_spot_tiles_are_centred():
    t = tiling.spot_tiles(np.array([300.2]), np.array([400.7]), 224, 1000, 1000)
    assert t["x0_px"].iloc[0] == 188 and t["y0_px"].iloc[0] == 289
    assert bool(t["inside_image"].iloc[0])
    edge = tiling.spot_tiles(np.array([50.0]), np.array([500.0]), 224, 1000, 1000)
    assert not bool(edge["inside_image"].iloc[0])


def test_rect_fractions_counts_pixel_centres():
    m = np.zeros((10, 10), bool)
    m[:, :5] = True  # left half is tissue
    # windows of 4 mask pixels (side 8 in a grid twice as fine)
    frac = mask.rect_fractions(m, x0=np.array([0, 6, 12]), y0=np.array([0, 0, 0]), size=8, mask_scale=0.5)
    np.testing.assert_allclose(frac, [1.0, 0.5, 0.0])


def test_tissue_mask_separates_glass_from_tissue():
    rng = np.random.default_rng(1)
    img = np.full((200, 300, 3), 240, np.uint8)
    img[:, 150:] = (225, 150, 190)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    m, info = mask.tissue_mask(img, mask.MaskParams(mask_mpp=4.0))
    assert m[:, 160:].mean() > 0.99 and m[:, :140].mean() < 0.01
    assert 0 < info["otsu_threshold_saturation"] < 1


def test_fill_small_holes_keeps_border_background():
    m = np.ones((50, 50), bool)
    m[20:25, 20:25] = False  # enclosed hole
    m[:, :3] = False  # background touching the border
    out = mask.fill_small_holes(m, max_area_px=100)
    assert out[20:25, 20:25].all() and not out[:, :3].any()


def test_aggregate_is_matrix_product():
    X = np.array([[1, 2, 0, 4], [0, 1, 1, 0]])  # 2 genes x 4 units
    counts = Counts(sparse.csc_matrix(X), np.array(["a", "b", "c", "d"], dtype=object),
                    np.array(["g1", "g2"], dtype=object), np.array(["G1", "G2"], dtype=object))
    assignment = pd.DataFrame({"barcode": ["a", "b", "d"], "tile_id": ["t1", "t1", "t2"]})
    Y, summary, report = st.aggregate(counts, assignment, np.array(["t1", "t2", "t3"], dtype=object))
    np.testing.assert_array_equal(Y.toarray(), [[3, 4, 0], [1, 0, 0]])
    assert list(summary["n_units_summed"]) == [2, 1, 0]
    assert report["n_units_found_in_matrix"] == 3


def test_aggregate_rejects_double_assignment():
    counts = Counts(sparse.csc_matrix(np.ones((1, 2))), np.array(["a", "b"], dtype=object),
                    np.array(["g"], dtype=object), np.array(["G"], dtype=object))
    assignment = pd.DataFrame({"barcode": ["a", "a"], "tile_id": ["t1", "t2"]})
    with pytest.raises(AssertionError):
        st.aggregate(counts, assignment, np.array(["t1", "t2"], dtype=object))


def test_counts_h5_roundtrip(tmp_path):
    Y = sparse.csc_matrix(np.array([[0, 5], [7, 0], [1, 1]]))
    path = tmp_path / "c.h5"
    st.write_counts_h5(str(path), Y, ["t1", "t2"], ["g1", "g2", "g3"], ["A", "B", "C"])
    Y2, tiles, ids, names = st.read_counts_h5(str(path))
    np.testing.assert_array_equal(Y2.toarray(), Y.toarray())
    assert list(tiles) == ["t1", "t2"] and list(names) == ["A", "B", "C"]
    empty = tmp_path / "e.h5"
    st.write_counts_h5(str(empty), sparse.csc_matrix((0, 2)), ["t1", "t2"], [], [])
    Y3, tiles3, ids3, _ = st.read_counts_h5(str(empty))
    assert Y3.shape == (0, 2) and len(ids3) == 0
