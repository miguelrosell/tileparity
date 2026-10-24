import numpy as np
import pandas as pd
import pytest
from scipy import ndimage, sparse

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


def test_window_fractions_outside_the_mask():
    m = np.ones((4, 4), bool)
    # window of 4 x 4 mask pixels, half of it beyond the right edge
    inside_only = mask.window_fractions(m, np.array([2.0]), np.array([0.0]), 4)
    as_background = mask.window_fractions(m, np.array([2.0]), np.array([0.0]), 4, outside_as_background=True)
    assert inside_only[0] == 1.0 and as_background[0] == 0.5


@pytest.mark.parametrize("method", ["saturation", "grey", "od"])
def test_tissue_mask_methods_separate_glass_from_tissue(method):
    rng = np.random.default_rng(1)
    img = np.full((200, 300, 3), 240, np.uint8)
    img[:, 150:] = (225, 150, 190)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    m, info = mask.tissue_mask(img, mask.MaskParams(mask_mpp=4.0, method=method))
    assert m[:, 160:].mean() > 0.99 and m[:, :140].mean() < 0.01
    assert info["method"] == method and info["otsu_threshold"] > 0


def test_black_padding_is_never_tissue():
    img = np.full((120, 120, 3), 240, np.uint8)
    img[:, 60:] = (225, 150, 190)
    img[:20] = 0  # scanner padding
    for method in mask.METHODS:
        m, _ = mask.tissue_mask(img, mask.MaskParams(mask_mpp=4.0, method=method, fill_holes_um2=0,
                                                     min_object_um2=0, close_um=0))
        assert not m[:20].any()


@pytest.mark.parametrize("radius", [1, 2, 3, 5, 8])
def test_distance_transform_closing_equals_morphological_closing(radius):
    rng = np.random.default_rng(radius)
    m = ndimage.binary_opening(rng.random((90, 110)) > 0.62)
    padded = np.pad(m, radius, mode="constant", constant_values=False)
    ref = ndimage.binary_closing(padded, structure=mask.disk(radius))[radius:-radius, radius:-radius]
    np.testing.assert_array_equal(mask.binary_closing(m, radius), ref)


def test_fill_small_holes_keeps_border_background():
    m = np.ones((50, 50), bool)
    m[20:25, 20:25] = False  # enclosed hole
    m[:, :3] = False  # background touching the border
    out = mask.fill_small_holes(m, max_area_px=100)
    assert out[20:25, 20:25].all() and not out[:, :3].any()


def test_external_mask_is_resized_and_checked(tmp_path):
    from PIL import Image

    src = np.zeros((40, 80), np.uint8)
    src[:, 40:] = 255
    Image.fromarray(src).save(tmp_path / "m.png")
    m, info = mask.load_external(tmp_path / "m.png", 20, 10, native_width=800, native_height=400)
    assert m.shape == (10, 20) and m[:, 11:].all() and not m[:, :9].any()
    assert info["method"] == "external"
    with pytest.raises(ValueError, match="aspect ratio"):
        mask.load_external(tmp_path / "m.png", 20, 20, native_width=800, native_height=800)


def test_dice_and_lookup():
    a = np.zeros((4, 4), bool)
    a[:2] = True
    b = np.zeros((4, 4), bool)
    b[1:3] = True
    assert mask.dice(a, b) == pytest.approx(0.5)
    assert list(mask.lookup(a, np.array([0.5, 0.5, 10]), np.array([0.5, 3.5, 0]))) == [True, False, False]


# ---- Visium spots on the grid ------------------------------------------------------------------
def test_disc_weights_split_area_between_tiles():
    T = 100
    # centred inside a tile, on a vertical border and on a corner of four tiles
    w = tiling.disc_tile_weights(np.array([50.0, 100.0, 100.0]), np.array([50.0, 50.0, 100.0]), radius=20, tile_px=T,
                                 n_rows=3, n_cols=3)
    one = w[w["unit"] == 0]
    assert len(one) == 1 and one["weight"].iloc[0] == pytest.approx(1.0)
    two = w[w["unit"] == 1]
    assert sorted(two["col"]) == [0, 1] and np.allclose(two["weight"], 0.5, atol=0.01)
    four = w[w["unit"] == 2]
    assert len(four) == 4 and np.allclose(four["weight"], 0.25, atol=0.01)
    assert w.groupby("unit")["weight"].sum().to_numpy() == pytest.approx([1, 1, 1])


def test_disc_weights_match_the_circular_segment_area():
    # disc of radius 1 cut by a line at distance d from its centre: segment area = r^2 acos(d/r) - d sqrt(r^2 - d^2)
    r, d = 30.0, 12.0
    w = tiling.disc_tile_weights(np.array([100.0 - d]), np.array([50.0]), radius=r, tile_px=100, n_rows=1, n_cols=2)
    segment = (r * r * np.arccos(d / r) - d * np.sqrt(r * r - d * d)) / (np.pi * r * r)
    assert w.loc[w["col"] == 1, "weight"].iloc[0] == pytest.approx(segment, abs=0.005)


def test_disc_weights_outside_the_lattice_are_lost():
    w = tiling.disc_tile_weights(np.array([10.0]), np.array([10.0]), radius=20, tile_px=100, n_rows=1, n_cols=1)
    # only the part inside the lattice counts: the disc minus two segments at distance 10 (19.6% each),
    # plus back the small corner region counted twice
    r, d = 20.0, 10.0
    segment = (r * r * np.arccos(d / r) - d * np.sqrt(r * r - d * d)) / (np.pi * r * r)
    assert 1 - 2 * segment < w["weight"].sum() < 1 - 2 * segment + 0.05


# ---- Process 3 ------------------------------------------------------------------------------
def _counts(X, barcodes):
    return Counts(sparse.csc_matrix(np.asarray(X)), np.asarray(barcodes, dtype=object),
                  np.array([f"g{i}" for i in range(len(X))], dtype=object),
                  np.array([f"G{i}" for i in range(len(X))], dtype=object))


def test_aggregate_is_matrix_product():
    counts = _counts([[1, 2, 0, 4], [0, 1, 1, 0]], ["a", "b", "c", "d"])  # 2 genes x 4 units
    assignment = pd.DataFrame({"barcode": ["a", "b", "d"], "tile_id": ["t1", "t1", "t2"]})
    Y, summary, report = st.aggregate(counts, assignment, np.array(["t1", "t2", "t3"], dtype=object))
    np.testing.assert_array_equal(Y.toarray(), [[3, 4, 0], [1, 0, 0]])
    assert list(summary["n_units_summed"]) == [2, 1, 0]
    assert report["n_assignments_found_in_matrix"] == 3 and report["share_of_counts_in_tiles"] == 1.0


def test_aggregate_rejects_double_assignment():
    counts = _counts([[1, 1]], ["a", "b"])
    assignment = pd.DataFrame({"barcode": ["a", "a"], "tile_id": ["t1", "t2"]})
    with pytest.raises(AssertionError):
        st.aggregate(counts, assignment, np.array(["t1", "t2"], dtype=object))


def _split_case():
    rng = np.random.default_rng(5)
    X = rng.poisson(40, size=(200, 2))  # 200 genes x 2 spots
    assignment = pd.DataFrame({"barcode": ["s1", "s1", "s1", "s2"], "tile_id": ["t1", "t2", "t3", "t3"],
                               "weight": [0.6, 0.3, 0.1, 0.7]})
    return _counts(X, ["s1", "s2"]), assignment, np.array(["t1", "t2", "t3"], dtype=object), X


def test_multinomial_split_keeps_integers_and_counts_every_molecule_once():
    counts, assignment, tiles, X = _split_case()
    Y, summary, report = st.aggregate(counts, assignment, tiles, split="multinomial", seed=3, stream="x")
    Y = Y.toarray()
    assert np.issubdtype(Y.dtype, np.integer)
    assert report["split"] == "multinomial"
    # s1 lies completely on t1-t3 (weights sum to 1); 70% of s2 lies on t3, the rest is dropped
    total_s1, total_s2 = X[:, 0].sum(), X[:, 1].sum()
    assert Y.sum() <= X.sum()
    assert Y[:, 0].sum() / total_s1 == pytest.approx(0.6, abs=0.02)
    assert Y[:, 1].sum() / total_s1 == pytest.approx(0.3, abs=0.02)
    assert Y[:, 2].sum() == pytest.approx(0.1 * total_s1 + 0.7 * total_s2, rel=0.03)
    # reproducible
    Y2, _, _ = st.aggregate(counts, assignment, tiles, split="multinomial", seed=3, stream="x")
    np.testing.assert_array_equal(Y, Y2.toarray())


def test_multinomial_split_of_a_complete_spot_is_exact():
    counts = _counts([[10, 0], [7, 3]], ["s1", "s2"])
    assignment = pd.DataFrame({"barcode": ["s1", "s1", "s2"], "tile_id": ["t1", "t2", "t2"], "weight": [0.5, 0.5, 1.0]})
    Y, _, _ = st.aggregate(counts, assignment, np.array(["t1", "t2"], dtype=object), split="multinomial", seed=1)
    Y = Y.toarray()
    np.testing.assert_array_equal(Y.sum(axis=1), [10, 10])  # all of s1 and s2 assigned
    assert Y[1, 1] >= 3  # s2's 3 molecules of gene 2 all go to t2


def test_fraction_split_is_the_expected_value():
    counts, assignment, tiles, X = _split_case()
    Y, _, report = st.aggregate(counts, assignment, tiles, split="fraction")
    W = np.array([[0.6, 0.3, 0.1], [0.0, 0.0, 0.7]])
    np.testing.assert_allclose(Y.toarray(), X @ W)
    assert report["split"] == "fraction" and not st.is_integer(Y)


def test_counts_h5_roundtrip(tmp_path):
    Y = sparse.csc_matrix(np.array([[0, 5], [7, 0], [1, 1]]))
    path = tmp_path / "c.h5"
    st.write_counts_h5(str(path), Y, ["t1", "t2"], ["g1", "g2", "g3"], ["A", "B", "C"])
    Y2, tiles, ids, names = st.read_counts_h5(str(path))
    np.testing.assert_array_equal(Y2.toarray(), Y.toarray())
    assert list(tiles) == ["t1", "t2"] and list(names) == ["A", "B", "C"]
    frac = tmp_path / "f.h5"
    st.write_counts_h5(str(frac), sparse.csc_matrix(np.array([[0.5, 1.25]])), ["t1", "t2"], ["g"], ["G"])
    np.testing.assert_allclose(st.read_counts_h5(str(frac))[0].toarray(), [[0.5, 1.25]])
    empty = tmp_path / "e.h5"
    st.write_counts_h5(str(empty), sparse.csc_matrix((0, 2)), ["t1", "t2"], [], [])
    Y3, tiles3, ids3, _ = st.read_counts_h5(str(empty))
    assert Y3.shape == (0, 2) and len(ids3) == 0
