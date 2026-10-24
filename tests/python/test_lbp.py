"""The per-tile LBP histograms must equal fastLBP's own whole-image computation."""

import numpy as np
import pyvips
import pytest

from tileparity_lib import lbp
from tileparity_lib.image import write_tiled_tiff

fastlbp = pytest.importorskip("fastlbp")


def _texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    # add flat plateaus: ties (g_p == g_c) are where floating-point details matter
    base[h // 4 : h // 2, w // 4 : (3 * w) // 4] = 128
    return base


@pytest.mark.parametrize("radii", [[1, 2, 3], [1, 5, 11]])
def test_tile_histograms_equal_fastlbp_whole_image(radii):
    T = 32
    img = _texture(4 * T, 5 * T)
    npoints = [lbp.npoints_for(r) for r in radii]
    spec = lbp.LbpSpec(radii=radii, npoints=npoints, channels=["H"], tile_px=T)
    ref = fastlbp.run_fastlbp(img, radii, npoints, T, ncpus=1)  # (rows, cols, features)
    pad = spec.pad
    for row in range(1, 3):  # interior tiles: margin lies inside the image
        for col in range(1, 4):
            y0, x0 = row * T, col * T
            crop = img[y0 - pad : y0 + T + pad, x0 - pad : x0 + T + pad]
            ours = lbp.tile_histograms(crop, spec, y0, x0)
            np.testing.assert_array_equal(ours, ref[row, col].astype(np.uint32))


def test_histograms_count_every_tile_pixel_once_per_radius():
    T = 32
    img = _texture(3 * T, 3 * T, seed=1)
    radii = [1, 2, 4]
    spec = lbp.LbpSpec(radii=radii, npoints=[lbp.npoints_for(r) for r in radii], channels=["H"], tile_px=T)
    pad = spec.pad
    crop = img[T - pad : 2 * T + pad, T - pad : 2 * T + pad]
    h = lbp.tile_histograms(crop, spec, T, T)
    offset = 0
    for p in spec.npoints:
        assert h[offset : offset + p + 2].sum() == T * T
        offset += p + 2


def test_extract_matches_single_tile_computation(tmp_path):
    T = 32
    rng = np.random.default_rng(3)
    bands = rng.integers(0, 256, size=(3 * T, 4 * T, 3), dtype=np.uint8)
    path = tmp_path / "channels.tif"
    write_tiled_tiff(pyvips.Image.new_from_array(bands).copy(interpretation="multiband"), path, compression="deflate", tile=64)
    import pandas as pd

    tiles = pd.DataFrame({"x0_px": [T, 2 * T, 0], "y0_px": [T, T, 0]})  # last tile touches the border
    spec = lbp.LbpSpec(radii=[1, 3], npoints=[8, 19], channels=["H", "grey"], tile_px=T)
    out = lbp.extract(str(path), tiles, spec, ncpus=2, batch_size=1)
    pad = spec.pad
    for i, (x0, y0) in enumerate(zip(tiles["x0_px"], tiles["y0_px"])):
        padded = np.pad(bands, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        crop = padded[y0 : y0 + T + 2 * pad, x0 : x0 + T + 2 * pad]
        np.testing.assert_array_equal(out["H"][i], lbp.tile_histograms(crop[:, :, 0], spec, y0, x0))
        np.testing.assert_array_equal(out["grey"][i], lbp.tile_histograms(crop[:, :, 2], spec, y0, x0))
    assert out["intensity"].shape == (3, 10)


def test_default_radii_follow_fastlbp_progression():
    radii = lbp.default_radii(56)
    assert radii == [1, 2, 3, 4, 5, 6, 8, 11, 14, 19, 25, 34, 45]
    assert [lbp.npoints_for(r) for r in (1, 2, 45)] == [7, 13, 283]
    # P + 2 codes per radius (0..P ones in a regular pattern, plus one 'irregular' code): 1145 per channel
    assert sum(lbp.npoints_for(r) + 2 for r in radii) == 1145


def test_center_block_is_the_mean_code_vector_of_the_central_pixels():
    from fastlbp.lbp import uniform_lbp_uint8_padded_absolute

    T = 32
    img = _texture(3 * T, 3 * T, seed=4)
    radii = [1, 3]
    spec = lbp.LbpSpec(radii=radii, npoints=[lbp.npoints_for(r) for r in radii], channels=["H"], tile_px=T)
    pad = spec.pad
    crop = img[T - pad : 2 * T + pad, T - pad : 2 * T + pad]
    hist, center = lbp.tile_histograms(crop, spec, T, T, center=True)
    np.testing.assert_array_equal(hist, lbp.tile_histograms(crop, spec, T, T))
    offset = 0
    for r, p in zip(spec.radii, spec.npoints):
        m = r + 1
        sub = np.ascontiguousarray(crop[pad - m : pad + T + m, pad - m : pad + T + m])
        codes = uniform_lbp_uint8_padded_absolute(sub, p, float(r), T, T, [m, m, m, m])
        expected = np.bincount(codes[15:17, 15:17].ravel(), minlength=p + 2) / 4.0  # the 2 x 2 central pixels
        np.testing.assert_allclose(center[offset : offset + p + 2], expected)
        assert center[offset : offset + p + 2].sum() == pytest.approx(1.0)
        offset += p + 2
    assert lbp.center_slice(224) == slice(111, 113) and lbp.center_slice(5) == slice(2, 3)


def test_irregular_fractions_read_the_last_bin_of_each_radius():
    spec = lbp.LbpSpec(radii=[1, 2], npoints=[7, 13], channels=["H"], tile_px=10)
    hist = np.zeros((1, spec.n_features_per_channel))
    hist[0, 8] = 25  # radius 1: code P + 1 = 8
    hist[0, 9 + 14] = 50  # radius 2: code P + 1 = 14, after the 9 bins of radius 1
    np.testing.assert_allclose(lbp.irregular_fractions(hist, spec), [[0.25, 0.5]])


def test_extract_with_channel_sets_and_center(tmp_path):
    from tileparity_lib import channels

    T = 32
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(3 * T, 3 * T, 3), dtype=np.uint8)
    bands = channels.band_order(["he", "rgb"])
    assert bands == ["H", "E", "R", "G", "B"]
    arr = channels.rgb_to_channels_numpy(rgb, 0.3, 0.15, bands)
    path = tmp_path / "channels.tif"
    write_tiled_tiff(pyvips.Image.new_from_array(arr).copy(interpretation="multiband"), path, compression="deflate",
                     tile=64)
    import pandas as pd

    tiles = pd.DataFrame({"x0_px": [T], "y0_px": [T]})
    spec = lbp.LbpSpec(radii=[1, 2], npoints=[7, 13], channels=channels.lbp_channels(["he", "rgb"]), tile_px=T,
                       bands=bands, center=True)
    out = lbp.extract(str(path), tiles, spec, ncpus=1)
    assert set(out) == {"H", "E", "R", "G", "B", "center_H", "center_E", "center_R", "center_G", "center_B",
                        "intensity"}
    pad = spec.pad
    crop = arr[T - pad : 2 * T + pad, T - pad : 2 * T + pad]
    np.testing.assert_array_equal(out["G"][0], lbp.tile_histograms(np.ascontiguousarray(crop[:, :, 3]), spec, T, T))
    # the R/G/B channels are exactly the RGB values UNI2-h reads
    np.testing.assert_array_equal(arr[:, :, 2:], rgb)
