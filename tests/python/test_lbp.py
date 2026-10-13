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
    assert lbp.default_radii(56) == [1, 2, 3, 4, 5, 6, 8, 11, 14, 19, 25, 34, 45]
    assert [lbp.npoints_for(r) for r in (1, 2, 45)] == [7, 13, 283]
