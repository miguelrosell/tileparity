import json

import numpy as np
import pyvips
import pytest

from conftest import TEST_DATA
from tileparity_lib import channels, image, spatial


def test_vips_channels_match_numpy_reference():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    rgb[0, 0] = (0, 0, 0)
    rgb[0, 1] = (255, 255, 255)
    vips = channels.rgb_to_channels_vips(pyvips.Image.new_from_array(rgb), 0.30, 0.15).numpy()
    ref = channels.rgb_to_channels_numpy(rgb, 0.30, 0.15)
    diff = np.abs(vips.astype(int) - ref.astype(int))
    assert diff.max() <= 1  # float32 vs float64 rounding at .5 boundaries
    assert (diff == 0).mean() > 0.999


def test_channels_follow_skimage_rgb2hed():
    from skimage.color import rgb2hed

    rgb = np.array([[[90, 60, 140], [230, 170, 200]]], dtype=np.uint8)
    hed = rgb2hed(rgb)
    ref = channels.rgb_to_channels_numpy(rgb, 0.30, 0.15)
    expected_h = np.floor(np.minimum(hed[..., 0] * 255 / 0.30, 255) + 0.5)
    np.testing.assert_array_equal(ref[..., 0], expected_h)


def test_read_region_mirrors_outside_the_image():
    arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
    img = pyvips.Image.new_from_array(np.stack([arr] * 3, -1))
    region = image.read_region(img, -2, 0, 5, 3)
    assert region.shape == (3, 5, 3)
    np.testing.assert_array_equal(region[0, :, 0], [2, 1, 0, 1, 2])


def test_mpp_from_tiff_resolution_tags():
    img = pyvips.Image.new_from_file(str(TEST_DATA / "he_synthetic.tif"))
    assert image.mpp_from_metadata(img) == pytest.approx(0.25)


@pytest.mark.parametrize("fmt,path,n_genes", [
    ("spaceranger_hd", "hd_test/outs", 30),
    ("spaceranger", "visium_test/outs", 30),
    ("hest", "hest_test.h5ad", 30),
])
def test_spatial_readers(fmt, path, n_genes):
    sp = spatial.resolve(TEST_DATA / path, fmt)
    pos = spatial.read_positions(sp)
    assert {"barcode", "in_tissue", "x_fullres", "y_fullres"} <= set(pos.columns)
    counts = spatial.read_counts(sp)
    assert counts.matrix.shape[0] == n_genes
    assert set(counts.barcodes) <= set(pos.loc[pos["in_tissue"] == 1, "barcode"])


def test_visium_and_hest_readers_agree():
    a = spatial.read_positions(spatial.resolve(TEST_DATA / "visium_test/outs", "spaceranger"))
    b = spatial.read_positions(spatial.resolve(TEST_DATA / "hest_test.h5ad", "hest"))
    a = a[a["in_tissue"] == 1].set_index("barcode").sort_index()
    b = b.set_index("barcode").sort_index()
    np.testing.assert_allclose(a[["x_fullres", "y_fullres"]].to_numpy(), b[["x_fullres", "y_fullres"]].to_numpy())


def test_hd_pixel_size_from_scalefactors():
    sp = spatial.resolve(TEST_DATA / "hd_test/outs", "spaceranger_hd")
    mpp, source = spatial.mpp_from_scalefactors(sp)
    assert mpp == pytest.approx(0.25) and source == "scalefactors:microns_per_pixel"
    assert json.loads(sp.scalefactors.read_text())["bin_size_um"] == 8.0
