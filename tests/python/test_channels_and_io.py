import json

import numpy as np
import pandas as pd
import pyvips
import pytest

from conftest import TEST_DATA
from tileparity_lib import channels, image, spatial


ALL_BANDS = ["H", "E", "R", "G", "B", "R_he", "G_he", "B_he"]


def _slide_basis():
    """Stain colours of a made-up laboratory: the standard ones turned by a few degrees."""
    std = channels.standard_basis()
    H = channels._unit(std.H + np.array([0.06, -0.05, 0.08]))
    E = channels._unit(std.E + np.array([0.12, -0.02, 0.07]))
    return channels.StainBasis(H, E, "test")


def _from_amounts(a_h, a_e, a_r=0.0, basis=None, as_float=False):
    """RGB pixels whose optical density is a_h * H + a_e * E + a_r * R. Realistic amounts are up to
    about 0.15 (H) and 0.1 (E): natural-log optical density 2 and 1.4, i.e. dark nuclei."""
    basis = basis or channels.standard_basis()
    od = (np.asarray(a_h)[..., None] * basis.H + np.asarray(a_e)[..., None] * basis.E
          + np.asarray(a_r)[..., None] * basis.R)
    rgb = 255.0 * np.exp(channels.LOG_1E6 * od)
    return rgb if as_float else np.floor(np.clip(rgb, 0, 255) + 0.5).astype(np.uint8)


@pytest.mark.parametrize("basis", [None, "slide"])
def test_vips_channels_match_numpy_reference(basis):
    basis = _slide_basis() if basis == "slide" else None
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    rgb[0, 0] = (0, 0, 0)
    rgb[0, 1] = (255, 255, 255)
    vips = channels.rgb_to_channels_vips(pyvips.Image.new_from_array(rgb), 0.30, 0.15, ALL_BANDS, basis).numpy()
    ref = channels.rgb_to_channels_numpy(rgb, 0.30, 0.15, ALL_BANDS, basis)
    assert vips.shape == (64, 64, 8)
    np.testing.assert_array_equal(vips[..., 2:5], rgb)  # the rgb set is the image itself
    diff = np.abs(vips.astype(int) - ref.astype(int))
    assert diff.max() <= 1  # libvips and numpy may round differently at exactly .5
    assert (diff == 0).mean() > 0.999


def test_rgb_he_keeps_a_pure_h_and_e_image():
    rng = np.random.default_rng(1)
    for basis in (channels.standard_basis(), _slide_basis()):
        rgb = _from_amounts(rng.uniform(0, 0.15, (60, 60)), rng.uniform(0, 0.1, (60, 60)), basis=basis)
        rebuilt = channels.rgb_he_numpy(rgb, basis)
        diff = np.abs(rebuilt.astype(int) - rgb.astype(int))
        assert diff.max() <= 1 and (diff == 0).mean() > 0.95  # 8-bit rounding only
        assert channels.residual_share(rgb, rebuilt) < 0.001


def test_rgb_he_removes_only_the_colour_that_is_neither_h_nor_e():
    rng = np.random.default_rng(2)
    basis = _slide_basis()
    a_h, a_e = rng.uniform(0.01, 0.15, (60, 60)), rng.uniform(0.01, 0.1, (60, 60))
    tinted = _from_amounts(a_h, a_e, a_r=0.02, basis=basis)  # an extra colour on top of H and E
    clean = _from_amounts(a_h, a_e, basis=basis)
    assert np.abs(tinted.astype(int) - clean.astype(int)).mean() > 10  # the tint is clearly visible
    rebuilt = channels.rgb_he_numpy(tinted, basis)
    assert np.abs(rebuilt.astype(int) - clean.astype(int)).max() <= 1  # and gone from rgb_he
    assert channels.residual_share(tinted, rebuilt) > 0.1
    # the H and E amounts are not changed by the removed colour (R is perpendicular to H and E)
    np.testing.assert_allclose(channels.stain_amounts(_from_amounts(a_h, a_e, 0.02, basis, True), basis)[..., :2],
                               np.stack([a_h, a_e], -1), atol=1e-9)


def test_standard_vectors_are_those_of_skimage():
    from skimage.color import rgb2hed, rgb_from_hed

    np.testing.assert_array_equal(channels.RUIFROK_H, rgb_from_hed[0])
    np.testing.assert_array_equal(channels.RUIFROK_E, rgb_from_hed[1])
    # for colours made of H and E only, the amounts equal rgb2hed's (up to the norm of the published vectors)
    rng = np.random.default_rng(3)
    a_h, a_e = rng.uniform(0, 0.15, (8, 8)), rng.uniform(0, 0.1, (8, 8))
    raw = rgb_from_hed[0] * a_h[..., None] + rgb_from_hed[1] * a_e[..., None]
    rgb = 255.0 * np.exp(channels.LOG_1E6 * raw)
    hed = rgb2hed(rgb / 255.0)
    ours = channels.stain_amounts(rgb, channels.standard_basis())
    np.testing.assert_allclose(ours[..., 0], hed[..., 0] * np.linalg.norm(rgb_from_hed[0]), atol=1e-9)
    np.testing.assert_allclose(ours[..., 1], hed[..., 1] * np.linalg.norm(rgb_from_hed[1]), atol=1e-9)


def test_stain_vectors_are_estimated_from_the_slide():
    rng = np.random.default_rng(4)
    truth = _slide_basis()
    n = 30000
    a_h = np.concatenate([rng.uniform(0.03, 0.15, n // 5), np.zeros(n // 5), rng.uniform(0.01, 0.12, 3 * n // 5)])
    a_e = np.concatenate([np.zeros(n // 5), rng.uniform(0.03, 0.1, n // 5), rng.uniform(0.01, 0.08, 3 * n // 5)])
    glass = np.full((5000, 3), 245, np.uint8)  # background is not tissue and must be ignored
    pixels = np.vstack([_from_amounts(a_h, a_e, basis=truth).reshape(-1, 3), glass])
    est = channels.estimate_stain_basis(pixels)
    assert est.method.startswith("slide") and est.note == ""
    assert channels._angle(est.H, truth.H) < 2 and channels._angle(est.E, truth.E) < 2
    assert est.plane_energy > 0.99 and est.n_pixels <= n
    d = est.to_dict()
    assert d["angle_to_standard_degrees"]["H"] > 3  # really differs from the standard vectors
    json.dumps(d)


def test_stain_vectors_fall_back_to_the_standard_ones():
    few = np.full((50000, 3), 240, np.uint8)  # glass only: no tissue pixels
    est = channels.estimate_stain_basis(few)
    assert est.method.startswith("standard") and "stained pixels" in est.note
    np.testing.assert_allclose(est.H, channels.standard_basis().H)


def test_input_sets_are_validated():
    assert channels.parse_sets("RGB_HE, rgb") == ["rgb", "rgb_he"]
    assert channels.parse_sets("none", allow_empty=True) == [] and channels.parse_sets("", allow_empty=True) == []
    with pytest.raises(ValueError, match="removed"):
        channels.parse_sets("rgb,grey")
    with pytest.raises(ValueError):
        channels.parse_sets("he", channels.UNI2H_INPUTS)  # UNI2-h reads 3-channel images only
    with pytest.raises(ValueError):
        channels.parse_sets("H,E")
    assert channels.lbp_channels(["he", "rgb_he"]) == ["R_he", "G_he", "B_he", "H", "E"]
    assert channels.band_order(["rgb"], ["rgb_he"]) == ALL_BANDS
    assert channels.band_order(["he"], ["rgb"]) == ["H", "E"]
    assert channels.band_order([], ["rgb", "rgb_he"]) == ["H", "E", "R_he", "G_he", "B_he"]


def test_qc_metrics_in_one_pass_equal_the_per_tile_ones():
    import tp_channels

    rng = np.random.default_rng(5)
    T = 32
    a = rng.integers(0, 256, size=(3 * T + 7, 4 * T + 5, 3), dtype=np.uint8)  # not a multiple of the tile size
    b = np.clip(a.astype(int) + rng.integers(-15, 16, size=a.shape), 0, 255).astype(np.uint8)
    tiles = pd.DataFrame([(f"t{r}{c}", c * T, r * T) for r in range(3) for c in range(4)],
                         columns=["tile_id", "x0_px", "y0_px"])
    assert tp_channels.on_grid(tiles, T, a.shape[1], a.shape[0])
    assert not tp_channels.on_grid(tiles.assign(x0_px=tiles["x0_px"] + 3), T, a.shape[1], a.shape[0])
    grid = tp_channels.grid_metrics(pyvips.Image.new_from_array(a), pyvips.Image.new_from_array(b), tiles, T)
    for i, t in tiles.iterrows():
        ref = tp_channels.tile_metrics(a[t.y0_px:t.y0_px + T, t.x0_px:t.x0_px + T], b[t.y0_px:t.y0_px + T, t.x0_px:t.x0_px + T])
        for key, value in ref.items():
            assert grid.loc[i, key] == pytest.approx(value, rel=1e-6, abs=1e-9)
    assert list(grid.columns) == tp_channels.QC_COLUMNS


def test_channels_step_writes_the_inputs_of_both_extractors(tmp_path):
    from PIL import Image

    import tp_channels
    from tileparity_lib import lineage, prepare
    from tileparity_lib.tiling import read_tiles

    prepare.run(prepare.PrepareParams(sample_id="s", image=TEST_DATA / "he_synthetic.tif", out_prefix=str(tmp_path / "s"),
                                      mpp_target=1.0))
    prefix = str(tmp_path / "s")
    tp_channels.main(["--image", f"{prefix}.resampled.tif", "--tiles", f"{prefix}.tiles.tsv.gz", "--prefix", prefix,
                      "--lbp-sets", "rgb,rgb_he,he", "--uni2h-inputs", "rgb,rgb_he",
                      "--lineage-parent", f"{prefix}.prepare.lineage.json"])
    report = json.loads((tmp_path / "s.channels.json").read_text())
    assert report["bands"] == ALL_BANDS
    assert report["stain_vectors"]["method"].startswith(("slide", "standard"))
    img = pyvips.Image.new_from_file(f"{prefix}.channels.tif")
    rgb = pyvips.Image.new_from_file(f"{prefix}.resampled.tif")
    assert img.bands == 8 and (img.width, img.height) == (rgb.width, rgb.height)
    np.testing.assert_array_equal(img.numpy()[..., 2:5], rgb.numpy()[..., :3])
    qc = pd.read_csv(tmp_path / "s.rgb_he_qc.tsv", sep="\t")
    assert list(qc["tile_id"]) == list(read_tiles(f"{prefix}.tiles.tsv.gz")["tile_id"])
    assert qc["residual_share"].between(0, 1).all() and qc["changed_fraction"].between(0, 1).all()
    assert report["rgb_he"]["n_tiles"] == len(qc)
    fig = Image.open(tmp_path / "s.rgb_he_qc.png")
    assert fig.width >= 960 and fig.height > 600
    lin = lineage.load(tmp_path / "s.channels.lineage.json")
    assert lin["record"]["process"] == "TILEPARITY_CHANNELS"
    assert [r["step"] for r in lineage.route(lin)] == ["prepare", "channels"]

    # without rgb_he no H&E-only image and no QC figure
    tp_channels.main(["--image", f"{prefix}.resampled.tif", "--tiles", f"{prefix}.tiles.tsv.gz", "--prefix",
                      str(tmp_path / "t"), "--lbp-sets", "rgb", "--stain-vectors", "standard"])
    assert json.loads((tmp_path / "t.channels.json").read_text())["bands"] == ["H", "E", "R", "G", "B"]
    assert not (tmp_path / "t.rgb_he_qc.png").exists()


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


def test_visium_spot_diameter_follows_the_65um_convention():
    # Space Ranger documents spot_diameter_fullres as a theoretical 65 µm spot, not 55 µm
    sp = spatial.resolve(TEST_DATA / "visium_test/outs", "spaceranger")
    mpp, source = spatial.mpp_from_scalefactors(sp)
    assert mpp == pytest.approx(0.25) and "65um" in source


@pytest.mark.parametrize("fmt,path", [("spaceranger_hd", "hd_test/outs"), ("spaceranger", "visium_test/outs")])
def test_pixel_size_from_unit_spacing(fmt, path):
    sp = spatial.resolve(TEST_DATA / path, fmt)
    pos = spatial.read_positions(sp)
    assert spatial.mpp_from_pitch(pos, spatial.unit_pitch_um(sp)) == pytest.approx(0.25, rel=1e-3)
    # large inputs are reduced to a window around the array centre: same answer
    assert spatial.mpp_from_pitch(pos, spatial.unit_pitch_um(sp), max_units=50) == pytest.approx(0.25, rel=1e-3)


def test_hest_spacing_is_not_assumed():
    sp = spatial.resolve(TEST_DATA / "hest_test.h5ad", "hest")
    assert spatial.unit_pitch_um(sp) is None
