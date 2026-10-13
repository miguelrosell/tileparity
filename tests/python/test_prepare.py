"""Process 1 on the synthetic slide: geometry and assignment invariants."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import TEST_DATA
from tileparity_lib import prepare
from tileparity_lib.tiling import read_tiles


def _run(tmp_path, **kw):
    p = prepare.PrepareParams(sample_id=kw.pop("sample_id", "s"), image=TEST_DATA / "he_synthetic.tif",
                              out_prefix=str(tmp_path / "s"), **kw)
    return prepare.run(p)


def test_hd_grid(tmp_path):
    rep = _run(tmp_path, mpp_target=0.5, spatial=TEST_DATA / "hd_test/outs", spatial_format="spaceranger_hd")
    assert rep["status"] == "ok" and rep["tiling"] == "grid"
    assert rep["native"]["mpp"] == pytest.approx(0.25)
    tiles = read_tiles(tmp_path / "s.tiles.tsv.gz", keep_only=False)
    kept = tiles[tiles["keep"]]
    assert (kept["x0_px"] % 224 == 0).all() and (kept["y0_px"] % 224 == 0).all()
    assert (kept["tissue_fraction"] >= 0.5).all()
    # full interior tiles hold exactly 14 x 14 bins of 8 µm (112 µm / 8 µm)
    assert kept["n_st_units"].max() == 196
    a = pd.read_csv(tmp_path / "s.assignment.tsv.gz", sep="\t")
    assert a["barcode"].is_unique  # every bin in at most one tile
    assert set(a["tile_id"]) <= set(tiles["tile_id"])
    # assignment agrees with the geometry: bin centre inside its tile
    pos = pd.read_parquet(TEST_DATA / "hd_test/outs/binned_outputs/square_008um/spatial/tissue_positions.parquet")
    m = a.merge(pos, on="barcode").merge(tiles, on="tile_id")
    x = m["pxl_col_in_fullres"] * rep["scale_factor"]
    y = m["pxl_row_in_fullres"] * rep["scale_factor"]
    assert ((x >= m["x0_px"]) & (x < m["x0_px"] + 224) & (y >= m["y0_px"]) & (y < m["y0_px"] + 224)).all()


def test_visium_spot_tiles(tmp_path):
    rep = _run(tmp_path, mpp_target=0.5, spatial=TEST_DATA / "visium_test/outs", spatial_format="spaceranger",
               mpp_native=0.25)
    assert rep["tiling"] == "spot"
    tiles = read_tiles(tmp_path / "s.tiles.tsv.gz", keep_only=False)
    pos = pd.read_csv(TEST_DATA / "visium_test/outs/spatial/tissue_positions.csv")
    pos = pos[pos["in_tissue"] == 1].set_index("barcode")
    t = tiles.set_index("barcode").loc[pos.index]
    # tile centre within half a pixel of the spot centre (in resampled pixels)
    np.testing.assert_allclose(t["cx_px"], pos["pxl_col_in_fullres"] * 0.5, atol=0.5)
    np.testing.assert_allclose(t["cy_px"], pos["pxl_row_in_fullres"] * 0.5, atol=0.5)
    assert (tiles["n_st_units"] == 1).all()


def test_upsampling_is_refused(tmp_path):
    rep = _run(tmp_path, mpp_target=0.2, spatial=TEST_DATA / "hd_test/outs", spatial_format="spaceranger_hd")
    assert rep["status"] == "skipped"
    assert json.loads((tmp_path / "s.prepare.json").read_text())["status"] == "skipped"
    assert not (tmp_path / "s.tiles.tsv.gz").exists()


def test_image_only_sample(tmp_path):
    rep = _run(tmp_path, mpp_target=1.0)
    assert rep["status"] == "ok"
    a = pd.read_csv(tmp_path / "s.assignment.tsv.gz", sep="\t")
    assert len(a) == 0


def test_conflicting_pixel_sizes_are_an_error(tmp_path):
    with pytest.raises(ValueError, match="differs from samplesheet"):
        _run(tmp_path, mpp_target=1.0, mpp_native=1.0, spatial=TEST_DATA / "hd_test/outs",
             spatial_format="spaceranger_hd")


def test_wrong_samplesheet_mpp_is_caught_by_the_spot_spacing(tmp_path):
    # standard Visium has no microns_per_pixel; the 100 µm spot spacing reveals the error
    with pytest.raises(ValueError, match="positions:unit_spacing .* differs from samplesheet"):
        _run(tmp_path, mpp_target=1.0, mpp_native=0.5, spatial=TEST_DATA / "visium_test/outs",
             spatial_format="spaceranger")


def test_pixel_size_sources_are_reported(tmp_path):
    rep = _run(tmp_path, mpp_target=1.0, spatial=TEST_DATA / "visium_test/outs", spatial_format="spaceranger")
    cands = rep["native"]["mpp_candidates"]
    assert rep["native"]["mpp_source"] == "positions:unit_spacing"
    assert set(cands) == {"positions:unit_spacing", "image_metadata", "scalefactors:65um/spot_diameter_fullres"}
    assert all(v == pytest.approx(0.25, rel=1e-3) for v in cands.values())
    assert rep["spatial"]["fraction_in_tissue_inside_image"] == 1.0
    assert rep["thumbnail"]["fullres_to_thumbnail"] == pytest.approx(0.25 / 4.0, rel=0.01)


def test_wrong_image_is_detected(tmp_path):
    import pyvips

    small = tmp_path / "half.tif"
    pyvips.Image.new_from_file(str(TEST_DATA / "he_synthetic.tif")).resize(0.5).write_to_file(str(small))
    p = prepare.PrepareParams(sample_id="s", image=small, out_prefix=str(tmp_path / "s"), mpp_target=1.0,
                              mpp_native=0.25, spatial=TEST_DATA / "hd_test/outs", spatial_format="spaceranger_hd")
    with pytest.raises(ValueError, match="not the one Space Ranger used"):
        prepare.run(p)
