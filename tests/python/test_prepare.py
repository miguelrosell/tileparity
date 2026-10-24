"""Process 1 on the synthetic slide: geometry, assignment, masks, figures and lineage."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import TEST_DATA
from tileparity_lib import lineage, mask, prepare
from tileparity_lib.tiling import read_tiles


def _run(tmp_path, **kw):
    p = prepare.PrepareParams(sample_id=kw.pop("sample_id", "s"), image=TEST_DATA / "he_synthetic.tif",
                              out_prefix=str(tmp_path / "s"), **kw)
    return prepare.run(p)


def _tiles(tmp_path, subset=None):
    return read_tiles(tmp_path / "s.tiles.tsv.gz", subset=subset)


def test_hd_grid(tmp_path):
    rep = _run(tmp_path, mpp_target=0.5, spatial=TEST_DATA / "hd_test/outs", spatial_format="spaceranger_hd")
    assert rep["status"] == "ok" and rep["tiling"] == "grid"
    assert rep["native"]["mpp"] == pytest.approx(0.25)
    tiles = _tiles(tmp_path)
    kept = tiles[tiles["keep"]]
    assert (kept["x0_px"] % 224 == 0).all() and (kept["y0_px"] % 224 == 0).all()
    assert (kept["tissue_fraction"] >= 0.5).all()
    assert (tiles["extract"] == tiles["keep"]).all()  # default: same threshold
    # full interior tiles hold exactly 14 x 14 bins of 8 µm (112 µm / 8 µm)
    assert kept["n_st_units"].max() == 196
    a = pd.read_csv(tmp_path / "s.assignment.tsv.gz", sep="\t")
    assert a["barcode"].is_unique and (a["weight"] == 1).all()  # every bin in at most one tile
    assert set(a["tile_id"]) <= set(tiles["tile_id"])
    # assignment agrees with the geometry: bin centre inside its tile
    pos = pd.read_parquet(TEST_DATA / "hd_test/outs/binned_outputs/square_008um/spatial/tissue_positions.parquet")
    m = a.merge(pos, on="barcode").merge(tiles, on="tile_id")
    x = m["pxl_col_in_fullres"] * rep["scale_factor"]
    y = m["pxl_row_in_fullres"] * rep["scale_factor"]
    assert ((x >= m["x0_px"]) & (x < m["x0_px"] + 224) & (y >= m["y0_px"]) & (y < m["y0_px"] + 224)).all()
    agreement = rep["mask"]["spaceranger_agreement"]
    assert agreement["in_tissue_units_inside_mask"] > 0.95 and agreement["background_units_outside_mask"] > 0.8


def test_visium_default_is_a_grid_with_spots_split_by_area(tmp_path):
    rep = _run(tmp_path, mpp_target=0.5, spatial=TEST_DATA / "visium_test/outs", spatial_format="spaceranger",
               mpp_native=0.25)
    assert rep["tiling"] == "grid"
    tiles = _tiles(tmp_path)
    assert (tiles["x0_px"] % 224 == 0).all() and (tiles["y0_px"] % 224 == 0).all()  # no overlaps
    a = pd.read_csv(tmp_path / "s.assignment.tsv.gz", sep="\t")
    per_spot = a.groupby("barcode")["weight"].sum()
    assert (per_spot <= 1 + 1e-9).all() and (a["weight"] > 0).all()
    assert (a.groupby("barcode").size() > 1).any()  # some spots straddle tile borders
    equiv = a.groupby("tile_id")["weight"].sum()
    t = tiles.set_index("tile_id")
    np.testing.assert_allclose(t.loc[equiv.index, "st_units_equiv"], equiv.to_numpy(), rtol=1e-6)
    # a 55 µm spot covers pi * 27.5^2 / 112^2 = 19% of a 112 µm tile
    np.testing.assert_allclose(t.loc[equiv.index, "st_coverage"], equiv.to_numpy() * np.pi * 27.5**2 / 112**2,
                               rtol=1e-3)
    assert rep["spatial"]["visium_grid"]["spot_area_on_the_lattice"] > 0.9


def test_visium_spot_tiles_on_request(tmp_path):
    rep = _run(tmp_path, mpp_target=0.5, spatial=TEST_DATA / "visium_test/outs", spatial_format="spaceranger",
               mpp_native=0.25, tiling="spot")
    assert rep["tiling"] == "spot"
    tiles = _tiles(tmp_path)
    pos = pd.read_csv(TEST_DATA / "visium_test/outs/spatial/tissue_positions.csv")
    pos = pos[pos["in_tissue"] == 1].set_index("barcode")
    t = tiles.set_index("barcode").loc[pos.index]
    # tile centre within half a pixel of the spot centre (in resampled pixels)
    np.testing.assert_allclose(t["cx_px"], pos["pxl_col_in_fullres"] * 0.5, atol=0.5)
    np.testing.assert_allclose(t["cy_px"], pos["pxl_row_in_fullres"] * 0.5, atol=0.5)
    assert (tiles["n_st_units"] == 1).all()
    # the same through the visium_tiling default
    rep2 = _run(tmp_path, mpp_target=0.5, spatial=TEST_DATA / "hest_test.h5ad", spatial_format="hest",
                mpp_native=0.25, visium_tiling="spot")
    assert rep2["tiling"] == "spot"


def test_upsampling_is_refused(tmp_path):
    rep = _run(tmp_path, mpp_target=0.2, spatial=TEST_DATA / "hd_test/outs", spatial_format="spaceranger_hd")
    assert rep["status"] == "skipped"
    assert json.loads((tmp_path / "s.prepare.json").read_text())["status"] == "skipped"
    assert not (tmp_path / "s.tiles.tsv.gz").exists()
    assert lineage.load(tmp_path / "s.prepare.lineage.json")["record"]["status"] == "skipped"


def test_image_only_sample(tmp_path):
    rep = _run(tmp_path, mpp_target=1.0)
    assert rep["status"] == "ok"
    a = pd.read_csv(tmp_path / "s.assignment.tsv.gz", sep="\t")
    assert len(a) == 0


def test_extraction_threshold_below_the_keep_threshold(tmp_path):
    # at 1 µm/px the left tile (224 µm) is about half glass
    rep = _run(tmp_path, mpp_target=1.0, min_tissue=0.8, min_tissue_extract=0.3)
    tiles = _tiles(tmp_path)
    assert (tiles.loc[tiles["keep"], "extract"]).all()
    between = tiles[(tiles["tissue_fraction"] >= 0.3) & (tiles["tissue_fraction"] < 0.8)]
    assert len(between) > 0 and between["extract"].all() and not between["keep"].any()
    assert between["qc_fail"].str.contains("low_tissue").all()
    assert rep["tiles"]["n_extract"] > rep["tiles"]["n_keep"]
    assert rep["tissue_thresholds"]["0.3"] >= rep["tissue_thresholds"]["0.5"] >= rep["tissue_thresholds"]["0.8"]
    assert len(read_tiles(tmp_path / "s.tiles.tsv.gz")) == rep["tiles"]["n_extract"]
    with pytest.raises(ValueError, match="min_tissue_extract"):
        _run(tmp_path, mpp_target=0.5, min_tissue=0.3, min_tissue_extract=0.5)


@pytest.mark.parametrize("method", ["grey", "od"])
def test_other_mask_methods(tmp_path, method):
    rep = _run(tmp_path, mpp_target=1.0, mask=mask.MaskParams(method=method))
    assert rep["mask"]["method"] == method and 0.3 < rep["mask"]["tissue_fraction_of_image"] < 0.9


def test_external_mask_replaces_the_automatic_one(tmp_path):
    auto = _run(tmp_path, mpp_target=1.0)
    auto_mask = np.asarray(Image.open(tmp_path / "s.mask.png")) > 0
    rep = _run(tmp_path, mpp_target=1.0, external_mask=TEST_DATA / "he_synthetic_mask.png")
    assert rep["mask"]["method"] == "external"
    ext_mask = np.asarray(Image.open(tmp_path / "s.mask.png")) > 0
    assert mask.dice(auto_mask, ext_mask) > 0.9
    assert auto["tiles"]["n_keep"] > 0 and rep["tiles"]["n_keep"] > 0
    rec = lineage.load(tmp_path / "s.prepare.lineage.json")["record"]
    assert rec["inputs"]["mask"]["name"] == "he_synthetic_mask.png"


def test_mask_comparison(tmp_path):
    rep = _run(tmp_path, mpp_target=0.5, spatial=TEST_DATA / "visium_test/outs", spatial_format="spaceranger",
               mpp_native=0.25, mask_compare=True, mask_compare_mpp=(1.0, 0.5, 0.1))
    table = pd.read_csv(tmp_path / "s.mask_compare.tsv", sep="\t")
    assert list(table["mask"][:3]) == ["used", "grey @ 4 um/px", "od @ 4 um/px"]
    used = table[table["mask"] == "used"].iloc[0]
    assert used["dice_vs_used"] == 1.0 and used["n_tiles_gained"] == 0 and used["n_tiles_lost"] == 0
    fine = table[table["mask"] == "saturation @ 0.5 um/px"].iloc[0]
    assert fine["dice_vs_used"] > 0.9
    assert table[table["mask"] == "saturation @ 0.1 um/px"]["note"].iloc[0].startswith("skipped")
    assert len(rep["mask_compare"]) == len(table)
    img = Image.open(tmp_path / "s.mask_compare.png")
    assert img.width >= 960


def test_mask_comparison_in_the_tissue_box_when_the_image_is_large(tmp_path, monkeypatch):
    from tileparity_lib import maskcompare

    monkeypatch.setattr(maskcompare, "MARGIN_UM", 10.0)  # the synthetic slide is smaller than the default margin
    # 0.70 Mpx at 0.5 µm/px; the tissue box (+ 10 µm) is about 0.57 Mpx
    _run(tmp_path, mpp_target=1.0, mask_compare=True, mask_compare_mpp=(0.5,), mask_compare_max_mpx=0.6)
    table = pd.read_csv(tmp_path / "s.mask_compare.tsv", sep="\t")
    row = table[table["mask"] == "saturation @ 0.5 um/px"].iloc[0]
    assert "bounding box" in row["note"] and row["dice_vs_used"] > 0.85


def test_figures_carry_their_explanation(tmp_path):
    _run(tmp_path, mpp_target=0.5, spatial=TEST_DATA / "hd_test/outs", spatial_format="spaceranger_hd")
    thumb = Image.open(tmp_path / "s.thumbnail.png")
    for name in ("overlay", "tissue_thresholds"):
        fig = Image.open(tmp_path / f"s.{name}.png")
        assert fig.height > thumb.height + 100 and fig.width >= 960  # title, legend and caption around the image


def test_lineage_of_process_1(tmp_path):
    _run(tmp_path, mpp_target=1.0, spatial=TEST_DATA / "visium_test/outs", spatial_format="spaceranger")
    lin = lineage.load(tmp_path / "s.prepare.lineage.json")
    rec = lin["record"]
    assert rec["step"] == "prepare" and rec["parents"] == [] and lin["ancestors"] == {}
    assert len(rec["inputs"]["image"]["sha256"]) == 64 and rec["inputs"]["spatial"]["directory"]
    assert {"tiles", "assignment", "overlay", "tissue_thresholds", "report"} <= set(rec["outputs"])
    assert rec["parameters"]["mpp_target"] == 1.0


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
