"""Process 1 - common preparation (the only step that makes geometric decisions).

Input1: a full-resolution H&E image (+ optionally the Space Ranger / HEST positions and a
user-provided tissue mask).
Output1 (= Input2 of every extractor):

* ``<prefix>.resampled.tif``  - the image at the target µm/px (tiled BigTIFF, lossless).
* ``<prefix>.tiles.tsv.gz``   - one row per candidate tile: position, tissue fraction, QC
  metrics, ``extract`` (features are computed) and ``keep`` (passes every threshold).
* ``<prefix>.assignment.tsv.gz`` - spatial-transcriptomics unit -> tile, with a weight
  (1 for Visium HD bins and spot-centred tiles; the share of the spot's area for
  Visium spots on the grid).
* ``<prefix>.thumbnail.png``, ``<prefix>.mask.png`` - the thumbnail and the tissue mask (data).
* ``<prefix>.overlay.png``, ``<prefix>.tissue_thresholds.png`` and, with
  ``--mask_compare``, ``<prefix>.mask_compare.png`` + ``.tsv`` - QC figures with their
  explanation printed underneath.
* ``<prefix>.prepare.json``   - every number needed to reproduce or audit the step.
* ``<prefix>.prepare.lineage.json`` - inputs, parameters, software and checksums.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import __version__, image, lineage, mask, maskcompare, panels, qc, spatial, tiling
from .spatial import VISIUM_SPOT_DIAMETER_UM

log = logging.getLogger(__name__)

VISIUM_FORMATS = ("spaceranger", "hest")


@dataclass
class PrepareParams:
    sample_id: str
    image: Path
    mpp_target: float
    out_prefix: str
    mpp_native: float | None = None  # from the samplesheet; overrides everything else
    spatial: Path | None = None
    spatial_format: str | None = None
    tiling: str = "auto"  # auto | grid | spot
    visium_tiling: str = "grid"  # what "auto" means for standard Visium and HEST: grid | spot
    visium_spot_um: float = VISIUM_SPOT_DIAMETER_UM
    hd_bin_um: int = 8
    tile_px: int = 224
    min_tissue: float = 0.5
    min_tissue_extract: float | None = None  # None: same as min_tissue
    min_st_coverage: float = 0.5  # Visium HD: share of the tile covered by bins
    min_spot_overlap: float = 0.25  # Visium grid: spots' worth of capture area inside the tile
    upsample_tolerance: float = 0.02
    mpp_mismatch_tolerance: float = 0.03
    resample_kernel: str = "lanczos3"
    compression: str = "deflate"
    blur_mad: float = 3.0
    drop_blurred: bool = False  # blurred tiles are flagged; dropped only on request
    mask: mask.MaskParams = field(default_factory=mask.MaskParams)
    external_mask: Path | None = None  # user-provided tissue mask (replaces the automatic one)
    mask_compare: bool = False
    mask_compare_mpp: tuple = (1.0, 0.5)
    mask_compare_max_mpx: float = 400.0
    tissue_thresholds: tuple = (0.3, 0.5, 0.8)
    metadata: dict = field(default_factory=dict)
    process: str = "TILEPARITY_PREPARE"

    @property
    def extract_threshold(self) -> float:
        return self.min_tissue if self.min_tissue_extract is None else self.min_tissue_extract


def _choose_tiling(p: PrepareParams) -> str:
    if p.tiling != "auto":
        return p.tiling
    return p.visium_tiling if p.spatial_format in VISIUM_FORMATS else "grid"


# Sources of the native pixel size, in order of priority. The first four describe the
# pixel grid of the spatial coordinates themselves, so two of them disagreeing stops the
# run; the image metadata and Space Ranger's plotting diameter only produce a warning.
MPP_SOURCES = ("samplesheet", "scalefactors:microns_per_pixel", "scalefactors:bin_size_um/spot_diameter_fullres",
               "positions:unit_spacing", "image_metadata", "scalefactors:65um/spot_diameter_fullres")
MPP_AUTHORITATIVE = frozenset(MPP_SOURCES[:4])


def _native_mpp(p: PrepareParams, img, sp: spatial.SpatialPaths | None,
                positions: pd.DataFrame | None) -> tuple[float, str, dict]:
    """Pick the native pixel size and cross-check every available source."""
    candidates: dict[str, float] = {}
    if p.mpp_native:
        candidates["samplesheet"] = float(p.mpp_native)
    if sp is not None:
        sf_mpp, sf_src = spatial.mpp_from_scalefactors(sp)
        if sf_mpp:
            candidates[sf_src] = sf_mpp
        pitch = spatial.unit_pitch_um(sp)
        if pitch and positions is not None:
            spacing_mpp = spatial.mpp_from_pitch(positions, pitch)
            if spacing_mpp:
                candidates["positions:unit_spacing"] = spacing_mpp
    meta_mpp = image.mpp_from_metadata(img)
    if meta_mpp:
        candidates["image_metadata"] = meta_mpp

    chosen = next((k for k in MPP_SOURCES if k in candidates), None)
    if chosen is None:
        raise ValueError(
            f"{p.sample_id}: cannot determine the native pixel size. Add an 'mpp' column (µm/px) "
            "to the samplesheet for this sample."
        )
    value = candidates[chosen]
    for k, v in candidates.items():
        if k == chosen or abs(v - value) / value <= p.mpp_mismatch_tolerance:
            continue
        msg = (f"{p.sample_id}: pixel size from {k} ({v:.5f} µm/px) differs from {chosen} ({value:.5f} µm/px) "
               f"by more than {100 * p.mpp_mismatch_tolerance:.0f}%")
        if k in MPP_AUTHORITATIVE and chosen in MPP_AUTHORITATIVE:
            raise ValueError(msg + ". Check the 'mpp' column of the samplesheet and that the image is the one "
                                   "Space Ranger was given.")
        log.warning("%s", msg)
    return value, chosen, candidates


def _write_tsv_gz(df: pd.DataFrame, path: str) -> None:
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, sep="\t", index=False)


def _inputs(p: PrepareParams) -> dict:
    return {"image": p.image, "spatial": p.spatial, "mask": p.external_mask}


def run(p: PrepareParams) -> dict:
    if not 0 <= p.extract_threshold <= p.min_tissue <= 1:
        raise ValueError("tissue thresholds must satisfy 0 <= min_tissue_extract <= min_tissue <= 1")
    report: dict = {"tileparity_version": __version__, "sample_id": p.sample_id, "mpp_target": p.mpp_target,
                    "scale_tag": tiling.scale_tag(p.mpp_target), "tile_px": p.tile_px,
                    "tile_um": p.tile_px * p.mpp_target, "params": _jsonable(asdict(p))}
    json_path = f"{p.out_prefix}.prepare.json"
    outputs: dict = {"report": json_path}

    native = image.open_rgb(p.image, access="sequential")  # only streamed through resize
    sp = spatial.resolve(p.spatial, p.spatial_format, p.hd_bin_um) if p.spatial else None
    positions = spatial.read_positions(sp) if sp is not None else None
    if positions is not None and len(positions) == 0:
        raise ValueError(f"{p.sample_id}: no spatial units found in {sp.positions}")
    mode = _choose_tiling(p)
    if mode == "spot" and (sp is None or sp.fmt not in VISIUM_FORMATS):
        raise ValueError(f"{p.sample_id}: tiling 'spot' needs standard Visium spots (spaceranger or hest)")
    visium_grid = mode == "grid" and sp is not None and sp.fmt in VISIUM_FORMATS

    mpp_native, mpp_source, mpp_candidates = _native_mpp(p, native, sp, positions)
    scale = mpp_native / p.mpp_target  # resampled px = native px * scale
    report["native"] = {"width": native.width, "height": native.height, "mpp": mpp_native,
                        "mpp_source": mpp_source, "mpp_candidates": mpp_candidates}
    report["scale_factor"] = scale
    report["tiling"] = mode

    if scale > 1.0 + p.upsample_tolerance:
        report.update(status="skipped", reason=(
            f"target {p.mpp_target} µm/px is finer than the native {mpp_native:.4f} µm/px; "
            f"reaching it would need upsampling by {scale:.3f}x (> {1 + p.upsample_tolerance:.2f}x)"))
        Path(json_path).write_text(json.dumps(report, indent=2))
        _write_lineage(p, report, outputs)
        log.warning("%s", report["reason"])
        return report

    # ---- 1. resample once, write a random-access copy -------------------------------
    resampled_path = f"{p.out_prefix}.resampled.tif"
    image.write_tiled_tiff(image.resample(native, scale, p.resample_kernel), resampled_path,
                           mpp=p.mpp_target, compression=p.compression)
    outputs["image"] = resampled_path
    target = image.open_rgb(resampled_path)
    W, H = target.width, target.height
    report["resampled"] = {"width": W, "height": H, "path": Path(resampled_path).name}

    # ---- 2. tissue mask on a thumbnail ------------------------------------------------
    thumb_img = image.resample(target, p.mpp_target / p.mask.mask_mpp, "linear")
    thumb = image.to_numpy(thumb_img)[:, :, :3]
    sx, sy = thumb.shape[1] / W, thumb.shape[0] / H  # thumbnail px per resampled px
    if p.external_mask:
        tissue, mask_info = mask.load_external(p.external_mask, thumb.shape[1], thumb.shape[0],
                                               native.width, native.height)
    else:
        tissue, mask_info = mask.tissue_mask(thumb, p.mask)
    # multiply a position by these factors to get thumbnail pixels (like Space Ranger's
    # tissue_lowres_scalef, which converts full-resolution pixels to the low-res image)
    report["thumbnail"] = {"width": thumb.shape[1], "height": thumb.shape[0], "mpp": p.mask.mask_mpp,
                           "resampled_to_thumbnail_x": sx, "resampled_to_thumbnail_y": sy,
                           "fullres_to_thumbnail": thumb.shape[1] / native.width}
    report["mask"] = mask_info
    if mask_info["tissue_fraction_of_image"] > 0.9:
        log.warning("%s: the tissue mask covers %.0f%% of the image. If the image is cropped to the tissue "
                    "(no glass visible), check %s.mask.png", p.sample_id,
                    100 * mask_info["tissue_fraction_of_image"], Path(p.out_prefix).name)
    Image.fromarray(thumb).save(f"{p.out_prefix}.thumbnail.png")
    Image.fromarray((tissue * 255).astype(np.uint8)).save(f"{p.out_prefix}.mask.png")
    outputs.update(thumbnail=f"{p.out_prefix}.thumbnail.png", mask=f"{p.out_prefix}.mask.png")

    # ---- 3. candidate tiles -------------------------------------------------------------
    T = p.tile_px
    if positions is not None:
        x_t, y_t = positions["x_fullres"].to_numpy() * scale, positions["y_fullres"].to_numpy() * scale
        inside = (x_t >= 0) & (y_t >= 0) & (x_t < W) & (y_t < H)
        # Space Ranger may place units of the capture area outside a tightly cropped image, but
        # the units it called in-tissue lie on the tissue, which the image must contain.
        in_tissue = positions["in_tissue"].to_numpy() == 1
        inside_tissue = float(inside[in_tissue].mean()) if in_tissue.any() else float(inside.mean())
        unit_um = p.visium_spot_um if sp.fmt in VISIUM_FORMATS else sp.unit_size_um
        report["spatial"] = {"format": sp.fmt, "root": str(sp.root), "unit_size_um": unit_um,
                             "n_units": int(len(positions)), "n_in_tissue": int(in_tissue.sum()),
                             "fraction_inside_image": float(inside.mean()),
                             "fraction_in_tissue_inside_image": inside_tissue}
        if inside_tissue < 0.99:
            raise ValueError(
                f"{p.sample_id}: only {inside_tissue:.1%} of the in-tissue spatial units fall inside the image. "
                "The image is probably not the one Space Ranger used (or the pixel size is wrong)."
            )
        # how well the tissue mask agrees with Space Ranger's own in-tissue call
        inside_mask = mask.lookup(tissue, x_t * sx, y_t * sy)
        report["mask"]["spaceranger_agreement"] = {
            "in_tissue_units_inside_mask": float(inside_mask[in_tissue].mean()) if in_tissue.any() else None,
            "background_units_outside_mask": float((~inside_mask[~in_tissue]).mean()) if (~in_tissue).any() else None,
        }

    n_rows, n_cols = H // T, W // T
    if mode == "grid":
        tiles = tiling.grid_tiles(W, H, T)
        tiles["inside_image"] = True
        tiles["barcode"] = ""
    else:
        units = positions[positions["in_tissue"] == 1].reset_index(drop=True)
        tiles = tiling.spot_tiles(units["x_fullres"].to_numpy() * scale, units["y_fullres"].to_numpy() * scale, T, W, H)
        tiles["barcode"] = units["barcode"].to_numpy()

    mask_scale = (sx + sy) / 2.0
    tiles["tissue_fraction"] = mask.rect_fractions(tissue, tiles["x0_px"].to_numpy(), tiles["y0_px"].to_numpy(),
                                                   T, mask_scale)
    lattice = tiles[["x0_px", "y0_px"]].copy()  # every possible tile, for the mask comparison
    n_grid_total = len(tiles)
    if mode == "grid":
        tiles = tiles[tiles["tissue_fraction"] > 0].reset_index(drop=True)
        tiles["tile_id"] = tiling.grid_tile_ids(p.sample_id, tiling.scale_tag(p.mpp_target),
                                                tiles["row"].to_numpy(), tiles["col"].to_numpy())
    else:
        tiles["tile_id"] = tiling.spot_tile_ids(p.sample_id, tiling.scale_tag(p.mpp_target), tiles["barcode"].to_numpy())

    # ---- 4. spatial units -> tiles (the only place where this assignment is decided) ----
    assignment = pd.DataFrame({"barcode": pd.Series(dtype=str), "tile_id": pd.Series(dtype=str),
                               "weight": pd.Series(dtype=float)})
    tiles["st_coverage"] = np.nan
    tiles["n_st_units"] = 0
    tiles["st_units_equiv"] = 0.0
    tile_area_px = float(T * T)
    if positions is not None:
        lookup = np.full((max(n_rows, 1), max(n_cols, 1)), -1, dtype=np.int64)
        if mode == "grid":
            lookup[tiles["row"].to_numpy(), tiles["col"].to_numpy()] = np.arange(len(tiles))
        if mode == "grid" and not visium_grid:  # Visium HD: each bin goes to the tile holding its centre
            row, col = tiling.assign_to_grid(x_t, y_t, T, n_rows, n_cols)
            tile_idx = np.where(row >= 0, lookup[np.maximum(row, 0), np.maximum(col, 0)], -1)
            any_count = np.bincount(tile_idx[tile_idx >= 0], minlength=len(tiles))
            tiles["st_coverage"] = any_count * (sp.unit_size_um / p.mpp_target) ** 2 / tile_area_px
            sel = (tile_idx >= 0) & in_tissue
            assignment = pd.DataFrame({"barcode": positions["barcode"].to_numpy()[sel],
                                       "tile_id": tiles["tile_id"].to_numpy()[tile_idx[sel]], "weight": 1.0})
        elif visium_grid:  # Visium spots on the grid: share of each spot's area in each tile
            units = positions[in_tissue].reset_index(drop=True)
            radius_px = p.visium_spot_um / 2.0 / p.mpp_target
            w = tiling.disc_tile_weights(units["x_fullres"].to_numpy() * scale, units["y_fullres"].to_numpy() * scale,
                                         radius_px, T, n_rows, n_cols)
            w["tile"] = lookup[w["row"].to_numpy(), w["col"].to_numpy()]
            spot_total = w.groupby("unit")["weight"].sum()
            w = w[w["tile"] >= 0]
            assignment = pd.DataFrame({"barcode": units["barcode"].to_numpy()[w["unit"].to_numpy()],
                                       "tile_id": tiles["tile_id"].to_numpy()[w["tile"].to_numpy()],
                                       "weight": w["weight"].to_numpy()})
            spot_area_px = math.pi * radius_px**2
            equiv = np.bincount(w["tile"].to_numpy(), weights=w["weight"].to_numpy(), minlength=len(tiles))
            tiles["st_units_equiv"] = equiv
            tiles["st_coverage"] = equiv * spot_area_px / tile_area_px
            report["spatial"]["visium_grid"] = {
                "spot_diameter_um": p.visium_spot_um,
                "spot_area_on_the_lattice": float(spot_total.sum() / max(len(units), 1)),
                "spot_area_on_candidate_tiles": float(w["weight"].sum() / max(len(units), 1)),
            }
        else:  # spot-centred tiles: one spot per tile
            assignment = pd.DataFrame({"barcode": tiles["barcode"].to_numpy(), "tile_id": tiles["tile_id"].to_numpy(),
                                       "weight": 1.0})
            radius_px = p.visium_spot_um / 2.0 / p.mpp_target
            tiles["st_coverage"] = math.pi * radius_px**2 / tile_area_px
        counts = assignment["tile_id"].value_counts()
        tiles["n_st_units"] = tiles["tile_id"].map(counts).fillna(0).astype(int)
        if not visium_grid:
            tiles["st_units_equiv"] = tiles["n_st_units"].astype(float)

    # ---- 5. QC metrics for tiles that pass the extraction threshold ------------------------
    eligible = (tiles["tissue_fraction"] >= p.extract_threshold) & tiles["inside_image"]
    lapvar = np.full(len(tiles), np.nan)
    pen = np.full(len(tiles), np.nan)
    dark = np.full(len(tiles), np.nan)
    for i in np.flatnonzero(eligible.to_numpy()):
        rgb = image.read_region(target, int(tiles.at[i, "x0_px"]), int(tiles.at[i, "y0_px"]), T, T)
        lapvar[i], pen[i], dark[i] = qc.tile_metrics(rgb[:, :, :3])
    tiles["lapvar"], tiles["pen_fraction"], tiles["dark_fraction"] = lapvar, pen, dark
    tiles["blur_flag"] = qc.blur_flags(lapvar, eligible.to_numpy(), p.blur_mad,
                                       reference=(tiles["tissue_fraction"] >= p.min_tissue).to_numpy())
    tiles["pen_flag"] = tiles["pen_fraction"] > 0.05
    tiles["dark_flag"] = tiles["dark_fraction"] > 0.25

    # ---- 6. extract / keep decisions ------------------------------------------------------------
    reasons = pd.DataFrame(index=tiles.index)
    reasons["outside_image"] = ~tiles["inside_image"]
    reasons["low_tissue"] = tiles["tissue_fraction"] < p.min_tissue
    if positions is not None:
        if mode == "grid" and not visium_grid:
            reasons["low_st_coverage"] = tiles["st_coverage"] < p.min_st_coverage
        elif visium_grid:
            reasons["low_st_coverage"] = tiles["st_units_equiv"] < p.min_spot_overlap
        reasons["no_st_units"] = tiles["n_st_units"] == 0
    if p.drop_blurred:
        reasons["blurred"] = tiles["blur_flag"]
    other = reasons.drop(columns="low_tissue").any(axis=1)
    tiles["keep"] = ~reasons.any(axis=1)
    tiles["extract"] = ~other & (tiles["tissue_fraction"] >= p.extract_threshold)
    tiles["qc_fail"] = reasons.apply(lambda r: ";".join(c for c in reasons.columns if r[c]), axis=1)

    # ---- 7. derived coordinates and sample-level metadata -------------------------------------
    tiles["cx_px"] = tiles["x0_px"] + T / 2.0
    tiles["cy_px"] = tiles["y0_px"] + T / 2.0
    tiles["cx_fullres"] = tiles["cx_px"] / scale
    tiles["cy_fullres"] = tiles["cy_px"] / scale
    tiles["cx_um"] = tiles["cx_px"] * p.mpp_target
    tiles["cy_um"] = tiles["cy_px"] * p.mpp_target
    tiles["unit_type"] = mode
    tiles["sample_id"] = p.sample_id
    tiles["scale_mpp"] = p.mpp_target
    tiles["tile_px"] = T
    tiles["tile_um"] = T * p.mpp_target
    for k, v in p.metadata.items():
        tiles[k] = v
    columns = ["tile_id", "sample_id", "scale_mpp", "tile_px", "tile_um", "unit_type", "row", "col", "barcode",
               "x0_px", "y0_px", "cx_px", "cy_px", "cx_fullres", "cy_fullres", "cx_um", "cy_um",
               "inside_image", "tissue_fraction", "st_coverage", "n_st_units", "st_units_equiv", "lapvar",
               "pen_fraction", "dark_fraction", "blur_flag", "pen_flag", "dark_flag", "extract", "keep",
               "qc_fail"] + list(p.metadata)
    tiles = tiles[columns]
    _write_tsv_gz(tiles, f"{p.out_prefix}.tiles.tsv.gz")
    if len(assignment):
        assignment = assignment[assignment["tile_id"].isin(set(tiles["tile_id"]))]
    _write_tsv_gz(assignment, f"{p.out_prefix}.assignment.tsv.gz")
    outputs.update(tiles=f"{p.out_prefix}.tiles.tsv.gz", assignment=f"{p.out_prefix}.assignment.tsv.gz")

    # ---- 8. QC figures (each with its explanation underneath) ----------------------------------
    source = lineage.source_text(p.process, _inputs(p), context=f"sample {p.sample_id}, {p.mpp_target:g} µm/px")
    unit_r = 0.0
    if sp is not None and sp.fmt in VISIUM_FORMATS:
        unit_r = p.visium_spot_um / 2.0 / p.mask.mask_mpp
    method_text = mask_info.get("method", "?") + ("" if p.external_mask else f" + Otsu at {p.mask.mask_mpp:g} µm/px")
    overlap_note = ""
    if mode == "spot" and T * p.mpp_target > spatial.VISIUM_SPOT_PITCH_UM:
        overlap_note = (f" Spot-centred tiles of {T * p.mpp_target:.0f} µm overlap their neighbours (spots are "
                        f"{spatial.VISIUM_SPOT_PITCH_UM:.0f} µm apart).")
    elif visium_grid:
        overlap_note = (" Grid tiles do not overlap; each spot's molecules are split between the tiles it touches, "
                        "in proportion to the spot area inside each tile.")
    fig = panels.overlay(thumb, tiles, positions, scale, sx, sy, unit_r, p.min_tissue, caption=dict(
        title=f"{p.sample_id} - {p.mpp_target:g} µm/px - {mode} tiles of {T * p.mpp_target:.0f} µm",
        what=(f"The slide (thumbnail at {p.mask.mask_mpp:g} µm/px) with the {int(tiles['keep'].sum())} kept tiles "
              f"(green) of {T} px = {T * p.mpp_target:.0f} µm" +
              (" and the in-tissue spatial units placed by Space Ranger (blue)." if positions is not None else ".")),
        how=("Blue must lie exactly on the tissue: a shift means that the image is not the one Space Ranger used "
             "or that the pixel size is wrong. Green tiles get features (and expression); orange tiles hold enough "
             "tissue but were dropped for another reason (column qc_fail of the tile table)."),
        caveats=(f"Tissue from the {method_text} mask; outlines are exact but drawn on the thumbnail." + overlap_note),
        source=source))
    fig.save(f"{p.out_prefix}.overlay.png")
    outputs["overlay"] = f"{p.out_prefix}.overlay.png"

    thr_fig, thr_counts = panels.thresholds(thumb, tiles, sx, sy, list(p.tissue_thresholds), caption=dict(
        title=f"{p.sample_id} - {p.mpp_target:g} µm/px - tissue-fraction thresholds",
        what=(f"Candidate tiles coloured by the highest tissue-fraction threshold they reach "
              f"({', '.join(f'{v:.0%}' for v in sorted(p.tissue_thresholds))}). This run extracts features from tiles "
              f"with >= {p.extract_threshold:.0%} tissue and marks tiles with >= {p.min_tissue:.0%} as keep."),
        how=("Compare how many tiles each threshold keeps and where the extra tiles are (tissue borders, holes, "
             "lumens). A threshold that removes whole structures is too strict; one that admits mostly glass is too "
             "loose."),
        caveats=(f"Only the tissue fraction is applied in this figure, from the {method_text} mask; the other "
                 "criteria (expression coverage, image border, blur with --drop_blurred) are not."),
        source=source))
    thr_fig.save(f"{p.out_prefix}.tissue_thresholds.png")
    outputs["tissue_thresholds"] = f"{p.out_prefix}.tissue_thresholds.png"
    report["tissue_thresholds"] = thr_counts

    if p.mask_compare:
        table, cells = maskcompare.compare(
            native_path=p.image, mpp_native=mpp_native, native_size=(native.width, native.height), thumb=thumb,
            sx=sx, sy=sy, scale=scale, mpp_target=p.mpp_target, used_mask=tissue, used_info=mask_info,
            params=p.mask, compare_mpp=list(p.mask_compare_mpp), max_mpx=p.mask_compare_max_mpx, lattice=lattice,
            tile_px=T, min_tissue=p.min_tissue, positions=positions, upsample_tolerance=p.upsample_tolerance)
        table.to_csv(f"{p.out_prefix}.mask_compare.tsv", sep="\t", index=False)
        cmp_fig = maskcompare.panel(thumb, table, cells, p.min_tissue, caption=dict(
            title=f"{p.sample_id} - {p.mpp_target:g} µm/px - tissue masks compared",
            what=("The tissue mask used for the tiles (first panel) and the alternatives: the other automatic methods "
                  f"at {p.mask.mask_mpp:g} µm/px and the {p.mask.method} method at finer resolutions, read from the "
                  "full-resolution image. Green: tissue in both masks; magenta: only in the alternative; orange: only "
                  "in the used mask."),
            how=("Small differences along the tissue border are expected. Compare the Dice overlap (1 = identical), "
                 f"the number of tiles that change their >= {p.min_tissue:.0%} decision and, for spatial samples, the "
                 "share of Space Ranger's in-tissue units inside the mask and of its background units outside it "
                 "(both high is good)."),
            caveats=("Space Ranger's tissue call is itself automatic (sometimes corrected by hand in Loupe): a "
                     "reference, not the truth. Fine resolutions of large slides are computed inside the tissue "
                     "bounding box, where Otsu's threshold sees less glass, and are drawn on the thumbnail grid."),
            source=source))
        cmp_fig.save(f"{p.out_prefix}.mask_compare.png")
        outputs.update(mask_compare=f"{p.out_prefix}.mask_compare.png",
                       mask_compare_table=f"{p.out_prefix}.mask_compare.tsv")
        report["mask_compare"] = json.loads(table.to_json(orient="records"))

    fail_counts = {c: int(reasons[c].sum()) for c in reasons.columns}
    report["tiles"] = {"n_grid_total": int(n_grid_total) if mode == "grid" else None,
                       "n_candidates": int(len(tiles)), "n_extract": int(tiles["extract"].sum()),
                       "n_keep": int(tiles["keep"].sum()), "fail_counts": fail_counts,
                       "n_blur_flag": int(tiles["blur_flag"].sum()), "n_pen_flag": int(tiles["pen_flag"].sum()),
                       "n_dark_flag": int(tiles["dark_flag"].sum())}
    if positions is not None:
        report["spatial"]["n_assignments_to_candidate_tiles"] = int(len(assignment))
        report["spatial"]["n_units_assigned_to_extracted_tiles"] = int(
            assignment.loc[assignment["tile_id"].isin(set(tiles.loc[tiles["extract"], "tile_id"])), "barcode"].nunique())
    report["status"] = "ok"
    Path(json_path).write_text(json.dumps(_jsonable(report), indent=2))
    _write_lineage(p, report, outputs)
    if tiles["keep"].sum() == 0:
        log.warning("%s: no tile passed QC at %s µm/px", p.sample_id, p.mpp_target)
    return report


def _write_lineage(p: PrepareParams, report: dict, outputs: dict) -> None:
    rec = lineage.build("prepare", p.process, Path(p.out_prefix).name, parameters=asdict(p), inputs=_inputs(p),
                        outputs=outputs, packages=["numpy", "scipy", "scikit-image", "pandas", "pyvips", "pillow"])
    rec["record"]["status"] = report.get("status")
    lineage.write(f"{p.out_prefix}.prepare.lineage.json", rec)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
