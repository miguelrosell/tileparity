"""Process 1 - common preparation (the only step that makes geometric decisions).

Input1: a full-resolution H&E image (+ optionally the Space Ranger / HEST positions).
Output1 (= Input2 of every extractor):

* ``<prefix>.resampled.tif``  - the image at the target µm/px (tiled BigTIFF, lossless).
* ``<prefix>.tiles.tsv.gz``   - one row per candidate tile: position, QC metrics, ``keep``.
* ``<prefix>.assignment.tsv.gz`` - spatial-transcriptomics unit -> tile (may be empty).
* ``<prefix>.thumbnail.png``, ``<prefix>.mask.png``, ``<prefix>.overlay.png`` - QC figures.
* ``<prefix>.prepare.json``   - every number needed to reproduce or audit the step.
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
from PIL import Image, ImageDraw

from . import __version__, image, mask, qc, spatial, tiling

log = logging.getLogger(__name__)


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
    hd_bin_um: int = 8
    tile_px: int = 224
    min_tissue: float = 0.5
    min_st_coverage: float = 0.5
    upsample_tolerance: float = 0.02
    mpp_mismatch_tolerance: float = 0.03
    resample_kernel: str = "lanczos3"
    compression: str = "deflate"
    blur_mad: float = 3.0
    drop_blurred: bool = True
    mask: mask.MaskParams = field(default_factory=mask.MaskParams)
    metadata: dict = field(default_factory=dict)


def _choose_tiling(p: PrepareParams) -> str:
    if p.tiling != "auto":
        return p.tiling
    return "spot" if p.spatial_format in ("spaceranger", "hest") else "grid"


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
    authoritative = MPP_AUTHORITATIVE
    for k, v in candidates.items():
        if k == chosen or abs(v - value) / value <= p.mpp_mismatch_tolerance:
            continue
        msg = (f"{p.sample_id}: pixel size from {k} ({v:.5f} µm/px) differs from {chosen} ({value:.5f} µm/px) "
               f"by more than {100 * p.mpp_mismatch_tolerance:.0f}%")
        if k in authoritative and chosen in authoritative:
            raise ValueError(msg + ". Check the 'mpp' column of the samplesheet and that the image is the one "
                                   "Space Ranger was given.")
        log.warning("%s", msg)
    return value, chosen, candidates


def _write_tsv_gz(df: pd.DataFrame, path: str) -> None:
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, sep="\t", index=False)


def run(p: PrepareParams) -> dict:
    report: dict = {"tileparity_version": __version__, "sample_id": p.sample_id, "mpp_target": p.mpp_target,
                    "scale_tag": tiling.scale_tag(p.mpp_target), "tile_px": p.tile_px,
                    "tile_um": p.tile_px * p.mpp_target, "params": _jsonable(asdict(p))}
    json_path = f"{p.out_prefix}.prepare.json"

    native = image.open_rgb(p.image, access="sequential")  # only streamed through resize
    sp = spatial.resolve(p.spatial, p.spatial_format, p.hd_bin_um) if p.spatial else None
    positions = spatial.read_positions(sp) if sp is not None else None
    if positions is not None and len(positions) == 0:
        raise ValueError(f"{p.sample_id}: no spatial units found in {sp.positions}")
    mode = _choose_tiling(p)
    if mode == "spot" and sp is None:
        raise ValueError(f"{p.sample_id}: tiling 'spot' needs spatial data (Visium positions)")

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
        log.warning("%s", report["reason"])
        return report

    # ---- 1. resample once, write a random-access copy -------------------------------
    resampled_path = f"{p.out_prefix}.resampled.tif"
    image.write_tiled_tiff(image.resample(native, scale, p.resample_kernel), resampled_path,
                           mpp=p.mpp_target, compression=p.compression)
    target = image.open_rgb(resampled_path)
    W, H = target.width, target.height
    report["resampled"] = {"width": W, "height": H, "path": Path(resampled_path).name}

    # ---- 2. tissue mask on a thumbnail ------------------------------------------------
    thumb_img = image.resample(target, p.mpp_target / p.mask.mask_mpp, "linear")
    thumb = image.to_numpy(thumb_img)[:, :, :3]
    sx, sy = thumb.shape[1] / W, thumb.shape[0] / H  # thumbnail px per resampled px
    tissue, mask_info = mask.tissue_mask(thumb, p.mask)
    # multiply a position by these factors to get thumbnail pixels (like Space Ranger's
    # tissue_lowres_scalef, which converts full-resolution pixels to the low-res image)
    report["thumbnail"] = {"width": thumb.shape[1], "height": thumb.shape[0], "mpp": p.mask.mask_mpp,
                           "resampled_to_thumbnail_x": sx, "resampled_to_thumbnail_y": sy,
                           "fullres_to_thumbnail": thumb.shape[1] / native.width}
    report["mask"] = mask_info
    if mask_info["tissue_fraction_of_slide"] > 0.9:
        log.warning("%s: the tissue mask covers %.0f%% of the image. If the image is cropped to the tissue "
                    "(no glass visible), check %s.mask.png", p.sample_id,
                    100 * mask_info["tissue_fraction_of_slide"], Path(p.out_prefix).name)
    Image.fromarray(thumb).save(f"{p.out_prefix}.thumbnail.png")
    Image.fromarray((tissue * 255).astype(np.uint8)).save(f"{p.out_prefix}.mask.png")

    # ---- 3. candidate tiles -------------------------------------------------------------
    T = p.tile_px
    if positions is not None:
        x_t, y_t = positions["x_fullres"].to_numpy() * scale, positions["y_fullres"].to_numpy() * scale
        inside = (x_t >= 0) & (y_t >= 0) & (x_t < W) & (y_t < H)
        # Space Ranger may place units of the capture area outside a tightly cropped image, but
        # the units it called in-tissue lie on the tissue, which the image must contain.
        in_tissue = positions["in_tissue"].to_numpy() == 1
        inside_tissue = float(inside[in_tissue].mean()) if in_tissue.any() else float(inside.mean())
        report["spatial"] = {"format": sp.fmt, "root": str(sp.root), "unit_size_um": sp.unit_size_um,
                             "n_units": int(len(positions)), "n_in_tissue": int(in_tissue.sum()),
                             "fraction_inside_image": float(inside.mean()),
                             "fraction_in_tissue_inside_image": inside_tissue}
        if inside_tissue < 0.99:
            raise ValueError(
                f"{p.sample_id}: only {inside_tissue:.1%} of the in-tissue spatial units fall inside the image. "
                "The image is probably not the one Space Ranger used (or the pixel size is wrong)."
            )

    if mode == "grid":
        tiles = tiling.grid_tiles(W, H, T)
        tiles["inside_image"] = True
        tiles["barcode"] = ""
        n_rows, n_cols = H // T, W // T
    else:
        units = positions[positions["in_tissue"] == 1].reset_index(drop=True)
        tiles = tiling.spot_tiles(units["x_fullres"].to_numpy() * scale, units["y_fullres"].to_numpy() * scale, T, W, H)
        tiles["barcode"] = units["barcode"].to_numpy()

    tiles["tissue_fraction"] = mask.rect_fractions(tissue, tiles["x0_px"].to_numpy(), tiles["y0_px"].to_numpy(),
                                                   T, (sx + sy) / 2.0)
    n_grid_total = len(tiles)
    if mode == "grid":
        tiles = tiles[tiles["tissue_fraction"] > 0].reset_index(drop=True)
        tiles["tile_id"] = tiling.grid_tile_ids(p.sample_id, tiling.scale_tag(p.mpp_target),
                                                tiles["row"].to_numpy(), tiles["col"].to_numpy())
    else:
        tiles["tile_id"] = tiling.spot_tile_ids(p.sample_id, tiling.scale_tag(p.mpp_target), tiles["barcode"].to_numpy())

    # ---- 4. spatial units -> tiles (the only place where this assignment is decided) ----
    assignment = pd.DataFrame({"barcode": pd.Series(dtype=str), "tile_id": pd.Series(dtype=str)})
    tiles["st_coverage"] = np.nan
    tiles["n_st_units"] = 0
    if positions is not None:
        if mode == "grid":
            row, col = tiling.assign_to_grid(x_t, y_t, T, n_rows, n_cols)
            # lookup table (row, col) -> position in `tiles`; -1 for tiles without tissue
            lookup = np.full((max(n_rows, 1), max(n_cols, 1)), -1, dtype=np.int64)
            lookup[tiles["row"].to_numpy(), tiles["col"].to_numpy()] = np.arange(len(tiles))
            tile_idx = np.where(row >= 0, lookup[np.maximum(row, 0), np.maximum(col, 0)], -1)
            if sp.fmt == "spaceranger_hd":
                any_count = np.bincount(tile_idx[tile_idx >= 0], minlength=len(tiles))
                tile_um2 = (T * p.mpp_target) ** 2
                tiles["st_coverage"] = any_count * sp.unit_size_um**2 / tile_um2
            sel = (tile_idx >= 0) & (positions["in_tissue"].to_numpy() == 1)
            assignment = pd.DataFrame({"barcode": positions["barcode"].to_numpy()[sel],
                                       "tile_id": tiles["tile_id"].to_numpy()[tile_idx[sel]]})
        else:
            assignment = pd.DataFrame({"barcode": tiles["barcode"].to_numpy(), "tile_id": tiles["tile_id"].to_numpy()})
        counts = assignment["tile_id"].value_counts()
        tiles["n_st_units"] = tiles["tile_id"].map(counts).fillna(0).astype(int)

    # ---- 5. QC metrics for tiles that pass the tissue filter ------------------------------
    eligible = (tiles["tissue_fraction"] >= p.min_tissue) & tiles["inside_image"]
    lapvar = np.full(len(tiles), np.nan)
    pen = np.full(len(tiles), np.nan)
    dark = np.full(len(tiles), np.nan)
    for i in np.flatnonzero(eligible.to_numpy()):
        rgb = image.read_region(target, int(tiles.at[i, "x0_px"]), int(tiles.at[i, "y0_px"]), T, T)
        lapvar[i], pen[i], dark[i] = qc.tile_metrics(rgb[:, :, :3])
    tiles["lapvar"], tiles["pen_fraction"], tiles["dark_fraction"] = lapvar, pen, dark
    tiles["blur_flag"] = qc.blur_flags(lapvar, eligible.to_numpy(), p.blur_mad)
    tiles["pen_flag"] = tiles["pen_fraction"] > 0.05
    tiles["dark_flag"] = tiles["dark_fraction"] > 0.25

    # ---- 6. keep decision ---------------------------------------------------------------------
    reasons = pd.DataFrame(index=tiles.index)
    reasons["outside_image"] = ~tiles["inside_image"]
    reasons["low_tissue"] = tiles["tissue_fraction"] < p.min_tissue
    if positions is not None:
        if mode == "grid" and sp.fmt == "spaceranger_hd":
            reasons["low_st_coverage"] = tiles["st_coverage"] < p.min_st_coverage
        reasons["no_st_units"] = tiles["n_st_units"] == 0
    if p.drop_blurred:
        reasons["blurred"] = tiles["blur_flag"]
    tiles["keep"] = ~reasons.any(axis=1)
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
               "inside_image", "tissue_fraction", "st_coverage", "n_st_units", "lapvar", "pen_fraction",
               "dark_fraction", "blur_flag", "pen_flag", "dark_flag", "keep", "qc_fail"] + list(p.metadata)
    tiles = tiles[columns]
    _write_tsv_gz(tiles, f"{p.out_prefix}.tiles.tsv.gz")
    if len(assignment):
        assignment = assignment[assignment["tile_id"].isin(set(tiles["tile_id"]))]
    _write_tsv_gz(assignment, f"{p.out_prefix}.assignment.tsv.gz")

    _overlay(thumb, tiles, positions, scale, sx, sy, sp, p.mpp_target, p.min_tissue, f"{p.out_prefix}.overlay.png")

    fail_counts = {c: int(reasons[c].sum()) for c in reasons.columns}
    report["tiles"] = {"n_grid_total": int(n_grid_total) if mode == "grid" else None,
                       "n_candidates": int(len(tiles)), "n_keep": int(tiles["keep"].sum()),
                       "fail_counts": fail_counts,
                       "n_blur_flag": int(tiles["blur_flag"].sum()), "n_pen_flag": int(tiles["pen_flag"].sum()),
                       "n_dark_flag": int(tiles["dark_flag"].sum())}
    if positions is not None:
        report["spatial"]["n_assigned_to_candidate_tiles"] = int(len(assignment))
        report["spatial"]["n_assigned_to_kept_tiles"] = int(
            assignment["tile_id"].isin(set(tiles.loc[tiles["keep"], "tile_id"])).sum())
    report["status"] = "ok"
    Path(json_path).write_text(json.dumps(_jsonable(report), indent=2))
    if tiles["keep"].sum() == 0:
        log.warning("%s: no tile passed QC at %s µm/px", p.sample_id, p.mpp_target)
    return report


def _overlay(thumb, tiles, positions, scale, sx, sy, sp, mpp_target, min_tissue, path) -> None:
    """Thumbnail with spatial units (blue tint) and tiles (green = kept, red = dropped
    although they hold enough tissue).

    The blue tint marks where Space Ranger placed the in-tissue units. If it is shifted
    relative to the tissue, the image is not the one Space Ranger was given."""
    canvas = thumb.astype(np.float32).copy()
    if positions is not None:
        units = positions[positions["in_tissue"] == 1]
        ux = np.floor(units["x_fullres"].to_numpy() * scale * sx).astype(int)
        uy = np.floor(units["y_fullres"].to_numpy() * scale * sy).astype(int)
        ok = (ux >= 0) & (uy >= 0) & (ux < canvas.shape[1]) & (uy < canvas.shape[0])
        layer = np.zeros(canvas.shape[:2], bool)
        radius = 0
        if sp is not None and sp.fmt in ("spaceranger", "hest"):
            radius = max(int(round(sp.unit_size_um / 2 / (mpp_target / sx))), 1)
        if radius <= 1:
            layer[uy[ok], ux[ok]] = True
        else:
            yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            disk = (xx * xx + yy * yy) <= radius * radius
            for x, y in zip(ux[ok], uy[ok]):
                y0, y1 = max(y - radius, 0), min(y + radius + 1, canvas.shape[0])
                x0, x1 = max(x - radius, 0), min(x + radius + 1, canvas.shape[1])
                layer[y0:y1, x0:x1] |= disk[y0 - (y - radius):y1 - (y - radius), x0 - (x - radius):x1 - (x - radius)]
        canvas[layer] = 0.6 * canvas[layer] + 0.4 * np.array([0, 110, 255], np.float32)
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for _, t in tiles.iterrows():
        if not t["keep"] and t["tissue_fraction"] < min_tissue:
            continue
        x0, y0 = t["x0_px"] * sx, t["y0_px"] * sy
        x1, y1 = (t["x0_px"] + t["tile_px"]) * sx, (t["y0_px"] + t["tile_px"]) * sy
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 170, 0) if t["keep"] else (220, 0, 0))
    img.save(path)


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
