# FAQ

Short answers to the questions that come up first. The reasoning behind each choice is in [design.md](design.md).

## Input

**Does the H&E image need a particular resolution, size or magnification?**
No. tileparity only needs the pixel size in µm/px. It takes it from the samplesheet `mpp` column, from Space Ranger's files or from the image metadata. It then resamples the image to each target scale. It never enlarges an image (2 % tolerance), so a scale finer than the scan is skipped with a warning:

| scan | native µm/px | default scales computed |
|---|---|---|
| 40× | ~0.25 | 0.2857, 0.5, 1.0 |
| 20× | ~0.5 | 0.5, 1.0 |
| 10× | ~1.0 | 1.0 |

The image must be at least one tile (224 px) wide and high at the target scale.

**Which spatial technologies are supported?**
* Visium HD (`spatial_format = spaceranger_hd`): grid tiles.
* Standard Visium (`spaceranger`, or a HEST-1k `.h5ad` with `hest`): one tile centred on each spot.
* No spatial data: grid tiles, image features only.

If there is spatial data, the image must be the one Space Ranger (or HEST) used, because the coordinates are pixels of that image. Process 1 stops if the pixel sizes disagree or if the in-tissue units fall outside the image. `prepare/<id>.overlay.png` shows the alignment.

**Which file formats can the image be in?**
Anything libvips reads: TIFF, BigTIFF (`.btf`), OME-TIFF, Aperio SVS with JPEG compression, PNG and JPEG. The image must be RGB brightfield. An alpha channel is dropped and 16-bit images are converted to 8-bit. Pyramidal files are read at full resolution.

## What each extractor receives

**What exactly does fastLBP receive (Input2b)?**
An image, not precomputed features: a tiled TIFF with three 8-bit bands (H, E and grey, 0–255), computed pixel by pixel from the resampled RGB image, plus the tile table. `FASTLBP_EXTRACT` reads each kept tile together with a margin, so border pixels are compared with their real neighbours. It then computes the LBP codes and the histograms with fastLBP's kernel.

**What exactly does UNI2-h receive (Input2a)?**
The resampled RGB image plus the tile table. Each kept tile is processed as follows:

1. It is cut as 224 × 224 px with three values per pixel (R, G, B).
2. The values are scaled to [0, 1] and normalised with the ImageNet mean and standard deviation, giving a tensor of shape [3, 224, 224].

Tiles are processed in batches, and the output is one 1536-dimensional vector per tile.

**Do I have to set the patch size?**
No. `--tile_px 224` is the default and the same for both extractors (UNI2-h needs 224). The physical size of a tile follows from the scale: 64, 112 and 224 µm at 0.2857, 0.5 and 1.0 µm/px.

**Where is the background (glass) removed? Is there a bounding box?**
In Process 1, not in the adapters 1a/1b, which only change how pixels are represented. A tissue mask (saturation + Otsu on a 4 µm/px thumbnail) gives every tile a tissue fraction, and only tiles with at least 50 % tissue are kept (`--min_tissue`). The image is not cropped, for two reasons: background tiles are simply not used, and keeping the original frame keeps Space Ranger's coordinates valid.

## Spatial transcriptomics

**Do Visium spots always fit in the tiles?**
Yes, at all default scales. Each spot gets its own tile, centred on it, and the 55 µm spot is smaller than the tile (64, 112 or 224 µm). Two cases are handled automatically:
* A scale finer than the native resolution is skipped.
* A spot whose tile would extend beyond the image is dropped (`qc_fail = outside_image`).

At 1.0 µm/px the 224 µm tile also shows neighbouring spots, but its expression is only that of the central spot.

**How are Visium HD bins assigned to tiles?**
Only Visium HD has bins; standard Visium has spots, one per tile. For Visium HD the rule is fixed: a bin belongs to the tile that contains its centre.

Process 1 computes this once and writes it to `assignment.tsv.gz`. A step is needed because Space Ranger's coordinates are pixels of the full-resolution image, whereas tiles are defined on the resampled image. Every later step reads that file, so no step can count a bin differently.

With 8 µm bins (`--hd_bin_um 8`, the default) a 112 µm tile holds 14 × 14 = 196 bins; with 2 µm bins it would hold 3136. The tile's expression is the sum of its bins.

**Why are counts summed rather than averaged?**
A sum of counts is still a count: it is what one spot with the tile's footprint would have measured. Count models and library-size normalisation therefore still apply.

## Running

**Which parameters and choices are there?**
Take the decisions in this order:

1. **Where and with which software:** `-profile <machine>,<software>[,gpu]`.
   * Machine: `laptop`, `uzh_rolandgarros` or `slurm`.
   * Software: `docker`, `apptainer`, `singularity`, `conda` or `mamba`.
   * `test` runs on the synthetic data.
2. **Data** (per sample, in the samplesheet): `spatial_format` (`spaceranger_hd`, `spaceranger`, `hest` or empty) decides the tiling, `grid` or `spot`. The `tiling` column can override it.
3. **Scales:** `--scales_mpp`.
4. **Branches:** `--skip_uni2h`, `--skip_lbp`, `--skip_spe`, `--skip_anndata`, `--export_combined`.
5. **UNI2-h weights:** `--uni2h_weights` (recommended) or a download with `HF_TOKEN`.
6. **Fine-tuning**, rarely needed:
   * tissue mask and QC: `--min_tissue`, `--mask_*`, `--blur_mad`, `--drop_blurred`;
   * LBP: `--lbp_channels`, `--lbp_radii`, `--lbp_npoints`, `--lbp_max_radius`, `--he_h_max`, `--he_e_max`;
   * Visium HD: `--hd_bin_um`, `--min_st_coverage`;
   * resources: `--max_cpus`, `--max_memory`, `--max_time`.

Every parameter is listed with its default and a comment in [`nextflow.config`](../nextflow.config).

**Can it run without a GPU?**
Yes. UNI2-h then runs on the CPU in fp32. This works but is much slower; `--scales_mpp 0.5` keeps the run short. The other processes never use a GPU.

**How do I run only part of the pipeline?**
Use the `--skip_*` options. For example, `--skip_uni2h` runs only fastLBP and the expression aggregation. Add `-resume` to reuse finished tasks after changing a parameter that only affects later steps.
