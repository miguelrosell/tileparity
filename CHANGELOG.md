# Changelog

## 0.2.0 (2026-10-24)

Changes discussed and agreed before the first run on real data.

**Tissue mask and tile selection (Process 1)**

* Three automatic mask methods, `--mask_method saturation` (default, as before), `grey` or `od` (mean optical density), and an optional external mask per sample (samplesheet column `mask`, any image of the whole slide).
* `--mask_compare`: figure and table comparing the mask used with the other methods at 4 µm/px and with the saturation mask at finer resolutions (`--mask_compare_mpp`, default 1.0 and 0.5 µm/px; inside the tissue's bounding box for very large slides), with Dice overlap, tiles gained and lost and agreement with Space Ranger's tissue call.
* Two tissue thresholds: features are computed for tiles with at least `--min_tissue_extract` tissue (new column `extract`; default: the same as `--min_tissue`), and `keep` still uses `--min_tissue`. Every later step reads the `extract` tiles. The blur reference uses the tiles with at least `--min_tissue`.
* `tissue_thresholds.png`: which tiles each of `--qc_tissue_thresholds` (30, 50, 80 %) would keep.

**Standard Visium on the grid**

* Standard Visium and HEST samples now use grid tiles by default (`--visium_tiling grid`), like Visium HD: no overlap, and a tile's image matches the area its expression comes from. Spot-centred tiles remain available (`--visium_tiling spot`, or `spot` in a samplesheet row). `scripts/hest_download.py` no longer writes `spot` into the samplesheet.
* Each spot (a 55 µm disc, `--visium_spot_um`) is split between the tiles it touches in proportion to its area in each (`assignment.tsv.gz` has a `weight` column). Process 3 gives each molecule to one tile at random with these probabilities (`--visium_split multinomial`, seed `--split_seed`, recorded), or uses the expected values (`--visium_split fraction`, non-integer counts). New column `st_units_equiv` (spots' worth of area in the tile); tiles need at least `--min_spot_overlap` (0.25) of it.

**LBP (Processes 1b, 2b)**

* Channel sets instead of single channels: `--lbp_channels he,grey` (default), plus `rgb` (R, G, B). Sets are alternatives and stored as separate blocks (`LBP_HE`, `LBP_GREY`, `LBP_RGB`). **Changed value format:** 0.1 accepted `H,E,grey`; 0.2 takes set names.
* `--lbp_center`: optional `LBP_CENTER_*` blocks with the codes of each tile's central 2 × 2 pixels.
* `lbp_qc.png` / `.tsv`: the share of irregular patterns per channel and radius.

**Traceability**

* Every step writes `<id>.<step>.lineage.json`: command, parameters, software versions, input and output files with checksums, and the records of every upstream step. The exported objects (per sample and combined) carry their lineage (`metadata(spe)$tileparity$lineage`, `uns["tileparity"]["lineage"]`).
* Every figure has a caption inside the PNG: what it shows, how to read it, caveats, where it comes from.

**Optional QC**

* `--qc_umap`: one interactive HTML page per scale with a 3-D UMAP of UNI2-h, each LBP channel set (proportions, Hellinger, equal weight per radius and channel) and expression.

**Compatibility, running and documentation**

* Nextflow ≥ 25.04 (0.1 declared ≥ 24.10, but Nextflow 24.10 cannot compile the samplesheet subworkflow); tested with the strict syntax of the current releases (26.04, 26.10): no variable declarations in configs, completion handler inside the workflow, no deprecated `when:` sections or implicit closure parameters. CI stub-runs with 25.04.0 and the latest stable release, and lints with the latest (the linter of 25.04.0 wrongly reports `meta` as undefined in the publishDir closures).
* SLURM partitions and options are parameters (`--slurm_queue`, `--slurm_gpu_queue`, `--slurm_options`, `--slurm_gpu_options`) for `-profile uzh_rolandgarros` and `-profile slurm`.
* New docs: [lbp.md](docs/lbp.md) (LBP step by step), [structure.md](docs/structure.md) (processes, files, lineage), [rolandgarros.md](docs/rolandgarros.md) (first real run). `design.md` corrects the effect of cutting tiles without context: at R = 45 the circles of 64 % of a tile's pixels leave the tile, but only 3–5 % of the codes change (1–5 % over all radii; `scripts/border_effect.py`).
* Container images 0.2.0; the Python image adds scikit-learn, numba, umap-learn and plotly for the QC page.

## 0.1.0 (2026-10-13)

First version.

* Process 1 (`TILEPARITY_PREPARE`): multi-scale resampling (libvips), saturation/Otsu tissue mask, grid or spot-centred 224 px tiles, QC metrics, unit→tile assignment, overlay figure.
* Processes 1a/2a (`UNI2H_EMBED`): explicit UNI2-h adapter (no resize/crop, ImageNet normalisation), class-token embeddings, local weights or Hugging Face download, test mode.
* Processes 1b/2b (`TILEPARITY_LBPCHANNELS`, `FASTLBP_EXTRACT`): H/E/grey channels; per-tile uniform LBP with fastLBP's kernel and real neighbouring context (identical to fastLBP's whole-image run); H/E intensity block.
* Process 3 (`TILEPARITY_AGGREGATEST`): Y = X·A aggregation for Visium HD bins, Visium spots and HEST h5ad.
* Process 4: SpatialExperiment (R) and AnnData exports per sample and combined per scale.
* Profiles: docker, apptainer, singularity, conda, mamba, wave, gpu, slurm, test, laptop, uzh_rolandgarros.
* QC flags (blur, pen, dark) only flag tiles by default; `--drop_blurred` also removes blur-flagged tiles.
* Input checks: pixel size cross-checked between the samplesheet, Space Ranger's scale factors, the spacing of the spots/bins and the image metadata; in-tissue units must lie inside the image; samplesheet and parameter validation before any task starts.
