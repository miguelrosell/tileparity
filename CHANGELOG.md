# Changelog

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
