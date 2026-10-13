# tileparity

**Harmonised H&E tiling so that a foundation model (UNI2-h), a classical texture descriptor (fastLBP) and spatial transcriptomics all describe exactly the same pieces of tissue.**

Comparing image feature extractors is only fair if they see the same pixels. `tileparity` makes every geometric decision once — pixel size, tissue mask, tile grid, assignment of Visium / Visium HD units to tiles — and then feeds the same tiles to each extractor and to the expression aggregation. The result is one object per sample (SpatialExperiment and AnnData) in which every column/row is a tile with its UNI2-h embedding, its LBP histograms and its gene counts.

```
Input1 --[1 PREPARE]--> Output1 = Input2 (resampled image, tile table, unit->tile assignment)
                           |-- [1a+2a UNI2H_EMBED] -------------------> Output2a  tiles x 1536
                           |-- [1b LBPCHANNELS] -> [2b FASTLBP_EXTRACT] -> Output2b  tiles x 2290 (+ grey, intensity)
                           |-- [3 AGGREGATEST] ------------------------> Output3   genes x tiles
                           '-- [4 EXPORTSPE / EXPORTANNDATA] ----------> SpatialExperiment (.rds), AnnData (.h5ad)
```

| Process | Module | What it does |
|---|---|---|
| 1 | `TILEPARITY_PREPARE` | Resamples the slide to each target µm/px, detects tissue, lays the tile grid (or centres tiles on Visium spots), computes QC, assigns spatial units to tiles. **The only step that makes geometric decisions.** |
| 1a + 2a | `UNI2H_EMBED` | Cuts each 224 × 224 px tile (no resize), normalises it as in the UNI2-h model card and returns the 1536-d class token. |
| 1b | `TILEPARITY_LBPCHANNELS` | Computes haematoxylin, eosin (colour deconvolution) and grey channels pixel by pixel from the same RGB image. |
| 2b | `FASTLBP_EXTRACT` | Uniform LBP histograms per tile and channel with fastLBP's kernel (13 radii, 1–45 px), using the real neighbouring pixels; plus H/E intensity statistics. |
| 3 | `TILEPARITY_AGGREGATEST` | Expression per tile, Y = X·A: the sum of the Visium HD bins inside each tile (or the Visium spot of a spot-centred tile). |
| 4 | `TILEPARITY_EXPORTSPE`, `TILEPARITY_EXPORTANNDATA` | One object per sample and one combined object per scale. |

The reasoning behind every default is in [docs/design.md](docs/design.md).

## Quick start

Requirements: [Nextflow](https://www.nextflow.io/docs/latest/install.html) ≥ 24.10 (Java 17+) and one of Docker, Apptainer/Singularity or Conda. A GPU is optional.

```bash
# 1. smoke test on a tiny synthetic slide (a few minutes, no data or token needed)
nextflow run miguelrosell/tileparity -profile test,docker

# 2. real data on a workstation with an NVIDIA GPU
export HF_TOKEN=hf_xxx                                   # access to MahmoodLab/UNI2-h
bash scripts/download_uni2h_weights.sh ~/models/uni2h    # once
nextflow run miguelrosell/tileparity -profile laptop,docker,gpu \
    --input samplesheet.csv --outdir results \
    --uni2h_weights ~/models/uni2h/pytorch_model.bin
```

The container images are built from [`containers/`](containers) by GitHub Actions and published as `ghcr.io/miguelrosell/tileparity-{python,uni2h,r}`. To build them locally instead: `bash containers/build_local.sh`.

## Running on the UZH server (SLURM)

```bash
# from the login node; pick the software route that is available
nextflow run miguelrosell/tileparity -profile uzh_rolandgarros,apptainer --input ... --outdir ...
nextflow run miguelrosell/tileparity -profile uzh_rolandgarros,conda     --input ... --outdir ...
```

`conf/uzh_rolandgarros.config` holds everything specific to that machine (executor, resource ceilings, GPU request). The partition names and the GPU `--gres` string are marked `TODO` until confirmed with the administrator. Any other machine only needs a similar small config file.

## Samplesheet

| column | required | description |
|---|---|---|
| `sample` | yes | unique id (letters, digits, `.`, `_`, `-`) |
| `image` | yes | full-resolution H&E image (TIFF/BigTIFF/BTF/SVS/…), **the same image Space Ranger was given** |
| `spatial` | no | Space Ranger `outs/` (or `binned_outputs/`) folder, or a HEST-1k `.h5ad` |
| `spatial_format` | with `spatial` | `spaceranger_hd`, `spaceranger` or `hest` |
| `tiling` | no | `auto` (default: `grid` for Visium HD and image-only samples, `spot` for Visium), `grid`, `spot` |
| `mpp` | no | native pixel size (µm/px); otherwise read from Space Ranger's scale factors or the image metadata |
| any other column | no | sample metadata (e.g. `patient`, `condition`, `cohort`), copied to every tile |

Relative paths are resolved against the samplesheet's folder. See [`assets/samplesheet_example.csv`](assets/samplesheet_example.csv) and [docs/usage.md](docs/usage.md).

## Main parameters

| parameter | default | meaning |
|---|---|---|
| `--scales_mpp` | `0.2857,0.5,1.0` | target pixel sizes; tiles are 224 px, i.e. 64, 112 and 224 µm. Scales finer than a slide's native resolution are skipped (no upsampling). |
| `--min_tissue` | `0.5` | minimum tissue fraction of a kept tile |
| `--hd_bin_um` | `8` | Visium HD bin size summed per tile |
| `--lbp_channels` | `H,E,grey` | channels for LBP |
| `--uni2h_weights` | — | local UNI2-h weights (recommended); otherwise downloaded with `HF_TOKEN` |
| `--skip_uni2h`, `--skip_lbp`, `--skip_spe`, `--skip_anndata` | `false` | switch branches off |

All parameters are listed with comments in [`nextflow.config`](nextflow.config).

## Outputs

```
results/
├── <sample>/mpp<scale>/prepare/   tile table, unit→tile assignment, thumbnail, mask, overlay, prepare.json
├── <sample>/mpp<scale>/uni2h/     <id>.uni2h.h5      (Output2a)
├── <sample>/mpp<scale>/lbp/       <id>.lbp.h5        (Output2b), channels.json
├── <sample>/mpp<scale>/st/        <id>.counts.h5     (Output3), per-tile summary
├── export/mpp<scale>/spe/         <id>.rds + combined_mpp<scale>.rds
├── export/mpp<scale>/anndata/     <id>.h5ad + combined_mpp<scale>.h5ad
└── pipeline_info/                 software versions, execution report, timeline, trace
```

Details in [docs/output.md](docs/output.md). Check `prepare/<id>.overlay.png` first: the blue tint marks where Space Ranger placed the spatial units, so a shift between tint and tissue reveals an image that does not match the expression data.

## Tests

```bash
python -m pytest tests/python          # unit tests (incl. LBP identical to fastLBP's own whole-image run)
nextflow run . -profile test -stub-run # workflow logic without software
nextflow run . -profile test,docker    # end to end on the synthetic slide
```

`tests/data/make_test_data.py` regenerates the synthetic slide and its Visium HD, Visium and HEST-style inputs.

## Extending

A new extractor is one module that reads `[meta, resampled.tif, tiles.tsv.gz]` and writes `<id>.<name>.h5` in the layout of `bin/tileparity_lib/features.py`; add it to `subworkflows/local/tileparity/main.nf` next to `UNI2H_EMBED` and to the export join. Nothing in Process 1 changes.

## Status and roadmap

Version 0.1.0. Tested here: the Python library (33 unit tests, including LBP histograms identical to fastLBP's whole-image run) and the whole workflow on the synthetic test data without containers. The container images, the SpatialExperiment export and the GPU path are exercised by the GitHub Actions workflows (`Containers` builds the images and runs the test profile in Docker).

Not yet implemented:

* stain normalisation (e.g. Macenko) as an option;
* an input mode for pre-tiled image collections such as NCT-CRC-HE-100K;
* cell-level units (segmented nuclei) as a third tiling mode.

## Citations

See [CITATIONS.md](CITATIONS.md). UNI2-h is released by the Mahmood Lab under CC-BY-NC-ND 4.0 for non-commercial academic use; the weights are not redistributed here.
