# tileparity

**Harmonised H&E tiling so that a foundation model (UNI2-h), a classical texture descriptor (fastLBP) and spatial transcriptomics all describe exactly the same pieces of tissue.**

Comparing image feature extractors is only fair if they see the same pixels. `tileparity` makes every geometric decision once — pixel size, tissue mask, tile grid, assignment of Visium / Visium HD units to tiles — computes the input images once, and then feeds the same tiles of the same images to each extractor and to the expression aggregation. The result is one object per sample (SpatialExperiment and AnnData) in which every column/row is a tile with its UNI2-h embeddings, its LBP histograms and its gene counts, and with the record of every step that produced it.

```
Input1 --[1 PREPARE]--> Output1 = Input2 (resampled image, tile table, unit->tile assignment)
                           |-- [1b CHANNELS] --------------------------> the input sets: rgb (the slide) and
                           |                                             rgb_he (the slide rebuilt from H and E only)
                           |-- [1a+2a UNI2H_EMBED] (rgb, rgb_he) ------> Output2a  tiles x 1536 per input set
                           |-- [2b FASTLBP_EXTRACT] (rgb, rgb_he) -----> Output2b  tiles x 1145 per channel (+ intensity)
                           |-- [3 AGGREGATEST] ------------------------> Output3   genes x tiles
                           |-- [CLUSTER] (optional, per slide) --------> clusters of every view, agreement between views
                           |-- [4 EXPORTSPE / EXPORTANNDATA] ----------> SpatialExperiment (.rds), AnnData (.h5ad)
                           '-- [QCUMAP] (optional) --------------------> interactive 3-D UMAP of every view
```

| Process | Module | What it does |
|---|---|---|
| 1 | `TILEPARITY_PREPARE` | Resamples the slide to each target µm/px, detects tissue (saturation, grey or optical density + Otsu, or your own mask), lays the tile grid (or centres tiles on Visium spots), computes QC, decides which tiles are extracted and kept, assigns spatial units to tiles (Visium spots by area share). **The only step that makes geometric decisions.** |
| 1b | `TILEPARITY_CHANNELS` | Computes the input images of both extractors once, pixel by pixel from the resampled image: the slide rebuilt from haematoxylin and eosin only (`rgb_he`), with H and E colours estimated on each slide (Macenko), the H and E amounts and the R, G, B channels; a QC figure of what `rgb_he` changes. |
| 1a + 2a | `UNI2H_EMBED` | For each input set, cuts each 224 × 224 px tile (no resize), normalises it as in the UNI2-h model card and returns the 1536-d class token. |
| 2b | `FASTLBP_EXTRACT` | Uniform LBP histograms per tile and channel of each input set with fastLBP's kernel (13 radii, 1–45 px), using the real neighbouring pixels; H/E intensity statistics; a QC figure of irregular patterns per radius. |
| 3 | `TILEPARITY_AGGREGATEST` | Expression per tile: the sum of the Visium HD bins inside each tile, or the Visium spots split between the tiles they touch (multinomially, by area share). |
| 4 | `TILEPARITY_EXPORTSPE`, `TILEPARITY_EXPORTANNDATA` | One object per sample and one combined object per scale, with their lineage. |
| QC | `TILEPARITY_QCUMAP` | With `--qc_umap`: one interactive page per scale with a 3-D UMAP of UNI2-h and LBP on each input set, and of expression. |
| Analysis | `TILEPARITY_CLUSTER` | With `--cluster`: per slide and scale, every view clustered on its own (Leiden with the resolution tuned to give exactly k clusters, and k-means, for k = 2 to 15), and how much the views agree (adjusted Rand index, NMI and a spatial null), with figures; the exports get the Leiden labels. |

### The same images for both extractors

| input set | the image | UNI2-h | fastLBP |
|---|---|---|---|
| `rgb` (default) | the slide as it is | `UNI2H_RGB` | `LBP_RGB`: one LBP per channel R, G, B |
| `rgb_he` (default) | the slide rebuilt from haematoxylin and eosin only: every pixel's colour replaced by the closest colour made of the two stains, so the colour that is neither H nor E (pen, pigment, a tint of the glass or the scanner) is removed | `UNI2H_RGB_HE` | `LBP_RGB_HE`: one LBP per channel R_he, G_he, B_he |
| `he` (optional) | the amounts of H and of E | — (UNI2-h needs three channels) | `LBP_HE` |

Comparing `rgb` with `rgb_he` shows how much each extractor relies on colour that is neither H nor E; comparing the extractors on the same set compares them on identical pixels. [docs/design.md](docs/design.md#the-input-images-of-both-extractors-1b) explains how `rgb_he` is computed.

Documentation:

* [docs/usage.md](docs/usage.md) — installation, data download, samplesheet, running on a laptop or a cluster;
* [docs/rolandgarros.md](docs/rolandgarros.md) — the first real run on the UZH server, step by step;
* [docs/output.md](docs/output.md) — every output file and object;
* [docs/structure.md](docs/structure.md) — map of processes, files, tiles used, lineage and figures;
* [docs/design.md](docs/design.md) — the reasoning behind every default;
* [docs/lbp.md](docs/lbp.md) — LBP step by step, from one pixel to 1145 numbers;
* [docs/faq.md](docs/faq.md) — short answers: input requirements, what each extractor receives, spots and bins, parameters.

## Quick start

Requirements: [Nextflow](https://www.nextflow.io/docs/latest/install.html) ≥ 25.04 (Java 17+) and one of Docker, Apptainer/Singularity or Conda. A GPU is optional.

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

The container images are built from [`containers/`](containers) by GitHub Actions and published as `ghcr.io/miguelrosell/tileparity-{python,uni2h,r}`. To build them locally instead: `bash containers/build_local.sh`. Without containers, `-profile conda` creates the same software environments from `containers/*/environment.yml`.

## Running on the UZH server (SLURM)

Without containers (conda in your own folder; the whole run is one SLURM job, inside which Nextflow runs every task on the node):

```bash
# inside an sbatch job with 64 CPUs, 400 GB and the GPU (docs/rolandgarros.md has the job script)
nextflow run ./tileparity -profile conda,gpu --max_cpus 64 --max_memory '400.GB' --input ... --outdir ...
```

With Apptainer, from the login node, every task as its own SLURM job:

```bash
nextflow run miguelrosell/tileparity -r main -profile uzh_rolandgarros,apptainer --input ... --outdir ... \
    --slurm_queue <partition> --slurm_gpu_options '--gres=gpu:1'
```

`conf/uzh_rolandgarros.config` holds what is specific to that machine (executor, resource ceilings, GPU request). Partitions and sbatch options are parameters (`--slurm_queue`, `--slurm_gpu_queue`, `--slurm_options`, `--slurm_gpu_options`), so nothing needs editing; `sinfo -o "%P %G"` lists the partitions and their GPUs. [docs/rolandgarros.md](docs/rolandgarros.md) walks through the first run step by step (conda route).

## Samplesheet

| column | required | description |
|---|---|---|
| `sample` | yes | unique id (letters, digits, `.`, `_`, `-`) |
| `image` | yes | full-resolution H&E image (TIFF/BigTIFF/BTF/SVS/…), **the same image Space Ranger was given** |
| `spatial` | no | Space Ranger `outs/` (or `binned_outputs/`) folder, or a HEST-1k `.h5ad` |
| `spatial_format` | with `spatial` | `spaceranger_hd`, `spaceranger` or `hest` |
| `tiling` | no | `auto` (default: grid tiles; for standard Visium `--visium_tiling` decides, `grid` by default), `grid`, `spot` |
| `mpp` | no | native pixel size (µm/px); otherwise read from Space Ranger's scale factors or the image metadata |
| `mask` | no | your own tissue mask of the whole slide (replaces the automatic one for that sample) |
| any other column | no | sample metadata (e.g. `patient`, `condition`, `cohort`), copied to every tile |

Relative paths are resolved against the samplesheet's folder. See [`assets/samplesheet_example.csv`](assets/samplesheet_example.csv) and [docs/usage.md](docs/usage.md).

## Main parameters

| parameter | default | meaning |
|---|---|---|
| `--scales_mpp` | `0.2857,0.5,1.0` | target pixel sizes; tiles are 224 px, i.e. 64, 112 and 224 µm. Scales finer than a slide's native resolution are skipped (no upsampling). |
| `--min_tissue` | `0.5` | minimum tissue fraction of a kept tile (`keep`) |
| `--min_tissue_extract` | = `--min_tissue` | minimum tissue fraction for computing features (`extract`) |
| `--mask_method` | `saturation` | automatic tissue mask: `saturation`, `grey` or `od` |
| `--mask_compare` | `false` | also compare the mask with the other methods and finer resolutions |
| `--hd_bin_um` | `8` | Visium HD bin size summed per tile |
| `--visium_tiling` | `grid` | standard Visium / HEST: `grid` (spots split by area) or `spot` (tiles centred on spots) |
| `--visium_split` | `multinomial` | how a spot's molecules are split between grid tiles: `multinomial` or `fraction` |
| `--uni2h_inputs` | `rgb,rgb_he` | UNI2-h input sets: `rgb`, `rgb_he` |
| `--lbp_channels` | `rgb,rgb_he` | fastLBP input sets: `rgb`, `rgb_he`, `he` |
| `--stain_vectors` | `slide` | H and E colours: estimated on each slide (`slide`) or the published ones (`standard`) |
| `--lbp_center` | `false` | also store the LBP codes of the tiles' central pixels |
| `--qc_umap` | `false` | interactive 3-D UMAP of every feature view per scale |
| `--cluster` | `false` | cluster every view of every slide on its own and measure how much the views agree (`--cluster_k`, `--cluster_k_ref`, `--cluster_sets`, … in [docs/usage.md](docs/usage.md)) |
| `--uni2h_weights` | — | local UNI2-h weights (recommended); otherwise downloaded with `HF_TOKEN` |
| `--skip_uni2h`, `--skip_lbp`, `--skip_spe`, `--skip_anndata` | `false` | switch branches off |

All parameters are listed with comments in [`nextflow.config`](nextflow.config).

## Outputs

```
results/
├── <sample>/mpp<scale>/prepare/   tile table, unit→tile assignment, thumbnail, mask, overlay, tissue thresholds, [mask comparison], prepare.json
├── <sample>/mpp<scale>/channels/  channels.json (bands, stain vectors), rgb_he_qc.png/.tsv
├── <sample>/mpp<scale>/uni2h/     <id>.uni2h.h5      (Output2a)
├── <sample>/mpp<scale>/lbp/       <id>.lbp.h5        (Output2b), lbp_qc.png
├── <sample>/mpp<scale>/st/        <id>.counts.h5     (Output3), per-tile summary
├── <sample>/mpp<scale>/cluster/   [clusters of every view, agreement between views, figures]
├── export/mpp<scale>/spe/         <id>.rds + combined_mpp<scale>.rds
├── export/mpp<scale>/anndata/     <id>.h5ad + combined_mpp<scale>.h5ad
├── qc/mpp<scale>/                 [3-D UMAP page]
└── pipeline_info/                 software versions, execution report, timeline, trace
```

Every step also writes a `*.lineage.json` (command, parameters, software, checksums, and every step upstream), and every figure has a caption saying what it shows, how to read it, its caveats and where it comes from. Details in [docs/output.md](docs/output.md). Check `prepare/<id>.overlay.png` first: the blue tint marks where Space Ranger placed the spatial units, so a shift between tint and tissue reveals an image that does not match the expression data.

## Tests

```bash
python -m pytest tests/python          # unit tests (incl. LBP identical to fastLBP's own whole-image run)
nextflow lint .                        # syntax check (also of the strict syntax of recent Nextflow)
nextflow run . -profile test -stub-run # workflow logic without software
nextflow run . -profile test,docker    # end to end on the synthetic slide
```

`tests/data/make_test_data.py` regenerates the synthetic slide and its Visium HD, Visium and HEST-style inputs and an external mask.

## Extending

A new extractor is one module that reads `[meta, resampled.tif, tiles.tsv.gz, prepare.lineage.json]` (and, for the input set `rgb_he`, the channels image and `channels.json` of Process 1b) and writes `<id>.<name>.h5` in the layout of `bin/tileparity_lib/features.py` and a lineage file (`bin/tileparity_lib/lineage.py`); add it to `subworkflows/local/tileparity/main.nf` next to `UNI2H_EMBED` and to the export join. Nothing in Process 1 changes.

## Status and roadmap

Version 0.4.0 (changes in [CHANGELOG.md](CHANGELOG.md)). What has been run so far:

* the Python library: 98 unit tests, including LBP histograms identical to fastLBP's whole-image run, the H&E-only image against a direct computation, and the clustering's expression recipe, neighbour graph and Leiden partition identical to Scanpy's;
* two real samples (the Visium HD colorectal cancer slide P1 and the HEST-1k Visium sample ZEN36) at 0.5 µm/px on the UZH server with tileparity 0.3.0: conda route, one SLURM job, UNI2-h in fp16 on an NVIDIA L4, about 35 minutes;
* the whole workflow on the synthetic test data without containers, with every option of the test profile switched on and both exports (AnnData; SpatialExperiment with R 4.3 / SpatialExperiment 1.12), with Nextflow 25.04.0 and 26.10.3; stub runs with 25.04.0, 25.04.7, 26.04.7 and 26.10.3, and `nextflow lint` without errors with 25.04.7, 26.04.7 and 26.10.3 (the linter of 25.04.0 wrongly reports `meta` as undefined in `conf/modules.config`);
* GitHub Actions: `CI` runs the unit tests, a stub run with the oldest supported and the latest Nextflow, and lint with the latest, on every push; `Containers` builds the three images and runs the test profile inside them in Docker.

The clustering (`--cluster`) has been run on the synthetic test data only so far; its first real run will be the two samples above.

Not yet implemented:

* stain normalisation to reference colours as an option (the H and E colours are already estimated per slide in Process 1b);
* an input mode for pre-tiled image collections such as NCT-CRC-HE-100K;
* cell-level units (segmented nuclei) as a third tiling mode;
* the analysis layer beyond the per-slide clustering (comparing and integrating the views across slides).

## Citations

See [CITATIONS.md](CITATIONS.md). UNI2-h is released by the Mahmood Lab under CC-BY-NC-ND 4.0 for non-commercial academic use; the weights are not redistributed here.
