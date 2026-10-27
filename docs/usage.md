# Usage

## 1. Install

* Nextflow ≥ 25.04 with Java 17+ (`curl -s https://get.nextflow.io | bash`). The pipeline is tested with 25.04.0 and with the latest release (26.10.3 at the time of writing), which reads scripts and configs with the strict syntax.
* One software route:
  * **Docker** (workstation). For the GPU also install the NVIDIA Container Toolkit, then add `-profile gpu`.
  * **Apptainer / Singularity** (clusters). Images are pulled from `ghcr.io/miguelrosell/tileparity-*`.
  * **Conda / Mamba** (no containers). Environments are created from `containers/*/environment.yml`.

## 2. Get the data

* **Visium HD (cohort A)**: from each 10x Genomics dataset page download the *microscope image* (BTF/TIFF) and the *binned outputs*. Put the links in `assets/cohort_a_10x_urls.tsv` and run `bash scripts/download_urls.sh assets/cohort_a_10x_urls.tsv /data/raw`. Use the microscope image, not the CytAssist image: Space Ranger's coordinates refer to it.
* **HEST-1k (cohort B)**: `python scripts/hest_download.py --ids ZEN36 ZEN37 ... ZEN49 --outdir /data/hest --samplesheet cohort_b.csv --cohort B` (needs `HF_TOKEN` with access to `MahmoodLab/hest`). Do not use HEST's Visium HD samples for aggregation: their h5ad pools bins into 128 µm pseudo-spots.
* **UNI2-h weights**: `HF_TOKEN=... bash scripts/download_uni2h_weights.sh ~/models/uni2h` (add `USE_DOCKER=1` to run it in the container).

## 3. Write the samplesheet

```csv
sample,image,spatial,spatial_format,tiling,mpp,mask,patient,condition,cohort
A_P1_CRC,raw/A_P1_CRC/tissue_image.btf,raw/A_P1_CRC/binned_outputs,spaceranger_hd,,,,P1,CRC,A
ZEN36,hest/wsis/ZEN36.tif,hest/st/ZEN36.h5ad,hest,,0.4598,,Patient 7,Cancer,B
```

* For `spaceranger_hd`, `spatial` may be `outs/`, `outs/binned_outputs/` or a `square_008um/` folder; the bin size is chosen with `--hd_bin_um`.
* For `spaceranger` (standard Visium), `spatial` is the `outs/` folder.
* Image-only samples: leave `spatial` and `spatial_format` empty (grid tiling, no counts).
* `tiling` can be left empty (`auto`): grid tiles for every technology (for standard Visium and HEST, `--visium_tiling` decides: `grid` by default, each spot split between the tiles it touches). Write `spot` in a row to get one tile centred on each spot for that sample (standard Visium and HEST only).
* `mpp` can be left empty when Space Ranger's files or the image metadata give the pixel size; if it is given and disagrees with them, the run stops (see [design.md](design.md#pixel-size)).
* `mask` (optional): a tissue mask of the whole slide (any image format libvips reads, any resolution, the same aspect ratio as the slide; non-zero or ≥ 128 = tissue), e.g. exported from QuPath. It replaces the automatic mask for that sample.
* Extra columns (e.g. `patient`, `condition`) are copied into every tile of that sample; they must not reuse a tile-table column name such as `row`, `col` or `keep`.

Common questions about inputs, spots and bins are answered in [faq.md](faq.md).

## 4. Run

```bash
nextflow run miguelrosell/tileparity -r main -profile laptop,docker,gpu \
    --input samplesheet.csv --outdir results \
    --uni2h_weights ~/models/uni2h/pytorch_model.bin
```

Useful options:

* `--scales_mpp 0.5` — main scale only (faster).
* `-resume` — continue after an interruption; finished tasks are reused.
* `--skip_uni2h` / `--skip_lbp` — run one branch only.
* `--mask_compare` — compare the tissue mask with the other methods and finer resolutions (`prepare/<id>.mask_compare.png`).
* `--min_tissue_extract 0.3` — compute features from 30 % tissue on (`keep` still uses `--min_tissue`), to compare thresholds later without rerunning.
* `--lbp_channels rgb,rgb_he,he` — add LBP on the H and E amounts to the default input sets; `--lbp_center` — add the central-pixel LBP blocks.
* `--uni2h_inputs rgb` — UNI2-h on the slide only, not on its H&E-only version (half the GPU time).
* `--stain_vectors standard` — the same published H and E colours for every slide instead of each slide's own.
* `--qc_umap` — write an interactive 3-D UMAP of every feature view per scale (`qc/`).
* `--visium_tiling spot` — spot-centred tiles for every standard Visium / HEST sample; `--visium_split fraction` — split spots by expected value instead of multinomially.
* `--publish_images` — also keep the resampled and channels images (large).
* `-profile debug` — keep work directories and print host names.

The first real run on the UZH server is described step by step in [rolandgarros.md](rolandgarros.md).

## 5. Running on another machine

Create `conf/<site>.config` modelled on `conf/uzh_rolandgarros.config` (executor, `resourceLimits`, GPU request), add a profile entry in `nextflow.config`, and combine it with a software profile: `-profile <site>,apptainer`. Nothing else changes. For any SLURM cluster, `-profile slurm` with `--slurm_queue`, `--slurm_options` and `--slurm_gpu_options` may already be enough.

## Resources (orientation)

Per sample at 0.5 µm/px (at most ≈3 400 tiles of 112 µm for a 6.5 × 6.5 mm Visium HD capture area), LBP takes roughly 2 CPU-seconds per tile and channel with the default 13 radii; the default input sets have 6 channels (R, G, B of `rgb` and of `rgb_he`), i.e. about 10 minutes on 64 cores or 1.5 hours on 8; the 0.2857 µm/px scale has about three times more tiles. UNI2-h needs about 0.35 TFLOP per tile and input set (ViT-H/14, 681 M parameters), i.e. in fp16 roughly a hundred tiles per second on an NVIDIA L4 and a few tens on a laptop RTX 3070 (estimates); the two default input sets double that time. The resampled image is written to the work directory (about 3 GB per sample at 0.5 µm/px, uncompressed equivalent). These are estimates; the first real run will give measured times in `pipeline_info/execution_report_*.html`.
