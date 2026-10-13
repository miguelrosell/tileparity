# Usage

## 1. Install

* Nextflow ≥ 24.10 with Java 17+ (`curl -s https://get.nextflow.io | bash`).
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
sample,image,spatial,spatial_format,tiling,mpp,patient,condition,cohort
A_P1_CRC,raw/A_P1_CRC/tissue_image.btf,raw/A_P1_CRC/binned_outputs,spaceranger_hd,grid,,P1,CRC,A
ZEN36,hest/wsis/ZEN36.tif,hest/st/ZEN36.h5ad,hest,spot,0.4598,Patient 7,Cancer,B
```

* For `spaceranger_hd`, `spatial` may be `outs/`, `outs/binned_outputs/` or a `square_008um/` folder; the bin size is chosen with `--hd_bin_um`.
* For `spaceranger` (standard Visium), `spatial` is the `outs/` folder.
* Image-only samples: leave `spatial` and `spatial_format` empty (grid tiling, no counts).
* Extra columns are copied into every tile of that sample.

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
* `--publish_images` — also keep the resampled and H/E/grey images (large).
* `-profile debug` — keep work directories and print host names.

## 5. Running on another machine

Create `conf/<site>.config` modelled on `conf/uzh_rolandgarros.config` (executor, `resourceLimits`, GPU request), add a profile entry in `nextflow.config`, and combine it with a software profile: `-profile <site>,apptainer`. Nothing else changes.

## Resources (orientation)

Per sample at 0.5 µm/px (≈3 000–4 000 tissue tiles for a 6.5 mm Visium HD capture area), LBP takes roughly 2 CPU-seconds per tile and channel with the default 13 radii, i.e. a few minutes on 64 cores or about an hour on 8; the 0.2857 µm/px scale has about three times more tiles. UNI2-h needs about 0.35 TFLOP per tile (ViT-H/14, 681 M parameters), i.e. in fp16 roughly a hundred tiles per second on an NVIDIA L4 and a few tens on a laptop RTX 3070 (estimates). The resampled image is written to the work directory (about 3 GB per sample at 0.5 µm/px, uncompressed equivalent).
