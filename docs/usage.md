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

Useful options, in the three levels of `nextflow.config`.

**Core** (the standard run):

* `--scales_mpp 0.5` — main scale only (faster).
* `-resume` — continue after an interruption; finished tasks are reused.
* `--skip_uni2h` / `--skip_lbp` — run one branch only.
* `--min_tissue_extract 0.3` — compute features from 30 % tissue on (`keep` still uses `--min_tissue`), to compare thresholds later without rerunning.
* `--lbp_channels rgb,rgb_he,he,h,e` — add LBP on the stain amounts to the default input sets: H and E side by side (`he`), H alone (`h`) and E alone (`e`); the LBP of H and of E is computed once for the three. They change neither the input images nor UNI2-h, so adding them to a finished run with `-resume` reruns only fastLBP and the steps after it (clustering, exports, QC page); [rolandgarros.md](rolandgarros.md#11-lbp-on-the-stain-amounts-he-h-e-tileparity-041-or-later) shows how.
* `--uni2h_inputs rgb` — UNI2-h on the slide only, not on its H&E-only version (half the GPU time).
* `--stain_vectors standard` — the same published H and E colours for every slide instead of each slide's own.
* `--visium_tiling spot` — spot-centred tiles for every standard Visium / HEST sample; `--visium_split fraction` — split spots by expected value instead of multinomially.
* `--publish_images` — also keep the resampled and channels images (large).
* `-profile debug` — keep work directories and print host names.

**Optional analyses** (off by default):

* `--qc_umap` — write an interactive 3-D UMAP of every feature view per scale (`qc/`).
* `--cluster` — cluster every feature view of every slide and scale on its own and measure how much the views agree (`<sample>/mpp<scale>/cluster/`). Options: `--cluster_k` (numbers of clusters, default `2,3,4,6,8,10,12,15`), `--cluster_k_ref` (the k of the maps and heatmaps: `auto`, the default, picks per method the k at which the image views agree best with EXPRESSION, by their mean corrected ARI; or a number), `--cluster_sets` (`keep,extended`; add `--min_tissue_extract 0.3` so that `extended` includes tiles with 30–50 % tissue), `--cluster_views` (default `all` computed views), `--cluster_methods` (`leiden,kmeans`), `--cluster_min_counts` (counts floor of `extended`, 0.1 of the median), `--cluster_n_pcs` (50), `--cluster_n_neighbors` (15), `--cluster_n_hvg` (2000), `--cluster_null_shifts` (100), `--cluster_seed` (0), `--cluster_stability_reps` (5 subsamples of `--cluster_stability_fraction` 0.9 of the tiles; 0 switches the stability check off). Besides the agreement it writes the corrected ARI, F1 per expression domain and the averages over k (`cluster_summary.tsv`, `cluster_f1.tsv`, `cluster_stability.tsv`). With it, by default:
  * `--cluster_gallery` (true) — for every view, the `--cluster_gallery_n` (8) most typical tiles of every cluster and the tiles at both ends of the first principal components (`<sample>/mpp<scale>/gallery/`);
  * `--cohort_summary` (true) — one summary per scale with every slide (`cohort/mpp<scale>/`): a long table, key numbers per slide and view, cohort figures and `report.html`. Groups of slides are compared by the samplesheet column `--cohort_group` (default `condition`), with sections of the same `--cohort_pair` (default `patient`) averaged first; with two groups and patients seen in both, a paired Wilcoxon test.

  The method is explained in [design.md](design.md#optional-clustering-of-every-view-per-slide---cluster).
* `--redundancy` (with `--cluster`) — how much of each image view the others predict, from the same principal components as the clustering, in a spatial cross-validation (`cluster/<id>.redundancy*`). Options: `--redundancy_views` (`images`: every clustered view except EXPRESSION, or a list), `--redundancy_folds` (5), `--redundancy_block_um` (1000: side of the blocks of tissue that go to one fold). `--redundancy_null_shifts` (20 moves of the spatial null of R2) and `--redundancy_bootstrap` (200 block-bootstrap resamples for its 95 % interval). Method: [design.md](design.md#optional-redundancy-between-views---redundancy).
* `--relevance` (with `--cluster`) — how much of the expression each image view predicts, what each adds to another (UNI2-h and LBP on the same pixels, LBP and INTENSITY_HE), and which genes the image predicts (`cluster/<id>.relevance*`). Same folds, blocks, null and interval as `--redundancy`; `--relevance_gene_views` (`UNI2H_RGB,LBP_RGB,INTENSITY_HE`, or `none`), `--relevance_gene_null_shifts` (20). Method: [design.md](design.md#optional-relevance-what-the-image-predicts-of-the-expression---relevance).
* `--patch_sweep` (Visium HD slides) — LBP and expression on patches of `--patch_sweep_sizes_um` (`16:224:8`: 16 to 224 µm in steps of 8), built from LBP counted once per `--patch_sweep_cell_um` (8) µm cell; input set `--patch_sweep_channels` (`rgb`), radii up to `--patch_sweep_max_radius` (8 px), ARI at `--patch_sweep_k` (`4,8`) (`<sample>/mpp<scale>/patch_sweep/`). Method: [design.md](design.md#optional-patch-size-of-fastlbp---patch_sweep).

**Diagnostics** (off by default; why each exists: [decisions.md](decisions.md)):

* `--mask_compare` — compare the tissue mask with the other methods and finer resolutions (`prepare/<id>.mask_compare.png`).
* `--lbp_center` — add the central-pixel LBP blocks.
* `--cluster_lbp_recipes hellinger,scanpy,zscore` (with `--cluster`) — also prepare every LBP view like expression (`scanpy`: counts per 10 000, log1p, the `--cluster_lbp_hvf` share of most variable bins, 0.3, z-scores) and like UNI2-h (`zscore`), cluster them as views `<view>.scanpy` and `<view>.zscore`, and compare the three preparations (`cluster/<id>.cluster_lbp_recipes_keep.png`).

The first real run on the UZH server is described step by step in [rolandgarros.md](rolandgarros.md).

## 5. Running on another machine

Create `conf/<site>.config` modelled on `conf/uzh_rolandgarros.config` (executor, `resourceLimits`, GPU request), add a profile entry in `nextflow.config`, and combine it with a software profile: `-profile <site>,apptainer`. Nothing else changes. For any SLURM cluster, `-profile slurm` with `--slurm_queue`, `--slurm_options` and `--slurm_gpu_options` may already be enough.

## Resources (orientation)

Per sample at 0.5 µm/px (at most ≈3 400 tiles of 112 µm for a 6.5 × 6.5 mm Visium HD capture area), LBP takes roughly 2 CPU-seconds per tile and channel with the default 13 radii; the default input sets have 6 channels (R, G, B of `rgb` and of `rgb_he`), i.e. about 10 minutes on 64 cores or 1.5 hours on 8, and `he`, `h` and `e` add 2 (H and E, a third more time); the 0.2857 µm/px scale has about three times more tiles. UNI2-h needs about 0.35 TFLOP per tile and input set (ViT-H/14, 681 M parameters), i.e. in fp16 roughly a hundred tiles per second on an NVIDIA L4 and a few tens on a laptop RTX 3070 (estimates); the two default input sets double that time. The resampled image is written to the work directory (about 3 GB per sample at 0.5 µm/px, uncompressed equivalent). These are estimates; the first real run will give measured times in `pipeline_info/execution_report_*.html`.
