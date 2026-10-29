# Outputs

`<id>` is `<sample>_mpp<scale>`, e.g. `A_P1_CRC_mpp0.5000`. Every step also writes `<id>.<step>.lineage.json` (command, parameters, software, checksums of inputs and outputs, and the records of every step upstream; see [structure.md](structure.md#lineage-where-every-output-comes-from)). Every PNG figure has a caption underneath that says what it shows, how to read it, its caveats and where it comes from.

## `<sample>/mpp<scale>/prepare/` — Output1

| file | content |
|---|---|
| `<id>.tiles.tsv.gz` | one row per candidate tile (any tissue): `tile_id`, `unit_type` (grid/spot), `row`, `col`, `barcode` (spot tiles), `x0_px`, `y0_px` (top-left corner in the resampled image), centre in resampled px (`cx_px`, `cy_px`), full-resolution px (`cx_fullres`, `cy_fullres`) and µm (`cx_um`, `cy_um`), `tissue_fraction` (share of the tile under the tissue mask), `st_coverage` (share of the tile's area covered by bins or spots), `n_st_units` (in-tissue units with a share in the tile), `st_units_equiv` (Visium grid: spots' worth of area in the tile, the sum of their shares; otherwise = `n_st_units`), QC metrics (`lapvar`, `pen_fraction`, `dark_fraction`) and flags, `extract` (features and counts computed), `keep` (default analysis set), `qc_fail` (reasons a tile is not kept), sample metadata |
| `<id>.assignment.tsv.gz` | `barcode`, `tile_id`, `weight`: every in-tissue Visium HD bin / Visium spot and the share of it in each candidate tile (1 for bins and spot-centred tiles; the area share for Visium spots on the grid) |
| `<id>.thumbnail.png`, `<id>.mask.png` | thumbnail at `--mask_mpp` and the tissue mask used |
| `<id>.overlay.png` | thumbnail with spatial units (blue tint) and tiles (green kept, orange dropped although they have enough tissue) |
| `<id>.tissue_thresholds.png` | the tiles each tissue threshold of `--qc_tissue_thresholds` would keep, with counts |
| `<id>.mask_compare.png`, `.tsv` | with `--mask_compare`: the mask used next to the other methods and finer resolutions; per mask: Otsu threshold, tissue share, Dice overlap with the mask used, tiles gained and lost, agreement with Space Ranger's tissue call |
| `<id>.prepare.json` | every pixel-size candidate and the value used (`native`), scale factor, image sizes, thumbnail scale factors (`fullres_to_thumbnail`), mask method and threshold, spatial-unit checks, tile counts (`n_extract`, `n_keep`) and drop reasons, all parameters; `"status": "skipped"` when a scale would need upsampling |

## `<sample>/mpp<scale>/channels/` — the input images of both extractors (Process 1b)

* `<id>.channels.json`: the input sets of each extractor, the bands of the channels image, the stain vectors (`stain_vectors`: method `slide` or `standard`, H, E and R = H × E in optical density, the angle between H and E and to the standard vectors, the number of stained pixels used, the share of their squared optical density in the H–E plane, and a `note` when the standard vectors had to be used or H and E are poorly separated), the H/E mapping and the share of pixels above it, and the `rgb_he` summary (median, 90th-percentile and largest share of optical density removed per tile; median change of R, G, B).
* `<id>.rgb_he_qc.png`: the slide next to its version rebuilt from H and E only, how much each pixel changed (white = unchanged, darkest = 40 levels or more), and the tiles with the median, 90th-percentile and largest change.
* `<id>.rgb_he_qc.tsv`: per extracted tile, `residual_share` (share of its optical density that `rgb_he` removes), `mean_abs_change` (mean change of R, G and B, in levels of 255) and `changed_fraction` (share of pixels where R, G or B moves by more than 10 levels).
* `<id>.channels.tif` (with `--publish_images`): the channels image, 8 bits per band, bands in the order listed in `channels.json` (H, E; R, G, B for fastLBP's `rgb`; R_he, G_he, B_he = `rgb_he`).

## `<sample>/mpp<scale>/uni2h/` — Output2a

`<id>.uni2h.h5`: `/tile_id` (n) and one block per input set, `/blocks/UNI2H_RGB/X` and `/blocks/UNI2H_RGB_HE/X` (n × 1536, float32; columns `uni2h_rgb_0000`… and `uni2h_rgb_he_0000`…); attributes record the model source, weights checksum, precision, normalisation, the input set and, for `rgb_he`, the stain vectors.

## `<sample>/mpp<scale>/lbp/` — Output2b

* `<id>.lbp.h5`: `/tile_id` and one block per channel, `/blocks/LBP_<channel>/X` (n × 1145 raw counts with the default radii; channels R, G, B, R_HE, G_HE, B_HE, H, E as requested by `--lbp_channels`; columns named `<channel>_r<R>_p<P>_u<code>`), with `--lbp_center` also `/blocks/LBP_CENTER_<channel>/X` (mean code vector of the central 2 × 2 pixels), and `/blocks/INTENSITY_HE/X` (n × 10). The exports group the channels into input sets: `LBP_RGB` (R, G, B), `LBP_RGB_HE` (R_he, G_he, B_he) and `LBP_HE` (H then E).
* `<id>.lbp_qc.png`, `.tsv`: the share of each tile's pixels with an irregular pattern, per channel and radius (mean, 10th, 50th and 90th percentile over the extracted tiles).

## `<sample>/mpp<scale>/st/` — Output3

`<id>.counts.h5` (10x-style CSC matrix, genes × tiles; `/matrix/barcodes` holds tile ids; integer counts, or float with `--visium_split fraction`, as the attribute `integer_counts` says), `<id>.counts_summary.tsv` (`n_units_summed`, `units_equiv`, `total_counts`, `n_genes_detected` per tile), `<id>.aggregate.json` (split method and seed, the share of the assigned units' molecules that landed in extracted tiles).

## `<sample>/mpp<scale>/cluster/` — clustering of every view (with `--cluster`)

* `<id>.clusters.tsv.gz`: one row per extracted tile: `tile_id`, `in_keep`, `in_extended` (the two tile sets) and the cluster of the tile in every clustering, in columns `<set>_<view>_<method>_k<k>` (e.g. `keep_UNI2H_RGB_leiden_k8`; empty for tiles outside the set). Clusters are numbered from 0, the largest, separately for every view and slide.
* `<id>.cluster_agreement.tsv`: one row per pair of views, tile set, method and k: `set`, `method`, `k`, `view_a`, `view_b`, `n_tiles`, `k_a` and `k_b` (clusters actually reached), `ari`, `nmi`, `null_moves`, `null_ari_mean`, `null_ari_p05`, `null_ari_p50`, `null_ari_p95` and `p_null` (the spatial null; `NA` without a tile grid). Rows with `method` = `leiden_vs_kmeans` compare the two methods on the same view.
* `<id>.cluster.json`: the parameters, and per tile set its definition, size, the values of k used, the reference view and, per view, the features, principal components, Leiden resolution and clusters reached for every k; the reason when a set or view was not clustered.
* `<id>.cluster_agreement_<set>.png`: ARI of every pair of views at `--cluster_k_ref` (Leiden and k-means; the grey diagonal compares the two methods on one view) and ARI of every view with the reference view for every k, with the spatial null.
* `<id>.cluster_maps_<set>.png`: the clusters of every view on the slide, coloured after the matching clusters of the reference view (`EXPRESSION`, or `UNI2H_RGB` for slides without expression).
* `<id>.cluster_contingency_keep.png`: for every view, where the tiles of each of its clusters fall among the reference view's clusters.

## `export/mpp<scale>/`

* `spe/<id>.rds` and `spe/combined_mpp<scale>.rds` — SpatialExperiment: `assay(spe, "counts")`, `colData(spe)` (tile table of the extracted tiles; with `--cluster` also the Leiden labels of the set `keep`, `cluster_<view>_leiden_k<k>`, text, `NA` outside the set), `spatialCoords(spe)` (tile centres, full-resolution px), `reducedDims(spe)` (`UNI2H_RGB`, `UNI2H_RGB_HE`, `LBP_RGB`, `LBP_RGB_HE`, `LBP_HE`, `LBP_CENTER_*`, `INTENSITY_HE`, those computed), `imgData(spe)` (thumbnail), `metadata(spe)$tileparity` (provenance — including `channels`, the content of `channels.json` with the stain vectors — and `lineage`, JSON text).
* `anndata/<id>.h5ad` and `anndata/combined_mpp<scale>.h5ad` — AnnData: `X` (tiles × genes counts, int32 or float32), `obs` (tile table; with `--cluster` also `cluster_<view>_leiden_k<k>`, text, empty outside the set `keep`), `obsm` (`X_uni2h_rgb`, `X_uni2h_rgb_he`, `X_lbp_rgb`, `X_lbp_rgb_he`, `X_lbp_he`, `X_lbp_center_*`, `X_intensity_he`, `spatial`), `uns["spatial"]` (thumbnail and scale factor, compatible with `scanpy.pl.spatial`), `uns["tileparity"]` (column names, provenance, `lineage`).
* `*.lineage.json` — the lineage of each object, including the checksum of the object file.

Both formats hold the same content. To write only one, add `--skip_anndata` or `--skip_spe`; [anndataR](https://bioconductor.org/packages/release/bioc/html/anndataR.html) converts between them (e.g. read an `.h5ad` in R as a SpatialExperiment), which may be the lighter option for large datasets.

```r
spe <- readRDS("results/export/mpp0.5000/spe/combined_mpp0.5000.rds")
spe <- spe[, spe$keep]                     # the default analysis set
uni <- reducedDim(spe, "UNI2H_RGB")        # tiles x 1536, UNI2-h on the slide
uni_he <- reducedDim(spe, "UNI2H_RGB_HE")  # tiles x 1536, UNI2-h on the slide rebuilt from H and E only
lbp <- reducedDim(spe, "LBP_RGB") / 50176  # LBP frequencies (224 x 224 px tiles), R then G then B
```

## `qc/mpp<scale>/`

With `--qc_umap`: `combined_mpp<scale>.qc_umap.html` (interactive 3-D UMAP of every view; works offline), `combined_mpp<scale>.qc_umap.tsv.gz` (the coordinates: `tile_id`, `view`, `umap1`–`umap3`) and its lineage.

## `pipeline_info/`

`software_versions.yml` (versions used by every process), Nextflow execution report, timeline, trace and DAG.
