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

## `<sample>/mpp<scale>/uni2h/` — Output2a

`<id>.uni2h.h5`: `/tile_id` (n), `/blocks/UNI2H/X` (n × 1536, float32), `/blocks/UNI2H/columns`; attributes record the model source, weights checksum, precision and normalisation.

## `<sample>/mpp<scale>/lbp/` — Output2b

* `<id>.lbp.h5`: `/tile_id` and one block per channel, `/blocks/LBP_<channel>/X` (n × 1145 raw counts with the default radii; channels H, E, GREY, R, G, B as requested by `--lbp_channels`; columns named `<channel>_r<R>_p<P>_u<code>`), with `--lbp_center` also `/blocks/LBP_CENTER_<channel>/X` (mean code vector of the central 2 × 2 pixels), and `/blocks/INTENSITY_HE/X` (n × 10). The exports group the channels into sets: `LBP_HE` (H then E), `LBP_GREY`, `LBP_RGB` (R, G, B).
* `<id>.lbp_qc.png`, `.tsv`: the share of each tile's pixels with an irregular pattern, per channel and radius (mean, 10th, 50th and 90th percentile over the extracted tiles).
* `<id>.channels.json`: the channels written, the H/E mapping and how many pixels were clipped.

## `<sample>/mpp<scale>/st/` — Output3

`<id>.counts.h5` (10x-style CSC matrix, genes × tiles; `/matrix/barcodes` holds tile ids; integer counts, or float with `--visium_split fraction`, as the attribute `integer_counts` says), `<id>.counts_summary.tsv` (`n_units_summed`, `units_equiv`, `total_counts`, `n_genes_detected` per tile), `<id>.aggregate.json` (split method and seed, the share of the assigned units' molecules that landed in extracted tiles).

## `export/mpp<scale>/`

* `spe/<id>.rds` and `spe/combined_mpp<scale>.rds` — SpatialExperiment: `assay(spe, "counts")`, `colData(spe)` (tile table of the extracted tiles), `spatialCoords(spe)` (tile centres, full-resolution px), `reducedDims(spe)` (`UNI2H`, `LBP_HE`, `LBP_GREY`, `LBP_RGB`, `LBP_CENTER_*`, `INTENSITY_HE`, those computed), `imgData(spe)` (thumbnail), `metadata(spe)$tileparity` (provenance and `lineage`, JSON text).
* `anndata/<id>.h5ad` and `anndata/combined_mpp<scale>.h5ad` — AnnData: `X` (tiles × genes counts, int32 or float32), `obs` (tile table), `obsm` (`X_uni2h`, `X_lbp_he`, `X_lbp_grey`, `X_lbp_rgb`, `X_lbp_center_*`, `X_intensity_he`, `spatial`), `uns["spatial"]` (thumbnail and scale factor, compatible with `scanpy.pl.spatial`), `uns["tileparity"]` (column names, provenance, `lineage`).
* `*.lineage.json` — the lineage of each object, including the checksum of the object file.

Both formats hold the same content. To write only one, add `--skip_anndata` or `--skip_spe`; [anndataR](https://bioconductor.org/packages/release/bioc/html/anndataR.html) converts between them (e.g. read an `.h5ad` in R as a SpatialExperiment), which may be the lighter option for large datasets.

```r
spe <- readRDS("results/export/mpp0.5000/spe/combined_mpp0.5000.rds")
spe <- spe[, spe$keep]                     # the default analysis set
uni <- reducedDim(spe, "UNI2H")            # tiles x 1536
lbp <- reducedDim(spe, "LBP_HE") / 50176   # LBP frequencies (224 x 224 px tiles)
```

## `qc/mpp<scale>/`

With `--qc_umap`: `combined_mpp<scale>.qc_umap.html` (interactive 3-D UMAP of every view; works offline), `combined_mpp<scale>.qc_umap.tsv.gz` (the coordinates: `tile_id`, `view`, `umap1`–`umap3`) and its lineage.

## `pipeline_info/`

`software_versions.yml` (versions used by every process), Nextflow execution report, timeline, trace and DAG.
