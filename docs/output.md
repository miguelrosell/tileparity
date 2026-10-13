# Outputs

`<id>` is `<sample>_mpp<scale>`, e.g. `A_P1_CRC_mpp0.5000`.

## `<sample>/mpp<scale>/prepare/` — Output1

| file | content |
|---|---|
| `<id>.tiles.tsv.gz` | one row per candidate tile (any tissue): `tile_id`, `unit_type` (grid/spot), `row`, `col`, `barcode` (spot tiles), `x0_px`, `y0_px` (top-left corner in the resampled image), centre in resampled px (`cx_px`, `cy_px`), full-resolution px (`cx_fullres`, `cy_fullres`) and µm (`cx_um`, `cy_um`), `tissue_fraction` (share of the tile under the tissue mask), `st_coverage` (Visium HD: share of the tile covered by bins of the capture area), `n_st_units` (in-tissue bins summed for the tile; 1 for a spot tile), QC metrics (`lapvar`, `pen_fraction`, `dark_fraction`) and flags, `keep`, `qc_fail` (reasons a tile was dropped), sample metadata |
| `<id>.assignment.tsv.gz` | `barcode` → `tile_id` for every in-tissue Visium HD bin / Visium spot that falls in a candidate tile |
| `<id>.thumbnail.png`, `<id>.mask.png` | thumbnail at `--mask_mpp` and the tissue mask |
| `<id>.overlay.png` | thumbnail with spatial units (blue tint) and tiles (green kept, red dropped) |
| `<id>.prepare.json` | every pixel-size candidate and the value used (`native`), scale factor, image sizes, thumbnail scale factors (`fullres_to_thumbnail`), mask threshold, spatial-unit checks, tile counts and drop reasons, all parameters; `"status": "skipped"` when a scale would need upsampling |

## `<sample>/mpp<scale>/uni2h/` — Output2a

`<id>.uni2h.h5`: `/tile_id` (n), `/blocks/UNI2H/X` (n × 1536, float32), `/blocks/UNI2H/columns`; attributes record the model source, weights checksum, precision and normalisation.

## `<sample>/mpp<scale>/lbp/` — Output2b

`<id>.lbp.h5`: `/tile_id`, `/blocks/LBP_H/X`, `/blocks/LBP_E/X`, `/blocks/LBP_GREY/X` (n × 1145 raw counts each, columns named `<channel>_r<R>_p<P>_u<code>`), `/blocks/INTENSITY_HE/X` (n × 10). `<id>.channels.json` records the H/E mapping and how many pixels were clipped.

## `<sample>/mpp<scale>/st/` — Output3

`<id>.counts.h5` (10x-style CSC matrix, genes × tiles; `/matrix/barcodes` holds tile ids), `<id>.counts_summary.tsv` (units summed, total counts, genes detected per tile), `<id>.aggregate.json`.

## `export/mpp<scale>/`

* `spe/<id>.rds` and `spe/combined_mpp<scale>.rds` — SpatialExperiment: `assay(spe, "counts")`, `colData(spe)` (tile table), `spatialCoords(spe)` (tile centres, full-resolution px), `reducedDims(spe)` (`UNI2H`, `LBP_HE`, `LBP_GREY`, `INTENSITY_HE`), `imgData(spe)` (thumbnail), `metadata(spe)$tileparity` (provenance).
* `anndata/<id>.h5ad` and `anndata/combined_mpp<scale>.h5ad` — AnnData: `X` (tiles × genes counts), `obs` (tile table), `obsm` (`X_uni2h`, `X_lbp_he`, `X_lbp_grey`, `X_intensity_he`, `spatial`), `uns["spatial"]` (thumbnail and scale factor, compatible with `scanpy.pl.spatial`), `uns["tileparity"]` (column names, provenance).

```r
spe <- readRDS("results/export/mpp0.5000/spe/combined_mpp0.5000.rds")
uni <- reducedDim(spe, "UNI2H")         # tiles x 1536
lbp <- reducedDim(spe, "LBP_HE") / 50176 # LBP frequencies
```

## `pipeline_info/`

`software_versions.yml` (versions used by every process), Nextflow execution report, timeline, trace and DAG.
