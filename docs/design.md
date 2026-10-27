# Design and rationale

This document records why each default was chosen. Every choice is a parameter, so it can be changed in `nextflow.config` (or on the command line) without touching code. The map of processes and files is in [structure.md](structure.md); LBP is explained step by step in [lbp.md](lbp.md).

## The principle: decide geometry once

UNI2-h and fastLBP expect different inputs: UNI2-h takes a 224 × 224 px image and returns one vector; fastLBP takes a whole image and builds its own grid of histograms. Tiling the slide separately for each tool produces vectors that do not describe the same regions. `tileparity` therefore separates:

* **Process 1 (shared)** — every geometric decision: pixel size, tissue mask, tile positions, which tiles are extracted and kept, which spatial-transcriptomics unit belongs to which tile (and with which share).
* **Process 1b (shared)** — every input image of the extractors, computed once: the slide as it is (input set `rgb`) and the slide rebuilt from haematoxylin and eosin only (`rgb_he`), so both extractors read exactly the same pixels in each set.
* **Adapter 1a and the reading of 2b** — only change how those pixels are represented for a given extractor (ImageNet normalisation for UNI2-h; one channel at a time for LBP), never which pixels enter.
* **Processes 2a / 2b (extractors)** and **3 (expression)** — read the tile table and add their own measurements.

| Aspect | UNI2-h (1a → 2a) | fastLBP (2b) | Same? |
|---|---|---|---|
| Source pixels | input sets `rgb` and `rgb_he` (Process 1b) | input sets `rgb` and `rgb_he` (+ optional `he`) | yes |
| Pixel size | target µm/px | target µm/px | yes |
| Tiles | tile table (x0, y0, 224 px) | tile table | yes |
| Tiles with features | `extract == true` | `extract == true` | yes |
| Pixel representation | the 3 channels together, (x/255 − mean)/sd with ImageNet statistics | each channel on its own, 0–255 | same channels; the representation differs by definition of each method |
| Context outside the tile | none | up to R_max + 1 px of real neighbouring tissue for border pixels | no (see LBP) |

## Scale (multi-scale by default)

UNI2-h always receives 224 × 224 px, so the scale is set by the pixel size. The defaults are 64/224 = 0.2857, 0.5 and 1.0 µm/px, i.e. tiles of 64, 112 and 224 µm. 112 µm at 0.5 µm/px is the convention of the HEST benchmark. All three are multiples of the 8 µm Visium HD bin (8 × 8, 14 × 14 and 28 × 28 bins). Images are only downsampled: a scale finer than a slide's native resolution would need invented pixels and is skipped (tolerance 2 %). Resampling uses libvips' Lanczos-3 kernel.

LBP radii are fixed in pixels, so their physical size changes with the scale exactly as UNI2-h's field of view does: both methods always look at the same 224 × 224 px.

## Pixel size

The native pixel size (µm/px) is taken from the first available source in this order:

1. the samplesheet (`mpp` column);
2. Space Ranger's `microns_per_pixel` (Visium HD);
3. Space Ranger's `bin_size_um / spot_diameter_fullres` (Visium HD);
4. the spacing of the spatial units: the median distance from each unit to its nearest neighbour, in full-resolution pixels, corresponds to 100 µm for Visium spots and to one bin for Visium HD bins;
5. the image metadata (TIFF resolution tags, OME-XML, Aperio `MPP`);
6. 65 µm / `spot_diameter_fullres` (standard Visium). Space Ranger documents `spot_diameter_fullres` as the diameter of a *theoretical 65 µm spot*, not of the 55 µm capture spot.

Sources 1–4 describe the pixel grid of the spatial coordinates themselves: if two of them disagree by more than 3 % the run stops (wrong `mpp` in the samplesheet, or an image other than Space Ranger's). A disagreement with sources 5–6 only produces a warning. Every candidate and the chosen value are written to `prepare.json`. The spacing check is not applied to HEST inputs, because HEST-1k also contains older arrays with other spacings; their pixel size comes from the samplesheet (`scripts/hest_download.py` writes HEST's own estimate, derived from the spot spacing).

## Tissue mask

On a thumbnail at 4 µm/px (`--mask_mpp`), one of three automatic methods (`--mask_method`), all with their parameters in micrometres:

1. A per-pixel *tissue score*, higher for tissue than for glass:
   * `saturation` (default): S = (max(R,G,B) − min(R,G,B)) / max(R,G,B). Glass is white or grey (R ≈ G ≈ B, S ≈ 0); stained tissue is pink or purple. Saturation separates them better than brightness because pale tissue can be bright but is still stained.
   * `grey`: darkness, 1 − (0.299 R + 0.587 G + 0.114 B)/255. Glass is bright.
   * `od`: mean optical density −log₁₀(I/255) over R, G and B, i.e. how much light the stain absorbs (Beer–Lambert); Macenko's stain normalisation uses OD < 0.15 for "background".
2. Gaussian smoothing (σ = 8 µm); pixels with V = max(R,G,B)/255 < 0.1 (black scanner padding) are ignored.
3. Otsu's threshold: the t maximising the between-class variance w₀·w₁·(μ₀ − μ₁)².
4. Closing with a 16 µm disk, filling enclosed holes < 20 000 µm² (gland lumens), removing objects < 20 000 µm² (dust).

**External mask.** A samplesheet column `mask` can give any image of the whole slide (e.g. exported from QuPath, ilastik or a segmentation network) instead: it is resized to the thumbnail with nearest-neighbour interpolation and used as it is (non-zero pixels are tissue in a 0/1 mask, values ≥ 128 in an 8-bit mask). Its aspect ratio must match the slide's within 2 %, so that a mask of a crop is not silently stretched.

**Comparison (`--mask_compare`).** `prepare/<id>.mask_compare.png` and `.tsv` put the mask that was used next to the other automatic methods at 4 µm/px and the saturation method at finer resolutions (`--mask_compare_mpp`, default 1.0 and 0.5 µm/px, read from the full-resolution image; resolutions finer than the scan are skipped). For each: the tissue share of the image, the Dice overlap with the mask used (2|A∩B|/(|A|+|B|), 1 = identical), how many tiles would change their decision, and, for spatial samples, the share of Space Ranger's in-tissue units inside the mask and of its background units outside it. Space Ranger's tissue call is itself automatic: it is a reference, not the truth. Fine resolutions of large slides (more than `--mask_compare_max_mpx` megapixels, default 400) are computed inside the tissue's bounding box plus 500 µm, where Otsu's threshold sees less glass, and skipped if even that box is too large; the table's `note` column says which.

**Thresholds.** A tile's `tissue_fraction` is the share of its area under the mask. Two thresholds use it:

* `--min_tissue_extract` (default: the same as `--min_tissue`): features and counts are computed for tiles with at least this much tissue (column `extract`).
* `--min_tissue` (0.5): tiles with at least this much tissue are `keep`, the default set for analysis.

A lower extraction threshold (e.g. 0.3) costs computation but lets an analysis compare thresholds without rerunning. `prepare/<id>.tissue_thresholds.png` shows which tiles each of `--qc_tissue_thresholds` (30, 50 and 80 %) would keep.

Known limitations: large areas of adipose tissue have low saturation and can be classified as background (raise `--mask_fill_holes_um2`, try `--mask_method grey`, or give an external mask); and Otsu's method needs some glass in the image — on an image cropped to the tissue it would split the tissue itself into two classes. Process 1 warns when the mask covers more than 90 % of the image.

## Tiles

* **Grid** (default for every technology): a lattice of 224 px tiles anchored at (0, 0) of the resampled image, without overlap, so image and expression cover exactly the same area and no tissue is counted twice.
  * Visium HD: a bin belongs to the tile holding its centre. A tile must be at least 50 % covered by bins (`--min_st_coverage`), so tiles outside the capture area are dropped.
  * Standard Visium and HEST: spots (55 µm discs, 100 µm apart) do not tile the slide and straddle tile borders, so each spot is split between the tiles it touches in proportion to its area in each (see Process 3). A tile must hold at least 0.25 spots' worth of capture area (`--min_spot_overlap`; column `st_units_equiv`, the sum of the area shares of the spots in the tile).
* **Spot** (`--visium_tiling spot`, or `spot` in a samplesheet row; standard Visium only): a 224 px tile centred on each in-tissue spot. The tile shows the spot plus its surroundings — 1.7×, 5.3× and 21× the spot's area at 0.2857, 0.5 and 1.0 µm/px — while the expression is that of the central spot only. Neighbouring tiles overlap at 0.5 and 1.0 µm/px, so the same tissue appears in several tiles. Spots whose tile would extend beyond the image are dropped (`qc_fail = outside_image`).
* **Cell level** is out of scope for this version: UNI2-h is built for tiles holding tens of cells, LBP histograms on a few hundred pixels are noisy, and per-cell Visium HD counts are sparse. The tile table only stores centre, size and `unit_type`, so a cell mode can be added to Process 1 later without changing the other processes.

The grid is the default for standard Visium as well (it was spot-centred in 0.1): one tiling for every technology, no overlap, and the image of a tile always matches the area its expression comes from. The price is that a tile's expression comes from the parts of the spots inside it, i.e. from a sample of the tile's area: spots cover about 27 % of the capture area, and the gaps between them are not measured. On the Visium lattice ([`scripts/visium_grid_coverage.py`](../scripts/visium_grid_coverage.py)) a tile holds on average 0.47, 1.45 and 5.8 spots' worth of area at 64, 112 and 224 µm (at least 1.0 at 112 µm and 4.9 at 224 µm); at 64 µm (0.2857 µm/px) about 17 % of the tiles inside the capture area hold less than 0.25 spots' worth and are not extracted, which `--min_spot_overlap` controls.

## Quality control

Stored as columns of the tile table. By default they are only flags: `keep` depends on the tissue fraction (and, for spatial samples, on the presence of expression), so no signal is thrown away before it can be checked, e.g. whether flagged tiles drive a clustering. `--drop_blurred` also removes blur-flagged tiles.

* `lapvar`: variance of the Laplacian of the grey tile. A tile is `blur_flag`ged when log₁₀(lapvar) is more than 3 robust SDs (1.4826 × MAD) below the median of the slide's tiles with at least `--min_tissue` tissue. Naturally smooth tissue (mucin, fat) can be flagged too, which is why the flag does not drop tiles by default.
* `pen_fraction` (green/cyan hue 70–200°, S > 0.25) and `dark_fraction` (V < 0.25): flags only.

## UNI2-h adapter (1a) and embedding (2a)

The architecture and normalisation follow the model card (ViT-H/14, 24 blocks, 1536-d, 8 register tokens, SwiGLU; ImageNet mean/std). The transform is written explicitly — no resize, no centre crop — so UNI2-h sees exactly the tile's pixels; timm's automatic `create_transform` could add a resize/crop depending on the model configuration. The embedding is the class token. Inference uses fp16 autocast on GPU (`--uni2h_precision`), fp32 on CPU. Each input set (`--uni2h_inputs`, next section) is embedded separately: `rgb` from the resampled image, `rgb_he` from the bands R_he, G_he, B_he of the channels image of Process 1b, giving the blocks `UNI2H_RGB` and `UNI2H_RGB_HE` (1536 columns each, `uni2h_rgb_0000`, … and `uni2h_rgb_he_0000`, …). Two input sets take twice the GPU time.

## The input images of both extractors (1b)

UNI2-h reads RGB images; LBP compares one number with another, so it reads one channel at a time. For a fair comparison both receive the same images, in **input sets** (`--uni2h_inputs`, `--lbp_channels`) computed once, pixel by pixel, from the resampled RGB image:

| input set | the image | UNI2-h | fastLBP | default |
|---|---|---|---|---|
| `rgb` | the slide as it is | reads the image: `UNI2H_RGB` | one LBP per channel R, G, B: `LBP_RGB` | on for both |
| `rgb_he` | the slide rebuilt from haematoxylin and eosin only | reads the image: `UNI2H_RGB_HE` | one LBP per channel R_he, G_he, B_he: `LBP_RGB_HE` | on for both |
| `he` | the amounts of H and of E, two single-channel images | — (needs 3 channels) | `LBP_H`, `LBP_E` = `LBP_HE` | off |

The sets are alternatives to compare, stored as separate blocks and never concatenated with each other. Comparing `rgb` with `rgb_he` in each extractor tells how much it relies on colour that is neither H nor E (scanner tint, pigment, pen): a batch effect that disappears in `rgb_he` came from that colour.

**How `rgb_he` is made** (colour deconvolution, Ruifrok & Johnston 2001):

1. optical density per pixel and channel, OD = log(max(RGB/255, 10⁻⁶)) / log(10⁻⁶). By Beer–Lambert, OD is proportional to the amount of dye, and the OD of two dyes on top of each other is the sum of their ODs;
2. every pixel's OD (a point in 3-D) is written as a_H · H + a_E · E + a_R · R, where H and E are the colours of the two stains (unit vectors in OD space) and R = H × E is the direction perpendicular to both;
3. `rgb_he` keeps a_H · H + a_E · E — the closest colour made of H and E only — and turns it back into RGB, 255 · 10⁻⁶^OD, rounded to 8 bits.

The removed part a_R · R is what two stains cannot explain: pen marks, pigment, the colour of the glass or a tint of the scanner, noise. For an H&E slide it is small: on the tiles of two real H&E images 3–5 % of the optical density, and R, G and B move by 2–3 levels (of 255) on average. Strongly red blood cells (haemoglobin's own colour on top of eosin) can lose part of their red. `channels/<id>.rgb_he_qc.png` shows, for every sample and scale, the slide, its rebuilt version, how much each pixel changed and the tiles with the median, 90th-percentile and largest change; `rgb_he_qc.tsv` has the numbers per tile.

Why not give UNI2-h the H and E amounts directly: its first layer takes exactly three channels (a 14 × 14 px patch embedding of R, G and B) and it was trained on ordinary RGB images of H&E slides. An image made of H, E and a third channel would be unlike anything it has seen. `rgb_he` is still an ordinary-looking H&E image — R, G, B built from two numbers per pixel — so UNI2-h can read it and fastLBP reads the same pixels. The grey channel set of tileparity 0.2 was removed for the same reason: a fixed mixture of R, G and B (0.299, 0.587, 0.114, tuned to human brightness) has no counterpart in UNI2-h, and its information is in R, G and B.

**Stain vectors** (`--stain_vectors`):

* `slide` (default): estimated on each slide with the method of Macenko et al. (2009), from about one million random pixels of up to 500 random extracted tiles (fixed seed): keep the stained pixels (length of the natural-log OD vector ≥ 0.15), find the plane through the origin that holds most of their optical density (the two largest singular vectors of the OD matrix), and take as H and E the directions at the 1st and 99th percentile of the angle in that plane (H: the one that absorbs more red). Common implementations of Macenko keep the pixels with OD ≥ 0.15 in every channel instead; that drops pale eosin-stained stroma, which absorbs little red, and on a real H&E image left 5.6 % of the OD outside the plane instead of 3.5 %.
* `standard`: Ruifrok and Johnston's published vectors (those of `skimage.color.rgb2hed`), the same for every slide. Also used when a slide's sample has fewer than 10 000 stained pixels or the estimate is not a plausible pair of stain colours (a clearly negative component, or H and E less than 5° apart); `channels.json` says so.

Stain colours vary between laboratories, protocols and scanners. On three real H&E images, the slides' own vectors left 1–4 % of the optical density outside the plane (R, G, B moved by 1–3 levels on average); the standard vectors 5–22 % (3–23 levels), the largest on a Visium image with a grey tint, whose rebuilt version visibly changed colour. The estimated vectors were 7–20° (H) and 7–34° (E) away from the standard ones. `rgb_he` depends only on the plane, not on the two directions within it; the directions matter for the `he` channels and the intensity block. When H and E come out less than 15° apart (a slide with little eosin, for example), `channels.json` notes that their amounts are poorly separated.

**`he` channels**: max(a, 0) mapped to 0–255 with a **fixed** range shared by all samples (H: 0–0.30, E: 0–0.15 in the OD units above; the fractions of pixels above the range are reported in `channels.json`). H and E separate nuclear texture from cytoplasm/stroma texture. They are always computed, because fastLBP's intensity block (below) is made from them.

## LBP features (2b)

For each pixel c, radius R and P points on the circle (bilinear interpolation), bit_p = 1 if g_p ≥ g_c. If the bit pattern has at most two 0/1 changes round the circle (a *regular* or "uniform" pattern) the code is the number of ones (0…P), otherwise P + 1, giving P + 2 rotation-invariant codes. Each tile's histogram counts the codes of its 224 × 224 pixels (50 176 per radius and channel). [lbp.md](lbp.md) explains every step with examples.

* **Radii**: fastLBP's progression round(1.499 · 1.327^i) up to a quarter of the tile: 1, 2, 3, 4, 5, 6, 8, 11, 14, 19, 25, 34, 45 px (0.5–22.5 µm at 0.5 µm/px), with P = ⌈2πR⌉. 1145 features per channel, 3435 for an RGB input set.
* **Context**: the codes of border pixels use the real neighbouring tissue (the tile is read with a margin of R_max + 1 px), so they equal those of a whole-slide computation. Cutting isolated tiles instead would give the circle points outside the tile the value 0 (fastLBP's convention outside the image). At R = 45 the circles of 64 % of a 224 px tile's pixels reach outside it, but most of those patterns are irregular with or without the real neighbours, so on a real stained section only 1–5 % of the codes change (depending on radius and channel; 3.0–4.5 % at R = 45), and the histogram by 0.6–3.9 % (total-variation distance; measured by [`scripts/border_effect.py`](../scripts/border_effect.py), table in [lbp.md](lbp.md#pixels-near-the-tile-border)). Small but systematic, hence the margin. The price is a small, documented asymmetry: for border pixels LBP looks up to 45 px beyond the tile, UNI2-h does not. Outside the slide the image is mirrored.
* **Implementation**: `fastlbp.lbp.uniform_lbp_uint8_padded_absolute` with absolute coordinates, i.e. fastLBP's own kernel. The unit test `tests/python/test_lbp.py` checks that the per-tile histograms are identical to `fastlbp.run_fastlbp` on the whole image.
* **Irregular patterns**: large circles have many points and are irregular more often; a radius at which nearly every pixel is irregular adds little information. `lbp/<id>.lbp_qc.png` and `.tsv` report the share per channel and radius on every slide, to choose `--lbp_radii` on real data.
* **Central pixels (optional, `--lbp_center`)**: `LBP_CENTER_<SET>` blocks with the codes of the tile's central 2 × 2 pixels only (mean of their one-hot code vectors), to compare "the tile's centre" with "the whole tile".
* **Intensity block**: LBP is invariant to any strictly increasing change of intensity (the sign of g_p − g_c does not change), so it ignores how much stain there is. Mean, SD and 10/50/90th percentiles of H and E (10 numbers per tile) are stored as a separate block (`INTENSITY_HE`), so analyses can separate "UNI2-h sees colour" from "UNI2-h sees structure".
* **Analysis recipe**: the stored values are raw counts. The blocks of a concatenated vector do not weigh the same (more columns at large radii; the irregular bins carry a large share of the variance), so analyses should use proportions, the Hellinger transform (square root) and equal weight per radius and channel (multiple factor analysis). The QC UMAP uses this recipe.
* **No stain normalisation to a reference**: LBP codes are unaffected by brightness/contrast changes, and UNI2-h was trained on varied staining. The input set `rgb_he` removes only the colour that is neither H nor E; it does not map the slide's H and E onto reference colours. The effect of full normalisation can be measured with the NCT-CRC-HE-100K benchmark, which ships with and without Macenko normalisation.

## Expression per tile (3)

With X (genes × units) the raw UMI counts and A (units × tiles) the assignment matrix written by Process 1, where A[b, t] is the share of unit b that lies in tile t:

* **Visium HD bins and spot-centred tiles**: A[b, t] = 1 for the tile holding the unit's centre (at most one 1 per row), and

  Y = X · A,  Y[g, t] = Σ_{b ∈ t} X[g, b]

  Y[g, t] is the number of molecules of gene g captured over the tile's area — what one spot shaped like the tile would have measured. Summing (not averaging) keeps counts: a sum of Poisson counts is Poisson, so count models and library-size normalisation remain valid, and a tile of ~196 bins is √196 = 14 times more precise than one bin.

* **Visium spots on the grid**: A[b, t] is the share of the 55 µm disc of spot b inside tile t, computed by sampling the disc with about 5 000 evenly spaced points (accurate to a few tenths of a percent). A spot's shares sum to 1, or less where part of it falls on tiles that are not extracted (that part is not counted). Two ways to split the molecules (`--visium_split`):
  * `multinomial` (default): every molecule of the spot is given to one tile at random, tile t with probability A[b, t]. Counts stay integers and every molecule is counted at most once, so neighbouring tiles share no molecules; if a spot's counts are Poisson, the parts are independent Poisson counts (the principle behind count splitting). The random seed (`--split_seed`, combined with the sample and scale) is fixed and recorded, so a rerun gives the same counts.
  * `fraction`: Y = X · A with the shares as weights, the expected value of the multinomial split. Counts become non-integers and one spot's molecules are spread over several tiles, which makes neighbouring tiles correlated.

  Both assume that the molecules of a spot are spread evenly over its area. `st_units_equiv` (spots' worth of area in the tile) and `st_coverage` (share of the tile's area covered by spots) are stored, because a tile's total counts depend on them.

The number of summed units and the total counts per tile are stored (`counts_summary.tsv`). Only the image Space Ranger was given has the pixel grid of the unit coordinates. Process 1 stops if fewer than 99 % of the in-tissue units fall inside the image (units of the capture area outside the tissue may legitimately lie outside a cropped image), and draws the units over the thumbnail (`overlay.png`) for a visual check.

## Exports (4)

SpatialExperiment (primary, R/Bioconductor) and AnnData (Python) are built independently from the same intermediate files. Per sample: `assay(counts)` genes × tiles, `colData` = tile table (the extracted tiles, with `extract`, `keep` and `qc_fail`), `spatialCoords` = tile centre in full-resolution pixels, `reducedDims` = `UNI2H_RGB`, `UNI2H_RGB_HE`, `LBP_RGB`, `LBP_RGB_HE`, `LBP_HE`, `LBP_CENTER_*` (those computed) and `INTENSITY_HE`, `imgData` = thumbnail. A combined object per scale keeps the genes shared by all samples with expression. LBP histograms are stored as raw counts; divide by 50 176 for frequencies. Counts are integers except with `--visium_split fraction`.

## Lineage and figures

Every step writes a lineage file with its command, parameters, software versions and the checksums of its input and output files, and inherits the records of the steps upstream; the exported objects carry the complete route from the original image ([structure.md](structure.md#lineage-where-every-output-comes-from)). This is tileparity's own JSON format rather than a Nextflow plugin, so it also works offline and travels inside the objects.

Every figure carries a caption in four parts (what it shows, how to read it, caveats, where it comes from) inside the PNG, so it keeps its meaning when it is copied out of the results folder.

## Optional QC: 3-D UMAP (`--qc_umap`)

One interactive HTML page per scale with a 3-D UMAP of every feature view: UNI2-h on each input set (z-scored), LBP on each input set (the analysis recipe above) and expression (counts per 10 000, log1p, 2 000 most variable genes; samples without expression are left out of this view), each after PCA to at most 50 components. Points can be coloured by sample, `keep`, tissue fraction, blur flag or total counts. It is a quick look for batch effects and artefacts, not an analysis: UMAP keeps neighbourhoods but distorts distances.
