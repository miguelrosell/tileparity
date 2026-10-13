# Design and rationale

This document records why each default was chosen. Every choice is a parameter, so it can be changed in `nextflow.config` (or on the command line) without touching code.

## The principle: decide geometry once

UNI2-h and fastLBP expect different inputs: UNI2-h takes a 224 × 224 px image and returns one vector; fastLBP takes a whole image and builds its own grid of histograms. Tiling the slide separately for each tool produces vectors that do not describe the same regions. `tileparity` therefore separates:

* **Process 1 (shared)** — every geometric decision: pixel size, tissue mask, tile positions, which tiles are kept, which spatial-transcriptomics unit belongs to which tile.
* **Processes 1a / 1b (adapters)** — only change how each pixel is represented for a given extractor, never which pixels enter.
* **Processes 2a / 2b (extractors)** and **3 (expression)** — read the tile table and add their own measurements.

| Aspect | UNI2-h (1a → 2a) | fastLBP (1b → 2b) | Same? |
|---|---|---|---|
| Source pixels | resampled RGB | resampled RGB | yes |
| Pixel size | target µm/px | target µm/px | yes |
| Tiles | tile table (x0, y0, 224 px) | tile table | yes |
| Kept tiles | `keep == true` | `keep == true` | yes |
| Pixel representation | 3 values, (x/255 − mean)/sd with ImageNet statistics | 1 value 0–255 per pass: H, E or grey | no, by definition of each method |
| Context outside the tile | none | up to R_max + 1 px of real neighbouring tissue for border pixels | no (see LBP) |

## Scale (multi-scale by default)

UNI2-h always receives 224 × 224 px, so the scale is set by the pixel size. The defaults are 64/224 = 0.2857, 0.5 and 1.0 µm/px, i.e. tiles of 64, 112 and 224 µm. 112 µm at 0.5 µm/px is the convention of the HEST benchmark. All three are multiples of the 8 µm Visium HD bin (8 × 8, 14 × 14 and 28 × 28 bins). Images are only downsampled: a scale finer than a slide's native resolution would need invented pixels and is skipped (tolerance 2 %). Resampling uses libvips' Lanczos-3 kernel.

LBP radii are fixed in pixels, so their physical size changes with the scale exactly as UNI2-h's field of view does: both methods always look at the same 224 × 224 px.

## Pixel size

The native pixel size is taken, in order of priority, from the samplesheet (`mpp`), Space Ranger's `microns_per_pixel`, the image metadata, or `bin_size_um / spot_diameter_fullres`. Two authoritative sources that disagree by more than 3 % stop the run; a disagreement with the image metadata only warns.

## Tissue mask

On a thumbnail at 4 µm/px:

1. Saturation S = (max(R,G,B) − min(R,G,B)) / max(R,G,B). Glass is white or grey (R ≈ G ≈ B, S ≈ 0); stained tissue is pink or purple. Saturation separates them better than brightness because pale tissue can be bright but is still stained.
2. Gaussian smoothing (σ = 8 µm); pixels with V = max(R,G,B)/255 < 0.1 (black scanner padding) are ignored.
3. Otsu's threshold: the t maximising the between-class variance w₀·w₁·(μ₀ − μ₁)².
4. Closing with a 16 µm disk, filling enclosed holes < 20 000 µm² (gland lumens), removing objects < 20 000 µm² (dust).

A tile is kept if at least 50 % of it is tissue. Known limitation: large areas of adipose tissue have low saturation and can be classified as background; raise `--mask_fill_holes_um2` if needed.

## Tiles

* **Grid** (Visium HD and image-only samples): a lattice anchored at (0, 0) of the resampled image. Image and expression cover exactly the same area. For Visium HD, a tile is also required to be at least 50 % covered by bins (`--min_st_coverage`), so tiles outside the capture area are dropped.
* **Spot** (standard Visium): a 224 px tile centred on each in-tissue spot (55 µm spot, 100 µm spacing). The image sees the spot plus its surroundings (about 5× the spot's area), the HEST convention. Neighbouring tiles overlap by 12 µm at 0.5 µm/px.
* **Cell level** is out of scope for this version: UNI2-h is built for tiles holding tens of cells, LBP histograms on a few hundred pixels are noisy, and per-cell Visium HD counts are sparse. The tile table only stores centre, size and `unit_type`, so a cell mode can be added to Process 1 later without changing the other processes.

## Quality control

Stored as columns of the tile table; only blur and tissue fraction enter the default `keep`:

* `lapvar`: variance of the Laplacian of the grey tile. A tile is `blur_flag`ged when log₁₀(lapvar) is more than 3 robust SDs (1.4826 × MAD) below the slide median.
* `pen_fraction` (green/cyan hue 70–200°, S > 0.25) and `dark_fraction` (V < 0.25): flags only.

## UNI2-h adapter (1a) and embedding (2a)

The architecture and normalisation follow the model card (ViT-H/14, 24 blocks, 1536-d, 8 register tokens, SwiGLU; ImageNet mean/std). The transform is written explicitly — no resize, no centre crop — so UNI2-h sees exactly the tile's pixels; timm's automatic `create_transform` could add a resize/crop depending on the model configuration. The embedding is the class token. Inference uses fp16 autocast on GPU (`--uni2h_precision`), fp32 on CPU.

## LBP adapter (1b)

LBP compares one number with another, so it needs single-channel images. All channels are computed pixel by pixel from the same RGB values:

* **H and E** by colour deconvolution (Ruifrok & Johnston 2001) as in `skimage.color.rgb2hed`: optical density OD = log(max(RGB/255, 10⁻⁶)) / log(10⁻⁶) (Beer–Lambert: OD is linear in the amount of dye), stains = max(OD · M⁻¹, 0), then a **fixed** mapping to 0–255 shared by all samples (H: 0–0.30, E: 0–0.15; fractions above the range are reported).
* **grey**: 0.299 R + 0.587 G + 0.114 B, as an ablation.

H and E separate nuclear texture from cytoplasm/stroma texture, which keeps the features interpretable.

## LBP features (2b)

For each pixel c, radius R and P points on the circle (bilinear interpolation), bit_p = 1 if g_p ≥ g_c. If the bit pattern has at most two 0/1 transitions ("uniform") the code is the number of ones (0…P), otherwise P + 1, giving P + 2 rotation-invariant codes. Each tile's histogram counts its 224 × 224 pixels (50 176 per radius and channel).

* **Radii**: fastLBP's progression round(1.499 · 1.327^i) up to a quarter of the tile: 1, 2, 3, 4, 5, 6, 8, 11, 14, 19, 25, 34, 45 px (0.5–22.5 µm at 0.5 µm/px), with P = ⌈2πR⌉. 1145 features per channel, 2290 for H + E.
* **Context**: the codes of border pixels use the real neighbouring tissue (the tile is read with a margin of R_max + 1 px). Cutting isolated tiles instead would compare border pixels with zeros (fastLBP fills outside the image with 0); in a 224 px tile that corrupts 1 − ((224 − 2R)/224)² of the pixels: 1.8 % at R = 1, 31 % at R = 19, 64 % at R = 45. The price is a small, documented asymmetry: for border pixels LBP looks up to 45 px beyond the tile, UNI2-h does not. Outside the slide the image is mirrored.
* **Implementation**: `fastlbp.lbp.uniform_lbp_uint8_padded_absolute` with absolute coordinates, i.e. fastLBP's own kernel. The unit test `tests/python/test_lbp.py` checks that the per-tile histograms are identical to `fastlbp.run_fastlbp` on the whole image.
* **Intensity block**: LBP is invariant to any strictly increasing change of intensity (the sign of g_p − g_c does not change), so it ignores how much stain there is. Mean, SD and 10/50/90th percentiles of H and E (10 numbers per tile) are stored separately, so analyses can separate "UNI2-h sees colour" from "UNI2-h sees structure".
* **No stain normalisation by default**: LBP codes are unaffected by brightness/contrast changes, and UNI2-h was trained on varied staining. The effect can be measured with the NCT-CRC-HE-100K benchmark, which ships with and without Macenko normalisation.

## Expression per tile (3)

With X (genes × units) the raw UMI counts and A (units × tiles) the assignment matrix written by Process 1 (A[b, t] = 1 if unit b's centre lies in tile t; at most one 1 per row):

Y = X · A,  Y[g, t] = Σ_{b ∈ t} X[g, b]

Y[g, t] is the number of molecules of gene g captured over the tile's area — what one spot shaped like the tile would have measured. Summing (not averaging) keeps counts: a sum of Poisson counts is Poisson, so count models and library-size normalisation remain valid, and a tile of ~196 bins is √196 = 14 times more precise than one bin. The number of summed units and the total counts per tile are stored. For spot-centred tiles each tile has exactly its own spot.

Only the image Space Ranger was given has the pixel grid of the unit coordinates. Process 1 stops if fewer than 99 % of the units fall inside the image, and draws the units over the thumbnail (`overlay.png`) for a visual check.

## Exports (4)

SpatialExperiment (primary, R/Bioconductor) and AnnData (Python) are built independently from the same intermediate files. Per sample: `assay(counts)` genes × tiles, `colData` = tile table, `spatialCoords` = tile centre in full-resolution pixels, `reducedDims` = `UNI2H`, `LBP_HE`, `LBP_GREY`, `INTENSITY_HE`, `imgData` = thumbnail. A combined object per scale keeps the genes shared by all samples. LBP histograms are stored as raw counts; divide by 50 176 for frequencies.
