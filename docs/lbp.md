# LBP step by step

This page explains, from the beginning, how `FASTLBP_EXTRACT` turns a tile into a list of numbers. Every term is defined when it first appears. Nothing here is specific to a library: the same steps are what fastLBP's kernel computes, and the tests check that tileparity's numbers equal fastLBP's own whole-image run.

**LBP** (local binary patterns) describes *texture*: how the values around each pixel go up and down. It does not describe colour or how much stain there is; that is why a separate intensity block exists (section 5).

## 1. The input: one number per pixel

LBP compares numbers, so it works on a single-channel image: one value from 0 to 255 per pixel. A colour image has three numbers per pixel, so Process 1b first computes single-channel images from the same RGB pixels (`--lbp_channels`, channel sets):

| set | channels | what a high value means |
|---|---|---|
| `he` (default) | H (haematoxylin) and E (eosin), by colour deconvolution | more of that stain (H: nuclei; E: cytoplasm, stroma) |
| `grey` (default) | grey = 0.299 R + 0.587 G + 0.114 B | brighter |
| `rgb` | R, G and B separately | more red, green or blue light |

Every channel is processed on its own, in exactly the same way. The rest of this page follows one channel.

## 2. One pixel and one radius

Take one pixel, the **centre** c, with value g_c.

### 2.1 The circle

Draw a circle of radius R pixels around c and place P points on it at equal angles. tileparity uses fastLBP's default number of points, **P = ⌈2πR⌉**: the length of the circle, 2πR pixels, rounded up (⌈x⌉ = the smallest whole number ≥ x). So the points are about one pixel apart along the circle:

| R | 2πR | P |
|---|---|---|
| 1 | 6.28 | 7 |
| 2 | 12.57 | 13 |
| 45 | 282.74 | 283 |

*Why 7 and not the 8 pixels around c?* The 8 pixels of the 3 × 3 square are not on a circle: the four diagonal ones are √2 ≈ 1.41 px away, the other four 1 px away. The classic textbook LBP uses those 8 pixels (P = 8); fastLBP instead puts the points on a true circle of radius R, which for R = 1 gives 7 points. Both are valid choices; the number of points can be set with `--lbp_npoints`.

### 2.2 The value at each point

The points rarely fall exactly on a pixel centre. The value at a point, g_p (p = 1, …, P), is computed from the four pixels around it by **bilinear interpolation**: a weighted average in which nearer pixels weigh more. A point exactly on a pixel takes that pixel's value.

### 2.3 Bits: is the neighbour at least as high as the centre?

Each point gets a **bit** (a 0 or a 1):

    bit_p = 1  if g_p ≥ g_c,   otherwise 0

Reading the P bits in order round the circle gives the **pattern** of the pixel, e.g. `1110000` for P = 7.

### 2.4 Changes round the circle

Count **U**, the number of changes 0→1 or 1→0 between consecutive bits, going once round the circle *including the step from the last bit back to the first*. Because you end where you started, U is always even: 0, 2, 4, …

* U = 0: all bits equal (`0000000` or `1111111`).
* U = 2: the ones form one unbroken arc (`1110000`, `0011100`, …).
* U ≥ 4: the ones are broken into two or more arcs (`1010000`, `1100110`, …).

### 2.5 Regular or irregular, and the code

* If **U ≤ 2** the pattern is **regular** (the literature says *uniform*). Its **code** is the number of ones, which can be 0, 1, 2, …, P.
* If **U ≥ 4** the pattern is **irregular**, and every irregular pattern gets the same code, **P + 1**.

The threshold U ≤ 2 is the same for every P. It does not count points; it asks a question about shape: *do the ones form at most one arc?* That question makes sense for 7 points and for 283.

**Why P + 2 codes.** The regular codes 0, 1, …, P are P + 1 different values (P + 1 and not P, because "no ones" is a code too). The irregular code P + 1 is one more value. Together: (P + 1) + 1 = **P + 2** possible codes. For P = 7: codes 0–7 (eight values) plus 8 (irregular) = 9 codes.

### 2.6 Two worked examples (P = 7, g_c = 120)

| | g_1 … g_7 (round the circle) | bits | U | regular? | code |
|---|---|---|---|---|---|
| A | 130, 140, 125, 90, 80, 100, 118 | 1 1 1 0 0 0 0 | 2 (1→0 after the third bit, and 0→1 from the last bit back to the first) | yes | 3 (three ones) |
| B | 130, 90, 125, 80, 140, 100, 118 | 1 0 1 0 1 0 0 | 6 | no | 8 (= P + 1) |

### 2.7 What the codes mean

* **0**: every neighbour is lower than the centre: c is a local peak (in the H channel: a spot with more haematoxylin than its surroundings).
* **P**: every neighbour is at least as high as the centre: c is a local pit, *or the area is flat* (equal values count as "at least as high"). Glass is nearly flat in the H and E channels (white to light-grey glass gives E = 0 and H a small, almost constant value, from 0 to about 10 out of 255), so much of it gets code P; scanner noise makes the rest irregular.
* **1 … P − 1**: an arc of higher neighbours on one side and lower on the other: an edge or a corner; the code says how wide the higher arc is.
* **P + 1**: anything more complicated (noise, fine or crossing structures).

The code does not change if the tissue is rotated: rotation moves the arc round the circle but does not change how many ones it has (**rotation invariance**). On a real image it is approximate, because the image is made of square pixels and the circle points are interpolated.

## 3. From 50 176 codes to one vector per tile

Every pixel of the tile is a centre in turn: a 224 × 224 px tile has 224 × 224 = **50 176 codes** for each radius.

They are not kept one by one. They are **counted**: the **histogram** of the tile for radius R is the list of P + 2 numbers "how many of the tile's pixels have code 0, how many code 1, …, how many code P + 1". The P + 2 numbers add up to 50 176. This is how evaluating every pixel gives a single, short vector per tile. The histogram says *how much* of each texture the tile contains, not *where* in the tile it is.

(The same thing in another vocabulary: give every pixel a **one-hot** vector of length P + 2, all zeros except a 1 at the position of its code; the histogram is the sum of the 50 176 one-hot vectors, and the histogram divided by 50 176 is their mean, i.e. "mean pooling".)

### Pixels near the tile border

For a pixel near the edge of the tile, part of its circle lies outside the tile. tileparity reads every tile with a **margin** of R_max + 1 = 46 px of the real surrounding slide, so those circle points get the real values of the neighbouring tissue. Only the 224 × 224 centres inside the tile are counted. Where the margin falls outside the slide itself, the slide is mirrored.

What would cutting the tile out alone change? fastLBP, like scikit-image, gives the value 0 to points outside the image. [`scripts/border_effect.py`](../scripts/border_effect.py) measures it on five 224 px tiles of a real stained section (scikit-image's immunohistochemistry image):

| R | pixels whose circle leaves the tile | codes changed, grey / H | histogram change, grey / H |
|---|---|---|---|
| 1 | 1.8 % | 1.3 % / 1.3 % | 0.7 % / 0.6 % |
| 8 | 13.8 % | 4.1 % / 3.9 % | 2.3 % / 2.3 % |
| 19 | 31.1 % | 4.1 % / 3.8 % | 2.7 % / 2.8 % |
| 45 | 64.2 % | 5.4 % / 3.0 % | 4.6 % / 2.3 % |

("Histogram change" is the total-variation distance between the two normalised histograms: the share of pixels that would have to move to another bin to turn one histogram into the other.) Far fewer codes change than circles leave the tile because, at large radii, most patterns are irregular with or without the real neighbours: at R = 45, 96 % of the pixels of these tiles are irregular (grey channel), and 90 % of the border pixels are irregular both with the real neighbours and with zeros outside. Small as it is, the change is not random noise: the part of the circle outside the tile always reads "lower than the centre", so it biases the border pixels of every tile in the same direction. That is why tileparity uses the real context: with the margin the codes are exactly those of a whole-slide computation.

## 4. Thirteen radii: 1145 numbers per channel

One radius sees structures of one size. tileparity uses 13 radii, fastLBP's progression round(1.499 × 1.327^i) up to a quarter of the tile (56 px): **1, 2, 3, 4, 5, 6, 8, 11, 14, 19, 25, 34, 45 px** (at 0.5 µm/px: 0.5 to 22.5 µm). Each radius gives its own histogram of P + 2 numbers, and the 13 histograms are written one after the other:

| R | P = ⌈2πR⌉ | P + 2 numbers | running total |
|---|---|---|---|
| 1 | 7 | 9 | 9 |
| 2 | 13 | 15 | 24 |
| 3 | 19 | 21 | 45 |
| 4 | 26 | 28 | 73 |
| 5 | 32 | 34 | 107 |
| 6 | 38 | 40 | 147 |
| 8 | 51 | 53 | 200 |
| 11 | 70 | 72 | 272 |
| 14 | 88 | 90 | 362 |
| 19 | 120 | 122 | 484 |
| 25 | 158 | 160 | 644 |
| 34 | 214 | 216 | 860 |
| 45 | 283 | 285 | **1145** |

1145 is not a multiple of 13 because every radius has a different number of points, so a different P + 2. Columns are named `<channel>_r<R>_p<P>_u<code>`, e.g. `H_r1_p7_u8` is the irregular bin of radius 1 in the H channel. A channel set puts its channels side by side: `LBP_HE` has 2 × 1145 = 2290 columns (H, then E), `LBP_GREY` 1145 and `LBP_RGB` 3435. Channel sets are alternatives to compare, never concatenated with each other.

## 5. Things to keep in mind when using the vector

1. **The irregular share grows with the radius.** More points on the circle means more chances for the arc to break. On the five tiles measured above (grey channel) 5 % of the pixels are irregular at R = 1, 74 % at R = 8, 90 % at R = 19 and 96 % at R = 45; how much this holds for your slides is what the QC figure below shows. A radius at which nearly every pixel is irregular puts almost all of its P + 2 numbers into one bin and adds little information. `lbp/<id>.lbp_qc.png` (and `.tsv`) shows this share for every channel and radius of every sample, so the useful radii can be chosen on real data (`--lbp_radii`, `--lbp_npoints`).
2. **The blocks of the vector do not weigh the same.** A raw concatenation gives large radii more columns, and a few bins dominate: in the test run (synthetic slide, 9 radii up to 14 px), the 9 irregular bins, 2.5 % of the 362 columns of a channel, carry 35–54 % of the variance, depending on channel and scale. Before computing distances or a PCA, use proportions (divide by 50 176), take square roots (the Hellinger transform, so that a few large bins do not dominate) and give each radius and channel the same total weight (multiple factor analysis: divide each block by the square root of its largest eigenvalue). The optional 3-D UMAP QC (`--qc_umap`) uses exactly this recipe; the files store the raw counts so that any other choice remains possible.
3. **LBP ignores how much stain there is.** Adding a constant to every value, or any change that keeps the order of values, gives the same bits. The `INTENSITY_HE` block stores what LBP throws away: the mean, standard deviation and 10th, 50th and 90th percentiles of H and of E (10 numbers per tile). It is a separate block, not part of the 1145 or 2290 numbers.

## 6. Optional: the codes of the central pixels only (`--lbp_center`)

`LBP_CENTER_<SET>` blocks hold, for each radius, the codes of the tile's **central pixels** only: the 2 × 2 pixels in the middle of the 224 px tile (an even size has no single middle pixel), stored as the mean of their four one-hot vectors (values 0, 0.25, 0.5, 0.75 or 1; each radius sums to 1). This answers "what does the tile look like at its centre" instead of "what does the whole tile contain". It is off by default and meant as a comparison with the histogram over all 50 176 pixels.

## 7. Implementation and checks

`FASTLBP_EXTRACT` calls fastLBP's kernel `fastlbp.lbp.uniform_lbp_uint8_padded_absolute` with absolute slide coordinates, the kernel that `fastlbp.run_fastlbp` uses for whole images. `tests/python/test_lbp.py` checks that:

* the per-tile histograms equal those of `fastlbp.run_fastlbp` on the whole image, for tiles whose margin lies inside the image (at the slide edge fastLBP reads 0 and tileparity mirrors);
* every radius of a tile sums to the number of pixels of the tile (each pixel counted once), and the default radii give 1145 numbers per channel;
* the central-pixel block and the irregular-pattern shares are computed as described here.
