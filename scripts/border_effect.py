#!/usr/bin/env python3
"""How much does cutting a tile out of the slide change its LBP codes and histogram?

Reproduces the numbers in docs/lbp.md ("Pixels near the tile border") and docs/design.md.

For five 224 x 224 px tiles of a real stained section (scikit-image's bundled
immunohistochemistry image, 512 x 512 px), the uniform rotation-invariant LBP codes of
every pixel are computed twice for each default radius R:

* with context: on the tile plus a margin of R + 1 px of the real neighbouring image, then
  keeping the 224 x 224 centres. This is what tileparity does;
* alone: on the 224 x 224 tile cut out of the image, where the circle points that fall
  outside the tile read 0 (scikit-image, like fastLBP, uses 0 outside the image).

Printed per radius: the share of pixels whose circle reaches outside the tile (geometry,
1 - ((224 - 2R) / 224)^2), the share of pixels whose code changes, and the change of the
tile's histogram as a total-variation distance (half the sum of the absolute differences
of the two normalised histograms: the share of pixels that would have to move to another
bin to turn one histogram into the other). Also printed: the share of pixels with an
irregular pattern (code P + 1) computed with context, and the share of the border pixels
(those whose circle leaves the tile) that are irregular both with and without context.

    python scripts/border_effect.py
"""

import sys
from pathlib import Path

import numpy as np
from skimage import color, data
from skimage.feature import local_binary_pattern

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from tileparity_lib.lbp import default_radii, npoints_for  # noqa: E402

T = 224
ORIGINS = [(46, 46), (46, 242), (242, 46), (242, 242), (140, 140)]  # top-left corners; 46 px of image around each


def channels() -> dict:
    rgb = data.immunohistochemistry()
    grey = np.round(color.rgb2gray(rgb) * 255).astype(np.uint8)
    h = color.rgb2hed(rgb)[..., 0]
    h = np.round(np.clip((h - h.min()) / (np.percentile(h, 99.9) - h.min()), 0, 1) * 255).astype(np.uint8)
    return {"grey": grey, "H": h}


def main() -> None:
    for name, img in channels().items():
        print(f"\nchannel {name}: R, P, % of pixels whose circle leaves the tile, % of codes changed "
              "(mean and range over 5 tiles), histogram change (total-variation distance, %)")
        yy, xx = np.mgrid[0:T, 0:T]
        for R in default_radii(45):
            P = npoints_for(R)
            border = (yy < R) | (yy >= T - R) | (xx < R) | (xx >= T - R)
            changed, tv, irregular, irregular_both = [], [], [], []
            for y0, x0 in ORIGINS:
                m = R + 1
                with_context = local_binary_pattern(img[y0 - m:y0 + T + m, x0 - m:x0 + T + m], P, R,
                                                    method="uniform")[m:m + T, m:m + T]
                alone = local_binary_pattern(img[y0:y0 + T, x0:x0 + T], P, R, method="uniform")
                changed.append(100 * np.mean(with_context != alone))
                h1 = np.bincount(with_context.astype(int).ravel(), minlength=P + 2) / T**2
                h2 = np.bincount(alone.astype(int).ravel(), minlength=P + 2) / T**2
                tv.append(100 * 0.5 * np.abs(h1 - h2).sum())
                irregular.append(100 * np.mean(with_context == P + 1))
                irregular_both.append(100 * np.mean(((with_context == P + 1) & (alone == P + 1))[border]))
            geometry = 100 * (1 - ((T - 2 * R) / T) ** 2)
            print(f"  R={R:2d}  P={P:3d}  circle leaves the tile {geometry:5.1f}%  codes changed {np.mean(changed):4.1f}% "
                  f"({min(changed):.1f}-{max(changed):.1f})  histogram change {np.mean(tv):4.1f}%  "
                  f"irregular {np.mean(irregular):3.0f}%  border pixels irregular both ways {np.mean(irregular_both):3.0f}%")


if __name__ == "__main__":
    main()
