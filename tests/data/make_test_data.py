#!/usr/bin/env python3
"""Generate the small synthetic dataset used by the test profile and the unit tests.

Creates (next to this script):

* he_synthetic.tif - 1792 x 1568 px H&E-like image at 0.25 µm/px (448 x 392 µm):
  glass on the left, stroma (pink, sparse nuclei) and an epithelium-like region
  (dense purple nuclei) on the right, plus a small lumen.
* hd_test/outs/binned_outputs/square_008um/ - Visium HD-like 8 µm bins (Space Ranger layout).
* visium_test/outs/ - standard Visium-like spots (55 µm, 100 µm hexagonal lattice).
* hest_test.h5ad - the Visium spots in HEST-1k's h5ad layout.
* samplesheet_test.csv - a Visium HD sample, a Visium sample and an image-only sample

Everything is deterministic (fixed random seed). Rerun after changing it with:
    python tests/data/make_test_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyvips

HERE = Path(__file__).resolve().parent
MPP = 0.25
W_PX, H_PX = 1792, 1568
N_GENES = 30
rng = np.random.default_rng(42)


def tissue_masks(x_um: np.ndarray, y_um: np.ndarray):
    tissue = x_um > 112 + 15 * np.sin(y_um / 40.0)
    lumen = (x_um - 330) ** 2 + (y_um - 300) ** 2 < 25**2
    epithelium = tissue & (y_um < 180) & (x_um > 250)
    return tissue & ~lumen, epithelium, lumen


def make_image() -> np.ndarray:
    yy, xx = np.mgrid[0:H_PX, 0:W_PX]
    x_um, y_um = (xx + 0.5) * MPP, (yy + 0.5) * MPP
    tissue, epi, lumen = tissue_masks(x_um, y_um)
    img = np.empty((H_PX, W_PX, 3), np.float32)
    img[:] = (243, 243, 245)
    low = 10 * np.sin(xx / 97.0) * np.cos(yy / 131.0)
    stroma = np.array([232, 165, 198], np.float32)
    img[tissue] = stroma + low[tissue, None]
    # nuclei: ellipses, denser in the epithelium-like region
    for region, n, colour, (rmin, rmax) in ((tissue & ~epi, 260, (105, 70, 150), (8, 14)),
                                            (epi, 520, (70, 45, 125), (10, 18))):
        ys, xs = np.nonzero(region)
        pick = rng.choice(len(xs), size=n, replace=False)
        for cx, cy in zip(xs[pick], ys[pick]):
            a, b = rng.uniform(rmin, rmax), rng.uniform(rmin * 0.6, rmax * 0.8)
            th = rng.uniform(0, np.pi)
            y0, y1 = max(cy - rmax, 0), min(cy + rmax + 1, H_PX)
            x0, x1 = max(cx - rmax, 0), min(cx + rmax + 1, W_PX)
            gy, gx = np.mgrid[y0:y1, x0:x1]
            u = (gx - cx) * np.cos(th) + (gy - cy) * np.sin(th)
            v = -(gx - cx) * np.sin(th) + (gy - cy) * np.cos(th)
            ell = (u / a) ** 2 + (v / b) ** 2 <= 1
            ell &= region[y0:y1, x0:x1]
            img[y0:y1, x0:x1][ell] = colour
    img += rng.normal(0, 5, img.shape).astype(np.float32)
    img[~tissue] = np.clip(img[~tissue], 230, 255)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_10x_h5(path: Path, X: np.ndarray, barcodes, gene_ids, gene_names) -> None:
    """X: genes x units dense int array -> 10x CSC HDF5."""
    from scipy import sparse

    M = sparse.csc_matrix(X)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = h5py.string_dtype()
    with h5py.File(path, "w") as h5:
        g = h5.create_group("matrix")
        g.create_dataset("data", data=M.data.astype(np.int32))
        g.create_dataset("indices", data=M.indices.astype(np.int64))
        g.create_dataset("indptr", data=M.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(M.shape, np.int32))
        g.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        f = g.create_group("features")
        f.create_dataset("id", data=np.array(gene_ids, dtype="S"))
        f.create_dataset("name", data=np.array(gene_names, dtype="S"))
        f.create_dataset("feature_type", data=np.array(["Gene Expression"] * len(gene_ids), dtype="S"))
        f.create_dataset("genome", data=np.array(["GRCh38"] * len(gene_ids), dtype="S"))
        f.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))
    _ = s


def expression(epi: np.ndarray, area_scale: float) -> np.ndarray:
    """Genes 0-9 epithelial, 10-19 stromal, 20-29 housekeeping."""
    rates = np.empty((N_GENES, len(epi)))
    rates[:10] = np.where(epi, 3.0, 0.3)
    rates[10:20] = np.where(epi, 0.3, 2.5)
    rates[20:] = 1.0
    return rng.poisson(rates * area_scale).astype(np.int64)


def main() -> None:
    img = make_image()
    vimg = pyvips.Image.new_from_array(img).copy(interpretation="srgb", xres=1000 / MPP, yres=1000 / MPP)
    vimg.tiffsave(str(HERE / "he_synthetic.tif"), tile=True, tile_width=256, tile_height=256,
                  compression="jpeg", Q=90, resunit="cm")
    gene_ids = [f"ENSG{i:011d}" for i in range(N_GENES)]
    gene_names = [f"EPI{i}" for i in range(10)] + [f"STR{i}" for i in range(10)] + [f"HK{i}" for i in range(10)]

    # ---- Visium HD-like: 8 µm bins over x in [64, 448) µm, y in [0, 392) µm -------------
    bin_um = 8.0
    xs = np.arange(64 + bin_um / 2, 448, bin_um)
    ys = np.arange(bin_um / 2, 392, bin_um)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    tissue, epi, _ = tissue_masks(gx, gy)
    rows = np.round((gy - bin_um / 2) / bin_um).astype(int)
    cols = np.round((gx - 64 - bin_um / 2) / bin_um).astype(int)
    barcodes = [f"s_008um_{r:05d}_{c:05d}-1" for r, c in zip(rows, cols)]
    pos = pd.DataFrame({"barcode": barcodes, "in_tissue": tissue.astype(int), "array_row": rows, "array_col": cols,
                        "pxl_row_in_fullres": gy / MPP, "pxl_col_in_fullres": gx / MPP})
    hd = HERE / "hd_test" / "outs" / "binned_outputs" / "square_008um"
    (hd / "spatial").mkdir(parents=True, exist_ok=True)
    pos.to_parquet(hd / "spatial" / "tissue_positions.parquet", index=False)
    (hd / "spatial" / "scalefactors_json.json").write_text(json.dumps(
        {"spot_diameter_fullres": bin_um / MPP, "bin_size_um": bin_um, "microns_per_pixel": MPP,
         "regist_target_img_scalef": 1.0, "tissue_lowres_scalef": 0.1, "tissue_hires_scalef": 0.5,
         "fiducial_diameter_fullres": 100.0}, indent=2))
    sel = tissue
    write_10x_h5(hd / "filtered_feature_bc_matrix.h5", expression(epi[sel], 1.0),
                 np.array(barcodes)[sel], gene_ids, gene_names)

    # ---- standard Visium-like spots ----------------------------------------------------------
    # Visium geometry: hexagonal lattice with 100 µm between neighbouring spot centres; along a
    # row array_col advances by 2 per spot and odd rows are shifted by half a spot (+1).
    row_um = 100.0 * np.sqrt(3) / 2
    spots = []
    for i, y in enumerate([80.0, 80.0 + row_um, 80.0 + 2 * row_um]):
        for x in ([120.0, 220.0, 320.0] if i % 2 == 0 else [170.0, 270.0, 370.0]):
            spots.append((x, y, i, int(round((x - 120.0) / 50.0))))
    sx = np.array([s[0] for s in spots]); sy = np.array([s[1] for s in spots])
    s_tissue, s_epi, _ = tissue_masks(sx, sy)
    s_barcodes = [f"AAAC{i:04d}GGTT-1" for i in range(len(spots))]
    spos = pd.DataFrame({"barcode": s_barcodes, "in_tissue": s_tissue.astype(int),
                         "array_row": [s[2] for s in spots], "array_col": [s[3] for s in spots],
                         "pxl_row_in_fullres": sy / MPP, "pxl_col_in_fullres": sx / MPP})
    vis = HERE / "visium_test" / "outs"
    (vis / "spatial").mkdir(parents=True, exist_ok=True)
    spos.to_csv(vis / "spatial" / "tissue_positions.csv", index=False)
    # Space Ranger's spot_diameter_fullres is the diameter of a theoretical 65 µm spot in pixels
    (vis / "spatial" / "scalefactors_json.json").write_text(json.dumps(
        {"spot_diameter_fullres": 65.0 / MPP, "fiducial_diameter_fullres": 100.0 / MPP,
         "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1}, indent=2))
    X_spots = expression(s_epi[s_tissue], 40.0)
    write_10x_h5(vis / "filtered_feature_bc_matrix.h5", X_spots, np.array(s_barcodes)[s_tissue], gene_ids, gene_names)

    # ---- the same spots in HEST's h5ad layout ------------------------------------------------
    import anndata as ad
    from scipy import sparse

    obs = spos[spos["in_tissue"] == 1].set_index("barcode")
    adata = ad.AnnData(X=sparse.csr_matrix(X_spots.T.astype(np.float32)), obs=obs,
                       var=pd.DataFrame({"gene_ids": gene_ids}, index=gene_names))
    adata.obsm["spatial"] = obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    adata.write_h5ad(HERE / "hest_test.h5ad")

    (HERE / "samplesheet_test.csv").write_text(
        "sample,image,spatial,spatial_format,tiling,mpp,patient,condition,cohort\n"
        "test_hd,he_synthetic.tif,hd_test/outs,spaceranger_hd,,,P1,CRC,A\n"
        "test_visium,he_synthetic.tif,visium_test/outs,spaceranger,,0.25,P2,CRC,B\n"
        "test_image_only,he_synthetic.tif,,,,,P3,none,C\n"
    )
    print("test data written to", HERE)


if __name__ == "__main__":
    main()
