"""tileparity 0.7.0 analyses: the per-gene ridge and its null, the split-half ceiling, the
gene partition, domain markers, texture features against genes (with the modified t-test),
count thinning, the cohort tables, and the clustering step end to end with every option."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tileparity_lib import cluster, cohort, lbp, markers, redundancy, relevance, texture_genes

pytest.importorskip("igraph")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _grid_tiles(H=20, W=20):
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    T = 224
    return pd.DataFrame({"tile_id": [f"t{i}" for i in range(H * W)], "row": r.ravel(), "col": c.ravel(),
                         "x0_px": c.ravel() * T, "y0_px": r.ravel() * T, "tile_px": T})


def _smooth(rng, H, W, sigma, k=1):
    from scipy.ndimage import gaussian_filter

    out = []
    for _ in range(k):
        z = gaussian_filter(rng.normal(size=(H, W)), sigma)
        out.append(((z - z.mean()) / z.std()).ravel())
    return np.column_stack(out)


# ---- relevance: per-gene ridge, null, ceiling ------------------------------------------------------------
def test_gcv_ridge_matches_leave_one_out_closely_and_picks_a_penalty_per_gene():
    from sklearn.linear_model import RidgeCV

    rng = np.random.default_rng(0)
    n, p, g = 300, 8, 20
    X = rng.normal(size=(n, p))
    Y = X @ (rng.normal(size=(p, g)) * (rng.random(g) < 0.5)) + rng.normal(size=(n, g))
    folds = rng.integers(0, 4, n)
    mine = relevance.ridge_gcv_predict(X, Y, folds)
    ref = np.zeros_like(Y)
    for f in range(4):
        te = folds == f
        ref[te] = RidgeCV(alphas=redundancy.ALPHAS, alpha_per_target=True).fit(X[~te], Y[~te]).predict(X[te])
    r_mine, r_ref = redundancy.r2_scores(Y, mine)[1], redundancy.r2_scores(Y, ref)[1]
    assert np.abs(r_mine - r_ref).max() < 0.01


def test_empirical_p_and_its_floor():
    real = np.array([0.5, 0.1, -0.1])
    null = np.array([[0.0, 0.2, 0.0], [0.1, 0.0, 0.0], [0.2, 0.3, 0.1]])
    assert relevance.empirical_p(real, null) == pytest.approx([1 / 4, 3 / 4, 1.0])


def test_per_gene_null_is_calibrated_for_noise_genes():
    """Genes of pure (spatially smooth) noise must not beat the null more often than chance."""
    rng = np.random.default_rng(3)
    H, W = 24, 24
    tiles = _grid_tiles(H, W)
    X = _smooth(rng, H, W, 2.0, 6)
    G = _smooth(rng, H, W, 2.0, 60)  # independent of X
    folds = redundancy.spatial_folds(*redundancy.tile_centres_um(tiles, 0.5)[:2], 1000.0, 5, 0)
    moves = cluster.torus_pairs(tiles["row"], tiles["col"], 39, 0)
    r2 = redundancy.r2_scores(G, relevance.ridge_gcv_predict(X, G, folds))[1]
    p = relevance.empirical_p(r2, relevance.gene_null(X, G, folds, moves))
    assert (p < 0.05).mean() <= 0.15


def test_relevance_gives_ceiling_pcs_reliability_and_gene_partition():
    rng = np.random.default_rng(1)
    H, W = 16, 16
    tiles = _grid_tiles(H, W)
    n = len(tiles)
    a = _smooth(rng, H, W, 2.0, 3)
    b = _smooth(rng, H, W, 2.0, 3)
    # counts driven by a (genes 0-19) and b (genes 20-39), plus noise genes
    rate = np.exp(np.hstack([np.repeat(a[:, :1], 20, 1), np.repeat(b[:, :1], 20, 1), np.zeros((n, 20))]) * 0.8 + 2)
    Y = sparse.csr_matrix(rng.poisson(rate))
    Z, idx = relevance.expression_genes(Y.T, 60)
    E = cluster.pca(Z, 10, 0)[0]
    scores = {"EXPRESSION": E, "UNI2H_RGB": a, "LBP_RGB": b, "INTENSITY_HE": rng.normal(size=(n, 3))}
    out = relevance.run(scores, {"EXPRESSION": 1.0}, tiles, ["UNI2H_RGB", "LBP_RGB", "INTENSITY_HE"], "keep", 0.5,
                        genes=(Z, np.array([f"g{i}" for i in idx]), np.array([f"G{i}" for i in idx])),
                        block_um=500, null_shifts=5, n_boot=10, gene_null_shifts=19, counts=Y, gene_index=idx, n_hvg=60)
    t = out["table"].set_index("predictor")
    assert {"r2_pc1", "r2_pc5", "r2_half", "r2_ceiling", "fraction"} <= set(t.columns)
    assert 0 < t.loc["UNI2H_RGB", "r2_ceiling"] <= 1 and t.loc["UNI2H_RGB", "fraction"] > 0.2
    assert t.loc["INTENSITY_HE", "fraction"] < 0.2
    g = out["genes"].set_index("gene_id")
    assert {"reliability", "p_UNI2H_RGB", "q_UNI2H_RGB", "r2_null_p95_LBP_RGB"} <= set(g.columns)
    assert g.filter(like="g0", axis=0)["reliability"].iloc[0] > 0.5
    assert g["p_UNI2H_RGB"].min() == pytest.approx(1 / 20)
    gp = out["gene_partition"]
    pair = gp[(gp["view_a"] == "UNI2H_RGB") & (gp["view_b"] == "LBP_RGB")].set_index("gene_id")
    assert pair["only_a"].filter(like="g0", axis=0).iloc[0] > 0.2  # gene of a: only UNI2-h predicts it
    assert np.allclose(pair["only_a"] + pair["only_b"] + pair["shared"], pair["r2_ab"])
    cat = relevance.gene_category(np.array([0.01, 0.01, 0.5, 0.5]), np.array([0.01, 0.5, 0.01, 0.5]))
    assert list(cat) == ["both", "only_x", "only_y", "neither"]


# ---- markers ----------------------------------------------------------------------------------------------
def test_markers_rank_the_gene_of_each_domain_first():
    rng = np.random.default_rng(2)
    lab = np.repeat([0, 1, 2], 40)
    counts = rng.poisson(1.0, (120, 30))
    for d in range(3):
        counts[lab == d, d] += rng.poisson(20, (lab == d).sum())  # gene d marks domain d
    t = markers.domain_markers(sparse.csr_matrix(counts), [f"g{i}" for i in range(30)], [f"G{i}" for i in range(30)],
                               {("kmeans", 3): lab}, n_top=5)
    first = t[t["rank"] == 1].set_index("domain")["gene_id"]
    assert first.to_dict() == {0: "g0", 1: "g1", 2: "g2"}
    assert (t["auroc"].between(0, 1)).all() and (t[t["rank"] == 1]["auroc"] > 0.95).all()
    f1 = pd.DataFrame({"set": "keep", "method": "kmeans", "k": 3, "view": "LBP_RGB", "domain": [0, 1, 2],
                       "f1": [0.9, 0.1, 0.5]})
    dom = markers.with_f1(t, f1)
    assert dom.set_index("domain")["f1_LBP_RGB"].to_dict() == {0: 0.9, 1: 0.1, 2: 0.5}


# ---- texture features against genes ---------------------------------------------------------------------------
def test_texture_features_are_fractions_of_four_types_per_radius():
    radii = [1, 2]
    spec = lbp.LbpSpec(radii=radii, npoints=[lbp.npoints_for(r) for r in radii], channels=["R", "G"], tile_px=4)
    cols = spec.columns("R") + spec.columns("G")
    X = np.ones((3, len(cols)))
    F, names, table = texture_genes.features(X, cols, "mean")
    assert names == ["spot_bright_r1", "spot_bright_r2", "spot_dark_or_flat_r1", "spot_dark_or_flat_r2", "edge_r1",
                     "edge_r2", "irregular_r1", "irregular_r2"]
    P1 = lbp.npoints_for(1)
    assert F[0, names.index("edge_r1")] == pytest.approx((P1 - 1) / (P1 + 2))
    assert np.allclose(F.reshape(3, 4, 2).sum(axis=1), 1.0)
    F_r, _, _ = texture_genes.features(X, cols, "R")
    assert np.allclose(F_r, F)
    with pytest.raises(ValueError):
        texture_genes.features(X, cols, "B")


def test_modified_t_test_holds_its_level_where_the_classical_test_does_not():
    from scipy.stats import pearsonr

    rng = np.random.default_rng(5)
    H, W, reps = 30, 30, 120
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    X, Y = _smooth(rng, H, W, 3.0, reps), _smooth(rng, H, W, 3.0, reps)
    cx, pairs = texture_genes.correlograms(X, r.ravel(), c.ravel())
    cy, _ = texture_genes.correlograms(Y, r.ravel(), c.ravel())
    n = H * W
    naive = modified = 0
    for i in range(reps):
        rho = np.corrcoef(X[:, i], Y[:, i])[0, 1]
        naive += pearsonr(X[:, i], Y[:, i])[1] < 0.05
        ne = texture_genes.effective_n(cx[i:i + 1], cy[i:i + 1], pairs, n)[0, 0]
        from scipy.stats import t as tdist

        modified += 2 * tdist.sf(abs(rho) * math.sqrt((ne - 2) / (1 - rho**2)), ne - 2) < 0.05
    assert naive / reps > 0.5 and modified / reps < 0.12
    assert pairs[0] == n and cx[:, 0] == pytest.approx(1.0)


def test_texture_genes_finds_the_linked_gene_and_removes_depth():
    rng = np.random.default_rng(6)
    H, W = 30, 30
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    field = _smooth(rng, H, W, 2.0)[:, 0]
    depth = _smooth(rng, H, W, 2.0)[:, 0]
    F = np.column_stack([field + rng.normal(0, 0.3, H * W), depth + rng.normal(0, 0.3, H * W)])
    G = np.column_stack([np.exp(field + rng.normal(0, 0.3, H * W)), np.exp(depth), rng.normal(size=H * W)])
    ft = pd.DataFrame({"feature": ["f_linked", "f_depth"], "type": ["edge", "edge"], "radius_px": [1, 2]})
    t = texture_genes.run(F, ["f_linked", "f_depth"], ft, G, ["a", "b", "c"], ["A", "B", "C"], depth, r.ravel(),
                          c.ravel())
    k = t.set_index(["feature", "gene_id"])
    assert k.loc[("f_linked", "a"), "q"] < 0.01 and k.loc[("f_linked", "a"), "rho"] > 0.5
    assert abs(k.loc[("f_depth", "b"), "rho"]) < 0.2  # both follow depth only: nothing left once depth is removed
    assert k.loc[("f_linked", "c"), "q"] > 0.05
    s = texture_genes.summary(t)
    assert s.set_index("feature").loc["f_linked", "top_gene"] == "A"


# ---- thinning ---------------------------------------------------------------------------------------------------
def test_thinning_brings_the_median_depth_to_the_target():
    rng = np.random.default_rng(7)
    Y = sparse.csc_matrix(rng.poisson(10, (50, 200)))  # genes x tiles, ~500 molecules per tile
    sample = {"counts": Y, "tiles": pd.DataFrame({"keep": np.ones(200, bool)})}
    info = cluster.thin_counts(sample, 100, 0)
    assert info["applied"] and info["median_after"] == pytest.approx(100, rel=0.1)
    assert (sample["counts"] - Y).max() <= 0
    assert not cluster.thin_counts(sample, 10_000, 0)["applied"]


# ---- cohort tables -----------------------------------------------------------------------------------------------
def test_cohort_ari_by_k_picks_the_best_k_by_the_median_corrected_ari():
    rows = []
    for s, shift in (("A", 0.0), ("B", 0.1), ("C", 0.05)):
        for k, v in ((2, 0.2), (4, 0.6), (6, 0.4)):
            for metric, val in (("ari", v + 0.2), ("ari_corrected", v + shift)):
                rows.append({"sample": s, "analysis": "agreement", "set": "keep", "method": "kmeans", "k": k,
                             "a": "LBP_RGB", "b": "EXPRESSION", "metric": metric, "value": val})
    t = cohort.ari_by_k(pd.DataFrame(rows), "EXPRESSION")
    assert t.loc[t["best_k"], "k"].tolist() == [4]
    assert t.set_index("k").loc[4, "ari_corrected_median"] == pytest.approx(0.65)


def test_texture_replication_counts_slides_with_the_same_sign():
    def slide(name, rho, q):
        return {"prefix": f"{name}_mpp0.5000", "info": {"sample_id": name},
                "texture": pd.DataFrame({"feature": ["edge_r1"], "type": ["edge"], "radius_px": [1], "gene_id": ["g"],
                                         "gene_name": ["G"], "rho": [rho], "n_tiles": [100], "n_eff": [30.0], "p": [q],
                                         "q": [q]})}
    t = cohort.texture_replication([slide("A", 0.5, 0.01), slide("B", 0.4, 0.02), slide("C", -0.3, 0.01)])
    r = t.iloc[0]
    assert r["slides"] == 3 and r["slides_sig_pos"] == 2 and r["slides_sig_neg"] == 1 and r["slides_sig_same_sign"] == 2


# ---- the clustering step and the cohort, end to end -----------------------------------------------------------
def test_cluster_step_writes_every_new_output_and_the_cohort_gathers_them(tmp_path, monkeypatch):
    from test_cluster import _write_slide

    import tp_cluster
    import tp_cohort

    monkeypatch.chdir(tmp_path)
    for i, s in enumerate(("A", "B")):
        _write_slide(tmp_path, prefix=f"{s}_mpp0.5000", H=14, W=16, genes=80, seed=i)
        tp_cluster.main(["--input-dir", ".", "--prefix", f"{s}_mpp0.5000", "--k", "2,4", "--null-shifts", "5",
                         "--stability-reps", "1", "--relevance", "--relevance-gene-null-shifts", "9", "--markers",
                         "--texture-genes", "--texture-genes-n", "20", "--expression-thin-to", "300"])
    for suffix in ("relevance_gene_partition.tsv.gz", "relevance_gene_partition_keep.png", "markers.tsv.gz",
                   "markers_domains.tsv", "cluster_markers_keep.png", "texture_genes.tsv.gz",
                   "texture_genes_summary.tsv", "texture_genes_keep.png", "relevance_genes_keep.png"):
        assert (tmp_path / f"A_mpp0.5000.{suffix}").exists(), suffix
    info = json.loads((tmp_path / "A_mpp0.5000.cluster.json").read_text())
    assert info["expression_thinning"]["applied"] and info["texture_genes"]["status"] == "computed"
    assert info["relevance"]["genes"]["null_moves"] == 9
    tp_cohort.main(["--input-dir", ".", "--prefix", "cohort_mpp0.5000"])
    for suffix in ("ari_by_k.tsv", "ari_by_k_corrected.png", "stability.png", "domains.tsv", "gene_partition.tsv",
                   "texture_genes.tsv.gz", "report.html"):
        assert (tmp_path / f"cohort_mpp0.5000.{suffix}").exists(), suffix
    dom = pd.read_csv(tmp_path / "cohort_mpp0.5000.domains.tsv", sep="\t")
    assert set(dom["sample"]) == {"A", "B"} and dom["markers"].str.len().gt(0).all()
