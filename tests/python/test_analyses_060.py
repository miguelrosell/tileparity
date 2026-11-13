"""tileparity 0.6.0 analyses: corrected ARI and the k shown, F1 per domain, stability,
representative tiles, the interval and null of R2, relevance (expression and genes), the
patch sweep's geometry, the cohort summary and the HEST cohort table."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tileparity_lib import cluster, cohort, patchsweep, redundancy, relevance

pytest.importorskip("igraph")
pytest.importorskip("umap")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


# ---- agreement: corrected ARI and the k shown -------------------------------------------------------------
def test_corrected_ari_is_the_share_of_the_way_from_the_null_to_one():
    assert cluster.corrected_ari(0.51, 0.11) == pytest.approx(0.4494, abs=1e-4)
    assert cluster.corrected_ari(0.51, 0.45) == pytest.approx(0.1091, abs=1e-4)
    assert np.isnan(cluster.corrected_ari(0.5, np.nan))


def _rows(scores: dict, method="kmeans", ref="EXPRESSION"):
    rows = []
    for k, per_view in scores.items():
        for v, a in per_view.items():
            rows.append({"set": "keep", "method": method, "k": k, "view_a": v, "view_b": ref, "n_tiles": 10, "k_a": k,
                         "k_b": k, "ari": a, "ari_corrected": a - 0.1, "null_ari_mean": 0.1})
    return pd.DataFrame(rows)


def test_k_shown_is_the_best_mean_agreement_with_the_reference_and_ties_go_to_the_smaller_k():
    rows = _rows({3: {"A": 0.5, "B": 0.3}, 4: {"A": 0.7, "B": 0.5}, 8: {"A": 0.9, "B": 0.1}})
    choice = cluster.choose_k(rows, "kmeans", ["A", "B", "EXPRESSION"], "EXPRESSION", [3, 4, 8])
    assert choice["k"] == 4 and choice["basis"] == "corrected"
    tie = _rows({3: {"A": 0.6}, 4: {"A": 0.6}})
    assert cluster.choose_k(tie, "kmeans", ["A", "EXPRESSION"], "EXPRESSION", [3, 4])["k"] == 3


def test_f1_of_every_domain_matches_one_to_one():
    ref = np.array([0] * 10 + [1] * 6)
    lab = np.array([5] * 8 + [7] * 2 + [7] * 4 + [5] * 2)  # domain 0 -> cluster 5 (8 of 10), domain 1 -> cluster 7
    d = cluster.domain_f1(ref, lab).set_index("domain")
    # cluster 5 has 10 tiles, 8 in domain 0: precision 0.8, recall 0.8
    assert d.loc[0, "matched_cluster"] == 5 and d.loc[0, "f1"] == pytest.approx(0.8)
    assert d.loc[1, "f1"] == pytest.approx(2 * 4 / (6 + 6))
    assert cluster.domain_f1(ref, ref)["f1"].eq(1.0).all()


def test_run_shows_the_best_k_and_writes_f1_stability_and_representatives(tmp_path):
    from test_cluster import _write_slide

    from tileparity_lib import export

    _write_slide(tmp_path, H=12, W=16)
    sample = export.load_sample(tmp_path, "S_mpp0.5000")
    res = cluster.run(sample, ks=(2, 4, 6), sets=("keep",), null_shifts=10)
    info = res["sets"]["keep"]
    assert set(info["k_show"]) == {"leiden", "kmeans"} and info["k_show_rule"].startswith("auto")
    assert info["k_show"]["kmeans"] == 4  # the slide has 4 regions that every view sees
    assert "ari_corrected" in res["agreement"]
    summ = cluster.summary_table(res["agreement"], res["sets"])
    assert set(summ.columns) == set(cluster.SUMMARY_COLUMNS) and summ["ari_mean_k"].between(-1, 1).all()
    f1 = cluster.f1_table(res)
    assert len(f1) and f1["f1"].between(0, 1).all()
    macro = cluster.macro_f1(f1)
    assert (macro.loc[macro["k"] == 4, "macro_f1"] > 0.8).all()
    st = cluster.stability(res, "keep", ["kmeans"], reps=2, fraction=0.9)
    assert set(st["rep"]) == {1, 2} and st["ari_with_full"].dropna().between(-1, 1).all()
    reps = cluster.representatives(sample, res, "keep", n=3)
    clusters = reps[(reps["kind"] == "cluster") & (reps["view"] == "UNI2H_RGB") & (reps["method"] == "kmeans")]
    assert clusters["group"].nunique() == 4 and clusters.groupby("group").size().max() <= 3
    assert set(reps["kind"]) == {"cluster", "pc_high", "pc_low"}


# ---- R2: interval and spatial null ------------------------------------------------------------------------
def _grid_tiles(H=20, W=20):
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    T = 224
    return pd.DataFrame({"tile_id": [f"t{i}" for i in range(H * W)], "row": r.ravel(), "col": c.ravel(),
                         "x0_px": c.ravel() * T, "y0_px": r.ravel() * T, "tile_px": T})


def test_r2_interval_and_null():
    rng = np.random.default_rng(0)
    tiles = _grid_tiles()
    n = len(tiles)
    # a smooth map: the predictor describes the target, moved maps do not
    base = np.sin(tiles["row"].to_numpy() / 3.0)[:, None] * rng.normal(size=(1, 5)) + \
        np.cos(tiles["col"].to_numpy() / 4.0)[:, None] * rng.normal(size=(1, 5))
    A = base + rng.normal(0, 0.3, (n, 5))
    B = base @ rng.normal(size=(5, 4)) + rng.normal(0, 0.3, (n, 4))
    out = redundancy.run({"A": A, "B": B}, {"A": 1.0, "B": 1.0}, tiles, ["A", "B"], "keep", 0.5,
                         block_um=600, null_shifts=10, n_boot=50)
    row = out["table"].set_index(["predictor", "target"]).loc[("A", "B")]
    assert row["r2"] > 0.8 and row["r2_ci_low"] <= row["r2"] <= row["r2_ci_high"]
    assert row["null_moves"] == 10 and row["r2_null_mean"] < row["r2"] - 0.3 and row["p_null"] == pytest.approx(1 / 11)


# ---- relevance --------------------------------------------------------------------------------------------
def test_relevance_partitions_the_expression_between_two_views():
    rng = np.random.default_rng(1)
    tiles = _grid_tiles(16, 16)
    n = len(tiles)
    a, b = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    E = np.hstack([a[:, :2], b[:, :2]]) + rng.normal(0, 0.1, (n, 4))  # half from A, half from B
    G = np.hstack([a[:, :1] * 2, b[:, :1], rng.normal(size=(n, 1))])  # gene 3 is noise
    scores = {"EXPRESSION": E, "UNI2H_RGB": a, "LBP_RGB": b, "INTENSITY_HE": rng.normal(size=(n, 3))}
    out = relevance.run(scores, {"EXPRESSION": 1.0}, tiles, ["UNI2H_RGB", "LBP_RGB", "INTENSITY_HE"], "keep", 0.5,
                        genes=(G, np.array(["g1", "g2", "g3"]), np.array(["G1", "G2", "G3"])),
                        block_um=500, null_shifts=5, n_boot=20, gene_null_shifts=5)
    part = out["partition"].set_index(["view_a", "view_b"]).loc[("UNI2H_RGB", "LBP_RGB")]
    assert part["unique_a"] > 0.3 and part["unique_b"] > 0.3 and abs(part["shared"]) < 0.1
    assert part["unique_a"] + part["unique_b"] + part["shared"] == pytest.approx(part["r2_ab"])
    g = out["genes"].set_index("gene_id")
    assert g.loc["g1", "r2_UNI2H_RGB"] > 0.9 and g.loc["g2", "r2_LBP_RGB"] > 0.9 and g.loc["g3", "r2_LBP_RGB"] < 0.1
    assert g.loc["g2", "adds_LBP_RGB_to_UNI2H_RGB"] > 0.8
    assert relevance._bh(np.array([0.01, 0.04, 0.03, np.nan]))[:3] == pytest.approx([0.03, 0.04, 0.04])


# ---- patch sweep ------------------------------------------------------------------------------------------
def test_sizes_patches_and_cells():
    assert patchsweep.parse_sizes("16:64:16", 8) == [16, 32, 48, 64]
    assert patchsweep.parse_sizes("12,16,20,8", 8) == [16]  # multiples of the cell, at least two cells
    cr, cc = np.meshgrid(np.arange(5), np.arange(4), indexing="ij")
    cr, cc = cr.ravel(), cc.ravel()
    M, pr, pc = patchsweep.patches(cr, cc, 2)
    assert M.shape == (4, 20) and np.all(np.asarray(M.sum(axis=1)).ravel() == 4)  # row 4 is incomplete
    # a molecule at pixel (x=20, y=35) with 16 px cells goes to cell (row 2, col 1)
    Y = patchsweep.cell_counts(np.array([20.0, 500.0]), np.array([35.0, 5.0]), sparse.csc_matrix(np.array([[3, 7]])),
                               cr, cc, (0, 0), 16)
    idx = int(np.flatnonzero((cr == 2) & (cc == 1))[0])
    assert Y[idx, 0] == 3 and Y.sum() == 3  # the second unit is outside the cells


def test_sweet_spot_is_the_smallest_size_within_one_standard_error():
    t = pd.DataFrame({"size_um": [16, 32, 48, 64], "fraction": [0.5, 0.78, 0.8, 0.7], "fraction_se": [0.05] * 4})
    spot = patchsweep.sweet_spot(t)
    assert spot["best_size_um"] == 48 and spot["sweet_spot_um"] == 32


# ---- cohort -----------------------------------------------------------------------------------------------
def test_cohort_tables_and_paired_comparison():
    slides = []
    for i, (s, lesion) in enumerate([("A1", "les"), ("A2", "non"), ("B1", "les"), ("B2", "non"), ("C1", "les"),
                                     ("C2", "non")]):
        r2 = 0.6 if lesion == "les" else 0.4
        slides.append({"prefix": f"{s}_mpp0.5000", "info": {"sample_id": s, "sets": {"keep": {
            "reference_view": "EXPRESSION", "views_clustered": ["UNI2H_RGB", "EXPRESSION"]}}},
            "relevance": pd.DataFrame({"set": ["keep"], "predictor": ["UNI2H_RGB"], "r2": [r2 + 0.01 * i],
                                       "r2_ci_low": [0.1], "r2_ci_high": [0.9], "r2_null_p95": [0.05]}),
            "summary": None, "f1": None, "stability": None, "agreement": None, "redundancy": None, "partition": None})
    key = cohort.key_table(slides)
    assert list(key["view"].unique()) == ["UNI2H_RGB"] and len(key) == 6
    long = cohort.long_table(slides)
    assert set(long["analysis"]) == {"relevance"}
    meta = pd.DataFrame({"sample": ["A1", "A2", "B1", "B2", "C1", "C2"], "lesion": ["les", "non"] * 3,
                         "patient": ["A", "A", "B", "B", "C", "C"]})
    comp = cohort.compare_groups(key, meta, "lesion", "patient")
    row = comp[comp["metric"] == "relevance_r2"].iloc[0]
    assert row["test"] == "paired Wilcoxon signed-rank" and row["n_pairs"] == 3
    assert row["median_paired_difference"] == pytest.approx(-0.19)


def test_hest_cohort_table_leaves_out_two_patient_and_coarse_sections(tmp_path):
    import hest_download

    t = tmp_path / "c.tsv"
    t.write_text("id\tpatient\tcondition\tsingle_biopsy\tmpp\n"
                 "X1\tP1\tpso_lesional\tTrue\t0.52\nX2\tP2+P3\tpso_lesional\tFalse\t0.52\nX3\tP4\tprp\tTrue\t1.05\n")
    ids, meta = hest_download.read_table(str(t), 0.6)
    assert ids == ["X1"] and meta["X1"]["patient"] == "P1"
    real = Path(__file__).resolve().parents[2] / "assets" / "cohort_c_skin_hest.tsv"
    ids, meta = hest_download.read_table(str(real), 0.6)
    assert len(ids) == 50 and len({m["patient"] for m in meta.values()}) == 13
