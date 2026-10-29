"""The optional clustering step (--cluster): agreement measures, Scanpy's recipe, Leiden at a
given k, the spatial null, the tile sets, the command line and the exported labels."""

import gzip
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from conftest import TEST_DATA
from tileparity_lib import cluster, export, features, lineage, st

pytest.importorskip("igraph")
pytest.importorskip("umap")


# ---- agreement -----------------------------------------------------------------------------------------
def test_ari_matches_scikit_learn():
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(1, 300))
        a = rng.integers(0, int(rng.integers(1, 10)), n)
        b = rng.integers(0, int(rng.integers(1, 10)), n)
        if trial % 4 == 0:
            b = a.copy()
        if trial % 9 == 0:
            b = np.arange(n)
        assert cluster.ari(a, b) == pytest.approx(adjusted_rand_score(a, b), abs=1e-12)
    assert cluster.ari(np.array(["x", "y", "y"]), np.array([5, 7, 7])) == 1.0


def test_clusters_are_numbered_by_size():
    assert cluster.relabel_by_size(np.array([5, 5, 1, 1, 1, 9])).tolist() == [1, 1, 0, 0, 0, 2]


def test_matching_and_contingency():
    ref = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    lab = np.array([2, 2, 2, 0, 0, 1, 1, 3])
    mapping = cluster.match_to_reference(ref, lab)
    assert mapping[2] == 0 and mapping[0] == 1 and mapping[1] == 2 and mapping[3] == 3  # 3 has no partner
    assert cluster.contingency(lab, ref).sum() == len(ref)


# ---- Scanpy's recipe ---------------------------------------------------------------------------------
def test_expression_recipe_and_graph_reproduce_scanpy():
    ref = np.load(TEST_DATA / "scanpy_reference.npz")
    Y = sparse.csc_matrix(ref["counts"].T.astype(float))  # genes x tiles
    Z, detected = cluster.expression_features(Y, 50)
    totals = ref["counts"].sum(axis=1, keepdims=True)
    logN = sparse.csr_matrix(np.log1p(ref["counts"] / np.maximum(totals, 1) * 1e4))
    assert cluster.highly_variable_genes(logN, 50).tolist() == ref["hvg"].tolist()
    np.testing.assert_allclose(Z, ref["scaled"], atol=1e-9)
    assert detected == int((ref["counts"].sum(axis=0) > 0).sum())

    n = len(ref["pcs"])
    graph = cluster.knn_graph(ref["pcs"], 10, 0)
    rows, cols = zip(*graph.get_edgelist())
    mine = sparse.coo_matrix((graph.es["weight"], (rows, cols)), shape=(n, n)).tocsr()
    scanpy = sparse.triu(sparse.csr_matrix((ref["conn_data"], ref["conn_indices"], ref["conn_indptr"]),
                                           shape=(n, n)), k=1).tocsr()
    assert abs(mine - scanpy).max() < 1e-6
    assert cluster.ari(cluster.leiden(graph, 1.0, 0), ref["leiden"]) == 1.0


# ---- Leiden at a given number of clusters --------------------------------------------------------------
def _blobs(n_per=60, centres=6, spread=1.0, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 3, (centres, 8))
    truth = np.repeat(np.arange(centres), n_per)
    return C[truth] + rng.normal(0, spread, (len(truth), 8)), truth


def test_leiden_resolution_is_searched_until_k_clusters():
    X, truth = _blobs(spread=1.5)
    graph = cluster.knn_graph(X, 15, 0)
    cache = {}
    for k in (2, 4, 6, 9):
        labels, resolution, reached = cluster.leiden_k(graph, k, 0, cache=cache)
        assert reached == k == cluster.n_clusters(labels)
        assert resolution > 0
    labels, _, _ = cluster.leiden_k(graph, 6, 0, cache=cache)
    assert cluster.ari(labels, truth) > 0.8  # the 6 groups overlap a little
    again, _, _ = cluster.leiden_k(cluster.knn_graph(X, 15, 0), 6, 0)
    assert np.array_equal(labels, again)  # same seed, same partition


def test_leiden_reports_when_k_cannot_be_reached():
    X, _ = _blobs(n_per=30, centres=5, spread=0.05)  # 5 separate groups: the graph has 5 components
    labels, _, reached = cluster.leiden_k(cluster.knn_graph(X, 10, 0), 2, 0)
    assert reached == cluster.n_clusters(labels) == 5


def test_kmeans_recovers_separated_groups():
    X, truth = _blobs(spread=0.8)
    assert cluster.ari(cluster.kmeans(X, 6, 0), truth) > 0.95


# ---- spatial null --------------------------------------------------------------------------------------
def _patchy_map(H=30, W=40, seed=0):
    rng = np.random.default_rng(seed)
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    centres = np.column_stack([rng.uniform(0, H, 6), rng.uniform(0, W, 6)])
    d = (r.ravel()[:, None] - centres[:, 0]) ** 2 + (c.ravel()[:, None] - centres[:, 1]) ** 2
    return r.ravel(), c.ravel(), np.argmin(d, axis=1)


def test_torus_moves_follow_the_rules():
    rows, cols, _ = _patchy_map()
    pairs = cluster.torus_pairs(rows, cols, 50, 0)
    assert len(pairs) == 50
    for i, j in pairs:
        assert len(i) == len(j) >= 0.3 * len(rows)
        dy = (rows[i] - rows[j]) % 30
        dx = (cols[i] - cols[j]) % 40
        assert len(set(zip(dy.tolist(), dx.tolist()))) == 1  # one move per item
        dy, dx = int(dy[0]), int(dx[0])
        assert np.hypot(min(dy, 30 - dy) / 30, min(dx, 40 - dx) / 40) >= 0.2


def test_null_is_chance_agreement_of_patchy_maps():
    rows, cols, a = _patchy_map(seed=0)
    _, _, b = _patchy_map(seed=1)  # another patchy map
    pairs = cluster.torus_pairs(rows, cols, 100, 0)
    rng = np.random.default_rng(3)
    # random labels agree by about 0 whatever their alignment ...
    assert abs(cluster.null_ari(rng.integers(0, 6, len(a)), rng.integers(0, 6, len(a)), pairs).mean()) < 0.02
    # ... but patchy maps agree partly by chance: what the null measures
    assert cluster.null_ari(a, b, pairs).mean() > 0.1
    noisy = a.copy()
    flip = rng.uniform(size=len(a)) < 0.2
    noisy[flip] = rng.integers(0, 6, int(flip.sum()))
    null = cluster.null_ari(a, noisy, pairs)  # moved, the map no longer matches its noisy copy
    assert cluster.ari(a, noisy) > 0.5 > np.percentile(null, 99)
    assert cluster.torus_pairs(np.array([0]), np.array([0]), 10, 0) == []


# ---- one synthetic slide ---------------------------------------------------------------------------------
def _write_slide(d, prefix="S_mpp0.5000", H=12, W=16, seed=0, genes=60, image_only=False):
    """A slide with 4 regions in the pipeline's file layout; every view sees the regions."""
    rng = np.random.default_rng(seed)
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    r, c = r.ravel(), c.ravel()
    # regions around 4 centres (not periodic, so that moved maps do not line up by chance)
    centres = np.array([(0.2, 0.15), (0.25, 0.8), (0.8, 0.3), (0.85, 0.85)]) * (H, W)
    region = np.argmin((r[:, None] - centres[:, 0]) ** 2 + (c[:, None] - centres[:, 1]) ** 2, axis=1)
    n, T = len(r), 224
    sample = prefix.split("_mpp")[0]
    ids = [f"{sample}_mpp0.5000_r{a:04d}_c{b:04d}" for a, b in zip(r, c)]
    tissue = np.where(rng.uniform(size=n) < 0.15, 0.4, 0.9)
    tiles = pd.DataFrame({
        "tile_id": ids, "sample_id": sample, "scale_mpp": 0.5, "tile_px": T, "tile_um": 112.0, "unit_type": "grid",
        "row": r, "col": c, "barcode": "", "x0_px": c * T, "y0_px": r * T, "cx_px": c * T + 112, "cy_px": r * T + 112,
        "cx_fullres": c * T + 112, "cy_fullres": r * T + 112, "cx_um": (c * T + 112) / 2, "cy_um": (r * T + 112) / 2,
        "inside_image": True, "tissue_fraction": tissue, "st_coverage": 1.0, "n_st_units": 196,
        "st_units_equiv": 196.0, "lapvar": 100.0, "pen_fraction": 0.0, "dark_fraction": 0.0, "blur_flag": False,
        "pen_flag": False, "dark_flag": False, "extract": True, "keep": tissue >= 0.5,
        "qc_fail": np.where(tissue >= 0.5, "", "low_tissue")})
    with gzip.open(d / f"{prefix}.tiles.tsv.gz", "wt") as fh:
        tiles.to_csv(fh, sep="\t", index=False)
    sx = 0.125
    from PIL import Image

    Image.fromarray(np.full((int(H * T * sx), int(W * T * sx), 3), 220, np.uint8)).save(d / f"{prefix}.thumbnail.png")
    (d / f"{prefix}.prepare.json").write_text(json.dumps({
        "sample_id": sample, "mpp_target": 0.5, "tile_px": T, "scale_factor": 1.0,
        "thumbnail": {"mpp": 4.0, "resampled_to_thumbnail_x": sx, "resampled_to_thumbnail_y": sx,
                      "fullres_to_thumbnail": sx}}))
    lineage.write(d / f"{prefix}.prepare.lineage.json",
                  lineage.build("prepare", "TILEPARITY_PREPARE", prefix, {}, {"image": "slide.tif"}, {}))
    for step in ("uni2h", "lbp", "st"):
        parent = d / f"{prefix}.prepare.lineage.json"
        lineage.write(d / f"{prefix}.{step}.lineage.json",
                      lineage.build(step, step.upper(), prefix, {}, {}, {}, parents=[parent]))
    centres = rng.normal(0, 2, (4, 32))
    uni = {name: (centres[region] + rng.normal(0, 1, (n, 32)), [f"u{i}" for i in range(32)])
           for name in ("UNI2H_RGB", "UNI2H_RGB_HE")}
    features.write(str(d / f"{prefix}.uni2h.h5"), ids, uni)
    lbp_blocks = {}
    for ch in ("R", "G", "B"):
        probs = rng.dirichlet(np.ones(10) * 2, size=4)
        cols = [f"{ch}_r1_p8_u{u}" for u in range(10)]
        lbp_blocks[f"LBP_{ch}"] = (np.vstack([rng.multinomial(T * T, probs[k]) for k in region]), cols)
    lbp_blocks["INTENSITY_HE"] = (rng.normal(0, 1, (4, 10))[region] + rng.normal(0, 1, (n, 10)),
                                  [f"i{i}" for i in range(10)])
    features.write(str(d / f"{prefix}.lbp.h5"), ids, lbp_blocks)
    if image_only:
        Y = sparse.csc_matrix((0, n))
        gene_ids = []
    else:
        profiles = rng.gamma(2, 2, (4, genes)) * rng.lognormal(0, 1, (4, genes))
        counts = rng.poisson(profiles[region] * 3).T
        counts[:, 0] = 0  # one tile without RNA: outside the set extended
        Y = sparse.csc_matrix(counts)
        gene_ids = [f"G{i}" for i in range(genes)]
    st.write_counts_h5(str(d / f"{prefix}.counts.h5"), Y, ids, gene_ids, gene_ids)
    return region


def test_cli_clusters_every_view_and_measures_agreement(tmp_path, monkeypatch):
    import tp_cluster

    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    region = _write_slide(data)
    monkeypatch.chdir(out)
    tp_cluster.main(["--input-dir", str(data), "--k", "2,4,6", "--k-ref", "4", "--null-shifts", "20"])
    prefix = "S_mpp0.5000"
    labels = pd.read_csv(out / f"{prefix}.clusters.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    tiles = export.load_sample(data, prefix)["tiles"]
    assert labels["tile_id"].tolist() == tiles["tile_id"].tolist()
    keep = tiles["keep"].to_numpy()
    col = "keep_UNI2H_RGB_leiden_k4"
    assert (labels.loc[keep, col] != "").all() and (labels.loc[~keep, col] == "").all()
    assert cluster.ari(labels.loc[keep, col].astype(int).to_numpy(), region[keep]) > 0.9
    views = ["UNI2H_RGB", "UNI2H_RGB_HE", "LBP_RGB", "INTENSITY_HE", "EXPRESSION"]
    for s in ("keep", "extended"):
        for v in views:
            for m in ("leiden", "kmeans"):
                assert f"{s}_{v}_{m}_k6" in labels.columns
    assert labels["in_extended"].eq("False").sum() == 1  # the tile without RNA

    table = pd.read_csv(out / f"{prefix}.cluster_agreement.tsv", sep="\t")
    assert list(table.columns) == cluster.AGREEMENT_COLUMNS
    pairs = table[(table["method"] == "leiden") & (table["set"] == "keep")]
    assert len(pairs) == 3 * 10  # 3 values of k x 10 pairs of 5 views
    at4 = pairs[pairs["k"] == 4]
    assert (at4["ari"] > at4["null_ari_p95"]).all() and (at4["null_moves"] == 20).all()
    assert len(table[table["method"] == cluster.METHOD_PAIR]) == 2 * 3 * 5

    summary = json.loads((out / f"{prefix}.cluster.json").read_text())
    keep_info = summary["sets"]["keep"]
    assert keep_info["status"] == "clustered" and keep_info["reference_view"] == "EXPRESSION"
    assert keep_info["k_ref"] == 4 and keep_info["views"]["UNI2H_RGB"]["leiden"]["4"]["k_reached"] == 4
    for name in ("agreement_keep", "agreement_extended", "maps_keep", "maps_extended", "contingency_keep"):
        assert (out / f"{prefix}.cluster_{name}.png").exists()
    lin = lineage.load(out / f"{prefix}.cluster.lineage.json")
    assert lin["record"]["step"] == "cluster" and len(lin["record"]["parents"]) == 3

    # the exports add the Leiden labels of the set keep to the tile table
    (data / f"{prefix}.clusters.tsv.gz").write_bytes((out / f"{prefix}.clusters.tsv.gz").read_bytes())
    (data / f"{prefix}.cluster.lineage.json").write_bytes((out / f"{prefix}.cluster.lineage.json").read_bytes())
    s = export.load_sample(data, prefix)
    assert "cluster_UNI2H_RGB_leiden_k4" in s["tiles"].columns
    assert not any(c.startswith("cluster_") and "kmeans" in c for c in s["tiles"].columns)
    assert any(str(p).endswith(".cluster.lineage.json") for p in s["lineage_parents"])
    a = export.to_anndata(s)
    assert a.obs["cluster_EXPRESSION_leiden_k4"].astype(str).isin(["", "0", "1", "2", "3"]).all()


def test_image_only_slide_uses_uni2h_as_reference(tmp_path, monkeypatch):
    import tp_cluster

    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    _write_slide(data, image_only=True)
    monkeypatch.chdir(out)
    tp_cluster.main(["--input-dir", str(data), "--k", "2,4", "--k-ref", "8", "--methods", "leiden",
                     "--views", "UNI2H_RGB,LBP_RGB"])
    summary = json.loads((out / "S_mpp0.5000.cluster.json").read_text())
    keep = summary["sets"]["keep"]
    assert keep["reference_view"] == "UNI2H_RGB" and keep["views_clustered"] == ["UNI2H_RGB", "LBP_RGB"]
    assert keep["k_ref"] == 4  # the closest k below --k-ref
    assert summary["sets"]["extended"]["definition"].startswith("every extracted tile")
    table = pd.read_csv(out / "S_mpp0.5000.cluster_agreement.tsv", sep="\t")
    assert set(table["method"]) == {"leiden"}
    assert not (out / "S_mpp0.5000.cluster_contingency_extended.png").exists()


def test_small_or_repeated_sets_are_not_clustered():
    tiles = pd.DataFrame({"tile_id": [f"t{i}" for i in range(5)], "keep": True, "tile_px": 224, "row": 0,
                          "col": np.arange(5), "x0_px": 0, "y0_px": 0, "st_units_equiv": 1.0})
    sample = {"tiles": tiles, "blocks": {"UNI2H_RGB": (np.random.default_rng(0).normal(size=(5, 4)), list("abcd"))},
              "counts": sparse.csc_matrix((0, 5))}
    result = cluster.run(sample, ks=(2, 3))
    assert result["sets"]["keep"]["status"].startswith("not clustered: 5 tiles")
    assert result["agreement"].empty and list(result["agreement"].columns) == cluster.AGREEMENT_COLUMNS
    tiles = pd.concat([tiles] * 4, ignore_index=True).assign(tile_id=[f"t{i}" for i in range(20)],
                                                             col=np.arange(20))
    sample = {"tiles": tiles, "blocks": {"UNI2H_RGB": (np.random.default_rng(0).normal(size=(20, 4)), list("abcd"))},
              "counts": sparse.csc_matrix((0, 20))}
    result = cluster.run(sample, ks=(2, 3), methods=("kmeans",))
    assert result["sets"]["keep"]["status"] == "clustered"
    assert result["sets"]["extended"]["status"] == "not clustered: the same 20 tiles as the set 'keep'"
    table = cluster.labels_table(sample, result)
    assert table.columns.tolist() == ["tile_id", "in_keep", "in_extended", "keep_UNI2H_RGB_kmeans_k2",
                                      "keep_UNI2H_RGB_kmeans_k3"]
