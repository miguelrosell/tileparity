"""The cohort summary (with --cluster, on by default: --cohort_summary): the per-slide
results of one scale gathered in one place, so that findings can be read across slides
rather than slide by slide.

* ``long``: one row per slide, analysis and measured value (agreement between views by k,
  corrected ARI, macro-F1, stability, redundancy R^2, relevance R^2), with the sample
  metadata of the samplesheet (condition, patient, ...) in every row: the table to analyse
  further in R or Python.
* ``key``: one row per slide and image view with the numbers that summarise it (averages
  over k, values at the k shown, F1, stability, relevance), the table to read.
* group comparisons: when the samplesheet has a grouping column (``--cohort_group``,
  default ``condition``) with at least two groups, every key number is compared between
  groups. Sections of the same patient (``--cohort_pair``, default ``patient``) and group
  are averaged first, so that the patient is the unit. With two groups and patients seen
  in both, a paired Wilcoxon signed-rank test; otherwise Mann-Whitney (two groups) or
  Kruskal-Wallis (more). These are descriptive with few patients.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LONG_COLUMNS = ["sample", "analysis", "set", "method", "k", "a", "b", "metric", "value"]
RELEVANCE_METRICS = ("r2", "r2_ci_low", "r2_ci_high", "r2_null_p95", "r2_pc1", "r2_pc2", "r2_pc3", "r2_pc4", "r2_pc5",
                     "r2_half", "r2_ceiling", "fraction")


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return None


def discover(directory: str | Path) -> list[str]:
    """Prefixes (<sample>_mpp<scale>) with a cluster.json in ``directory``."""
    return sorted(p.name[: -len(".cluster.json")] for p in Path(directory).glob("*.cluster.json"))


def load(directory: str | Path, prefix: str) -> dict:
    d = Path(directory)
    out = {"prefix": prefix}
    for key, suffix in (("agreement", "cluster_agreement.tsv"), ("summary", "cluster_summary.tsv"),
                        ("f1", "cluster_f1.tsv"), ("stability", "cluster_stability.tsv"),
                        ("redundancy", "redundancy.tsv"), ("relevance", "relevance.tsv"),
                        ("partition", "relevance_partition.tsv")):
        out[key] = _read(d / f"{prefix}.{suffix}")
    for key, suffix in (("genes", "relevance_genes.tsv.gz"), ("gene_partition", "relevance_gene_partition.tsv.gz"),
                        ("domains", "markers_domains.tsv"), ("markers", "markers.tsv.gz"),
                        ("texture", "texture_genes.tsv.gz")):
        path = d / f"{prefix}.{suffix}"
        out[key] = _read(path) if path.exists() else None
    out["info"] = json.loads((d / f"{prefix}.cluster.json").read_text())
    return out


def sample_name(slide: dict) -> str:
    return str(slide["info"].get("sample_id") or slide["prefix"].rsplit("_mpp", 1)[0])


def long_table(slides: list[dict]) -> pd.DataFrame:
    """Every measured value of every slide in one long table."""
    parts = []
    for sl in slides:
        s = sample_name(sl)
        a = sl.get("agreement")
        if a is not None and len(a):
            t = a[(a["k_a"] == a["k"]) & (a["k_b"] == a["k"])]
            for metric in ("ari", "ari_corrected", "null_ari_p95", "nmi"):
                if metric in t:
                    parts.append(pd.DataFrame({"sample": s, "analysis": "agreement", "set": t["set"],
                                               "method": t["method"], "k": t["k"], "a": t["view_a"], "b": t["view_b"],
                                               "metric": metric, "value": t[metric]}))
        f1 = sl.get("f1")
        if f1 is not None and len(f1):
            m = f1.groupby(["set", "method", "k", "view", "reference"], sort=False)["f1"].mean().reset_index()
            parts.append(pd.DataFrame({"sample": s, "analysis": "f1", "set": m["set"], "method": m["method"],
                                       "k": m["k"], "a": m["view"], "b": m["reference"], "metric": "macro_f1",
                                       "value": m["f1"]}))
        st = sl.get("stability")
        if st is not None and len(st):
            g = st.groupby(["set", "method", "k", "view"], sort=False)
            for col in ("ari_with_reference", "ari_with_full"):
                for stat in ("mean", "std"):
                    v = getattr(g[col], stat)().reset_index()
                    parts.append(pd.DataFrame({"sample": s, "analysis": "stability", "set": v["set"],
                                               "method": v["method"], "k": v["k"], "a": v["view"], "b": "",
                                               "metric": f"{col}_{'sd' if stat == 'std' else stat}", "value": v[col]}))
        r = sl.get("redundancy")
        if r is not None and len(r):
            for metric in ("r2", "r2_ci_low", "r2_ci_high", "r2_null_p95"):
                if metric in r:
                    parts.append(pd.DataFrame({"sample": s, "analysis": "redundancy", "set": r["set"], "method": "",
                                               "k": np.nan, "a": r["predictor"], "b": r["target"], "metric": metric,
                                               "value": r[metric]}))
        rel = sl.get("relevance")
        if rel is not None and len(rel):
            for metric in [c for c in RELEVANCE_METRICS if c in rel]:
                parts.append(pd.DataFrame({"sample": s, "analysis": "relevance", "set": rel["set"], "method": "",
                                           "k": np.nan, "a": rel["predictor"], "b": "EXPRESSION", "metric": metric,
                                           "value": rel[metric]}))
        p = sl.get("partition")
        if p is not None and len(p):
            for metric in [m for m in ("r2_joint", "only_a", "only_b", "shared", "not_explained") if m in p]:
                parts.append(pd.DataFrame({"sample": s, "analysis": "relevance_partition", "set": p["set"],
                                           "method": "", "k": np.nan, "a": p["view_a"], "b": p["view_b"],
                                           "metric": metric, "value": p[metric]}))
    if not parts:
        return pd.DataFrame(columns=LONG_COLUMNS)
    return pd.concat(parts, ignore_index=True)[LONG_COLUMNS]


def key_table(slides: list[dict]) -> pd.DataFrame:
    """One row per slide and image view (set keep): the numbers that summarise it."""
    rows = []
    for sl in slides:
        s, info = sample_name(sl), sl["info"].get("sets", {}).get("keep", {})
        ref = info.get("reference_view")
        views = [v for v in info.get("views_clustered", []) if v != ref]
        summ, f1, st, rel = sl.get("summary"), sl.get("f1"), sl.get("stability"), sl.get("relevance")
        for v in views:
            row = {"sample": s, "view": v, "reference": ref}
            for m in ("kmeans", "leiden"):
                if summ is not None and len(summ):
                    t = summ[(summ["set"] == "keep") & (summ["method"] == m) &
                             (((summ["view_a"] == v) & (summ["view_b"] == ref)) |
                              ((summ["view_b"] == v) & (summ["view_a"] == ref)))]
                    if len(t):
                        r = t.iloc[0]
                        row[f"ari_mean_k_{m}"] = r["ari_mean_k"]
                        row[f"ari_corrected_mean_k_{m}"] = r["ari_corrected_mean_k"]
                        row[f"k_show_{m}"] = r["k_show"]
                        row[f"ari_k_show_{m}"] = r["ari_k_show"]
                        row[f"ari_corrected_k_show_{m}"] = r["ari_corrected_k_show"]
                if f1 is not None and len(f1):
                    t = f1[(f1["set"] == "keep") & (f1["method"] == m) & (f1["view"] == v)]
                    if len(t):
                        row[f"macro_f1_mean_k_{m}"] = t.groupby("k")["f1"].mean().mean()
                if st is not None and len(st):
                    t = st[(st["set"] == "keep") & (st["method"] == m) & (st["view"] == v)]
                    if len(t):
                        row[f"stability_sd_mean_k_{m}"] = t.groupby("k")["ari_with_reference"].std().mean()
                        row[f"stability_self_mean_k_{m}"] = t.groupby("k")["ari_with_full"].mean().mean()
            if rel is not None and len(rel):
                t = rel[rel["predictor"] == v]
                if len(t):
                    row["relevance_r2"] = t["r2"].iloc[0]
                    row["relevance_r2_null_p95"] = t["r2_null_p95"].iloc[0]
                    for c, name in (("r2_pc1", "relevance_r2_pc1"), ("fraction", "relevance_fraction")):
                        if c in t:
                            row[name] = t[c].iloc[0]
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["sample", "view", "reference"])


KEY_METRICS = {"ari_corrected_mean_k_kmeans": "corrected ARI with expression, mean over k (k-means)",
               "ari_corrected_mean_k_leiden": "corrected ARI with expression, mean over k (Leiden)",
               "macro_f1_mean_k_kmeans": "macro-F1, mean over k (k-means)",
               "relevance_r2": "R2 of the expression from the view",
               "relevance_fraction": "share of the reachable expression predicted from the view"}


# ---- 0.7.0: corrected ARI by k, genes, domains, texture across slides ------------------------------------------------
def ari_by_k(long: pd.DataFrame, ref: str) -> pd.DataFrame:
    """Per method, view and k: raw and corrected ARI with ``ref`` over the slides (mean,
    median, lowest, highest, slides), and whether k is the best k of the view (largest median
    corrected ARI)."""
    t = long[(long["analysis"] == "agreement") & long["metric"].isin(["ari", "ari_corrected"]) &
             (long["set"] == "keep") & ((long["a"] == ref) | (long["b"] == ref))]
    if t.empty:
        return pd.DataFrame()
    t = t.assign(view=np.where(t["a"] == ref, t["b"], t["a"]))
    t = t[~t["view"].astype(str).str.contains(r"\.", regex=True)]
    g = t.groupby(["method", "view", "k", "metric"])["value"]
    out = g.agg(mean="mean", median="median", lowest="min", highest="max", slides="count").reset_index()
    wide = out.pivot_table(index=["method", "view", "k"], columns="metric",
                           values=["mean", "median", "lowest", "highest", "slides"])
    wide.columns = [f"{m}_{s}" for s, m in wide.columns]
    order = [f"{m}_{s}" for m in ("ari", "ari_corrected") for s in ("mean", "median", "lowest", "highest", "slides")]
    wide = wide[[c for c in order if c in wide.columns]].reset_index()
    if "ari_corrected_median" in wide:
        ok = wide.dropna(subset=["ari_corrected_median"])
        best = ok.loc[ok.groupby(["method", "view"])["ari_corrected_median"].idxmax(), ["method", "view", "k"]]
        wide["best_k"] = wide.set_index(["method", "view", "k"]).index.isin(best.set_index(["method", "view", "k"]).index)
    return wide


def gene_table(slides: list[dict], level: float = 0.05) -> pd.DataFrame:
    """Per gene: mean R2 from every view over the slides where it was tested, in how many
    slides, and in how many it beats the spatial null (q < level)."""
    parts = [s["genes"].assign(sample=sample_name(s)) for s in slides if s.get("genes") is not None]
    if not parts:
        return pd.DataFrame()
    g = pd.concat(parts, ignore_index=True)
    r2cols = [c for c in g.columns if c.startswith("r2_") and "null" not in c]
    agg = g.groupby(["gene_id", "gene_name"])[r2cols].mean()
    agg.columns = [f"{c}_mean" for c in agg.columns]
    agg["slides"] = g.groupby(["gene_id", "gene_name"]).size()
    for c in [c for c in g.columns if c.startswith("q_")]:
        agg[f"slides_{c[2:]}_q{str(level)[2:]}"] = g.assign(_s=g[c] < level).groupby(["gene_id", "gene_name"])["_s"].sum()
    if "reliability" in g:
        agg["reliability_mean"] = g.groupby(["gene_id", "gene_name"])["reliability"].mean()
    return agg.reset_index().sort_values(agg.columns[0] if len(agg.columns) else "gene_id", ascending=False)


def gene_partition_table(slides: list[dict]) -> pd.DataFrame:
    """Per slide and pair of views: the mean over the genes of every part of the gene's
    variance (only A, shared, only B, joint, not explained) and of the reliability, for all
    genes and for the well-measured ones (reliability >= relevance.WELL_MEASURED)."""
    from .relevance import WELL_MEASURED

    rows = []
    for s in slides:
        gp = s.get("gene_partition")
        if gp is None or gp.empty or "r2_joint" not in gp:
            continue
        for (a, b), t in gp.groupby(["view_a", "view_b"], sort=False):
            for group, u in (("all", t), ("well_measured", t[t["reliability"] >= WELL_MEASURED] if "reliability" in t
                                          else t.iloc[0:0])):
                row = {"sample": sample_name(s), "view_a": a, "view_b": b, "genes": group, "n_genes": len(u)}
                for c in ("only_a", "shared", "only_b", "r2_joint", "not_explained", "reliability"):
                    row[f"mean_{c}"] = float(u[c].mean()) if c in u and len(u) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def partition_table(slides: list[dict]) -> pd.DataFrame:
    """Per slide and pair of views, the partition of the expression's components (relevance)."""
    parts = [s["partition"].assign(sample=sample_name(s)) for s in slides
             if s.get("partition") is not None and len(s["partition"]) and "r2_joint" in s["partition"]]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def domains_table(slides: list[dict], n_markers: int = 10) -> pd.DataFrame:
    """Per slide, method (at its k shown) and expression domain: its size, the F1 of every
    image view and its best markers, to name the domains across slides."""
    rows = []
    for s in slides:
        dom, mk = s.get("domains"), s.get("markers")
        if dom is None or dom.empty or mk is None or mk.empty:
            continue
        info = s["info"].get("sets", {}).get("keep", {})
        for m, k in (info.get("k_show") or {}).items():
            d = dom[(dom["method"] == m) & (dom["k"] == k)]
            for r in d.to_dict("records"):
                g = mk[(mk["method"] == m) & (mk["k"] == k) & (mk["domain"] == r["domain"])].sort_values("rank")
                r = {"sample": sample_name(s), **r,
                     "markers": ", ".join(f"{x.gene_name} ({x.auroc:.2f})" for x in g.head(n_markers).itertuples())}
                rows.append(r)
    return pd.DataFrame(rows)


def texture_replication(slides: list[dict], level: float = 0.05) -> pd.DataFrame:
    """Per texture feature and gene: in how many slides it was tested, its median partial
    Spearman rho, and in how many slides it reaches q < level with a positive and with a
    negative rho."""
    parts = [s["texture"].assign(sample=sample_name(s)) for s in slides if s.get("texture") is not None]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts, ignore_index=True)
    t["sig_pos"] = (t["q"] < level) & (t["rho"] > 0)
    t["sig_neg"] = (t["q"] < level) & (t["rho"] < 0)
    g = t.groupby(["feature", "type", "radius_px", "gene_id", "gene_name"], sort=False)
    out = g.agg(slides=("rho", "count"), rho_median=("rho", "median"), slides_sig_pos=("sig_pos", "sum"),
                slides_sig_neg=("sig_neg", "sum"), n_eff_median=("n_eff", "median")).reset_index()
    out["slides_sig_same_sign"] = np.where(out["rho_median"] >= 0, out["slides_sig_pos"], out["slides_sig_neg"])
    return out.sort_values(["slides_sig_same_sign", "rho_median"], ascending=[False, False], key=lambda c: c.abs()
                           if c.name == "rho_median" else c)


def patient_level(key: pd.DataFrame, meta: pd.DataFrame, group: str, pair: str | None) -> pd.DataFrame:
    """Key numbers averaged over the sections of the same patient (``pair``) and group."""
    t = key.merge(meta, on="sample", how="left")
    if group not in t:
        return pd.DataFrame()
    unit = pair if (pair and pair in t and t[pair].astype(str).str.len().gt(0).all()) else "sample"
    cols = [c for c in KEY_METRICS if c in t]
    return t.groupby(["view", group, unit], dropna=False)[cols].mean().reset_index().rename(columns={unit: "unit"})


def compare_groups(key: pd.DataFrame, meta: pd.DataFrame, group: str, pair: str | None) -> pd.DataFrame:
    """Every key number compared between the groups of ``group``, per view (see the module
    docstring for the tests)."""
    from scipy import stats

    pl = patient_level(key, meta, group, pair)
    if pl.empty:
        return pd.DataFrame()
    groups = [g for g in pd.unique(pl[group].astype(str)) if g not in ("", "nan")]
    rows = []
    if len(groups) < 2:
        return pd.DataFrame()
    for v, t in pl.groupby("view", sort=False):
        for metric in [c for c in KEY_METRICS if c in t]:
            vals = {g: t.loc[t[group].astype(str) == g, ["unit", metric]].dropna() for g in groups}
            row = {"view": v, "metric": metric, "group_column": group, "unit": pair or "sample"}
            for g in groups:
                row[f"n_{g}"] = len(vals[g])
                row[f"mean_{g}"] = float(vals[g][metric].mean()) if len(vals[g]) else np.nan
            test, p, n_pairs = "", np.nan, 0
            if len(groups) == 2:
                a, b = vals[groups[0]].set_index("unit")[metric], vals[groups[1]].set_index("unit")[metric]
                both = a.index.intersection(b.index)
                n_pairs = len(both)
                if n_pairs >= 3:
                    d = (b.loc[both] - a.loc[both]).to_numpy(float)
                    test = "paired Wilcoxon signed-rank"
                    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                    row["median_paired_difference"] = float(np.median(d))
                elif len(a) >= 2 and len(b) >= 2:
                    test, p = "Mann-Whitney U", float(stats.mannwhitneyu(a, b).pvalue)
            else:
                samples = [vals[g][metric].to_numpy(float) for g in groups if len(vals[g]) >= 2]
                if len(samples) >= 2:
                    test, p = "Kruskal-Wallis", float(stats.kruskal(*samples).pvalue)
            row.update(test=test, p_value=p, n_pairs=n_pairs)
            rows.append(row)
    return pd.DataFrame(rows)
