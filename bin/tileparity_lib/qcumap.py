"""Optional QC: 3-D UMAP of every feature view of one scale (``--qc_umap``).

One interactive HTML page per scale (plotly; it can be rotated, zoomed and hovered, and
works offline) with one 3-D UMAP per view, coloured by sample or by a QC column:

* ``UNI2H_RGB``, ``UNI2H_RGB_HE``: columns standardised (z-scores), then PCA.
* ``LBP_RGB``, ``LBP_RGB_HE``, ``LBP_HE``: the analysis recipe for LBP histograms. Each
  radius block of each channel is turned into proportions (divided by the tile's pixel
  count, so it sums to 1), square-rooted (Hellinger transform, so that a few large bins
  such as the irregular one do not dominate distances), and divided by the square root
  of its largest eigenvalue so that every radius and channel weighs the same (multiple
  factor analysis), then PCA.
* ``EXPRESSION``: counts per tile normalised to 10 000, log1p, the 2 000 most variable
  genes, then PCA.

UMAP (n_components = 3) runs on the first 50 principal components. This is a quick look
for batch effects and outliers, not an analysis: UMAP distorts distances, so the size
and separation of groups are not interpretable.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

# LBP column names: <channel>_r<R>_p<P>_u<code>, e.g. R_r3_p19_u4 or R_he_r3_p19_u4
COLUMN_RE = re.compile(r"^(?:center_)?(?P<ch>[A-Za-z][A-Za-z_]*?)_r(?P<r>\d+)_p(?P<p>\d+)_u(?P<u>\d+)$")
UNI2H_VIEWS = ("UNI2H_RGB", "UNI2H_RGB_HE")
LBP_VIEWS = ("LBP_RGB", "LBP_RGB_HE", "LBP_HE")
VIEW_INPUT = {"UNI2H_RGB": "the slide", "UNI2H_RGB_HE": "the slide rebuilt from H and E only",
              "LBP_RGB": "R, G, B of the slide", "LBP_RGB_HE": "R, G, B of the slide rebuilt from H and E only",
              "LBP_HE": "the H and E amounts"}


def lbp_recipe(X: np.ndarray, columns, tile_px: int) -> np.ndarray:
    """Proportions -> Hellinger -> equal weight per (channel, radius) block."""
    X = np.asarray(X, float) / float(tile_px**2)
    keys = []
    for c in columns:
        m = COLUMN_RE.match(str(c))
        keys.append((m.group("ch"), int(m.group("r"))) if m else ("?", 0))
    keys = np.array([f"{a}:{b}" for a, b in keys])
    out = []
    for k in dict.fromkeys(keys):
        B = np.sqrt(np.clip(X[:, keys == k], 0, None))
        B = B - B.mean(axis=0)
        if B.shape[0] > 1:
            top = np.linalg.svd(B, compute_uv=False)[0] ** 2 / max(B.shape[0] - 1, 1)
            if top > 0:
                B = B / np.sqrt(top)
        out.append(B)
    return np.hstack(out)


def expression_matrix(Y, n_hvg: int = 2000) -> np.ndarray | None:
    """log1p(counts / total * 1e4) of the most variable genes; ``Y`` is genes x tiles,
    the result tiles x genes."""
    from scipy import sparse

    Y = sparse.csr_matrix(sparse.csc_matrix(Y).T).astype(float)
    if Y.shape[1] == 0:
        return None
    totals = np.asarray(Y.sum(axis=1)).ravel()
    totals[totals == 0] = 1
    N = sparse.diags(1e4 / totals) @ Y
    N.data = np.log1p(N.data)
    N = N.toarray()
    if N.shape[1] > n_hvg:
        N = N[:, np.argsort(N.var(axis=0))[::-1][:n_hvg]]
    return N


def embed3d(X: np.ndarray, n_pcs: int = 50, seed: int = 0) -> tuple[np.ndarray | None, str]:
    from sklearn.decomposition import PCA

    n = X.shape[0]
    if n < 10:
        return None, f"only {n} tiles: no UMAP"
    X = np.asarray(X, float)
    X = X[:, X.std(axis=0) > 0] if X.shape[1] else X
    if X.shape[1] == 0:
        return None, "no variable features"
    k = int(min(n_pcs, n - 1, X.shape[1]))
    pcs = PCA(n_components=k, random_state=seed).fit_transform(X)
    import warnings

    import umap

    reducer = umap.UMAP(n_components=3, n_neighbors=int(min(15, n - 1)), min_dist=0.3, random_state=seed)
    with warnings.catch_warnings():
        # a fixed seed makes umap run single-threaded and say so; that is intended here
        warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)
        coords = reducer.fit_transform(pcs)
    return coords, f"PCA {k} components -> UMAP 3-D"


def views(samples: list[dict]) -> dict:
    """{"obs": table of all tiles, "views": {view name: (tiles x d matrix, description, obs of
    those tiles)}}. Image views use every sample; the expression view uses the samples with
    expression data, on the genes they share (image-only samples are left out of it)."""
    from scipy import sparse

    def tile_table(s):
        t = s["tiles"].copy()
        has_genes = s["counts"].shape[0] > 0
        t["total_counts"] = (np.asarray(sparse.csc_matrix(s["counts"]).sum(axis=0)).ravel() if has_genes
                             else np.full(len(t), np.nan))
        return t

    obs = pd.concat([tile_table(s) for s in samples], ignore_index=True)
    out = {}
    names = set.intersection(*(set(s["blocks"]) for s in samples)) if samples else set()
    for v in UNI2H_VIEWS:
        if v in names:
            X = np.vstack([s["blocks"][v][0] for s in samples]).astype(float)
            sd = X.std(axis=0)
            sd[sd == 0] = 1
            out[v] = ((X - X.mean(axis=0)) / sd, f"UNI2-h class token of {VIEW_INPUT[v]}, z-scored", obs)
    for v in LBP_VIEWS:
        if v in names:
            tile_px = int(obs["tile_px"].iloc[0])
            X = np.vstack([s["blocks"][v][0] for s in samples])
            out[v] = (lbp_recipe(X, samples[0]["blocks"][v][1], tile_px),
                      f"LBP histograms of {VIEW_INPUT[v]}: proportions, Hellinger, equal weight per radius and "
                      "channel", obs)
    with_genes = [s for s in samples if s["counts"].shape[0] > 0]
    if with_genes:
        genes = sorted(set.intersection(*(set(map(str, s["gene_ids"])) for s in with_genes)))
        if genes:
            mats = []
            for s in with_genes:
                idx = pd.Series(np.arange(len(s["gene_ids"])), index=np.asarray(s["gene_ids"], str)).loc[genes].to_numpy()
                mats.append(sparse.csc_matrix(s["counts"])[idx, :])
            E = expression_matrix(sparse.hstack(mats))
            if E is not None:
                left_out = len(samples) - len(with_genes)
                note = "expression: counts per 10 000, log1p, 2 000 most variable genes"
                if left_out:
                    note += f" ({left_out} image-only sample(s) left out)"
                out["EXPRESSION"] = (E, note, pd.concat([tile_table(s) for s in with_genes], ignore_index=True))
    return {"obs": obs, "views": out}


COLOUR_COLUMNS = (("sample_id", "categorical"), ("keep", "categorical"), ("tissue_fraction", "continuous"),
                  ("blur_flag", "categorical"), ("total_counts", "continuous"))


def figure(coords: np.ndarray, obs: pd.DataFrame, title: str):
    import plotly.graph_objects as go

    hover = (obs["tile_id"].astype(str) + "<br>sample " + obs["sample_id"].astype(str) +
             "<br>x " + obs["cx_um"].round(0).astype(str) + " µm, y " + obs["cy_um"].round(0).astype(str) + " µm" +
             "<br>tissue " + (100 * obs["tissue_fraction"].astype(float)).round(0).astype(str) + "%")
    traces, groups = [], []
    for col, kind in COLOUR_COLUMNS:
        if col not in obs.columns:
            continue
        if kind == "continuous" and pd.to_numeric(obs[col], errors="coerce").isna().all():
            continue  # e.g. total_counts when no sample has expression data
        start = len(traces)
        if kind == "categorical":
            values = obs[col].astype(str)
            for v in sorted(values.unique()):
                sel = (values == v).to_numpy()
                traces.append(go.Scatter3d(x=coords[sel, 0], y=coords[sel, 1], z=coords[sel, 2], mode="markers",
                                           name=f"{col} = {v}", text=hover[sel], hoverinfo="text",
                                           marker=dict(size=2.5, opacity=0.8), visible=False))
        else:
            vals = pd.to_numeric(obs[col], errors="coerce").to_numpy(float)
            traces.append(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
                                       name=col, text=hover, hoverinfo="text", visible=False,
                                       marker=dict(size=2.5, opacity=0.8, color=vals, colorscale="Viridis",
                                                   colorbar=dict(title=col))))
        groups.append((col, list(range(start, len(traces)))))
    for i in groups[0][1]:
        traces[i].visible = True
    buttons = []
    for col, idx in groups:
        vis = [i in idx for i in range(len(traces))]
        buttons.append(dict(label=f"colour: {col}", method="update", args=[{"visible": vis}]))
    fig = go.Figure(data=traces)
    # title on the right, so that it does not sit under the colour menu on the left
    fig.update_layout(title=dict(text=title, x=0.98, xanchor="right"), height=700, margin=dict(l=0, r=0, t=50, b=0),
                      scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
                      updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.08, xanchor="left")],
                      legend=dict(itemsizing="constant"))
    return fig


def page(sections: list[tuple[str, str, str]], title: str, caption: dict) -> str:
    """HTML page: one (heading, plot html, note) per view, then the explanation."""
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
             "<style>body{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px;color:#222}"
             "h2{margin-top:32px}.cap p{margin:6px 0;color:#444}.cap b{color:#111}.note{color:#666}</style>",
             f"</head><body><h1>{html.escape(title)}</h1>"]
    for heading, plot, note in sections:
        parts.append(f"<h2>{html.escape(heading)}</h2><p class='note'>{html.escape(note)}</p>{plot}")
    parts.append("<div class='cap'>")
    for label in ("What it shows", "How to read it", "Caveats", "Where it comes from"):
        parts.append(f"<p><b>{label}:</b> {html.escape(caption.get(label, ''))}</p>")
    parts.append("</div></body></html>")
    return "\n".join(parts)


def to_html(fig, first: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=True if first else False)
