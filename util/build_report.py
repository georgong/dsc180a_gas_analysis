from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import io, base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go  
import plotly.io as pio


def _nonna_sampling_html(df: pd.DataFrame, max_rows: int = 5) -> str:
    sample = df[df.notna().any(axis=1)].head(max_rows)
    return sample.to_html(classes="dataframe compact", border=0, index=False, escape=False)

def _series_table_html(s: pd.Series, key_name: str, val_name: str) -> str:
    t = pd.DataFrame({key_name: s.index.astype(str).tolist(), val_name: s.values.tolist()})
    return t.to_html(classes="dataframe mini", border=0, index=False, escape=False)

def _describe_html(df: pd.DataFrame) -> str:
    tmp = df.copy()
    dt_cols = tmp.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    if dt_cols:
        for c in dt_cols:
            s = tmp[c]
            mask = s.notna()
            arr = np.full(len(s), np.nan, dtype="float64")
            arr[mask.values] = s[mask].astype("int64").to_numpy() / 1e9
            tmp[c] = arr
    d = tmp.describe(include="all").T
    d.index.name = "column"
    d = d.reset_index()
    return d.to_html(classes="dataframe compact", border=0, index=False, escape=False)

def _compute_shared(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], List[str]]]:
    names = list(dfs.keys())
    shared_matrix = pd.DataFrame(0, index=names, columns=names, dtype=int)
    pair_to_cols: Dict[Tuple[str, str], List[str]] = {}
    for i, a in enumerate(names):
        cols_a = set(map(str, dfs[a].columns))
        for j, b in enumerate(names):
            cols_b = set(map(str, dfs[b].columns))
            shared = sorted(cols_a & cols_b)
            shared_matrix.loc[a, b] = len(shared) if i != j else len(cols_a)
            if i < j and shared:
                pair_to_cols[(a, b)] = shared
    return shared_matrix, pair_to_cols

def _build_graph_png_b64(dfs: Dict[str, pd.DataFrame]) -> str:
    names = list(dfs.keys())
    G = nx.Graph()
    G.add_nodes_from(names)
    for i in range(len(names)):
        cols_i = set(map(str, dfs[names[i]].columns))
        for j in range(i + 1, len(names)):
            cols_j = set(map(str, dfs[names[j]].columns))
            shared = cols_i & cols_j
            if shared:
                G.add_edge(names[i], names[j], weight=len(shared), label=str(len(shared)))
    fig = plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Shared Columns Between Tables")
    plt.axis("off")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _make_corr_frame(df: pd.DataFrame, min_unique: int = 5) -> pd.DataFrame:
    """
    """
    parts = []

    
    num = df.select_dtypes(include=[np.number]).copy()
    parts.append(num)

    
    dt = df.select_dtypes(include=["datetime64[ns]", "datetimetz"])
    for c in dt.columns:
        s = dt[c]
        bundle = pd.DataFrame({
            f"{c}__year":   s.dt.year,
            f"{c}__month":  s.dt.month,
            f"{c}__day":    s.dt.day,
            f"{c}__hour":   s.dt.hour,
            f"{c}__minute": s.dt.minute,
            f"{c}__second": s.dt.second,
        }, index=s.index)
        parts.append(bundle)

    if not parts:
        return pd.DataFrame(index=df.index)

    X = pd.concat(parts, axis=1)

    
    X = X.dropna(axis=1, how="all")
    if X.empty:
        return X

    
    nunq = X.nunique(dropna=True)
    var0 = X.var(numeric_only=True).reindex(X.columns).fillna(0) == 0
    low_card = nunq < min_unique
    keep_cols = X.columns[~(var0 | low_card)]
    X = X[keep_cols]

    return X


def _corr_plotly_html(df: pd.DataFrame) -> Optional[str]:
    """
    """
    X = _make_corr_frame(df)
    if X.shape[1] < 2:
        return None

    c = X.corr(numeric_only=True)
    if c.shape[0] < 2 or c.isna().all().all():
        return None

    # Heatmap with hovertext
    z = c.values
    xlabels = c.columns.tolist()
    ylabels = c.index.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=xlabels,
        y=ylabels,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        colorbar=dict(title="corr"),
        hoverongaps=False,
        hovertemplate="row=%{y}<br>col=%{x}<br>corr=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Correlation (numeric + datetime parts)",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(automargin=True),
        height=520,  
    )

    
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})
    return html


def _css() -> str:
    return """
<style>
:root {
  --bg: #0f172a; --panel: #111827; --muted: #9ca3af; --text: #e5e7eb;
  --accent: #38bdf8; --card: #0b1220; --card-border: #1f2937; --table-header: #0f172a;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: linear-gradient(180deg,#0b0f1a 0%,#0f172a 100%); color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji"; line-height: 1.5; }
.container { max-width: 1200px; margin: 32px auto; padding: 0 16px 48px 16px; }
.header { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 16px; }
.title { font-size: 28px; font-weight: 700; letter-spacing: 0.2px; }
.meta { color: var(--muted); font-size: 14px; }
.badge { display: inline-block; padding: 2px 8px; font-size: 12px; border: 1px solid rgba(56,189,248,0.5); border-radius: 999px; color: #a5f3fc; }
.panel { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 16px 20px; margin: 16px 0 28px 0; box-shadow: 0 8px 24px rgba(0,0,0,0.25); }
.anchor-list { display: flex; flex-wrap: wrap; gap: 8px; }
.anchor-list a { color: #93c5fd; text-decoration: none; font-size: 13px; }
.anchor-list a:hover { text-decoration: underline; }
.graph-wrap { display: grid; grid-template-columns: 1.2fr 1fr; gap: 14px; margin: 8px 0 24px 0; }
.graph-card { border: 1px solid var(--card-border); background: var(--card); border-radius: 14px; padding: 12px; }
.graph-img { width: 100%; height: auto; border-radius: 10px; display: block; }
.table-section { margin: 24px 0 40px 0; padding: 16px; border: 1px solid rgba(255,255,255,0.06); border-radius: 16px;
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02)); box-shadow: 0 8px 24px rgba(0,0,0,0.18); }
.section-header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
.table-name { margin: 0; font-size: 22px; letter-spacing: 0.2px; }
.desc { font-size: 14px; color: var(--muted); margin-bottom: 12px; }
.cards-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr)); /* 4 列 */
  grid-auto-rows: minmax(40px, auto);
  gap: 14px;
}
.card { border: 1px solid var(--card-border); background: var(--card); border-radius: 14px; padding: 12px; overflow: auto; }
.card-title { font-size: 14px; color: var(--muted); margin-bottom: 8px; letter-spacing: 0.2px; }
table.dataframe { width: 100%; border-collapse: collapse; border: 1px solid rgba(255,255,255,0.06);
  background: rgba(17,24,39,0.4); border-radius: 10px; overflow: hidden; font-size: 13px; }
table.dataframe thead tr { background: var(--table-header); }
table.dataframe th, table.dataframe td { padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.06); text-align: left; vertical-align: top; }
table.dataframe th { color: #cbd5e1; font-weight: 600; }
table.dataframe tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
table.dataframe tr:hover td { background: rgba(56,189,248,0.08); }
.hr { height: 1px; background: linear-gradient(90deg, rgba(56,189,248,0), rgba(56,189,248,0.35), rgba(56,189,248,0)); margin: 18px 0; }
.footer { color: var(--muted); font-size: 12px; text-align: center; margin-top: 24px; }
.kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace; background: #111827; border: 1px solid #1f2937; border-radius: 6px; padding: 0 6px; }
.plotly-wrap { width: 100%; }
</style>
"""


def generate_tables_report(
    dfs: Dict[str, pd.DataFrame],
    descriptions: Optional[Dict[str, str]] = None,
    out_html_path: str | Path = "tables_overview.html",
    max_rows: int = 5,
) -> Path:
    if descriptions is None:
        descriptions = {name: "" for name in dfs.keys()}
    else:
        for name in dfs.keys():
            descriptions.setdefault(name, "")

    anchors = " ".join([f'<a href="#{name}">{name}</a>' for name in dfs.keys()])
    shared_matrix, pair_to_cols = _compute_shared(dfs)
    shared_matrix_html = shared_matrix.to_html(classes="dataframe compact", border=0, index=True, escape=False)
    graph_b64 = _build_graph_png_b64(dfs)
    graph_img_html = f'<img src="{graph_b64}" alt="Shared Columns Graph" class="graph-img" />'

    if pair_to_cols:
        pairs_html = []
        for (a, b), cols in sorted(pair_to_cols.items()):
            col_list_html = "<code>" + "</code>, <code>".join(map(str, cols)) + "</code>"
            pairs_html.append(f"<li><span class='kbd'>{a}</span> ↔ <span class='kbd'>{b}</span>: {col_list_html}</li>")
        shared_pairs_block = f"<details open><summary>Shared columns (by table pairs)</summary><ul>{''.join(pairs_html)}</ul></details>"
    else:
        shared_pairs_block = "<p class='meta'>(No shared columns between tables)</p>"

    sections = []
    for name, df in dfs.items():
        nonna_html    = _nonna_sampling_html(df, max_rows=max_rows)
        dtypes_html   = _series_table_html(df.dtypes.astype(str), key_name="column", val_name="dtype")
        nan_prop      = df.isna().mean().round(2)
        nan_html      = _series_table_html(nan_prop, key_name="column", val_name="nan_prop")
        describe_html = _describe_html(df)

        corr_html = _corr_plotly_html(df)
        corr_card = f"""
          <div class="card" style="grid-row: 4 / 8; grid-column: 1 / 5;">
            <div class="card-title">corr() heatmap (numeric + datetime parts)</div>
            <div class="plotly-wrap">
              {corr_html}
            </div>
          </div>
        """ if corr_html else ""

        desc_html = descriptions.get(name) or "<em>(fill in later)</em>"

        section = f"""
        <section class="table-section" id="{name}">
          <div class="section-header">
            <h2 class="table-name">{name}</h2>
          </div>
          <div class="desc"><span class="desc-label">Description:</span> <span class="desc-text">{desc_html}</span></div>

          <div class="cards-grid">
            <!-- non_sampling_table [0, 0:4] -->
            <div class="card" style="grid-row: 1 / 2; grid-column: 1 / 5;">
              <div class="card-title">nonna_sampling (first {max_rows})</div>
              {nonna_html}
            </div>

            <!-- dtypes [1:0:2] -->
            <div class="card" style="grid-row: 2 / 3; grid-column: 1 / 3;">
              <div class="card-title">DTypes</div>
              {dtypes_html}
            </div>

            <!-- nan_prop [1:2:4] -->
            <div class="card" style="grid-row: 2 / 3; grid-column: 3 / 5;">
              <div class="card-title">NaN Proportion</div>
              {nan_html}
            </div>

            <!-- describe [2, 0:4] -->
            <div class="card" style="grid-row: 3 / 4; grid-column: 1 / 5;">
              <div class="card-title">describe(include='all')</div>
              {describe_html}
            </div>

            <!-- corr [3:7,0:4] -->
            {corr_card}
          </div>
        </section>
        """
        sections.append(section)

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Tables Overview</title>{_css()}
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title">Tables Overview</div>
    </div>

    <div class="panel">
      <h3>Index</h3>
      <div class="anchor-list">{anchors}</div>
    </div>

    <div class="graph-wrap">
      <div class="graph-card">
        <div class="graph-title">Shared Columns Graph (tables as nodes, edge label = number of shared columns)</div>
        {graph_img_html}
      </div>
      <div class="graph-card">
        <div class="graph-title">Shared Columns Matrix (counts)</div>
        {shared_matrix_html}
        {shared_pairs_block}
      </div>
    </div>

    <div class="hr"></div>
    {''.join(sections)}
    <div class="footer">Generated report • Fill in descriptions via <code>descriptions</code>.</div>
  </div>
</body></html>"""
    out_path = Path(out_html_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path

