"""Tree vs graph structural encoding, compared like-for-like.

Every tree run is paired with the graph run that has the same requested
configuration (encoder, training setup, path depth, text-config flags), and the
gap delta = F1(tree) - F1(graph) is analysed conditionally on every other factor.

Pairing uses the *requested* configuration: a flag that cannot change the input
at a given depth (e.g. edge flags for tree + text classifier at d=1) is mapped to
the run that was actually executed, exactly as in the explorer. Tree + text
classifier at d=0 is excluded: its input is identical to graph by construction.
GCN/GAT pairs exist only on the all-flags-on slice (reduced design).

Outputs (results_v2/report/structure/):
  pairs_<key>.csv          every matched pair
  conditional_<key>.csv    mean/median delta, share tree better, Wilcoxon p, per factor level
  cells_<key>.csv          delta per two-factor cell (setup x depth, encoder x setup)
  moderators_<key>.csv     eta^2 of each factor on delta: which factors change the gap
  best_<key>.csv           best tree vs best graph, overall and per level ("tuned" comparison)
  reach_<key>.csv          text classifier at equal reach: tree d+1 vs graph d
  figures/structure_gap.{png,pdf}
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import node_cls_report as R  # noqa: E402

OUT = Path("results_v2/report/structure")


def canonical_flags(flags, structure, setup, d, names):
    text = setup == "Text Cls"
    min_depth = 2 if (structure == "tree" and text) else 1
    return tuple(v if (n == "use_attributes" or d >= min_depth) else None for n, v in zip(names, flags))


def make_pairs(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    idx = {}
    for _, r in df.iterrows():
        idx[(r.structure, r.encoder_key, r.setup, r.path_depth, tuple(r[n] for n in names))] = r
    rows = []
    for enc, setup, d in itertools.product(R.ENC_ORDER, R.SETUP_ORDER, range(4)):
        if setup == "Text Cls" and d == 0:
            continue  # identical input in both structures
        for flags in itertools.product([False, True], repeat=len(names)):
            g = idx.get(("graph", enc, setup, d, canonical_flags(flags, "graph", setup, d, names)))
            t = idx.get(("tree", enc, setup, d, canonical_flags(flags, "tree", setup, d, names)))
            if g is None or t is None:
                continue
            rows.append({"encoder": R.ENC_LABEL[enc], "setup": setup, "path_depth": d,
                         **{R.FLAG_NAME[n]: f for n, f in zip(names, flags)},
                         "f1_graph": g.f1_macro, "f1_tree": t.f1_macro, "delta": t.f1_macro - g.f1_macro})
    # A requested flag combination that maps to the same pair of runs is one comparison, not several.
    p = pd.DataFrame(rows)
    return p.drop_duplicates(subset=["encoder", "setup", "path_depth", "f1_graph", "f1_tree"]).reset_index(drop=True)


def summarise(d: pd.Series) -> dict:
    out = {"n_pairs": len(d), "mean_delta": d.mean(), "median_delta": d.median(),
           "share_tree_better": float((d > 0).mean()), "min_delta": d.min(), "max_delta": d.max()}
    nz = d[d != 0]
    out["wilcoxon_p"] = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else np.nan
    return out


def eta2(cats, vals):
    tmp = pd.DataFrame({"c": cats, "v": vals}).dropna()
    gm = tmp.v.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for _, g in tmp.groupby("c").v)
    sst = ((tmp.v - gm) ** 2).sum()
    return ssb / sst if sst else 0.0


def analyse(key: str):
    name, ds, tg, _ = R.DATASETS[key]
    df = R.load_runs("results_v2", ds, tg)
    names = R.dataset_flags(df)
    flag_cols = [R.FLAG_NAME[n] for n in names]
    pairs = make_pairs(df, names)
    pairs.to_csv(OUT / f"pairs_{key}.csv", index=False)

    # Flags only count where they can change the input in *both* structures.
    def applicable(p, col):
        n = [k for k, v in R.FLAG_NAME.items() if v == col][0]
        if n == "use_attributes":
            return p.index == p.index
        return (p.path_depth >= 2) | ((p.path_depth >= 1) & (p.setup != "Text Cls"))

    cond = [{"factor": "all", "level": "all", **summarise(pairs.delta)}]
    for fac in ["setup", "encoder", "path_depth", *flag_cols]:
        sub = pairs[applicable(pairs, fac)] if fac in flag_cols else pairs
        for lvl, g in sub.groupby(fac):
            cond.append({"factor": fac, "level": str(lvl), **summarise(g.delta)})
    cond = pd.DataFrame(cond)
    cond.to_csv(OUT / f"conditional_{key}.csv", index=False)

    cells = []
    for a, b in [("setup", "path_depth"), ("encoder", "setup"), ("encoder", "path_depth")]:
        for (la, lb), g in pairs.groupby([a, b]):
            cells.append({"rows": a, "cols": b, "row": str(la), "col": str(lb), "n_pairs": len(g),
                          "mean_delta": g.delta.mean(), "share_tree_better": float((g.delta > 0).mean())})
    cells = pd.DataFrame(cells)
    cells.to_csv(OUT / f"cells_{key}.csv", index=False)

    core = pairs[pairs.setup.isin(["Text Cls", "GNN Cls: SAGE"])]
    mods = []
    for fac in ["setup", "encoder", "path_depth", *flag_cols]:
        sub = core[applicable(core, fac)] if fac in flag_cols else (pairs if fac == "setup" else core)
        mods.append({"factor": fac, "eta2_on_delta": eta2(sub[fac].astype(str), sub.delta),
                     "pairs_used": "all pairs" if fac == "setup" else "text + SAGE pairs", "n_pairs": len(sub)})
    mods = pd.DataFrame(mods).sort_values("eta2_on_delta", ascending=False)
    mods.to_csv(OUT / f"moderators_{key}.csv", index=False)

    # Tuned comparison: best achievable F1 per structure, overall and within each level.
    runs = R.with_tree_text_d0(df)
    runs = runs[~runs.reused_from_graph]
    runs["encoder"] = runs.encoder_key.map(R.ENC_LABEL)
    best = []
    for fac in [None, "setup", "encoder", "path_depth"]:
        groups = [("all", runs)] if fac is None else list(runs.groupby(fac))
        for lvl, g in groups:
            bg = g[g.structure == "graph"].sort_values("f1_macro").tail(1)
            bt = g[g.structure == "tree"].sort_values("f1_macro").tail(1)
            if bg.empty or bt.empty:
                continue
            best.append({"factor": fac or "all", "level": str(lvl),
                         "best_graph": bg.f1_macro.iat[0], "best_tree": bt.f1_macro.iat[0],
                         "gap": bt.f1_macro.iat[0] - bg.f1_macro.iat[0],
                         "best_graph_cfg": describe(bg.iloc[0], names), "best_tree_cfg": describe(bt.iloc[0], names)})
    best = pd.DataFrame(best)
    best.to_csv(OUT / f"best_{key}.csv", index=False)

    # Equal reach for the text classifier: a tree OBJ node needs two hops (OBJ -> ASSOCS -> REF)
    # to reach what a graph node reaches in one, so compare tree d+1 with graph d.
    reach = []
    tx = df[df.setup == "Text Cls"]
    for enc in R.ENC_ORDER:
        for d in range(3):
            for flags in itertools.product([False, True], repeat=len(names)):
                g = tx[(tx.structure == "graph") & (tx.encoder_key == enc) & (tx.path_depth == d)]
                t = tx[(tx.structure == "tree") & (tx.encoder_key == enc) & (tx.path_depth == d + 1)]
                gf = canonical_flags(flags, "graph", "Text Cls", d, names)
                tf = canonical_flags(flags, "tree", "Text Cls", d + 1, names)
                g = g[np.logical_and.reduce([g[n].apply(lambda v, x=x: v == x if x is not None else v is None) for n, x in zip(names, gf)])]
                t = t[np.logical_and.reduce([t[n].apply(lambda v, x=x: v == x if x is not None else v is None) for n, x in zip(names, tf)])]
                if len(g) and len(t):
                    reach.append({"encoder": R.ENC_LABEL[enc], "graph_d": d, "tree_d": d + 1,
                                  "f1_graph": g.f1_macro.iat[0], "f1_tree": t.f1_macro.iat[0]})
    reach = pd.DataFrame(reach).drop_duplicates()
    reach["delta"] = reach.f1_tree - reach.f1_graph
    reach.to_csv(OUT / f"reach_{key}.csv", index=False)
    return name, pairs, cond, cells, mods, best, reach


def describe(r, names) -> str:
    fl = "-".join(f"{R.FLAG_SHORT[n]}{'x' if r[n] is None else int(bool(r[n]))}" for n in names)
    return f"{R.ENC_LABEL[r.encoder_key]} | {r.setup} | d={r.path_depth} | {fl}"


def figure(results):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=len(results), subplot_titles=[r[0] for r in results], horizontal_spacing=0.07)
    zmax = max(r[3][(r[3].rows == "setup") & (r[3].cols == "path_depth")].mean_delta.abs().max() for r in results)
    for i, (name, _, _, cells, *_rest) in enumerate(results, start=1):
        c = cells[(cells.rows == "setup") & (cells.cols == "path_depth")]
        m = c.pivot(index="row", columns="col", values="mean_delta").reindex(R.SETUP_ORDER)
        fig.add_trace(go.Heatmap(
            z=m.values, x=[f"d={x}" for x in m.columns], y=[s.replace("GNN Cls: ", "") for s in m.index],
            zmin=-zmax, zmax=zmax, colorscale=[[0, "#c0504d"], [0.5, "#f2f2f2"], [1, "#2f7f9e"]],
            text=[[("" if np.isnan(v) else f"{v:+.3f}") for v in row] for row in m.values],
            texttemplate="%{text}", textfont=dict(size=15), showscale=(i == len(results)),
            colorbar=dict(title="F1 tree − graph"), hovertemplate="%{y}, %{x}: %{z:+.3f}<extra></extra>"),
            row=1, col=i)
        fig.update_yaxes(autorange="reversed", row=1, col=i)
    fig.update_layout(template="plotly_white", height=430, width=1450, font=dict(size=16, color="#253d63"),
                      title="Tree minus graph macro-F1, matched configurations (mean over encoders and text configs)",
                      margin=dict(l=70, r=30, t=100, b=50))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT / "figures" / "structure_gap.png"), scale=2)
    fig.write_image(str(OUT / "figures" / "structure_gap.pdf"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = [analyse(k) for k in R.DATASETS]
    figure(results)
    pd.set_option("display.width", 250)
    for name, pairs, cond, cells, mods, best, reach in results:
        print(f"\n=================== {name}")
        print(cond.round(3).to_string(index=False))
        print(mods.round(3).to_string(index=False))
        print(best.round(4).to_string(index=False))
        for (a, b), g in cells.groupby(["rows", "cols"], sort=False):
            print(f"-- mean delta {a} x {b}")
            print(g.pivot(index="row", columns="col", values="mean_delta").round(3).to_string())
        print("-- equal reach (tree d+1 vs graph d), text classifier:")
        print(reach.groupby(["graph_d", "tree_d"]).delta.agg(["mean", "count", lambda s: (s > 0).mean()]).round(3).to_string())


if __name__ == "__main__":
    main()
