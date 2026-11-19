"""Tables, factor-importance figures and explorer data for the node-classification grid.

Reads results_v2/node_cls_grid_<dataset>_<target>*.jsonl (reduced design, see
run_all_node_cls_combinations.py) and writes to results_v2/report/:

  top5_<key>.csv, top10_<key>.csv     best configurations by macro-F1
  top5_table.tex                      Table-13-style LaTeX table (all datasets present)
  figures/<key>_state_correlations.{png,pdf}   Pearson r of parameter states with F1
  figures/<key>_factor_association.{png,pdf}   one-way eta^2 per factor
  factor_stats_<key>.csv              the numbers behind both figures (+ subset sizes)
  matched_pairs_<key>.csv             mean F1 change when one factor changes, all else fixed
  explorer_data.json                  every run, keyed for the interactive explorer

Which runs each statistic uses (the reduced design is not a full factorial):
  * Training-setup states (Text / SAGE / GCN / GAT) and the factors "Training Setup"
    and "Classification Type" use the GNN-architecture *slice*: the one text config
    per cell with every available flag on, crossed with all encoders, depths,
    structures and the four training setups. GCN and GAT were run only there.
  * Encoder, path length and text-config flags use the *core*: text classifier +
    SAGE over every text config. A flag is only counted where it can have an effect
    (it is n/a at d=0, and for tree + text classifier also at d=1).
  * Tree + text classifier at d=0 is not run because its input is identical to
    graph + text classifier at d=0 (tests/verify_real_data.py, section B); those
    runs are reused in the tree panel so that panel is balanced.

Usage: python analysis/node_cls_report.py [--results results_v2] [--out results_v2/report]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {
    # key: (display name, dataset, target, number of classes)
    "ea_type": ("EA Type", "eamodelset", "type", 63),
    "ea_layer": ("EA Layer", "eamodelset", "layer", 7),
    "modelset_abstract": ("Modelset Abstract", "modelset", "abstract", 2),
}
FLAGS = ["use_attributes", "use_edge_label", "use_edge_type", "use_node_type"]
FLAG_SHORT = {"use_attributes": "A", "use_edge_label": "L", "use_edge_type": "T", "use_node_type": "N"}
FLAG_NAME = {"use_attributes": "Use Attributes", "use_edge_label": "Use Edge Label",
             "use_edge_type": "Use Edge Type", "use_node_type": "Use Node Type"}
ENC_ORDER = ["bow", "tfidf", "w2v", "bert-encoder"]
ENC_LABEL = {"bow": "BoW", "tfidf": "TF-IDF", "w2v": "W2V", "bert-encoder": "BERT"}
ENC_STATE = {"bow": "Encoder: BoW", "tfidf": "Encoder: TFIDF", "w2v": "Encoder: Word2vec",
             "bert-encoder": "Encoder: BERT"}
SETUP_ORDER = ["Text Cls", "GNN Cls: SAGE", "GNN Cls: GCN", "GNN Cls: GAT"]
SETUP_STATE = {"Text Cls": "Classification: Text", "GNN Cls: SAGE": "GNN: SAGE",
               "GNN Cls: GCN": "GNN: GCN", "GNN Cls: GAT": "GNN: GAT"}

# Plot style of the paper's existing figures (generate_plots.py, removed in 218a1bb).
PLOT_FONT = dict(size=18, color="#253d63")
POS, NEG, BAR = "#2a9d8f", "#e76f51", "#4f81a3"


# --------------------------------------------------------------------------- data

def load_runs(results_dir: str, dataset: str, target: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(f"{results_dir}/node_cls_grid_{dataset}_{target}*.jsonl")):
        if f.endswith("_summary.json"):
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            c, m = r["config"], r["result"]["metrics"]
            gnn = c["gnn_model_name"] if c["classification_mode"] == "gnn" else "none"
            rows.append({
                "signature": r["signature"], "base_signature": r.get("base_signature"),
                "structure": c["structure"], "encoder_key": c["encoder_key"],
                "classification_mode": c["classification_mode"], "gnn_model_name": gnn,
                "path_depth": int(c["path_depth"]), "seed": c["seed"],
                "design_part": c.get("design_part", "core"),
                **{f: c.get(f) for f in FLAGS},
                "accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["setup"] = np.where(df.classification_mode == "text", "Text Cls",
                           "GNN Cls: " + df.gnn_model_name.str.upper())
    # One row per configuration: average over seeds if replicate runs exist.
    keys = ["structure", "encoder_key", "classification_mode", "gnn_model_name", "setup",
            "path_depth", "design_part", *FLAGS]
    agg = {k: "mean" for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]}
    agg.update(signature="first", base_signature="first", seed="count")
    df = (df.assign(**{f: df[f].astype(object) for f in FLAGS})
            .fillna({f: "n/a" for f in FLAGS})
            .groupby(keys, dropna=False, as_index=False).agg(agg)
            .rename(columns={"seed": "n_seeds"}))
    for f in FLAGS:
        df[f] = df[f].map(lambda v: None if v == "n/a" else bool(v))
    return df


def dataset_flags(df: pd.DataFrame) -> list[str]:
    """Text-config flags that are part of this dataset's design."""
    return [f for f in FLAGS if df[f].notna().any()]


def with_tree_text_d0(df: pd.DataFrame) -> pd.DataFrame:
    """Add tree + text + d=0 rows (identical input to graph + text + d=0)."""
    extra = df[(df.structure == "graph") & (df.classification_mode == "text") & (df.path_depth == 0)].copy()
    extra["structure"] = "tree"
    extra["reused_from_graph"] = True
    out = pd.concat([df.assign(reused_from_graph=False), extra], ignore_index=True)
    return out


def is_slice(row, flags) -> bool:
    return all(row[f] is None or row[f] is True for f in flags)


# --------------------------------------------------------------------------- statistics

def eta_squared(cats: pd.Series, vals: pd.Series) -> float:
    tmp = pd.DataFrame({"c": cats, "v": vals}).dropna()
    if tmp.c.nunique() < 2:
        return float("nan")
    gm = tmp.v.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for _, g in tmp.groupby("c").v)
    sst = ((tmp.v - gm) ** 2).sum()
    return float(ssb / sst) if sst else 0.0


def pearson(x: pd.Series, y: pd.Series) -> float:
    x, y = x.astype(float), y.astype(float)
    if x.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def factor_stats(df: pd.DataFrame, structure: str, flags: list[str]) -> pd.DataFrame:
    """Pearson r per parameter state and eta^2 per factor for one structural encoding."""
    work = with_tree_text_d0(df)
    work = work[work.structure == structure]
    sl = work[work.apply(lambda r: is_slice(r, flags), axis=1)]
    core = work[work.setup.isin(["Text Cls", "GNN Cls: SAGE"])]
    out = []

    def add(kind, name, value, subset, n):
        out.append({"structure": structure, "kind": kind, "name": name, "value": value,
                    "subset": subset, "n_runs": int(n)})

    # Training setup states and factors: slice
    for s in SETUP_ORDER:
        if s in set(sl.setup):
            add("pearson", SETUP_STATE[s], pearson(sl.setup == s, sl.f1_macro), "slice", len(sl))
    add("eta2", "Training Setup", eta_squared(sl.setup, sl.f1_macro), "slice", len(sl))
    add("eta2", "Classification Type", eta_squared(sl.classification_mode, sl.f1_macro), "slice", len(sl))

    # Encoder, path length, flags: core
    for e in ENC_ORDER:
        add("pearson", ENC_STATE[e], pearson(core.encoder_key == e, core.f1_macro), "core", len(core))
    add("pearson", "Path Length", pearson(core.path_depth, core.f1_macro), "core", len(core))
    add("eta2", "Text Encoder", eta_squared(core.encoder_key, core.f1_macro), "core", len(core))
    add("eta2", "Bag of Paths", eta_squared(core.path_depth.astype(str), core.f1_macro), "core", len(core))
    for f in flags:
        sub = core[core[f].notna()]
        if sub.empty or sub[f].nunique() < 2:
            continue
        add("pearson", FLAG_NAME[f], pearson(sub[f].astype(bool), sub.f1_macro), "core, flag applicable", len(sub))
        add("eta2", FLAG_NAME[f], eta_squared(sub[f].astype(str), sub.f1_macro), "core, flag applicable", len(sub))
    return pd.DataFrame(out)


def matched_pairs(df: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    """Mean F1 change when exactly one factor changes and every other factor is held fixed.

    Encoder and text-flag changes are paired within the core (text + SAGE, every text
    config). Training-setup, depth and structure changes are paired within the
    all-flags-on slice, because the set of applicable flags itself changes with depth,
    structure and classifier (flags are then 'on where applicable' on both sides).
    """
    work = with_tree_text_d0(df)
    for f in flags:
        work[f] = work[f].map(lambda v: "n/a" if v is None else v)
    core = work[work.setup.isin(["Text Cls", "GNN Cls: SAGE"])]
    sl = work[work.apply(lambda r: all(r[f] in ("n/a", True) for f in flags), axis=1)]
    rows = []

    def compare(factor, base, alt, data, keys):
        others = [k for k in keys if k != factor]
        a = data[data[factor] == base].set_index(others).f1_macro
        b = data[data[factor] == alt].set_index(others).f1_macro
        common = a.index.intersection(b.index)
        if len(common) == 0:
            return
        d = b.loc[common] - a.loc[common]
        groups = [("all", d)]
        if factor != "structure":
            lv = d.index.get_level_values("structure")
            groups += [(st, d[lv == st]) for st in ("graph", "tree")]
        for st, dd in groups:
            if len(dd):
                rows.append({"factor": factor, "from": str(base), "to": str(alt), "structure": st,
                             "pairs_from": "core" if data is core else "slice", "n_pairs": len(dd),
                             "mean_delta_f1": dd.mean(), "median_delta_f1": dd.median(),
                             "share_improved": float((dd > 0).mean()),
                             "min_delta": dd.min(), "max_delta": dd.max()})

    core_keys = ["structure", "encoder_key", "setup", "path_depth", *flags]
    slice_keys = ["structure", "encoder_key", "setup", "path_depth"]
    for e in ENC_ORDER[1:]:
        compare("encoder_key", "bow", e, core, core_keys)
    for f in flags:
        compare(f, False, True, core, core_keys)
    for s in SETUP_ORDER[1:]:
        compare("setup", "Text Cls", s, sl, slice_keys)
    for d in (1, 2, 3):
        compare("path_depth", d - 1, d, sl, slice_keys)
    compare("structure", "graph", "tree", sl, slice_keys)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- tables

def fmt_flag(v, in_design: bool) -> str:
    if not in_design or v is None:
        return "n/a"
    return "True" if v else "False"


def top_table(df: pd.DataFrame, name: str, flags: list[str], k: int) -> pd.DataFrame:
    top = df.sort_values("f1_macro", ascending=False).head(k)
    out = pd.DataFrame({
        "Dataset": name,
        "Encoder": top.encoder_key.map(ENC_LABEL),
        "Cls.": top.classification_mode.map({"text": "Text", "gnn": "GNN"}),
        "Tree": top.structure.eq("tree").map({True: "True", False: "False"}),
        "d": top.path_depth,
        "Attr.": top.use_attributes.map(lambda v: fmt_flag(v, "use_attributes" in flags)),
        "EType": top.use_edge_type.map(lambda v: fmt_flag(v, "use_edge_type" in flags)),
        "ELabel": top.use_edge_label.map(lambda v: fmt_flag(v, "use_edge_label" in flags)),
        "NType": top.use_node_type.map(lambda v: fmt_flag(v, "use_node_type" in flags)),
        "GNN": top.gnn_model_name.map(lambda g: "n/a" if g == "none" else g.upper()),
        "Accuracy": top.accuracy, "Precision_M": top.precision_macro,
        "Recall_M": top.recall_macro, "F1_M": top.f1_macro, "n_seeds": top.n_seeds,
    })
    return out.reset_index(drop=True)


def latex_table(blocks: list[tuple[str, int, pd.DataFrame]]) -> str:
    cols = ["Dataset", "Encoder", "Cls.", "Tree", "d", "Attr.", "EType", "ELabel", "NType", "GNN",
            "Accuracy", "Precision_M", "Recall_M", "F1_M"]
    head = ["Dataset", "Encoder", "Cls.", "Tree", "$d$", "Attr.", "EType", "ELabel", "NType", "GNN",
            "Accuracy", r"Precision$_M$", r"Recall$_M$", r"F1$_M$"]
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Top five configurations per dataset ranked by macro-F1. ``n/a'' marks a text-configuration "
        r"flag that is not part of the design for the dataset (Attr.\ and ELabel for EA, NType for Modelset) or "
        r"that cannot change the input at the given depth (edge and neighbour flags at $d=0$, and at $d=1$ "
        r"for tree + text classifier), and the GNN column for text-classifier runs.}",
        r"\label{tab:top5}",
        r"\small",
        r"\begin{tabular}{llllrlllllrrrr}",
        r"\toprule",
        " & ".join(head) + r" \\",
        r"\midrule",
    ]
    for i, (title, ncls, t) in enumerate(blocks):
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{14}}{{l}}{{\textit{{{title} classification ({ncls} classes)}}}} \\")
        for _, r in t.iterrows():
            cells = [str(r[c]) if c not in ("Accuracy", "Precision_M", "Recall_M", "F1_M") else f"{r[c]:.4f}"
                     for c in cols]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures

STATE_ORDER = ["Classification: Text", "GNN: SAGE", "GNN: GCN", "GNN: GAT",
               "Encoder: BoW", "Encoder: TFIDF", "Encoder: Word2vec", "Encoder: BERT", "Path Length",
               "Use Attributes", "Use Edge Label", "Use Edge Type", "Use Node Type"]
FACTOR_ORDER = ["Text Encoder", "Training Setup", "Classification Type", "Bag of Paths",
                "Use Attributes", "Use Edge Label", "Use Edge Type", "Use Node Type"]


def _two_panel(stats: pd.DataFrame, kind: str, order: list[str], title: str, ytitle: str | None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Graph Encoding", "Tree Encoding"],
                        horizontal_spacing=0.10)
    vals = []
    for col, struct in enumerate(["graph", "tree"], start=1):
        s = stats[(stats.structure == struct) & (stats.kind == kind)].set_index("name").value.dropna()
        s = s.reindex([o for o in order if o in s.index])
        vals += list(s.values)
        colors = [POS if v >= 0 else NEG for v in s.values] if kind == "pearson" else BAR
        fig.add_trace(go.Bar(x=list(s.index), y=list(s.values), marker_color=colors, showlegend=False,
                             hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"), row=1, col=col)
    fig.update_layout(template="plotly_white", title=title, height=650, width=1450,
                      margin=dict(l=85, r=35, t=105, b=190), font=PLOT_FONT, title_font_size=24)
    fig.update_annotations(font_size=20)
    fig.update_xaxes(tickangle=-45 if kind == "pearson" else -30, tickfont_size=15)
    if kind == "pearson":
        lo, hi = min(vals + [0]), max(vals + [0])
        pad = (hi - lo) * 0.08 or 0.1
        rng = [lo - pad, hi + pad]
        fig.add_hline(y=0, line_width=1, line_color="gray")
    else:
        rng = [0, max(vals) * 1.08 if vals else 1]
    for col in (1, 2):
        fig.update_yaxes(range=rng, tickfont_size=15, title_text=ytitle if col == 1 else None,
                         title_font_size=18, row=1, col=col)
    return fig


def write_figures(stats: pd.DataFrame, name: str, key: str, out: Path):
    figs = {
        "state_correlations": _two_panel(
            stats, "pearson", STATE_ORDER,
            f"{name}: Correlation of Parameter States with F1 by Structural Encoding (Pearson)", None),
        "factor_association": _two_panel(
            stats, "eta2", FACTOR_ORDER,
            f"{name}: Factor-Level Association with F1 by Structural Encoding (Eta²)", "Eta²"),
    }
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for stem, fig in figs.items():
        for ext in ("png", "pdf"):
            fig.write_image(str(out / "figures" / f"{key}_{stem}.{ext}"), scale=2 if ext == "png" else 1)


# --------------------------------------------------------------------------- explorer

def explorer_records(df: pd.DataFrame, flags: list[str]) -> list[dict]:
    recs = []
    for _, r in df.iterrows():
        recs.append({
            "s": r.structure, "e": r.encoder_key, "d": int(r.path_depth), "t": r.setup,
            "f": [None if r[f] is None else int(bool(r[f])) for f in flags],
            "f1": round(r.f1_macro, 4), "acc": round(r.accuracy, 4),
            "p": round(r.precision_macro, 4), "r": round(r.recall_macro, 4),
            "part": r.design_part, "n": int(r.n_seeds),
        })
    return recs


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--out", default="results_v2/report")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    blocks, explorer = [], {}
    for key in args.datasets:
        name, dataset, target, ncls = DATASETS[key]
        df = load_runs(args.results, dataset, target)
        if df.empty:
            print(f"{key}: no results, skipped")
            continue
        flags = dataset_flags(df)
        top_table(df, name, flags, 10).to_csv(out / f"top10_{key}.csv", index=False)
        t5 = top_table(df, name, flags, 5)
        t5.to_csv(out / f"top5_{key}.csv", index=False)
        blocks.append((name, ncls, t5))

        stats = pd.concat([factor_stats(df, s, flags) for s in ["graph", "tree"]], ignore_index=True)
        stats.to_csv(out / f"factor_stats_{key}.csv", index=False)
        matched_pairs(df, flags).to_csv(out / f"matched_pairs_{key}.csv", index=False)
        if not args.no_figures:
            write_figures(stats, name, key, out)

        explorer[key] = {
            "name": name, "classes": ncls, "flags": flags,
            "flag_short": [FLAG_SHORT[f] for f in flags], "flag_names": [FLAG_NAME[f] for f in flags],
            "runs": explorer_records(df, flags), "n_runs": int(len(df)),
            "top5": t5.drop(columns=["Dataset"]).to_dict(orient="records"),
            "stats": stats.to_dict(orient="records"),
        }
        print(f"{key}: {len(df)} configurations, flags={flags}, best F1={df.f1_macro.max():.4f}")

    (out / "top5_table.tex").write_text(latex_table(blocks))
    (out / "explorer_data.json").write_text(json.dumps(explorer, separators=(",", ":"),
                                                       default=lambda o: None if isinstance(o, float) and math.isnan(o) else o))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
