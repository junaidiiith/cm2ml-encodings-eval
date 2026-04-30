import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


df_ea_type = pd.read_csv("results/analysis_rigorous/evaluation_package/eamodelset_type/flat_results.csv")
df_ea_type.drop(
    columns=[
        'dataset', 'encoder_name', 'target_attr', 'label_attr', 'attributes_attr',
        'use_attributes', 'use_edge_label', 'graph_encoder_family',
        'structural_encoding_label', 'duplicate_count', 'num_samples', 'num_classes'
    ],
    inplace=True,
    errors="ignore"
)

df_ea_layer = pd.read_csv("results/analysis_rigorous/evaluation_package/eamodelset_layer/flat_results.csv")
df_ea_layer.drop(
    columns=[
        'dataset', 'encoder_name', 'target_attr', 'label_attr', 'attributes_attr',
        'use_attributes', 'use_edge_label', 'graph_encoder_family',
        'structural_encoding_label', 'duplicate_count', 'num_samples', 'num_classes'
    ],
    inplace=True,
    errors="ignore"
)

# modelset_abstract
df_modelset = pd.read_csv("results/analysis_rigorous/evaluation_package/modelset_abstract/flat_results.csv")
df_modelset.drop(
    columns=[
        'dataset', 'encoder_name', 'target_attr', 'label_attr', 'attributes_attr',
        'graph_encoder_family', 'structural_encoding_label',
        'duplicate_count', 'num_samples', 'num_classes'
    ],
    inplace=True,
    errors="ignore"
)

DATASETS = {
    "EA Type": df_ea_type,
    "EA Layer": df_ea_layer,
    "Modelset Abstract": df_modelset,
}



ALL_DATASETS = {
    "EA Layer": df_ea_layer,
    "EA Type": df_ea_type,
    "Modelset Abstract": df_modelset,
}

DEFAULTS = {
    "tree": False,
    "encoder_key": "bow",
    "use_attributes": False,
    "use_edge_label": False,
    "use_edge_type": False,
    "path_depth": 0,
    "classification_mode": "text",
    "gnn_model_name": "sage",
}

COLOR_SEQ = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
]

FACTOR_ORDER = [
    "Text Encoder",
    "Training Setup",
    "Classification Type",
    "Bag of Paths",
    "Use Edge Type",
    "Use Attributes",
    "Use Edge Label",
]

PARAMETER_STATE_ORDER = [
    "Classification: Text",
    "GNN: SAGE",
    "GNN",
    "GNN: GCN",
    "GNN: GAT",
    "Encoder: BoW",
    "Encoder: TFIDF",
    "Encoder: Word2vec",
    "Encoder: BERT",
    "Path Length",
    "Use Edge Type",
]

PLOT_FONT = dict(size=18, color="#253d63")
TITLE_FONT_SIZE = 24
SUBTITLE_FONT_SIZE = 20
AXIS_TITLE_FONT_SIZE = 18
TICK_FONT_SIZE = 15
MIN_FACTOR_ETA_SQUARED = 1e-3


def _normalize_plot_df(df):
    work = df.copy()

    required = [
        "encoder_key", "classification_mode", "tree",
        "use_edge_type", "path_depth", "gnn_model_name", "f1_macro"
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "use_attributes" not in work.columns:
        work["use_attributes"] = False
    if "use_edge_label" not in work.columns:
        work["use_edge_label"] = False

    work["tree"] = work["tree"].astype(bool)
    work["use_attributes"] = work["use_attributes"].astype(bool)
    work["use_edge_label"] = work["use_edge_label"].astype(bool)
    work["use_edge_type"] = work["use_edge_type"].astype(bool)
    work["path_depth"] = work["path_depth"].astype(int)
    work["classification_mode"] = work["classification_mode"].astype(str)
    work["encoder_key"] = work["encoder_key"].astype(str)

    def normalize_gnn(row):
        if row["classification_mode"] == "text":
            return "none"
        val = row.get("gnn_model_name", "sage")
        if pd.isna(val):
            return "sage"
        val = str(val).lower()
        if val == "unknown":
            return "sage"
        return val

    work["gnn_model_name"] = work.apply(normalize_gnn, axis=1)
    return work


def _pretty_encoder_label(value):
    mapping = {
        "bow": "Encoder: BoW",
        "tfidf": "Encoder: TFIDF",
        "w2v": "Encoder: Word2vec",
        "bert-encoder": "Encoder: BERT",
        "bert": "Encoder: BERT",
    }
    return mapping.get(value, f"Encoder: {value}")


def _pretty_gnn_label(value):
    mapping = {
        "sage": "GNN: SAGE",
        "gcn": "GNN: GCN",
        "gat": "GNN: GAT",
        "unknown": "GNN: SAGE",
    }
    return mapping.get(value, f"GNN: {str(value).upper()}")


def _ordered_existing(labels, order):
    seen = set(labels)
    ordered = [label for label in order if label in seen]
    ordered.extend(label for label in labels if label not in order)
    return ordered


def _positive_axis_range(fig, padding=1.08):
    values = []
    for trace in fig.data:
        if trace.y is not None:
            values.extend(v for v in trace.y if pd.notna(v))

    if not values:
        return None

    upper = max(values)
    if upper <= 0:
        return None
    return [0, upper * padding]


def _signed_axis_range(fig, padding=1.08):
    values = []
    for trace in fig.data:
        if trace.y is not None:
            values.extend(v for v in trace.y if pd.notna(v))

    if not values:
        return None

    lower = min(values)
    upper = max(values)
    span = upper - lower
    if span == 0:
        pad = abs(upper) * (padding - 1) or 0.1
        return [lower - pad, upper + pad]

    pad = span * (padding - 1)
    return [lower - pad, upper + pad]


def _sync_yaxes(fig, axis_range, title_text=None):
    yaxis_kwargs = {
        "range": axis_range,
        "title_font_size": AXIS_TITLE_FONT_SIZE,
        "tickfont_size": TICK_FONT_SIZE,
    }
    if title_text is not None:
        yaxis_kwargs["title_text"] = title_text

    fig.update_yaxes(**yaxis_kwargs, row=1, col=1)
    fig.update_yaxes(
        **yaxis_kwargs,
        matches="y",
        showticklabels=True,
        row=1,
        col=2,
    )


def _keep_all_tree_gnn_models(dataset_name):
    return dataset_name == "EA Layer"


def _filter_tree_gnn_runs(work, tree_value, keep_all_tree_gnn_models):
    if not tree_value or keep_all_tree_gnn_models:
        return work

    is_non_sage_gnn = (work["classification_mode"] == "gnn") & (work["gnn_model_name"] != "sage")
    return work[~is_non_sage_gnn].copy()


def _state_matrix(df, tree_value, keep_all_tree_gnn_models=False):
    work = _normalize_plot_df(df)
    work = work[work["tree"] == tree_value].copy()
    work = _filter_tree_gnn_runs(work, tree_value, keep_all_tree_gnn_models)
    if work.empty:
        return pd.DataFrame()

    parts = []

    # Encoder states
    enc_labels = work["encoder_key"].map(_pretty_encoder_label)
    enc = pd.get_dummies(enc_labels, dtype=int)
    parts.append(enc)

    # Only keep Classification: Text
    cls_text = (work["classification_mode"] == "text").astype(int)
    cls_text.name = "Classification: Text"
    parts.append(cls_text.to_frame())

    # GNN states only for actual GNN runs
    if tree_value and not keep_all_tree_gnn_models:
        gnn_labels = work.apply(
            lambda r: "GNN" if r["classification_mode"] == "gnn" and r["gnn_model_name"] == "sage" else None,
            axis=1
        )
    else:
        gnn_labels = work.apply(
            lambda r: _pretty_gnn_label(r["gnn_model_name"]) if r["classification_mode"] == "gnn" else None,
            axis=1
        )
    gnn = pd.get_dummies(gnn_labels, dtype=int)
    if not gnn.empty:
        parts.append(gnn)

    # Numeric states
    path_part = work[["path_depth"]].rename(columns={"path_depth": "Path Length"})
    parts.append(path_part)

    edge_part = work[["use_edge_type"]].astype(int).rename(columns={"use_edge_type": "Use Edge Type"})
    parts.append(edge_part)

    X = pd.concat(parts, axis=1)
    X["f1_macro"] = work["f1_macro"].astype(float)

    # Drop constant columns
    constant_cols = [c for c in X.columns if c != "f1_macro" and X[c].nunique() <= 1]
    X = X.drop(columns=constant_cols, errors="ignore")

    ordered_cols = _ordered_existing([c for c in X.columns if c != "f1_macro"], PARAMETER_STATE_ORDER)
    X = X[ordered_cols + ["f1_macro"]]

    return X


def plot_parameter_state_correlations_by_tree(df, dataset_name, method="pearson"):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Graph Encoding", "Tree Encoding"],
        horizontal_spacing=0.10,
    )

    for idx, tree_value in enumerate([False, True], start=1):
        X = _state_matrix(df, tree_value, keep_all_tree_gnn_models=_keep_all_tree_gnn_models(dataset_name))
        if X.empty:
            continue

        corr = X.corr(method=method)["f1_macro"].drop("f1_macro")
        corr = corr.reindex(_ordered_existing(corr.index, PARAMETER_STATE_ORDER))

        colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in corr.values]

        fig.add_trace(
            go.Bar(
                x=list(corr.index),
                y=list(corr.values),
                marker_color=colors,
                hovertemplate="%{x}<br>corr with F1: %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=idx,
        )

    fig.update_layout(
        template="plotly_white",
        title=f"{dataset_name}: Correlation of Parameter States with F1 by Structural Encoding ({method.title()})",
        height=650,
        width=1450,
        margin=dict(l=85, r=35, t=105, b=175),
        font=PLOT_FONT,
        title_font_size=TITLE_FONT_SIZE,
    )
    fig.update_annotations(font_size=SUBTITLE_FONT_SIZE)
    fig.update_xaxes(
        tickangle=-45,
        tickfont_size=TICK_FONT_SIZE,
    )
    _sync_yaxes(fig, _signed_axis_range(fig))
    fig.add_hline(y=0, line_width=1, line_color="gray")
    return fig


def _correlation_ratio(categories, values):
    tmp = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if tmp.empty:
        return np.nan

    grand_mean = tmp["val"].mean()
    grouped = tmp.groupby("cat")["val"]

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in grouped)
    ss_total = ((tmp["val"] - grand_mean) ** 2).sum()

    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


def _factor_scores(df, tree_value, keep_all_tree_gnn_models=False):
    work = _normalize_plot_df(df)
    work = work[work["tree"] == tree_value].copy()
    work = _filter_tree_gnn_runs(work, tree_value, keep_all_tree_gnn_models)
    if work.empty:
        return pd.DataFrame(columns=["factor", "eta_squared"])

    factors = {
        "Text Encoder": work["encoder_key"],
        "Training Setup": work.apply(
            lambda r: "Text Cls"
            if r["classification_mode"] == "text"
            else (
                f"GNN Cls: {r['gnn_model_name'].upper()}"
                if not tree_value or keep_all_tree_gnn_models
                else "GNN Cls"
            ),
            axis=1
        ),
        "Classification Type": work["classification_mode"].map({"text": "Text", "gnn": "GNN"}),
        "Bag of Paths": work["path_depth"].astype(str),
        "Use Edge Type": work["use_edge_type"].map({False: "False", True: "True"}),
    }

    if "use_attributes" in df.columns:
        factors["Use Attributes"] = work["use_attributes"].map({False: "False", True: "True"})
    if "use_edge_label" in df.columns:
        factors["Use Edge Label"] = work["use_edge_label"].map({False: "False", True: "True"})

    rows = []
    for factor_name, cats in factors.items():
        if cats.dropna().nunique() <= 1:
            continue

        eta_squared = _correlation_ratio(cats, work["f1_macro"])
        if pd.isna(eta_squared) or eta_squared < MIN_FACTOR_ETA_SQUARED:
            continue

        rows.append({
            "factor": factor_name,
            "eta_squared": eta_squared
        })

    out = pd.DataFrame(rows)
    out["factor"] = pd.Categorical(out["factor"], categories=FACTOR_ORDER, ordered=True)
    out = out.sort_values("factor")
    out["factor"] = out["factor"].astype(str)
    return out


def plot_factor_level_association_by_tree(df, dataset_name):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Graph Encoding", "Tree Encoding"],
        horizontal_spacing=0.10,
    )

    for idx, tree_value in enumerate([False, True], start=1):
        eff = _factor_scores(df, tree_value, keep_all_tree_gnn_models=_keep_all_tree_gnn_models(dataset_name))

        fig.add_trace(
            go.Bar(
                x=eff["factor"],
                y=eff["eta_squared"],
                marker_color="#4f81a3",
                hovertemplate="%{x}<br>Eta²: %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=idx,
        )

    fig.update_layout(
        template="plotly_white",
        title=f"{dataset_name}: Factor-Level Association with F1 by Structural Encoding (Eta²)",
        height=650,
        width=1450,
        margin=dict(l=85, r=35, t=105, b=155),
        font=PLOT_FONT,
        title_font_size=TITLE_FONT_SIZE,
    )
    fig.update_annotations(font_size=SUBTITLE_FONT_SIZE)
    fig.update_xaxes(
        tickangle=-30,
        tickfont_size=TICK_FONT_SIZE,
    )
    _sync_yaxes(fig, _positive_axis_range(fig), title_text="Eta²")
    return fig


if __name__ == "__main__":
    name_map = {
        "EA Layer": "ea_layer",
        "EA Type": "ea_type",
        "Modelset Abstract": "modelset_abstract",
    }
    for name, df_ in ALL_DATASETS.items():
        ds_name = name_map.get(name, name.lower().replace(" ", "_"))
        plot_parameter_state_correlations_by_tree(df_, name, method="pearson").write_image(
            f"results/plots/{ds_name}_state_correlations_pub.png"
        )
        plot_factor_level_association_by_tree(df_, name).write_image(
            f"results/plots/{ds_name}_factor_association_pub.png"
        )
