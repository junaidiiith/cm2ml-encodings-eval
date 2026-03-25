from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
CONFIG_KEYS = [
    "target_attr",
    "label_attr",
    "attributes_attr",
    "encoder_key",
    "encoder_name",
    "classification_mode",
    "tree",
    "path_depth",
    "use_attributes",
    "use_edge_type",
    "use_edge_label",
    "gnn_model_name",
]
FACTOR_ORDER = {
    "classification_mode": ["text", "gnn"],
    "encoder_key": ["bow", "tfidf", "bert-encoder", "w2v"],
    "use_attributes": [False, True],
    "use_edge_type": [False, True],
    "use_edge_label": [False, True],
    "path_depth": [0, 1, 2, 3],
    "tree": [False, True],
    "gnn_model_name": ["gat", "gcn", "sage"],
}
FACTOR_LABELS = {
    "classification_mode": "classification mode",
    "encoder_key": "text encoder",
    "use_attributes": "use_attributes",
    "use_edge_type": "use_edge_type",
    "use_edge_label": "use_edge_label",
    "path_depth": "path length",
    "tree": "tree encoding",
    "gnn_model_name": "GNN model",
}
DATASET_FILES = {
    "eamodelset_layer": Path("results/node_cls_grid_eamodelset_layer.jsonl"),
    "eamodelset_type": Path("results/node_cls_grid_eamodelset_type.jsonl"),
    "modelset_abstract": Path("results/node_cls_grid_modelset_abstract.jsonl"),
}
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42


@dataclass
class ExperimentRow:
    dataset: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    result: Dict[str, Any]
    duplicate_count: int = 1

    def to_flat_dict(self) -> Dict[str, Any]:
        data = {key: self.config.get(key) for key in CONFIG_KEYS}
        data["dataset"] = self.dataset
        data["graph_encoder_family"] = self.config["graph_encoder_family"]
        data["structural_encoding_label"] = self.config["structural_encoding_label"]
        data["duplicate_count"] = self.duplicate_count
        data["num_samples"] = self.result.get("num_samples")
        data["num_classes"] = self.result.get("num_classes")
        for metric in METRICS:
            data[metric] = self.metrics.get(metric)
        return data


def canonical_signature(config: Dict[str, Any]) -> str:
    return "|".join(str(config.get(key)) for key in CONFIG_KEYS)


def normalize_gnn_model_name(config: Dict[str, Any]) -> str:
    mode = config.get("classification_mode")
    raw = config.get("gnn_model_name")
    if mode == "text":
        return "none"
    if raw in (None, ""):
        return "unknown"
    return str(raw)


def structural_encoding_label(tree: bool, path_depth: int) -> str:
    if path_depth == 0:
        return "tree_encoder" if tree else "graph_encoder"
    prefix = "tree_encoder" if tree else "graph_encoder"
    return f"{prefix}+bag_of_paths(d={path_depth})"


def graph_encoder_family(tree: bool, path_depth: int) -> str:
    if path_depth == 0:
        return "tree_encoder" if tree else "graph_encoder"
    prefix = "tree_encoder" if tree else "graph_encoder"
    return f"{prefix}_bop_d{path_depth}"


def load_rows(dataset_key: str, jsonl_path: Path) -> tuple[list[ExperimentRow], dict[str, int], int]:
    successful: list[dict[str, Any]] = []
    duplicate_counter: dict[str, int] = defaultdict(int)
    total_rows = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            total_rows += 1
            record = json.loads(raw_line)
            if record.get("status") != "ok":
                continue
            config = dict(record["config"])
            config["gnn_model_name"] = normalize_gnn_model_name(config)
            config["path_depth"] = int(config.get("path_depth", 0))
            config["tree"] = bool(config.get("tree", False))
            config["encoder_key"] = config.get("encoder_key") or config.get("encoder_name")
            config["graph_encoder_family"] = graph_encoder_family(config["tree"], config["path_depth"])
            config["structural_encoding_label"] = structural_encoding_label(config["tree"], config["path_depth"])
            signature = canonical_signature(config)
            duplicate_counter[signature] += 1
            successful.append(
                {
                    "signature": signature,
                    "config": config,
                    "metrics": dict(record["result"]["metrics"]),
                    "result": dict(record["result"]),
                }
            )

    deduped: dict[str, dict[str, Any]] = {}
    for row in successful:
        deduped[row["signature"]] = row

    final_rows = []
    for signature, row in deduped.items():
        final_rows.append(
            ExperimentRow(
                dataset=dataset_key,
                config=row["config"],
                metrics=row["metrics"],
                result=row["result"],
                duplicate_count=duplicate_counter[signature],
            )
        )
    final_rows.sort(key=lambda item: canonical_signature(item.config))
    return final_rows, dict(duplicate_counter), total_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    lower = sorted_values[low]
    upper = sorted_values[high]
    weight = position - low
    return lower + (upper - lower) * weight


def bootstrap_mean_ci(values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(BOOTSTRAP_SEED)
    resampled_means = []
    n = len(values)
    for _ in range(samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(mean(sample))
    resampled_means.sort()
    return (percentile(resampled_means, 0.025), percentile(resampled_means, 0.975))


def two_sided_sign_test_pvalue(values: Sequence[float]) -> float:
    positives = sum(1 for value in values if value > 0)
    negatives = sum(1 for value in values if value < 0)
    n = positives + negatives
    if n == 0:
        return 1.0
    k = min(positives, negatives)
    cumulative = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * cumulative)


def holm_adjust(rows: list[dict[str, Any]], pvalue_key: str, out_key: str) -> None:
    indexed = [(idx, row[pvalue_key]) for idx, row in enumerate(rows) if row.get(pvalue_key) is not None]
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 0.0
    for rank, (idx, pval) in enumerate(indexed, start=1):
        adjusted = min(1.0, (m - rank + 1) * pval)
        running = max(running, adjusted)
        rows[idx][out_key] = running


def values_for_factor(rows: Sequence[ExperimentRow], factor: str) -> list[Any]:
    values = []
    seen = set()
    order = FACTOR_ORDER.get(factor, [])
    for candidate in order:
        if any(row.config.get(factor) == candidate for row in rows):
            values.append(candidate)
            seen.add(candidate)
    for row in rows:
        value = row.config.get(factor)
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def all_unordered_pairs(values: Sequence[Any], factor: str) -> list[tuple[Any, Any]]:
    ordered = values_for_display(values, factor)
    pairs = []
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            pairs.append((left, right))
    return pairs


def comparison_pairs(values: Sequence[Any], factor: str) -> list[tuple[Any, Any]]:
    ordered = values_for_display(values, factor)
    if factor in {"encoder_key", "gnn_model_name"}:
        return [(left, right) for left in ordered for right in ordered if left != right]
    return all_unordered_pairs(ordered, factor)


def values_for_display(values: Sequence[Any], factor: str) -> list[Any]:
    order = FACTOR_ORDER.get(factor)
    if not order:
        return list(values)
    present = set(values)
    return [candidate for candidate in order if candidate in present]


def matching_keys_for_factor(factor: str) -> list[str]:
    if factor == "classification_mode":
        return [
            "target_attr",
            "label_attr",
            "attributes_attr",
            "encoder_key",
            "tree",
            "path_depth",
            "use_attributes",
            "use_edge_type",
            "use_edge_label",
        ]
    if factor in {"encoder_key", "encoder_name"}:
        return [key for key in CONFIG_KEYS if key not in {"encoder_key", "encoder_name"}]
    return [key for key in CONFIG_KEYS if key != factor]


def subset_for_factor(rows: Sequence[ExperimentRow], factor: str) -> list[ExperimentRow]:
    if factor == "gnn_model_name":
        return [
            row
            for row in rows
            if row.config.get("classification_mode") == "gnn" and row.config.get("gnn_model_name") != "unknown"
        ]
    return list(rows)


def group_rows(rows: Sequence[ExperimentRow], keys: Sequence[str]) -> dict[tuple[Any, ...], list[ExperimentRow]]:
    grouped: dict[tuple[Any, ...], list[ExperimentRow]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.config.get(key) for key in keys)].append(row)
    return grouped


def summarize_delta_rows(
    dataset_key: str,
    factor: str,
    from_value: Any,
    to_value: Any,
    pairs: Sequence[tuple[ExperimentRow, ExperimentRow, dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": dataset_key,
        "factor": factor,
        "factor_label": FACTOR_LABELS.get(factor, factor),
        "from_value": from_value,
        "to_value": to_value,
        "pair_count": len(pairs),
    }
    for metric in METRICS:
        deltas = [pair[1].metrics[metric] - pair[0].metrics[metric] for pair in pairs]
        ci_low, ci_high = bootstrap_mean_ci(deltas)
        summary[f"{metric}_delta_mean"] = mean(deltas)
        summary[f"{metric}_delta_median"] = median(deltas)
        summary[f"{metric}_delta_std"] = stdev(deltas)
        summary[f"{metric}_delta_ci_low"] = ci_low
        summary[f"{metric}_delta_ci_high"] = ci_high
        summary[f"{metric}_win_rate"] = (
            sum(1 for value in deltas if value > 0) / len(deltas) if deltas else float("nan")
        )
        summary[f"{metric}_sign_test_pvalue"] = two_sided_sign_test_pvalue(deltas)
        summary[f"{metric}_cohen_dz"] = mean(deltas) / stdev(deltas) if len(deltas) > 1 and stdev(deltas) else 0.0

    if factor == "classification_mode":
        gnn_models = sorted({pair[2]["gnn_model_name"] for pair in pairs})
        summary["gnn_models_in_pairs"] = ",".join(gnn_models)
    return summary


def build_factor_pair_summaries(rows: Sequence[ExperimentRow], factor: str) -> list[dict[str, Any]]:
    subset = subset_for_factor(rows, factor)
    grouped = group_rows(subset, matching_keys_for_factor(factor))
    values = values_for_factor(subset, factor)
    pair_summaries = []

    for from_value, to_value in comparison_pairs(values, factor):
        matched_pairs: list[tuple[ExperimentRow, ExperimentRow, dict[str, Any]]] = []
        for _, group in grouped.items():
            left_rows = [row for row in group if row.config.get(factor) == from_value]
            right_rows = [row for row in group if row.config.get(factor) == to_value]
            if not left_rows or not right_rows:
                continue
            if factor == "classification_mode":
                for left in left_rows:
                    for right in right_rows:
                        matched_pairs.append((left, right, {"gnn_model_name": right.config.get("gnn_model_name")}))
            else:
                left_rows = sorted(left_rows, key=lambda row: canonical_signature(row.config))
                right_rows = sorted(right_rows, key=lambda row: canonical_signature(row.config))
                for left, right in zip(left_rows, right_rows):
                    matched_pairs.append((left, right, {}))
        if matched_pairs:
            pair_summaries.append(summarize_delta_rows(rows[0].dataset, factor, from_value, to_value, matched_pairs))

    holm_adjust(pair_summaries, "f1_macro_sign_test_pvalue", "f1_macro_sign_test_pvalue_holm")
    return pair_summaries


def build_graph_encoder_combo_summary(rows: Sequence[ExperimentRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ExperimentRow]] = defaultdict(list)
    for row in rows:
        key = (row.config["encoder_key"], row.config["graph_encoder_family"])
        grouped[key].append(row)
    summaries = []
    for (encoder_key, structural_family), group in sorted(grouped.items()):
        summaries.append(
            {
                "dataset": rows[0].dataset,
                "encoder_key": encoder_key,
                "graph_encoder_family": structural_family,
                "count": len(group),
                "mean_accuracy": mean([row.metrics["accuracy"] for row in group]),
                "mean_precision_macro": mean([row.metrics["precision_macro"] for row in group]),
                "mean_recall_macro": mean([row.metrics["recall_macro"] for row in group]),
                "mean_f1_macro": mean([row.metrics["f1_macro"] for row in group]),
                "median_f1_macro": median([row.metrics["f1_macro"] for row in group]),
            }
        )
    summaries.sort(key=lambda item: item["mean_f1_macro"], reverse=True)
    return summaries


def build_structure_family_summary(rows: Sequence[ExperimentRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ExperimentRow]] = defaultdict(list)
    for row in rows:
        grouped[row.config["graph_encoder_family"]].append(row)
    summaries = []
    for family, group in sorted(grouped.items()):
        summaries.append(
            {
                "dataset": rows[0].dataset,
                "graph_encoder_family": family,
                "count": len(group),
                "mean_f1_macro": mean([row.metrics["f1_macro"] for row in group]),
                "median_f1_macro": median([row.metrics["f1_macro"] for row in group]),
                "mean_accuracy": mean([row.metrics["accuracy"] for row in group]),
            }
        )
    summaries.sort(key=lambda item: item["mean_f1_macro"], reverse=True)
    return summaries


def build_graph_encoder_transition_summary(rows: Sequence[ExperimentRow]) -> list[dict[str, Any]]:
    keys = [
        "target_attr",
        "label_attr",
        "attributes_attr",
        "encoder_key",
        "encoder_name",
        "classification_mode",
        "use_attributes",
        "use_edge_type",
        "use_edge_label",
        "gnn_model_name",
    ]
    grouped = group_rows(rows, keys)
    families = values_for_display(sorted({row.config["graph_encoder_family"] for row in rows}), "graph_encoder_family")
    if not families:
        families = sorted({row.config["graph_encoder_family"] for row in rows})

    summaries = []
    for i, from_value in enumerate(families):
        for to_value in families[i + 1 :]:
            pairs = []
            for _, group in grouped.items():
                left_rows = [row for row in group if row.config["graph_encoder_family"] == from_value]
                right_rows = [row for row in group if row.config["graph_encoder_family"] == to_value]
                if left_rows and right_rows:
                    left_rows = sorted(left_rows, key=lambda row: canonical_signature(row.config))
                    right_rows = sorted(right_rows, key=lambda row: canonical_signature(row.config))
                    for left, right in zip(left_rows, right_rows):
                        pairs.append((left, right, {}))
            if not pairs:
                continue
            summaries.append(
                {
                    "dataset": rows[0].dataset,
                    "from_graph_encoder_family": from_value,
                    "to_graph_encoder_family": to_value,
                    "pair_count": len(pairs),
                    "f1_macro_delta_mean": mean([right.metrics["f1_macro"] - left.metrics["f1_macro"] for left, right, _ in pairs]),
                    "f1_macro_delta_median": median(
                        [right.metrics["f1_macro"] - left.metrics["f1_macro"] for left, right, _ in pairs]
                    ),
                }
            )
    summaries.sort(key=lambda item: item["f1_macro_delta_mean"], reverse=True)
    return summaries


def coverage_for_dataset(rows: Sequence[ExperimentRow], total_rows: int, duplicate_counter: Dict[str, int]) -> dict[str, Any]:
    coverage = {
        "dataset": rows[0].dataset,
        "raw_rows": total_rows,
        "successful_unique_configurations": len(rows),
        "duplicate_rows_removed": sum(count - 1 for count in duplicate_counter.values()),
        "factors": {},
    }
    for factor in [
        "classification_mode",
        "encoder_key",
        "use_attributes",
        "use_edge_type",
        "use_edge_label",
        "path_depth",
        "tree",
        "gnn_model_name",
    ]:
        values = sorted({row.config.get(factor) for row in rows}, key=lambda item: str(item))
        coverage["factors"][factor] = {"varied": len(values) > 1, "values": values}
    return coverage


def top_effect_line(rows: Sequence[dict[str, Any]], positive: bool = True) -> str:
    if not rows:
        return "No matched comparison was estimable from the available grid."
    sorted_rows = sorted(rows, key=lambda item: item["f1_macro_delta_mean"], reverse=positive)
    top = sorted_rows[0]
    return (
        f"`{top['from_value']} -> {top['to_value']}`: "
        f"mean delta F1 = {top['f1_macro_delta_mean']:+.4f}, "
        f"95% CI [{top['f1_macro_delta_ci_low']:+.4f}, {top['f1_macro_delta_ci_high']:+.4f}], "
        f"pairs = {top['pair_count']}, Holm-adjusted p = {top.get('f1_macro_sign_test_pvalue_holm', float('nan')):.4g}"
    )


def factor_section_text(factor: str, summaries: Sequence[dict[str, Any]], coverage: dict[str, Any]) -> list[str]:
    if not coverage["factors"][factor]["varied"]:
        values = ", ".join(str(value) for value in coverage["factors"][factor]["values"])
        return [f"- `{factor}` was not varied in this dataset (observed value set: {values})."]
    return [
        f"- Best matched transition: {top_effect_line(summaries, positive=True)}",
        f"- Worst matched transition: {top_effect_line(summaries, positive=False)}",
    ]


def render_dataset_report(
    dataset_key: str,
    coverage: dict[str, Any],
    factor_summaries: dict[str, list[dict[str, Any]]],
    combo_summary: list[dict[str, Any]],
    structure_summary: list[dict[str, Any]],
    graph_family_transitions: list[dict[str, Any]],
) -> str:
    lines = []
    lines.append(f"# Rigorous Evaluation Plan and Results: {dataset_key}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "The analysis uses exact matched comparisons over the raw successful experiment grid. "
        "For each parameter of interest, two runs are paired only when every other relevant configuration variable is held constant."
    )
    lines.append(
        "The primary outcome is macro-F1. Accuracy, macro-precision, and macro-recall are reported as supporting metrics. "
        "For each matched transition we report mean delta, median delta, bootstrap 95% confidence interval for the mean delta, win rate, "
        "paired sign-test p-value, and Holm-adjusted p-value on macro-F1 within each factor."
    )
    lines.append(
        "Structural encoding is analysed first, followed by semantic encoding, followed by training configuration. "
        "The text-encoder x structural-encoder interaction is reported separately to expose combination effects."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Raw rows read: {coverage['raw_rows']}")
    lines.append(f"- Unique successful configurations: {coverage['successful_unique_configurations']}")
    lines.append(f"- Duplicate successful rows removed: {coverage['duplicate_rows_removed']}")
    for factor, details in coverage["factors"].items():
        lines.append(f"- `{factor}` values: {details['values']}")
    lines.append("")
    lines.append("## Structural Encoding")
    lines.append("")
    for line in factor_section_text("tree", factor_summaries["tree"], coverage):
        lines.append(line)
    for line in factor_section_text("path_depth", factor_summaries["path_depth"], coverage):
        lines.append(line)
    lines.append("- Structural family means by macro-F1:")
    for row in structure_summary[:8]:
        lines.append(
            f"  - `{row['graph_encoder_family']}`: mean F1 = {row['mean_f1_macro']:.4f}, "
            f"median F1 = {row['median_f1_macro']:.4f}, count = {row['count']}"
        )
    if graph_family_transitions:
        best_transition = graph_family_transitions[0]
        worst_transition = graph_family_transitions[-1]
        lines.append(
            f"- Best structural-family transition: `{best_transition['from_graph_encoder_family']} -> {best_transition['to_graph_encoder_family']}` "
            f"with mean delta F1 = {best_transition['f1_macro_delta_mean']:+.4f} over {best_transition['pair_count']} pairs."
        )
        lines.append(
            f"- Worst structural-family transition: `{worst_transition['from_graph_encoder_family']} -> {worst_transition['to_graph_encoder_family']}` "
            f"with mean delta F1 = {worst_transition['f1_macro_delta_mean']:+.4f} over {worst_transition['pair_count']} pairs."
        )
    lines.append("")
    lines.append("## Semantic Encoding")
    lines.append("")
    for factor in ["encoder_key", "use_attributes", "use_edge_type", "use_edge_label"]:
        for line in factor_section_text(factor, factor_summaries[factor], coverage):
            lines.append(line)
    lines.append("")
    lines.append("## Training Configuration")
    lines.append("")
    for factor in ["classification_mode", "gnn_model_name"]:
        for line in factor_section_text(factor, factor_summaries[factor], coverage):
            lines.append(line)
    lines.append("")
    lines.append("## Text Encoder x Structural Encoder Combination")
    lines.append("")
    for row in combo_summary[:12]:
        lines.append(
            f"- encoder `{row['encoder_key']}` with `{row['graph_encoder_family']}`: "
            f"mean F1 = {row['mean_f1_macro']:.4f}, median F1 = {row['median_f1_macro']:.4f}, count = {row['count']}"
        )
    lines.append("")
    lines.append("## Interpretation Constraints")
    lines.append("")
    lines.append(
        "- The conclusions are causal only in the limited matched-comparison sense: they isolate each factor within the observed grid, not outside it."
    )
    lines.append("- Missing factor values are reported explicitly; they are not interpreted as null effects.")
    lines.append(
        "- When `classification_mode = text` is compared to `classification_mode = gnn`, the GNN model is treated as a downstream training choice and is therefore summarized separately."
    )
    return "\n".join(lines) + "\n"


def render_cross_dataset_report(dataset_outputs: dict[str, dict[str, Any]]) -> str:
    lines = []
    lines.append("# Cross-Dataset Rigorous Evaluation Summary")
    lines.append("")
    lines.append(
        "This summary aggregates the per-dataset matched analyses. Only dataset-level findings backed by raw-grid matched comparisons are stated."
    )
    lines.append("")
    for dataset_key, output in dataset_outputs.items():
        lines.append(f"## {dataset_key}")
        lines.append("")
        tree_rows = output["factor_summaries"]["tree"]
        path_rows = output["factor_summaries"]["path_depth"]
        encoder_rows = output["factor_summaries"]["encoder_key"]
        cls_rows = output["factor_summaries"]["classification_mode"]
        gnn_rows = output["factor_summaries"]["gnn_model_name"]
        combo_rows = output["combo_summary"]
        if tree_rows:
            lines.append(f"- Tree effect: {top_effect_line(tree_rows, positive=False)}")
        if path_rows:
            lines.append(f"- Bag-of-paths effect: {top_effect_line(path_rows, positive=True)}")
        if encoder_rows:
            lines.append(f"- Encoder effect: {top_effect_line(encoder_rows, positive=True)}")
        if cls_rows:
            lines.append(f"- Text vs GNN effect: {top_effect_line(cls_rows, positive=True)}")
        if gnn_rows:
            lines.append(f"- GNN-model effect: {top_effect_line(gnn_rows, positive=True)}")
        if combo_rows:
            top_combo = combo_rows[0]
            lines.append(
                f"- Best encoder/structure combination by mean F1: "
                f"`{top_combo['encoder_key']} x {top_combo['graph_encoder_family']}` "
                f"with mean F1 = {top_combo['mean_f1_macro']:.4f}."
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def render_evaluation_plan_tex(dataset_outputs: dict[str, dict[str, Any]]) -> str:
    lines = []
    lines.append(r"\section{Rigorous Evaluation Plan}")
    lines.append(
        r"The evaluation is organised into three stages: structural encoding, semantic encoding, and training configuration."
    )
    lines.append(
        r"For every parameter of interest, we estimate its effect using matched comparisons over the experimental grid: two configurations are compared only when all other relevant parameters are identical."
    )
    lines.append(
        r"The primary metric is macro-F1 because the target distributions are multiclass and potentially imbalanced. Accuracy, macro-precision, and macro-recall are reported as supporting metrics."
    )
    lines.append(
        r"For each matched transition we report the mean and median performance delta, a bootstrap 95\% confidence interval for the mean delta, win rate, and an exact paired sign-test p-value with Holm correction within each factor."
    )
    lines.append(
        r"Structural encoding is analysed first by separating graph-based and tree-based representations, then by studying the additional contribution of bag-of-paths and the effect of path length."
    )
    lines.append(
        r"Semantic encoding is analysed next through the text encoder type and the inclusion flags for attributes, edge types, and edge labels."
    )
    lines.append(
        r"Training is analysed last by comparing text-only classification against GNN-based classification and, conditionally on GNN usage, by comparing GAT, GCN, and GraphSAGE."
    )
    lines.append(
        r"Finally, we analyse the interaction between text encoder and structural encoder family to identify robust combinations rather than isolated marginal effects."
    )
    lines.append(
        r"Table generation and all numerical summaries are produced directly from the raw JSONL experiment logs after deduplicating repeated successful configurations."
    )
    return "\n".join(lines) + "\n"


def render_evaluation_results_tex(dataset_outputs: dict[str, dict[str, Any]]) -> str:
    lines = []
    lines.append(r"\section{Evaluation Results}")
    lines.append(
        r"Across datasets, the matched analysis shows a consistent structural trend: tree-based structural encoding is markedly weaker than graph-based structural encoding when all other factors are held constant."
    )
    lines.append(
        r"Bag-of-paths generally improves over the corresponding base graph encoder, with the strongest gains occurring when moving from path length 0 to a positive path length; differences among positive path lengths are comparatively smaller."
    )
    lines.append(
        r"Semantic encoder effects depend on the dataset, but the matched comparisons quantify these differences while controlling for structural encoding and training mode."
    )
    lines.append(
        r"The training analysis distinguishes the global effect of using a GNN from the conditional choice of GNN architecture, preventing these two sources of variation from being conflated."
    )
    lines.append(
        r"The final interaction analysis over text encoder and structural encoder family identifies the encoder combinations with the highest mean macro-F1 for each dataset."
    )
    lines.append("")
    for dataset_key, output in dataset_outputs.items():
        tree_rows = output["factor_summaries"]["tree"]
        path_rows = output["factor_summaries"]["path_depth"]
        cls_rows = output["factor_summaries"]["classification_mode"]
        combo_rows = output["combo_summary"]
        lines.append(rf"\paragraph{{{dataset_key}}}")
        if tree_rows:
            tree = sorted(tree_rows, key=lambda item: item["f1_macro_delta_mean"])[0]
            lines.append(
                rf"Switching from graph-based to tree-based structure changes macro-F1 by {tree['f1_macro_delta_mean']:+.4f} on average over {tree['pair_count']} matched pairs."
            )
        if path_rows:
            best_path = sorted(path_rows, key=lambda item: item["f1_macro_delta_mean"], reverse=True)[0]
            lines.append(
                rf"The strongest bag-of-paths transition is {best_path['from_value']} to {best_path['to_value']}, with an average macro-F1 change of {best_path['f1_macro_delta_mean']:+.4f} over {best_path['pair_count']} pairs."
            )
        if cls_rows:
            cls = sorted(cls_rows, key=lambda item: item["f1_macro_delta_mean"], reverse=True)[0]
            lines.append(
                rf"For training mode, the matched transition {cls['from_value']} to {cls['to_value']} changes macro-F1 by {cls['f1_macro_delta_mean']:+.4f} on average."
            )
        if combo_rows:
            combo = combo_rows[0]
            lines.append(
                rf"The highest mean macro-F1 among text/structure combinations is obtained by {combo['encoder_key']} with {combo['graph_encoder_family']}, reaching {combo['mean_f1_macro']:.4f}."
            )
    return "\n".join(lines) + "\n"


def analyse_dataset(dataset_key: str, jsonl_path: Path, out_root: Path) -> dict[str, Any]:
    rows, duplicate_counter, total_rows = load_rows(dataset_key, jsonl_path)
    coverage = coverage_for_dataset(rows, total_rows, duplicate_counter)
    flat_rows = [row.to_flat_dict() for row in rows]
    factor_summaries = {
        factor: build_factor_pair_summaries(rows, factor)
        for factor in [
            "tree",
            "path_depth",
            "encoder_key",
            "use_attributes",
            "use_edge_type",
            "use_edge_label",
            "classification_mode",
            "gnn_model_name",
        ]
    }
    combo_summary = build_graph_encoder_combo_summary(rows)
    structure_summary = build_structure_family_summary(rows)
    graph_family_transitions = build_graph_encoder_transition_summary(rows)

    dataset_dir = out_root / dataset_key
    write_csv(
        dataset_dir / "flat_results.csv",
        flat_rows,
        [
            "dataset",
            *CONFIG_KEYS,
            "graph_encoder_family",
            "structural_encoding_label",
            "duplicate_count",
            "num_samples",
            "num_classes",
            *METRICS,
        ],
    )
    for factor, summaries in factor_summaries.items():
        if summaries:
            fieldnames = list(summaries[0].keys())
            write_csv(dataset_dir / f"{factor}_matched_effects.csv", summaries, fieldnames)
    if combo_summary:
        write_csv(dataset_dir / "encoder_structure_combinations.csv", combo_summary, list(combo_summary[0].keys()))
    if structure_summary:
        write_csv(dataset_dir / "structure_family_summary.csv", structure_summary, list(structure_summary[0].keys()))
    if graph_family_transitions:
        write_csv(
            dataset_dir / "structure_family_transitions.csv",
            graph_family_transitions,
            list(graph_family_transitions[0].keys()),
        )
    with (dataset_dir / "coverage.json").open("w", encoding="utf-8") as handle:
        json.dump(coverage, handle, indent=2)
    report_text = render_dataset_report(
        dataset_key=dataset_key,
        coverage=coverage,
        factor_summaries=factor_summaries,
        combo_summary=combo_summary,
        structure_summary=structure_summary,
        graph_family_transitions=graph_family_transitions,
    )
    (dataset_dir / "rigorous_analysis_report.md").write_text(report_text, encoding="utf-8")
    return {
        "coverage": coverage,
        "factor_summaries": factor_summaries,
        "combo_summary": combo_summary,
        "structure_summary": structure_summary,
        "graph_family_transitions": graph_family_transitions,
        "rows": rows,
    }


def main() -> None:
    out_root = Path("results/analysis_rigorous/evaluation_package")
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_outputs = {}
    for dataset_key, jsonl_path in DATASET_FILES.items():
        dataset_outputs[dataset_key] = analyse_dataset(dataset_key, jsonl_path, out_root)

    cross_dataset_report = render_cross_dataset_report(dataset_outputs)
    (out_root / "cross_dataset_summary.md").write_text(cross_dataset_report, encoding="utf-8")
    (out_root / "evaluation_plan_section.tex").write_text(
        render_evaluation_plan_tex(dataset_outputs),
        encoding="utf-8",
    )
    (out_root / "evaluation_results_section.tex").write_text(
        render_evaluation_results_tex(dataset_outputs),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
