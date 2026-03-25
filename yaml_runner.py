from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from data_loading.models_dataset import get_models_dataset

from cm2ml_encodings_eval.config import (
    BertClassifierConfig,
    GNNExperimentConfig,
    MetaDataConfig,
    SplitConfig,
    TextBuildConfig,
    TextExperimentConfig,
    TreeSerializationConfig,
)
from cm2ml_encodings_eval.pipeline import (
    run_gnn_edge_classification,
    run_gnn_node_classification,
    run_text_edge_classification,
    run_text_node_classification,
    run_tree_gnn_node_classification,
    run_tree_text_node_classification,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_metadata(cfg: Dict[str, Any] | None) -> MetaDataConfig:
    cfg = cfg or {}
    return MetaDataConfig(**cfg)


def _build_text_cfg(cfg: Dict[str, Any] | None, metadata_cfg: Dict[str, Any] | None = None) -> TextBuildConfig:
    cfg = dict(cfg or {})
    nested_meta = cfg.pop("metadata", None)
    text_cfg = TextBuildConfig(**cfg)
    text_cfg.metadata = _build_metadata(nested_meta or metadata_cfg)
    return text_cfg


def _build_split_cfg(cfg: Dict[str, Any] | None) -> SplitConfig:
    return SplitConfig(**(cfg or {}))


def _build_text_exp_cfg(cfg: Dict[str, Any] | None) -> TextExperimentConfig:
    return TextExperimentConfig(**(cfg or {}))


def _build_bert_cfg(cfg: Dict[str, Any] | None) -> BertClassifierConfig:
    return BertClassifierConfig(**(cfg or {}))


def _build_gnn_cfg(cfg: Dict[str, Any] | None) -> GNNExperimentConfig:
    return GNNExperimentConfig(**(cfg or {}))


def _build_tree_cfg(cfg: Dict[str, Any] | None) -> TreeSerializationConfig:
    return TreeSerializationConfig(**(cfg or {}))


def _mask_graph_nodes(graphs: List, ratio: float, seed: int, reset_existing: bool = True) -> None:
    rng = random.Random(seed)
    for graph in graphs:
        nodes = list(graph.nodes())
        if not nodes:
            continue
        if reset_existing:
            for n in nodes:
                graph.nodes[n]["masked"] = False

        n_mask = int(round(len(nodes) * ratio))
        n_mask = max(1, min(n_mask, len(nodes) - 1)) if len(nodes) > 1 else len(nodes)
        masked_nodes = rng.sample(nodes, n_mask) if n_mask > 0 else []
        for node in masked_nodes:
            graph.nodes[node]["masked"] = True


def _ensure_mask_present(graphs: List) -> None:
    for graph in graphs:
        has_any = any(bool(graph.nodes[n].get("masked", False)) for n in graph.nodes())
        if not has_any and len(graph.nodes()) > 1:
            first = next(iter(graph.nodes()))
            graph.nodes[first]["masked"] = True


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_from_yaml(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/object.")

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    data_cfg = cfg.get("dataset", {})
    dataset_name = data_cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("dataset.dataset_name is required in YAML.")

    dataset_params = dict(data_cfg.get("params", {}))
    dataset = get_models_dataset(dataset_name=dataset_name, **dataset_params)
    graphs = list(dataset.graphs)

    max_graphs = data_cfg.get("max_graphs")
    if max_graphs is not None:
        graphs = graphs[: int(max_graphs)]

    mask_cfg = cfg.get("masking", {})
    if mask_cfg.get("enabled", True):
        _mask_graph_nodes(
            graphs,
            ratio=float(mask_cfg.get("ratio", 0.2)),
            seed=int(mask_cfg.get("seed", seed)),
            reset_existing=bool(mask_cfg.get("reset_existing", True)),
        )
    else:
        _ensure_mask_present(graphs)

    base_metadata_cfg = cfg.get("metadata", {})

    runs = cfg.get("runs", [])
    if not runs:
        runs = [{"name": "default_gnn_node", "pipeline": "gnn_node"}]

    outputs: List[Dict[str, Any]] = []

    for run in runs:
        run_name = run.get("name", run.get("pipeline", "unnamed"))
        pipeline_name = str(run.get("pipeline", "")).strip().lower()
        if not pipeline_name:
            raise ValueError(f"Run '{run_name}' is missing 'pipeline'.")

        text_cfg = _build_text_cfg(run.get("text_build", deepcopy(cfg.get("text_build", {}))), run.get("metadata", base_metadata_cfg))
        text_exp_cfg = _build_text_exp_cfg(run.get("text_experiment", deepcopy(cfg.get("text_experiment", {}))))
        bert_cfg = _build_bert_cfg(run.get("bert", deepcopy(cfg.get("bert", {}))))
        gnn_cfg = _build_gnn_cfg(run.get("gnn", deepcopy(cfg.get("gnn", {}))))
        tree_cfg = _build_tree_cfg(run.get("tree", deepcopy(cfg.get("tree", {}))))

        node_encoder_name = run.get("node_encoder_name", run.get("encoder_name", "tfidf"))
        edge_encoder_name = run.get("edge_encoder_name", "tfidf")
        run_graphs = graphs
        run_max_graphs = run.get("max_graphs")
        if run_max_graphs is not None:
            run_graphs = run_graphs[: int(run_max_graphs)]

        if pipeline_name == "text_node":
            result = run_text_node_classification(
                run_graphs,
                text_cfg=text_cfg,
                exp_cfg=text_exp_cfg,
                bert_cfg=bert_cfg,
            )
        elif pipeline_name == "text_edge":
            result = run_text_edge_classification(
                run_graphs,
                target_attr=run.get("target_attr", text_cfg.metadata.cls),
                text_cfg=text_cfg,
                exp_cfg=text_exp_cfg,
                bert_cfg=bert_cfg,
            )
        elif pipeline_name == "gnn_node":
            result = run_gnn_node_classification(
                run_graphs,
                node_encoder_name=node_encoder_name,
                text_cfg=text_cfg,
                gnn_cfg=gnn_cfg,
            )
        elif pipeline_name == "gnn_edge":
            if not run_graphs:
                raise ValueError("No graphs available for gnn_edge.")
            graph_index = int(run.get("graph_index", 0))
            graph = run_graphs[graph_index]
            result = run_gnn_edge_classification(
                graph,
                metadata=text_cfg.metadata,
                node_encoder_name=node_encoder_name,
                edge_encoder_name=edge_encoder_name,
                text_cfg=text_cfg,
                gnn_cfg=gnn_cfg,
            )
        elif pipeline_name == "tree_text_node":
            result = run_tree_text_node_classification(
                run_graphs,
                text_cfg=text_cfg,
                exp_cfg=text_exp_cfg,
                bert_cfg=bert_cfg,
                tree_cfg=tree_cfg,
            )
        elif pipeline_name == "tree_gnn_node":
            if "graph_index" in run:
                raise ValueError(
                    "tree_gnn_node now runs on a dataset/list of graphs. "
                    "Remove 'graph_index' from this run."
                )
            result = run_tree_gnn_node_classification(
                run_graphs,
                node_encoder_name=node_encoder_name,
                text_cfg=text_cfg,
                gnn_cfg=gnn_cfg,
                tree_cfg=tree_cfg,
            )
        else:
            raise ValueError(
                f"Unsupported pipeline '{pipeline_name}'. Supported: "
                "text_node, text_edge, gnn_node, gnn_edge, tree_text_node, tree_gnn_node"
            )

        outputs.append(
            {
                "name": run_name,
                "pipeline": pipeline_name,
                "result": _to_jsonable(result),
            }
        )

    return {
        "config_path": str(config_path),
        "dataset_name": dataset_name,
        "num_graphs": len(graphs),
        "seed": seed,
        "runs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cm2ml pipelines from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = run_from_yaml(args.config)
    payload = json.dumps(result, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"Saved results to {output_path}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
