from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

from data_loading.models_dataset import get_models_dataset

from cm2ml_encodings_eval.config import GNNExperimentConfig, MetaDataConfig, TextBuildConfig, TextExperimentConfig
from cm2ml_encodings_eval.pipeline import (
    run_gnn_node_classification,
    run_text_node_classification,
    run_tree_gnn_node_classification,
    run_tree_text_node_classification,
)


ENCODER_MAP = {
    "bow": "bow",
    "tfidf": "tfidf",
    "w2v": "word2vec",
    "bert-encoder": "bert",
    "custom-w2v": "custom_w2v"
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def deterministic_mask_graph_nodes(graphs: Iterable, mask_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    for graph in graphs:
        nodes = list(graph.nodes())
        for n in nodes:
            graph.nodes[n]["masked"] = False

        if len(nodes) <= 1:
            if len(nodes) == 1:
                graph.nodes[nodes[0]]["masked"] = True
            continue

        n_mask = int(round(len(nodes) * mask_ratio))
        n_mask = max(1, min(n_mask, len(nodes) - 1))
        for n in rng.sample(nodes, n_mask):
            graph.nodes[n]["masked"] = True


def make_text_cfg(
    target_attr: str,
    label_attr: str,
    attributes_attr: str,
    path_depth: int,
    use_attributes: bool,
    use_edge_type: bool,
    use_edge_label: bool,
) -> TextBuildConfig:
    text_cfg = TextBuildConfig(
        include_node_label=True,
        include_node_type=False,
        include_node_attributes=use_attributes,
        include_edge_label=use_edge_label,
        include_edge_type=use_edge_type,
        path_depth=path_depth,
    )
    text_cfg.metadata = MetaDataConfig(cls=target_attr, label=label_attr, attributes=attributes_attr)
    return text_cfg


def run_one(
    graphs,
    cfg: Dict,
    text_classifier: str,
    gnn_cfg: GNNExperimentConfig,
) -> Dict:
    text_cfg = make_text_cfg(
        target_attr=cfg["target_attr"],
        label_attr=cfg["label_attr"],
        attributes_attr=cfg["attributes_attr"],
        path_depth=cfg["path_depth"],
        use_attributes=cfg["use_attributes"],
        use_edge_type=cfg["use_edge_type"],
        use_edge_label=cfg["use_edge_label"],
    )

    encoder_name = cfg["encoder_name"]

    if cfg["classification_mode"] == "text":
        exp_cfg = TextExperimentConfig(encoder_name=encoder_name, classifier_name=text_classifier)
        if cfg["tree"]:
            text_cfg.is_tree = True
            return run_tree_text_node_classification(
                graphs,
                text_cfg=text_cfg,
                exp_cfg=exp_cfg,
            )
        return run_text_node_classification(
            graphs,
            text_cfg=text_cfg,
            exp_cfg=exp_cfg,
        )

    # GNN mode
    if cfg["tree"]:
        text_cfg.is_tree = True
        return run_tree_gnn_node_classification(
            graphs,
            node_encoder_name=encoder_name,
            text_cfg=text_cfg,
            gnn_cfg=gnn_cfg,
        )
    return run_gnn_node_classification(
        graphs,
        node_encoder_name=encoder_name,
        text_cfg=text_cfg,
        gnn_cfg=gnn_cfg,
    )

def get_config_str(config):
    config_str = "".join([
        str(config["config"]["classification_mode"]),
        str(config["config"]["tree"]),
        str(config["config"]["use_attributes"]),
        str(config["config"]["use_edge_type"]),
        str(config["config"]["use_edge_label"]),
        str(config["config"]["encoder_key"]),
        str(config["config"]["path_depth"]),
        str(config["config"].get("gnn_model_name", "none"))
    ])
    return config_str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all node classification configuration combinations")
    parser.add_argument("--dataset", default="modelset")
    parser.add_argument("--min-enr", type=float, default=1.2)
    parser.add_argument("--min-edges", type=int, default=10)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--max-graphs", type=int, default=None)

    parser.add_argument("--target", default="abstract")
    parser.add_argument("--label-attr", default="name")
    parser.add_argument("--attributes-attr", default="attributes")
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--text-classifier", default="logreg", choices=["logreg", "linearsvm"])
    parser.add_argument("--gnn-model", default="graphsage", choices=["gcn", "graphsage", "sage", "gat"])
    parser.add_argument("--gnn-epochs", type=int, default=30)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument("--gnn-dropout", type=float, default=0.3)
    parser.add_argument("--gnn-lr", type=float, default=1e-3)
    parser.add_argument("--gnn-weight-decay", type=float, default=5e-4)

    parser.add_argument("--out", default="results/node_cls_grid")
    parser.add_argument("--summary-out", default="results/node_cls_grid_summary")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N combinations (for smoke testing)")
    
    parser.add_argument("--cuda_devices", default="1", help="Comma-separated list of CUDA device IDs to use (e.g., '0,1,2')")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    set_seed(args.seed)

    dataset = get_models_dataset(
        dataset_name=args.dataset,
        min_enr=args.min_enr,
        min_edges=args.min_edges,
        reload=args.reload,
    )
    graphs = list(dataset.graphs)
    if args.max_graphs is not None:
        graphs = graphs[: args.max_graphs]

    out_path = Path(f"{args.out}_{args.dataset}_{args.target}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(f"{args.summary_out}_{args.dataset}_{args.target}.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    gnn_cfg = GNNExperimentConfig(
        model_name=args.gnn_model,
        hidden_dim=args.gnn_hidden_dim,
        num_layers=args.gnn_num_layers,
        dropout=args.gnn_dropout,
        learning_rate=args.gnn_lr,
        weight_decay=args.gnn_weight_decay,
        epochs=args.gnn_epochs,
    )

    grid = itertools.product(
        
        ['sage', 'gcn', 'gat'],  # gnn model
        [False, True],  # tree transformation
        
        [True],  # use_attributes
        [False, True],  # use_edge_type
        [True],  # use_edge_label
        [1, 0, 2, 3],
        ["text", "gnn"],
        ["bow", "tfidf", "w2v", "bert-encoder"],
    )
    # grid = itertools.product(
        
    #     ['sage'],
    #     [False, True],  # tree transformation
        
    #     [True],  # use_attributes
    #     [True],  # use_edge_type
    #     [True],  # use_edge_label
    #     [1],
    #     ["text"],
    #     ["bow"],
    # )

    total = 0
    success = 0
    failed = 0
    runs = set()
    if os.path.exists(out_path):
        with out_path.open("r", encoding="utf-8") as f:
            d = f.read()
            
            for s in d.strip().split("\n"):
                # try:
                config = json.loads(s)
                config_str = get_config_str(config)
                # if "status" in config and config["status"] == "ok":
                runs.add(f"{config_str}_{'tree' if config['config']['tree'] else ''}")
                # except Exception as e:  # noqa: BLE001
                #     print(s)
                #     break
    
    print(f"Already completed {len(runs)} runs, skipping those configurations.")
    for combo in grid:
        with out_path.open("a", encoding="utf-8") as f:
            total += 1
            if args.limit is not None and total > args.limit:
                break

            gnn_model, tree_mode, use_attrs, use_edge_type, use_edge_label, depth, cls_mode, text_encoder = combo
            if total in runs:
                continue
            gnn_cfg.model_name = gnn_model
            cfg = {
                "target_attr": args.target,
                "label_attr": args.label_attr,
                "attributes_attr": args.attributes_attr,
                "encoder_name": ENCODER_MAP[text_encoder],
                "encoder_key": text_encoder,
                "classification_mode": cls_mode,
                "tree": tree_mode,
                "path_depth": depth,
                "use_attributes": use_attrs,
                "use_edge_type": use_edge_type,
                "use_edge_label": use_edge_label,
                "gnn_model_name": gnn_model if cls_mode == "gnn" else "none",
            }
            config_str = get_config_str({"config": cfg})
            config_str = f"{config_str}_{'tree' if cfg['tree'] else ''}"
            if config_str in runs:
                continue
            deterministic_mask_graph_nodes(graphs, mask_ratio=args.mask_ratio, seed=args.seed)
            set_seed(args.seed)

            print("Running combination:")
            print(f"[{total}] | mode={cls_mode} tree={tree_mode} enc={text_encoder} d={depth} attrs={use_attrs} et={use_edge_type} el={use_edge_label}")
            
            record = {"id": total, "config": cfg}
            try:
                result = run_one(
                    graphs=graphs,
                    cfg=cfg,
                    text_classifier=args.text_classifier,
                    gnn_cfg=gnn_cfg,
                )
                record["status"] = "ok"
                record["result"] = result
                success += 1
            except Exception as exc:
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1

            f.write(json.dumps(record) + "\n")
            f.flush()
            print("Completed combination:")
            print(f"[{total}] {record['status']} | mode={cls_mode} tree={tree_mode} enc={text_encoder} d={depth} attrs={use_attrs} et={use_edge_type} el={use_edge_label}")

    summary = {
        "dataset": args.dataset,
        "num_graphs": len(graphs),
        "total": total if args.limit is None else min(total, args.limit),
        "success": success,
        "failed": failed,
        "output": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
