"""Check that no two grid configurations feed the model identical inputs.

For every (dataset, mode, structure, depth) cell, the text-config variants in the
grid must produce different input texts: for the text classifier the texts of the
classified nodes, for the GNN the texts of every node. Identical inputs mean a
factor is redundant and the grid would contain duplicate experiments.

Runs on the first N graphs (default 60): inputs that differ on a subset also
differ on the full data. CPU only.

Usage: CUDA_VISIBLE_DEVICES="" python tests/check_grid_duplicates.py [--max-graphs 60]
"""

import argparse
import hashlib
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cm2ml_encodings_eval.text import build_graph_texts, build_node_corpus  # noqa: E402
from cm2ml_encodings_eval.tree_transform import serialize_graphs_to_trees  # noqa: E402
from data_loading.models_dataset import get_models_dataset  # noqa: E402
import run_all_node_cls_combinations as runner  # noqa: E402

TASKS = [("eamodelset", "type"), ("eamodelset", "layer"), ("modelset", "abstract")]


def digest(texts) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=60)
    args = ap.parse_args()
    problems = []
    for dataset, target in TASKS:
        graphs = list(get_models_dataset(dataset_name=dataset, min_enr=1.2, min_edges=10, reload=False).graphs)
        runner.deterministic_mask_graph_nodes(graphs, 0.2, 42)
        graphs = graphs[: args.max_graphs]
        meta = runner.make_metadata(dataset, target)
        structures = {"graph": graphs, "tree": serialize_graphs_to_trees(graphs, metadata=meta)}
        grid_args = Namespace(dataset=dataset, target=target, label_attr="name", structures=["graph", "tree"],
                              depths=[0, 1, 2, 3], modes=["text", "gnn"], encoders=["bow"], gnn_models=["sage"],
                              mask_ratio=0.2, mask_seed=42, seed=42, edge_direction=None)
        grid = runner.build_grid(grid_args, runner.available_fields(graphs, meta))

        cells = defaultdict(list)
        for cfg in grid:
            cells[(cfg["classification_mode"], cfg["structure"], cfg["path_depth"])].append(cfg)

        n_checked = 0
        for (mode, structure, depth), cfgs in sorted(cells.items()):
            seen = {}
            for cfg in cfgs:
                text_cfg = runner.make_text_cfg(cfg, meta)
                G = structures[structure]
                if mode == "text":
                    corpus, _, _ = build_node_corpus(
                        G, meta, text_cfg.include_node_label, text_cfg.include_node_type,
                        text_cfg.include_node_attributes, text_cfg.include_edge_label,
                        text_cfg.include_edge_type, depth, text_cfg.max_paths_per_node)
                    key = digest(corpus["train"] + corpus["test"])
                else:
                    key = digest(t for g in G for t in build_graph_texts(g, text_cfg))
                flags = {f: cfg[f] for f in runner.TEXT_FLAGS}
                if key in seen:
                    problems.append(f"{dataset}/{target} {mode} {structure} d={depth}: {flags} == {seen[key]}")
                else:
                    seen[key] = flags
                n_checked += 1
        print(f"{dataset}/{target}: {n_checked} configs in {len(cells)} cells checked", flush=True)

    if problems:
        print(f"\n{len(problems)} DUPLICATE INPUTS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("\nNO DUPLICATE INPUTS")


if __name__ == "__main__":
    main()
