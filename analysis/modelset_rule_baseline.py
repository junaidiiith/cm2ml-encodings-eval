"""Rule baseline for ModelSet Abstract: predict "abstract" iff the class has a subclass.

Uses the same test nodes as the grid (20% of nodes, mask seed 42) and the declared
EClass@abstract labels. Prints macro-F1 and accuracy on the test nodes.

Usage: CUDA_VISIBLE_DEVICES="" python analysis/modelset_rule_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

import run_all_node_cls_combinations as runner  # noqa: E402
from data_loading.models_dataset import get_models_dataset  # noqa: E402


def as_bool(v):
    return str(v).strip().lower() == "true" if isinstance(v, str) else bool(v)


def main():
    graphs = list(get_models_dataset(dataset_name="modelset", min_enr=1.2, min_edges=10, reload=False).graphs)
    runner.deterministic_mask_graph_nodes(graphs, 0.2, 42)
    y, p = [], []
    for g in graphs:
        for n, d in g.nodes(data=True):
            if d.get("masked"):
                y.append(as_bool(d["abstract"]))
                p.append(any(e.get("type") == "supertype" for _, _, e in g.in_edges(n, data=True)))
    print(f"test nodes {len(y)} | rule 'abstract iff it has a subclass': "
          f"macro-F1 = {f1_score(y, p, average='macro'):.4f}, accuracy = {accuracy_score(y, p):.4f}")


if __name__ == "__main__":
    main()
