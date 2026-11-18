"""Bug-fix checks on the real datasets (EAModelSet Type/Layer, ModelSet Abstract).

A. Leak audit: test-node types, targets and deny-listed meta-properties are
   (and planted CM2ML-style key/value meta-properties such as isAbstract)
   are replaced by unique sentinel tokens; no sentinel may appear in any text the
   text classifier or the GNN sees (graph + tree, d = 0..3, every flag on).
   Positive control: train-node type sentinels must appear (for d > 0) and a
   node's own type sentinel must never appear in its own text.
B. Tree and graph d=0 texts of classified nodes are identical.
C. GNN node texts change with d.
D. d=0 text-classifier results reproduce the old runs exactly (that code path
   was not buggy, so any change here would be a regression).
E. Corpus build time on EA (previously O(n^2) per graph).

Usage: python tests/verify_real_data.py [--skip-regression]
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cm2ml_encodings_eval.config import TextBuildConfig, TextExperimentConfig  # noqa: E402
from cm2ml_encodings_eval.pipeline import run_text_node_classification  # noqa: E402
from cm2ml_encodings_eval.text import build_graph_texts, build_node_corpus, is_labelled  # noqa: E402
from cm2ml_encodings_eval.tree_transform import serialize_graphs_to_trees  # noqa: E402
from data_loading.models_dataset import get_models_dataset  # noqa: E402
import run_all_node_cls_combinations as runner  # noqa: E402

TASKS = [("eamodelset", "type"), ("eamodelset", "layer"), ("modelset", "abstract")]
OLD_D0_BOW_F1 = {  # from all_results/*.jsonl, text mode, graph, d=0, bow
    ("eamodelset", "type", False): 0.4226573089202567,
    ("eamodelset", "layer", False): 0.6145335372683735,
    ("modelset", "abstract", False): 0.6737748357715714,
    ("modelset", "abstract", True): 0.6722045720847556,
}

failures = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def load(dataset, target):
    graphs = list(get_models_dataset(dataset_name=dataset, min_enr=1.2, min_edges=10, reload=False).graphs)
    runner.deterministic_mask_graph_nodes(graphs, 0.2, 42)
    return graphs, runner.make_metadata(dataset, target)


def all_on(meta, d):
    return TextBuildConfig(include_node_label=True, include_node_type=True, include_node_attributes=True,
                           include_edge_label=True, include_edge_type=True, path_depth=d, metadata=meta)


def plant_sentinels(graphs, meta):
    """Return copies whose leak-relevant values are unique sentinel tokens."""
    gs = copy.deepcopy(graphs)
    k = 0
    for g in gs:
        for _, d in g.nodes(data=True):
            k += 1
            if meta.type in d:
                d[meta.type] = (f"zzleaktype{k}q" if d.get("masked") else f"zzvis{k}q")
                d["_own"] = d[meta.type]
            if meta.cls != meta.type and meta.cls in d:
                d[meta.cls] = f"zzleaklabel{k}q"
            if isinstance(d.get(meta.attributes), list):
                # CM2ML-style key/value meta-properties that state the target
                d[meta.attributes] = list(d[meta.attributes]) + [(key, f"zzleakmeta{k}q") for key in meta.leak_keys]
    return gs


def audit_texts(texts, label):
    joined = "\n".join(texts)
    check("zzleak" not in joined, f"{label}: no test-node type / target sentinel in {len(texts)} texts")
    return joined


def section_a(graphs, meta, task):
    print(f"[A] leak audit {task}")
    gs = plant_sentinels(graphs, meta)
    trees = serialize_graphs_to_trees(gs, metadata=meta)
    for structure, G in (("graph", gs), ("tree", trees)):
        for d in (0, 1, 2, 3):
            c = all_on(meta, d)
            t0 = time.time()
            corpus, _, _ = build_node_corpus(G, meta, True, True, True, True, True, d, 64)
            gnn_texts, own = [], []
            for g in G:
                texts = build_graph_texts(g, c)
                gnn_texts.extend(texts)
                own.extend((data.get("_own"), t) for (_, data), t in zip(g.nodes(data=True), texts))
            tag = f"{structure} d={d}"
            audit_texts(corpus["train"] + corpus["test"], f"{tag} text-clf")
            joined = audit_texts(gnn_texts, f"{tag} GNN features")
            if meta.leak_keys:
                check("zzleakmeta" not in joined, f"{tag}: deny-listed meta-properties {meta.leak_keys} never emitted")
            check(not any(o and o in t for o, t in own), f"{tag}: no node shows its own type")
            if d > 0 and any(meta.type in data for g in G for _, data in g.nodes(data=True)):
                check("zzvis" in joined, f"{tag}: positive control, train-node types visible via paths")
            print(f"    ({time.time() - t0:.1f}s)")


def section_b(graphs, meta, task):
    print(f"[B] tree d=0 == graph d=0 {task}")
    trees = serialize_graphs_to_trees(graphs, metadata=meta)
    g_corpus, g_lab, _ = build_node_corpus(graphs, meta, True, True, True, True, True, 0, 64)
    t_corpus, t_lab, _ = build_node_corpus(trees, meta, True, True, True, True, True, 0, 64)
    check(g_corpus == t_corpus and g_lab == t_lab,
          f"{len(g_corpus['train'])}/{len(g_corpus['test'])} train/test texts and labels identical")
    n_cls = len(set(g_lab["train"] + g_lab["test"]))
    t_cls = len(set(t_lab["train"] + t_lab["test"]))
    check(n_cls == t_cls, f"label space graph={n_cls} tree={t_cls}")
    distinct = len(set(t_corpus["train"]))
    check(distinct > 1, f"tree OBJ texts are not constant ({distinct} distinct train texts)")


def section_c(graphs, meta, task):
    print(f"[C] GNN features depend on d {task}")
    c0, c1 = all_on(meta, 0), all_on(meta, 1)
    changed = total = with_nbrs = 0
    for g in graphs:
        for n, a, b in zip(g.nodes(), build_graph_texts(g, c0), build_graph_texts(g, c1)):
            total += 1
            changed += a != b
            # self-loops are not paths, so only count neighbours other than the node itself
            with_nbrs += any(v != n for v in nx.all_neighbors(g, n))
    check(changed == with_nbrs, f"{changed}/{total} node features change d0->d1 "
                                f"(= nodes with a neighbour other than themselves: {with_nbrs})")


def section_d(graphs, meta, dataset, target):
    print(f"[D] old d=0 results reproduce {dataset}/{target}")
    for (ds, tg, attrs), old in OLD_D0_BOW_F1.items():
        if (ds, tg) != (dataset, target):
            continue
        c = TextBuildConfig(include_node_label=True, include_node_type=False, include_node_attributes=attrs,
                            include_edge_label=False, include_edge_type=False, path_depth=0, metadata=meta)
        r = run_text_node_classification(graphs, c, TextExperimentConfig(encoder_name="bow", classifier_name="logreg"))
        new = r["metrics"]["f1_macro"]
        check(abs(new - old) < 1e-9, f"attrs={attrs}: F1 old={old:.6f} new={new:.6f}")


def section_e(graphs, meta):
    print("[E] corpus build time EA d=0")
    t0 = time.time()
    build_node_corpus(graphs, meta, True, False, False, False, False, 0, 64)
    dt = time.time() - t0
    check(dt < 10, f"EA d=0 corpus built in {dt:.1f}s (was 42.7s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-regression", action="store_true")
    args = ap.parse_args()
    for dataset, target in TASKS:
        graphs, meta = load(dataset, target)
        task = f"{dataset}/{target}"
        if (dataset, target) == ("eamodelset", "type"):
            section_e(graphs, meta)
        section_a(graphs, meta, task)
        section_b(graphs, meta, task)
        section_c(graphs, meta, task)
        if not args.skip_regression:
            section_d(graphs, meta, dataset, target)
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED:\n  " + "\n  ".join(failures)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
