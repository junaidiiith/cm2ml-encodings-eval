"""Run the node-classification grid: semantic encoders x structural encodings.

Grid axes
---------
* encoder            bow | tfidf | w2v | bert-encoder
* structure          graph | tree
* path_depth (d)     0..3   (bag of paths over the undirected neighbourhood)
* classification     text (logistic regression) | gnn (sage | gcn | gat)
* text config        use_attributes, use_edge_type, use_edge_label, use_node_type

Factors that cannot change the input are stored as ``None`` (reported as n/a)
and not expanded into duplicate runs:

* use_edge_type / use_edge_label / use_node_type when no original edge can be
  reached: d == 0 (a node's own type is always hidden, and a single-node path has
  no edges), and d == 1 for tree + text classifier (see text_flag_combinations);
* any flag whose field does not occur in the dataset (e.g. EA has no
  attributes or edge labels, ModelSet has no node types).

Tree + text classifier + d == 0 is the same input as graph + text + d == 0, so it
is not run twice.

Designs (``--design``)
----------------------
* ``full``    -- every configuration above.
* ``reduced`` -- (A) "core": every text-classifier run and every run of the
  ``--full-gnn-models`` (default SAGE) over all text configs; plus (B) "slice":
  the other GNN models on every structure x encoder x d, but only the one text
  config with every available flag on. The GNN-model effect is estimated on
  matched pairs (same config, only the GNN differs); all other factors on the core.

Follow-ups use ``--design full --only-base-signatures FILE``: (C) fill in the
remaining text configs of a GNN cell near the top 5, (D) re-run top configs with
another ``--seed``. A base signature identifies a configuration without its seed.

Parallel jobs: ``--shard i/N`` runs every N-th configuration (stable hash) and
writes its own file; resuming reads every result file of the dataset/target.
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    # GPU must be chosen before torch (or settings.py) is imported; see gpu_select.py.
    import argparse as _ap
    from gpu_select import DEFAULT_NEED_MB, configure_gpu, parse_device_list

    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument("--cuda_devices", default=None)
    _pre.add_argument("--gpu-mem-mb", type=int, default=DEFAULT_NEED_MB)
    _pre.add_argument("--dry-run", action="store_true")
    _pre.add_argument("--torch-threads", type=int, default=8)
    _known, _ = _pre.parse_known_args(sys.argv[1:])
    # Cap BLAS/OpenMP threads before numpy/torch load: the CPU, not the GPU, is the
    # bottleneck of these jobs and the machine is shared.
    import os as _os
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _os.environ.setdefault(_var, str(_known.torch_threads))
    # Graphs of very different sizes fragment PyTorch's CUDA cache: EA tree runs
    # allocate <= 5.2 GB but the cache grew to 22.5 GB. Expandable segments avoid
    # that; main() also caps the process at --gpu-mem-mb (see gpu_select rule).
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not _known.dry_run:
        try:
            _GPU_INDEX = configure_gpu(_known.gpu_mem_mb, parse_device_list(_known.cuda_devices))
        except RuntimeError as _exc:
            sys.exit(f"[gpu_select] {_exc}")

import argparse
import hashlib
import itertools
import json
import random
import subprocess
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from cm2ml_encodings_eval.config import GNNExperimentConfig, MetaDataConfig, TextBuildConfig, TextExperimentConfig
from cm2ml_encodings_eval.text import filter_attributes

ENCODER_MAP = {
    "bow": "bow",
    "tfidf": "tfidf",
    "w2v": "word2vec",
    "bert-encoder": "bert",
    "custom-w2v": "custom_w2v",
}

# Fields per (dataset, target). leak_keys are attribute / meta-property names that
# reveal the target and are removed everywhere.
DATASET_METADATA = {
    ("modelset", "abstract"): dict(cls="abstract", label="name", attributes="attributes", type="type",
                                   edge_label="name", edge_type="type", leak_keys=("abstract", "isAbstract")),
    ("eamodelset", "type"): dict(cls="type", label="name", attributes="attributes", type="type",
                                 edge_label="name", edge_type="type", leak_keys=()),
    ("eamodelset", "layer"): dict(cls="layer", label="name", attributes="attributes", type="type",
                                  edge_label="name", edge_type="type", leak_keys=()),
}

TEXT_FLAGS = ("use_attributes", "use_edge_type", "use_edge_label", "use_node_type")
DESIGNS = ("full", "reduced")
SIGNATURE_KEYS = (
    "dataset", "target_attr", "classification_mode", "structure", "encoder_key", "gnn_model_name",
    "path_depth", *TEXT_FLAGS, "edge_direction", "mask_seed", "seed",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_metadata(dataset: str, target: str) -> MetaDataConfig:
    key = (dataset, target)
    if key not in DATASET_METADATA:
        raise ValueError(f"No metadata preset for dataset={dataset!r} target={target!r}; add one to DATASET_METADATA.")
    return MetaDataConfig(**DATASET_METADATA[key])


def available_fields(graphs: List, meta: MetaDataConfig) -> Dict[str, bool]:
    """Which optional text fields actually occur in the data."""
    has_attrs = has_type = has_elabel = has_etype = False
    for g in graphs:
        for _, d in g.nodes(data=True):
            has_attrs = has_attrs or bool(filter_attributes(d.get(meta.attributes), meta))
            has_type = has_type or d.get(meta.type) not in (None, "")
        for *_, d in g.edges(data=True):
            has_elabel = has_elabel or d.get(meta.edge_label) not in (None, "")
            has_etype = has_etype or d.get(meta.edge_type) not in (None, "")
        if has_attrs and has_type and has_elabel and has_etype:
            break
    return {
        "use_attributes": has_attrs,
        "use_node_type": has_type,
        "use_edge_label": has_elabel,
        "use_edge_type": has_etype,
    }


def text_flag_combinations(depth: int, fields: Dict[str, bool], structure: str = "graph",
                           mode: str = "text") -> List[Dict[str, bool | None]]:
    """All non-redundant text configs for one cell; redundant flags are None.

    Edge flags and neighbour types need at least one traversed original edge:
    d >= 1 in a graph. In a tree, a classified OBJ node first reaches an
    original edge (ASSOCS -> REF) at d = 2, so for tree + text classifier they
    need d >= 2. (Tree + GNN uses every tree node's text, and an ASSOCS node
    reaches its REF children at d = 1.)
    """
    min_depth = 2 if (structure == "tree" and mode == "text") else 1
    axes = {}
    for flag in TEXT_FLAGS:
        meaningful = fields[flag] and (flag == "use_attributes" or depth >= min_depth)
        axes[flag] = [False, True] if meaningful else [None]
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


def is_slice_config(flags: Dict[str, bool | None]) -> bool:
    """The single text config per cell used for the GNN-model slice: every available flag on."""
    return all(v is None or v is True for v in flags.values())


def build_grid(args, fields: Dict[str, bool]) -> List[Dict]:
    design = getattr(args, "design", "full")
    full_gnn_models = set(getattr(args, "full_gnn_models", ["sage"]))
    grid = []
    for structure, depth, mode, encoder in itertools.product(args.structures, args.depths, args.modes, args.encoders):
        if mode == "text" and structure == "tree" and depth == 0:
            continue  # identical input to graph + text + d=0
        gnn_models = args.gnn_models if mode == "gnn" else ["none"]
        for gnn_model in gnn_models:
            for flags in text_flag_combinations(depth, fields, structure, mode):
                core = mode == "text" or gnn_model in full_gnn_models
                if design == "reduced" and not core and not is_slice_config(flags):
                    continue
                grid.append({
                    "dataset": args.dataset,
                    "target_attr": args.target,
                    "label_attr": args.label_attr,
                    "classification_mode": mode,
                    "structure": structure,
                    "tree": structure == "tree",
                    "encoder_key": encoder,
                    "encoder_name": ENCODER_MAP[encoder],
                    "gnn_model_name": gnn_model,
                    "edge_direction": (args.edge_direction or ("reverse" if structure == "tree" else "directed"))
                                      if mode == "gnn" else None,
                    "path_depth": depth,
                    **flags,
                    "mask_ratio": args.mask_ratio,
                    "mask_seed": args.mask_seed,
                    "seed": args.seed,
                    "design_part": "core" if core else "slice",
                })
    return grid


def config_signature(cfg: Dict) -> str:
    raw = "|".join(f"{k}={cfg.get(k)}" for k in SIGNATURE_KEYS)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def base_signature(cfg: Dict) -> str:
    """Signature of the configuration without its training seed (for seed replicates)."""
    raw = "|".join(f"{k}={cfg.get(k)}" for k in SIGNATURE_KEYS if k != "seed")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def in_shard(cfg: Dict, shard: str | None) -> bool:
    if not shard:
        return True
    i, n = (int(x) for x in shard.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"--shard {shard}: need 0 <= i < N")
    return int(base_signature(cfg), 16) % n == i


def make_text_cfg(cfg: Dict, meta: MetaDataConfig) -> TextBuildConfig:
    return TextBuildConfig(
        include_node_label=True,
        include_node_type=bool(cfg["use_node_type"]),
        include_node_attributes=bool(cfg["use_attributes"]),
        include_edge_label=bool(cfg["use_edge_label"]),
        include_edge_type=bool(cfg["use_edge_type"]),
        path_depth=cfg["path_depth"],
        is_tree=cfg["tree"],
        metadata=meta,
    )


def run_one(graphs, cfg: Dict, meta: MetaDataConfig, text_classifier: str, gnn_cfg: GNNExperimentConfig) -> Dict:
    from cm2ml_encodings_eval.pipeline import (
        run_gnn_node_classification,
        run_text_node_classification,
        run_tree_gnn_node_classification,
        run_tree_text_node_classification,
    )

    text_cfg = make_text_cfg(cfg, meta)
    if cfg["classification_mode"] == "text":
        exp_cfg = TextExperimentConfig(encoder_name=cfg["encoder_name"], classifier_name=text_classifier)
        fn = run_tree_text_node_classification if cfg["tree"] else run_text_node_classification
        return fn(graphs, text_cfg=text_cfg, exp_cfg=exp_cfg)

    gnn_cfg.model_name = cfg["gnn_model_name"]
    fn = run_tree_gnn_node_classification if cfg["tree"] else run_gnn_node_classification
    return fn(graphs, node_encoder_name=cfg["encoder_name"], text_cfg=text_cfg, gnn_cfg=gnn_cfg)


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def completed_signatures(*paths: Path) -> set:
    done = set()
    for out_path in paths:
        if not out_path.exists():
            continue
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok" and "signature" in rec:
                done.add(rec["signature"])
    return done


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run all node classification configuration combinations")
    p.add_argument("--dataset", default="modelset")
    p.add_argument("--target", default="abstract")
    p.add_argument("--label-attr", default="name")
    p.add_argument("--min-enr", type=float, default=1.2)
    p.add_argument("--min-edges", type=int, default=10)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--max-graphs", type=int, default=None)

    p.add_argument("--mask-ratio", type=float, default=0.2)
    p.add_argument("--mask-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42, help="training seed")

    p.add_argument("--encoders", nargs="+", default=["bow", "tfidf", "w2v", "bert-encoder"])
    p.add_argument("--structures", nargs="+", default=["graph", "tree"], choices=["graph", "tree"])
    p.add_argument("--depths", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--modes", nargs="+", default=["text", "gnn"], choices=["text", "gnn"])
    p.add_argument("--gnn-models", nargs="+", default=["sage", "gcn", "gat"])

    p.add_argument("--text-classifier", default="logreg", choices=["logreg", "linearsvm"])
    p.add_argument("--gnn-epochs", type=int, default=30)
    p.add_argument("--gnn-hidden-dim", type=int, default=128)
    p.add_argument("--gnn-num-layers", type=int, default=2)
    p.add_argument("--gnn-dropout", type=float, default=0.3)
    p.add_argument("--gnn-lr", type=float, default=1e-3)
    p.add_argument("--gnn-weight-decay", type=float, default=5e-4)
    p.add_argument("--edge-direction", default=None, choices=["directed", "reverse", "undirected"],
                   help="Default: directed for graphs (original behaviour), reverse/bottom-up for trees.")

    p.add_argument("--design", default="reduced", choices=DESIGNS)
    p.add_argument("--full-gnn-models", nargs="+", default=["sage"],
                   help="GNN models run over every text config in the reduced design")
    p.add_argument("--only-base-signatures", default=None,
                   help="File with one base signature per line; run only those configs (follow-ups C/D)")
    p.add_argument("--shard", default=None, help="i/N: run every N-th config (0-based i)")
    p.add_argument("--torch-threads", type=int, default=8)

    p.add_argument("--out-dir", default="results")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N pending combinations")
    p.add_argument("--dry-run", action="store_true", help="Print the grid and exit")
    p.add_argument("--cuda_devices", default=None,
                   help="Physical GPU indices (nvidia-smi order) allowed for this job, e.g. '1,2,3'. "
                        "Default: any GPU that passes the gpu_select rule.")
    p.add_argument("--gpu-mem-mb", type=int, default=4000,
                   help="Memory this job needs; a GPU is used only with >= 2x this free and <= 20 GB used.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    gpu_index = globals().get("_GPU_INDEX")

    from data_loading.models_dataset import get_models_dataset

    meta = make_metadata(args.dataset, args.target)
    dataset = get_models_dataset(
        dataset_name=args.dataset, min_enr=args.min_enr, min_edges=args.min_edges, reload=args.reload,
    )
    graphs = list(dataset.graphs)
    if args.max_graphs is not None:
        graphs = graphs[: args.max_graphs]
    deterministic_mask_graph_nodes(graphs, mask_ratio=args.mask_ratio, seed=args.mask_seed)

    fields = available_fields(graphs, meta)
    grid = build_grid(args, fields)
    if args.only_base_signatures:
        wanted = {l.strip() for l in Path(args.only_base_signatures).read_text().splitlines() if l.strip()}
        grid = [c for c in grid if base_signature(c) in wanted]
    grid = [c for c in grid if in_shard(c, args.shard)]
    torch.set_num_threads(args.torch_threads)
    if torch.cuda.is_available():
        total_mb = torch.cuda.get_device_properties(0).total_memory / 2**20
        # Hard cap at the declared need: the allocator frees its cache before it
        # would exceed this, so the job never holds more than it declared.
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_mb / total_mb))
    print(f"Fields present: {fields}")
    print(f"Design: {args.design}  shard: {args.shard or 'all'}  threads: {args.torch_threads}")
    print(f"Grid size: {len(grid)} runs")
    if args.dry_run:
        for cfg in grid:
            print(json.dumps(cfg))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"node_cls_grid_{args.dataset}_{args.target}"
    suffix = f"_shard{args.shard.replace('/', 'of')}" if args.shard else ""
    out_path = out_dir / f"{stem}{suffix}.jsonl"
    done = completed_signatures(*out_dir.glob(f"{stem}*.jsonl"))
    commit = git_commit()

    gnn_cfg = GNNExperimentConfig(
        hidden_dim=args.gnn_hidden_dim, num_layers=args.gnn_num_layers, dropout=args.gnn_dropout,
        learning_rate=args.gnn_lr, weight_decay=args.gnn_weight_decay, epochs=args.gnn_epochs,
        edge_direction=args.edge_direction,
    )

    ran = ok = failed = 0
    for cfg in grid:
        sig = config_signature(cfg)
        if sig in done:
            continue
        if args.limit is not None and ran >= args.limit:
            break
        ran += 1
        set_seed(args.seed)
        print(f"[{ran}] {sig} {json.dumps({k: cfg[k] for k in SIGNATURE_KEYS[2:-2]})}", flush=True)
        record = {
            "signature": sig,
            "base_signature": base_signature(cfg),
            "design": args.design,
            "shard": args.shard,
            "config": cfg,
            "metadata": DATASET_METADATA[(args.dataset, args.target)],
            "gnn": vars(gnn_cfg) if cfg["classification_mode"] == "gnn" else None,
            "text_classifier": args.text_classifier if cfg["classification_mode"] == "text" else None,
            "num_graphs": len(graphs),
            "git_commit": commit,
        }
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            record["result"] = run_one(graphs, cfg, meta, args.text_classifier, gnn_cfg)
            record["status"] = "ok"
            ok += 1
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            failed += 1
        record["seconds"] = round(time.time() - start, 2)
        record["gpu_mem_cap_mb"] = args.gpu_mem_mb if torch.cuda.is_available() else None
        record["gpu_index"] = gpu_index
        if torch.cuda.is_available():
            record["gpu_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
            record["gpu_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
            torch.cuda.empty_cache()
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"    -> {record['status']} in {record['seconds']}s "
              f"{record.get('result', {}).get('metrics', record.get('error'))}", flush=True)

    summary = {"dataset": args.dataset, "target": args.target, "design": args.design, "shard": args.shard,
               "grid": len(grid), "ran": ran,
               "ok": ok, "failed": failed, "output": str(out_path)}
    (out_dir / f"{stem}{suffix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
