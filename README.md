# cm2ml-encodings-eval

Package for evaluating node/edge classification on typed `networkx` graphs with:
- A1: text-only node/edge classification from node/edge text
- A2: bag-of-paths text (`k`-hop path depth; `k=0` recovers A1)
- A3: GNN training with encoded node/edge features
- Tree serialization: `MODEL -> OBJ/ASSOC -> ATTS/...` then node classification on tree nodes

## Results (CM2ML paper, Section 5)

The node-classification results used in the paper are in [`results/`](results/). [`results/README.md`](results/README.md) documents the pipeline fixes, the experimental setup, the top configurations, the factor analyses, the tree vs graph comparison, and how to reproduce them with `run_all_node_cls_combinations.py` and the scripts in [`analysis/`](analysis/).

This pipeline implements the encoding choices studied in the paper in Python (graph or tree structure, bag-of-paths context, and the text encoders listed below). It does not call the [CM2ML](https://github.com/borkdominik/CM2ML) framework itself.

## Install

Python 3.10 or later. The paper's grid uses W2V, so install the `word2vec` extra:

```bash
pip install -e ".[word2vec]"
```

To also run the analysis scripts (figures need `kaleido`) and the tests:

```bash
pip install -e ".[word2vec,analysis,test]"
```

Without an NVIDIA GPU, the scripts run on the CPU.

## Graph Assumptions

- Node target label is stored in a node attribute, e.g. `graph.nodes[n]["cls"]`
- Edge target label is stored in an edge attribute, e.g. `graph.edges[u, v]["edge_cls"]`
- Node text fields can come from node id/label, `name`, `type`, and `attributes`
- Edge text fields can come from edge `name` and `type`

## Quick Start

```python
import networkx as nx
from cm2ml_encodings_eval import (
    SplitConfig, TextBuildConfig, TextExperimentConfig, GNNExperimentConfig,
    run_text_node_classification, run_text_edge_classification,
    run_gnn_node_classification, run_gnn_edge_classification,
)

graphs = [your_networkx_digraph]

split_cfg = SplitConfig(test_ratio=0.2, random_state=42)
text_cfg = TextBuildConfig(path_depth=1)  # bag-of-paths; use 0 for A1

res_a2_node = run_text_node_classification(
    graphs,
    target_attr="cls",
    split_cfg=split_cfg,
    text_cfg=text_cfg,
    exp_cfg=TextExperimentConfig(encoder_name="tfidf", classifier_name="logreg"),
)

res_a2_edge = run_text_edge_classification(
    graphs,
    target_attr="edge_cls",
    split_cfg=split_cfg,
    text_cfg=text_cfg,
    exp_cfg=TextExperimentConfig(encoder_name="bert", classifier_name="linearsvm"),
)

res_a3_node = run_gnn_node_classification(
    graphs[0],
    target_attr="cls",
    node_encoder_name="tfidf",
    split_cfg=split_cfg,
    text_cfg=text_cfg,
    gnn_cfg=GNNExperimentConfig(model_name="gcn", epochs=100),
)

res_a3_edge = run_gnn_edge_classification(
    graphs[0],
    target_attr="edge_cls",
    node_encoder_name="tfidf",
    edge_encoder_name="tfidf",
    split_cfg=split_cfg,
    text_cfg=text_cfg,
    gnn_cfg=GNNExperimentConfig(model_name="graphsage", epochs=100),
)
```

## Tree-Transformed Node Classification

```python
from cm2ml_encodings_eval import (
    TreeSerializationConfig,
    tree_to_bracket_text,
    serialize_graph_to_tree,
    run_tree_text_node_classification,
    run_tree_gnn_node_classification,
)

tree_cfg = TreeSerializationConfig()
tree_graph = serialize_graph_to_tree(your_networkx_digraph, node_target_attr="cls", tree_cfg=tree_cfg)
tree_string = tree_to_bracket_text(tree_graph)  # optional bracket serialization

res_tree_text = run_tree_text_node_classification(
    graphs=[your_networkx_digraph],
    tree_cfg=tree_cfg,
)

res_tree_gnn = run_tree_gnn_node_classification(
    graphs_or_dataset=[your_networkx_digraph],
    tree_cfg=tree_cfg,
)
```

## Supported Encoders

- `onehot`
- `bow`
- `tfidf`
- `word2vec` (requires `gensim`)
- `custom_w2v` (loads `out/skip_gram_modelling/skip_gram_vectors.kv`)
- `custom_w2v:/absolute/or/relative/path/to/model.kv`
- `bert` (frozen `bert-base-uncased`; the embedding of the `[CLS]` token)

## Supported Text Classifiers

- `logreg`
- `linearsvm`

## Supported GNN Models

- `gcn`
- `graphsage` / `sage`
- `gat`

## Data

The datasets are not part of this repository. Place them in `datasets/` (ignored by git):

| Dataset | Source | Expected location |
| --- | --- | --- |
| EAModelSet (ArchiMate) | Glaser et al., PoEM 2023, [doi:10.1007/978-3-031-48583-1_2](https://doi.org/10.1007/978-3-031-48583-1_2) | `datasets/eamodelset/processed-models/<model id>/model.json` |
| ModelSet (Ecore part) | López et al., SoSyM 2022, [doi:10.1007/s10270-021-00929-3](https://doi.org/10.1007/s10270-021-00929-3) | `datasets/modelset/ecore.jsonl` |

On the first run, the loaders parse the models and cache them in `datasets/pickles/`.

## Reproducing the Paper Results

All commands run from the repository root. The full procedure, including the ModelSet cache, the sharded ModelSet run, and the checks, is in [`results/README.md`, section 10](results/README.md#10-reproducing).

```bash
# ModelSet cache with the declared abstract labels (once)
python -c "from data_loading.models_dataset import get_models_dataset as g; \
           g('modelset', min_enr=1.2, min_edges=10, reload=True, remove_duplicates=True)"

# Grid (reduced design) for the three tasks; results go to results/ and resume automatically
python run_all_node_cls_combinations.py --dataset eamodelset --target layer
python run_all_node_cls_combinations.py --dataset eamodelset --target type
python run_all_node_cls_combinations.py --dataset modelset --target abstract

# Reports, figures, and the interactive explorer, from the jsonl files in results/
python analysis/node_cls_report.py
python analysis/structure_comparison.py
python analysis/build_explorer.py
python analysis/modelset_rule_baseline.py
```

The first three analysis scripts need only the committed `results/*.jsonl` files; the rule baseline also needs the ModelSet data.

To check the setup on a small sample before a full run, print the grid or run a few configurations into a separate folder:

```bash
python run_all_node_cls_combinations.py --dataset eamodelset --target layer --dry-run
python run_all_node_cls_combinations.py --dataset eamodelset --target layer --max-graphs 20 --limit 4 --out-dir out/smoke
```

In the grid, the encoders are named `bow`, `tfidf`, `w2v` (Word2Vec trained on the corpus), and `bert-encoder` (the `bert` encoder above). `python run_all_node_cls_combinations.py --help` lists all options.

## YAML Runner

Run individual experiment configurations from YAML (the schema is in [`configs/README.md`](configs/README.md)):

```bash
python yaml_runner.py --config configs/pipeline_text_gnn.yaml --output out/text_gnn.json
python yaml_runner.py --config configs/pipeline_tree.yaml --output out/tree.json
```

Both example configs use ModelSet (see Data).
