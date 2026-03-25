# cm2ml-encodings-eval

Package for evaluating node/edge classification on typed `networkx` graphs with:
- A1: text-only node/edge classification from node/edge text
- A2: bag-of-paths text (`k`-hop path depth; `k=0` recovers A1)
- A3: GNN training with encoded node/edge features
- Tree serialization: `MODEL -> OBJ/ASSOC -> ATTS/...` then node classification on tree nodes

## Install

```bash
pip install -e .
```

Optional Word2Vec support:

```bash
pip install -e .[word2vec]
```

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
- `bert` (BERT CLS embeddings)

## Supported Text Classifiers

- `logreg`
- `linearsvm`
- `bert_classifier` (fine-tune `AutoModelForSequenceClassification`)

## Supported GNN Models

- `gcn`
- `graphsage` / `sage`
- `gat`

## Reproducible Example

Run:

```bash
python3 examples/run_pipeline.py
python3 examples/run_tree_pipeline.py
```

## YAML Runner

Run multiple experiment configurations from YAML:

```bash
python3 -m cm2ml_encodings_eval.yaml_runner --config configs/pipeline_text_gnn.yaml --output results/text_gnn.json
python3 -m cm2ml_encodings_eval.yaml_runner --config configs/pipeline_tree.yaml --output results/tree.json
```

YAML files in this repo:

- `configs/pipeline_text_gnn.yaml`
- `configs/pipeline_tree.yaml`
