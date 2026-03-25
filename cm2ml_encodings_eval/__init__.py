"""cm2ml encodings evaluation package."""

import random
import torch
import numpy as np


from .config import (
    SplitConfig,
    TextBuildConfig,
    TextExperimentConfig,
    BertClassifierConfig,
    GNNExperimentConfig,
    TreeSerializationConfig,
)
from .pipeline import (
    run_text_node_classification,
    run_text_edge_classification,
    run_gnn_node_classification,
    run_gnn_edge_classification,
    run_tree_text_node_classification,
    run_tree_gnn_node_classification,
)
from .tree_transform import serialize_graph_to_tree, serialize_graphs_to_trees, tree_to_bracket_text


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



__all__ = [
    "SplitConfig",
    "TextBuildConfig",
    "TextExperimentConfig",
    "BertClassifierConfig",
    "GNNExperimentConfig",
    "TreeSerializationConfig",
    "run_text_node_classification",
    "run_text_edge_classification",
    "run_gnn_node_classification",
    "run_gnn_edge_classification",
    "run_tree_text_node_classification",
    "run_tree_gnn_node_classification",
    "serialize_graph_to_tree",
    "serialize_graphs_to_trees",
    "tree_to_bracket_text",
]
