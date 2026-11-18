from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SplitConfig:
    test_ratio: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class MetaDataConfig:
    """Which node/edge fields play which role.

    cls        -- the node field being predicted (never emitted into text)
    label      -- the node field used as the node's name
    attributes -- the node field holding attributes / meta-properties
    type       -- the node field holding the node's (metamodel) type
    edge_label -- the edge field holding the edge's name
    edge_type  -- the edge field holding the edge's type
    leak_keys  -- attribute / meta-property names that reveal `cls`; they are
                  removed from every node (e.g. ``isAbstract`` for UML
                  abstractness). The node type is handled separately: it is
                  always hidden for the classified node and for test nodes.
    """

    cls: str = "abstract"
    label: str = "name"
    attributes: str = "attributes"
    type: str = "type"
    edge_label: str = "name"
    edge_type: str = "type"
    leak_keys: Tuple[str, ...] = ()


@dataclass
class TextBuildConfig:
    include_node_label: bool = True
    include_node_type: bool = False
    include_node_attributes: bool = True
    include_edge_label: bool = True
    include_edge_type: bool = True
    path_depth: int = 0
    max_paths_per_node: int = 64
    
    is_tree: bool = False

    metadata: MetaDataConfig = field(default_factory=MetaDataConfig)
    

@dataclass
class TextExperimentConfig:
    encoder_name: str = "tfidf"
    classifier_name: str = "logreg"


@dataclass
class GNNExperimentConfig:
    model_name: str = "gcn"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 200
    # How graph edges feed message passing (PyG passes messages source -> target):
    #   "directed"   -- along the original edges (graph default, as in the original runs)
    #   "reverse"    -- against them; for trees this is bottom-up, so each OBJ node
    #                   aggregates its own subtree (tree default)
    #   "undirected" -- both ways, aggregated together (loses direction)
    #   None         -- pick the default for the structure
    edge_direction: str | None = None


@dataclass
class TreeSerializationConfig:
    model_token: str = "MODEL"
    object_token: str = "OBJ"
    atts_token: str = "ATTS"
    assoc_token: str = "ASSOC"
    include_associations: bool = True
