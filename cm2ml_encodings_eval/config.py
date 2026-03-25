from dataclasses import dataclass


@dataclass
class SplitConfig:
    test_ratio: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class MetaDataConfig:
    cls: str = "abstract"
    label: str = "name"
    attributes: str = "attributes"
    

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
    
    metadata = MetaDataConfig()
    

@dataclass
class TextExperimentConfig:
    encoder_name: str = "tfidf"
    classifier_name: str = "logreg"


@dataclass
class BertClassifierConfig:
    model_name: str = "bert-base-uncased"
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    max_length: int = 256
    weight_decay: float = 0.01


@dataclass
class GNNExperimentConfig:
    model_name: str = "gcn"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 200


@dataclass
class TreeSerializationConfig:
    model_token: str = "MODEL"
    object_token: str = "OBJ"
    atts_token: str = "ATTS"
    assoc_token: str = "ASSOC"
    include_associations: bool = True
