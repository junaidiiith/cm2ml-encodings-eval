from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import numpy as np


@dataclass
class TextTaskData:
    texts: List[str]
    y: np.ndarray
    indices: np.ndarray
    class_names: List[str]


@dataclass
class SplitData:
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class TypedGraphCollection:
    graphs: List[nx.DiGraph]

    def __len__(self) -> int:
        return len(self.graphs)


NodeRecord = Dict[str, Any]
EdgeRecord = Dict[str, Any]
