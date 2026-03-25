import numpy as np
from sklearn.model_selection import train_test_split
import random
import networkx as nx


def train_test_indices(y: np.ndarray, test_ratio: float, random_state: int, stratify: bool = True):
    idx = np.arange(len(y))
    strat = y if stratify and len(np.unique(y)) > 1 else None
    try:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_ratio,
            random_state=random_state,
            stratify=strat,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_ratio,
            random_state=random_state,
            stratify=None,
        )
    return np.asarray(train_idx), np.asarray(test_idx)


def mask_from_indices(n: int, idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def mask_graph_nodes(graphs: list[nx.DiGraph], mask_ratio: float = 0.2):
    for graph in graphs:
        total_nodes = graph.number_of_nodes()
        num_mask = int(total_nodes * mask_ratio)
        nodes = list(graph.nodes())
        masked_nodes = set(random.sample(nodes, num_mask))
        for node in masked_nodes:
            graph.nodes[node]['masked'] = True
