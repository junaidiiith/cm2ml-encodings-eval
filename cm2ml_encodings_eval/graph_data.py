from __future__ import annotations

from typing import Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

from cm2ml_encodings_eval.config import MetaDataConfig

from .encoders import BaseTextEncoder
from .text import edge_text, node_text


def _to_dense_if_sparse(x):
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def encode_node_texts(
    graph: nx.DiGraph, encoder: BaseTextEncoder, metadata: MetaDataConfig, include_node_label=True, include_node_type=True, include_node_attributes=True):
    node_ids = list(graph.nodes())
    texts = [
        node_text(graph.nodes[n], metadata, include_node_label, include_node_type, include_node_attributes)
        for n in node_ids
    ]
    x = encoder.fit_transform(texts)
    return node_ids, _to_dense_if_sparse(x)


def encode_edge_texts(graph: nx.DiGraph, encoder: BaseTextEncoder, include_edge_label=True, include_edge_type=True):
    edges = list(graph.edges())
    texts = [edge_text(graph.edges[e], include_edge_label, include_edge_type) for e in edges]
    e = encoder.fit_transform(texts)
    return edges, _to_dense_if_sparse(e)


def pyg_from_nx(
    graph: nx.DiGraph,
    node_encoder: BaseTextEncoder,
    edge_encoder: BaseTextEncoder | None,
    metadata: MetaDataConfig,
    task: str,
    include_node_label: bool = True,
    include_node_type: bool = True,
    include_node_attributes: bool = True,
    include_edge_label: bool = True,
    include_edge_type: bool = True,
) -> Tuple[Data, LabelEncoder]:

    node_ids, x_np = encode_node_texts(
        graph,
        node_encoder,
        metadata=metadata,
        include_node_label=include_node_label,
        include_node_type=include_node_type,
        include_node_attributes=include_node_attributes,
    )
    nid_to_i = {n: i for i, n in enumerate(node_ids)}

    edges = [(nid_to_i[u], nid_to_i[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    x = torch.tensor(x_np, dtype=torch.float32)

    if task == "node":
        y_raw = []
        valid_idx = []
        for i, n in enumerate(node_ids):
            y = graph.nodes[n].get(metadata.cls)
            if y is not None:
                y_raw.append(str(y))
                valid_idx.append(i)

        if not y_raw:
            raise ValueError(f"No node labels found for metadata class '{metadata.cls}'")
        if len(y_raw) < 2:
            raise ValueError("Need at least 2 labeled nodes for train/test split.")

        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw)
        y = torch.full((len(node_ids),), -1, dtype=torch.long)
        y[torch.tensor(valid_idx, dtype=torch.long)] = torch.tensor(y_enc, dtype=torch.long)

        test_nodes = [n for n in graph.nodes if 'masked' in graph.nodes[n] and graph.nodes[n]['masked']]
        test_idx = [nid_to_i[n] for n in test_nodes]
        train_idx = [i for i in range(len(node_ids)) if i not in test_idx]
        train_mask = torch.zeros(len(node_ids), dtype=torch.bool)
        test_mask = torch.zeros(len(node_ids), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
        return data, le

    if task == "edge":
        edge_pairs_raw = list(graph.edges())
        y_raw = []
        edge_pairs = []
        for (u, v) in edge_pairs_raw:
            lbl = graph.edges[u, v].get(metadata.cls)
            if lbl is None:
                continue
            y_raw.append(str(lbl))
            edge_pairs.append((nid_to_i[u], nid_to_i[v]))

        if not y_raw:
            raise ValueError(f"No edge labels found for metadata class '{metadata.cls}'")
        if len(y_raw) < 2:
            raise ValueError("Need at least 2 labeled edges for train/test split.")

        le = LabelEncoder()
        y_np = le.fit_transform(y_raw)

        test_edges = [(u, v) for (u, v) in edge_pairs if 'masked' in graph.edges[u, v] and graph.edges[u, v]['masked']]
        test_idx = [i for i, (u, v) in enumerate(edge_pairs) if (u, v) in test_edges]
        train_idx = [i for i in range(len(edge_pairs)) if i not in test_idx]
        

        edge_pairs_t = torch.tensor(edge_pairs, dtype=torch.long)
        y = torch.tensor(y_np, dtype=torch.long)
        train_mask = torch.zeros(len(y_np), dtype=torch.bool)
        test_mask = torch.zeros(len(y_np), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        if edge_encoder is not None and len(edge_pairs_raw) > 0:
            edge_texts = [edge_text(graph.edges[u, v], include_edge_label, include_edge_type) for u, v in graph.edges()]
            edge_attr_all = _to_dense_if_sparse(edge_encoder.fit_transform(edge_texts))
            lookup = {(nid_to_i[u], nid_to_i[v]): i for i, (u, v) in enumerate(graph.edges())}
            chosen = [lookup[p] for p in edge_pairs]
            edge_attr = torch.tensor(edge_attr_all[chosen], dtype=torch.float32)
        else:
            edge_attr = None

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_pairs=edge_pairs_t,
            edge_y=y,
            edge_train_mask=train_mask,
            edge_test_mask=test_mask,
            edge_attr=edge_attr,
        )
        return data, le

    raise ValueError("task must be one of {'node', 'edge'}")
