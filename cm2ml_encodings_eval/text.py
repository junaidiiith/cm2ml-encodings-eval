from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx

from cm2ml_encodings_eval.config import MetaDataConfig


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k}:{v}" for k, v in sorted(value.items()))
    return str(value)


def node_text(data: Dict, metadata: MetaDataConfig, include_label: bool, include_type: bool, include_attrs: bool) -> str:
    parts: List[str] = []
    label_key = metadata.label
    attributes_key = metadata.attributes
    cls_key = metadata.cls
    
    if include_label:
        parts.append(_stringify(data.get(label_key, None)))
    if include_type:
        parts.append(f"{cls_key} = {_stringify(data.get(cls_key, None))}")
    if include_attrs:
        attrs = data.get(attributes_key, [])
        if attrs:
            parts.append(f"{attributes_key} = {_stringify(attrs)}")
        
    return " ".join(p for p in parts if p).strip()


def edge_text(data: Dict, metadata: MetaDataConfig, include_label: bool, include_type: bool) -> str:
    parts: List[str] = []
    label_key = metadata.label
    type_key = metadata.cls
    if include_label:
        parts.append(_stringify(data.get(label_key, None)))
    if include_type:
        parts.append(_stringify(data.get(type_key, None)))
    return " ".join(p for p in parts if p).strip()


def paths_from_node(graph: nx.DiGraph, start: str, depth: int, max_paths: int) -> List[List[str]]:
    if depth <= 0:
        return [[start]]
    queue = deque([(start, [start])])
    out: List[List[str]] = []
    while queue and len(out) < max_paths:
        curr, path = queue.popleft()
        out.append(path)
        if len(path) - 1 >= depth:
            continue
        for nbr in graph.neighbors(curr):
            if nbr in path:
                continue
            queue.append((nbr, path + [nbr]))
    return out


def path_to_text(
    graph: nx.DiGraph,
    path: Sequence[str],
    metadata: MetaDataConfig,
    include_node_label: bool,
    include_node_type: bool,
    include_node_attributes: bool,
    include_edge_label: bool,
    include_edge_type: bool,
) -> str:
    if not path:
        return ""
    start_node = path[0]
    initial_mask = graph.nodes[start_node].get("masked", False)
    graph.nodes[start_node]["masked"] = True
    parts = [
        node_text(
            graph.nodes[path[0]],
            metadata,
            include_node_label,
            include_node_type,
            include_node_attributes,
        )
    ]
    graph.nodes[start_node]["masked"] = initial_mask
    
    for i in range(1, len(path)):
        u, v = path[i - 1], path[i]
        parts.append(edge_text(graph.edges[u, v], metadata, include_edge_label, include_edge_type))
        parts.append(
            node_text(
                graph.nodes[v],
                metadata,
                include_node_label,
                include_node_type,
                include_node_attributes,
            )
        )
    return " | ".join(p for p in parts if p)


def build_node_corpus(
    graphs: Iterable[nx.DiGraph],
    metadata: MetaDataConfig,
    include_node_label: bool,
    include_node_type: bool,
    include_node_attributes: bool,
    include_edge_label: bool,
    include_edge_type: bool,
    path_depth: int,
    max_paths_per_node: int,
) -> Tuple[List[str], List[str], List[Tuple[int, str]]]:
    texts: Dict[List[str]] = {'train': [], 'test': []}
    labels: Dict[List[str]] = {'train': [], 'test': []}
    indices: Dict[List[Tuple[int, str]]] = {'train': [], 'test': []}

    for gid, graph in enumerate(graphs):
        for node_id, data in graph.nodes(data=True):
            y = data.get(metadata.cls)
            if y is None:
                continue
            paths = paths_from_node(graph, node_id, path_depth, max_paths_per_node)
            
            path_texts = [
                path_to_text(
                    graph,
                    p,
                    metadata,
                    include_node_label,
                    include_node_type,
                    include_node_attributes,
                    include_edge_label,
                    include_edge_type,
                )
                for p in paths
            ]
            text = " || ".join(t for t in path_texts if t).strip()
            masked_nodes = [n for n in graph.nodes if 'masked' in graph.nodes[n] and graph.nodes[n]['masked']]
            unmasked_nodes = [n for n in graph.nodes if 'masked' in graph.nodes[n] and graph.nodes[n]['masked'] == False]
            if node_id in masked_nodes:
                texts['test'].append(text)
                labels['test'].append(str(y))
                indices['test'].append((gid, str(node_id)))
            elif node_id in unmasked_nodes:
                texts['train'].append(text)
                labels['train'].append(str(y))
                indices['train'].append((gid, str(node_id)))
                
    return texts, labels, indices


def build_edge_corpus(
    graphs: Iterable[nx.DiGraph],
    metadata: MetaDataConfig,
    include_node_label: bool,
    include_node_type: bool,
    include_node_attributes: bool,
    include_edge_label: bool,
    include_edge_type: bool,
    path_depth: int,
    max_paths_per_node: int,
    test_ratio: float = 0.2,
) -> Tuple[List[str], List[str], List[Tuple[int, str, str]]]:
    texts: List[str] = []
    labels: List[str] = []
    indices: List[Tuple[int, str, str]] = []

    for gid, graph in enumerate(graphs):
        for u, v, data in graph.edges(data=True):
            y = data.get(metadata.cls)
            if y is None:
                continue
            src_paths = paths_from_node(graph, u, path_depth, max_paths_per_node)
            dst_paths = paths_from_node(graph, v, path_depth, max_paths_per_node)
            src_txt = [
                path_to_text(
                    graph,
                    p,
                    metadata,
                    include_node_label,
                    include_node_type,
                    include_node_attributes,
                    include_edge_label,
                    include_edge_type,
                )
                for p in src_paths
            ]
            dst_txt = [
                path_to_text(
                    graph,
                    p,
                    metadata,
                    include_node_label,
                    include_node_type,
                    include_node_attributes,
                    include_edge_label,
                    include_edge_type,
                )
                for p in dst_paths
            ]
            e_txt = edge_text(data, metadata, include_edge_label, include_edge_type)
            text = f"SRC: {' || '.join(src_txt)} [EDGE] {e_txt} [DST] {' || '.join(dst_txt)}"
            texts.append(text.strip())
            labels.append(str(y))
            indices.append((gid, str(u), str(v)))

    return texts, labels, indices
