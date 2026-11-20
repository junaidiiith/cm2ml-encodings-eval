"""Node / path text construction shared by every pipeline.

Leakage rules (enforced here and nowhere else):

* The target field (``metadata.cls``) is never written into any text.
* Key/value meta-properties named in ``metadata.leak_keys`` (or named like the
  target) are removed from every node, e.g. ``isAbstract: 1`` for UML. Plain
  feature-name lists are domain content and are kept (see filter_attributes).
* The node *type* (``metadata.type``) is hidden
    - for the node whose text is being built (it is the node being
      classified, or -- in a GNN -- the node's own input feature), and
    - for every test node (``masked=True``) and every tree reference node that
      points at a test node (``ref_masked=True``), wherever they appear.
  Types of train nodes reached through paths stay visible when
  ``include_node_type`` is on.

Paths are taken over the *undirected* neighbourhood; the direction of each
traversed edge is kept in the text as ``->`` (along the edge) or ``<-``
(against it).  Tree root nodes (``tree_role == "MODEL"``) are never traversed,
so a tree node's bag of paths stays inside its own object subtree.
"""

from collections import deque
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import networkx as nx

from cm2ml_encodings_eval.config import MetaDataConfig, TextBuildConfig

FORWARD = "->"
BACKWARD = "<-"
NON_TRAVERSABLE_ROLES = {"MODEL"}

Path = Tuple[List[str], List[str]]  # (nodes, directions between consecutive nodes)


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k}:{v}" for k, v in sorted(value.items()))
    return str(value)


def banned_keys(metadata: MetaDataConfig) -> set:
    return {metadata.cls, *metadata.leak_keys}


def filter_attributes(attrs, metadata: MetaDataConfig):
    """Drop meta-properties (key -> value) whose key is the target or a leak key.

    Only key/value meta-properties are filtered: dict items and ``(name, value)``
    pairs, e.g. CM2ML's ``isAbstract: 1``, which state the node's own target.
    Plain lists of feature names (ModelSet's ``['request', 'response']``) are
    domain content -- a metaclass ``Class`` owning a feature called
    ``isAbstract`` says nothing about whether ``Class`` itself is abstract -- so
    they are kept.
    """
    banned = banned_keys(metadata)
    if attrs is None:
        return None
    if isinstance(attrs, dict):
        return {k: v for k, v in attrs.items() if str(k) not in banned}
    if isinstance(attrs, (list, tuple, set)):
        return [
            item for item in attrs
            if not (isinstance(item, (list, tuple)) and len(item) == 2 and str(item[0]) in banned)
        ]
    return attrs


def is_hidden(data: Dict) -> bool:
    """True for test nodes and for tree nodes that reference a test node."""
    return bool(data.get("masked", False)) or bool(data.get("ref_masked", False))


def is_labelled(data: Dict, metadata: MetaDataConfig) -> bool:
    """A node that takes part in node classification (train or test)."""
    return (
        data.get(metadata.cls) is not None
        and "masked" in data
        and data.get("tree_role", "OBJ") == "OBJ"
    )


def node_text(
    data: Dict,
    metadata: MetaDataConfig,
    include_label: bool,
    include_type: bool,
    include_attrs: bool,
    hide_type: bool = False,
) -> str:
    parts: List[str] = []
    if include_label:
        parts.append(_stringify(data.get(metadata.label, None)))
    type_key = metadata.type
    if include_type and not hide_type:
        type_val = data.get(type_key, None)
        if type_val is not None:
            parts.append(f"{type_key} = {_stringify(type_val)}")
    if include_attrs:
        attrs = filter_attributes(data.get(metadata.attributes, []), metadata)
        if attrs:
            parts.append(f"{metadata.attributes} = {_stringify(attrs)}")
    return " ".join(p for p in parts if p).strip()


def edge_text(data: Dict, metadata: MetaDataConfig, include_label: bool, include_type: bool) -> str:
    parts: List[str] = []
    if include_label:
        parts.append(_stringify(data.get(metadata.edge_label, None)))
    if include_type:
        parts.append(_stringify(data.get(metadata.edge_type, None)))
    return " ".join(p for p in parts if p).strip()


def _neighbours(graph: nx.DiGraph, node: str) -> Iterator[Tuple[str, str]]:
    for v in graph.successors(node):
        yield v, FORWARD
    for v in graph.predecessors(node):
        yield v, BACKWARD


def _traversable(graph: nx.DiGraph, node: str) -> bool:
    return graph.nodes[node].get("tree_role") not in NON_TRAVERSABLE_ROLES


def paths_from_node(graph: nx.DiGraph, start: str, depth: int, max_paths: int) -> List[Path]:
    """Breadth-first simple paths of length <= depth over the undirected neighbourhood."""
    if depth <= 0:
        return [([start], [])]
    queue = deque([([start], [])])
    out: List[Path] = []
    while queue and len(out) < max_paths:
        nodes, dirs = queue.popleft()
        out.append((nodes, dirs))
        if len(nodes) - 1 >= depth:
            continue
        for nbr, direction in _neighbours(graph, nodes[-1]):
            if nbr in nodes or not _traversable(graph, nbr):
                continue
            queue.append((nodes + [nbr], dirs + [direction]))
    return out


def path_to_text(
    graph: nx.DiGraph,
    path: Path,
    metadata: MetaDataConfig,
    include_node_label: bool,
    include_node_type: bool,
    include_node_attributes: bool,
    include_edge_label: bool,
    include_edge_type: bool,
) -> str:
    nodes, dirs = path
    if not nodes:
        return ""
    # The start node is the node being described: its type is always hidden.
    parts = [
        node_text(
            graph.nodes[nodes[0]], metadata,
            include_node_label, include_node_type, include_node_attributes,
            hide_type=True,
        )
    ]
    for i in range(1, len(nodes)):
        u, v, direction = nodes[i - 1], nodes[i], dirs[i - 1]
        e_data = graph.edges[u, v] if direction == FORWARD else graph.edges[v, u]
        e_txt = edge_text(e_data, metadata, include_edge_label, include_edge_type)
        parts.append(f"{direction} {e_txt}".strip())
        v_data = graph.nodes[v]
        parts.append(
            node_text(
                v_data, metadata,
                include_node_label, include_node_type, include_node_attributes,
                hide_type=is_hidden(v_data),
            )
        )
    return " ".join(p for p in parts if p)


def build_node_text(graph: nx.DiGraph, node: str, cfg: TextBuildConfig) -> str:
    """Text for one node: its own information (d=0) or its bag of paths (d>0)."""
    paths = paths_from_node(graph, node, cfg.path_depth, cfg.max_paths_per_node)
    path_texts = [
        path_to_text(
            graph, p, cfg.metadata,
            cfg.include_node_label, cfg.include_node_type, cfg.include_node_attributes,
            cfg.include_edge_label, cfg.include_edge_type,
        )
        for p in paths
    ]
    return " || ".join(t for t in path_texts if t).strip()


def build_graph_texts(graph: nx.DiGraph, cfg: TextBuildConfig) -> List[str]:
    """Texts for every node of a graph, in ``graph.nodes()`` order (GNN features)."""
    return [build_node_text(graph, n, cfg) for n in graph.nodes()]


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
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[Tuple[int, str]]]]:
    cfg = TextBuildConfig(
        include_node_label=include_node_label,
        include_node_type=include_node_type,
        include_node_attributes=include_node_attributes,
        include_edge_label=include_edge_label,
        include_edge_type=include_edge_type,
        path_depth=path_depth,
        max_paths_per_node=max_paths_per_node,
        metadata=metadata,
    )
    texts: Dict[str, List[str]] = {"train": [], "test": []}
    labels: Dict[str, List[str]] = {"train": [], "test": []}
    indices: Dict[str, List[Tuple[int, str]]] = {"train": [], "test": []}

    for gid, graph in enumerate(graphs):
        for node_id, data in graph.nodes(data=True):
            if not is_labelled(data, metadata):
                continue
            split = "test" if data["masked"] else "train"
            texts[split].append(build_node_text(graph, node_id, cfg))
            labels[split].append(str(data[metadata.cls]))
            indices[split].append((gid, str(node_id)))
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
    flags = (include_node_label, include_node_type, include_node_attributes, include_edge_label, include_edge_type)

    for gid, graph in enumerate(graphs):
        for u, v, data in graph.edges(data=True):
            y = data.get(metadata.cls)
            if y is None:
                continue
            src_txt = [path_to_text(graph, p, metadata, *flags) for p in paths_from_node(graph, u, path_depth, max_paths_per_node)]
            dst_txt = [path_to_text(graph, p, metadata, *flags) for p in paths_from_node(graph, v, path_depth, max_paths_per_node)]
            e_txt = edge_text(data, metadata, include_edge_label, include_edge_type)
            text = f"SRC: {' || '.join(src_txt)} [EDGE] {e_txt} [DST] {' || '.join(dst_txt)}"
            texts.append(text.strip())
            labels.append(str(y))
            indices.append((gid, str(u), str(v)))

    return texts, labels, indices
