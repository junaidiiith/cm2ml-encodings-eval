"""Graph -> tree serialization used by the tree-based structural encoding.

Tree shape (one tree per model)::

    MODEL
    └── OBJ  (name, type, attributes, target, masked)       <- classified node
        ├── ATTS
        │   └── ATTR_NAME ── ATTR_VALUE
        └── ASSOCS
            └── REF  (name/type of the target object)       <- one per outgoing edge;
                                                               the tree edge ASSOCS->REF
                                                               carries the original
                                                               edge's label and type

* OBJ nodes copy the source node's own information, so the d=0 text of an OBJ
  node is identical to the d=0 text of the source graph node; tree and graph
  differ only in adjacency.
* Only OBJ nodes carry the target and the ``masked`` flag, so only they are
  classified.
* Tree roles live in ``tree_role``; structural tree nodes have no ``type``
  field, so they never collide with a ``type`` target (EA Type).
* REF nodes carry the referenced object's type with ``ref_masked`` set when that
  object is a test node, so the text builder hides it.
* Key/value meta-properties named like the target or listed in
  ``metadata.leak_keys`` are dropped from the tree entirely (e.g. ``isAbstract: 1``
  for UML abstractness); plain feature-name lists are kept (see
  ``text.filter_attributes``).
"""

from __future__ import annotations

from typing import Iterable, List

import networkx as nx

from .config import MetaDataConfig, TreeSerializationConfig
from .text import filter_attributes


def _as_attr_pairs(raw_attrs):
    if raw_attrs is None:
        return []
    if isinstance(raw_attrs, dict):
        pairs = []
        for k, v in raw_attrs.items():
            if isinstance(v, (list, tuple, set)):
                for item in v:
                    pairs.append((str(k), str(item)))
            else:
                pairs.append((str(k), str(v)))
        return pairs
    if isinstance(raw_attrs, (list, tuple, set)):
        return [(str(a), "") for a in raw_attrs]
    return [("value", str(raw_attrs))]


def serialize_graph_to_tree(
    graph: nx.DiGraph,
    node_target_attr: str | None = None,
    tree_cfg: TreeSerializationConfig | None = None,
    metadata: MetaDataConfig | None = None,
) -> nx.DiGraph:
    cfg = tree_cfg or TreeSerializationConfig()
    meta = metadata or MetaDataConfig(cls=node_target_attr or "abstract")
    if node_target_attr is not None and node_target_attr != meta.cls:
        raise ValueError(f"node_target_attr={node_target_attr!r} does not match metadata.cls={meta.cls!r}")

    tree = nx.DiGraph()
    root = cfg.model_token
    tree.add_node(root, **{meta.label: cfg.model_token, "tree_role": "MODEL"})

    obj_id = {n: f"OBJ::{i}" for i, n in enumerate(graph.nodes())}

    for src_node, src_data in graph.nodes(data=True):
        oid = obj_id[src_node]
        obj = {
            "tree_role": "OBJ",
            "source_node": str(src_node),
            meta.label: src_data.get(meta.label, ""),
            meta.attributes: filter_attributes(src_data.get(meta.attributes), meta),
        }
        if meta.type in src_data:
            obj[meta.type] = src_data[meta.type]
        if src_data.get(meta.cls) is not None:
            obj[meta.cls] = src_data[meta.cls]
        if "masked" in src_data:
            obj["masked"] = bool(src_data["masked"])
        tree.add_node(oid, **obj)
        tree.add_edge(root, oid)

        attr_pairs = _as_attr_pairs(filter_attributes(src_data.get(meta.attributes), meta))
        if attr_pairs:
            atts_id = f"ATTS::{oid}"
            tree.add_node(atts_id, **{meta.label: cfg.atts_token, "tree_role": "ATTS"})
            tree.add_edge(oid, atts_id)
            for j, (attr_name, attr_value) in enumerate(attr_pairs):
                an_id = f"ATTR::{oid}::{j}"
                tree.add_node(an_id, **{meta.label: attr_name, "tree_role": "ATTR_NAME"})
                tree.add_edge(atts_id, an_id)
                if attr_value:
                    av_id = f"ATTR_VAL::{oid}::{j}"
                    tree.add_node(av_id, **{meta.label: attr_value, "tree_role": "ATTR_VALUE"})
                    tree.add_edge(an_id, av_id)

    if cfg.include_associations:
        for src_node in graph.nodes():
            out_edges = list(graph.out_edges(src_node, data=True))
            if not out_edges:
                continue
            oid = obj_id[src_node]
            assocs_id = f"ASSOCS::{oid}"
            tree.add_node(assocs_id, **{meta.label: cfg.assoc_token, "tree_role": "ASSOCS"})
            tree.add_edge(oid, assocs_id)
            for k, (_, dst, e_data) in enumerate(out_edges):
                dst_data = graph.nodes[dst]
                ref_id = f"REF::{oid}::{k}"
                ref = {
                    "tree_role": "REF",
                    "ref_node": str(dst),
                    meta.label: dst_data.get(meta.label, ""),
                    "ref_masked": bool(dst_data.get("masked", False)),
                }
                if meta.type in dst_data:
                    # Shown only when the referenced object is a train node (ref_masked=False).
                    ref[meta.type] = dst_data[meta.type]
                tree.add_node(ref_id, **ref)
                edge = {}
                if meta.edge_label in e_data:
                    edge[meta.edge_label] = e_data[meta.edge_label]
                if meta.edge_type in e_data:
                    edge[meta.edge_type] = e_data[meta.edge_type]
                tree.add_edge(assocs_id, ref_id, **edge)

    return tree


def serialize_graphs_to_trees(
    graphs: Iterable[nx.DiGraph],
    node_target_attr: str | None = None,
    tree_cfg: TreeSerializationConfig | None = None,
    metadata: MetaDataConfig | None = None,
) -> List[nx.DiGraph]:
    return [
        serialize_graph_to_tree(g, node_target_attr=node_target_attr, tree_cfg=tree_cfg, metadata=metadata)
        for g in graphs
    ]


def tree_to_bracket_text(tree: nx.DiGraph, root: str = "MODEL", label_key: str = "name") -> str:
    def _dfs(node_id: str) -> str:
        name = str(tree.nodes[node_id].get(label_key, node_id))
        children = list(tree.successors(node_id))
        if not children:
            return name
        inner = " ".join(_dfs(c) for c in children)
        return f"({name} {inner})"

    if root not in tree:
        roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
        if not roots:
            return ""
        root = roots[0]
    return _dfs(root)
