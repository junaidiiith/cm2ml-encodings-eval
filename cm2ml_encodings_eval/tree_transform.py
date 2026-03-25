from __future__ import annotations

from typing import Iterable, List

import networkx as nx

from .config import TreeSerializationConfig


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
    node_target_attr: str,
    tree_cfg: TreeSerializationConfig | None = None,
) -> nx.DiGraph:
    cfg = tree_cfg or TreeSerializationConfig()

    tree = nx.DiGraph()
    root = cfg.model_token
    tree.add_node(root, name=cfg.model_token, type="model", tree_role="MODEL", attributes=[])

    node_order = {n: i for i, n in enumerate(graph.nodes())}
    edge_order = {(u, v): i for i, (u, v) in enumerate(graph.edges())}

    for src_node, src_data in graph.nodes(data=True):
        idx = node_order[src_node]
        obj_id = f"OBJ::{idx}"
        obj_name_id = f"OBJ_NAME::{idx}"
        atts_id = f"ATTS::{idx}"

        obj_target = src_data.get(node_target_attr)

        tree.add_node(
            obj_id,
            name=cfg.object_token,
            type=str(src_data.get("type", "object")),
            tree_role="OBJ",
            source_node=str(src_node),
            attributes=[],
            masked=bool(src_data.get("masked", False)),
        )
        if obj_target is not None:
            tree.nodes[obj_id][node_target_attr] = obj_target

        tree.add_node(
            obj_name_id,
            name=str(src_data.get("name", src_node)),
            type="object_name",
            tree_role="OBJ_NAME",
            attributes=[],
        )
        tree.add_node(
            atts_id,
            name=cfg.atts_token,
            type="attributes_group",
            tree_role="ATTS",
            attributes=[],
        )

        tree.add_edge(root, obj_id, name="hasObject", type="containment")
        tree.add_edge(obj_id, obj_name_id, name="objectName", type="containment")
        tree.add_edge(obj_id, atts_id, name="hasAttributes", type="containment")

        attr_pairs = _as_attr_pairs(src_data.get("attributes"))
        for j, (attr_name, attr_value) in enumerate(attr_pairs):
            attr_name_id = f"ATTR::{idx}::{j}"
            tree.add_node(
                attr_name_id,
                name=attr_name,
                type="attribute_name",
                tree_role="ATTR_NAME",
                attributes=[],
            )
            tree.add_edge(atts_id, attr_name_id, name="attr", type="containment")

            if attr_value:
                attr_val_id = f"ATTR_VAL::{idx}::{j}"
                tree.add_node(
                    attr_val_id,
                    name=attr_value,
                    type="attribute_value",
                    tree_role="ATTR_VALUE",
                    attributes=[],
                )
                tree.add_edge(attr_name_id, attr_val_id, name="value", type="containment")

    if cfg.include_associations:
        for (u, v), edge_idx in edge_order.items():
            e_data = graph.edges[u, v]
            assoc_id = f"ASSOC::{edge_idx}"
            assoc_name_id = f"ASSOC_NAME::{edge_idx}"
            assoc_src_id = f"ASSOC_SRC::{edge_idx}"
            assoc_dst_id = f"ASSOC_DST::{edge_idx}"

            edge_name = str(e_data.get("name", "assoc"))
            src_name = str(graph.nodes[u].get("name", u))
            dst_name = str(graph.nodes[v].get("name", v))

            tree.add_node(
                assoc_id,
                name=cfg.assoc_token,
                type=str(e_data.get("type", "association")),
                tree_role="ASSOC",
                source_edge=(str(u), str(v)),
                attributes=[],
            )
            tree.add_node(
                assoc_name_id,
                name=edge_name,
                type="association_name",
                tree_role="ASSOC_NAME",
                attributes=[],
            )
            tree.add_node(
                assoc_src_id,
                name=src_name,
                type="association_src",
                tree_role="ASSOC_SRC",
                attributes=[],
            )
            tree.add_node(
                assoc_dst_id,
                name=dst_name,
                type="association_dst",
                tree_role="ASSOC_DST",
                attributes=[],
            )

            tree.add_edge(root, assoc_id, name="hasAssociation", type="containment")
            tree.add_edge(assoc_id, assoc_name_id, name="assocName", type="containment")
            tree.add_edge(assoc_id, assoc_src_id, name="source", type="containment")
            tree.add_edge(assoc_id, assoc_dst_id, name="target", type="containment")

    return tree


def serialize_graphs_to_trees(
    graphs: Iterable[nx.DiGraph],
    node_target_attr: str,
    tree_cfg: TreeSerializationConfig | None = None,
) -> List[nx.DiGraph]:
    return [serialize_graph_to_tree(g, node_target_attr=node_target_attr, tree_cfg=tree_cfg) for g in graphs]


def tree_to_bracket_text(tree: nx.DiGraph, root: str = "MODEL") -> str:
    def _dfs(node_id: str) -> str:
        name = str(tree.nodes[node_id].get("name", node_id))
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
