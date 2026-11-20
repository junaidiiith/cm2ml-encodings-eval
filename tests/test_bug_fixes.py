"""Regression tests, one or more per bug in the node-classification pipeline.

Bug ids refer to the review: G = graph/text pipeline, T = tree, X = infrastructure.
All tests use tiny hand-built graphs so the expected text is known exactly.
"""

import time
from argparse import Namespace

import networkx as nx
import pytest
import torch

import cm2ml_encodings_eval.pipeline as pipeline
from cm2ml_encodings_eval.config import MetaDataConfig, TextBuildConfig
from cm2ml_encodings_eval.pipeline import build_gnn_graphs
from cm2ml_encodings_eval.text import build_graph_texts, build_node_corpus, build_node_text, paths_from_node
from cm2ml_encodings_eval.tree_transform import serialize_graph_to_tree

import run_all_node_cls_combinations as runner

UML_META = MetaDataConfig(cls="abstract", leak_keys=("abstract", "isAbstract"))
EA_TYPE_META = MetaDataConfig(cls="type")
EA_LAYER_META = MetaDataConfig(cls="layer")


def uml_graph() -> nx.DiGraph:
    """Dog --supertype--> Animal ; Kennel --containment 'dogs'--> Dog."""
    g = nx.DiGraph()
    g.add_node("A", name="Animal", attributes=["age", "isAbstract"], abstract="True", masked=False)
    g.add_node("D", name="Dog", attributes=["bark"], abstract="False", masked=True)
    g.add_node("K", name="Kennel", attributes=[], abstract="False", masked=False)
    g.add_edge("D", "A", name="", type="supertype")
    g.add_edge("K", "D", name="dogs", type="containment")
    return g


def ea_graph() -> nx.DiGraph:
    """Actor --Assignment--> Role --Serving--> Service ; Role is a test node."""
    g = nx.DiGraph()
    g.add_node("a", name="Clerk", type="BusinessActor", layer="business", masked=False)
    g.add_node("r", name="Teller", type="BusinessRole", layer="business", masked=True)
    g.add_node("s", name="Payments", type="ApplicationService", layer="application", masked=False)
    g.add_edge("a", "r", type="Assignment")
    g.add_edge("r", "s", type="Serving")
    return g


def cfg(meta, d=0, **flags) -> TextBuildConfig:
    base = dict(include_node_label=True, include_node_type=False, include_node_attributes=False,
                include_edge_label=False, include_edge_type=False)
    base.update(flags)
    return TextBuildConfig(path_depth=d, metadata=meta, **base)


# --- G1: GNN node features must use path depth and edge flags -------------------------

def test_g1_gnn_features_change_with_depth():
    g = ea_graph()
    _, _, texts_d0 = build_gnn_graphs([g], "bow", cfg(EA_TYPE_META, d=0))
    _, _, texts_d1 = build_gnn_graphs([g], "bow", cfg(EA_TYPE_META, d=1, include_edge_type=True))
    assert texts_d0[0] != texts_d1[0]
    assert "Teller" in texts_d1[0][0] and "Assignment" in texts_d1[0][0]


def test_g1_gnn_d0_features_equal_text_pipeline_d0():
    g = uml_graph()
    c = cfg(UML_META, d=0, include_node_attributes=True)
    _, _, gnn_texts = build_gnn_graphs([g], "bow", c)
    corpus, _, idx = build_node_corpus([g], UML_META, True, False, True, False, False, 0, 64)
    by_node = dict(zip(g.nodes(), gnn_texts[0]))
    for split in ("train", "test"):
        for (_, node), text in zip(idx[split], corpus[split]):
            assert by_node[node] == text


# --- G2: include_node_type must emit the node type, never the target ------------------

def test_g2_node_type_flag_emits_type_not_target():
    text = build_node_text(ea_graph(), "s", cfg(EA_LAYER_META, d=2, include_node_type=True))
    assert "BusinessActor" in text  # train neighbour's type is visible
    assert "business" not in text.replace("BusinessActor", "").replace("BusinessRole", "")  # no layer values
    assert "application" not in text.lower().replace("applicationservice", "")


# --- G3: types of the classified node and of test nodes are hidden --------------------

def test_g3_own_type_and_test_neighbour_type_hidden():
    g = ea_graph()
    c = cfg(EA_TYPE_META, d=2, include_node_type=True)
    a_text = build_node_text(g, "a", c)
    assert "BusinessActor" not in a_text      # own type (= the label for EA Type)
    assert "BusinessRole" not in a_text       # test neighbour's type
    assert "Teller" in a_text                 # test neighbour's name is still allowed
    assert "ApplicationService" in a_text     # train node two hops away stays visible
    for n in g.nodes():
        for t in build_graph_texts(g, c):
            assert "BusinessRole" not in t


# --- G4: paths cover the undirected neighbourhood ------------------------------------

def test_g4_paths_follow_incoming_edges():
    g = uml_graph()
    reached = {nodes[-1] for nodes, _ in paths_from_node(g, "A", 2, 64)}
    assert reached == {"A", "D", "K"}  # A has no outgoing edges at all
    text = build_node_text(g, "A", cfg(UML_META, d=1))
    assert "<- Dog" in text


# --- G5: GNN edge direction is explicit; graphs keep the original direction -----------

def test_g5_graph_gnn_keeps_original_edge_direction():
    g = uml_graph()
    pyg, _, _ = build_gnn_graphs([g], "bow", cfg(UML_META))
    idx = {n: i for i, n in enumerate(g.nodes())}
    assert set(map(tuple, pyg[0].edge_index.t().tolist())) == {(idx["D"], idx["A"]), (idx["K"], idx["D"])}


def test_g5_edge_direction_options():
    g = uml_graph()
    n_edges = {d: build_gnn_graphs([g], "bow", cfg(UML_META), d)[0][0].edge_index.size(1)
               for d in ("directed", "reverse", "undirected")}
    assert n_edges == {"directed": 2, "reverse": 2, "undirected": 4}
    with pytest.raises(ValueError):
        build_gnn_graphs([g], "bow", cfg(UML_META), "sideways")


# --- G6: edge types must be read from the edge-type field -----------------------------

@pytest.mark.parametrize("meta", [UML_META, EA_LAYER_META, EA_TYPE_META])
def test_g6_edge_type_present_for_every_target(meta):
    g = uml_graph() if meta is UML_META else ea_graph()
    start = "A" if meta is UML_META else "a"
    text = build_node_text(g, start, cfg(meta, d=1, include_edge_type=True))
    assert ("supertype" in text) if meta is UML_META else ("Assignment" in text)


def test_g6_edge_label_flag():
    text = build_node_text(uml_graph(), "D", cfg(UML_META, d=1, include_edge_label=True))
    assert "dogs" in text


# --- leakage of the target / deny-listed meta-properties -----------------------------

def uml_meta_graph() -> nx.DiGraph:
    """CM2ML-style nodes whose attributes are key -> value meta-properties."""
    g = nx.DiGraph()
    g.add_node("A", name="Animal", attributes={"isAbstract": "true", "visibility": "public"}, abstract="True", masked=False)
    g.add_node("D", name="Dog", attributes={"isAbstract": "false", "visibility": "public"}, abstract="False", masked=True)
    g.add_edge("D", "A", name="", type="supertype")
    return g


ALL_FLAGS = dict(include_node_attributes=True, include_node_type=True, include_edge_label=True, include_edge_type=True)


def test_meta_property_leak_keys_removed_everywhere():
    g = uml_meta_graph()
    tree = serialize_graph_to_tree(g, metadata=UML_META)
    for d in (0, 1, 2, 3):
        c = cfg(UML_META, d=d, **ALL_FLAGS)
        for t in build_graph_texts(g, c) + build_graph_texts(tree, c):
            assert "isAbstract" not in t and "true" not in t.split() and "false" not in t.split()
        assert any("visibility" in t for t in build_graph_texts(g, c))  # other meta-properties stay
    assert not any(d.get("name") == "isAbstract" for _, d in tree.nodes(data=True))
    assert any(d.get("name") == "visibility" for _, d in tree.nodes(data=True))


def test_plain_feature_names_are_kept():
    # A metaclass owning a *feature* called isAbstract says nothing about its own abstractness.
    c = cfg(UML_META, d=0, include_node_attributes=True)
    assert "isAbstract" in build_node_text(uml_graph(), "A", c)
    tree = serialize_graph_to_tree(uml_graph(), metadata=UML_META)
    assert any(d.get("name") == "isAbstract" for _, d in tree.nodes(data=True))


def test_target_field_never_in_text():
    g = uml_graph()
    for n in g.nodes():
        g.nodes[n]["abstract"] = "zzlabel"
    tree = serialize_graph_to_tree(g, metadata=UML_META)
    for d in (0, 1, 2, 3):
        c = cfg(UML_META, d=d, **ALL_FLAGS)
        for t in build_graph_texts(g, c) + build_graph_texts(tree, c):
            assert "zzlabel" not in t


# --- T1: tree OBJ d=0 text equals graph d=0 text --------------------------------------

@pytest.mark.parametrize("meta,graph_fn", [(UML_META, uml_graph), (EA_TYPE_META, ea_graph), (EA_LAYER_META, ea_graph)])
def test_t1_tree_d0_text_equals_graph_d0_text(meta, graph_fn):
    g = graph_fn()
    tree = serialize_graph_to_tree(g, metadata=meta)
    c = cfg(meta, d=0, include_node_attributes=True, include_node_type=True)
    g_corpus, g_lab, g_idx = build_node_corpus([g], meta, True, True, True, False, False, 0, 64)
    t_corpus, t_lab, t_idx = build_node_corpus([tree], meta, True, True, True, False, False, 0, 64)
    assert g_corpus == t_corpus and g_lab == t_lab
    assert len({build_node_text(tree, n, c) for n, d in tree.nodes(data=True) if d.get("tree_role") == "OBJ"}) == 3


# --- T2: OBJ nodes receive messages from their subtree -------------------------------

def test_t2_tree_obj_receives_messages_from_children():
    tree = serialize_graph_to_tree(uml_graph(), metadata=UML_META)
    tree_cfg = cfg(UML_META)
    tree_cfg.is_tree = True
    pyg, _, _ = build_gnn_graphs([tree], "bow", tree_cfg)  # default for trees: bottom-up
    idx = {n: i for i, n in enumerate(tree.nodes())}
    obj = next(n for n, d in tree.nodes(data=True) if d.get("tree_role") == "OBJ" and d["name"] == "Dog")
    pairs = set(map(tuple, pyg[0].edge_index.t().tolist()))
    for child in tree.successors(obj):
        assert (idx[child], idx[obj]) in pairs  # source=child -> target=OBJ: OBJ aggregates its subtree
    assert (idx["MODEL"], idx[obj]) not in pairs  # the root no longer is the OBJ's only input


# --- T3: associations live under their source object ---------------------------------

def test_t3_tree_paths_reach_association_targets():
    tree = serialize_graph_to_tree(uml_graph(), metadata=UML_META)
    kennel = next(n for n, d in tree.nodes(data=True) if d.get("name") == "Kennel" and d.get("tree_role") == "OBJ")
    text = build_node_text(tree, kennel, cfg(UML_META, d=2, include_edge_label=True))
    assert "dogs" in text and "Dog" in text
    # the MODEL root is not traversed, so other objects are not reached through it
    assert "Animal" not in text


# --- T4: tree label space equals graph label space ------------------------------------

@pytest.mark.parametrize("meta,graph_fn", [(UML_META, uml_graph), (EA_TYPE_META, ea_graph), (EA_LAYER_META, ea_graph)])
def test_t4_tree_label_space(meta, graph_fn):
    g = graph_fn()
    tree = serialize_graph_to_tree(g, metadata=meta)
    pg, le_g, _ = build_gnn_graphs([g], "bow", cfg(meta))
    pt, le_t, _ = build_gnn_graphs([tree], "bow", cfg(meta))
    assert list(le_g.classes_) == list(le_t.classes_)
    assert int(pg[0].train_mask.sum()) == int(pt[0].train_mask.sum())
    assert int(pg[0].test_mask.sum()) == int(pt[0].test_mask.sum())


# --- T5: no type leakage through tree nodes -------------------------------------------

def test_t5_tree_hides_test_types_and_drops_leaky_attributes():
    tree = serialize_graph_to_tree(ea_graph(), metadata=EA_TYPE_META)
    c = cfg(EA_TYPE_META, d=3, include_node_type=True, include_edge_type=True)
    for t in build_graph_texts(tree, c):
        assert "BusinessRole" not in t  # the test node's type (= its label)
    assert any("ApplicationService" in t for t in build_graph_texts(tree, c))  # train types stay usable
    roles = {d.get("tree_role") for _, d in tree.nodes(data=True) if "type" in d}
    assert roles <= {"OBJ", "REF"}  # structural tree nodes carry no `type`


# --- X1: no process-randomised corpus cache --------------------------------------------

def test_x1_hash_cache_removed():
    for name in ("get_corpus_path", "node_corpus_exists", "load_node_corpus", "save_node_corpus"):
        assert not hasattr(pipeline, name)


# --- X2: corpus construction is linear in graph size ---------------------------------

def test_x2_corpus_build_is_fast_on_large_graph():
    g = nx.DiGraph()
    n = 4000
    for i in range(n):
        g.add_node(i, name=f"n{i}", attributes=[], abstract="False", masked=(i % 5 == 0))
    for i in range(n - 1):
        g.add_edge(i, i + 1, name="", type="reference")
    start = time.time()
    build_node_corpus([g], UML_META, True, False, False, False, False, 0, 64)
    assert time.time() - start < 2.0


# --- X3: the grid runner records complete, non-duplicated configs ----------------------

def _args(**kw):
    base = dict(dataset="modelset", target="abstract", label_attr="name", structures=["graph", "tree"],
                depths=[0, 1, 2, 3], modes=["text", "gnn"], encoders=["bow", "tfidf", "w2v", "bert-encoder"],
                gnn_models=["sage", "gcn", "gat"], mask_ratio=0.2, mask_seed=42, seed=42, edge_direction=None)
    base.update(kw)
    return Namespace(**base)


def test_x3_grid_sizes_and_signatures():
    uml_fields = {"use_attributes": True, "use_node_type": False, "use_edge_label": True, "use_edge_type": True}
    ea_fields = {"use_attributes": False, "use_node_type": True, "use_edge_label": False, "use_edge_type": True}
    uml = runner.build_grid(_args(), uml_fields)
    ea = runner.build_grid(_args(dataset="eamodelset", target="type"), ea_fields)
    assert (len(uml), len(ea)) == (800, 400)
    for grid in (uml, ea):
        sigs = [runner.config_signature(c) for c in grid]
        assert len(set(sigs)) == len(sigs)
        assert all(c["gnn_model_name"] in {"sage", "gcn", "gat"} for c in grid if c["classification_mode"] == "gnn")
        assert all(c["use_edge_type"] is None and c["use_node_type"] is None for c in grid if c["path_depth"] == 0)
        assert not any(c["tree"] and c["classification_mode"] == "text" and c["path_depth"] == 0 for c in grid)
        for c in grid:
            expected = None if c["classification_mode"] == "text" else ("reverse" if c["tree"] else "directed")
            assert c["edge_direction"] == expected
    assert all(c["use_node_type"] is None for c in uml)
    assert all(c["use_attributes"] is None and c["use_edge_label"] is None for c in ea)


def test_x3_resume_reads_signatures(tmp_path):
    out = tmp_path / "runs.jsonl"
    out.write_text('{"signature": "abc", "status": "ok"}\n{"signature": "def", "status": "error"}\n')
    assert runner.completed_signatures(out) == {"abc"}


# --- BERT is used only as a frozen encoder; fine-tuning was removed --------------------

def test_bert_finetuning_removed():
    import cm2ml_encodings_eval
    import cm2ml_encodings_eval.config as config
    import cm2ml_encodings_eval.models as models
    from cm2ml_encodings_eval.models import ClassicTextClassifier

    for mod, name in ((models, "BertTextClassifier"), (config, "BertClassifierConfig"),
                      (cm2ml_encodings_eval, "BertClassifierConfig")):
        assert not hasattr(mod, name)
    with pytest.raises(ValueError):
        ClassicTextClassifier("bert_classifier")


def test_bert_encoder_still_available():
    from cm2ml_encodings_eval.encoders import BertEmbeddingEncoder, get_encoder

    assert isinstance(get_encoder("bert"), BertEmbeddingEncoder)
    assert runner.ENCODER_MAP["bert-encoder"] == "bert"


def test_tree_text_d1_is_invariant_to_edge_and_type_flags():
    """Justifies the grid rule: tree OBJ texts need d >= 2 to see original edges/types."""
    tree = serialize_graph_to_tree(ea_graph(), metadata=EA_LAYER_META)
    for d, should_differ in ((1, False), (2, True)):
        off = build_node_corpus([tree], EA_LAYER_META, True, False, False, False, False, d, 64)[0]
        on = build_node_corpus([tree], EA_LAYER_META, True, True, False, True, True, d, 64)[0]
        assert (off != on) == should_differ
    tree_cfgs = runner.text_flag_combinations(1, {"use_attributes": False, "use_node_type": True,
                                                  "use_edge_label": False, "use_edge_type": True}, "tree", "text")
    assert tree_cfgs == [{"use_attributes": None, "use_edge_type": None, "use_edge_label": None, "use_node_type": None}]


# --- reduced design (A core + B slice), shards, follow-ups -----------------------------

UML_FIELDS = {"use_attributes": True, "use_node_type": False, "use_edge_label": True, "use_edge_type": True}
EA_FIELDS = {"use_attributes": False, "use_node_type": True, "use_edge_label": False, "use_edge_type": True}


@pytest.mark.parametrize("fields,dataset,target,expected", [
    (UML_FIELDS, "modelset", "abstract", 448), (EA_FIELDS, "eamodelset", "type", 256)])
def test_reduced_design(fields, dataset, target, expected):
    full = runner.build_grid(_args(dataset=dataset, target=target, design="full"), fields)
    red = runner.build_grid(_args(dataset=dataset, target=target, design="reduced", full_gnn_models=["sage"]), fields)
    assert len(red) == expected
    key = lambda c: (c["classification_mode"], c["structure"], c["encoder_key"], c["path_depth"],
                     c["gnn_model_name"], *(c[f] for f in runner.TEXT_FLAGS))
    full_keys = {key(c) for c in full}
    assert all(key(c) in full_keys for c in red)  # reduced is a subset of full
    # (A) core: every text and every SAGE config of the full grid is kept
    core = [c for c in full if c["classification_mode"] == "text" or c["gnn_model_name"] == "sage"]
    assert {key(c) for c in core} <= {key(c) for c in red}
    # (B) slice: exactly one GCN and one GAT config per structure x encoder x d, all flags on,
    #     each with an identical SAGE config to compare against
    sl = [c for c in red if c["design_part"] == "slice"]
    cells = {}
    for c in sl:
        cells.setdefault((c["structure"], c["encoder_key"], c["path_depth"], c["gnn_model_name"]), []).append(c)
    assert len(cells) == 2 * 4 * 4 * 2 and all(len(v) == 1 for v in cells.values())
    for c in sl:
        assert runner.is_slice_config({f: c[f] for f in runner.TEXT_FLAGS})
        twin = key({**c, "gnn_model_name": "sage"})
        assert twin in {key(x) for x in red}


def test_shards_partition_grid():
    grid = runner.build_grid(_args(design="reduced", full_gnn_models=["sage"]), UML_FIELDS)
    shards = [[c for c in grid if runner.in_shard(c, f"{i}/3")] for i in range(3)]
    sigs = [{runner.config_signature(c) for c in s} for s in shards]
    assert sum(map(len, sigs)) == len(grid) and set.union(*sigs) == {runner.config_signature(c) for c in grid}
    with pytest.raises(ValueError):
        runner.in_shard(grid[0], "3/3")


def test_base_signature_ignores_seed_only():
    c = runner.build_grid(_args(), UML_FIELDS)[0]
    other_seed = {**c, "seed": 7}
    assert runner.base_signature(c) == runner.base_signature(other_seed)
    assert runner.config_signature(c) != runner.config_signature(other_seed)
    assert runner.base_signature(c) != runner.base_signature({**c, "path_depth": 3})
