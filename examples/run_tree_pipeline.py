import networkx as nx

from cm2ml_encodings_eval import (
    GNNExperimentConfig,
    TextBuildConfig,
    TextExperimentConfig,
    TreeSerializationConfig,
    run_tree_gnn_node_classification,
    run_tree_text_node_classification,
    set_random_seed,
)
from cm2ml_encodings_eval.split import mask_graph_nodes
from data_loading.models_dataset import get_models_dataset


def build_sample_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("a", name="A", type="package", attributes={"visibility": "public"}, cls="package")
    g.add_node("b", name="B", type="class", attributes={"isAbstract": "true"}, cls="class")
    g.add_node("c", name="C", type="attribute", attributes={"visibility": "protected"}, cls="attribute")

    g.add_edge("a", "b", name="elementImport", type="association")
    g.add_edge("b", "c", name="attribute", type="association")
    return g


def main():
    seed = 42
    set_random_seed(seed)
    
    graph = build_sample_graph()
    graphs = [graph]

    config_params = dict(
        dataset_name = "modelset",
        min_enr = 1.2,
        min_edges = 10,
        reload = False
    )
    print("Loading dataset...")
    graphs = get_models_dataset(**config_params)
    print(f"Loaded {len(graphs)} graphs.")
    print("Masking graph nodes for test set...")
    mask_graph_nodes(graphs)
    
    
    tree_cfg = TreeSerializationConfig()
    # tree = serialize_graph_to_tree(graph, node_target_attr="cls", tree_cfg=tree_cfg)
    # print("Sample tree nodes:", tree.number_of_nodes(), "Sample tree edges:", tree.number_of_edges())

    text_cfg = TextBuildConfig(path_depth=2)

    text_res = run_tree_text_node_classification(
        graphs=graphs,
        text_cfg=text_cfg,
        exp_cfg=TextExperimentConfig(encoder_name="tfidf", classifier_name="logreg"),
        tree_cfg=tree_cfg,
    )
    print("Tree text node cls:", text_res['metrics'])

    gnn_res = run_tree_gnn_node_classification(
        graphs_or_dataset=graphs,
        node_encoder_name="tfidf",
        text_cfg=text_cfg,
        gnn_cfg=GNNExperimentConfig(model_name="gcn", epochs=30),
        tree_cfg=tree_cfg,
    )
    print("Tree gnn node cls:", gnn_res['metrics'])


if __name__ == "__main__":
    main()
