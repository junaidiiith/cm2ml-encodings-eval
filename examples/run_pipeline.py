import networkx as nx


from cm2ml_encodings_eval import (
    GNNExperimentConfig,
    TextBuildConfig,
    TextExperimentConfig,
    run_gnn_edge_classification,
    run_gnn_node_classification,
    run_text_edge_classification,
    run_text_node_classification,
    set_random_seed,
)

from cm2ml_encodings_eval.split import mask_graph_nodes
from data_loading.models_dataset import get_models_dataset    

def build_sample_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("Scheme", name="Scheme", type="metaclass", attributes=["name"], abstract=False, cls="schema")
    g.add_node("Table", name="Table", type="metaclass", attributes=["name"], abstract=False, cls="entity")
    g.add_node("Column", name="Column", type="metaclass", attributes=["name"], abstract=False, cls="entity")
    g.add_node("FKey", name="FKey", type="metaclass", attributes=[], abstract=False, cls="key")
    g.add_node("PKey", name="PKey", type="metaclass", attributes=[], abstract=False, cls="key")

    g.add_edge("Scheme", "Table", name="tables", type="containment", edge_cls="containment")
    g.add_edge("Scheme", "FKey", name="keys", type="containment", edge_cls="containment")
    g.add_edge("Table", "Column", name="columns", type="containment", edge_cls="containment")
    g.add_edge("Table", "Scheme", name="scheme", type="reference", edge_cls="reference")
    g.add_edge("Table", "PKey", name="key", type="containment", edge_cls="containment")
    g.add_edge("Column", "Table", name="table", type="reference", edge_cls="reference")
    g.add_edge("FKey", "PKey", name="refersTo", type="reference", edge_cls="reference")
    g.add_edge("FKey", "Column", name="column", type="reference", edge_cls="reference")
    g.add_edge("FKey", "Scheme", name="scheme", type="reference", edge_cls="reference")
    g.add_edge("PKey", "Column", name="column", type="reference", edge_cls="reference")
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

    text_cfg = TextBuildConfig(path_depth=1)

    print("A1/A2 node text")
    node_text_res = run_text_node_classification(
        graphs,
        text_cfg=text_cfg,
        exp_cfg=TextExperimentConfig(encoder_name="bert", classifier_name="logreg"),
    )
    print(node_text_res['metrics'])

    # print("A1/A2 edge text")
    # edge_text_res = run_text_edge_classification(
    #     graphs,
    #     target_attr="edge_cls",
    #     split_cfg=split_cfg,
    #     text_cfg=text_cfg,
    #     exp_cfg=TextExperimentConfig(encoder_name="bow", classifier_name="linearsvm"),
    # )
    # print(edge_text_res)

    # print("A3 GNN node")
    gnn_node_res = run_gnn_node_classification(
        graphs,
        node_encoder_name="tfidf",
        text_cfg=text_cfg,
        gnn_cfg=GNNExperimentConfig(model_name="gcn", epochs=30),
    )
    print(gnn_node_res['metrics'])

    # print("A3 GNN edge")
    # gnn_edge_res = run_gnn_edge_classification(
    #     graph,
    #     target_attr="edge_cls",
    #     node_encoder_name="tfidf",
    #     edge_encoder_name="tfidf",
    #     split_cfg=split_cfg,
    #     text_cfg=text_cfg,
    #     gnn_cfg=GNNExperimentConfig(model_name="graphsage", epochs=30),
    # )
    # print(gnn_edge_res)


if __name__ == "__main__":
    main()
