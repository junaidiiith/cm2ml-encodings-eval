from abc import abstractmethod
from typing import List
import networkx as nx
from uuid import uuid4
import numpy as np
import torch
from data_loading.metadata import GraphMetadata
from tokenization.utils import doc_tokenizer
from settings import (
    SUPERTYPE,
    REFERENCE,
    CONTAINMENT,
    
    EDGE_CLS_TASK,
    LINK_PRED_TASK,
)
from utils import md5_hash, get_node_neighbours
from tokenization.special_tokens import (
    NODE_BEGIN,
    NODE_END,
    EDGE_START,
    EDGE_END
)


class LangGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.id = uuid4().hex
        self.node_label_to_id = {}
        self.id_to_node_label = {}
        self.edge_label_to_id = {}
        self.id_to_edge_label = {}


    @abstractmethod
    def create_graph(self):
        pass


    def set_numbered_labels(self):
        self.node_label_to_id = {label: i for i, label in enumerate(self.nodes())}
        self.id_to_node_label = {i: label for i, label in enumerate(self.nodes())}

        self.edge_label_to_id = {label: i for i, label in enumerate(self.edges())}
        self.id_to_edge_label = {i: label for i, label in enumerate(self.edges())}

        self.numbered_graph = self.get_numbered_graph()
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.numbered_graph.edges())}
        self.idx_to_edge = {idx: edge for idx, edge in enumerate(self.numbered_graph.edges())}



    def get_numbered_graph(self) -> nx.DiGraph:
        nodes = [(self.node_label_to_id[i], data) for i, data in list(self.nodes(data=True))]
        edges = [(self.node_label_to_id[i], self.node_label_to_id[j], data) for i, j, data in list(self.edges(data=True))]
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        return graph


    @property
    def enr(self):
        if self.number_of_nodes() == 0:
            return -1
        return self.number_of_edges() / self.number_of_nodes()


    @property
    def edge_index(self):
        edge_index = torch.tensor(list(self.numbered_graph.edges)).t().contiguous().numpy()
        return edge_index

    @property
    def hash(self):
        return md5_hash(str(sorted(self.edges)))
    
    def get_edge_id(self, edge):
        return self.edge_label_to_id[edge]

    def get_edge_label(self, edge_id):
        return self.edge_label_to_id[edge_id]

    
    def get_node_id(self, node):
        return self.node_label_to_id[node]
    
    def get_node_label(self, node_id):
        return self.node_label_to_id[node_id]


def create_graph_from_edge_index(graph: LangGraph, edge_index: np.ndarray):
    """
    Create a subgraph from G using only the edges specified in edge_index.
    
    Parameters:
    G (networkx.Graph): The original graph.
    edge_index (numpy.ndarray): A numpy containing edge indices.
    
    Returns:
    networkx.Graph: A subgraph of G containing only the edges in edge_index.
    """

    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    # Add nodes and edges from the edge_index to the subgraph
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(list(graph.numbered_graph.nodes(data=True)))
    subgraph.add_edges_from([(u, v, graph.numbered_graph.edges[u, v]) for u, v in edge_index.T])
    for node, data in subgraph.nodes(data=True):
        data = graph.numbered_graph.nodes[node]
        subgraph.nodes[node].update(data)



    subgraph.node_label_to_id = graph.node_label_to_id
    subgraph.id_to_node_label = graph.id_to_node_label
    subgraph.edge_label_to_id = graph.edge_label_to_id
    subgraph.id_to_edge_label = graph.id_to_edge_label
    if len(edge_index) > 0:
        try:
            assert subgraph.number_of_edges() == edge_index.shape[1]
        except AssertionError as e:
            print(f"Number of edges mismatch {subgraph.number_of_edges()} != {edge_index.size(1)}")
            import pickle
            pickle.dump([graph, edge_index], open("subgraph.pkl", "wb"))
            raise e

    return subgraph


def format_path(
    graph: LangGraph, 
    path: List, 
    metadata: GraphMetadata, 
    use_node_attributes = False, 
    use_node_types = False, 
    use_edge_label = False, 
    use_edge_types = False, 
    node_cls_label=None,
    edge_cls_label='type',
    use_special_tokens = False, 
    no_labels = False,
    preprocessor=doc_tokenizer,
    neg_sample=False
):
    """Format a path into a string representation."""
    def get_node_label(node):
        
        masked = graph.nodes[node].get('masked')
        node_type = f"{graph.nodes[node].get(f'{node_cls_label}', '')}" \
            if use_node_types and not masked and node_cls_label else ''    
        
        if node_type != '':
            if isinstance(graph.nodes[node].get(f'{node_cls_label}'), bool):
                node_type = node_cls_label.title() if graph.nodes[node].get(f'{node_cls_label}') else ''
                
            
        node_label = get_node_name(
            graph.nodes[node], 
            metadata.node_label, 
            use_node_attributes, 
            metadata.node_attributes
        ) if not no_labels else ''
        
        
        if preprocessor:
            node_label = preprocessor(node_label)
        
        node_label = f"{node_type} {node_label}".strip()
        if use_special_tokens:
            node_label = f"{NODE_BEGIN}{node_label}{NODE_END}"
        
        return node_label.strip()

    def get_edge_label(n1, n2):
        edge_data = graph.get_edge_data(n1, n2)
        masked = edge_data.get('masked')
        edge_label = edge_data.get(metadata.edge_label, '') if use_edge_label and not no_labels else ''
        edge_type = f"{edge_cls_label}:{get_edge_data(edge_data, f'{edge_cls_label}', metadata.type)}" if use_edge_types and not masked  and edge_cls_label else ''
        
        if preprocessor:
            edge_label = preprocessor(edge_label)
        
        edge_label = f"{edge_type} {edge_label}".strip()
        
        if use_special_tokens:
            edge_label = f"{EDGE_START}{edge_label}{EDGE_END}"
        
        return edge_label.strip()

    # import code; code.interact(local=locals())
    assert len(path) > 0, "Path must contain at least one node."
    formatted = []
    for i in range(1, len(path)):
        n1 = path[i - 1]
        n2 = path[i]

        if not neg_sample:
            formatted.append(get_edge_label(n1, n2))
        formatted.append(get_node_label(n2))

    node_str = get_node_label(path[0])
    if len(formatted) > 0:
        node_str += " | " + " ".join(formatted).strip()
    
    return node_str


def get_edge_texts(
    graph: LangGraph, 
    edge: tuple, 
    d: int, 
    task_type: str,
    metadata: GraphMetadata, 
    use_node_attributes=False, 
    use_node_types=False, 
    use_edge_types=False,
    use_edge_label=False,
    node_cls_label=None,
    edge_cls_label='type',
    use_special_tokens=False, 
    no_labels=False,
    preprocessor: callable = doc_tokenizer,
    neg_samples=False
):
    n1, n2 = edge
    if not neg_samples:
        masked = graph.edges[n1, n2].get('masked')
        graph.edges[n1, n2]['masked'] = True
        
    
    n1_text = get_node_text(
        graph=graph,
        node=n1,
        d=d,
        metadata=metadata,
        use_node_attributes=use_node_attributes,
        use_node_types=use_node_types,
        use_edge_types=use_edge_types,
        use_edge_label=use_edge_label,
        node_cls_label=node_cls_label,
        edge_cls_label=edge_cls_label,
        use_special_tokens=use_special_tokens,
        no_labels=no_labels,
        preprocessor=preprocessor,
        exclude_edges=[edge]
    )
    n2_text = get_node_text(
        graph=graph,
        node=n2,
        d=d,
        metadata=metadata,
        use_node_attributes=use_node_attributes,
        use_node_types=use_node_types,
        use_edge_types=use_edge_types,
        use_edge_label=use_edge_label,
        node_cls_label=node_cls_label,
        edge_cls_label=edge_cls_label,
        use_special_tokens=use_special_tokens,
        no_labels=no_labels,
        preprocessor=preprocessor,
        exclude_edges=[edge]
    )
    

    edge_text = ""    
    
    if not neg_samples:
        graph.edges[n1, n2]['masked'] = masked or False
        
        edge_data = graph.get_edge_data(n1, n2)
        edge_type = get_edge_data(edge_data, edge_cls_label, metadata.type)
        edge_label = edge_data.get(metadata.edge_label, '') if use_edge_label and not no_labels else ''
        
        if task_type not in [EDGE_CLS_TASK, LINK_PRED_TASK]:
            if use_edge_types :
                edge_text += f" {edge_cls_label}: {edge_type} " if not no_labels else ''
                
            if use_edge_label:
                edge_text += f" {edge_label} " if not no_labels else ''

    
    return n1_text + EDGE_START + f"{edge_text}" + EDGE_END + n2_text


def get_node_text(
    graph: LangGraph, 
    node, 
    d: int, 
    metadata: GraphMetadata, 
    use_node_attributes=False, 
    use_node_types=False, 
    use_edge_types=False,
    use_edge_label=False,
    node_cls_label=None,
    edge_cls_label='type',
    use_special_tokens=False, 
    no_labels=False,
    preprocessor: callable = doc_tokenizer,
    exclude_edges=None
):
    masked = graph.nodes[node].get('masked')
    graph.nodes[node]['masked'] = True
    # raw_paths = utils.bfs(graph=graph, start_node=node, d=d, exclude_edges=exclude_edges)
    # unique_paths = utils.remove_subsets(list_of_lists=raw_paths)
    node_neighbour_texts = list()
    node_neighbours = get_node_neighbours(graph, node, d, exclude_edges=exclude_edges)
    for neighbour in node_neighbours:
        unique_paths = [p for p in nx.all_simple_paths(graph, node, neighbour, cutoff=d)]

        node_neighbour_texts.extend([
            format_path(
                graph=graph, 
                path=path, 
                metadata=metadata, 
                use_node_attributes=use_node_attributes, 
                use_node_types=use_node_types, 
                use_edge_types=use_edge_types, 
                use_edge_label=use_edge_label,
                node_cls_label=node_cls_label,
                edge_cls_label=edge_cls_label,
                use_special_tokens=use_special_tokens,
                no_labels=no_labels,
                preprocessor=preprocessor, 
                neg_sample=False
            )
            for path in unique_paths
        ])
    
    graph.nodes[node]['masked'] = masked or False
    node_str = "\n".join(node_neighbour_texts).strip() if node_neighbour_texts else ''
    
    if node_cls_label == 'stereotype':
        node_str = graph.nodes[node]['type'].title() + " " + node_str
    
    return node_str.strip()


def get_node_texts(
        graph: LangGraph, 
        d: int, 
        metadata: GraphMetadata, 
        use_node_attributes=False, 
        use_node_types=False, 
        use_edge_types=False,
        use_edge_label=False,
        node_cls_label=None,
        edge_cls_label='type',
        use_special_tokens=False, 
        no_labels=False,
        preprocessor: callable = doc_tokenizer
    ):
    paths_dict = {}
    for node in graph.nodes:
        paths_dict[node] = get_node_text(
            graph=graph,
            node=node,
            d=d,
            metadata=metadata,
            use_node_attributes=use_node_attributes,
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            use_edge_label=use_edge_label,
            node_cls_label=node_cls_label,
            edge_cls_label=edge_cls_label,
            use_special_tokens=use_special_tokens,
            no_labels=no_labels,
            preprocessor=preprocessor
        )

    return paths_dict


def get_attribute_labels(node_data, attribute_labels):
    if isinstance(node_data[attribute_labels], list):
        if not node_data[attribute_labels]:
            return ''
        if isinstance(node_data[attribute_labels][0], str):
            return ", ".join(node_data[attribute_labels])
        if isinstance(node_data[attribute_labels][0], tuple):
            return ", ".join([f"{k}: {v}" for k, v in node_data[attribute_labels]])
        elif isinstance(node_data[attribute_labels][0], dict):
            return ", ".join([f"{k}: {v}" for d in node_data[attribute_labels] for k, v in d.items()])
        return ", ".join(node_data[attribute_labels])
    if isinstance(node_data[attribute_labels], dict):
        return ", ".join([f"{k}: {v}" for k, v in node_data[attribute_labels].items()])
    return node_data[attribute_labels]


def get_node_name(
        node_data, 
        label, 
        use_attributes,
        attribute_labels,
    ):
    if use_attributes and attribute_labels in node_data:
        attributes_str = "(" + get_attribute_labels(node_data, attribute_labels) + ")"
    else:
        attributes_str = ''
    
    node_label = node_data.get(label, '') if node_data.get(label, '') else ''
    node_label = '' if node_label and node_label.lower() in ['null', 'none'] else node_label
    # if attributes_str:
    #     print(f"Node label: {node_label} | Attributes: {attributes_str}")
        
    return f"{node_label}{attributes_str}".strip()


def get_node_data(
    node_data: dict,
    node_label: str,
    model_type: str,
):
    if model_type == 'archimate':
        return get_archimate_node_data(node_data, node_label)
    elif model_type == 'ecore':
        return get_uml_node_data(node_data, node_label)
    elif model_type == 'ontouml':
        return get_ontouml_node_data(node_data, node_label)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



def get_edge_data(
    edge_data: dict,
    edge_label: str,
    model_type: str,
):
    if model_type == 'archimate':
        return get_archimate_edge_data(edge_data, edge_label)
    elif model_type == 'ecore':
        return get_uml_edge_data(edge_data, edge_label)
    elif model_type == 'ontouml':
        return get_ontouml_edge_data(edge_data, edge_label)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_archimate_node_data(edge_data: dict, node_label: str):
    return edge_data.get(node_label)

def get_uml_node_data(node_data: dict, node_label: str):
    return node_data.get(node_label, '')

def get_ontouml_node_data(node_data: dict, node_label: str):
    return node_data.get(node_label, '')


def get_archimate_edge_data(edge_data: dict, edge_label: str):
    return edge_data.get(edge_label)


def get_uml_edge_data(edge_data: dict, edge_label: str):
    if edge_label == 'type':
        return get_uml_edge_type(edge_data)
    elif edge_label in edge_data:
        return edge_data[edge_label]
    else:
        raise ValueError(f"Unknown edge label: {edge_label}")

def get_ontouml_edge_data(edge_data: dict, edge_label: str):
    try:
        return {'rel': "relates", "gen": "generalizes"}[edge_data.get(edge_label)]
    except KeyError:
        raise ValueError(f"Unknown edge label: {edge_label}")

def get_uml_edge_type(edge_data):
    edge_type = edge_data.get('type')
    if edge_type == SUPERTYPE:
        return SUPERTYPE
    if edge_type == CONTAINMENT:
        return CONTAINMENT
    return REFERENCE
