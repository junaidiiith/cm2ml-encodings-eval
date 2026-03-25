from argparse import ArgumentParser
from ast import Dict
import random
import numpy as np
import torch
import os
import fnmatch
import json
import xmltodict
from torch_geometric.data import Data
import hashlib
import networkx as nx
from collections import deque
import struct
from tensorboardX.proto import event_pb2
from collections import deque
from typing import Any, List, Tuple, Optional, Set
import networkx as nx





def find_files_with_extension(root_dir, extension):
    matching_files: List[str] = list()

    # Recursively search for files with the specified extension
    for root, _, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, f'*.{extension}'):
            matching_files.append(os.path.join(root, filename))

    return matching_files


def xml_to_json(xml_string):
    xml_dict = xmltodict.parse(xml_string)
    json_data = json.dumps(xml_dict, indent=4)
    return json_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_dummy_graph(num_nodes, num_edges):
    # Node and edge attribute types
    node_types = ['nt1', 'nt2', 'nt3', 'nt4', 'nt5', 'nt6', 'nt7']
    edge_types = ['et1', 'et2', 'et3', 'et4']

    # Create a graph
    G = nx.Graph()

    # Add nodes with attributes
    for i in range(1, num_nodes + 1):
        G.add_node(i, name=f'node_{i}', type=random.choice(node_types))

    # Add edges with attributes
    edges_added = 0
    while edges_added < num_edges:
        u = random.randint(1, num_nodes)
        v = random.randint(1, num_nodes)
        if u != v and not G.has_edge(u, v):  # Ensure no self-loops and no duplicate edges
            G.add_edge(u, v, name=f'edge_{edges_added + 1}', type=random.choice(edge_types))
            edges_added += 1

    return G


def bfs(graph: nx.Graph, start_node, d, exclude_edges: List[str] = None):
        """Perform BFS to get all paths up to a given depth."""
        if exclude_edges is None:
            exclude_edges = []
            
        queue = deque([(start_node, [start_node])])
        paths = []

        while queue:
            current_node, path = queue.popleft()

            # Stop if the path length exceeds the maximum depth
            if len(path) - 1 > d:
                continue

            # Store the path
            if len(path) >= 1:  # Exclude single-node paths
                paths.append(path)

            # Add neighbors to the queue
            for neighbor in graph.neighbors(current_node):
                edge = (current_node, neighbor)
                if neighbor not in path and edge not in exclude_edges:
                    queue.append((neighbor, path + [neighbor]))
        
        return paths


def remove_subsets(list_of_lists):
        sorted_lists = sorted(list_of_lists, key=len, reverse=True)
        unique_lists = []
        for lst in sorted_lists:
            current_set = set(lst)
            if not any(current_set <= set(ul) for ul in unique_lists):
                unique_lists.append(lst)
        
        return unique_lists


def get_size_format(sz):
	for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
		if sz < 1024.0:
			return "%3.1f %s" % (sz, x)
		sz /= 1024.0
	

def get_file_size(file_path):
	sz = os.path.getsize(file_path)
	return get_size_format(sz)

def get_directory_size(directory):
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return get_size_format(total_size)

def get_tensor_size(tensor: torch.Tensor):
	return get_size_format(tensor.element_size() * tensor.nelement())

def get_size_of_data(data: Data):
	size = 0
	for _, value in data:
		if isinstance(value, torch.Tensor):
			size += value.element_size() * value.nelement()
		elif isinstance(value, int):
			size += value.bit_length() // 8
						
	return get_size_format(size)


def md5_hash(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()

def randomize_features(dataset: List[Data], num_feats, mode):
    for data in dataset:
        num_nodes = data.num_nodes
        num_edges = data.overall_edge_index.shape[1] if hasattr(data, 'overall_edge_index') else data.edge_index.shape[1]
        if mode == 'node':
            data.x = torch.randn((num_nodes, num_feats))
        elif mode == 'edge':
            data.edge_attr = torch.randn((num_edges, num_feats))
        else:
            raise ValueError("Invalid mode. Choose 'node' or 'edge'.")
        

def merge_argument_parsers(p1: ArgumentParser, p2: ArgumentParser):
    merged_parser = ArgumentParser(description="Merged Parser")

    # Combine arguments from parser1
    for action in p1._actions:
        if action.dest != "help":  # Skip the help action
            merged_parser._add_action(action)

    # Combine arguments from parser2
    for action in p2._actions:
        if action.dest != "help":  # Skip the help action
            merged_parser._add_action(action)

    return merged_parser


def is_meaningful_line(line: str):
    stripped_line: str = line.strip()
    # Ignore empty lines, comments, and docstrings
    if stripped_line == "" or stripped_line.startswith("#") or stripped_line.startswith('"""') or stripped_line.startswith("'''"):
        return False
    return True

def count_lines_of_code_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        meaningful_lines = [line for line in lines if is_meaningful_line(line)]
    return len(meaningful_lines)

def count_total_lines_of_code(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                total_lines += count_lines_of_code_in_file(file_path)
    return total_lines


def snake_to_title(snake_str: str):
    return snake_str.replace("_", " ").title()


def parse_event_file(filepath):
    """
    Generator that yields `Event` protocol buffer messages
    from a single TensorBoard event file, using TFRecord-like
    parsing without TensorFlow.
    """
    with open(filepath, 'rb') as f:
        while True:
            # 1) Read the length of the next record (8 bytes, little-endian).
            header = f.read(8)
            if len(header) < 8:
                break  # no more data

            record_length = struct.unpack('Q', header)[0]

            # 2) Skip the 4-byte length CRC (unused here).
            _ = f.read(4)

            # 3) Read the actual record data.
            record_data = f.read(record_length)

            # 4) Skip the 4-byte data CRC.
            _ = f.read(4)

            if len(record_data) < record_length:
                # Incomplete record at end of file
                break

            # Parse the record into an Event proto.
            event = event_pb2.Event()
            event.ParseFromString(record_data)
            yield event


def get_max_scalars_with_step_epoch(logdir, epoch_tag="eval/epoch"):
    """
    Scans all `events.out.tfevents.*` files in `logdir` and returns a dict:
        {
          scalar_tag: {
            "max_value": float,
            "step": int or None,
            "epoch": int or float or None
          },
          ...
        }
    By default, it looks for an 'eval/epoch' scalar to determine the epoch.
    If that scalar isn't found, epoch will be None.
    """
    max_scalars = {}

    # Gather all event files in the directory
    event_files = [
        os.path.join(logdir, f)
        for f in os.listdir(logdir)
        if f.startswith("events.out.tfevents")
    ]

    for filepath in event_files:
        for event in parse_event_file(filepath):
            # In proto3, "step" is always present as an int64.
            # If it's not explicitly set, it'll be 0.
            step = event.step
            if step == 0:
                step = None  # treat zero as "no step logged"

            # Try to find an epoch value in the same event (if you're logging it).
            epoch_val = None
            if event.summary and event.summary.value:
                # First pass: see if there's a dedicated epoch tag in this event
                for v in event.summary.value:
                    if v.tag == epoch_tag and v.HasField("simple_value"):
                        epoch_val = v.simple_value
                        break

                # Second pass: for each scalar, update the max if we see a bigger value
                for v in event.summary.value:
                    if v.HasField('simple_value'):
                        tag = v.tag
                        val = v.simple_value

                        # Ignore the epoch tag itself; we only want other scalar tags
                        if tag == epoch_tag:
                            continue

                        # Update if this tag is new or if we found a bigger value
                        if (tag not in max_scalars) or (val > max_scalars[tag]["max_value"]):
                            max_scalars[tag] = {
                                "max_value": val,
                                "step": step,
                                "epoch": epoch_val
                            }

    return max_scalars


def update_config_results(logs_dir='logs'):
    
    logs_dir = "logs"
    
    def is_tf_dir(dir_path):
        return any([f.startswith("events.out.tfevents") for f in os.listdir(dir_path)])
    
    graph_data_dir = "datasets/graph_data"
    dataset_config = dict()

    for dataset_dir in os.listdir(graph_data_dir):
        if dataset_dir not in ['ecore_555', 'eamodelset', 'modelset', 'ontouml']:
            continue
        with open(os.path.join(graph_data_dir, dataset_dir, 'configs.json')) as f:
            dataset_config[dataset_dir] = json.load(f)
    

    for dataset_dir in os.listdir(logs_dir):
        if dataset_dir not in ['ecore_555', 'eamodelset', 'modelset', 'ontouml']:
            continue
        for task in os.listdir(os.path.join(logs_dir, dataset_dir)):
            task_dir = os.path.join(logs_dir, dataset_dir, task)
            if not os.path.isdir(task_dir) or "_comp_" in task:
                continue
            
            for root, dirs, _ in os.walk(task_dir):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if is_tf_dir(dir_path):
                        
                        config_id = dir_name.split(os.sep)[-1]
                        
                        if "_" in config_id:
                            config_id = config_id.split("_")[0]
                            
                        
                        assert config_id in dataset_config[dataset_dir], f"Config {config_id} not found in {dataset_dir}"
                        config = dataset_config[dataset_dir][config_id]
                        config_results = get_max_scalars_with_step_epoch(dir_path)
                        
                        if 'results' not in config:
                            config['results'] = list()
                        
                        config['results'].append(config_results)
                        config['task'] = task
                        
        
    for dataset_dir in os.listdir(graph_data_dir):
        if dataset_dir not in ['ecore_555', 'eamodelset', 'modelset', 'ontouml']:
            continue
        with open(os.path.join(graph_data_dir, dataset_dir, 'configs.json'), 'w') as f:
             json.dump(dataset_config[dataset_dir], f)
    


def set_encoded_labels(train_ds, test_ds):
    train_labels = train_ds.inputs['labels']
    test_labels = test_ds.inputs['labels']
    all_labels = torch.cat([train_labels, test_labels])
    unique_labels = torch.unique(all_labels)
    label_to_encoded = dict()
    encoded_to_label = dict()
    for i, label in enumerate(unique_labels):
        label_to_encoded[label.item()] = i
        encoded_to_label[i] = label.item()
    
    train_ds.inputs['labels'] = torch.tensor([label_to_encoded[label.item()] for label in train_labels])
    test_ds.inputs['labels'] = torch.tensor([label_to_encoded[label.item()] for label in test_labels])
    

def set_torch_encoding_labels(dataset: list, cls_label, exclude_labels: List[str] = None):
    print(f"Setting encoding labels for {cls_label}")
    labels = [getattr(data, cls_label) for data in dataset]
    all_labels = torch.cat(labels)
    unique_labels = [label for label in torch.unique(all_labels) if label not in exclude_labels]
    label_to_encoded = dict()
    encoded_to_label = dict()
    for i, label in enumerate(unique_labels):
        label_to_encoded[label.item()] = i
        encoded_to_label[i] = label.item()
            
    for i, data in enumerate(dataset):
        setattr(
            data, 
            cls_label, 
            torch.tensor([label_to_encoded.get(label.item(), -1) for label in getattr(data, cls_label)])
        )
    
    print(f"Set encoding labels for {cls_label}")
    

def find_nodes_within_distance(
    graph: nx.DiGraph,
    start_node: Any,
    distance: int,
    exclude_edges: Optional[List[Tuple[Any, Any]]] = None
) -> List[Tuple[Any, int]]:
    """
    Find all nodes reachable from start_node within a given distance,
    optionally excluding specified edges.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph to traverse.
    start_node : Any
        Node from which to start the search.
    distance : int
        Maximum graph distance (number of edges) to traverse.
    exclude_edges : Optional[List[Tuple[Any, Any]]]
        List of directed edges to exclude, each as a (u, v) tuple.

    Returns
    -------
    List[Tuple[Any, int]]
        Sorted list of (node, distance) pairs.
    """
    # Normalize exclude_edges to a set for fast lookups
    excluded: Set[Tuple[Any, Any]] = set(exclude_edges or [])

    # BFS initialization
    queue = deque([(start_node, 0)])
    visited: Dict[Any, int] = {}

    while queue:
        node, dist = queue.popleft()
        # Only process within the allowed distance
        if dist > distance:
            continue

        # Record the shortest distance to this node
        if node not in visited or dist < visited[node]:
            visited[node] = dist

        # Explore neighbors if we haven't reached max distance yet
        if dist < distance:
            for nbr in graph.neighbors(node):
                # Skip self-loops
                if nbr == node:
                    continue
                # Skip if edge is excluded
                if (node, nbr) in excluded:
                    continue
                # Skip already-visited nodes at shorter or equal distance
                if nbr in visited and visited[nbr] <= dist + 1:
                    continue
                queue.append((nbr, dist + 1))

    # Return nodes sorted by distance
    return sorted(visited.items(), key=lambda x: x[1])




def get_node_neighbours(graph, start_node, distance, exclude_edges: List[str] = None):
    neighbours = find_nodes_within_distance(graph, start_node, distance, exclude_edges)
    max_distance = max(distance for _, distance in neighbours)
    distance = min(distance, max_distance)
    return [node for node, d in neighbours if d == distance]



def extract_top_rows(file_path):
    keys = [
        "encoder_name", 
        "classification_mode", 
        "tree", 
        "path_depth", 
        "use_attributes", 
        "use_edge_type", 
        "use_edge_label", 
        "gnn_model_name"
    ]
    def get_rows(f):
        with open(f) as fp:
            d = fp.read()
        rows = []
        for s in d.strip().split("\n"):
            rows.append(json.loads(s))
        return rows
    
    def get_config_str(c):
        conf_str = "$".join([f"{k}={c[k]}" if k in c and c[k] is not None else f"{k}=none" for k in keys])
        return conf_str
    
    ea_layer_rows = get_rows(file_path)

    ea_layer_results = dict()
    for r in ea_layer_rows:
        config_str = get_config_str(r['config'])
        if config_str not in ea_layer_results:
            ea_layer_results[config_str] = list()
        ea_layer_results[config_str].append(r)
    
    def write_top_rows(rows, output_file):
        with open(output_file, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
            
    write_top_rows([v[-1] for v in ea_layer_results.values()], file_path.replace(".jsonl", "_top.jsonl"))
