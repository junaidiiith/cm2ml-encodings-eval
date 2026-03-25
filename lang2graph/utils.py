import signal
import networkx as nx


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Took too long")

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=5):
    # Set the signal handler and a timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        result = None
    finally:
        # Disable the alarm
        signal.alarm(0)
    return result


def get_triple_text(node, edge_data, neighbour):
    src = f'Class {node}'
    dest = f'Class {neighbour}'

    if edge_data is None:
        return f'{src} -> {dest}'
    
    if edge_data['type'] == 'reference' and 'name' in edge_data:
        return f'{src} -> ({edge_data["name"]}) -> {dest}'
    
    return f'{src} -> {dest}'
    

def find_node_str_upto_distance(node, distance=1):
    nodes_with_distance = find_nodes_within_distance(
        node, 
        distance=distance
    )
    if distance == 0:
        return f'Class {node}'
    
    d2n = {dd[0]: set() for _, dd in nodes_with_distance}
    for neighbour, dis_data in nodes_with_distance:
        d, edge_data = dis_data
        if d == 0:
            continue

        node_text = get_triple_text(
            node, edge_data, neighbour
        )
        if node_text:
            d2n[d].add(node_text)


    d2n = sorted(d2n.items(), key=lambda x: x[0])
    node_buckets = [f" ".join(nbs) for _, nbs in d2n]
    path_str = " | ".join(node_buckets)
    
    return path_str


def find_nodes_within_distance(g: nx.DiGraph, n, distance=1):
    visited = {n: (0, None)}
    queue = [(n, 0)]
    
    while queue:
        node, d = queue.pop(0)
        if d == distance:
            continue
        for neighbor in g.neighbors(node):
            if neighbor not in visited:
                visited[neighbor] = (d+1, g.edges[node, neighbor])
                queue.append((neighbor, d+1))
    
    visited = sorted(visited.items(), key=lambda x: x[1][0])
    return visited