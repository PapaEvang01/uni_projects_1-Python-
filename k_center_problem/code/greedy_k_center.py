"""
Greedy k-Center Solver for GEXF Graphs
--------------------------------------

This script randomly selects a graph from the "graphs/" folder (filtered by node count),
and applies a greedy approximation algorithm to solve the k-center problem.

The greedy algorithm starts with either a random or specified first center, and iteratively
adds the farthest node from the current centers until k centers are selected.

"""

import networkx as nx
import random
import os
import re
from timeit import default_timer


def k_centers_objective_value(G, centers):
    """
    Computes the maximum shortest-path distance from any node to the nearest center.
    """
    path = nx.multi_source_dijkstra_path_length(G, centers)
    return max(path.values())


def get_key_by_value(d, target_value):
    """
    Utility function to return the first key in a dict that has the given value.
    """
    for key, value in d.items():
        if value == target_value:
            return key
    return None


def k_centers_greedy(G, k, first_center=None):
    """
    Greedy 2-approximation for the k-center problem.

    Parameters:
        G : networkx.Graph
        k : number of centers
        first_center : optional, node to use as the first center

    Returns:
        (cost, centers) : tuple of final cost and selected centers
    """
    nodes = list(G.nodes)

    # Select initial center
    if first_center is None:
        first_center = random.choice(nodes)

    centers = {first_center}

    while len(centers) < k:
        path_lengths = nx.multi_source_dijkstra_path_length(G, centers)
        farthest_distance = max(path_lengths.values())
        new_center = get_key_by_value(path_lengths, farthest_distance)
        centers.add(new_center)

    cost = k_centers_objective_value(G, centers)
    return cost, sorted(centers)


def k_centers_greedy_best_of_all(G, k, time_limit):
    """
    Runs greedy algorithm multiple times with each possible node as starting point,
    keeping the best result. Stops if time limit is reached.

    Returns:
        best_cost : float
        best_centers : list of selected centers
        best_start_node : the starting node that led to best result
        optimal_solution_found : bool
    """
    nodes = list(G.nodes)
    best_cost = float('inf')
    best_centers = []
    best_start_node = None
    optimal_solution_found = True

    start = default_timer()

    for node in nodes:
        duration = default_timer() - start
        if duration > time_limit:
            optimal_solution_found = False
            break

        cost, centers = k_centers_greedy(G, k, first_center=node)
        print(f"Start from node {node}, cost = {cost}")

        if cost < best_cost:
            best_cost = cost
            best_centers = centers
            best_start_node = node
            print(f"New best solution: {best_centers}, cost = {best_cost}, from node {node}")

    return best_cost, best_centers, best_start_node, optimal_solution_found


def extract_node_count(filename):
    match = re.search(r'graph_(\d+)_\d+\.gexf', filename)
    return int(match.group(1)) if match else -1


def run_greedy_on_random_graph(folder_path, k=3, time_limit=30, max_nodes=100):
    files = [f for f in os.listdir(folder_path) if f.endswith(".gexf") and extract_node_count(f) <= max_nodes]
    if not files:
        print(f"No graphs with ≤ {max_nodes} nodes found.")
        return

    random_file = random.choice(files)
    full_path = os.path.join(folder_path, random_file)
    node_count = extract_node_count(random_file)

    print(f"Selected random graph: {random_file} with {node_count} nodes")

    G = nx.read_gexf(full_path)
    cost, centers, start_node, success = k_centers_greedy_best_of_all(G, k, time_limit)

    print("\n=== Final Greedy Result ===")
    print(f"Best centers: {centers}")
    print(f"Best starting node: {start_node}")
    print(f"Final cost: {cost}")
    print(f"Optimal solution found: {'Yes' if success else 'No'}")


# Entry point
if __name__ == "__main__":
    run_greedy_on_random_graph("graphs", k=3, time_limit=60)
