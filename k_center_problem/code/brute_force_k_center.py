"""
k-center Brute-Force Solver for GEXF Graphs
-------------------------------------------

This script randomly selects a graph from the "graphs/" folder, reads it using NetworkX,
and applies a brute-force algorithm to solve the k-center problem with a specified value of k.

The goal is to choose k center nodes such that the maximum shortest-path distance from
any node to its nearest center is minimized.

Due to the high computational cost, a time limit is imposed to stop the search early.
Only graphs with a node count below a specified threshold (default: 100) are considered.

"""

import networkx as nx
import itertools
import os
import random
from timeit import default_timer
import re


def k_centers_objective_value(G, centers):
    """
    Computes the objective function for the k-center problem:
    the maximum shortest-path distance from any node in the graph
    to its closest center.

    Parameters:
        G : networkx.Graph
        centers : list of center nodes

    Returns:
        float : the maximum shortest path to the nearest center
    """
    shortest_paths = nx.multi_source_dijkstra_path_length(G, centers)
    return max(shortest_paths.values())


def k_centers_brute_force(G, k, time_limit):
    """
    Brute-force search over all combinations of k nodes in G to find
    the optimal center set that minimizes the maximum distance to any node.

    Parameters:
        G : networkx.Graph
        k : int, number of centers to choose
        time_limit : float, maximum time in seconds before stopping

    Returns:
        best_combination : tuple of best center nodes
        best_centers : list version of best center nodes
        optimal_solution_found : bool, True if finished within time
        execution_time : float, time spent (in seconds)
    """
    start_time = default_timer()
    best_combination = None
    best_cost = float('inf')
    optimal_solution_found = True
    total_combinations = 0

    print(f"Running brute-force algorithm for k = {k} on graph with {len(G.nodes)} nodes.")

    # Check all combinations of k nodes
    for comb in itertools.combinations(G.nodes, k):
        total_combinations += 1
        elapsed = default_timer() - start_time
        if elapsed > time_limit:
            optimal_solution_found = False
            print("Time limit exceeded.")
            break

        # Evaluate current combination
        cost = k_centers_objective_value(G, comb)
        print(f"Checked centers: {comb}, cost = {cost:.4f}")

        # Update best solution if current one is better
        if cost < best_cost:
            best_cost = cost
            best_combination = comb
            print(f"New best solution: {best_combination}, cost = {best_cost:.4f}")

    execution_time = default_timer() - start_time
    print(f"Finished. Total combinations tried: {total_combinations}")
    print(f"Execution time: {execution_time:.2f} seconds")

    best_centers = list(best_combination) if best_combination else []
    return best_combination, best_centers, optimal_solution_found, execution_time


def extract_node_count(filename):
    """
    Extracts the number of nodes from a filename of the form:
    graph_XXXX_YYYY.gexf

    Returns:
        int : number of nodes
    """
    match = re.search(r'graph_(\d+)_\d+\.gexf', filename)
    return int(match.group(1)) if match else -1


def run_brute_force_on_random_graph(folder_path, k=3, time_limit=30, max_nodes=100):
    """
    Selects a random .gexf graph file from the folder, subject to a maximum
    node count, and runs the brute-force k-center algorithm on it.

    Parameters:
        folder_path : str, directory containing .gexf files
        k : int, number of centers
        time_limit : float, max execution time per run
        max_nodes : int, only use graphs with ≤ max_nodes
    """
    # Filter files by extension and node count
    files = [f for f in os.listdir(folder_path) if f.endswith(".gexf") and extract_node_count(f) <= max_nodes]
    if not files:
        print(f"No graphs with ≤ {max_nodes} nodes found.")
        return

    # Randomly select a graph
    random_file = random.choice(files)
    full_path = os.path.join(folder_path, random_file)
    node_count = extract_node_count(random_file)

    print(f"Selected random graph: {random_file} with {node_count} nodes")

    # Load and solve
    G = nx.read_gexf(full_path)
    best_tuple, best_list, success, exec_time = k_centers_brute_force(G, k, time_limit)

    # Final output
    print("\n=== Final Result ===")
    print(f"Best centers: {best_list}")
    print(f"Optimal solution found: {'Yes' if success else 'No'}")
    print(f"Execution time: {exec_time:.2f} seconds")


# Entry point
if __name__ == "__main__":
    run_brute_force_on_random_graph("graphs", k=3, time_limit=60)
