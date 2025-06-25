"""
Custom k-Center Solver (Local Search + Random Init)
---------------------------------------------------

This script defines a custom heuristic algorithm for the k-center problem using:
1. Random sampling to find an initial set of centers.
2. Local search to improve the solution by trying center replacements.

"""

import networkx as nx
import random
from random import Random
from timeit import default_timer

def k_centers_objective_value(G, centers):
    """
    Compute max distance from any node to its nearest center.
    """
    path = nx.multi_source_dijkstra_path_length(G, centers)
    return max(path.values())

def my_k_center(G, k, sample_count):
    """
    Random sampling to find a good starting solution.

    Parameters:
        G : networkx.Graph
        k : number of centers
        sample_count : number of random samples to try

    Returns:
        list : best center list found
    """
    best_cost = float('inf')
    best_centers = []
    seed = random.randint(0, 1000000)
    rng = Random(seed)

    for _ in range(sample_count):
        centers = rng.sample(list(G.nodes), k)
        cost = k_centers_objective_value(G, centers)
        if cost < best_cost:
            best_cost = cost
            best_centers = centers

    return best_centers

def k_centers_custom(G, centers, time_limit):
    """
    Local search algorithm: tries to improve an initial set of centers
    by swapping each one with other nodes.

    Parameters:
        G : networkx.Graph
        centers : list, initial center nodes
        time_limit : float, in seconds

    Returns:
        new_centers : list of final center nodes
        best_cost : objective value of the final solution
    """
    start = default_timer()
    best_centers = centers[:]
    best_cost = k_centers_objective_value(G, best_centers)
    optimal_solution_found = False

    while True:
        improved = False

        for i, old_center in enumerate(best_centers):
            for candidate in G.nodes:
                # Check time limit
                if default_timer() - start > time_limit:
                    optimal_solution_found = False
                    print("Time limit reached during local search.")
                    return best_centers, best_cost

                if candidate in best_centers:
                    continue

                # Try replacing one center
                new_centers = best_centers[:]
                new_centers[i] = candidate
                new_cost = k_centers_objective_value(G, new_centers)

                if new_cost < best_cost:
                    best_centers = new_centers
                    best_cost = new_cost
                    improved = True
                    print(f"Improved solution: {best_centers}, cost = {best_cost:.4f}")

        if not improved:
            break

    optimal_solution_found = True
    return best_centers, best_cost

if __name__ == "__main__":
    G = nx.read_gexf("graphs/graph_0050_07481.gexf")
    initial_centers = my_k_center(G, k=3, sample_count=100)
    final_centers, final_cost = k_centers_custom(G, initial_centers, time_limit=30)

    print("\n=== Custom Algorithm Result ===")
    print(f"Initial centers: {initial_centers}")
    print(f"Final centers: {final_centers}")
    print(f"Final cost: {final_cost:.4f}")
