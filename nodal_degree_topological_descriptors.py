import numpy as np
import networkx as nx

def k_counts_calculation(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Node degree counts.

    This function calculates the node degree counts in an (undirected)
    graph, where the node degree can be between 1 and 8, inclusive.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Node degree counts, where the node degree can be between 1 and 8, inclusive.
    
    """
    # Initialize the node degree counts, where k\in[1, 8]. Thus,
    # k_counts[k-1] = counts, i.e., k_counts[0] = number of nodes with
    # k = 1, k_counts[1] = number of nodes with k = 2, ...,
    # k_counts[7] = number of nodes with k = 8.
    k_counts = np.zeros(8, dtype=int)

    # Calculate number of occurrences for each node degree
    graph_k, graph_k_counts = np.unique(
        np.asarray(list(graph.degree()), dtype=int)[:, 1], return_counts=True)

    # Store the node degree counts
    for k_indx in range(np.shape(graph_k)[0]):
        if graph_k[k_indx] == 0: continue
        else: k_counts[graph_k[k_indx]-1] = graph_k_counts[k_indx]
    
    return k_counts