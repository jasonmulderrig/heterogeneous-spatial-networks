import numpy as np
import networkx as nx
from general_topological_descriptors import n_func

def k_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Node degree.

    This function calculates the node degree in an (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Node degree.
    
    """
    return np.asarray(list(graph.degree()), dtype=int)[:, 1]

def k_avrg_nn_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average nearest neighbor degree.

    This function calculates the average nearest neighbor degree for
    each node in an (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Nodewise average nearest neighbor degree.
    
    """
    return np.asarray(list(nx.average_neighbor_degree(graph).items()))[:, 1]

def k_diff_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Degree difference.

    This function calculates the degree difference for each edge in an
    (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Edgewise degree difference.
    
    """
    # Calculate and store graph degree in a dictionary
    k_dict = dict(graph.degree())
    
    # Gather edges and initialize degree difference np.ndarray
    edges = list(graph.edges())
    k_diff = np.empty(len(edges), dtype=int)

    # Calculate and store the degree difference of each edge
    for edge_indx, edge in enumerate(edges):
        # Node numbers
        node_0 = int(edge[0])
        node_1 = int(edge[1])

        # Calculate and store the degree difference
        k_diff[edge_indx] = np.abs(k_dict[node_0]-k_dict[node_1], dtype=int)
    
    return k_diff

def c_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Clustering coefficient.

    This function calculates the clustering coefficient for each node in
    an (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Nodewise clustering coefficient.
    
    """
    # NetworkX clustering function only accepts graphs of type
    # nx.Graph()
    if graph.is_multigraph(): graph = nx.Graph(graph)
    
    return np.asarray(list(nx.clustering(graph).items()))[:, 1]

def kappa_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Nodal connectivity.

    This function calculates the nodal connectivity for all pairs of
    nodes in an (undirected) graph (excluding all self-loop node pairs).

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Node-pairwise nodal connectivity, excluding all
        self-loop node pairs. Thus, for a network with n nodes labeled
        {0, 1, ..., n-1}, the first n-1 entries are associated with the
        nodal connectivities for all non-self-loop node pairs for the
        zeroth node (0-1, 0-2, ..., 0-(n-1)). The next n-1 entries are
        associated with that for the first node (1-0, 1-2, 1-3, ...,
        1-(n-1)), and so on.
    
    """
    # Remove self-loops and isolate nodes from the as-provided graph
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    graph.remove_nodes_from(list(nx.isolates(graph)))

    kappa_dict = nx.all_pairs_node_connectivity(graph)
    node_list = list(graph.nodes())
    n = n_func(graph)
    n_pairs = n * (n-1)
    kappa = np.empty(n_pairs, dtype=int)

    indx = 0
    for node_0 in node_list:
        for node_1 in node_list:
            if node_0 == node_1: continue
            else:
                kappa[indx] = kappa_dict[node_0][node_1]
                indx += 1
    
    return kappa

def avrg_kappa_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Average nodal connectivity.

    This function calculates the average nodal connectivity for all
    pairs of nodes in an (undirected) graph. The nodal connectivity for
    each pair of nodes is averaged over all n choose 2 pairs of nodes.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        float: Average node-pairwise nodal connectivity.
    
    """
    # Remove self-loops and isolate nodes from the as-provided graph
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    graph.remove_nodes_from(list(nx.isolates(graph)))

    return nx.average_node_connectivity(graph)

def lambda_1_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Algebraic connectivity.

    This function calculates the algebraic connectivity (the
    second-smallest eigenvalue of the Laplacian matrix) for an
    (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        float: Algebraic connectivity.
    
    """
    # Remove self-loops and isolate nodes from the as-provided graph
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    
    # NetworkX algebraic connectivity function is best applied to fully
    # connected graphs
    if nx.is_connected(graph):
        return nx.algebraic_connectivity(graph)
    else:
        # Calculate the algebraic connectivity on the maximal component
        # of the graph
        mx_cmp_nodes = max(nx.connected_components(graph), key=len)
        mx_cmp_graph = graph.subgraph(mx_cmp_nodes).copy()
        return nx.algebraic_connectivity(mx_cmp_graph)

def r_pearson_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Degree assortativity coefficient using the Pearson correlation
    coefficient.

    This function calculates the degree assortativity coefficient using
    the Pearson correlation coefficient for an (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        float: Degree assortativity coefficient.
    
    """
    return nx.degree_pearson_correlation_coefficient(graph)
