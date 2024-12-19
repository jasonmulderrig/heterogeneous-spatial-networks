import numpy as np
import networkx as nx
from general_topological_descriptors import (
    n_func,
    l_func
)
from graph_utils import largest_connected_component

def r_func(graph: nx.Graph | nx.MultiGraph) -> int:
    """Graph radius.

    This function calculates the radius (minimum eccentricity) of a
    given graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        int: Graph radius.
    
    """
    # NetworkX radius function is best applied to fully connected graphs
    return nx.radius(largest_connected_component(graph))

def sigma_func(graph: nx.Graph | nx.MultiGraph) -> int:
    """Graph diameter.

    This function calculates the diameter (maximum eccentricity) of a
    given graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        int: Graph diameter.
    
    """
    # NetworkX diameter function is best applied to fully connected
    # graphs
    return nx.diameter(largest_connected_component(graph))

def epsilon_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Graph eccentricity.

    This function calculates the eccentricity (the maximum shortest
    path) for each node in an (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Nodewise graph eccentricity.
    
    """
    # NetworkX eccentricity function is best applied to fully connected
    # graphs
    return (
        np.asarray(
            list(
                nx.eccentricity(
                    largest_connected_component(graph)).values()), dtype=int)
    )

def d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Shortest path length.

    This function calculates the shortest path lenth for all pairs of
    nodes in an (undirected) graph (excluding all self-loop node pairs).

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Node-pairwise shortest path length, excluding all
        self-loop node pairs. Thus, for a network with n nodes labeled
        {0, 1, ..., n-1}, the first n-1 entries are associated with the
        shortest path length for all non-self-loop node pairs for the
        zeroth node (0-1, 0-2, ..., 0-(n-1)). The next n-1 entries are
        associated with that for the first node (1-0, 1-2, 1-3, ...,
        1-(n-1)), and so on.
    
    """
    # NetworkX shortest path length function is best applied to fully
    # connected graphs
    graph = largest_connected_component(graph)

    d_dict = dict(nx.shortest_path_length(graph))
    node_list = list(graph.nodes())
    n = n_func(graph)
    n_pairs = n * (n-1)
    d = np.empty(n_pairs, dtype=int)

    indx = 0
    for node_0 in node_list:
        for node_1 in node_list:
            if node_0 == node_1: continue
            else:
                d[indx] = d_dict[node_0][node_1]
                indx += 1
    
    return d

def avrg_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average shortest path length.

    This function calculates the average shortest path lenth for each
    node in an (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Nodewise average shortest path length.
    
    """
    # NetworkX shortest path length function is best applied to fully
    # connected graphs
    graph = largest_connected_component(graph)

    d_dict = dict(nx.shortest_path_length(graph))
    n = n_func(graph)
    avrg_d = np.empty(n)

    indx = 0
    for node in d_dict:
        avrg_d[indx] = sum(d_dict[node].values()) / (n-1)
        indx += 1
    
    return avrg_d

def e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Graph efficiency.

    This function calculates the efficiency for all pairs of nodes in an
    (undirected) graph (excluding all self-loop node pairs).

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Node-pairwise efficiency, excluding all self-loop
        node pairs. Thus, for a network with n nodes labeled {0, 1, ...,
        n-1}, the first n-1 entries are associated with the efficiency
        for all non-self-loop node pairs for the zeroth node (0-1, 0-2,
        ..., 0-(n-1)). The next n-1 entries are associated with that for
        the first node (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    return np.reciprocal(d_func(graph), dtype=float)

def avrg_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average graph efficiency.

    This function calculates the average efficiency for each node in an
    (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Nodewise average graph efficiency.
    
    """
    # NetworkX shortest path length function is best applied to fully
    # connected graphs
    graph = largest_connected_component(graph)

    d_dict = dict(nx.shortest_path_length(graph))
    node_list = list(graph.nodes())
    n = n_func(graph)
    avrg_e = np.empty(n)

    indx = 0
    for node_0 in node_list:
        avrg_e_sum = 0.
        for node_1 in node_list:
            if node_0 == node_1: continue
            else:
                avrg_e_sum += np.reciprocal(d_dict[node_0][node_1], dtype=float)
        avrg_e[indx] = avrg_e_sum / (n-1)
        indx += 1
    
    return avrg_e

def lcl_e_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Local graph efficiency.

    This function calculates the average local efficiency for an
    (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        float: Graph local efficiency.
    
    """
    # NetworkX local efficiency function is best applied to fully
    # connected graphs
    return nx.local_efficiency(largest_connected_component(graph))

def glbl_e_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Global graph efficiency.

    This function calculates the average global efficiency for an
    (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        float: Graph global efficiency.
    
    """
    # NetworkX global efficiency function is best applied to fully
    # connected graphs
    return nx.global_efficiency(largest_connected_component(graph))

def n_bc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Node shortest path betweenness centrality.

    This function calculates the shortest path betweenness centrality
    for each node in an (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Nodewise betweenness centrality.
    
    """
    # NetworkX (node) betweenness centrality function is best applied to
    # fully connected graphs
    return (
        np.asarray(
            list(
                nx.betweenness_centrality(
                    largest_connected_component(graph)).values()))
    )

def m_bc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Edge shortest path betweenness centrality.

    This function calculates the shortest path betweenness centrality
    for each edge in an (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Edgewise betweenness centrality.
    
    """
    # NetworkX edge betweenness centrality function is best applied to
    # fully connected graphs
    return (
        np.asarray(
            list(
                nx.edge_betweenness_centrality(
                    largest_connected_component(graph)).values()))
    )

def cc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Closeness centrality.

    This function calculates the closeness centrality for each node in
    an (undirected) graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        np.ndarray: Nodewise closeness centrality.
    
    """
    # NetworkX closeness centrality function is best applied to fully
    # connected graphs
    return np.asarray(list(nx.closeness_centrality(graph).values()))

def scc_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Spatial (Amamoto) closeness centrality.

    This function calculates the spatial closeness centrality for each
    node in an (undirected) graph, as defined by Amamoto et al.,
    Patterns (2020).

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Nodewise spatial (Amamoto) closeness centrality.
    
    """
    # NetworkX closeness centrality function is best applied to fully
    # connected graphs
    conn_graph = largest_connected_component(conn_graph)
    
    # Remove/Discard self-loops, which render a spatial (Amamoto)
    # closeness centrality value of infinity
    conn_graph.remove_edges_from(list(nx.selfloop_edges(conn_graph)))

    # Extract the core and periodic boundary graphs
    conn_core_graph = conn_core_graph.subgraph(list(conn_graph.nodes())).copy()
    conn_pb_graph = conn_pb_graph.subgraph(list(conn_graph.nodes())).copy()

    # Calculate inverse edge length
    l_inv = np.reciprocal(
        l_func(conn_core_graph, conn_pb_graph, conn_graph, coords, L))
    
    # Initialize edge weight attribute
    nx.set_edge_attributes(conn_graph, values=1, name="l_inv")

    # Set edge weight attribute to the inverse edge length for each edge
    for edge_indx, edge in enumerate(list(conn_graph.edges())):
        # Node numbers
        node_0 = int(edge[0])
        node_1 = int(edge[1])

        # Set edge weight attribute to the inverse edge length
        if conn_graph.is_multigraph():
            for multiedge_indx in range(conn_graph.number_of_edges(node_0, node_1)):
                conn_graph.edges[node_0, node_1, multiedge_indx]["l_inv"] = (
                    l_inv[edge_indx]
                )
        else:
            conn_graph.edges[node_0, node_1]["l_inv"] = l_inv[edge_indx]
    
    # Calculate spatial (Amamoto) closeness centrality
    return (
        np.asarray(
            list(
                nx.closeness_centrality(
                    conn_graph, distance="l_inv").values()))
    )