import numpy as np
import networkx as nx
from topological_descriptors.general_topological_descriptors import (
    n_func,
    l_func
)

def shrt_path_l_attr_graph(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> nx.Graph | nx.MultiGraph:
    """Euclidean edge length edge attributed graph for shortest path
    length-related analysis.

    This function adds Euclidean edge lengths as edge attributes to an
    undirected NetworkX graph for shortest path length-related graph
    analysis.
    
    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph that
        captures the periodic connections between the core nodes with
        Euclidean edge length edge attributes. Intended for use in
        shortest path length-related analysis.
    
    """
    # Calculate Euclidean edge lengths
    l = l_func(conn_core_graph, conn_pb_graph, conn_graph, coords, L)

    # Clear edge attributes
    for node_0, node_1, attr in list(conn_graph.edges(data=True)): attr.clear()
    
    # Set edge weight attribute to the Euclidean edge length for each
    # edge
    for edge_indx, edge in enumerate(list(conn_graph.edges())):
        # Node numbers
        node_0 = int(edge[0])
        node_1 = int(edge[1])

        # Set edge weight attribute to the Euclidean edge length
        if conn_graph.is_multigraph():
            for multiedge_indx in range(conn_graph.number_of_edges(node_0, node_1)):
                conn_graph.edges[node_0, node_1, multiedge_indx]["l"] = (
                    l[edge_indx]
                )
        else:
            conn_graph.edges[node_0, node_1]["l"] = l[edge_indx]
    
    return conn_graph

def shrt_path_l_inv_attr_graph(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> nx.Graph | nx.MultiGraph:
    """Inverse Euclidean edge length edge attributed graph for shortest
    path length-related analysis.

    This function adds inverse Euclidean edge lengths as edge attributes
    to an undirected NetworkX graph for shortest path length-related
    graph analysis.
    
    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph that
        captures the periodic connections between the core nodes with
        inverse Euclidean edge length edge attributes. Intended for use
        in shortest path length-related analysis.
    
    """
    # Calculate Euclidean edge lengths
    l = l_func(conn_core_graph, conn_pb_graph, conn_graph, coords, L)

    # Calculate inverse Euclidean edge length (where the inverse
    # Euclidean edge length for self-loops is set to zero)
    l_inv = np.reciprocal(l, where=l!=0.0)

    # Clear edge attributes
    for node_0, node_1, attr in list(conn_graph.edges(data=True)): attr.clear()
    
    # Set edge weight attribute to the inverse Euclidean edge length for
    # each edge
    for edge_indx, edge in enumerate(list(conn_graph.edges())):
        # Node numbers
        node_0 = int(edge[0])
        node_1 = int(edge[1])

        # Set edge weight attribute to the inverse Euclidean edge length
        if conn_graph.is_multigraph():
            for multiedge_indx in range(conn_graph.number_of_edges(node_0, node_1)):
                conn_graph.edges[node_0, node_1, multiedge_indx]["l_inv"] = (
                    l_inv[edge_indx]
                )
        else:
            conn_graph.edges[node_0, node_1]["l_inv"] = l_inv[edge_indx]
    
    return conn_graph

def r_func(graph: nx.Graph | nx.MultiGraph) -> int:
    """Geodesic graph radius.

    This function calculates the geodesic radius (minimum eccentricity)
    of a given graph, where each edge has a unit weight. This function
    is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        int: Geodesic graph radius.
    
    """
    return nx.radius(graph)

def l_attr_r_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> float:
    """Euclidean edge length-weighted graph radius.

    This function calculates the Euclidean edge length-weighted radius
    (minimum eccentricity) of a given graph. This function is best
    applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        float: Euclidean edge length-weighted graph radius.
    
    """
    graph = shrt_path_l_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return nx.radius(graph, weight="l")

def l_inv_attr_r_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> float:
    """Inverse Euclidean edge length-weighted graph radius.

    This function calculates the inverse Euclidean edge length-weighted
    radius (minimum eccentricity) of a given graph. This function is
    best applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        float: Inverse Euclidean edge length-weighted graph radius.
    
    """
    graph = shrt_path_l_inv_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return nx.radius(graph, weight="l_inv")

def sigma_func(graph: nx.Graph | nx.MultiGraph) -> int:
    """Geodesic graph diameter.

    This function calculates the geodesic diameter (maximum
    eccentricity) of a given graph, where each edge has a unit weight.
    This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        int: Geodesic graph diameter.
    
    """
    return nx.diameter(graph)

def l_attr_sigma_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> float:
    """Euclidean edge length-weighted graph diameter.

    This function calculates the Euclidean edge length-weighted diameter
    (maximum eccentricity) of a given graph. This function is best
    applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        float: Euclidean edge length-weighted graph diameter.
    
    """
    graph = shrt_path_l_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return nx.diameter(graph, weight="l")

def l_inv_attr_sigma_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> float:
    """Inverse Euclidean edge length-weighted graph diameter.

    This function calculates the inverse Euclidean edge length-weighted
    diameter (maximum eccentricity) of a given graph. This function is
    best applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        float: Inverse Euclidean edge length-weighted graph diameter.
    
    """
    graph = shrt_path_l_inv_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return nx.diameter(graph, weight="l_inv")

def epsilon_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic graph eccentricity.

    This function calculates the geodesic eccentricity (the maximum
    shortest path) for each node in an (undirected) graph, where each
    edge has a unit weight. This function is best applied to fully
    connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise geodesic graph eccentricity.
    
    """
    return np.asarray(list(nx.eccentricity(graph).values()), dtype=int)

def l_attr_epsilon_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Euclidean edge length-weighted graph eccentricity.

    This function calculates the Euclidean edge length-weighted
    eccentricity (the maximum shortest path) for each node in an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Euclidean edge length-weighted nodewise graph
        eccentricity.
    
    """
    graph = shrt_path_l_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return np.asarray(list(nx.eccentricity(graph, weight="l").values()))

def l_inv_attr_epsilon_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Inverse Euclidean edge length-weighted graph eccentricity.

    This function calculates the inverse Euclidean edge length-weighted
    eccentricity (the maximum shortest path) for each node in an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Inverse Euclidean edge length-weighted nodewise
        graph eccentricity.
    
    """
    graph = shrt_path_l_inv_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return np.asarray(list(nx.eccentricity(graph, weight="l_inv").values()))

def d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic shortest path length.

    This function calculates the geodesic shortest path length for all
    pairs of nodes in an (undirected) graph (excluding all self-loop
    node pairs), where each edge has a unit weight. This function is
    best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise geodesic shortest path length,
        excluding all self-loop node pairs. Thus, for a network with n
        nodes labeled {0, 1, ..., n-1}, the first n-1 entries are
        associated with the shortest path length for all non-self-loop
        node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
        next n-1 entries are associated with that for the first node
        (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
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

def l_attr_d_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Euclidean edge length-weighted shortest path length.

    This function calculates the Euclidean edge length-weighted shortest
    path length for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Node-pairwise Euclidean edge length-weighted
        shortest path length, excluding all self-loop node pairs. Thus,
        for a network with n nodes labeled {0, 1, ..., n-1}, the first
        n-1 entries are associated with the shortest path length for all
        non-self-loop node pairs for the zeroth node (0-1, 0-2, ...,
        0-(n-1)). The next n-1 entries are associated with that for the
        first node (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    graph = shrt_path_l_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    d_dict = dict(nx.shortest_path_length(graph, weight="l"))
    node_list = list(graph.nodes())
    n = n_func(graph)
    n_pairs = n * (n-1)
    d = np.empty(n_pairs)

    indx = 0
    for node_0 in node_list:
        for node_1 in node_list:
            if node_0 == node_1: continue
            else:
                d[indx] = d_dict[node_0][node_1]
                indx += 1
    
    return d

def l_inv_attr_d_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Inverse Euclidean edge length-weighted shortest path length.

    This function calculates the inverse Euclidean edge length-weighted
    shortest path length for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Node-pairwise inverse Euclidean edge length-weighted
        shortest path length, excluding all self-loop node pairs. Thus,
        for a network with n nodes labeled {0, 1, ..., n-1}, the first
        n-1 entries are associated with the shortest path length for all
        non-self-loop node pairs for the zeroth node (0-1, 0-2, ...,
        0-(n-1)). The next n-1 entries are associated with that for the
        first node (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    graph = shrt_path_l_inv_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    d_dict = dict(nx.shortest_path_length(graph, weight="l_inv"))
    node_list = list(graph.nodes())
    n = n_func(graph)
    n_pairs = n * (n-1)
    d = np.empty(n_pairs)

    indx = 0
    for node_0 in node_list:
        for node_1 in node_list:
            if node_0 == node_1: continue
            else:
                d[indx] = d_dict[node_0][node_1]
                indx += 1
    
    return d

def avrg_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average geodesic shortest path length.

    This function calculates the average geodesic shortest path length
    for each node in an (undirected) graph, where each edge has a unit
    weight. This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average geodesic shortest path length.
    
    """
    d_dict = dict(nx.shortest_path_length(graph))
    n = n_func(graph)
    avrg_d = np.empty(n)

    indx = 0
    for node in d_dict:
        avrg_d[indx] = sum(d_dict[node].values()) / (n-1)
        indx += 1
    
    return avrg_d

def avrg_l_attr_d_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Average Euclidean edge length-weighted shortest path length.

    This function calculates the average Euclidean edge length-weighted
    shortest path length for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Nodewise average Euclidean edge length-weighted
        shortest path length.
    
    """
    graph = shrt_path_l_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    d_dict = dict(nx.shortest_path_length(graph, weight="l"))
    n = n_func(graph)
    avrg_d = np.empty(n)

    indx = 0
    for node in d_dict:
        avrg_d[indx] = sum(d_dict[node].values()) / (n-1)
        indx += 1
    
    return avrg_d

def avrg_l_inv_attr_d_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Average inverse Euclidean edge length-weighted shortest path
    length.

    This function calculates the average inverse Euclidean edge
    length-weighted shortest path length for each node in an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Nodewise average inverse Euclidean edge
        length-weighted shortest path length.
    
    """
    graph = shrt_path_l_inv_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    d_dict = dict(nx.shortest_path_length(graph, weight="l_inv"))
    n = n_func(graph)
    avrg_d = np.empty(n)

    indx = 0
    for node in d_dict:
        avrg_d[indx] = sum(d_dict[node].values()) / (n-1)
        indx += 1
    
    return avrg_d

def e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic graph efficiency.

    This function calculates the geodesic efficiency for all pairs of
    nodes in an (undirected) graph (excluding all self-loop node pairs),
    where each edge has a unit weight. This function is best applied to
    fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise geodesic graph efficiency, excluding
        all self-loop node pairs. Thus, for a network with n nodes
        labeled {0, 1, ..., n-1}, the first n-1 entries are associated
        with the efficiency for all non-self-loop node pairs for the
        zeroth node (0-1, 0-2, ..., 0-(n-1)). The next n-1 entries are
        associated with that for the first node (1-0, 1-2, 1-3, ...,
        1-(n-1)), and so on.
    
    """
    d = d_func(graph)
    d = d.astype(float)
    return np.reciprocal(d, where=d!=0.0)

def l_attr_e_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Euclidean edge length-weighted graph efficiency.

    This function calculates the Euclidean edge length-weighted
    efficiency for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Node-pairwise Euclidean edge length-weighted
        efficiency, excluding all self-loop node pairs. Thus, for a
        network with n nodes labeled {0, 1, ..., n-1}, the first n-1
        entries are associated with the efficiency for all non-self-loop
        node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
        next n-1 entries are associated with that for the first node
        (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    d = l_attr_d_func(conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return np.reciprocal(d, where=d!=0.0)

def l_inv_attr_e_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Inverse Euclidean edge length-weighted graph efficiency.

    This function calculates the inverse Euclidean edge length-weighted
    efficiency for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Node-pairwise inverse Euclidean edge length-weighted
        efficiency, excluding all self-loop node pairs. Thus, for a
        network with n nodes labeled {0, 1, ..., n-1}, the first n-1
        entries are associated with the efficiency for all non-self-loop
        node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
        next n-1 entries are associated with that for the first node
        (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    d = l_inv_attr_d_func(conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return np.reciprocal(d, where=d!=0.0)

def avrg_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average geodesic graph efficiency.

    This function calculates the average geodesic efficiency for each
    node in an (undirected) graph, where each edge has a unit weight.
    This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average geodesic graph efficiency.
    
    """
    d_dict = dict(nx.shortest_path_length(graph))
    node_list = list(graph.nodes())
    n = n_func(graph)
    avrg_e = np.empty(n)

    indx = 0
    for node_0 in node_list:
        avrg_e_sum = 0.
        for node_1 in node_list:
            d = d_dict[node_0][node_1] * 1.0
            avrg_e_sum += np.reciprocal(d, where=d!=0.0)
        avrg_e[indx] = avrg_e_sum / (n-1)
        indx += 1
    
    return avrg_e

def avrg_l_attr_e_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Average Euclidean edge length-weighted graph efficiency.

    This function calculates the average Euclidean edge length-weighted
    efficiency for each node in an (undirected) graph, where each edge
    has a unit weight. This function is best applied to fully connected
    graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Nodewise average Euclidean edge length-weighted
        graph efficiency.
    
    """
    graph = shrt_path_l_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    d_dict = dict(nx.shortest_path_length(graph, weight="l"))
    node_list = list(graph.nodes())
    n = n_func(graph)
    avrg_e = np.empty(n)

    indx = 0
    for node_0 in node_list:
        avrg_e_sum = 0.
        for node_1 in node_list:
            d = d_dict[node_0][node_1] * 1.0
            avrg_e_sum += np.reciprocal(d, where=d!=0.0)
        avrg_e[indx] = avrg_e_sum / (n-1)
        indx += 1
    
    return avrg_e

def avrg_l_inv_attr_e_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Average inverse Euclidean edge length-weighted graph efficiency.

    This function calculates the average inverse Euclidean edge
    length-weighted efficiency for each node in an (undirected) graph,
    where each edge has a unit weight. This function is best applied to
    fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Nodewise average inverse Euclidean edge
        length-weighted graph efficiency.
    
    """
    graph = shrt_path_l_inv_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    d_dict = dict(nx.shortest_path_length(graph, weight="l_inv"))
    node_list = list(graph.nodes())
    n = n_func(graph)
    avrg_e = np.empty(n)

    indx = 0
    for node_0 in node_list:
        avrg_e_sum = 0.
        for node_1 in node_list:
            d = d_dict[node_0][node_1] * 1.0
            avrg_e_sum += np.reciprocal(d, where=d!=0.0)
        avrg_e[indx] = avrg_e_sum / (n-1)
        indx += 1
    
    return avrg_e

def bc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic node betweenness centrality.

    This function calculates the geodesic betweenness centrality for
    each node in an (undirected) graph, where each edge has a unit
    weight. This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise geodesic betweenness centrality.
    
    """
    return np.asarray(list(nx.betweenness_centrality(graph).values()))

def l_attr_bc_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Euclidean edge length-weighted node betweenness centrality.

    This function calculates the Euclidean edge length-weighted
    betweenness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Nodewise Euclidean edge length-weighted betweenness
        centrality.
    
    """
    graph = shrt_path_l_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return (
        np.asarray(list(nx.betweenness_centrality(graph, weight="l").values()))
    )

def l_inv_attr_bc_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Inverse Euclidean edge length-weighted node betweenness
    centrality.

    This function calculates the inverse Euclidean edge length-weighted
    betweenness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Nodewise inverse Euclidean edge length-weighted
        betweenness centrality.
    
    """
    graph = shrt_path_l_inv_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return (
        np.asarray(
            list(nx.betweenness_centrality(graph, weight="l_inv").values()))
    )

def ebc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic edge betweenness centrality.

    This function calculates the geodesic edge betweenness
    centrality for each edge in an (undirected) graph, where each edge
    has a unit weight. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Geodesic edgewise edge betweenness centrality.
    
    """
    return np.asarray(list(nx.edge_betweenness_centrality(graph).values()))

def l_attr_ebc_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Euclidean edge length-weighted edge betweenness centrality.

    This function calculates the Euclidean edge length-weighted edge
    betweenness centrality for each edge in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Edgewise Euclidean edge length-weighted edge
        betweenness centrality.
    
    """
    graph = shrt_path_l_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return (
        np.asarray(
            list(nx.edge_betweenness_centrality(graph, weight="l").values()))
    )

def l_inv_attr_ebc_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Inverse Euclidean edge length-weighted edge betweenness
    centrality.

    This function calculates the inverse Euclidean edge length-weighted
    edge betweenness centrality for each edge in an (undirected) graph.
    This function is best applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Edgewise inverse Euclidean edge length-weighted edge
        betweenness centrality.
    
    """
    graph = shrt_path_l_inv_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return (
        np.asarray(
            list(nx.edge_betweenness_centrality(graph, weight="l_inv").values()))
    )

def cc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic closeness centrality.

    This function calculates the geodesic closeness centrality for each
    node in an (undirected) graph, where each edge has a unit weight.
    This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise geodesic closeness centrality.
    
    """
    return np.asarray(list(nx.closeness_centrality(graph).values()))

def l_attr_cc_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Euclidean edge length-weighted closeness centrality.

    This function calculates the Euclidean edge length-weighted
    closeness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Nodewise Euclidean edge length-weighted closeness
        centrality.
    
    """
    graph = shrt_path_l_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return (
        np.asarray(list(nx.closeness_centrality(graph, distance="l").values()))
    )

def l_inv_attr_cc_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Inverse Euclidean edge length-weighted closeness centrality.

    This function calculates the inverse Euclidean edge length-weighted
    closeness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Nodewise inverse Euclidean edge length-weighted
        closeness centrality.
    
    """
    graph = shrt_path_l_inv_attr_graph(
        conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    return (
        np.asarray(
            list(nx.closeness_centrality(graph, distance="l_inv").values()))
    )

# import numpy as np
# import networkx as nx
# from topological_descriptors.general_topological_descriptors import (
#     n_func,
#     l_func
# )

# def shrt_path_l_attr_graph(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> nx.Graph | nx.MultiGraph:
#     """Euclidean edge length edge attributed graph for shortest path
#     length-related analysis.

#     This function adds Euclidean edge lengths as edge attributes to an
#     undirected NetworkX graph for shortest path length-related graph
#     analysis.
    
#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph that
#         captures the periodic connections between the core nodes with
#         Euclidean edge length edge attributes. Intended for use in
#         shortest path length-related analysis.
    
#     """
#     # Calculate Euclidean edge lengths
#     l = l_func(conn_core_graph, conn_pb_graph, conn_graph, coords, L)

#     # Clear edge attributes
#     for node_0, node_1, attr in list(conn_graph.edges(data=True)): attr.clear()
    
#     # Set edge weight attribute to the Euclidean edge length for each
#     # edge
#     for edge_indx, edge in enumerate(list(conn_graph.edges())):
#         # Node numbers
#         node_0 = int(edge[0])
#         node_1 = int(edge[1])

#         # Set edge weight attribute to the Euclidean edge length
#         if conn_graph.is_multigraph():
#             for multiedge_indx in range(conn_graph.number_of_edges(node_0, node_1)):
#                 conn_graph.edges[node_0, node_1, multiedge_indx]["l"] = (
#                     l[edge_indx]
#                 )
#         else:
#             conn_graph.edges[node_0, node_1]["l"] = l[edge_indx]
    
#     return conn_graph

# def shrt_path_l_inv_attr_graph(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> nx.Graph | nx.MultiGraph:
#     """Inverse Euclidean edge length edge attributed graph for shortest
#     path length-related analysis.

#     This function adds inverse Euclidean edge lengths as edge attributes
#     to an undirected NetworkX graph for shortest path length-related
#     graph analysis.
    
#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph that
#         captures the periodic connections between the core nodes with
#         inverse Euclidean edge length edge attributes. Intended for use
#         in shortest path length-related analysis.
    
#     """
#     # Calculate Euclidean edge lengths
#     l = l_func(conn_core_graph, conn_pb_graph, conn_graph, coords, L)

#     # Calculate inverse Euclidean edge length (where the inverse
#     # Euclidean edge length for self-loops is set to zero)
#     l_inv = np.reciprocal(l, where=l!=0.0)

#     # Clear edge attributes
#     for node_0, node_1, attr in list(conn_graph.edges(data=True)): attr.clear()
    
#     # Set edge weight attribute to the inverse Euclidean edge length for
#     # each edge
#     for edge_indx, edge in enumerate(list(conn_graph.edges())):
#         # Node numbers
#         node_0 = int(edge[0])
#         node_1 = int(edge[1])

#         # Set edge weight attribute to the inverse Euclidean edge length
#         if conn_graph.is_multigraph():
#             for multiedge_indx in range(conn_graph.number_of_edges(node_0, node_1)):
#                 conn_graph.edges[node_0, node_1, multiedge_indx]["l_inv"] = (
#                     l_inv[edge_indx]
#                 )
#         else:
#             conn_graph.edges[node_0, node_1]["l_inv"] = l_inv[edge_indx]
    
#     return conn_graph

# def r_func(graph: nx.Graph | nx.MultiGraph) -> int:
#     """Geodesic graph radius.

#     This function calculates the geodesic radius (minimum eccentricity)
#     of a given graph, where each edge has a unit weight. This function
#     is best applied to fully connected graphs.

#     Args:
#         graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
#     Returns:
#         int: Geodesic graph radius.
    
#     """
#     return nx.radius(graph)

# def l_attr_r_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> float:
#     """Euclidean edge length-weighted graph radius.

#     This function calculates the Euclidean edge length-weighted radius
#     (minimum eccentricity) of a given graph. This function is best
#     applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         float: Euclidean edge length-weighted graph radius.
    
#     """
#     graph = shrt_path_l_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return nx.radius(graph, weight="l")

# def l_inv_attr_r_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> float:
#     """Inverse Euclidean edge length-weighted graph radius.

#     This function calculates the inverse Euclidean edge length-weighted
#     radius (minimum eccentricity) of a given graph. This function is
#     best applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         float: Inverse Euclidean edge length-weighted graph radius.
    
#     """
#     graph = shrt_path_l_inv_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return nx.radius(graph, weight="l_inv")

# def sigma_func(graph: nx.Graph | nx.MultiGraph) -> int:
#     """Geodesic graph diameter.

#     This function calculates the geodesic diameter (maximum
#     eccentricity) of a given graph, where each edge has a unit weight.
#     This function is best applied to fully connected graphs.

#     Args:
#         graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
#     Returns:
#         int: Geodesic graph diameter.
    
#     """
#     return nx.diameter(graph)

# def l_attr_sigma_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> float:
#     """Euclidean edge length-weighted graph diameter.

#     This function calculates the Euclidean edge length-weighted diameter
#     (maximum eccentricity) of a given graph. This function is best
#     applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         float: Euclidean edge length-weighted graph diameter.
    
#     """
#     graph = shrt_path_l_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return nx.diameter(graph, weight="l")

# def l_inv_attr_sigma_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> float:
#     """Inverse Euclidean edge length-weighted graph diameter.

#     This function calculates the inverse Euclidean edge length-weighted
#     diameter (maximum eccentricity) of a given graph. This function is
#     best applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         float: Inverse Euclidean edge length-weighted graph diameter.
    
#     """
#     graph = shrt_path_l_inv_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return nx.diameter(graph, weight="l_inv")

# def epsilon_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
#     """Geodesic graph eccentricity.

#     This function calculates the geodesic eccentricity (the maximum
#     shortest path) for each node in an (undirected) graph, where each
#     edge has a unit weight. This function is best applied to fully
#     connected graphs.

#     Args:
#         graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
#     Returns:
#         np.ndarray: Nodewise geodesic graph eccentricity.
    
#     """
#     return np.asarray(list(nx.eccentricity(graph).values()), dtype=int)

# def l_attr_epsilon_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Euclidean edge length-weighted graph eccentricity.

#     This function calculates the Euclidean edge length-weighted
#     eccentricity (the maximum shortest path) for each node in an
#     (undirected) graph. This function is best applied to fully connected
#     graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Euclidean edge length-weighted nodewise graph
#         eccentricity.
    
#     """
#     graph = shrt_path_l_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return np.asarray(list(nx.eccentricity(graph, weight="l").values()))

# def l_inv_attr_epsilon_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Inverse Euclidean edge length-weighted graph eccentricity.

#     This function calculates the inverse Euclidean edge length-weighted
#     eccentricity (the maximum shortest path) for each node in an
#     (undirected) graph. This function is best applied to fully connected
#     graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Inverse Euclidean edge length-weighted nodewise
#         graph eccentricity.
    
#     """
#     graph = shrt_path_l_inv_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return np.asarray(list(nx.eccentricity(graph, weight="l_inv").values()))

# def d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
#     """Geodesic shortest path length.

#     This function calculates the geodesic shortest path length for all
#     pairs of nodes in an (undirected) graph (excluding all self-loop
#     node pairs), where each edge has a unit weight. This function is
#     best applied to fully connected graphs.

#     Args:
#         graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
#     Returns:
#         np.ndarray: Node-pairwise geodesic shortest path length,
#         excluding all self-loop node pairs. Thus, for a network with n
#         nodes labeled {0, 1, ..., n-1}, the first n-1 entries are
#         associated with the shortest path length for all non-self-loop
#         node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
#         next n-1 entries are associated with that for the first node
#         (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
#     """
#     d_dict = dict(nx.shortest_path_length(graph))
#     node_list = list(graph.nodes())
#     n = n_func(graph)
#     n_pairs = n * (n-1)
#     d = np.empty(n_pairs, dtype=int)

#     indx = 0
#     for node_0 in node_list:
#         for node_1 in node_list:
#             if node_0 == node_1: continue
#             else:
#                 d[indx] = d_dict[node_0][node_1]
#                 indx += 1
    
#     return d

# def l_attr_d_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Euclidean edge length-weighted shortest path length.

#     This function calculates the Euclidean edge length-weighted shortest
#     path length for all pairs of nodes in an (undirected) graph
#     (excluding all self-loop node pairs). This function is best applied
#     to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Node-pairwise Euclidean edge length-weighted
#         shortest path length, excluding all self-loop node pairs. Thus,
#         for a network with n nodes labeled {0, 1, ..., n-1}, the first
#         n-1 entries are associated with the shortest path length for all
#         non-self-loop node pairs for the zeroth node (0-1, 0-2, ...,
#         0-(n-1)). The next n-1 entries are associated with that for the
#         first node (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
#     """
#     graph = shrt_path_l_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     d_dict = dict(nx.shortest_path_length(graph, weight="l"))
#     node_list = list(graph.nodes())
#     n = n_func(graph)
#     n_pairs = n * (n-1)
#     d = np.empty(n_pairs)

#     indx = 0
#     for node_0 in node_list:
#         for node_1 in node_list:
#             if node_0 == node_1: continue
#             else:
#                 d[indx] = d_dict[node_0][node_1]
#                 indx += 1
    
#     return d

# def l_inv_attr_d_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Inverse Euclidean edge length-weighted shortest path length.

#     This function calculates the inverse Euclidean edge length-weighted
#     shortest path length for all pairs of nodes in an (undirected) graph
#     (excluding all self-loop node pairs). This function is best applied
#     to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Node-pairwise inverse Euclidean edge length-weighted
#         shortest path length, excluding all self-loop node pairs. Thus,
#         for a network with n nodes labeled {0, 1, ..., n-1}, the first
#         n-1 entries are associated with the shortest path length for all
#         non-self-loop node pairs for the zeroth node (0-1, 0-2, ...,
#         0-(n-1)). The next n-1 entries are associated with that for the
#         first node (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
#     """
#     graph = shrt_path_l_inv_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     d_dict = dict(nx.shortest_path_length(graph, weight="l_inv"))
#     node_list = list(graph.nodes())
#     n = n_func(graph)
#     n_pairs = n * (n-1)
#     d = np.empty(n_pairs)

#     indx = 0
#     for node_0 in node_list:
#         for node_1 in node_list:
#             if node_0 == node_1: continue
#             else:
#                 d[indx] = d_dict[node_0][node_1]
#                 indx += 1
    
#     return d

# def avrg_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
#     """Average geodesic shortest path length.

#     This function calculates the average geodesic shortest path length
#     for each node in an (undirected) graph, where each edge has a unit
#     weight. This function is best applied to fully connected graphs.

#     Args:
#         graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
#     Returns:
#         np.ndarray: Nodewise average geodesic shortest path length.
    
#     """
#     d_dict = dict(nx.shortest_path_length(graph))
#     n = n_func(graph)
#     avrg_d = np.empty(n)

#     indx = 0
#     for node in d_dict:
#         avrg_d[indx] = sum(d_dict[node].values()) / (n-1)
#         indx += 1
    
#     return avrg_d

# def avrg_l_attr_d_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Average Euclidean edge length-weighted shortest path length.

#     This function calculates the average Euclidean edge length-weighted
#     shortest path length for each node in an (undirected) graph. This
#     function is best applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Nodewise average Euclidean edge length-weighted
#         shortest path length.
    
#     """
#     graph = shrt_path_l_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     d_dict = dict(nx.shortest_path_length(graph, weight="l"))
#     n = n_func(graph)
#     avrg_d = np.empty(n)

#     indx = 0
#     for node in d_dict:
#         avrg_d[indx] = sum(d_dict[node].values()) / (n-1)
#         indx += 1
    
#     return avrg_d

# def avrg_l_inv_attr_d_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Average inverse Euclidean edge length-weighted shortest path
#     length.

#     This function calculates the average inverse Euclidean edge
#     length-weighted shortest path length for each node in an
#     (undirected) graph. This function is best applied to fully connected
#     graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Nodewise average inverse Euclidean edge
#         length-weighted shortest path length.
    
#     """
#     graph = shrt_path_l_inv_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     d_dict = dict(nx.shortest_path_length(graph, weight="l_inv"))
#     n = n_func(graph)
#     avrg_d = np.empty(n)

#     indx = 0
#     for node in d_dict:
#         avrg_d[indx] = sum(d_dict[node].values()) / (n-1)
#         indx += 1
    
#     return avrg_d

# def e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
#     """Geodesic graph efficiency.

#     This function calculates the geodesic efficiency for all pairs of
#     nodes in an (undirected) graph (excluding all self-loop node pairs),
#     where each edge has a unit weight. This function is best applied to
#     fully connected graphs.

#     Args:
#         graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
#     Returns:
#         np.ndarray: Node-pairwise geodesic graph efficiency, excluding
#         all self-loop node pairs. Thus, for a network with n nodes
#         labeled {0, 1, ..., n-1}, the first n-1 entries are associated
#         with the efficiency for all non-self-loop node pairs for the
#         zeroth node (0-1, 0-2, ..., 0-(n-1)). The next n-1 entries are
#         associated with that for the first node (1-0, 1-2, 1-3, ...,
#         1-(n-1)), and so on.
    
#     """
#     d = d_func(graph)
#     d = d.astype(float)
#     return np.reciprocal(d, where=d!=0.0)

# def l_attr_e_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Euclidean edge length-weighted graph efficiency.

#     This function calculates the Euclidean edge length-weighted
#     efficiency for all pairs of nodes in an (undirected) graph
#     (excluding all self-loop node pairs). This function is best applied
#     to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Node-pairwise Euclidean edge length-weighted
#         efficiency, excluding all self-loop node pairs. Thus, for a
#         network with n nodes labeled {0, 1, ..., n-1}, the first n-1
#         entries are associated with the efficiency for all non-self-loop
#         node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
#         next n-1 entries are associated with that for the first node
#         (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
#     """
#     d = l_attr_d_func(conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return np.reciprocal(d, where=d!=0.0)

# def l_inv_attr_e_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Inverse Euclidean edge length-weighted graph efficiency.

#     This function calculates the inverse Euclidean edge length-weighted
#     efficiency for all pairs of nodes in an (undirected) graph
#     (excluding all self-loop node pairs). This function is best applied
#     to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Node-pairwise inverse Euclidean edge length-weighted
#         efficiency, excluding all self-loop node pairs. Thus, for a
#         network with n nodes labeled {0, 1, ..., n-1}, the first n-1
#         entries are associated with the efficiency for all non-self-loop
#         node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
#         next n-1 entries are associated with that for the first node
#         (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
#     """
#     d = l_inv_attr_d_func(conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return np.reciprocal(d, where=d!=0.0)

# def avrg_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
#     """Average geodesic graph efficiency.

#     This function calculates the average geodesic efficiency for each
#     node in an (undirected) graph, where each edge has a unit weight.
#     This function is best applied to fully connected graphs.

#     Args:
#         graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
#     Returns:
#         np.ndarray: Nodewise average geodesic graph efficiency.
    
#     """
#     d_dict = dict(nx.shortest_path_length(graph))
#     node_list = list(graph.nodes())
#     n = n_func(graph)
#     avrg_e = np.empty(n)

#     indx = 0
#     for node_0 in node_list:
#         avrg_e_sum = 0.
#         for node_1 in node_list:
#             d = d_dict[node_0][node_1] * 1.0
#             avrg_e_sum += np.reciprocal(d, where=d!=0.0)
#         avrg_e[indx] = avrg_e_sum / (n-1)
#         indx += 1
    
#     return avrg_e

# def avrg_l_attr_e_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Average Euclidean edge length-weighted graph efficiency.

#     This function calculates the average Euclidean edge length-weighted
#     efficiency for each node in an (undirected) graph, where each edge
#     has a unit weight. This function is best applied to fully connected
#     graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Nodewise average Euclidean edge length-weighted
#         graph efficiency.
    
#     """
#     graph = shrt_path_l_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     d_dict = dict(nx.shortest_path_length(graph, weight="l"))
#     node_list = list(graph.nodes())
#     n = n_func(graph)
#     avrg_e = np.empty(n)

#     indx = 0
#     for node_0 in node_list:
#         avrg_e_sum = 0.
#         for node_1 in node_list:
#             d = d_dict[node_0][node_1] * 1.0
#             avrg_e_sum += np.reciprocal(d, where=d!=0.0)
#         avrg_e[indx] = avrg_e_sum / (n-1)
#         indx += 1
    
#     return avrg_e

# def avrg_l_inv_attr_e_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Average inverse Euclidean edge length-weighted graph efficiency.

#     This function calculates the average inverse Euclidean edge
#     length-weighted efficiency for each node in an (undirected) graph,
#     where each edge has a unit weight. This function is best applied to
#     fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Nodewise average inverse Euclidean edge
#         length-weighted graph efficiency.
    
#     """
#     graph = shrt_path_l_inv_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     d_dict = dict(nx.shortest_path_length(graph, weight="l_inv"))
#     node_list = list(graph.nodes())
#     n = n_func(graph)
#     avrg_e = np.empty(n)

#     indx = 0
#     for node_0 in node_list:
#         avrg_e_sum = 0.
#         for node_1 in node_list:
#             d = d_dict[node_0][node_1] * 1.0
#             avrg_e_sum += np.reciprocal(d, where=d!=0.0)
#         avrg_e[indx] = avrg_e_sum / (n-1)
#         indx += 1
    
#     return avrg_e

# def bc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
#     """Geodesic node betweenness centrality.

#     This function calculates the geodesic betweenness centrality for
#     each node in an (undirected) graph, where each edge has a unit
#     weight. This function is best applied to fully connected graphs.

#     Args:
#         graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
#     Returns:
#         np.ndarray: Nodewise geodesic betweenness centrality.
    
#     """
#     return np.asarray(list(nx.betweenness_centrality(graph).values()))

# def l_attr_bc_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Euclidean edge length-weighted node betweenness centrality.

#     This function calculates the Euclidean edge length-weighted
#     betweenness centrality for each node in an (undirected) graph. This
#     function is best applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Nodewise Euclidean edge length-weighted betweenness
#         centrality.
    
#     """
#     graph = shrt_path_l_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return (
#         np.asarray(list(nx.betweenness_centrality(graph, weight="l").values()))
#     )

# def l_inv_attr_bc_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Inverse Euclidean edge length-weighted node betweenness
#     centrality.

#     This function calculates the inverse Euclidean edge length-weighted
#     betweenness centrality for each node in an (undirected) graph. This
#     function is best applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Nodewise inverse Euclidean edge length-weighted
#         betweenness centrality.
    
#     """
#     graph = shrt_path_l_inv_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return (
#         np.asarray(
#             list(nx.betweenness_centrality(graph, weight="l_inv").values()))
#     )

# def ebc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
#     """Geodesic edge betweenness centrality.

#     This function calculates the geodesic edge betweenness
#     centrality for each edge in an (undirected) graph, where each edge
#     has a unit weight. This function is best applied to fully connected
#     graphs.

#     Args:
#         graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
#     Returns:
#         np.ndarray: Geodesic edgewise edge betweenness centrality.
    
#     """
#     return np.asarray(list(nx.edge_betweenness_centrality(graph).values()))

# def l_attr_ebc_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Euclidean edge length-weighted edge betweenness centrality.

#     This function calculates the Euclidean edge length-weighted edge
#     betweenness centrality for each edge in an (undirected) graph. This
#     function is best applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Edgewise Euclidean edge length-weighted edge
#         betweenness centrality.
    
#     """
#     graph = shrt_path_l_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return (
#         np.asarray(
#             list(nx.edge_betweenness_centrality(graph, weight="l").values()))
#     )

# def l_inv_attr_ebc_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Inverse Euclidean edge length-weighted edge betweenness
#     centrality.

#     This function calculates the inverse Euclidean edge length-weighted
#     edge betweenness centrality for each edge in an (undirected) graph.
#     This function is best applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Edgewise inverse Euclidean edge length-weighted edge
#         betweenness centrality.
    
#     """
#     graph = shrt_path_l_inv_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return (
#         np.asarray(
#             list(nx.edge_betweenness_centrality(graph, weight="l_inv").values()))
#     )

# def cc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
#     """Geodesic closeness centrality.

#     This function calculates the geodesic closeness centrality for each
#     node in an (undirected) graph, where each edge has a unit weight.
#     This function is best applied to fully connected graphs.

#     Args:
#         graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
#     Returns:
#         np.ndarray: Nodewise geodesic closeness centrality.
    
#     """
#     return np.asarray(list(nx.closeness_centrality(graph).values()))

# def l_attr_cc_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Euclidean edge length-weighted closeness centrality.

#     This function calculates the Euclidean edge length-weighted
#     closeness centrality for each node in an (undirected) graph. This
#     function is best applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Nodewise Euclidean edge length-weighted closeness
#         centrality.
    
#     """
#     graph = shrt_path_l_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return (
#         np.asarray(list(nx.closeness_centrality(graph, distance="l").values()))
#     )

# def l_inv_attr_cc_func(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Inverse Euclidean edge length-weighted closeness centrality.

#     This function calculates the inverse Euclidean edge length-weighted
#     closeness centrality for each node in an (undirected) graph. This
#     function is best applied to fully connected graphs.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes. Ideally, this graph ought to be fully connected as-is.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Nodewise inverse Euclidean edge length-weighted
#         closeness centrality.
    
#     """
#     graph = shrt_path_l_inv_attr_graph(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
#     return (
#         np.asarray(
#             list(nx.closeness_centrality(graph, distance="l_inv").values()))
#     )