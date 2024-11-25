import numpy as np
import networkx as nx
from network_topology_initialization_utils import (
    tessellation_protocol,
    tessellation
)
from scipy.special import comb
from graph_utils import (
    elastically_effective_graph,
    elastically_effective_end_linked_graph
)

def proportion_elastically_effective_nodes(graph) -> float:
    """Proportion of elastically-effective nodes in a given graph.

    This function calculates and returns the proportion of
    elastically-effective nodes in a given graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        float: (Undirected) Proportion of elastically-effective nodes.
    
    """
    # Number of nodes involved in the given graph
    graph_n = np.shape(np.unique(np.asarray(list(graph.edges()), dtype=int)))[0]
    # Elastically-effective graph
    ee_graph = elastically_effective_graph(graph)
    # Number of nodes involved in the elastically-effective graph
    ee_graph_n = (
        np.shape(np.unique(np.asarray(list(ee_graph.edges()), dtype=int)))[0]
    )

    return ee_graph_n / graph_n

def proportion_elastically_effective_end_linked_nodes(graph) -> float:
    """Proportion of elastically-effective end-linked nodes in a given
    graph.

    This function calculates and returns the proportion of
    elastically-effective end-linked nodes in a given graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        float: (Undirected) Proportion of elastically-effective end-linked nodes.
    
    """
    # Number of nodes involved in the given graph
    graph_n = np.shape(np.unique(np.asarray(list(graph.edges()), dtype=int)))[0]
    # Elastically-effective end-linked graph
    eeel_graph = elastically_effective_end_linked_graph(graph)
    # Number of nodes involved in the elastically-effective end-linked
    # graph
    eeel_graph_n = (
        np.shape(np.unique(np.asarray(list(eeel_graph.edges()), dtype=int)))[0]
    )

    return eeel_graph_n / graph_n

def proportion_elastically_effective_edges(graph):
    """Proportion of elastically-effective edges in a given graph.

    This function calculates and returns the proportion of
    elastically-effective edges in a given graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        float: (Undirected) Proportion of elastically-effective edges.
    
    """
    # Number of edges in given graph
    graph_m = len(list(graph.edges()))
    # Elastically-effective graph
    ee_graph = elastically_effective_graph(graph)
    # Number of edges in elastically-effective graph
    ee_graph_m = len(list(ee_graph.edges()))

    return ee_graph_m / graph_m

def proportion_elastically_effective_end_linked_edges(graph):
    """Proportion of elastically-effective end-linked edges in a given
    graph.

    This function calculates and returns the proportion of
    elastically-effective end-linked edges in a given graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
    
    Returns:
        float: (Undirected) Proportion of elastically-effective end-linked edges.
    
    """
    # Number of edges in given graph
    graph_m = len(list(graph.edges()))
    # Elastically-effective end-linked graph
    eeel_graph = elastically_effective_graph(graph)
    # Number of edges in elastically-effective end-linked graph
    eeel_graph_m = len(list(eeel_graph.edges()))

    return eeel_graph_m / graph_m

# def h_counts_calculation(graph, l_bound: int) -> np.ndarray:
#     """Chordless cycle counts. -> RING COUNTS

#     This function calculates the chordless cycle counts in an
#     (undirected) graph, where the chordless cycle order can be between 1
#     and a maximum value, inclusive.

#     Args:
#         graph: (Undirected) NetworkX graph that can be of type nx.Graph or nx.MultiGraph.
#         l_bound (int): Maximal chordless cycle order.
    
#     Returns:
#         np.ndarray: Chordless cycle counts, where the chordless cycle order can be between 1 and l_bound, inclusive.
    
#     """
#     # Initialize the chordless cycle counts, where h\in[1, l_bound].
#     # Thus, h_counts[h-1] = counts, i.e., h_counts[0] = number of
#     # self-loops, h_counts[1] = number of second-order cycles induced by
#     # redundant multi-edges, h_counts[2] = number of third-order cycles, 
#     # h_counts[3] = number of fourth-order cycles, ...,
#     # h_counts[l_bound-1] = number of l_bound-order cycles.
#     h_counts = np.zeros(l_bound, dtype=int)

#     # Calculate and store the number of self-loops
#     self_loop_num = int(nx.number_of_selfloops(graph))
#     h_counts[0] = self_loop_num

#     # Self-loop pruning procedure
#     if self_loop_num > 0:
#         graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    
#     # If the graph is of type nx.MultiGraph, then address multi-edges,
#     # prune redundant edges, and convert resulting graph to type
#     # nx.Graph
#     if graph.is_multigraph():    
#         # Gather edges and edges counts
#         graph_edges, graph_edges_counts = (
#             np.unique(
#                 np.sort(np.asarray(list(graph.edges()), dtype=int), axis=1),
#                 return_counts=True, axis=0)
#         )
        
#         # Address multi-edges by calculating and storing the number of
#         # second-order cycles and by pruning redundant edges
#         if np.any(graph_edges_counts > 1):
#             # Extract multi-edges
#             multiedges = np.where(graph_edges_counts > 1)[0]
#             for multiedge in np.nditer(multiedges):
#                 multiedge = int(multiedge)
#                 # Number of edges in the multiedge
#                 edge_num = graph_edges_counts[multiedge]
#                 # Calculate the number of second-order cycles induced by
#                 # redundant multi-edges
#                 h_counts[1] += int(comb(edge_num, 2))
#                 # Remove redundant edges in the multiedge (thereby
#                 # leaving one edge)
#                 graph.remove_edges_from(
#                     list((int(graph_edges[multiedge, 0]), int(graph_edges[multiedge, 1])) for _ in range(edge_num-1)))
        
#         # Convert graph to type nx.Graph
#         graph = nx.Graph(graph)

#     # Degree of nodes
#     graph_nodes_k = np.asarray(list(graph.degree()), dtype=int)
#     graph_k = graph_nodes_k[:, 1]

#     # Dangling edge pruning procedure
#     while np.any(graph_k == 1):
#         # Extract dangling nodes (with k = 1)
#         dangling_node_indcs = np.where(graph_k == 1)[0]
#         for dangling_node_indx in np.nditer(dangling_node_indcs):
#             dangling_node_indx = int(dangling_node_indx)
#             # Dangling node (with k = 1)
#             dangling_node = int(graph_nodes_k[dangling_node_indx, 0])
#             # Neighbor of dangling node (with k = 1)
#             dangling_node_nghbr_arr = np.asarray(
#                 list(graph.neighbors(dangling_node)), dtype=int)
#             # Check to see if the dangling edge was previously removed
#             if np.shape(dangling_node_nghbr_arr)[0] == 0: continue
#             else:
#                 # Remove dangling edge
#                 graph.remove_edge(dangling_node, int(dangling_node_nghbr_arr[0]))
#         # Update degree of nodes
#         graph_nodes_k = np.asarray(list(graph.degree()), dtype=int)
#         graph_k = graph_nodes_k[:, 1]
    
#     # Remove isolate nodes
#     if nx.number_of_isolates(graph) > 0:
#         graph.remove_nodes_from(list(nx.isolates(graph)))
    
#     # Find chordless cycles
#     chrdls_cycls = list(nx.chordless_cycles(graph, length_bound=l_bound))
    
#     # Calculate number of occurrences for each chordless cycle order
#     graph_h, graph_h_counts = np.unique(
#         np.asarray(list(len(cycl) for cycl in chrdls_cycls), dtype=int),
#         return_counts=True)

#     # Store the chordless cycle counts
#     for h_indx in range(np.shape(graph_h)[0]):
#         if graph_h[h_indx] == 0: continue
#         else: h_counts[graph_h[h_indx]-1] = graph_h_counts[h_indx]
    
#     return h_counts

def core_pb_edge_identification(
        core_node_0_coords: np.ndarray,
        core_node_1_coords: np.ndarray,
        L: float) -> tuple[np.ndarray, float]:
    """Periodic boundary edge and node identification.

    This function uses the minimum image criterion to determine/identify
    the node coordinates of a particular periodic boundary edge.

    Args:
        core_node_0_coords (np.ndarray): Coordinates of the core node in the periodic boundary edge.
        core_node_1_coords (np.ndarray): Coordinates of the core node that translates/tessellates to the periodic node in the periodic boundary edge.
        L (float): Tessellation scaling distance (i.e., simulation box size).

    Returns:
        tuple[np.ndarray, float]: Coordinates of the periodic node in
        the periodic boundary edge, and the length of the periodic
        boundary edge, respectively.
    
    """
    # Confirm that coordinate dimensions match
    if np.shape(core_node_0_coords)[0] != np.shape(core_node_1_coords)[0]:
        import sys

        error_str = (
            "The dimensionality of the core node coordinates at hand "
            + "in the periodic boundary edge and node identification "
            + "do not match." 
        )
        sys.exit(error_str)
    
    # Calculate network dimension
    dim = np.shape(core_node_0_coords)[0]

    # Tessellation protocol
    tsslltn, tsslltn_num = tessellation_protocol(dim)
    
    # Use tessellation protocol to tessellate core_node_1
    core_node_1_tsslltn_coords = tessellation(core_node_1_coords, tsslltn, L)
    
    # Use minimum image/distance criterion to select the correct
    # periodic boundary node and edge corresponding to core_node_1
    l_pb_nodes_1 = np.empty(tsslltn_num)
    for pb_node_1 in range(tsslltn_num):
        l_pb_nodes_1[pb_node_1] = np.linalg.norm(
            core_node_1_tsslltn_coords[pb_node_1]-core_node_0_coords)
    pb_node_1 = np.argmin(l_pb_nodes_1)
    
    return core_node_1_tsslltn_coords[pb_node_1], l_pb_nodes_1[pb_node_1]

def l_edges_calculation(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Edge length calculation.

    This function calculates the length of each core and periodic
    boundary edge. Note that the length of each edge is calculated as
    the true spatial length, not the naive length present in the graph.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Edge length.
    
    """
    # Gather edges and initialize edge length np.ndarray
    conn_edges = list(conn_graph.edges())
    conn_m = len(conn_edges)
    l_edges = np.empty(conn_m)

    # Calculate and store the length of each edge
    for edge_indx, edge in enumerate(conn_edges):
        # Node numbers
        core_node_0 = int(edge[0])
        core_node_1 = int(edge[1])

        # Coordinates of each node
        core_node_0_coords = coords[core_node_0]
        core_node_1_coords = coords[core_node_1]

        # Edge is a core edge
        if conn_core_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store core edge length
            l_edges[edge_indx] = np.linalg.norm(
                core_node_1_coords-core_node_0_coords)
        # Edge is a periodic boundary edge
        elif conn_pb_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store periodic boundary edge length
            _, l_pb_edge = core_pb_edge_identification(
                core_node_0_coords, core_node_1_coords, L)
            l_edges[edge_indx] = l_pb_edge
        else:
            import sys
            
            error_str = (
                "The edge in the overall graph was not detected in "
                + "either the core edge graph or the periodic boundary "
                + "edge graph."
            )
            sys.exit(error_str)
        
    return l_edges

# def l_nrmlzd_edges_calculation(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Normalized edge length calculation.

#     This function calculates the normalized length of each core and
#     periodic boundary edge. Note that the length of each edge is
#     calculated as the true spatial length, not the naive length present
#     in the graph.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Normalized edge length.
    
#     """
#     # Edge length
#     l_edges = l_edges_calculation(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    
#     # Edge length normalization by L
#     return l_edges / L

def l_cmpnts_edges_calculation(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Edge length component calculation.

    This function calculates the length components of each core and
    periodic boundary edge. Note that the length components of each edge
    are calculated as the true spatial length components, not the naive
    length components present in the graph.

    Args:
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        np.ndarray: Edge length components.
    
    """
    # Gather edges and initialize edge length components np.ndarray
    conn_edges = list(conn_graph.edges())
    conn_m = len(conn_edges)
    l_cmpnt_edges = np.empty((conn_m, np.shape(coords)[1]))

    # Calculate and store the length component of each edge
    for edge_indx, edge in enumerate(conn_edges):
        # Node numbers
        core_node_0 = int(edge[0])
        core_node_1 = int(edge[1])

        # Coordinates of each node
        core_node_0_coords = coords[core_node_0]
        core_node_1_coords = coords[core_node_1]

        # Edge is a core edge
        if conn_core_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store core edge length components
            l_cmpnt_edges[edge_indx] = core_node_1_coords - core_node_0_coords
        # Edge is a periodic boundary edge
        elif conn_pb_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store periodic boundary edge length
            pb_node_1_coords, _ = core_pb_edge_identification(
                core_node_0_coords, core_node_1_coords, L)
            l_cmpnt_edges[edge_indx] = pb_node_1_coords - core_node_0_coords
        else:
            import sys
            
            error_str = (
                "The edge in the overall graph was not detected in "
                + "either the core edge graph or the periodic boundary "
                + "edge graph."
            )
            sys.exit(error_str)
        
    return l_cmpnt_edges

# def l_cmpnts_nrmlzd_edges_calculation(
#         conn_core_graph: nx.Graph | nx.MultiGraph,
#         conn_pb_graph: nx.Graph | nx.MultiGraph,
#         conn_graph: nx.Graph | nx.MultiGraph,
#         coords: np.ndarray,
#         L: float) -> np.ndarray:
#     """Normalized edge length component calculation.

#     This function calculates the normalized length components of each
#     core and periodic boundary edge. Note that the length components of
#     each edge are calculated as the true spatial length components, not
#     the naive length components present in the graph.

#     Args:
#         conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes.
#         conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
#         conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
#         coords (np.ndarray): Coordinates of the core nodes.
#         L (float): Tessellation scaling distance (i.e., simulation box size).
    
#     Returns:
#         np.ndarray: Normalized edge length components.
    
#     """
#     # Edge length
#     l_cmpnts_edges = l_cmpnts_edges_calculation(
#         conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    
#     # Edge length components normalization by L
#     return l_cmpnts_edges / L
