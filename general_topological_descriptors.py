import numpy as np
import networkx as nx
from network_topology_initialization_utils import (
    tessellation_protocol,
    tessellation
)
from scipy.special import comb
from graph_utils import (
    elastically_effective_graph,
    elastically_effective_end_linked_graph,
    edge_id
)

def core_pb_edge_id(
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
        error_str = (
            "The dimensionality of the core node coordinates at hand "
            + "in the periodic boundary edge and node identification "
            + "do not match." 
        )
        print(error_str)
        return None
    
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

def l_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Edge lengths.

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
    l = np.empty(len(conn_edges))

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
            l[edge_indx] = np.linalg.norm(core_node_1_coords-core_node_0_coords)
        # Edge is a periodic boundary edge
        elif conn_pb_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store periodic boundary edge length
            _, l[edge_indx] = core_pb_edge_id(
                core_node_0_coords, core_node_1_coords, L)
        else:
            error_str = (
                "The edge in the overall graph was not detected in "
                + "either the core edge graph or the periodic boundary "
                + "edge graph."
            )
            print(error_str)
            return None
        
    return l

def l_cmpnts_func(
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Edge length components.

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
    l_cmpnt = np.empty((len(conn_edges), np.shape(coords)[1]))

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
            l_cmpnt[edge_indx] = core_node_1_coords - core_node_0_coords
        # Edge is a periodic boundary edge
        elif conn_pb_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store periodic boundary edge length
            pb_node_1_coords, _ = core_pb_edge_id(
                core_node_0_coords, core_node_1_coords, L)
            l_cmpnt[edge_indx] = pb_node_1_coords - core_node_0_coords
        else:
            error_str = (
                "The edge in the overall graph was not detected in "
                + "either the core edge graph or the periodic boundary "
                + "edge graph."
            )
            print(error_str)
            return None
        
    return l_cmpnt

def l_arr_func(
        conn_core_edges: np.ndarray,
        conn_pb_edges: np.ndarray,
        coords: np.ndarray,
        L: float) -> tuple[np.ndarray, np.ndarray]:
    """Array-wise edge lengths.

    This function calculates the length of each core and periodic
    boundary edge supplied in np.ndarrays. Note that the length of each
    edge is calculated as the true spatial length, not the naive length
    present in the graph. Also note that each edge length entry
    corresponds to the respective edge supplied in the input edge
    arrays.

    Args:
        conn_core_graph (np.ndarray): Core edges from the graph capturing the periodic connections between the core nodes.
        conn_pb_graph (np.ndarray): Periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Length of each core and periodic
        boundary edge, respectively. Each edge length entry corresponds
        to the respective edge supplied in the input edge arrays.
    
    """
    # Initialize core edge length np.ndarray
    conn_core_m = np.shape(conn_core_edges)[0]
    l_core_edges = np.empty(conn_core_m)

    # Calculate and store the length of each core edge
    for edge in range(conn_core_m):
        # Coordinates of each node
        core_node_0_coords = coords[int(conn_core_edges[edge, 0])]
        core_node_1_coords = coords[int(conn_core_edges[edge, 1])]

        # Core edge length
        l_core_edges[edge] = np.linalg.norm(
            core_node_1_coords-core_node_0_coords)
    
    # Initialize periodic boundary edge length np.ndarray
    conn_pb_m = np.shape(conn_pb_edges)[0]
    l_pb_edges = np.empty(conn_pb_m)

    # Calculate and store the length of each periodic boundary edge
    for edge in range(conn_pb_m):
        # Coordinates of each node
        core_node_0_coords = coords[int(conn_pb_edges[edge, 0])]
        core_node_1_coords = coords[int(conn_pb_edges[edge, 1])]

        # Periodic boundary edge length
        _, l_pb_edges[edge] = core_pb_edge_id(
            core_node_0_coords, core_node_1_coords, L)
        
    return l_core_edges, l_pb_edges

def l_cmpnts_arr_func(
        conn_core_edges: np.ndarray,
        conn_pb_edges: np.ndarray,
        coords: np.ndarray,
        L: float) -> tuple[np.ndarray, np.ndarray]:
    """Array-wise edge length components.

    This function calculates the length components of each core and
    periodic boundary edge supplied in np.ndarrays. Note that the length
    components of each edge are calculated as the true spatial length
    components, not the naive length components present in the graph.
    Also note that each edge length component entry corresponds to the
    respective edge supplied in the input edge arrays.

    Args:
        conn_core_graph (np.ndarray): Core edges from the graph capturing the periodic connections between the core nodes.
        conn_pb_graph (np.ndarray): Periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box size).
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Edge length components of each
        core and periodic boundary edge, respectively. Each edge length
        component entry corresponds to the respective edge supplied in
        the input edge arrays.
    
    """
    # Initialize core edge length components np.ndarray
    conn_core_m = np.shape(conn_core_edges)[0]
    l_cmpnt_core_edges = np.empty((conn_core_m, np.shape(coords)[1]))

    # Calculate and store the length components of each core edge
    for edge in range(conn_core_m):
        # Coordinates of each node
        core_node_0_coords = coords[int(conn_core_edges[edge, 0])]
        core_node_1_coords = coords[int(conn_core_edges[edge, 1])]

        # Core edge length components
        l_cmpnt_core_edges[edge] = core_node_1_coords - core_node_0_coords
    
    # Initialize periodic boundary edge length components np.ndarray
    conn_pb_m = np.shape(conn_pb_edges)[0]
    l_cmpnt_pb_edges = np.empty((conn_pb_m, np.shape(coords)[1]))

    # Calculate and store the length components of each periodic
    # boundary edge
    for edge in range(conn_pb_m):
        # Coordinates of each node
        core_node_0_coords = coords[int(conn_pb_edges[edge, 0])]
        core_node_1_coords = coords[int(conn_pb_edges[edge, 1])]

        # Periodic boundary edge length components
        pb_node_1_coords, _ = core_pb_edge_id(
            core_node_0_coords, core_node_1_coords, L)
        l_cmpnt_pb_edges[edge] = pb_node_1_coords - core_node_0_coords
    
    return l_cmpnt_core_edges, l_cmpnt_pb_edges

def n_func(graph: nx.Graph | nx.MultiGraph) -> int:
    """Number of nodes.

    This function calculates the number of nodes in a given graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        int: Number of nodes.
    
    """
    return np.shape(np.unique(np.asarray(list(graph.edges()), dtype=int)))[0]

def m_func(graph: nx.Graph | nx.MultiGraph) -> int:
    """Number of edges.

    This function calculates the number of edges in a given graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        int: Number of edges.
    
    """
    return len(list(graph.edges()))

def prop_ee_n_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Proportion of elastically-effective nodes.

    This function calculates the proportion of elastically-effective
    nodes in a given graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        float: Proportion of elastically-effective nodes.
    
    """
    return (
        n_func(elastically_effective_graph(graph)) / n_func(graph)
    )

def prop_eeel_n_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Proportion of elastically-effective end-linked nodes.

    This function calculates the proportion of elastically-effective
    end-linked nodes in a given graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        float: Proportion of elastically-effective end-linked nodes.
    
    """
    return (
        n_func(elastically_effective_end_linked_graph(graph)) / n_func(graph)
    )

def prop_ee_m_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Proportion of elastically-effective edges.

    This function calculates the proportion of elastically-effective
    edges in a given graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        float: Proportion of elastically-effective edges.
    
    """
    return (
        m_func(elastically_effective_graph(graph)) / m_func(graph)
    )

def prop_eeel_m_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Proportion of elastically-effective end-linked edges.

    This function calculates the proportion of elastically-effective
    end-linked edges in a given graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        float: Proportion of elastically-effective end-linked edges.
    
    """
    return (
        m_func(elastically_effective_end_linked_graph(graph)) / m_func(graph)
    )

def rho_graph_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Graph density.

    This function calculates the density of a given graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        float: Graph density.
    
    """
    return nx.density(graph)

def h_func(
        graph: nx.Graph | nx.MultiGraph,
        length_bound: int,
        coords: np.ndarray,
        L: float) -> np.ndarray:
    """Periodic boundary-compliant Franzblau shortest-path ring order.

    This function finds all the periodic boundary-compliant Franzblau
    shortest-path rings in a(n undirected) graph. Each of the rings are
    found up to an order inclusively up to a specified maximum value.
    The function uses the node coordinates (normalized by the simulation
    box size) in the graph to determine if a ring loops over the entire
    periodic simulation box. All rings that are found to loop over the
    entire periodic simulation box are omitted (since they are
    non-physical rings). Note that the execution runtime of this
    function for some graphs can be prohibitively expensive, so please
    use with caution.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        length_bound (int): Maximum ring order (inclusive).
        coords (np.ndarray): Node coordinates.
        L (float): Simulation box size.
    
    Returns:
        np.ndarray: Periodic boundary-compliant Franzblau shortest-path
        ring order of each such ring in the graph.
    
    """
    # Normalize coordinates by simulation box size
    nrmlzd_coords = coords / L
    
    # Calculate and store the number of self-loops
    self_loop_num = int(nx.number_of_selfloops(graph))
    h = np.ones(self_loop_num, dtype=int)
    
    # Remove self-loops
    if self_loop_num > 0:
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    
    # Gather edge counts
    _, graph_edges_counts = edge_id(graph)

    # Calculate and store the number of redundant multiedges as
    # second-order cycles
    if np.any(graph_edges_counts > 1):
        # Extract multiedges
        multiedges = np.where(graph_edges_counts > 1)[0]
        for multiedge in np.nditer(multiedges):
            multiedge = int(multiedge)
            # Number of edges in the multiedge
            edge_num = graph_edges_counts[multiedge]
            # Calculate the number of second-order cycles induced by
            # redundant multiedges
            h = np.concatenate(
                (h, np.repeat(2, int(comb(edge_num, 2)))), dtype=int)
    
    # Remove redundant edges
    if graph.is_multigraph(): graph = nx.Graph(graph)

    # Find all chordless cycles of cycle order inclusively up to
    # length_bound
    chrdlss_cycls = list(nx.chordless_cycles(graph, length_bound=length_bound))

    # Filter out the chordless cycles that periodically span the entire
    # simulation box (which are topologically present but not physically
    # relevant) in order to yield the periodic boundary-compliant
    # chordless cycles
    tol = 1e-10
    pb_cmplnt_chrdlss_cycls = []
    for chrdlss_cycl in chrdlss_cycls:
        cycl_l_cmpnts = np.zeros_like(nrmlzd_coords[chrdlss_cycl[0]])
        for node_0, node_1 in zip(chrdlss_cycl[-1:]+chrdlss_cycl[:-1], chrdlss_cycl):
            edge_l_cmpnts = nrmlzd_coords[node_1] - nrmlzd_coords[node_0]
            # Minimum image criterion in normalized coordinates
            edge_l_cmpnts -= np.floor(edge_l_cmpnts+0.5)
            cycl_l_cmpnts += edge_l_cmpnts
        # Reject chordless cycle that periodically spans the entire
        # simulation box
        if np.any(np.abs(cycl_l_cmpnts) > tol): continue
        else: pb_cmplnt_chrdlss_cycls.append(chrdlss_cycl)
    
    del chrdlss_cycls
    
    # Filter out remaining chordless cycles that do not meet the
    # Franzblau shortest-path ring criterion in order to yield the
    # Franzblau shortest-path rings
    F_sp_rings = []
    for chrdlss_cycl in pb_cmplnt_chrdlss_cycls:
        # Gather the cycle-wise path length and the graph-wise shortest
        # path length between each unique node pair in each chordless
        # cycle
        k = len(chrdlss_cycl)
        node_pairs_cycl_d = []
        node_pairs_d = []
        for indx_0 in range(k):
            for indx_1 in range(indx_0+1, k):
                forward_cycl_d = indx_1 - indx_0
                backward_cycl_d = k - forward_cycl_d
                node_pairs_cycl_d.append(min(forward_cycl_d, backward_cycl_d))
                node_pairs_d.append(
                    nx.shortest_path_length(
                        graph, source=chrdlss_cycl[indx_0], target=chrdlss_cycl[indx_1]))
        # Compare the cycle-wise path length and graph-wise shortest
        # path length for each unique node pair in each chordless cycle
        F_sp_ring_criterion = True
        for node_pair_cycl_d, node_pair_d in zip(node_pairs_cycl_d, node_pairs_d):
            if node_pair_d < node_pair_cycl_d:
                F_sp_ring_criterion = False
                break
        # Chordless cycles that have at least one unique node pair where
        # the graph-wise shortest path length (for the node pair) is
        # less than the corresponding cycle-wise path length are not
        # Franzblau shortest-path rings
        if F_sp_ring_criterion: F_sp_rings.append(chrdlss_cycl)
    
    del pb_cmplnt_chrdlss_cycls

    # Calculate and store the Franzblau shortest-path rings in the
    # graph. These are necessarily third-order and higher-order rings.
    h = np.concatenate(
        (h, np.asarray(list(len(ring) for ring in F_sp_rings), dtype=int)))
    
    return h
