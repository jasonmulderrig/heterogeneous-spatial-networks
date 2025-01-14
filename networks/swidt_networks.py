import numpy as np
import networkx as nx
from file_io.file_io import (
    L_filename_str,
    config_filename_str,
    config_pruning_filename_str
)
from helpers.simulation_box_utils import L_arg_eta_func
from networks.delaunay_networks import delaunay_network_topology_initialization
from topological_descriptors.nodal_degree_topological_descriptors import k_func
from helpers.graph_utils import (
    add_nodes_from_numpy_array,
    add_edges_from_numpy_array
)
from topological_descriptors.network_topological_descriptors import (
    network_topological_descriptor
)

def swidt_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        pruning: int) -> str:
    """Filename prefix associated with spider web-inspired
    Delaunay-triangulated network data files.

    This function returns the filename prefix associated with spider
    web-inspired Delaunay-triangulated network data files.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
    
    Returns:
        str: The filename prefix associated with spider web-inspired
        Delaunay-triangulated network data files.
    
    """
    # This filename prefix convention is only applicable for data files
    # associated with spider web-inspired Delaunay-triangulated
    # networks. Exit if a different type of network is passed.
    if network != "swidt":
        error_str = (
            "This filename prefix convention is only applicable for "
            + "data files associated with spider web-inspired "
            + "Delaunay-triangulated networks. This filename prefix "
            + "will only be supplied if network = ``swidt''."
        )
        print(error_str)
        return None
    return config_pruning_filename_str(
        network, date, batch, sample, config, pruning)

def swidt_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float) -> None:
    """Simulation box size for spider web-inspired Delaunay-triangulated
    networks.

    This function calculates the simulation box size for spider
    web-inspired Delaunay-triangulated networks.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended number of core nodes.
        eta_n (float): Node packing density.
    
    """
    # This calculation for L is only applicable for spider web-inspired
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "swidt":
        error_str = (
            "This calculation for L is only applicable for spider "
            + "web-inspired Delaunay-triangulated networks. This "
            + "calculation will only proceed if network = ``swidt''."
        )
        print(error_str)
        return None
    
    # Calculate and save L
    np.savetxt(
        L_filename_str(network, date, batch, sample),
        [L_arg_eta_func(dim, b, n, eta_n)])

def swidt_network_topology(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Spider web-inspired Delaunay-triangulated network topology.

    This function confirms that the network being called for is a spider
    web-inspired Delaunay-triangulated network. Then, the function calls
    the Delaunay-triangulated network initialization function to create
    the initial Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Network topology initialization procedure is only applicable for 
    # spider web-inspired Delaunay-triangulated networks. Exit if a
    # different type of network is passed.
    if network != "swidt":
        error_str = (
            "Network topology initialization procedure is only "
            + "applicable for spider web-inspired "
            + "Delaunay-triangulated networks. This procedure will "
            + "only proceed if network = ``swidt''."
        )
        print(error_str)
        return None
    delaunay_network_topology_initialization(
        network, date, batch, sample, scheme, dim, n, config)

def swidt_network_edge_pruning_procedure(
        network: str,
        date: str,
        batch: str,
        sample: int,
        n: int,
        k: int,
        config: int,
        pruning: int) -> None:
    """Edge pruning procedure for the initialized topology of spider
    web-inspired Delaunay-triangulated networks.

    This function loads fundamental graph constituents along with core
    node coordinates, performs a random edge pruning procedure such that
    each node in the network is connected to, at most, k edges, and
    isolates the maximum connected component from the resulting network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        n (int): Number of core nodes.
        k (int): Maximum node degree/functionality; either 3, 4, 5, 6, 7, or 8.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
    
    """
    # Edge pruning procedure is only applicable for the initialized
    # topology of spider web-inspired Delaunay-triangulated networks.
    # Exit if a different type of network is passed.
    if network != "swidt":
        error_str = (
            "Edge pruning procedure is only applicable for the "
            + "initialized topology of spider web-inspired "
            + "Delaunay-triangulated networks. This procedure will "
            + "only proceed if network = ``swidt''."
        )
        print(error_str)
        return None
    
    # Initialize random number generator
    rng = np.random.default_rng()

    # Initialize node number integer constants
    core_node_0 = 0
    core_node_1 = 0

    # Generate configuration filename prefix
    config_filename_prefix = config_filename_str(
        network, date, batch, sample, config)
    coords_filename = config_filename_prefix + ".coords"
    conn_core_edges_filename = (
        config_filename_prefix + "-conn_core_edges" + ".dat"
    )
    conn_pb_edges_filename = config_filename_prefix + "-conn_pb_edges" + ".dat"

    # Generate configuration and pruning filename prefix. This
    # establishes the configuration and pruning filename prefix as the
    # filename prefix associated with spider web-inspired
    # Delaunay-triangulated network data files, which is reflected in
    # the swidt_filename_str() function.
    config_pruning_filename_prefix = config_pruning_filename_str(
        network, date, batch, sample, config, pruning)
    mx_cmp_pruned_coords_filename = config_pruning_filename_prefix + ".coords"
    mx_cmp_pruned_conn_core_edges_filename = (
        config_pruning_filename_prefix + "-conn_core_edges" + ".dat"
    )
    mx_cmp_pruned_conn_pb_edges_filename = (
        config_pruning_filename_prefix + "-conn_pb_edges" + ".dat"
    )
    
    # Load fundamental graph constituents
    core_nodes = np.arange(n, dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Load core node coordinates
    coords = np.loadtxt(coords_filename)
    
    # Create nx.Graphs, and add nodes before edges
    conn_core_graph = nx.Graph()
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_core_graph = add_edges_from_numpy_array(conn_core_graph, conn_core_edges)

    conn_pb_graph = nx.Graph()
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)

    conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Degree of nodes in the graph
    conn_graph_k = k_func(conn_graph)

    if np.any(conn_graph_k > k):
        # Explicit edge pruning procedure
        while np.any(conn_graph_k > k):
            # Identify the nodes connected to more than k edges in the
            # graph, i.e., hyperconnected nodes
            conn_graph_hyprconn_nodes = np.where(conn_graph_k > k)[0]
            # Identify the edges connected to the hyperconnected nodes
            conn_graph_hyprconn_edges = np.logical_or(
                np.isin(conn_edges[:, 0], conn_graph_hyprconn_nodes),
                np.isin(conn_edges[:, 1], conn_graph_hyprconn_nodes))
            conn_graph_hyprconn_edge_indcs = (
                np.where(conn_graph_hyprconn_edges)[0]
            )
            # Randomly select a hyperconnected edge to remove
            edge_indcs_indx2remove_indx = (
                rng.integers(
                    np.shape(conn_graph_hyprconn_edge_indcs)[0], dtype=int)
            )
            edge_indx2remove = (
                conn_graph_hyprconn_edge_indcs[edge_indcs_indx2remove_indx]
            )
            core_node_0 = int(conn_edges[edge_indx2remove, 0])
            core_node_1 = int(conn_edges[edge_indx2remove, 1])

            # Remove hyperconnected edge in the graphs
            if conn_graph.has_edge(core_node_0, core_node_1):
                conn_graph.remove_edge(core_node_0, core_node_1)
                conn_edges = np.delete(conn_edges, edge_indx2remove, axis=0)
                if conn_core_graph.has_edge(core_node_0, core_node_1):
                    conn_core_graph.remove_edge(core_node_0, core_node_1)
                elif conn_pb_graph.has_edge(core_node_0, core_node_1):
                    conn_pb_graph.remove_edge(core_node_0, core_node_1)

                # Update degree of nodes in the graph
                conn_graph_k[core_node_0] -= 1
                conn_graph_k[core_node_1] -= 1
                
        # Isolate largest/maximum connected component in a nodewise
        # fashion
        mx_cmp_pruned_conn_graph_nodes = max(
            nx.connected_components(conn_graph), key=len)
        mx_cmp_pruned_conn_core_graph = (
            conn_core_graph.subgraph(mx_cmp_pruned_conn_graph_nodes).copy()
        )
        mx_cmp_pruned_conn_pb_graph = (
            conn_pb_graph.subgraph(mx_cmp_pruned_conn_graph_nodes).copy()
        )
        mx_cmp_pruned_conn_core_graph_edges = np.asarray(
            list(mx_cmp_pruned_conn_core_graph.edges()), dtype=int)
        mx_cmp_pruned_conn_pb_graph_edges = np.asarray(
            list(mx_cmp_pruned_conn_pb_graph.edges()), dtype=int)
        # Number of edges in the largest/maximum connected component
        mx_cmp_pruned_conn_core_graph_m = (
            np.shape(mx_cmp_pruned_conn_core_graph_edges)[0]
        )
        mx_cmp_pruned_conn_pb_graph_m = (
            np.shape(mx_cmp_pruned_conn_pb_graph_edges)[0]
        )
        # Nodes from the largest/maximum connected component, sorted in
        # ascending order
        mx_cmp_pruned_conn_graph_nodes = (
            np.sort(np.fromiter(mx_cmp_pruned_conn_graph_nodes, dtype=int))
        )
        # Construct an np.ndarray that returns the index for each node
        # number in the mx_cmp_pruned_conn_graph_nodes np.ndarray
        mx_cmp_pruned_conn_graph_nodes_indcs = (
            -1 * np.ones(np.max(mx_cmp_pruned_conn_graph_nodes)+1, dtype=int)
        )
        mx_cmp_pruned_conn_graph_nodes_indcs[mx_cmp_pruned_conn_graph_nodes] = (
            np.arange(np.shape(mx_cmp_pruned_conn_graph_nodes)[0], dtype=int)
        )

        # Isolate the core node coordinates for the largest/maximum
        # connected component
        mx_cmp_pruned_coords = coords[mx_cmp_pruned_conn_graph_nodes]

        # Update all original node values with updated node values
        for edge in range(mx_cmp_pruned_conn_core_graph_m):
            mx_cmp_pruned_conn_core_graph_edges[edge, 0] = int(
                mx_cmp_pruned_conn_graph_nodes_indcs[mx_cmp_pruned_conn_core_graph_edges[edge, 0]])
            mx_cmp_pruned_conn_core_graph_edges[edge, 1] = int(
                mx_cmp_pruned_conn_graph_nodes_indcs[mx_cmp_pruned_conn_core_graph_edges[edge, 1]])
        for edge in range(mx_cmp_pruned_conn_pb_graph_m):
            mx_cmp_pruned_conn_pb_graph_edges[edge, 0] = int(
                mx_cmp_pruned_conn_graph_nodes_indcs[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]])
            mx_cmp_pruned_conn_pb_graph_edges[edge, 1] = int(
                mx_cmp_pruned_conn_graph_nodes_indcs[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]])
                
        # Save fundamental graph constituents from this topology
        np.savetxt(
            mx_cmp_pruned_conn_core_edges_filename,
            mx_cmp_pruned_conn_core_graph_edges, fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_pb_edges_filename,
            mx_cmp_pruned_conn_pb_graph_edges, fmt="%d")
        
        # Save the core node coordinates
        np.savetxt(mx_cmp_pruned_coords_filename, mx_cmp_pruned_coords)
    else:
        # Save fundamental graph constituents from this topology
        np.savetxt(
            mx_cmp_pruned_conn_core_edges_filename, conn_core_edges, fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_pb_edges_filename, conn_pb_edges, fmt="%d")
        
        # Save the core node coordinates
        np.savetxt(mx_cmp_pruned_coords_filename, coords)

def swidt_network_topological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        pruning: int,
        length_bound: int,
        tplgcl_dscrptr: str,
        np_oprtn: str,
        eeel_ntwrk: bool,
        save_tplgcl_dscrptr_result: bool,
        return_tplgcl_dscrptr_result: bool) -> np.ndarray | float | int | None:
    """Spider web-inspired Delaunay-triangulated network topological
    descriptor.
    
    This function extracts a spider web-inspired Delaunay-triangulated
    network and sets a variety of input parameters corresponding to a
    particular topological descriptor (and numpy function) of interest.
    These are then passed to the master network_topological_descriptor()
    function, which calculates (and, if called for, saves) the result of
    the topological descriptor for the spider web-inspired
    Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider-web inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
        length_bound (int): Maximum ring order (inclusive).
        tplgcl_dscrptr (str): Topological descriptor name.
        np_oprtn (str): numpy function/operation name.
        eeel_ntwrk (bool): Boolean indicating if the elastically-effective end-linked network ought to be supplied for the topological descriptor calculation.
        save_tplgcl_dscrptr_result (bool): Boolean indicating if the topological descriptor result ought to be saved.
        return_tplgcl_dscrptr_result (bool): Boolean indicating if the topological descriptor result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Topological descriptor result.
    
    """
    # This topological descriptor calculation is only applicable for
    # data files associated with spider web-inspired
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "swidt":
        error_str = (
            "This topological descriptor calculation is only "
            + "applicable for data files associated with spider "
            + "web-inspired Delaunay-triangulated networks. This "
            + "calculation will proceed only if network = ``swidt'."
        )
        print(error_str)
        return None
    
    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    swidt_filename = swidt_filename_str(
        network, date, batch, sample, config, pruning)
    coords_filename = swidt_filename + ".coords"
    conn_core_edges_filename = swidt_filename + "-conn_core_edges.dat"
    conn_pb_edges_filename = swidt_filename + "-conn_pb_edges.dat"
    
    if eeel_ntwrk == True:
        if np_oprtn == "":
            tplgcl_dscrptr_result_filename = (
                swidt_filename + "-eeel-" + tplgcl_dscrptr + ".dat"
            )
        else:
            tplgcl_dscrptr_result_filename = (
                swidt_filename + "-eeel-" + np_oprtn + "-" + tplgcl_dscrptr
                + ".dat"
            )
    else:
        if np_oprtn == "":
            tplgcl_dscrptr_result_filename = (
                swidt_filename + "-" + tplgcl_dscrptr + ".dat"
            )
        else:
            tplgcl_dscrptr_result_filename = (
                swidt_filename + "-" + np_oprtn + "-" + tplgcl_dscrptr
                + ".dat"
            )

    # Load simulation box size and node coordinates
    L = np.loadtxt(L_filename)
    coords = np.loadtxt(coords_filename)

    # Load fundamental graph constituents
    core_nodes = np.arange(np.shape(coords)[0], dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Create nx.Graphs, and add nodes before edges
    conn_core_graph = nx.Graph()
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_core_graph = add_edges_from_numpy_array(conn_core_graph, conn_core_edges)

    conn_pb_graph = nx.Graph()
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)

    conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Call the master network_topological_descriptor() function
    return network_topological_descriptor(
        tplgcl_dscrptr, np_oprtn, conn_core_graph, conn_pb_graph, conn_graph,
        coords, L, length_bound, eeel_ntwrk, tplgcl_dscrptr_result_filename,
        save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)