import numpy as np
import networkx as nx
from file_io.file_io import (
    L_filename_str,
    config_filename_str
)
from helpers.simulation_box_utils import L_arg_eta_func
from scipy.spatial import Delaunay
from helpers.network_topology_initialization_utils import (
    core_node_tessellation,
    unique_sorted_edges
)
from helpers.graph_utils import (
    add_nodes_from_numpy_array,
    add_edges_from_numpy_array
)
from topological_descriptors.network_topological_descriptors import (
    network_topological_descriptor
)

def delaunay_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> str:
    """Filename prefix associated with Delaunay-triangulated network
    data files.

    This function returns the filename prefix associated with
    Delaunay-triangulated network data files.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    Returns:
        str: The filename prefix associated with Delaunay-triangulated
        network data files.
    
    """
    # This filename prefix convention is only applicable for data files
    # associated with Delaunay-triangulated networks. Exit if a
    # different type of network is passed.
    if network != "delaunay":
        error_str = (
            "This filename prefix convention is only applicable for "
            + "data files associated with Delaunay-triangulated "
            + "networks. This filename prefix will only be supplied if "
            + "network = ``delaunay''."
        )
        print(error_str)
        return None
    return config_filename_str(network, date, batch, sample, config)

def delaunay_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float) -> None:
    """Simulation box size for Delaunay-triangulated networks.

    This function calculates and saves the simulation box size for
    Delaunay-triangulated networks.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended number of core nodes.
        eta_n (float): Node packing density.
    
    """
    # This calculation for L is only applicable for
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "delaunay":
        error_str = (
            "This calculation for L is only applicable for "
            + "Delaunay-triangulated networks. This calculation will "
            + "only proceed if network = ``delaunay''."
        )
        print(error_str)
        return None
    
    # Calculate and save L
    np.savetxt(
        L_filename_str(network, date, batch, sample),
        [L_arg_eta_func(dim, b, n, eta_n)])

def delaunay_network_topology_initialization(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Network topology initialization procedure for
    Delaunay-triangulated networks.

    This function loads the simulation box size and the core node
    coordinates. Then, this function ``tessellates'' the core nodes
    about themselves, applies Delaunay triangulation to the resulting
    tessellated network via the scipy.spatial.Delaunay() function,
    acquires back the periodic network topology of the core nodes, and
    ascertains fundamental graph constituents (node and edge
    information) from this topology.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" or "swidt" are applicable (corresponding to Delaunay-triangulated networks ("delaunay") and spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Load simulation box size
    L = np.loadtxt(L_filename_str(network, date, batch, sample))

    # Generate configuration filename prefix. This establishes the
    # configuration filename prefix as the filename prefix associated
    # with Delaunay-triangulated network data files, which is reflected
    # in the delaunay_filename_str() function.
    config_filename_prefix = config_filename_str(
        network, date, batch, sample, config)

    # Generate filenames
    coords_filename = config_filename_prefix + ".coords"
    conn_core_edges_filename = (
        config_filename_prefix + "-conn_core_edges" + ".dat"
    )
    conn_pb_edges_filename = config_filename_prefix + "-conn_pb_edges" + ".dat"

    # Call appropriate helper function to initialize network topology
    if (scheme == "random") or (scheme == "prhd") or (scheme == "pdhu"):
        # Load core node coordinates
        coords = np.loadtxt(coords_filename)
    elif scheme == "lammps":
        skiprows_num = 15
        # Load core node coordinates
        coords = np.loadtxt(coords_filename, skiprows=skiprows_num, max_rows=n)
    
    # Actual number of core nodes
    n = np.shape(coords)[0]

    # Core nodes
    core_nodes = np.arange(n, dtype=int)

    # Tessellate the core node coordinates and construct the
    # pb2core_nodes np.ndarray
    tsslltd_coords, pb2core_nodes = core_node_tessellation(
        dim, core_nodes, coords, L)
    
    del core_nodes

    # Shift the coordinate origin to the center of the simulation box
    # for improved Delaunay triangulation performance
    tsslltd_coords -= 0.5 * L

    # Apply Delaunay triangulation
    tsslltd_core_delaunay = Delaunay(tsslltd_coords)

    del tsslltd_coords

    # Extract the simplices from the Delaunay triangulation
    simplices = tsslltd_core_delaunay.simplices

    # List for edges of the core and periodic boundary nodes
    tsslltd_core_pb_edges = []

    for simplex in simplices:
        # In two dimensions, each simplex is a triangle
        if dim == 2:
            node_0 = int(simplex[0])
            node_1 = int(simplex[1])
            node_2 = int(simplex[2])

            # If any of the nodes involved in any simplex edge
            # correspond to the original core nodes, then add that edge
            # to the edge list. Duplicate entries will arise.
            if (node_0 < n) or (node_1 < n):
                tsslltd_core_pb_edges.append((node_0, node_1))
            if (node_1 < n) or (node_2 < n):
                tsslltd_core_pb_edges.append((node_1, node_2))
            if (node_2 < n) or (node_0 < n):
                tsslltd_core_pb_edges.append((node_2, node_0))
            else: pass
        # In three dimensions, each simplex is a tetrahedron
        elif dim == 3:
            node_0 = int(simplex[0])
            node_1 = int(simplex[1])
            node_2 = int(simplex[2])
            node_3 = int(simplex[3])

            # If any of the nodes involved in any simplex edge
            # correspond to the original core nodes, then add those
            # nodes and that edge to the appropriate lists. Duplicate
            # entries will arise.
            if (node_0 < n) or (node_1 < n):
                tsslltd_core_pb_edges.append((node_0, node_1))
            if (node_1 < n) or (node_2 < n):
                tsslltd_core_pb_edges.append((node_1, node_2))
            if (node_2 < n) or (node_0 < n):
                tsslltd_core_pb_edges.append((node_2, node_0))
            if (node_3 < n) or (node_0 < n):
                tsslltd_core_pb_edges.append((node_3, node_0))
            if (node_3 < n) or (node_1 < n):
                tsslltd_core_pb_edges.append((node_3, node_1))
            if (node_3 < n) or (node_2 < n):
                tsslltd_core_pb_edges.append((node_3, node_2))
            else: pass
    
    del simplex, simplices, tsslltd_core_delaunay

    # Convert edge list to np.ndarray, and retain the unique edges from
    # the core and periodic boundary nodes
    tsslltd_core_pb_edges = unique_sorted_edges(tsslltd_core_pb_edges)

    # Lists for the edges of the graph capturing the periodic
    # connections between the core nodes
    conn_core_edges = []
    conn_pb_edges = []

    for edge in range(np.shape(tsslltd_core_pb_edges)[0]):
        node_0 = int(tsslltd_core_pb_edges[edge, 0])
        node_1 = int(tsslltd_core_pb_edges[edge, 1])

        # Edge is a core edge
        if (node_0 < n) and (node_1 < n):
            conn_core_edges.append((node_0, node_1))
        # Edge is a periodic boundary edge
        else:
            node_0 = int(pb2core_nodes[node_0])
            node_1 = int(pb2core_nodes[node_1])
            conn_pb_edges.append((node_0, node_1))
    
    # Convert edge lists to np.ndarrays, and retain unique edges
    conn_core_edges = unique_sorted_edges(conn_core_edges)
    conn_pb_edges = unique_sorted_edges(conn_pb_edges)

    # Save fundamental graph constituents from this topology
    np.savetxt(conn_core_edges_filename, conn_core_edges, fmt="%d")
    np.savetxt(conn_pb_edges_filename, conn_pb_edges, fmt="%d")
    
def delaunay_network_topology(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Delaunay-triangulated network topology.

    This function confirms that the network being called for is a
    Delaunay-triangulated network. Then, the function calls the
    Delaunay-triangulated network initialization function to create the
    Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Network topology initialization procedure is only applicable for 
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "delaunay":
        error_str = (
            "Network topology initialization procedure is only "
            + "applicable for Delaunay-triangulated networks. This "
            + "procedure will only proceed if network = ``delaunay''."
        )
        print(error_str)
        return None
    delaunay_network_topology_initialization(
        network, date, batch, sample, scheme, dim, n, config)

def delaunay_network_topological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        length_bound: int,
        tplgcl_dscrptr: str,
        np_oprtn: str,
        save_tplgcl_dscrptr_result: bool,
        return_tplgcl_dscrptr_result: bool) -> np.ndarray | float | int | None:
    """Delaunay-triangulated network topological descriptor.
    
    This function extracts a Delaunay-triangulated network and sets a
    variety of input parameters corresponding to a particular
    topological descriptor (and numpy function) of interest. These are
    then passed to the master network_topological_descriptor() function,
    which calculates (and, if called for, saves) the result of the
    topological descriptor for the Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        length_bound (int): Maximum ring order (inclusive).
        tplgcl_dscrptr (str): Topological descriptor name.
        np_oprtn (str): numpy function/operation name.
        save_tplgcl_dscrptr_result (bool): Boolean indicating if the topological descriptor result ought to be saved.
        return_tplgcl_dscrptr_result (bool): Boolean indicating if the topological descriptor result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Topological descriptor result.
    
    """
    # This topological descriptor calculation is only applicable for
    # data files associated with Delaunay-triangulated networks. Exit if a
    # different type of network is passed.
    if network != "delaunay":
        error_str = (
            "This topological descriptor calculation is only "
            + "applicable for data files associated with "
            + "Delaunay-triangulated networks. This calculation will "
            + "proceed only if network = ``delaunay'."
        )
        print(error_str)
        return None
    
    # Delaunay-triangulated networks are completely
    # elastically-effective, and thus there is no need to specify if an
    # elastically-effective end-linked network is desired
    eeel_ntwrk = False

    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    delaunay_filename = delaunay_filename_str(network, date, batch, sample, config)
    coords_filename = delaunay_filename + ".coords"
    conn_core_edges_filename = delaunay_filename + "-conn_core_edges.dat"
    conn_pb_edges_filename = delaunay_filename + "-conn_pb_edges.dat"
    
    if eeel_ntwrk == True:
        if np_oprtn == "":
            tplgcl_dscrptr_result_filename = (
                delaunay_filename + "-eeel-" + tplgcl_dscrptr + ".dat"
            )
        else:
            tplgcl_dscrptr_result_filename = (
                delaunay_filename + "-eeel-" + np_oprtn + "-" + tplgcl_dscrptr
                + ".dat"
            )
    else:
        if np_oprtn == "":
            tplgcl_dscrptr_result_filename = (
                delaunay_filename + "-" + tplgcl_dscrptr + ".dat"
            )
        else:
            tplgcl_dscrptr_result_filename = (
                delaunay_filename + "-" + np_oprtn + "-" + tplgcl_dscrptr
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