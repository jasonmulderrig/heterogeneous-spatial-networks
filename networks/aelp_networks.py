import numpy as np
import networkx as nx
from file_io.file_io import (
    L_filename_str,
    config_filename_str
)
from helpers.network_utils import (
    m_arg_stoich_func,
    n_nu_arg_m_func
)
from helpers.simulation_box_utils import L_arg_rho_func
from helpers.network_topology_initialization_utils import (
    tessellation_protocol,
    tessellation,
    orb_neighborhood_id
)
from helpers.graph_utils import (
    add_nodes_from_numpy_array,
    add_edges_from_numpy_array
)
from topological_descriptors.network_topological_descriptors import (
    network_topological_descriptor
)

def aelp_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> str:
    """Filename prefix associated with artificial end-linked polymer
    networks.

    This function returns the filename prefix associated with artificial
    end-linked polymer network data files.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, either "auelp", "abelp", or "apelp" are applicable (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial bimodal end-linked polymer networks ("abelp"), or artificial polydisperse end-linked polymer networks ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    Returns:
        str: The filename prefix associated with artificial end-linked
        polymer network data files.
    
    """
    # This filename prefix convention is only applicable for artificial
    # end-linked polymer networks. Exit if a different type of network
    # is passed.
    if network not in ["auelp", "abelp", "apelp"]:
        error_str = (
            "This filename prefix convention is only applicable for "
            + "artificial end-linked polymer networks. This "
            + "calculation will only proceed if network = ``auelp'', "
            + "network = ``abelp'', or network = ``apelp''."
        )
        print(error_str)
        return None
    return config_filename_str(network, date, batch, sample, config)

def aelp_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        rho_nu: float,
        k: int,
        n: int,
        nu: int) -> None:
    """Simulation box size for artificial end-linked polymer networks.

    This function calculates the simulation box size for artificial
    end-linked polymer networks.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, either "auelp", "abelp", or "apelp" are applicable (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial bimodal end-linked polymer networks ("abelp"), or artificial polydisperse end-linked polymer networks ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        rho_nu (float): Segment number density.
        k (int): Maximum cross-linker degree/functionality; either 3, 4, 5, 6, 7, or 8.
        n (int): Intended number of core cross-linkers.
        nu (int): (Average) Number of segments per chain.
    
    """
    # This calculation for L is only applicable for artificial
    # end-linked polymer networks. Exit if a different type of network
    # is passed.
    if network not in ["auelp", "abelp", "apelp"]:
        error_str = (
            "This calculation for L is only applicable for artificial"
            + "end-linked polymer networks. This calculation will only "
            + "proceed if network = ``auelp'', network = ``abelp'', or "
            + "network = ``apelp''."
        )
        print(error_str)
        return None
    
    # Calculate the stoichiometric (average) number of chain segments in
    # the simulation box
    n_nu = n_nu_arg_m_func(m_arg_stoich_func(n, k), nu)

    # Calculate and save L
    np.savetxt(
        L_filename_str(network, date, batch, sample),
        [L_arg_rho_func(dim, n_nu, rho_nu)])

def core_node_update_func(
        core_node_updt: int,
        core_node_updt_active_site_indx: int,
        core_nodes_active_sites: np.ndarray,
        core_nodes_active_sites_num: int,
        core_nodes_anti_k: np.ndarray,
        core_nodes: np.ndarray,
        core2pb_nodes: list[np.ndarray],
        core_nodes_nghbrhd: list[np.ndarray],
        r_core_nodes_nghbrhd: list[np.ndarray]) -> tuple[np.ndarray, int, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Core node update protocol.

    This function updates the core node active cross-linking sites,
    anti-degree (the number of remaining active cross-linking sites),
    and core node sampling neighborhood list in light of a particular
    core node active cross-linking site becoming inactive.

    Args:
        core_node_updt (int): Core node number corresponding to where an active cross-linking site is becoming inactive.
        core_node_updt_active_site_indx (int): Index corresponding to the core node active cross-linking site that is becoming inactive.
        core_nodes_active_sites (np.ndarray): np.ndarray of the core node numbers that have active cross-linking sites.
        core_nodes_active_sites_num (int): Number of core node active cross-linking sites.
        core_nodes_anti_k (np.ndarray): np.ndarray of the number of remaining active cross-linking sites for each core node.
        core_nodes (np.ndarray): np.ndarray of the core node numbers.
        core2pb_nodes (list[np.ndarray]): List of np.ndarrays corresponding to the periodic boundary nodes associated with each core node.
        core_nodes_nghbrhd (list[np.ndarray]): List of np.ndarrays corresponding to the core and periodic boundary node numbers in the sampling neighborhood of each core node.
        r_core_nodes_nghbrhd (list[np.ndarray]): List of np.ndarrays corresponding to the neighbor node-to-core node distances for each core node.
    
    Returns:
        tuple[np.ndarray, int, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        Core node numbers that have active cross-linking sites, number
        of core node active cross-linking sites, number of remaining
        active cross-linking sites for each core node, core and periodic
        boundary node numbers in the sampling neighborhood of each core
        node, and the neighbor node-to-core node distances for each core
        node.
    
    """
    # Update active cross-linking sites
    core_nodes_active_sites = np.delete(
        core_nodes_active_sites, core_node_updt_active_site_indx, axis=0)
    core_nodes_active_sites_num -= 1
    # Update anti-degree
    core_nodes_anti_k[core_node_updt] -= 1
    # Update core node sampling neighborhood list if the core node now
    # has no more active cross-linking sites
    if core_nodes_anti_k[core_node_updt] == 0:
        # Gather periodic boundary nodes associated with the inactive
        # core node
        pb_nodes_updt = core2pb_nodes[core_node_updt]
        # Assess and update each core node sampling neighborhood list
        for core_node in np.nditer(core_nodes):
            core_node = int(core_node)
            nghbr_nodes = core_nodes_nghbrhd[core_node]
            r_nghbr_nodes = r_core_nodes_nghbrhd[core_node]
            if np.shape(nghbr_nodes)[0] == 0: pass
            else:
                # Address inactive core node
                core_node_updt_indx_arr = (
                    np.where(nghbr_nodes == core_node_updt)[0]
                )
                if np.shape(core_node_updt_indx_arr)[0] == 0: pass
                else:
                    core_node_updt_indx = int(core_node_updt_indx_arr[0])
                    nghbr_nodes = np.delete(
                        nghbr_nodes, core_node_updt_indx, axis=0)
                    r_nghbr_nodes = np.delete(
                        r_nghbr_nodes, core_node_updt_indx, axis=0)
                # Address all associated inactive periodic boundary
                # nodes
                if np.shape(pb_nodes_updt)[0] == 0: pass
                else:
                    for pb_node_updt in np.nditer(pb_nodes_updt):
                        pb_node_updt = int(pb_node_updt)
                        pb_node_updt_indx_arr = (
                            np.where(nghbr_nodes == pb_node_updt)[0]
                        )
                        if np.shape(pb_node_updt_indx_arr)[0] == 0: pass
                        else:
                            pb_node_updt_indx = int(pb_node_updt_indx_arr[0])
                            nghbr_nodes = np.delete(
                                nghbr_nodes, pb_node_updt_indx, axis=0)
                            r_nghbr_nodes = np.delete(
                                r_nghbr_nodes, pb_node_updt_indx, axis=0)
                core_nodes_nghbrhd[core_node] = nghbr_nodes
                r_core_nodes_nghbrhd[core_node] = r_nghbr_nodes
    
    return (
        core_nodes_active_sites, core_nodes_active_sites_num, core_nodes_anti_k,
        core_nodes_nghbrhd, r_core_nodes_nghbrhd
    )

def dangling_chains_update_func(
        core_node_updt: int,
        max_try: int,
        rng: np.random.Generator,
        r_nghbrhd_chns: np.ndarray,
        p_nghbrhd_chns: np.ndarray,
        dim: int,
        b: float,
        L: float,
        dnglng_chn_fail: bool,
        dnglng_n: int,
        core_dnglng_m: int,
        pb_dnglng_m: int,
        conn_core_dnglng_edges: list[list[int, int]],
        conn_pb_dnglng_edges: list[list[int, int]],
        core_dnglng_chns_coords: list[np.ndarray],
        core_pb_dnglng_chns_coords: np.ndarray) -> tuple[bool, int, int, int, list[list[int, int]], list[list[int, int]], list[np.ndarray], np.ndarray]:
    """Dangling chains update protocol.

    This function instantiates a dangling chain about a particular core
    cross-linker node by randomly placing the dangling chain free end in
    the network and confirming that this free end is at least a distance
    b away from all other nodes.

    Args:
        core_node_updt (int): Core node number corresponding to where an active cross-linking site is becoming inactive. This node is also where the dangling chain will emanate out from.
        max_try (int): Maximum number of dangling chain instantiation attempts.
        rng (np.random.Generator): np.random.Generator object.
        r_nghbrhd_chns (np.ndarray): np.ndarray of the core cross-linker sampling neighborhood radius.
        p_nghbrhd_chns (np.ndarray): np.ndarray of the core cross-linker sampling neighborhood polymer chain probability distribution.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Chain segment and/or cross-linker diameter.
        L (float): Simulation box size.
        dnglng_chn_fail (bool): Tracker for failure of dangling chain creation.
        dnglng_n (int): Number of dangling chains, and dangling chain core node number.
        core_dnglng_m (int): Number of core dangling chains.
        pb_dnglng_m (int): Number of periodic boundary dangling chains.
        conn_core_dnglng_edges (list[list[int, int]]): Core dangling chain edge list, where each edge is represented as [core_node_updt, dnglng_n].
        conn_pb_dnglng_edges (list[list[int, int]]): Periodic boundary dangling chain edge list, where each edge is represented as [core_node_updt, dnglng_n].
        core_dnglng_chns_coords (list[np.ndarray]): List of dangling chain free end node coordinates in the simulation box. Each coordinate is an individual np.ndarray.
        core_pb_dnglng_chns_coords (np.ndarray): np.ndarray of core, periodic boundary, and dangling chain free end node coordinates.
    
    Returns:
        tuple[bool, int, int, int, list[list[int, int]], list[list[int, int]], list[np.ndarray], np.ndarray]:
        Tracker for failure of dangling chain creation, number of
        dangling chains, number of core dangling chains, number of
        periodic boundary dangling chains, core dangling chain edge
        list, periodic boundary dangling chain edge list, list of
        dangling chain free end node coordinates in the simulation box,
        and core, periodic boundary, and dangling chain free end node
        coordinates.
    
    """
    # Tessellation protocol
    tsslltn, _ = tessellation_protocol(dim)
    
    # Begin dangling chain update procedure
    num_try = 0
    node_dnglng_chn_cnddt = core_pb_dnglng_chns_coords[core_node_updt].copy()

    while num_try < max_try:
        # Isotropically place a candidate node representing the free end
        # of the dangling chain
        r = rng.choice(r_nghbrhd_chns, size=None, p=p_nghbrhd_chns)
        # In two dimensions, a polar coordinate-based sampling method is
        # used for the dangling chain free end placement
        if dim == 2:
            theta = 2 * np.pi * rng.uniform()
            node_dnglng_chn_cnddt[0] += r * np.cos(theta)
            node_dnglng_chn_cnddt[1] += r * np.sin(theta)
        # In three dimensions, a spherical coordinate-based sampling
        # method is used for the dangling chain free end placement
        elif dim == 3:
            theta = np.pi * rng.uniform()
            phi = 2 * np.pi * rng.uniform()
            node_dnglng_chn_cnddt[0] += r * np.sin(theta) * np.cos(phi)
            node_dnglng_chn_cnddt[1] += r * np.sin(theta) * np.sin(phi)
            node_dnglng_chn_cnddt[2] += r * np.cos(theta)
        
        # Downselect the local orb neighborhood with radius b about
        # the free end of the dangling chain
        _, orb_nghbr_num = orb_neighborhood_id(
            dim, core_pb_dnglng_chns_coords, node_dnglng_chn_cnddt, b,
            inclusive=False, indices=True)
        
        # Try again if the local orb neighborhood has at least one
        # neighbor in it
        if orb_nghbr_num > 0:
            num_try += 1
            continue
        
        # Accept and tessellate the node candidate if no local orb
        # neighborhood of tessellated nodes exists about the node
        # candidate
        dnglng_n += 1
        core_node_dnglng_chn = node_dnglng_chn_cnddt
        
        # Free end is a core node
        if np.all(np.logical_and(core_node_dnglng_chn>=0, core_node_dnglng_chn<L)):
            # Increase core dangling chain number
            core_dnglng_m += 1
            # Add to edge list
            conn_core_dnglng_edges.append([core_node_updt, dnglng_n])
        # Free end is a periodic boundary node
        else:
            # Gather core node coordinates via the minimum image
            # convention
            core_node_dnglng_chn[core_node_dnglng_chn<0] += L
            core_node_dnglng_chn[core_node_dnglng_chn>=L] -= L
            # Increase periodic boundary dangling chain number
            pb_dnglng_m += 1
            # Add to edge lists
            conn_pb_dnglng_edges.append([core_node_updt, dnglng_n])
        
        # Add the accepted candidate core node to list of dangling chain
        # free end node core coordinates
        core_dnglng_chns_coords.append(core_node_dnglng_chn)

        # Tessellate the accepted candidate core node, and add the
        # result to the np.ndarray of core, periodic boundary, and
        # dangling chain free end node coordinates
        core_pb_dnglng_chns_coords = np.vstack(
            (core_pb_dnglng_chns_coords, tessellation(core_node_dnglng_chn, tsslltn, L)))

        break

    # Update the dangling chain creation failure tracker if the number
    # of attempts to instantiate the dangling chain is equal to its
    # maximal value
    if num_try == max_try: dnglng_chn_fail = True

    return (
        dnglng_chn_fail, dnglng_n, core_dnglng_m, pb_dnglng_m,
        conn_core_dnglng_edges, conn_pb_dnglng_edges,
        core_dnglng_chns_coords, core_pb_dnglng_chns_coords
    )

def aelp_network_topological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        length_bound: int,
        tplgcl_dscrptr: str,
        np_oprtn: str,
        eeel_ntwrk: bool,
        save_tplgcl_dscrptr_result: bool,
        return_tplgcl_dscrptr_result: bool) -> np.ndarray | float | int | None:
    """Artificial end-linked polymer network topological descriptor.
    
    This function extracts an artificial end-linked polymer network
    and sets a variety of input parameters corresponding to a particular
    topological descriptor (and numpy function) of interest. These are
    then passed to the master network_topological_descriptor() function,
    which calculates (and, if called for, saves) the result of the
    topological descriptor for the artificial end-linked polymer
    network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, either "auelp" or "apelp" are applicable (corresponding to artificial uniform end-linked polymer networks ("auelp") or artificial polydisperse end-linked polymer networks ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
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
    # data files associated with artificial end-linked polymer networks.
    # Exit if a different type of network is passed.
    if network not in ["auelp", "abelp", "apelp"]:
        error_str = (
            "This topological descriptor calculation is only applicable "
            + "for artificial end-linked polymer networks. This "
            + "calculation will only proceed if network = ``auelp'', "
            + "network = ``abelp'', or network = ``apelp''."
        )
        print(error_str)
        return None
    
    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    aelp_filename = aelp_filename_str(network, date, batch, sample, config)
    coords_filename = aelp_filename + ".coords"
    conn_core_edges_filename = aelp_filename + "-conn_core_edges.dat"
    conn_pb_edges_filename = aelp_filename + "-conn_pb_edges.dat"
    
    if eeel_ntwrk == True:
        if np_oprtn == "":
            tplgcl_dscrptr_result_filename = (
                aelp_filename + "-eeel-" + tplgcl_dscrptr + ".dat"
            )
        else:
            tplgcl_dscrptr_result_filename = (
                aelp_filename + "-eeel-" + np_oprtn + "-" + tplgcl_dscrptr
                + ".dat"
            )
    else:
        if np_oprtn == "":
            tplgcl_dscrptr_result_filename = (
                aelp_filename + "-" + tplgcl_dscrptr + ".dat"
            )
        else:
            tplgcl_dscrptr_result_filename = (
                aelp_filename + "-" + np_oprtn + "-" + tplgcl_dscrptr
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

    # Create nx.MultiGraphs, and add nodes before edges
    conn_core_graph = nx.MultiGraph()
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_core_graph = add_edges_from_numpy_array(
        conn_core_graph, conn_core_edges)
    
    conn_pb_graph = nx.MultiGraph()
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)
    
    conn_graph = nx.MultiGraph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Call the master network_topological_descriptor() function
    return network_topological_descriptor(
        tplgcl_dscrptr, np_oprtn, conn_core_graph, conn_pb_graph, conn_graph,
        coords, L, length_bound, eeel_ntwrk, tplgcl_dscrptr_result_filename,
        save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)