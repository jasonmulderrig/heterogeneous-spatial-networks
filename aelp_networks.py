import numpy as np
import networkx as nx
from file_io import (
    L_filename_str,
    config_filename_str
)
from network_utils import (
    m_arg_stoich_func,
    n_nu_arg_m_func
)
from simulation_box_utils import L_arg_rho_func
from network_topology_initialization_utils import (
    tessellation_protocol,
    tessellation,
    core_node_tessellation,
    core2pb_nodes_func,
    orb_neighborhood_id
)
from polymer_network_chain_statistics import (
    p_gaussian_cnfrmtn_func,
    p_net_gaussian_cnfrmtn_func,
    p_rel_net_gaussian_cnfrmtn_func
)
from graph_utils import (
    add_nodes_from_numpy_array,
    add_edges_from_numpy_array
)
from general_topological_descriptors import l_func
from network_topological_descriptors import network_topological_descriptor

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
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, either "auelp" or "apelp" are applicable (corresponding to artificial uniform end-linked polymer networks ("auelp") or artificial polydisperse end-linked polymer networks ("apelp")).
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
    if network != "auelp":
        if network != "apelp":
            error_str = (
                "This filename prefix convention is only applicable "
                + "for artificial end-linked polymer networks. This "
                + "calculation will only proceed if "
                + "network = ``auelp'' or network = ``apelp''."
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
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, either "auelp" or "apelp" are applicable (corresponding to artificial uniform end-linked polymer networks ("auelp") or artificial polydisperse end-linked polymer networks ("apelp")).
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
    if network != "auelp":
        if network != "apelp":
            error_str = (
                "This calculation for L is only applicable for "
                + "artificial end-linked polymer networks. This "
                + "calculation will only proceed if "
                + "network = ``auelp'' or network = ``apelp''."
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

def apelp_network_nu_assignment(
        rng: np.random.Generator,
        b: float,
        nu: int,
        nu_max: int,
        conn_core_m: int,
        conn_pb_m: int,
        r_chn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chain segment number assignment procedure for artificial
    polydisperse end-linked polymer networks.

    This function assigns a chain segment number to each chain in an
    artificial polydisperse end-linked polymer network via a modified
    Hanson protocol.

    Args:
        rng (np.random.Generator): np.random.Generator object.
        b (float): Chain segment and/or cross-linker diameter.
        nu (int): (Average) Number of segments per chain.
        nu_max (int): Maximum number of segments that could possibly be assigned to a chain.
        conn_core_m (int): Number of core edge chains.
        conn_pb_m (int): Number of periodic edge chains.
        r_chn (np.ndarray): End-to-end chain lengths.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Chain segment numbers for core
        and periodic edge chains, respectively.
    
    """
    # Initialize chain segment number array
    conn_nu_edges = np.empty(conn_core_m+conn_pb_m, dtype=int)

    # Minimum and maximum end-to-end chain lengths
    r_chn_min = np.min(r_chn)
    r_chn_max = np.max(r_chn)

    # Number of end-to-end chain length bins, as per Hanson
    num_bins = 20
    
    # End-to-end chain length bins
    r_chn_bins = np.linspace(r_chn_min, r_chn_max, num_bins+1)
    r_chn_bins_midpoints = (r_chn_bins[:-1] + r_chn_bins[1:]) / 2
    
    # Pad the first and last bin with a small tolerance value
    tol = 1e-10
    r_chn_bins[0] -= tol
    r_chn_bins[-1] += tol
    
    # Bin the end-to-end chain lengths
    bin_indcs = np.digitize(r_chn, r_chn_bins) - 1
    bincount = np.bincount(bin_indcs, minlength=num_bins)

    # Assign chain segment numbers to chains on a bin-by-bin basis
    for bin in range(num_bins):
        num_chns_bin = int(bincount[bin])
        # Continue to next bin if empty
        if num_chns_bin == 0: continue
        else:
            # Bin midpoint
            r_chn_bin_midpoint = r_chn_bins_midpoints[bin]
            # Determine the bin chain indices
            chn_bin_indcs = np.where(bin_indcs == bin)[0]
            # Determine the physically-constrained minimum chain segment
            # number for each chain in the bin
            nu_min_bin = np.ceil(r_chn[chn_bin_indcs]).astype(int)
            # Confirm that the maximal physically-constrained minimum
            # chain segment number is less than or equal to the
            # specified maximum chain segment number
            if int(np.max(nu_min_bin)) > nu_max:
                error_str = (
                    "The specified maximum chain segment number is "
                    + "less than the largest physically-constrained "
                    + "minimum chain segment number in the network. "
                    + "Please modify the specified maximum chain "
                    + "segment number accordingly."
                )
                print(error_str)
                return None
            # Determine minimum chain segment number for all chains in
            # the bin
            nu_min = int(np.min(nu_min_bin))
            # Confirm that the minimal physically-constrained minimum
            # chain segment number is less than or equal to the
            # specified maximum chain segment number
            if nu_min > nu_max:
                error_str = (
                    "The specified maximum chain segment number is "
                    + "less than the smallest physically-constrained "
                    + "minimum chain segment number in the network. "
                    + "Please modify the specified maximum chain "
                    + "segment number accordingly."
                )
                print(error_str)
                return None
            # Range of available chain segment numbers in the bin
            nu_bin = np.arange(nu_min, nu_max+1, dtype=int)
            # Calculate and normalize the probability distribution of
            # chains in the bin accounting for dispersity in chain
            # segment number and Gaussian end-to-end distance polymer
            # chain conformation
            p_bin = p_net_gaussian_cnfrmtn_func(
                b, nu, nu_bin.astype(float), r_chn_bin_midpoint)
            Z_p_bin = np.sum(p_bin, dtype=float)
            p_bin /= Z_p_bin
            p_bin[np.isnan(p_bin)] = 0.
            Z_p_bin = np.sum(p_bin, dtype=float)
            if Z_p_bin == 0:
                p_bin = np.ones(np.shape(p_bin)[0]) / np.shape(p_bin)[0]
            else:
                p_bin /= Z_p_bin
            if num_chns_bin == 1:
                # Randomly select a chain segment number, and assign
                # this to the appropriate chain in the polymer network
                conn_nu_edges[chn_bin_indcs[0]] = int(
                    rng.choice(nu_bin, size=None, p=p_bin))
            else:
                # Initialize arrays to define chain segment number
                # intervals and store indices for each chain in the bin
                nu_min_chn_bin = np.empty(num_chns_bin, dtype=int)
                nu_max_chn_bin = np.empty(num_chns_bin, dtype=int)
                chn_bin = np.empty(num_chns_bin, dtype=int)
                # Calculate the chain segment number intervals for each
                # chain in the bin via performing a cumulative summation
                # on the normalized polymer chain segment probability
                # distribution and multiplying that by the number of
                # chains in the bin
                chn_bin_wrt_nu_bin = (
                    np.floor(np.cumsum(p_bin)*num_chns_bin).astype(int)
                )
                # Correct the last entry
                chn_bin_wrt_nu_bin[-1] = int(chn_bin_wrt_nu_bin[-2])
                # Determine the chain segment number intervals for each
                # chain in the bin
                for chn in range(num_chns_bin):
                    nu_chn_bin_indcs = np.where(chn_bin_wrt_nu_bin == chn)[0]
                    # Chain appears in the bin
                    if np.shape(nu_chn_bin_indcs)[0] > 0:
                        nu_chn_bin = nu_chn_bin_indcs + nu_min
                        nu_min_chn_bin[chn] = int(nu_chn_bin[0])
                        nu_max_chn_bin[chn] = int(nu_chn_bin[-1])
                    # chain_1, chain_2, chain_3, etc., do not appear in
                    # the bin due to the cumulative summation of the
                    # normalized polymer chain segment probability
                    # distribution skipping over their number, so simply
                    # adopt the chain segment number interval from the
                    # previous chain
                    elif chn > 0:
                        nu_min_chn_bin[chn] = int(nu_min_chn_bin[chn-1])
                        nu_max_chn_bin[chn] = int(nu_max_chn_bin[chn-1])
                # Handle the case of chain_0 (or chain_0 and chain_1 and
                # so on) being ostensibly excluded from the cumulative
                # summation of the normalized polymer chain segment
                # probability distribution
                min_chn_bin = int(chn_bin_wrt_nu_bin[0])
                if min_chn_bin > 0:
                    for chn in range(min_chn_bin):
                        nu_min_chn_bin[chn] = int(nu_min_chn_bin[min_chn_bin])
                        nu_max_chn_bin[chn] = int(nu_max_chn_bin[min_chn_bin])
                
                # Assign the chains in the bin to the chains in the
                # polymer network. In so doing, explicitly enforce the
                # physically-constrained minimum chain segment number to
                # the chain segment number interval for each chain in
                # the bin.
                for chn in range(num_chns_bin):
                    # Determine the chains in the bin whose
                    # physically-constrained minimum chain segment
                    # number is less than or equal to the probability
                    # distribution-governed minimum chain segment number
                    psbl_chns_indcs = (
                        np.where(nu_min_bin <= int(nu_min_chn_bin[chn]))[0]
                    )
                    # Address the case where no chains in the bin have a
                    # physically-constrained minimum chain segment
                    # number that is less than or equal to the
                    # probability distribution-governed minimum chain
                    # segment number
                    if np.shape(psbl_chns_indcs)[0] == 0:
                        # Reset the probability distribution-governed
                        # minimum chain segment number to equal the
                        # smallest physically-constrained minimum chain
                        # segment number in the bin
                        min_nu_min_bin = int(np.min(nu_min_bin))
                        nu_min_chn_bin[chn] = min_nu_min_bin
                        # If the reset (and now increased)
                        # probability distribution-governed minimum
                        # chain segment number is now greater than its
                        # maximum counterpart, then reset the latter to
                        # equal the former
                        if nu_min_chn_bin[chn] > nu_max_chn_bin[chn]:
                            nu_max_chn_bin[chn] = int(nu_min_chn_bin[chn])
                        # Determine the chains in the bin who exhibit
                        # the smallest physically-constrained minimum
                        # chain segment number
                        psbl_chns_indcs = (
                            np.where(nu_min_bin == min_nu_min_bin)[0]
                        )
                    # Properly select the chain index from the candidate
                    # chains
                    if np.shape(psbl_chns_indcs)[0] == 1:
                        chn_indx = int(psbl_chns_indcs[0])
                    else:
                        chn_indx = rng.integers(
                            np.shape(psbl_chns_indcs)[0], dtype=int)
                        chn_indx = int(psbl_chns_indcs[chn_indx])
                    # Save the chain index
                    chn_bin[chn] = int(chn_bin_indcs[chn_indx])
                    # Update the indices and minimum chain segment
                    # number for all of the chains in the bin
                    chn_bin_indcs = np.delete(chn_bin_indcs, chn_indx, axis=0)
                    nu_min_bin = np.delete(nu_min_bin, chn_indx, axis=0)

                # Assign a chain segment number to each bin chain in the
                # polymer network
                for chn in range(num_chns_bin):
                    nu_min_chn = int(nu_min_chn_bin[chn])
                    nu_max_chn = int(nu_max_chn_bin[chn])
                    # Address the case when the minimum and maximum
                    # chain segment numbers are equal to one another
                    if nu_min_chn == nu_max_chn:
                        nu_chn = nu_min_chn
                        conn_nu_edges[chn_bin[chn]] = int(nu_chn)
                    # Address the case when the minimum and maximum
                    # chain segment numbers are not equal to one another
                    else:
                        # Determine chain segment number interval 
                        nu_chn_bin = (
                            nu_bin[nu_min_chn-nu_min:nu_max_chn+1-nu_min]
                        )
                        # Extract and normalize polymer chain segment 
                        # probability distribution
                        p_chn_bin = p_bin[nu_min_chn-nu_min:nu_max_chn+1-nu_min]
                        Z_p_chn_bin = np.sum(p_chn_bin, dtype=float)
                        p_chn_bin /= Z_p_chn_bin
                        p_chn_bin[np.isnan(p_chn_bin)] = 0.
                        Z_p_chn_bin = np.sum(p_chn_bin, dtype=float)
                        if Z_p_chn_bin == 0:
                            p_chn_bin = (
                                np.ones(np.shape(p_chn_bin)[0])
                                / np.shape(p_chn_bin)[0]
                            )
                        else:
                            p_chn_bin /= Z_p_chn_bin
                        # Randomly select a chain segment number, and
                        # assign this to the appropriate chain in the
                        # polymer network
                        conn_nu_edges[chn_bin[chn]] = int(
                            rng.choice(nu_chn_bin, size=None, p=p_chn_bin))
    
    # Return the chain segment numbers for core and periodic edge
    # chains, respectively
    return conn_nu_edges[:conn_core_m], conn_nu_edges[conn_core_m:]

def aelp_network_topology_initialization(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        b: float,
        xi: float,
        k: int,
        n: int,
        nu: int,
        nu_max: int,
        config: int,
        max_try: int) -> None:
    """Network topology initialization procedure for artificial
    end-linked polymer networks.

    This function loads the simulation box size and the core
    cross-linker coordinates. Then, this function initializes and saves
    the topology of an artificial end-linked polymer network via a
    modified Gusev-Hanson protocol (which is a Monte Carlo procedure).

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, either "auelp" or "apelp" are applicable (corresponding to artificial uniform end-linked polymer networks ("auelp") or artificial polydisperse end-linked polymer networks ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Chain segment and/or cross-linker diameter.
        xi (float): Chain-to-cross-link connection probability.
        k (int): Maximum cross-linker degree/functionality; either 3, 4, 5, 6, 7, or 8.
        n (int): Number of core cross-linkers.
        nu (int): (Average) Number of segments per chain.
        nu_max (int): Maximum number of segments that could possibly be assigned to a chain.
        config (int): Configuration number.
        max_try (int): Maximum number of dangling chain instantiation attempts.
    
    """
    # Load simulation box size
    L = np.loadtxt(L_filename_str(network, date, batch, sample))

    # Generate configuration filename prefix. This establishes the
    # configuration filename prefix as the filename prefix associated
    # with artificial end-linked polymer network data files, which is
    # reflected in the aelp_filename_str() function.
    config_filename_prefix = config_filename_str(
        network, date, batch, sample, config)

    # Generate filenames
    coords_filename = config_filename_prefix + ".coords"
    mx_cmp_core_node_type_filename = (
        config_filename_prefix + "-node_type" + ".dat"
    )
    mx_cmp_conn_core_edges_filename = (
        config_filename_prefix + "-conn_core_edges" + ".dat"
    )
    mx_cmp_conn_pb_edges_filename = (
        config_filename_prefix + "-conn_pb_edges" + ".dat"
    )
    mx_cmp_coords_filename = config_filename_prefix + ".coords"
    conn_nu_core_edges_filename = (
        config_filename_prefix + "-conn_nu_core_edges" + ".dat"
    )
    conn_nu_pb_edges_filename = (
        config_filename_prefix + "-conn_nu_pb_edges" + ".dat"
    )

    # Call appropriate helper function to initialize network topology
    if (scheme == "random") or (scheme == "prhd") or (scheme == "pdhu"):
        # Load core cross-linker coordinates
        coords = np.loadtxt(coords_filename)
    elif scheme == "lammps":
        skiprows_num = 15
        # Load core cross-linker coordinates
        coords = np.loadtxt(coords_filename, skiprows=skiprows_num, max_rows=n)
    
    # Actual number of core cross-linkers
    n = np.shape(coords)[0]
    
    # Calculate the stoichiometric number of chains
    m = m_arg_stoich_func(n, k)

    # As a fail-safe check, force int-valued parameters to be ints
    k = int(np.floor(k))
    n = int(np.floor(n))
    nu = int(np.floor(nu))
    nu_max = int(np.floor(nu_max))
    max_try = int(np.floor(max_try))
    m = int(np.floor(m))

    # Core cross-linker nodes
    core_nodes = np.arange(n, dtype=int)

    # Identify core nodes as cross-linkers
    core_node_type = np.ones(n, dtype=int)

    # Tessellate the core node coordinates and construct the
    # pb2core_nodes np.ndarray
    tsslltd_coords, pb2core_nodes = core_node_tessellation(
        dim, core_nodes, coords, L)

    # Maximal core cross-linker sampling neighborhood radius as per the
    # minimum image criterion 
    r_mic = L / 2.
    # Polymer chain contour length
    l_cntr = nu * b
    # Determine the core cross-linker sampling neighborhood radius
    r_nghbrhd = np.min(np.asarray([r_mic, l_cntr]))

    # Finely discretize the core cross-linker sampling neighborhood
    # radius
    r_nghbrhd_chns = np.linspace(0, r_nghbrhd, 10001)
    # Depending on the type of artificial end-linked polymer network,
    # calculate the polymer chain probability distribution
    p_nghbrhd_chns = np.asarray([])
    if network == "auelp":
        p_nghbrhd_chns = p_gaussian_cnfrmtn_func(b, nu, r_nghbrhd_chns)
    elif network == "apelp":
        p_nghbrhd_chns = p_rel_net_gaussian_cnfrmtn_func(b, nu, r_nghbrhd_chns)
    # Normalize the polymer chain probability distribution
    p_nghbrhd_chns /= np.sum(p_nghbrhd_chns, dtype=float)

    # Initialize core cross-linker sampling neighborhood list
    core_nodes_nghbrhd_init = []
    r_core_nodes_nghbrhd_init = []

    # Determine core cross-linker sampling neighborhood
    for node in np.nditer(core_nodes):
        core_node = coords[int(node)]

        # Downselect the local orb neighborhood with radius r_nghbrhd
        # about the core node cross-linker
        orb_nghbr_indcs, orb_nghbr_num = orb_neighborhood_id(
            dim, tsslltd_coords, core_node, r_nghbrhd, inclusive=True,
            indices=True)
        
        if orb_nghbr_num > 0:
            # Calculate the distance between the core node cross-linker
            # and its local orb neighbors
            orb_nghbrs = tsslltd_coords[orb_nghbr_indcs]
            r_orb_nghbrs = np.asarray(
            [
                np.linalg.norm(core_node-orb_nghbrs[orb_nghbr_indx])
                for orb_nghbr_indx in range(orb_nghbr_num)
            ])

            # Add the cross-linker neighbor nodes array to the core
            # cross-linker sampling neighborhood list, and add the array
            # of the distance between the core node cross-linker
            # and its local orb neighbors to the core cross-linker
            # sampling neighborhood distance list
            core_nodes_nghbrhd_init.append(orb_nghbr_indcs)
            r_core_nodes_nghbrhd_init.append(r_orb_nghbrs)
        else:
            core_nodes_nghbrhd_init.append(np.asarray([]))
            r_core_nodes_nghbrhd_init.append(np.asarray([]))
    
    # Retain unique nodes from the core and periodic boundary
    # cross-linkers in the core cross-linker sampling neighborhood list
    core_pb_nodes = np.unique(
        np.concatenate(
            tuple(nghbrhd for nghbrhd in core_nodes_nghbrhd_init), dtype=int))
    
    # Construct an np.ndarray that returns the index for each node
    # number in the core_pb_nodes np.ndarray
    core_pb_nodes_indcs = -1 * np.ones(np.max(core_pb_nodes)+1, dtype=int)
    core_pb_nodes_indcs[core_pb_nodes] = np.arange(
        np.shape(core_pb_nodes)[0], dtype=int)

    # Extract the core and periodic boundary cross-linker coordinates
    # using the corresponding node numbers
    core_pb_coords = tsslltd_coords[core_pb_nodes].copy()

    del tsslltd_coords

    # Extract the core and periodic boundary cross-linker nodes in the
    # pb2core_nodes np.ndarray
    pb2core_nodes = pb2core_nodes[core_pb_nodes].copy()

    # Construct the core2pb_nodes list
    core2pb_nodes = core2pb_nodes_func(core_nodes, pb2core_nodes)

    # Refactor the node numbers in the core cross-linker sampling
    # neighborhood list
    for core_node in np.nditer(core_nodes):
        core_node = int(core_node)
        nghbr_nodes = core_nodes_nghbrhd_init[core_node]
        for nghbr_node_indx in range(np.shape(nghbr_nodes)[0]):
            nghbr_nodes[nghbr_node_indx] = int(
                core_pb_nodes_indcs[nghbr_nodes[nghbr_node_indx]])
        core_nodes_nghbrhd_init[core_node] = nghbr_nodes
    
    # Initialize tracker for failure of dangling chain creation
    dnglng_chn_fail = False

    # Initialize random number generator
    rng = np.random.default_rng()

    # Initialize dangling chain free end node coordinates list
    core_dnglng_chns_coords = []

    # Initialize edge lists
    conn_core_edges = []
    conn_pb_edges = []
    conn_core_dnglng_edges = []
    conn_pb_dnglng_edges = []

    # Initialize dangling chain node and edge numbers
    core_dnglng_m = 0
    pb_dnglng_m = 0
    dnglng_n = -1

    # Network topology initialization
    while True:
        # Initialize anti-degree of the core cross-linker nodes
        core_nodes_anti_k = k*np.ones(n, dtype=int)
        # Initialize core cross-linker node active sites
        core_nodes_active_sites = np.repeat(core_nodes, k)
        core_nodes_active_sites_num = np.shape(core_nodes_active_sites)[0]

        # Initialize core cross-linker sampling neighborhood list
        core_nodes_nghbrhd = core_nodes_nghbrhd_init.copy()
        r_core_nodes_nghbrhd = r_core_nodes_nghbrhd_init.copy()
        
        # Initialize cross-linker node coordinates for dangling chains
        core_pb_dnglng_chns_coords = core_pb_coords.copy()

        # Initialize dangling chain free end node coordinates list
        core_dnglng_chns_coords = []

        # Initialize edge array and lists
        edges = np.full((m, 2), np.inf)
        conn_core_edges = []
        conn_pb_edges = []
        conn_core_dnglng_edges = []
        conn_pb_dnglng_edges = []

        # Initialize dangling chain node and edge numbers
        core_dnglng_m = 0
        pb_dnglng_m = 0
        dnglng_n = -1

        # Initialize node number integer constants
        core_node = 0
        nghbr_node = 0

        # Initialize and randomly shuffle chain end numbers
        chn_ends = np.arange(2*m, dtype=int)
        rng.shuffle(chn_ends)

        # Initialize network topology on a chain end-by-chain end basis
        for chn_end in np.nditer(chn_ends):
            # Chain end is a free chain end
            if rng.random() > xi: continue
            else:
                # Identify the chain and its ends
                chn_end = int(chn_end)
                chn = int(np.floor(chn_end/2))
                chn_end_0 = int(chn_end%2)
                chn_end_1 = 1 - chn_end_0
                # If this chain has not yet been instantiated, then
                # randomly select an active site from the core nodes to
                # instantiate the chain
                if (edges[chn, 0] == np.inf) and (edges[chn, 1] == np.inf):
                    core_node_active_site_indx = rng.integers(
                        np.shape(core_nodes_active_sites)[0], dtype=int)
                    core_node = (
                        core_nodes_active_sites[core_node_active_site_indx]
                    )
                    # Instantiate the chain at one end
                    edges[chn, chn_end_0] = core_node
                    # Update inactive sites, anti-degree, active core
                    # cross-linker nodes, and the sampling neighborhood
                    # list for the parent core cross-linker node
                    (core_nodes_active_sites, core_nodes_active_sites_num, 
                     core_nodes_anti_k, core_nodes_nghbrhd,
                     r_core_nodes_nghbrhd) = core_node_update_func(
                         core_node, core_node_active_site_indx,
                         core_nodes_active_sites, core_nodes_active_sites_num,
                         core_nodes_anti_k, core_nodes, core2pb_nodes,
                         core_nodes_nghbrhd, r_core_nodes_nghbrhd)
                # Otherwise, the chain has previously been instantiated
                else:
                    # Extract sampling neighborhood about the
                    # previously-instantiated chain end
                    core_node = int(edges[chn, chn_end_1])
                    nghbr_nodes = core_nodes_nghbrhd[core_node]
                    nghbr_nodes_num = np.shape(nghbr_nodes)[0]
                    r_nghbr_nodes = r_core_nodes_nghbrhd[core_node]
                    # Check if the sampling neighborhood is empty. If
                    # so, then the chain is a dangling chain.
                    if nghbr_nodes_num == 0:
                        continue
                    else:
                        # Check if there is only one neighbor in the
                        # sampling neighborhood
                        if nghbr_nodes_num == 1:
                            nghbr_node = nghbr_nodes[0]
                        else:
                            # Calculate and normalize weighting factors
                            p_nghbrhd = np.asarray([])
                            if network == "auelp":
                                p_nghbrhd = p_gaussian_cnfrmtn_func(
                                    b, nu, r_nghbr_nodes)
                            elif network == "apelp":
                                p_nghbrhd = np.empty(nghbr_nodes_num)
                                for nghbr_node_indx in range(nghbr_nodes_num):
                                    r_nghbr_node = (
                                        r_nghbr_nodes[nghbr_node_indx]
                                    )
                                    if r_nghbr_node == 0.0:
                                        p_nghbrhd[nghbr_node_indx] = 0.0
                                    else:
                                        # Linear interpolation
                                        r_nghbrhd_chns_diff = (
                                            r_nghbrhd_chns - r_nghbr_node
                                        )
                                        r_nghbrhd_chns_diff[r_nghbrhd_chns_diff < 0] = (
                                            np.inf
                                        )
                                        indx_right = np.argmin(
                                            r_nghbrhd_chns_diff)
                                        r_right = r_nghbrhd_chns[indx_right]
                                        p_right = p_nghbrhd_chns[indx_right]
                                        indx_left = indx_right - 1
                                        r_left = r_nghbrhd_chns[indx_left]
                                        p_left = p_nghbrhd_chns[indx_left]
                                        p_nghbrhd[nghbr_node_indx] = (
                                            p_left + (r_nghbr_node-r_left)
                                            * (p_right-p_left) / (r_right-r_left)
                                        )
                            p_nghbrhd /= np.sum(p_nghbrhd, dtype=float)
                            # Randomly select a neighbor cross-linker
                            # node to host the other end of the chain
                            nghbr_node = int(
                                rng.choice(nghbr_nodes, size=None, p=p_nghbrhd))
                        # Instantiate the other end of the chain
                        edges[chn, chn_end_0] = nghbr_node
                        core_node = int(pb2core_nodes[nghbr_node])
                        core_node_active_site_indx = int(
                            np.where(core_nodes_active_sites == core_node)[0][0])
                        # Update inactive sites, anti-degree, active
                        # core cross-linker nodes, and the sampling
                        # neighborhood list for the parent core
                        # cross-linker node
                        (core_nodes_active_sites, core_nodes_active_sites_num,
                         core_nodes_anti_k, core_nodes_nghbrhd,
                         r_core_nodes_nghbrhd) = core_node_update_func(
                             core_node, core_node_active_site_indx,
                             core_nodes_active_sites, core_nodes_active_sites_num,
                             core_nodes_anti_k, core_nodes, core2pb_nodes,
                             core_nodes_nghbrhd, r_core_nodes_nghbrhd)
        
        # Post-process the edges array (that was populated during the
        # network topology initialization) on a chain-by-chain basis
        # into edges lists
        for chn in range(m):
            # Chain is a core edge
            if (edges[chn, 0] < n) and (edges[chn, 1] < n):
                conn_core_edges.append(
                    (int(edges[chn, 0]), int(edges[chn, 1])))
            # Chain is a periodic edge
            elif (edges[chn, 0] < n) and (n <= edges[chn, 1] < np.inf):
                conn_pb_edges.append(
                    (int(edges[chn, 0]), int(pb2core_nodes[int(edges[chn, 1])])))
            elif (n <= edges[chn, 0] < np.inf) and (edges[chn, 1] < n):
                conn_pb_edges.append(
                    (int(pb2core_nodes[int(edges[chn, 0])]), int(edges[chn, 1])))
            # Chain is a dangling chain
            elif (edges[chn, 0] < n) and (edges[chn, 1] == np.inf):
                (dnglng_chn_fail, dnglng_n, core_dnglng_m, pb_dnglng_m,
                 conn_core_dnglng_edges, conn_pb_dnglng_edges,
                 core_dnglng_chns_coords, core_pb_dnglng_chns_coords) = dangling_chains_update_func(
                     int(edges[chn, 0]), max_try, rng, r_nghbrhd_chns,
                     p_nghbrhd_chns, dim, b, L, dnglng_chn_fail, dnglng_n,
                     core_dnglng_m, pb_dnglng_m, conn_core_dnglng_edges,
                     conn_pb_dnglng_edges, core_dnglng_chns_coords,
                     core_pb_dnglng_chns_coords)
                # Break if a dangling chain failed to be instantiated
                if dnglng_chn_fail == True: break
            elif (edges[chn, 0] == np.inf) and (edges[chn, 1] < n):
                (dnglng_chn_fail, dnglng_n, core_dnglng_m, pb_dnglng_m,
                 conn_core_dnglng_edges, conn_pb_dnglng_edges,
                 core_dnglng_chns_coords, core_pb_dnglng_chns_coords) = dangling_chains_update_func(
                     int(edges[chn, 1]), max_try, rng, r_nghbrhd_chns,
                     p_nghbrhd_chns, dim, b, L, dnglng_chn_fail, dnglng_n,
                     core_dnglng_m, pb_dnglng_m, conn_core_dnglng_edges,
                     conn_pb_dnglng_edges, core_dnglng_chns_coords,
                     core_pb_dnglng_chns_coords)
                # Break if a dangling chain failed to be instantiated
                if dnglng_chn_fail == True: break
            # Chain is a free chain
            elif (edges[chn, 0] == np.inf) and (edges[chn, 1] == np.inf):
                continue

        # Restart the network topology initialization protocol if a
        # dangling chain failed to be instantiated
        if dnglng_chn_fail == True: continue
        
        del core_nodes_anti_k, core_nodes_active_sites
        del core_pb_dnglng_chns_coords
        del edges, chn_ends
        # Break out of acceptable initialized topology 
        break

    del core_pb_nodes, pb2core_nodes, core2pb_nodes
    del core_pb_coords
    del core_nodes_nghbrhd_init, core_nodes_nghbrhd
    del r_core_nodes_nghbrhd_init, r_core_nodes_nghbrhd
    del r_nghbrhd_chns, p_nghbrhd_chns

    # Refactor edge lists to np.ndarrays
    conn_core_edges = np.asarray(conn_core_edges, dtype=int)
    conn_pb_edges = np.asarray(conn_pb_edges, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)
    
    # Recalibrate number of dangling chains
    dnglng_n += 1

    # No dangling chains were added to the network
    if dnglng_n == 0: pass
    # Only core dangling chains were added to the network
    elif core_dnglng_m > 0 and pb_dnglng_m == 0:
        # Update core dangling chain node numbers
        for edge in conn_core_dnglng_edges: edge[1] += n
        # Update edge lists
        conn_core_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_core_dnglng_edges), dtype=int)
        conn_core_edges = np.vstack(
            (conn_core_edges, conn_core_dnglng_edges), dtype=int)
        conn_edges = np.vstack((conn_edges, conn_core_dnglng_edges), dtype=int)
    # Only periodic boundary dangling chains were added to the network
    elif core_dnglng_m == 0 and pb_dnglng_m > 0:
        # Update periodic boundary dangling chain node numbers
        for edge in conn_pb_dnglng_edges: edge[1] += n
        # Update edge lists
        conn_pb_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_pb_dnglng_edges), dtype=int)
        conn_pb_edges = np.vstack(
            (conn_pb_edges, conn_pb_dnglng_edges), dtype=int)
        conn_edges = np.vstack((conn_edges, conn_pb_dnglng_edges), dtype=int)
    # Both core and periodic boundary dangling chains were added to the
    # network
    else:
        # Update core and periodic boundary dangling chain node numbers
        for edge in conn_core_dnglng_edges: edge[1] += n
        for edge in conn_pb_dnglng_edges: edge[1] += n
        # Update edge lists
        conn_core_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_core_dnglng_edges), dtype=int)
        conn_pb_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_pb_dnglng_edges), dtype=int)
        conn_core_edges = np.vstack(
            (conn_core_edges, conn_core_dnglng_edges), dtype=int)
        conn_pb_edges = np.vstack(
            (conn_pb_edges, conn_pb_dnglng_edges), dtype=int)
        conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)
    # Update core_node_type correspondingly to end-linked polymer
    # network code
    core_node_type = np.concatenate(
        (core_node_type, np.repeat(3, dnglng_n)), dtype=int)
    # Add core coordinates from dangling chains
    coords = np.vstack((coords, np.asarray(core_dnglng_chns_coords)))
    # Update core_nodes
    core_nodes = np.arange(n+dnglng_n, dtype=int)
    
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

    # Isolate largest/maximum connected component in a nodewise fashion
    mx_cmp_conn_graph_nodes = max(nx.connected_components(conn_graph), key=len)
    # Extract largest/maximum connected component subgraphs
    mx_cmp_conn_core_graph = (
        conn_core_graph.subgraph(mx_cmp_conn_graph_nodes).copy()
    )
    mx_cmp_conn_pb_graph = (
        conn_pb_graph.subgraph(mx_cmp_conn_graph_nodes).copy()
    )
    # Extract edges
    mx_cmp_conn_core_graph_edges = np.asarray(
        list(mx_cmp_conn_core_graph.edges()), dtype=int)
    mx_cmp_conn_pb_graph_edges = np.asarray(
        list(mx_cmp_conn_pb_graph.edges()), dtype=int)
    # Number of edges in the largest/maximum connected component
    mx_cmp_conn_core_graph_m = np.shape(mx_cmp_conn_core_graph_edges)[0]
    mx_cmp_conn_pb_graph_m = np.shape(mx_cmp_conn_pb_graph_edges)[0]
    # Nodes from the largest/maximum connected component, sorted in
    # ascending order
    mx_cmp_conn_graph_nodes = np.sort(
        np.fromiter(mx_cmp_conn_graph_nodes, dtype=int))
    # Construct an np.ndarray that returns the index for each node
    # number in the mx_cmp_conn_graph_nodes np.ndarray
    mx_cmp_conn_graph_nodes_indcs = (
        -1 * np.ones(np.max(mx_cmp_conn_graph_nodes)+1, dtype=int)
    )
    mx_cmp_conn_graph_nodes_indcs[mx_cmp_conn_graph_nodes] = np.arange(
        np.shape(mx_cmp_conn_graph_nodes)[0], dtype=int)

    # Isolate core_node_type for the largest/maximum connected component
    mx_cmp_core_node_type = core_node_type[mx_cmp_conn_graph_nodes]
    # Isolate the cross-linker coordinates for the largest/maximum
    # connected component
    mx_cmp_coords = coords[mx_cmp_conn_graph_nodes]

    # Update all original node values with updated node values
    for edge in range(mx_cmp_conn_core_graph_m):
        mx_cmp_conn_core_graph_edges[edge, 0] = int(
            mx_cmp_conn_graph_nodes_indcs[mx_cmp_conn_core_graph_edges[edge, 0]])
        mx_cmp_conn_core_graph_edges[edge, 1] = int(
            mx_cmp_conn_graph_nodes_indcs[mx_cmp_conn_core_graph_edges[edge, 1]])
    for edge in range(mx_cmp_conn_pb_graph_m):
        mx_cmp_conn_pb_graph_edges[edge, 0] = int(
            mx_cmp_conn_graph_nodes_indcs[mx_cmp_conn_pb_graph_edges[edge, 0]])
        mx_cmp_conn_pb_graph_edges[edge, 1] = int(
            mx_cmp_conn_graph_nodes_indcs[mx_cmp_conn_pb_graph_edges[edge, 1]])
    
    # Save fundamental graph constituents
    np.savetxt(mx_cmp_core_node_type_filename, mx_cmp_core_node_type, fmt="%d")
    np.savetxt(
        mx_cmp_conn_core_edges_filename, mx_cmp_conn_core_graph_edges, fmt="%d")
    np.savetxt(
        mx_cmp_conn_pb_edges_filename, mx_cmp_conn_pb_graph_edges, fmt="%d")
    
    # Save the core node coordinates
    np.savetxt(mx_cmp_coords_filename, mx_cmp_coords)

    # Assign a chain segment number to each chain in the artificial
    # polydisperse end-linked polymer network
    if network == "apelp":
        # Acquire fundamental graph constituents
        core_nodes = np.arange(np.shape(mx_cmp_conn_graph_nodes)[0], dtype=int)
        conn_core_edges = mx_cmp_conn_core_graph_edges.copy()
        conn_core_m = mx_cmp_conn_core_graph_m
        conn_pb_edges = mx_cmp_conn_pb_graph_edges.copy()
        conn_pb_m = mx_cmp_conn_pb_graph_m
        conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)
        coords = mx_cmp_coords.copy()

        # Create nx.MultiGraphs and add nodes before edges
        conn_core_graph = nx.MultiGraph()
        conn_core_graph = add_nodes_from_numpy_array(
            conn_core_graph, core_nodes)
        conn_core_graph = add_edges_from_numpy_array(
            conn_core_graph, conn_core_edges)
        
        conn_pb_graph = nx.MultiGraph()
        conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
        conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)
        
        conn_graph = nx.MultiGraph()
        conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
        conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

        # Calculate end-to-end chain length (Euclidean edge length)
        r_chn = l_func(conn_core_graph, conn_pb_graph, conn_graph, coords, L)
        
        # Assign a chain segment number to each chain
        conn_nu_core_edges, conn_nu_pb_edges = apelp_network_nu_assignment(
            rng, b, nu, nu_max, conn_core_m, conn_pb_m, r_chn)
        
        # Save the chain segment numbers
        np.savetxt(conn_nu_core_edges_filename, conn_nu_core_edges, fmt="%d")
        np.savetxt(conn_nu_pb_edges_filename, conn_nu_pb_edges, fmt="%d")

def aelp_network_topology(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        b: float,
        xi: float,
        k: int,
        n: int,
        nu: int,
        nu_max: int,
        config: int,
        max_try: int) -> None:
    """Artificial end-linked polymer network topology.

    This function confirms that the network being called for is an
    artificial end-linked polymer network. Then, the function calls the
    artificial end-linked polymer network initialization function to
    create the artificial end-linked polymer network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, either "auelp" or "apelp" are applicable (corresponding to artificial uniform end-linked polymer networks ("auelp") or artificial polydisperse end-linked polymer networks ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Chain segment and/or cross-linker diameter.
        xi (float): Chain-to-cross-link connection probability.
        k (int): Maximum cross-linker degree/functionality; either 3, 4, 5, 6, 7, or 8.
        n (int): Number of core cross-linkers.
        nu (int): (Average) Number of segments per chain.
        nu_max (int): Maximum number of segments that could possibly be assigned to a chain.
        config (int): Configuration number.
        max_try (int): Maximum number of dangling chain instantiation attempts.
    
    """
    # Network topology initialization procedure is only applicable for
    # artificial end-linked polymer networks. Exit if a different type
    # of network is passed.
    if network != "auelp":
        if network != "apelp":
            error_str = (
                "Network topology initialization procedure is only "
                + "applicable for artificial end-linked polymer "
                + "networks. This calculation will only proceed if "
                + "network = ``auelp'' or network = ``apelp''."
            )
            print(error_str)
            return None
    aelp_network_topology_initialization(
        network, date, batch, sample, scheme, dim, b, xi, k, n, nu, nu_max,
        config, max_try)

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
    """Spider web-inspired Delaunay-triangulated network topological
    descriptor.
    
    This function extracts a Spider web-inspired Delaunay-triangulated
    network and sets a variety of input parameters corresponding to a
    particular topological descriptor (and numpy function) of interest.
    These are then passed to the master network_topological_descriptor()
    function, which calculates (and, if called for, saves) the result of
    the topological descriptor for the Spider web-inspired
    Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider-web inspired Delaunay-triangulated networks ("swidt")).
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
    if network != "auelp":
        if network != "apelp":
            error_str = (
                "This topological descriptor calculation is only "
                + "applicable for data files associated with "
                + "artificial end-linked polymer networks. This "
                + "calculation will only proceed if "
                + "network = ``auelp'' or network = ``apelp''."
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