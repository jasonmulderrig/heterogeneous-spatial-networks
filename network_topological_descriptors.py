import numpy as np
import networkx as nx
from graph_utils import (
    largest_connected_component,
    elastically_effective_end_linked_graph
)
import nodal_degree_topological_descriptors
import shortest_path_topological_descriptors
import general_topological_descriptors

def network_topological_descriptor(
        tplgcl_dscrptr: str,
        np_oprtn: str,
        conn_core_graph: nx.Graph | nx.MultiGraph,
        conn_pb_graph: nx.Graph | nx.MultiGraph,
        conn_graph: nx.Graph | nx.MultiGraph,
        coords: np.ndarray,
        L: float,
        length_bound: int,
        eeel_ntwrk: bool,
        tplgcl_dscrptr_result_filename: str,
        save_tplgcl_dscrptr_result: bool,
        return_tplgcl_dscrptr_result: bool) -> np.ndarray | float | int | None:
    """Network topological descriptor.
    
    This function calculates the result of a topological descriptor for
    a supplied network. If called for, the elastically-effective
    end-linked network in the supplied network will be extracted for the
    topological descriptor calculation. Additionally, if called for, a
    numpy function will operate on the result of the topological
    descriptor calculation. Finally, if called for, the resulting
    topological descriptor will be saved and returned.

    Args:
        tplgcl_dscrptr (str): Topological descriptor name.
        np_oprtn (str): numpy function/operation name.
        conn_core_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the core edges from the graph capturing the periodic connections between the core nodes.
        conn_pb_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that represents the periodic boundary edges from the graph capturing the periodic connections between the core nodes.
        conn_graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph that captures the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (float): Simulation box size.
        length_bound (int): Maximum ring order (inclusive).
        eeel_ntwrk (bool): Boolean indicating if the elastically-effective end-linked network in the supplied network ought to be extracted.
        tplgcl_dscrptr_result_filename (str): Filename for the topological descriptor result.
        save_tplgcl_dscrptr_result (bool): Boolean indicating if the topological descriptor result ought to be saved.
        return_tplgcl_dscrptr_result (bool): Boolean indicating if the topological descriptor result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Topological descriptor result.
    
    """
    # Modify topological descriptor string to match function name
    # convention
    tplgcl_dscrptr_func_str = tplgcl_dscrptr + "_func"

    # Probe each topological descriptors module to identify the
    # topological descriptor calculation function
    if hasattr(nodal_degree_topological_descriptors, tplgcl_dscrptr_func_str):
        tplgcl_dscrptr_func = getattr(
            nodal_degree_topological_descriptors, tplgcl_dscrptr_func_str)
    elif hasattr(shortest_path_topological_descriptors, tplgcl_dscrptr_func_str):
        tplgcl_dscrptr_func = getattr(
            shortest_path_topological_descriptors, tplgcl_dscrptr_func_str)
    elif hasattr(general_topological_descriptors, tplgcl_dscrptr_func_str):
        tplgcl_dscrptr_func = getattr(
            general_topological_descriptors, tplgcl_dscrptr_func_str)
    else:
        error_str = (
            "The topological descriptor ``" + tplgcl_dscrptr + "'' is "
            + "not implemented!"
        )
        print(error_str)
        return None
    
    # Extract elastically-effective end-linked network
    if eeel_ntwrk == True:
        # The elastically-effective end-linked network function returns
        # the largest connected component
        conn_graph = elastically_effective_end_linked_graph(conn_graph)
        if (tplgcl_dscrptr == "prop_eeel_n") or (tplgcl_dscrptr == "prop_eeel_m"):
            error_str = (
                "Only the original conn_graph ought to be supplied "
                + "to the prop_eeel_n and prop_eeel_m topological "
                + "descriptor functions. Please correct this and "
                + "rerun this calculation after doing so."
            )
            print(error_str)
            return None
    # Extract largest connected component from network if the
    # elastically-effective end-linked network is not called for
    else: conn_graph = largest_connected_component(conn_graph)
    # Complete the extraction of the largest connected component
    conn_core_graph = conn_core_graph.subgraph(list(conn_graph.nodes())).copy()
    conn_pb_graph = conn_pb_graph.subgraph(list(conn_graph.nodes())).copy()
    
    # Remove isolate nodes
    conn_graph.remove_nodes_from(list(nx.isolates(conn_graph)))
    conn_core_graph.remove_nodes_from(list(nx.isolates(conn_core_graph)))
    conn_pb_graph.remove_nodes_from(list(nx.isolates(conn_pb_graph)))
    
    # Deploy topological descriptor calculation function and carefully
    # handle the input parameter set in the process
    if ((tplgcl_dscrptr == "l") or (tplgcl_dscrptr == "l_cmpnts")
        or (tplgcl_dscrptr == "scc")):
        tplgcl_dscrptr_result = tplgcl_dscrptr_func(
            conn_core_graph, conn_pb_graph, conn_graph, coords, L)
    elif tplgcl_dscrptr == "h":
        tplgcl_dscrptr_result = tplgcl_dscrptr_func(
            conn_graph, length_bound, coords, L)
    else:
        tplgcl_dscrptr_result = tplgcl_dscrptr_func(conn_graph)

    # Probe the numpy module to identify the numpy function to send the
    # topological descriptor result to (if called for)
    if np_oprtn == "": pass
    elif hasattr(np, np_oprtn):
        # Deploy numpy function and carefully handle the input parameter
        # set in the process
        np_func = getattr(np, np_oprtn)
        if tplgcl_dscrptr == "l_cmpnts":
            tplgcl_dscrptr_result = np_func(tplgcl_dscrptr_result, axis=0)
        else:
            tplgcl_dscrptr_result = np_func(tplgcl_dscrptr_result)
    else:
        error_str = "The numpy function ``" + np_oprtn + "'' does not exist!"
        print(error_str)
        return None
    
    # Save topological descriptor result (if called for)
    if save_tplgcl_dscrptr_result:
        if isinstance(tplgcl_dscrptr_result, int):
            np.savetxt(
                tplgcl_dscrptr_result_filename,
                np.asarray([tplgcl_dscrptr_result]), fmt="%d")
        elif isinstance(tplgcl_dscrptr_result, float):
            np.savetxt(
                tplgcl_dscrptr_result_filename,
                np.asarray([tplgcl_dscrptr_result]))
        elif tplgcl_dscrptr_result.ndim == 1:
            if isinstance(tplgcl_dscrptr_result[0], np.int_):
                np.savetxt(
                    tplgcl_dscrptr_result_filename, tplgcl_dscrptr_result,
                    fmt="%d")
            else:
                np.savetxt(
                    tplgcl_dscrptr_result_filename, tplgcl_dscrptr_result)
        else:
            if isinstance(tplgcl_dscrptr_result[0, 0], np.int_):
                np.savetxt(
                    tplgcl_dscrptr_result_filename, tplgcl_dscrptr_result,
                    fmt="%d")
            else:
                np.savetxt(
                    tplgcl_dscrptr_result_filename, tplgcl_dscrptr_result)

    # Return topological descriptor result (if called for)
    if return_tplgcl_dscrptr_result: return tplgcl_dscrptr_result
    else: return None