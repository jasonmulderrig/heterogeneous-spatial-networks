import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from file_io.file_io import L_filename_str
from helpers.graph_utils import (
    add_nodes_from_numpy_array,
    add_edges_from_numpy_array,
    elastically_effective_end_linked_graph
)
from topological_descriptors.general_topological_descriptors import (
    core_pb_edge_id
)
from helpers.plotting_utils import (
    dim_2_network_topology_axes_formatter,
    dim_3_network_topology_axes_formatter
)
from networks.aelp_networks import aelp_filename_str

def aelp_network_core_node_marker_style(
        core_node: int,
        node_type: int,
        selfloop_edges: list[tuple[int, int]]) -> tuple[str, str, str]:
    marker = ""
    markerfacecolor = ""
    markeredgecolor = ""
    if node_type == 1:
        markerfacecolor = "black"
        markeredgecolor = "black"
        if len(selfloop_edges) == 0: marker = "."
        else:
            core_node_selfloop_edge_order = 0
            for selfloop_edge in selfloop_edges:
                if (selfloop_edge[0] == core_node) and (selfloop_edge[1] == core_node):
                    core_node_selfloop_edge_order += 1
            if core_node_selfloop_edge_order == 0: marker = "."
            elif core_node_selfloop_edge_order == 1: marker = "s"
            elif core_node_selfloop_edge_order == 2: marker = "h"
            elif core_node_selfloop_edge_order >= 3: marker = "8"
    elif node_type == 3:
        marker = "."
        markerfacecolor = "red"
        markeredgecolor = "red"
    return (marker, markerfacecolor, markeredgecolor)

def aelp_network_edge_alpha(
        core_node_0: int,
        core_node_1: int,
        conn_graph: nx.Graph | nx.MultiGraph) -> float:
    alpha = 0.25 * conn_graph.number_of_edges(core_node_0, core_node_1)
    alpha = np.minimum(alpha, 1.0)
    return alpha

def aelp_network_topology_plotter(
        plt_pad_prefactor: float,
        core_tick_inc_prefactor: float,
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> None:
    # Generate filenames
    aelp_filename = aelp_filename_str(network, date, batch, sample, config)
    coords_filename = aelp_filename + ".coords"
    node_type_filename = aelp_filename + "-node_type.dat"
    conn_core_edges_filename = aelp_filename + "-conn_core_edges.dat"
    conn_pb_edges_filename = aelp_filename + "-conn_pb_edges.dat"

    # Load simulation box size
    L = np.loadtxt(L_filename_str(network, date, batch, sample))

    # Load node coordinates
    coords = np.loadtxt(coords_filename)
    n, dim = np.shape(coords)

    # Load fundamental graph constituents
    core_nodes = np.arange(n, dtype=int)
    node_type = np.loadtxt(node_type_filename, dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Create nx.MultiGraphs and add nodes before edges
    conn_core_graph = nx.MultiGraph()
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_core_graph = add_edges_from_numpy_array(conn_core_graph, conn_core_edges)

    conn_pb_graph = nx.MultiGraph()
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)

    conn_graph = nx.MultiGraph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Extract list of self-loop edges
    conn_graph_selfloop_edges = list(nx.selfloop_edges(conn_graph))
    
    # Number of core edges and periodic boundary edges
    core_m = np.shape(conn_core_edges)[0]
    pb_m = np.shape(conn_pb_edges)[0]

    # Plot formatting parameters
    plt_pad = plt_pad_prefactor * L
    core_tick_inc = core_tick_inc_prefactor * L
    min_core = -plt_pad
    max_core = L + plt_pad
    core_tick_steps = int(np.around((max_core-min_core)/core_tick_inc)) + 1

    xlim = np.asarray([min_core, max_core])
    ylim = np.asarray([min_core, max_core])

    xticks = np.linspace(min_core, max_core, core_tick_steps)
    yticks = np.linspace(min_core, max_core, core_tick_steps)

    xlabel = "x"
    ylabel = "y"

    grid_alpha = 0.25
    grid_zorder = 0

    # Core square box coordinates and preformating
    core_square = np.asarray(
        [
            [0, 0], [L, 0], [L, L], [0, L], [0, 0]
        ]
    )
    core_square_color = "red"
    core_square_linewidth = 0.5

    # Plot formatting parameters and preformatting
    zlim = np.asarray([min_core, max_core])
    zticks = np.linspace(min_core, max_core, core_tick_steps)
    zlabel = "z"

    # Core cube box coordinates and preformating
    core_cube = np.asarray(
        [
            [[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, 0]],
            [[0, 0, L], [L, 0, L], [L, L, L], [0, L, L], [0, 0, L]],
            [[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L], [0, 0, 0]],
            [[L, 0, 0], [L, L, 0], [L, L, L], [L, 0, L], [L, 0, 0]],
            [[L, L, 0], [0, L, 0], [0, L, L], [L, L, L], [L, L, 0]],
            [[0, L, 0], [0, 0, 0], [0, 0, L], [0, L, L], [0, L, 0]]

        ]
    )
    core_cube_color = "red"
    core_cube_linewidth = 0.5

    def core_pb_graph_topology_plotting_func(colored=False, eeel_ntwrk=False):
        """Plot of the core and periodic boundary cross-linkers and
        edges for the graph capturing the spatial topology of the core
        and periodic boundary nodes and edges. Here, the edges could all
        be represented as blue lines, or the core and periodic boundary
        edges could each be represented by purple or olive lines,
        respectively.
        
        """
        if dim == 2:
            fig, ax = plt.subplots()
            for edge in range(core_m):
                core_node_0 = conn_core_edges[edge, 0]
                core_node_1 = conn_core_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = node_type[core_node_0]
                core_node_1_type = node_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                edge_x = np.asarray(
                    [
                        core_node_0_x,
                        core_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        core_node_0_y,
                        core_node_1_y
                    ]
                )
                ax.plot(
                    edge_x, edge_y,
                    color="tab:purple" if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_0, core_node_0_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_1, core_node_1_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
            for edge in range(pb_m):
                core_node_0 = conn_pb_edges[edge, 0]
                core_node_1 = conn_pb_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = node_type[core_node_0]
                core_node_1_type = node_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                core_node_0_coords = np.asarray([core_node_0_x, core_node_0_y])
                core_node_1_coords = np.asarray([core_node_1_x, core_node_1_y])
                pb_node_0_coords, _ = core_pb_edge_id(
                    core_node_1_coords, core_node_0_coords, L)
                pb_node_1_coords, _ = core_pb_edge_id(
                    core_node_0_coords, core_node_1_coords, L)
                pb_node_0_x = pb_node_0_coords[0]
                pb_node_0_y = pb_node_0_coords[1]
                pb_node_1_x = pb_node_1_coords[0]
                pb_node_1_y = pb_node_1_coords[1]
                edge_x = np.asarray(
                    [
                        core_node_0_x,
                        pb_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        core_node_0_y,
                        pb_node_1_y
                    ]
                )
                ax.plot(
                    edge_x, edge_y,
                    color="tab:olive" if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                edge_x = np.asarray(
                    [
                        pb_node_0_x,
                        core_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        pb_node_0_y,
                        core_node_1_y
                    ]
                )
                ax.plot(
                    edge_x, edge_y,
                    color="tab:olive" if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_0, core_node_0_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                ax.plot(
                    pb_node_0_x, pb_node_0_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_1, core_node_1_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                ax.plot(
                    pb_node_1_x, pb_node_1_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
            ax = dim_2_network_topology_axes_formatter(
                ax, core_square, core_square_color, core_square_linewidth,
                xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha,
                grid_zorder)
        elif dim == 3:
            fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
            for edge in range(core_m):
                core_node_0 = conn_core_edges[edge, 0]
                core_node_1 = conn_core_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = node_type[core_node_0]
                core_node_1_type = node_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_0_z = coords[core_node_0, 2]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                core_node_1_z = coords[core_node_1, 2]
                edge_x = np.asarray(
                    [
                        core_node_0_x,
                        core_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        core_node_0_y,
                        core_node_1_y
                    ]
                )
                edge_z = np.asarray(
                    [
                        core_node_0_z,
                        core_node_1_z
                    ]
                )
                ax.plot(
                    edge_x, edge_y, edge_z,
                    color="tab:purple" if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_0, core_node_0_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_1, core_node_1_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
            for edge in range(pb_m):
                core_node_0 = conn_pb_edges[edge, 0]
                core_node_1 = conn_pb_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = node_type[core_node_0]
                core_node_1_type = node_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_0_z = coords[core_node_0, 2]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                core_node_1_z = coords[core_node_1, 2]
                core_node_0_coords = np.asarray(
                    [
                        core_node_0_x,
                        core_node_0_y,
                        core_node_0_z
                    ]
                )
                core_node_1_coords = np.asarray(
                    [
                        core_node_1_x,
                        core_node_1_y,
                        core_node_1_z
                    ]
                )
                pb_node_0_coords, _ = core_pb_edge_id(
                    core_node_1_coords, core_node_0_coords, L)
                pb_node_1_coords, _ = core_pb_edge_id(
                    core_node_0_coords, core_node_1_coords, L)
                pb_node_0_x = pb_node_0_coords[0]
                pb_node_0_y = pb_node_0_coords[1]
                pb_node_0_z = pb_node_0_coords[2]
                pb_node_1_x = pb_node_1_coords[0]
                pb_node_1_y = pb_node_1_coords[1]
                pb_node_1_z = pb_node_1_coords[2]
                edge_x = np.asarray(
                    [
                        core_node_0_x,
                        pb_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        core_node_0_y,
                        pb_node_1_y
                    ]
                )
                edge_z = np.asarray(
                    [
                        core_node_0_z,
                        pb_node_1_z
                    ]
                )
                ax.plot(
                    edge_x, edge_y, edge_z,
                    color="tab:olive" if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                edge_x = np.asarray(
                    [
                        pb_node_0_x,
                        core_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        pb_node_0_y,
                        core_node_1_y
                    ]
                )
                edge_z = np.asarray(
                    [
                        pb_node_0_z,
                        core_node_1_z
                    ]
                )
                ax.plot(
                    edge_x, edge_y, edge_z,
                    color="tab:olive" if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_0, core_node_0_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                ax.plot(
                    pb_node_0_x, pb_node_0_y, pb_node_0_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_1, core_node_1_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                ax.plot(
                    pb_node_1_x, pb_node_1_y, pb_node_1_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
            ax = dim_3_network_topology_axes_formatter(
                ax, core_cube, core_cube_color, core_cube_linewidth,
                xlim, ylim, zlim, xticks, yticks, zticks,
                xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        core_pb_graph_topology = (
            "-eeel_core_pb_graph" if eeel_ntwrk else "-core_pb_graph"
        )
        core_pb_graph_topology = (
            core_pb_graph_topology+"_colored_topology" if colored
            else core_pb_graph_topology+"_topology"
        )
        fig.savefig(aelp_filename+core_pb_graph_topology+".png")
        plt.close()
        
        return None
    
    def conn_graph_topology_plotting_func(colored=False, eeel_ntwrk=False):
        """Plot of the core and periodic boundary cross-linkers and
        edges for the graph capturing the periodic connections between
        the core nodes. Here, the edges could all be represented as blue
        lines, or the core and periodic boundary edges could each be
        represented by purple or olive lines, respectively.
        
        """
        if dim == 2:
            fig, ax = plt.subplots()
            for edge in range(core_m):
                core_node_0 = conn_core_edges[edge, 0]
                core_node_1 = conn_core_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = node_type[core_node_0]
                core_node_1_type = node_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                edge_x = np.asarray(
                    [
                        core_node_0_x,
                        core_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        core_node_0_y,
                        core_node_1_y
                    ]
                )
                ax.plot(
                    edge_x, edge_y, color="tab:purple" if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_0, core_node_0_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_1, core_node_1_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
            for edge in range(pb_m):
                core_node_0 = conn_pb_edges[edge, 0]
                core_node_1 = conn_pb_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = node_type[core_node_0]
                core_node_1_type = node_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                edge_x = np.asarray(
                    [
                        core_node_0_x,
                        core_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        core_node_0_y,
                        core_node_1_y
                    ]
                )
                ax.plot(
                    edge_x, edge_y, color="tab:olive" if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_0, core_node_0_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_1, core_node_1_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
            ax = dim_2_network_topology_axes_formatter(
                ax, core_square, core_square_color, core_square_linewidth,
                xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha,
                grid_zorder)
        elif dim == 3:
            fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
            for edge in range(core_m):
                core_node_0 = conn_core_edges[edge, 0]
                core_node_1 = conn_core_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = node_type[core_node_0]
                core_node_1_type = node_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_0_z = coords[core_node_0, 2]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                core_node_1_z = coords[core_node_1, 2]
                edge_x = np.asarray(
                    [
                        core_node_0_x,
                        core_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        core_node_0_y,
                        core_node_1_y
                    ]
                )
                edge_z = np.asarray(
                    [
                        core_node_0_z,
                        core_node_1_z
                    ]
                )
                ax.plot(
                    edge_x, edge_y, edge_z,
                    color="tab:purple" if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_0, core_node_0_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_1, core_node_1_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
            for edge in range(pb_m):
                core_node_0 = conn_pb_edges[edge, 0]
                core_node_1 = conn_pb_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = node_type[core_node_0]
                core_node_1_type = node_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_0_z = coords[core_node_0, 2]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                core_node_1_z = coords[core_node_1, 2]
                edge_x = np.asarray(
                    [
                        core_node_0_x,
                        core_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        core_node_0_y,
                        core_node_1_y
                    ]
                )
                edge_z = np.asarray(
                    [
                        core_node_0_z,
                        core_node_1_z
                    ]
                )
                ax.plot(
                    edge_x, edge_y, edge_z,
                    color="tab:olive" if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_0, core_node_0_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_1, core_node_1_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
            ax = dim_3_network_topology_axes_formatter(
                ax, core_cube, core_cube_color, core_cube_linewidth,
                xlim, ylim, zlim, xticks, yticks, zticks,
                xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        conn_graph_topology = (
            "-eeel_conn_graph" if eeel_ntwrk else "-conn_graph"
        )
        conn_graph_topology = (
            conn_graph_topology+"_colored_topology" if colored
            else conn_graph_topology+"_topology"
        )
        fig.savefig(aelp_filename+conn_graph_topology+".png")
        plt.close()
        
        return None
    
    core_pb_graph_topology_plotting_func(colored=False, eeel_ntwrk=False)
    core_pb_graph_topology_plotting_func(colored=True, eeel_ntwrk=False)
    conn_graph_topology_plotting_func(colored=False, eeel_ntwrk=False)
    conn_graph_topology_plotting_func(colored=True, eeel_ntwrk=False)
    
    # Extract elastically-effective end-linked network
    conn_graph = elastically_effective_end_linked_graph(conn_graph).copy()
    conn_core_graph = conn_core_graph.subgraph(list(conn_graph.nodes())).copy()
    conn_pb_graph = conn_pb_graph.subgraph(list(conn_graph.nodes())).copy()

    # Extract list of self-loop edges
    conn_graph_selfloop_edges = list(nx.selfloop_edges(conn_graph))
    
    # Extract edges
    conn_core_edges = np.asarray(list(conn_core_graph.edges()), dtype=int)
    conn_pb_edges = np.asarray(list(conn_pb_graph.edges()), dtype=int)

    # Number of core edges and periodic boundary edges
    core_m = np.shape(conn_core_edges)[0]
    pb_m = np.shape(conn_pb_edges)[0]

    core_pb_graph_topology_plotting_func(colored=False, eeel_ntwrk=True)
    core_pb_graph_topology_plotting_func(colored=True, eeel_ntwrk=True)
    conn_graph_topology_plotting_func(colored=False, eeel_ntwrk=True)
    conn_graph_topology_plotting_func(colored=True, eeel_ntwrk=True)