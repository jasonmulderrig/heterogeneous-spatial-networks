# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import numpy as np
from file_io.file_io import (
    L_filename_str,
    filepath_str
)
from topological_descriptors.general_topological_descriptors import l_arr_func
from networks.aelp_networks import aelp_filename_str

def main():
    # Initialization of identification information for these batches of
    # artificial polydisperse end-linked polymer networks
    network = "apelp"
    date = "20250102"
    batch = "A"
    scheme = "prhd"

    network_str = f"network_{network}"
    date_str = f"date_{date}"
    batch_str = f"batch_{batch}"
    scheme_str = f"scheme_{scheme}"

    dim_str = "dim"
    b_str = "b"
    xi_str = "xi"
    rho_nu_str = "rho_nu"
    k_str = "k"
    n_str = "n"
    nu_str = "nu"
    nu_max_str = "nu_max"
    config_str = "config"

    filepath = filepath_str(network)
    filename_prefix = filepath + f"{date}{batch}"

    identifier_filename = filename_prefix + "-identifier" + ".txt"
    dim_filename = filename_prefix + f"-{dim_str}" + ".dat"
    b_filename = filename_prefix + f"-{b_str}" + ".dat"
    xi_filename = filename_prefix + f"-{xi_str}" + ".dat"
    rho_nu_filename = filename_prefix + f"-{rho_nu_str}" + ".dat"
    k_filename = filename_prefix + f"-{k_str}" + ".dat"
    n_filename = filename_prefix + f"-{n_str}" + ".dat"
    nu_filename = filename_prefix + f"-{nu_str}" + ".dat"
    config_filename = filename_prefix + f"-{config_str}" + ".dat"
    sample_params_filename = filename_prefix + "-sample_params" + ".dat"
    sample_config_params_filename = (
        filename_prefix + "-sample_config_params" + ".dat"
    )

    identifier_arr = np.loadtxt(
        identifier_filename, dtype=str, usecols=0, ndmin=1)
    dim_arr = np.loadtxt(dim_filename, dtype=int, ndmin=1)
    b_arr = np.loadtxt(b_filename, ndmin=1)
    xi_arr = np.loadtxt(xi_filename, ndmin=1)
    rho_nu_arr = np.loadtxt(rho_nu_filename, ndmin=1)
    k_arr = np.loadtxt(k_filename, dtype=int, ndmin=1)
    n_arr = np.loadtxt(n_filename, dtype=int, ndmin=1)
    nu_arr = np.loadtxt(nu_filename, dtype=int, ndmin=1)
    config_arr = np.loadtxt(config_filename, dtype=int, ndmin=1)
    
    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    xi_num = np.shape(xi_arr)[0]
    rho_nu_num = np.shape(rho_nu_arr)[0]
    k_num = np.shape(k_arr)[0]
    n_num = np.shape(n_arr)[0]
    if nu_arr.ndim == 1: nu_arr = nu_arr.reshape(1, -1)
    nu_num = np.shape(nu_arr)[0]
    config_num = np.shape(config_arr)[0]

    sample_num = dim_num * b_num * xi_num * rho_nu_num * k_num * n_num * nu_num

    b = b_arr[0]
    
    mean_perc_errant_edges = 0.
    for sample in range(sample_num):
        for config in range(config_num):
            # Generate filenames
            L_filename = L_filename_str(network, date, batch, sample)
            aelp_filename = aelp_filename_str(network, date, batch, sample, config)
            coords_filename = aelp_filename + ".coords"
            conn_core_edges_filename = (
                aelp_filename + "-conn_core_edges" + ".dat"
            )
            conn_pb_edges_filename = aelp_filename + "-conn_pb_edges" + ".dat"
            conn_nu_core_edges_filename = (
                aelp_filename + "-conn_nu_core_edges" + ".dat"
            )
            conn_nu_pb_edges_filename = (
                aelp_filename + "-conn_nu_pb_edges" + ".dat"
            )

            # Load simulation box size and node coordinates
            L = np.loadtxt(L_filename)
            coords = np.loadtxt(coords_filename)

            # Load fundamental graph constituents
            conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
            conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
            m = np.shape(conn_core_edges)[0] + np.shape(conn_pb_edges)[0]
            conn_nu_core_edges = np.loadtxt(conn_nu_core_edges_filename, dtype=int)
            conn_nu_pb_edges = np.loadtxt(conn_nu_pb_edges_filename, dtype=int)
            conn_nu_edges = np.concatenate((conn_nu_core_edges, conn_nu_pb_edges), dtype=int)

            # Calculate end-to-end chain length (Euclidean edge length)
            l_core_chn, l_pb_chn = l_arr_func(
                conn_core_edges, conn_pb_edges, coords, L)
            l_edges = np.concatenate((l_core_chn, l_pb_chn))
            
            errant_edges = 0
            for edge in range(m):
                if (b*conn_nu_edges[edge]) < l_edges[edge]:
                    errant_edges += 1
                    print("apelp sample {} config {} edge {} nu = {} < l = {}".format(sample, config, edge, conn_nu_edges[edge], l_edges[edge]))
            perc_errant_edges = errant_edges / m * 100
            if perc_errant_edges > 0:
                print("apelp sample {} config {} has {} errant edges, {} percent of edges in the network".format(sample, config, errant_edges, perc_errant_edges))
            mean_perc_errant_edges += perc_errant_edges
    mean_perc_errant_edges /= config_num

    print("{} percent of apelp edges are errant (on average)".format(mean_perc_errant_edges))

if __name__ == "__main__":
    main()