import numpy as np
import matplotlib.pyplot as plt
from file_io import (
    filepath_str,
    L_filename_str
)
from aelp_networks import aelp_filename_str

def main():
    # Initialization of identification information for these batches of
    # artificial end-linked polymer networks
    network_auelp = "auelp"
    network_apelp = "apelp"
    date = "20241210"
    batch_A = "A"
    batch_B = "B"
    scheme = "prhd"

    network_auelp_str = f"network_{network_auelp}"
    network_apelp_str = f"network_{network_apelp}"
    date_str = f"date_{date}"
    batch_A_str = f"batch_{batch_A}"
    batch_B_str = f"batch_{batch_B}"
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

    network_auelp_filepath = filepath_str(network_auelp)
    network_apelp_filepath = filepath_str(network_apelp)

    network_auelp_batch_A_filename_prefix = (
        network_auelp_filepath + f"{date}{batch_A}"
    )
    network_auelp_batch_B_filename_prefix = (
        network_auelp_filepath + f"{date}{batch_B}"
    )
    network_apelp_batch_A_filename_prefix = (
        network_apelp_filepath + f"{date}{batch_A}"
    )
    network_apelp_batch_B_filename_prefix = (
        network_apelp_filepath + f"{date}{batch_B}"
    )

    network_auelp_batch_A_identifier_filename = (
        network_auelp_batch_A_filename_prefix + "-identifier" + ".txt"
    )
    network_auelp_batch_B_identifier_filename = (
        network_auelp_batch_B_filename_prefix + "-identifier" + ".txt"
    )
    network_apelp_batch_A_identifier_filename = (
        network_apelp_batch_A_filename_prefix + "-identifier" + ".txt"
    )
    network_apelp_batch_B_identifier_filename = (
        network_apelp_batch_B_filename_prefix + "-identifier" + ".txt"
    )
    network_auelp_batch_A_dim_filename = (
        network_auelp_batch_A_filename_prefix + f"-{dim_str}" + ".dat"
    )
    network_auelp_batch_B_dim_filename = (
        network_auelp_batch_B_filename_prefix + f"-{dim_str}" + ".dat"
    )
    network_apelp_batch_A_dim_filename = (
        network_apelp_batch_A_filename_prefix + f"-{dim_str}" + ".dat"
    )
    network_apelp_batch_B_dim_filename = (
        network_apelp_batch_B_filename_prefix + f"-{dim_str}" + ".dat"
    )
    network_auelp_batch_A_b_filename = (
        network_auelp_batch_A_filename_prefix + f"-{b_str}" + ".dat"
    )
    network_auelp_batch_B_b_filename = (
        network_auelp_batch_B_filename_prefix + f"-{b_str}" + ".dat"
    )
    network_apelp_batch_A_b_filename = (
        network_apelp_batch_A_filename_prefix + f"-{b_str}" + ".dat"
    )
    network_apelp_batch_B_b_filename = (
        network_apelp_batch_B_filename_prefix + f"-{b_str}" + ".dat"
    )
    network_auelp_batch_A_xi_filename = (
        network_auelp_batch_A_filename_prefix + f"-{xi_str}" + ".dat"
    )
    network_auelp_batch_B_xi_filename = (
        network_auelp_batch_B_filename_prefix + f"-{xi_str}" + ".dat"
    )
    network_apelp_batch_A_xi_filename = (
        network_apelp_batch_A_filename_prefix + f"-{xi_str}" + ".dat"
    )
    network_apelp_batch_B_xi_filename = (
        network_apelp_batch_B_filename_prefix + f"-{xi_str}" + ".dat"
    )
    network_auelp_batch_A_rho_nu_filename = (
        network_auelp_batch_A_filename_prefix + f"-{rho_nu_str}" + ".dat"
    )
    network_auelp_batch_B_rho_nu_filename = (
        network_auelp_batch_B_filename_prefix + f"-{rho_nu_str}" + ".dat"
    )
    network_apelp_batch_A_rho_nu_filename = (
        network_apelp_batch_A_filename_prefix + f"-{rho_nu_str}" + ".dat"
    )
    network_apelp_batch_B_rho_nu_filename = (
        network_apelp_batch_B_filename_prefix + f"-{rho_nu_str}" + ".dat"
    )
    network_auelp_batch_A_k_filename = (
        network_auelp_batch_A_filename_prefix + f"-{k_str}" + ".dat"
    )
    network_auelp_batch_B_k_filename = (
        network_auelp_batch_B_filename_prefix + f"-{k_str}" + ".dat"
    )
    network_apelp_batch_A_k_filename = (
        network_apelp_batch_A_filename_prefix + f"-{k_str}" + ".dat"
    )
    network_apelp_batch_B_k_filename = (
        network_apelp_batch_B_filename_prefix + f"-{k_str}" + ".dat"
    )
    network_auelp_batch_A_n_filename = (
        network_auelp_batch_A_filename_prefix + f"-{n_str}" + ".dat"
    )
    network_auelp_batch_B_n_filename = (
        network_auelp_batch_B_filename_prefix + f"-{n_str}" + ".dat"
    )
    network_apelp_batch_A_n_filename = (
        network_apelp_batch_A_filename_prefix + f"-{n_str}" + ".dat"
    )
    network_apelp_batch_B_n_filename = (
        network_apelp_batch_B_filename_prefix + f"-{n_str}" + ".dat"
    )
    network_auelp_batch_A_nu_filename = (
        network_auelp_batch_A_filename_prefix + f"-{nu_str}" + ".dat"
    )
    network_auelp_batch_B_nu_filename = (
        network_auelp_batch_B_filename_prefix + f"-{nu_str}" + ".dat"
    )
    network_apelp_batch_A_nu_filename = (
        network_apelp_batch_A_filename_prefix + f"-{nu_str}" + ".dat"
    )
    network_apelp_batch_B_nu_filename = (
        network_apelp_batch_B_filename_prefix + f"-{nu_str}" + ".dat"
    )
    network_auelp_batch_A_nu_max_filename = (
        network_auelp_batch_A_filename_prefix + f"-{nu_max_str}" + ".dat"
    )
    network_auelp_batch_B_nu_max_filename = (
        network_auelp_batch_B_filename_prefix + f"-{nu_max_str}" + ".dat"
    )
    network_apelp_batch_A_nu_max_filename = (
        network_apelp_batch_A_filename_prefix + f"-{nu_max_str}" + ".dat"
    )
    network_apelp_batch_B_nu_max_filename = (
        network_apelp_batch_B_filename_prefix + f"-{nu_max_str}" + ".dat"
    )
    network_auelp_batch_A_config_filename = (
        network_auelp_batch_A_filename_prefix + f"-{config_str}" + ".dat"
    )
    network_auelp_batch_B_config_filename = (
        network_auelp_batch_B_filename_prefix + f"-{config_str}" + ".dat"
    )
    network_apelp_batch_A_config_filename = (
        network_apelp_batch_A_filename_prefix + f"-{config_str}" + ".dat"
    )
    network_apelp_batch_B_config_filename = (
        network_apelp_batch_B_filename_prefix + f"-{config_str}" + ".dat"
    )
    network_auelp_batch_A_params_filename = (
        network_auelp_batch_A_filename_prefix + "-params" + ".dat"
    )
    network_auelp_batch_B_params_filename = (
        network_auelp_batch_B_filename_prefix + "-params" + ".dat"
    )
    network_apelp_batch_A_params_filename = (
        network_apelp_batch_A_filename_prefix + "-params" + ".dat"
    )
    network_apelp_batch_B_params_filename = (
        network_apelp_batch_B_filename_prefix + "-params" + ".dat"
    )
    network_auelp_batch_A_sample_params_filename = (
        network_auelp_batch_A_filename_prefix + "-sample_params" + ".dat"
    )
    network_auelp_batch_B_sample_params_filename = (
        network_auelp_batch_B_filename_prefix + "-sample_params" + ".dat"
    )
    network_apelp_batch_A_sample_params_filename = (
        network_apelp_batch_A_filename_prefix + "-sample_params" + ".dat"
    )
    network_apelp_batch_B_sample_params_filename = (
        network_apelp_batch_B_filename_prefix + "-sample_params" + ".dat"
    )
    network_auelp_batch_A_sample_config_params_filename = (
        network_auelp_batch_A_filename_prefix + "-sample_config_params"
        + ".dat"
    )
    network_auelp_batch_B_sample_config_params_filename = (
        network_auelp_batch_B_filename_prefix + "-sample_config_params"
        + ".dat"
    )
    network_apelp_batch_A_sample_config_params_filename = (
        network_apelp_batch_A_filename_prefix + "-sample_config_params"
        + ".dat"
    )
    network_apelp_batch_B_sample_config_params_filename = (
        network_apelp_batch_B_filename_prefix + "-sample_config_params"
        + ".dat"
    )

    network_auelp_batch_A_identifier_arr = np.loadtxt(
        network_auelp_batch_A_identifier_filename, dtype=str, usecols=0, ndmin=1)
    network_apelp_batch_A_identifier_arr = np.loadtxt(
        network_apelp_batch_A_identifier_filename, dtype=str, usecols=0, ndmin=1)
    network_auelp_batch_B_identifier_arr = np.loadtxt(
        network_auelp_batch_B_identifier_filename, dtype=str, usecols=0, ndmin=1)
    network_apelp_batch_B_identifier_arr = np.loadtxt(
        network_apelp_batch_B_identifier_filename, dtype=str, usecols=0, ndmin=1)
    dim_2_arr = np.loadtxt(network_auelp_batch_A_dim_filename, dtype=int, ndmin=1)
    dim_3_arr = np.loadtxt(network_auelp_batch_B_dim_filename, dtype=int, ndmin=1)
    b_arr = np.loadtxt(network_auelp_batch_A_b_filename, ndmin=1)
    xi_arr = np.loadtxt(network_auelp_batch_A_xi_filename, ndmin=1)
    rho_nu_arr = np.loadtxt(network_auelp_batch_A_rho_nu_filename, ndmin=1)
    k_arr = np.loadtxt(network_auelp_batch_A_k_filename, dtype=int, ndmin=1)
    n_arr = np.loadtxt(network_auelp_batch_A_n_filename, dtype=int, ndmin=1)
    dim_2_nu_arr = np.loadtxt(network_auelp_batch_A_nu_filename, dtype=int, ndmin=1)
    dim_3_nu_arr = np.loadtxt(network_auelp_batch_B_nu_filename, dtype=int, ndmin=1)
    dim_2_nu_max_arr = np.loadtxt(
        network_auelp_batch_A_nu_max_filename, dtype=int, ndmin=1)
    dim_3_nu_max_arr = np.loadtxt(
        network_auelp_batch_B_nu_max_filename, dtype=int, ndmin=1)
    config_arr = np.loadtxt(network_auelp_batch_A_config_filename, dtype=int, ndmin=1)
    
    network_auelp_batch_A_params_arr = np.loadtxt(
        network_auelp_batch_A_params_filename, ndmin=1)
    network_auelp_batch_B_params_arr = np.loadtxt(
        network_auelp_batch_B_params_filename, ndmin=1)
    network_apelp_batch_A_params_arr = np.loadtxt(
        network_apelp_batch_A_params_filename, ndmin=1)
    network_apelp_batch_B_params_arr = np.loadtxt(
        network_apelp_batch_B_params_filename, ndmin=1)
    network_auelp_batch_A_sample_params_arr = np.loadtxt(
        network_auelp_batch_A_sample_params_filename, ndmin=1)
    network_auelp_batch_B_sample_params_arr = np.loadtxt(
        network_auelp_batch_B_sample_params_filename, ndmin=1)
    network_apelp_batch_A_sample_params_arr = np.loadtxt(
        network_apelp_batch_A_sample_params_filename, ndmin=1)
    network_apelp_batch_B_sample_params_arr = np.loadtxt(
        network_apelp_batch_B_sample_params_filename, ndmin=1)
    network_auelp_batch_A_sample_config_params_arr = np.loadtxt(
        network_auelp_batch_A_sample_config_params_filename, ndmin=1)
    network_auelp_batch_B_sample_config_params_arr = np.loadtxt(
        network_auelp_batch_B_sample_config_params_filename, ndmin=1)
    network_apelp_batch_A_sample_config_params_arr = np.loadtxt(
        network_apelp_batch_A_sample_config_params_filename, ndmin=1)
    network_apelp_batch_B_sample_config_params_arr = np.loadtxt(
        network_apelp_batch_B_sample_config_params_filename, ndmin=1)
    
    dim_2_num = np.shape(dim_2_arr)[0]
    dim_3_num = np.shape(dim_3_arr)[0]
    b_num = np.shape(b_arr)[0]
    xi_num = np.shape(xi_arr)[0]
    rho_nu_num = np.shape(rho_nu_arr)[0]
    k_num = np.shape(k_arr)[0]
    n_num = np.shape(n_arr)[0]
    dim_2_nu_num = np.shape(dim_2_nu_arr)[0]
    dim_3_nu_num = np.shape(dim_3_nu_arr)[0]
    dim_2_nu_max_num = np.shape(dim_2_nu_max_arr)[0]
    dim_3_nu_max_num = np.shape(dim_3_nu_max_arr)[0]
    config_num = np.shape(config_arr)[0]

    network_auelp_batch_A_sample_num = (
        dim_2_num * b_num * xi_num * rho_nu_num * k_num * n_num
        * dim_2_nu_num * dim_2_nu_max_num
    )
    network_auelp_batch_B_sample_num = (
        dim_3_num * b_num * xi_num * rho_nu_num * k_num * n_num
        * dim_3_nu_num * dim_3_nu_max_num
    )
    network_apelp_batch_A_sample_num = (
        dim_2_num * b_num * xi_num * rho_nu_num * k_num * n_num
        * dim_2_nu_num * dim_2_nu_max_num
    )
    network_apelp_batch_B_sample_num = (
        dim_3_num * b_num * xi_num * rho_nu_num * k_num * n_num
        * dim_3_nu_num * dim_3_nu_max_num
    )
    network_auelp_batch_A_sample_config_num = (
        network_auelp_batch_A_sample_num * config_num
    )
    network_auelp_batch_B_sample_config_num = (
        network_auelp_batch_B_sample_num * config_num
    )
    network_apelp_batch_A_sample_config_num = (
        network_apelp_batch_A_sample_num * config_num
    )
    network_apelp_batch_B_sample_config_num = (
        network_apelp_batch_B_sample_num * config_num
    )

    dim_2 = dim_2_arr[0]
    dim_3 = dim_3_arr[0]

    # Calculate number of nodes and the dimensionality in each network
    # configuration
    network_auelp_batch_A_coords_filename = (
        aelp_filename_str(network_auelp, date, batch_A, sample=0, config=0) + ".coords"
    )
    network_auelp_batch_A_n = np.shape(np.loadtxt(network_auelp_batch_A_coords_filename))[0]
    network_auelp_batch_B_coords_filename = (
        aelp_filename_str(network_auelp, date, batch_B, sample=0, config=0) + ".coords"
    )
    network_auelp_batch_B_n = np.shape(np.loadtxt(network_auelp_batch_B_coords_filename))[0]

    network_apelp_batch_A_coords_filename = (
        aelp_filename_str(network_apelp, date, batch_A, sample=0, config=0) + ".coords"
    )
    network_apelp_batch_A_n = np.shape(np.loadtxt(network_apelp_batch_A_coords_filename))[0]
    network_apelp_batch_B_coords_filename = (
        aelp_filename_str(network_apelp, date, batch_B, sample=0, config=0) + ".coords"
    )
    network_apelp_batch_B_n = np.shape(np.loadtxt(network_apelp_batch_B_coords_filename))[0]

    network_auelp_batch_A_nrmlzd_coords = np.empty((network_auelp_batch_A_sample_num, config_num, network_auelp_batch_A_n, dim_2))
    network_auelp_batch_B_nrmlzd_coords = np.empty((network_auelp_batch_B_sample_num, config_num, network_auelp_batch_B_n, dim_3))
    network_apelp_batch_A_nrmlzd_coords = np.empty((network_apelp_batch_A_sample_num, config_num, network_apelp_batch_A_n, dim_2))
    network_apelp_batch_B_nrmlzd_coords = np.empty((network_apelp_batch_B_sample_num, config_num, network_apelp_batch_B_n, dim_3))

    for sample in range(network_auelp_batch_A_sample_num):
        L = np.loadtxt(L_filename_str(network_auelp, date, batch_A, sample))
        for config in range(config_num):
            coords_filename = (
                aelp_filename_str(network_auelp, date, batch_A, sample, config) + ".coords"
            )
            nrmlzd_coords = np.loadtxt(coords_filename) / L
            network_auelp_batch_A_nrmlzd_coords[sample, config, :, :] = nrmlzd_coords
    for sample in range(network_auelp_batch_B_sample_num):
        L = np.loadtxt(L_filename_str(network_auelp, date, batch_B, sample))
        for config in range(config_num):
            coords_filename = (
                aelp_filename_str(network_auelp, date, batch_B, sample, config) + ".coords"
            )
            nrmlzd_coords = np.loadtxt(coords_filename) / L
            network_auelp_batch_B_nrmlzd_coords[sample, config, :, :] = nrmlzd_coords
    for sample in range(network_apelp_batch_A_sample_num):
        L = np.loadtxt(L_filename_str(network_apelp, date, batch_A, sample))
        for config in range(config_num):
            coords_filename = (
                aelp_filename_str(network_apelp, date, batch_A, sample, config) + ".coords"
            )
            nrmlzd_coords = np.loadtxt(coords_filename) / L
            network_apelp_batch_A_nrmlzd_coords[sample, config, :, :] = nrmlzd_coords
    for sample in range(network_apelp_batch_B_sample_num):
        L = np.loadtxt(L_filename_str(network_apelp, date, batch_B, sample))
        for config in range(config_num):
            coords_filename = (
                aelp_filename_str(network_apelp, date, batch_B, sample, config) + ".coords"
            )
            nrmlzd_coords = np.loadtxt(coords_filename) / L
            network_apelp_batch_B_nrmlzd_coords[sample, config, :, :] = nrmlzd_coords

    network_auelp_batch_A_nrmlzd_coords_min = np.empty((network_auelp_batch_A_sample_num, network_auelp_batch_A_n, dim_2))
    network_auelp_batch_A_nrmlzd_coords_mean_minus_std = np.empty((network_auelp_batch_A_sample_num, network_auelp_batch_A_n, dim_2))
    network_auelp_batch_A_nrmlzd_coords_mean = np.empty((network_auelp_batch_A_sample_num, network_auelp_batch_A_n, dim_2))
    network_auelp_batch_A_nrmlzd_coords_mean_plus_std = np.empty((network_auelp_batch_A_sample_num, network_auelp_batch_A_n, dim_2))
    network_auelp_batch_A_nrmlzd_coords_max = np.empty((network_auelp_batch_A_sample_num, network_auelp_batch_A_n, dim_2))

    network_auelp_batch_B_nrmlzd_coords_min = np.empty((network_auelp_batch_B_sample_num, network_auelp_batch_B_n, dim_3))
    network_auelp_batch_B_nrmlzd_coords_mean_minus_std = np.empty((network_auelp_batch_B_sample_num, network_auelp_batch_B_n, dim_3))
    network_auelp_batch_B_nrmlzd_coords_mean = np.empty((network_auelp_batch_B_sample_num, network_auelp_batch_B_n, dim_3))
    network_auelp_batch_B_nrmlzd_coords_mean_plus_std = np.empty((network_auelp_batch_B_sample_num, network_auelp_batch_B_n, dim_3))
    network_auelp_batch_B_nrmlzd_coords_max = np.empty((network_auelp_batch_B_sample_num, network_auelp_batch_B_n, dim_3))

    network_apelp_batch_A_nrmlzd_coords_min = np.empty((network_apelp_batch_A_sample_num, network_apelp_batch_A_n, dim_2))
    network_apelp_batch_A_nrmlzd_coords_mean_minus_std = np.empty((network_apelp_batch_A_sample_num, network_apelp_batch_A_n, dim_2))
    network_apelp_batch_A_nrmlzd_coords_mean = np.empty((network_apelp_batch_A_sample_num, network_apelp_batch_A_n, dim_2))
    network_apelp_batch_A_nrmlzd_coords_mean_plus_std = np.empty((network_apelp_batch_A_sample_num, network_apelp_batch_A_n, dim_2))
    network_apelp_batch_A_nrmlzd_coords_max = np.empty((network_apelp_batch_A_sample_num, network_apelp_batch_A_n, dim_2))

    network_apelp_batch_B_nrmlzd_coords_min = np.empty((network_apelp_batch_B_sample_num, network_apelp_batch_B_n, dim_3))
    network_apelp_batch_B_nrmlzd_coords_mean_minus_std = np.empty((network_apelp_batch_B_sample_num, network_apelp_batch_B_n, dim_3))
    network_apelp_batch_B_nrmlzd_coords_mean = np.empty((network_apelp_batch_B_sample_num, network_apelp_batch_B_n, dim_3))
    network_apelp_batch_B_nrmlzd_coords_mean_plus_std = np.empty((network_apelp_batch_B_sample_num, network_apelp_batch_B_n, dim_3))
    network_apelp_batch_B_nrmlzd_coords_max = np.empty((network_apelp_batch_B_sample_num, network_apelp_batch_B_n, dim_3))

    for sample in range(network_auelp_batch_A_sample_num):
        for node_label in range(network_auelp_batch_A_n):
            for coord in range(dim_2):
                mean_val = np.mean(
                    network_auelp_batch_A_nrmlzd_coords[sample, :, node_label, coord])
                std_val = np.std(
                    network_auelp_batch_A_nrmlzd_coords[sample, :, node_label, coord])
                network_auelp_batch_A_nrmlzd_coords_min[sample, node_label, coord] = np.min(
                    network_auelp_batch_A_nrmlzd_coords[sample, :, node_label, coord])
                network_auelp_batch_A_nrmlzd_coords_mean_minus_std[sample, node_label, coord] = (
                    mean_val - std_val
                )
                network_auelp_batch_A_nrmlzd_coords_mean[sample, node_label, coord] = mean_val
                network_auelp_batch_A_nrmlzd_coords_mean_plus_std[sample, node_label, coord] = (
                    mean_val + std_val
                )
                network_auelp_batch_A_nrmlzd_coords_max[sample, node_label, coord] = np.max(
                    network_auelp_batch_A_nrmlzd_coords[sample, :, node_label, coord])
    for sample in range(network_auelp_batch_B_sample_num):
        for node_label in range(network_auelp_batch_B_n):
            for coord in range(dim_3):
                mean_val = np.mean(
                    network_auelp_batch_B_nrmlzd_coords[sample, :, node_label, coord])
                std_val = np.std(
                    network_auelp_batch_B_nrmlzd_coords[sample, :, node_label, coord])
                network_auelp_batch_B_nrmlzd_coords_min[sample, node_label, coord] = np.min(
                    network_auelp_batch_B_nrmlzd_coords[sample, :, node_label, coord])
                network_auelp_batch_B_nrmlzd_coords_mean_minus_std[sample, node_label, coord] = (
                    mean_val - std_val
                )
                network_auelp_batch_B_nrmlzd_coords_mean[sample, node_label, coord] = mean_val
                network_auelp_batch_B_nrmlzd_coords_mean_plus_std[sample, node_label, coord] = (
                    mean_val + std_val
                )
                network_auelp_batch_B_nrmlzd_coords_max[sample, node_label, coord] = np.max(
                    network_auelp_batch_B_nrmlzd_coords[sample, :, node_label, coord])
    for sample in range(network_apelp_batch_A_sample_num):
        for node_label in range(network_apelp_batch_A_n):
            for coord in range(dim_2):
                mean_val = np.mean(
                    network_apelp_batch_A_nrmlzd_coords[sample, :, node_label, coord])
                std_val = np.std(
                    network_apelp_batch_A_nrmlzd_coords[sample, :, node_label, coord])
                network_apelp_batch_A_nrmlzd_coords_min[sample, node_label, coord] = np.min(
                    network_apelp_batch_A_nrmlzd_coords[sample, :, node_label, coord])
                network_apelp_batch_A_nrmlzd_coords_mean_minus_std[sample, node_label, coord] = (
                    mean_val - std_val
                )
                network_apelp_batch_A_nrmlzd_coords_mean[sample, node_label, coord] = mean_val
                network_apelp_batch_A_nrmlzd_coords_mean_plus_std[sample, node_label, coord] = (
                    mean_val + std_val
                )
                network_apelp_batch_A_nrmlzd_coords_max[sample, node_label, coord] = np.max(
                    network_apelp_batch_A_nrmlzd_coords[sample, :, node_label, coord])
    for sample in range(network_apelp_batch_B_sample_num):
        for node_label in range(network_apelp_batch_B_n):
            for coord in range(dim_3):
                mean_val = np.mean(
                    network_apelp_batch_B_nrmlzd_coords[sample, :, node_label, coord])
                std_val = np.std(
                    network_apelp_batch_B_nrmlzd_coords[sample, :, node_label, coord])
                network_apelp_batch_B_nrmlzd_coords_min[sample, node_label, coord] = np.min(
                    network_apelp_batch_B_nrmlzd_coords[sample, :, node_label, coord])
                network_apelp_batch_B_nrmlzd_coords_mean_minus_std[sample, node_label, coord] = (
                    mean_val - std_val
                )
                network_apelp_batch_B_nrmlzd_coords_mean[sample, node_label, coord] = mean_val
                network_apelp_batch_B_nrmlzd_coords_mean_plus_std[sample, node_label, coord] = (
                    mean_val + std_val
                )
                network_apelp_batch_B_nrmlzd_coords_max[sample, node_label, coord] = np.max(
                    network_apelp_batch_B_nrmlzd_coords[sample, :, node_label, coord])
    
    network_auelp_batch_A_node_labels = np.arange(network_auelp_batch_A_n, dtype=int)
    network_auelp_batch_B_node_labels = np.arange(network_auelp_batch_B_n, dtype=int)
    network_apelp_batch_A_node_labels = np.arange(network_apelp_batch_A_n, dtype=int)
    network_apelp_batch_B_node_labels = np.arange(network_apelp_batch_B_n, dtype=int)

    for sample in range(network_auelp_batch_A_sample_num):
        for coord in range(dim_2):
            plt.fill_between(
                network_auelp_batch_A_node_labels,
                network_auelp_batch_A_nrmlzd_coords_min[sample, :, coord],
                network_auelp_batch_A_nrmlzd_coords_max[sample, :, coord],
                color="skyblue", alpha=0.25)
            plt.fill_between(
                network_auelp_batch_A_node_labels,
                network_auelp_batch_A_nrmlzd_coords_mean_minus_std[sample, :, coord],
                network_auelp_batch_A_nrmlzd_coords_mean_plus_std[sample, :, coord],
                color="steelblue", alpha=0.25)
            plt.plot(
                network_auelp_batch_A_node_labels,
                network_auelp_batch_A_nrmlzd_coords_mean[sample, :, coord],
                linestyle="-", color="tab:blue")
            plt.xlabel("Node label", fontsize=16)
            if coord == 0:
                plt.ylabel("x/L", fontsize=16)
                filename_str = "x_nrmlzd_vs_node_label"
            elif coord == 1:
                plt.ylabel("y/L", fontsize=16)
                filename_str = "y_nrmlzd_vs_node_label"
            elif coord == 2:
                plt.ylabel("z/L", fontsize=16)
                filename_str = "z_nrmlzd_vs_node_label"
            plt.title("{}D uniform aelp network".format(dim_2), fontsize=16)
            plt.grid(True, alpha=0.25, zorder=0)
            plt.tight_layout()
            filename_str += "_{}d_auelp_network_finalized.png".format(dim_2)
            plt.savefig(filename_str)
            plt.close()
    for sample in range(network_auelp_batch_B_sample_num):
        for coord in range(dim_3):
            plt.fill_between(
                network_auelp_batch_B_node_labels,
                network_auelp_batch_B_nrmlzd_coords_min[sample, :, coord],
                network_auelp_batch_B_nrmlzd_coords_max[sample, :, coord],
                color="skyblue", alpha=0.25)
            plt.fill_between(
                network_auelp_batch_B_node_labels,
                network_auelp_batch_B_nrmlzd_coords_mean_minus_std[sample, :, coord],
                network_auelp_batch_B_nrmlzd_coords_mean_plus_std[sample, :, coord],
                color="steelblue", alpha=0.25)
            plt.plot(
                network_auelp_batch_B_node_labels,
                network_auelp_batch_B_nrmlzd_coords_mean[sample, :, coord],
                linestyle="-", color="tab:blue")
            plt.xlabel("Node label", fontsize=16)
            if coord == 0:
                plt.ylabel("x/L", fontsize=16)
                filename_str = "x_nrmlzd_vs_node_label"
            elif coord == 1:
                plt.ylabel("y/L", fontsize=16)
                filename_str = "y_nrmlzd_vs_node_label"
            elif coord == 2:
                plt.ylabel("z/L", fontsize=16)
                filename_str = "z_nrmlzd_vs_node_label"
            plt.title("{}D uniform aelp network".format(dim_3), fontsize=16)
            plt.grid(True, alpha=0.25, zorder=0)
            plt.tight_layout()
            filename_str += "_{}d_auelp_network_finalized.png".format(dim_3)
            plt.savefig(filename_str)
            plt.close()
    for sample in range(network_apelp_batch_A_sample_num):
        for coord in range(dim_2):
            plt.fill_between(
                network_apelp_batch_A_node_labels,
                network_apelp_batch_A_nrmlzd_coords_min[sample, :, coord],
                network_apelp_batch_A_nrmlzd_coords_max[sample, :, coord],
                color="skyblue", alpha=0.25)
            plt.fill_between(
                network_apelp_batch_A_node_labels,
                network_apelp_batch_A_nrmlzd_coords_mean_minus_std[sample, :, coord],
                network_apelp_batch_A_nrmlzd_coords_mean_plus_std[sample, :, coord],
                color="steelblue", alpha=0.25)
            plt.plot(
                network_apelp_batch_A_node_labels,
                network_apelp_batch_A_nrmlzd_coords_mean[sample, :, coord],
                linestyle="-", color="tab:blue")
            plt.xlabel("Node label", fontsize=16)
            if coord == 0:
                plt.ylabel("x/L", fontsize=16)
                filename_str = "x_nrmlzd_vs_node_label"
            elif coord == 1:
                plt.ylabel("y/L", fontsize=16)
                filename_str = "y_nrmlzd_vs_node_label"
            elif coord == 2:
                plt.ylabel("z/L", fontsize=16)
                filename_str = "z_nrmlzd_vs_node_label"
            plt.title("{}D polydisperse aelp network".format(dim_2), fontsize=16)
            plt.grid(True, alpha=0.25, zorder=0)
            plt.tight_layout()
            filename_str += "_{}d_apelp_network_finalized.png".format(dim_2)
            plt.savefig(filename_str)
            plt.close()
    for sample in range(network_apelp_batch_B_sample_num):
        for coord in range(dim_3):
            plt.fill_between(
                network_apelp_batch_B_node_labels,
                network_apelp_batch_B_nrmlzd_coords_min[sample, :, coord],
                network_apelp_batch_B_nrmlzd_coords_max[sample, :, coord],
                color="skyblue", alpha=0.25)
            plt.fill_between(
                network_apelp_batch_B_node_labels,
                network_apelp_batch_B_nrmlzd_coords_mean_minus_std[sample, :, coord],
                network_apelp_batch_B_nrmlzd_coords_mean_plus_std[sample, :, coord],
                color="steelblue", alpha=0.25)
            plt.plot(
                network_apelp_batch_B_node_labels,
                network_apelp_batch_B_nrmlzd_coords_mean[sample, :, coord],
                linestyle="-", color="tab:blue")
            plt.xlabel("Node label", fontsize=16)
            if coord == 0:
                plt.ylabel("x/L", fontsize=16)
                filename_str = "x_nrmlzd_vs_node_label"
            elif coord == 1:
                plt.ylabel("y/L", fontsize=16)
                filename_str = "y_nrmlzd_vs_node_label"
            elif coord == 2:
                plt.ylabel("z/L", fontsize=16)
                filename_str = "z_nrmlzd_vs_node_label"
            plt.title("{}D polydisperse aelp network".format(dim_3), fontsize=16)
            plt.grid(True, alpha=0.25, zorder=0)
            plt.tight_layout()
            filename_str += "_{}d_apelp_network_finalized.png".format(dim_3)
            plt.savefig(filename_str)
            plt.close()

if __name__ == "__main__":
    main()