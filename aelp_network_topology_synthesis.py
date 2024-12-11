import multiprocessing
import random
import numpy as np
from file_io import (
    filepath_str,
    config_filename_str
)
from multiprocessing_utils import (
    run_aelp_L,
    run_initial_node_seeding,
    run_aelp_network_topology,
    run_aelp_network_additional_node_seeding,
    run_aelp_network_hilbert_node_label_assignment
)
from aelp_networks import aelp_filename_str

def params_list_func(params_arr: np.ndarray) -> list[tuple]:
    if params_arr.ndim == 1:
        return [tuple(params_arr)]
    else:
        return list(map(tuple, params_arr))

def main():
    ##### Load in artificial end-linked network topology configuration
    print(
        "Loading in artificial end-linked network topology configuration",
        flush=True)
    
    # This may or may not correspond to the number of cpus for optimal
    # parallelization performance. Feel free to modify if you see fit.
    cpu_num = int(np.floor(multiprocessing.cpu_count()/2))

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
        dim_2_num * b_num * xi_num * rho_nu_num * k_num * n_num * dim_2_nu_num
    )
    network_auelp_batch_B_sample_num = (
        dim_3_num * b_num * xi_num * rho_nu_num * k_num * n_num * dim_3_nu_num
    )
    network_apelp_batch_A_sample_num = (
        dim_2_num * b_num * xi_num * rho_nu_num * k_num * n_num * dim_2_nu_num
    )
    network_apelp_batch_B_sample_num = (
        dim_3_num * b_num * xi_num * rho_nu_num * k_num * n_num * dim_3_nu_num
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

    ##### Calculate and save L for each artificial end-linked polymer
    ##### network parameter sample
    print("Calculating simulation box size", flush=True)

    network_auelp_batch_A_L_params_arr = (
        network_auelp_batch_A_sample_params_arr[[0, 1, 4, 5, 6, 7]]
    ) # sample, dim, rho_nu, k, n, nu
    network_auelp_batch_B_L_params_arr = (
        network_auelp_batch_B_sample_params_arr[[0, 1, 4, 5, 6, 7]]
    ) # sample, dim, rho_nu, k, n, nu
    network_apelp_batch_A_L_params_arr = (
        network_apelp_batch_A_sample_params_arr[[0, 1, 4, 5, 6, 7]]
    ) # sample, dim, rho_nu, k, n, nu
    network_apelp_batch_B_L_params_arr = (
        network_apelp_batch_B_sample_params_arr[[0, 1, 4, 5, 6, 7]]
    ) # sample, dim, rho_nu, k, n, nu
    # network_auelp_batch_A_L_params_arr = (
    #     network_auelp_batch_A_sample_params_arr[:, [0, 1, 4, 5, 6, 7]]
    # ) # sample, dim, rho_nu, k, n, nu
    # network_auelp_batch_B_L_params_arr = (
    #     network_auelp_batch_B_sample_params_arr[:, [0, 1, 4, 5, 6, 7]]
    # ) # sample, dim, rho_nu, k, n, nu
    # network_apelp_batch_A_L_params_arr = (
    #     network_apelp_batch_A_sample_params_arr[:, [0, 1, 4, 5, 6, 7]]
    # ) # sample, dim, rho_nu, k, n, nu
    # network_apelp_batch_B_L_params_arr = (
    #     network_apelp_batch_B_sample_params_arr[:, [0, 1, 4, 5, 6, 7]]
    # ) # sample, dim, rho_nu, k, n, nu
    network_auelp_batch_A_L_params_list = params_list_func(
        network_auelp_batch_A_L_params_arr)
    network_auelp_batch_B_L_params_list = params_list_func(
        network_auelp_batch_B_L_params_arr)
    network_apelp_batch_A_L_params_list = params_list_func(
        network_apelp_batch_A_L_params_arr)
    network_apelp_batch_B_L_params_list = params_list_func(
        network_apelp_batch_B_L_params_arr)
    network_auelp_batch_A_L_args = (
        [
            (network_auelp, date, batch_A, int(sample), int(dim), rho_nu, int(k), int(n), int(nu))
            for (sample, dim, rho_nu, k, n, nu) in network_auelp_batch_A_L_params_list
        ]
    )
    network_auelp_batch_B_L_args = (
        [
            (network_auelp, date, batch_B, int(sample), int(dim), rho_nu, int(k), int(n), int(nu))
            for (sample, dim, rho_nu, k, n, nu) in network_auelp_batch_B_L_params_list
        ]
    )
    network_apelp_batch_A_L_args = (
        [
            (network_apelp, date, batch_A, int(sample), int(dim), rho_nu, int(k), int(n), int(nu))
            for (sample, dim, rho_nu, k, n, nu) in network_apelp_batch_A_L_params_list
        ]
    )
    network_apelp_batch_B_L_args = (
        [
            (network_apelp, date, batch_B, int(sample), int(dim), rho_nu, int(k), int(n), int(nu))
            for (sample, dim, rho_nu, k, n, nu) in network_apelp_batch_B_L_params_list
        ]
    )
    random.shuffle(network_auelp_batch_A_L_args)
    random.shuffle(network_auelp_batch_B_L_args)
    random.shuffle(network_apelp_batch_A_L_args)
    random.shuffle(network_apelp_batch_B_L_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(run_aelp_L, network_auelp_batch_A_L_args)
        pool.map(run_aelp_L, network_auelp_batch_B_L_args)
        pool.map(run_aelp_L, network_apelp_batch_A_L_args)
        pool.map(run_aelp_L, network_apelp_batch_B_L_args)
    
    ##### Perform the initial node seeding procedure for each artificial
    ##### end-linked polymer network parameter sample
    print("Performing the initial node seeding", flush=True)

    max_try = 500

    network_auelp_batch_A_initial_node_seeding_params_arr = (
        network_auelp_batch_A_sample_config_params_arr[:, [0, 1, 2, 6, 9]]
    ) # sample, dim, b, n, config
    network_auelp_batch_B_initial_node_seeding_params_arr = (
        network_auelp_batch_B_sample_config_params_arr[:, [0, 1, 2, 6, 9]]
    ) # sample, dim, b, n, config
    network_apelp_batch_A_initial_node_seeding_params_arr = (
        network_apelp_batch_A_sample_config_params_arr[:, [0, 1, 2, 6, 9]]
    ) # sample, dim, b, n, config
    network_apelp_batch_B_initial_node_seeding_params_arr = (
        network_apelp_batch_B_sample_config_params_arr[:, [0, 1, 2, 6, 9]]
    ) # sample, dim, b, n, config
    network_auelp_batch_A_initial_node_seeding_params_list = params_list_func(
        network_auelp_batch_A_initial_node_seeding_params_arr)
    network_auelp_batch_B_initial_node_seeding_params_list = params_list_func(
        network_auelp_batch_B_initial_node_seeding_params_arr)
    network_apelp_batch_A_initial_node_seeding_params_list = params_list_func(
        network_apelp_batch_A_initial_node_seeding_params_arr)
    network_apelp_batch_B_initial_node_seeding_params_list = params_list_func(
        network_apelp_batch_B_initial_node_seeding_params_arr)
    network_auelp_batch_A_initial_node_seeding_args = (
        [
            (network_auelp, date, batch_A, int(sample), scheme, int(dim), b, int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in network_auelp_batch_A_initial_node_seeding_params_list
        ]
    )
    network_auelp_batch_B_initial_node_seeding_args = (
        [
            (network_auelp, date, batch_B, int(sample), scheme, int(dim), b, int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in network_auelp_batch_B_initial_node_seeding_params_list
        ]
    )
    network_apelp_batch_A_initial_node_seeding_args = (
        [
            (network_apelp, date, batch_A, int(sample), scheme, int(dim), b, int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in network_apelp_batch_A_initial_node_seeding_params_list
        ]
    )
    network_apelp_batch_B_initial_node_seeding_args = (
        [
            (network_apelp, date, batch_B, int(sample), scheme, int(dim), b, int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in network_apelp_batch_B_initial_node_seeding_params_list
        ]
    )
    random.shuffle(network_auelp_batch_A_initial_node_seeding_args)
    random.shuffle(network_auelp_batch_B_initial_node_seeding_args)
    random.shuffle(network_apelp_batch_A_initial_node_seeding_args)
    random.shuffle(network_apelp_batch_B_initial_node_seeding_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_initial_node_seeding,
            network_auelp_batch_A_initial_node_seeding_args)
        pool.map(
            run_initial_node_seeding,
            network_auelp_batch_B_initial_node_seeding_args)
        pool.map(
            run_initial_node_seeding,
            network_apelp_batch_A_initial_node_seeding_args)
        pool.map(
            run_initial_node_seeding,
            network_apelp_batch_B_initial_node_seeding_args)
    
    # Check to see if the number of seeded nodes, prhd_n, equals the
    # intended/specified number of nodes to be seeded, n. Continue to
    # the topology initialization procedure ONLY IF prhd_n = n. If
    # prhd_n != n for any specified network, then the code block
    # identifies which particular set(s) of network parameters
    # prhd_n != n occurred for.
    if scheme == "prhd":
        network_auelp_batch_A_prhd_n_vs_n = np.zeros(
            network_auelp_batch_A_sample_config_num)
        network_auelp_batch_B_prhd_n_vs_n = np.zeros(
            network_auelp_batch_B_sample_config_num)
        network_apelp_batch_A_prhd_n_vs_n = np.zeros(
            network_apelp_batch_A_sample_config_num)
        network_apelp_batch_B_prhd_n_vs_n = np.zeros(
            network_apelp_batch_B_sample_config_num)
        
        for indx in range(network_auelp_batch_A_sample_config_num):
            sample = int(network_auelp_batch_A_sample_config_params_arr[indx, 0])
            n = int(network_auelp_batch_A_sample_config_params_arr[indx, 6])
            config = int(network_auelp_batch_A_sample_config_params_arr[indx, 9])
            
            coords_filename = (
                config_filename_str(network_auelp, date, batch_A, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            
            if np.shape(coords)[0] == n:
                network_auelp_batch_A_prhd_n_vs_n[indx] = 1
            else: pass
        
        for indx in range(network_auelp_batch_B_sample_config_num):
            sample = int(network_auelp_batch_B_sample_config_params_arr[indx, 0])
            n = int(network_auelp_batch_B_sample_config_params_arr[indx, 6])
            config = int(network_auelp_batch_B_sample_config_params_arr[indx, 9])
            
            coords_filename = (
                config_filename_str(network_auelp, date, batch_B, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            
            if np.shape(coords)[0] == n:
                network_auelp_batch_B_prhd_n_vs_n[indx] = 1
            else: pass
        
        for indx in range(network_apelp_batch_A_sample_config_num):
            sample = int(network_apelp_batch_A_sample_config_params_arr[indx, 0])
            n = int(network_apelp_batch_A_sample_config_params_arr[indx, 6])
            config = int(network_apelp_batch_A_sample_config_params_arr[indx, 9])
            
            coords_filename = (
                config_filename_str(network_apelp, date, batch_A, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            
            if np.shape(coords)[0] == n:
                network_apelp_batch_A_prhd_n_vs_n[indx] = 1
            else: pass
        
        for indx in range(network_apelp_batch_B_sample_config_num):
            sample = int(network_apelp_batch_B_sample_config_params_arr[indx, 0])
            n = int(network_apelp_batch_B_sample_config_params_arr[indx, 6])
            config = int(network_apelp_batch_B_sample_config_params_arr[indx, 9])
            
            coords_filename = (
                config_filename_str(network_apelp, date, batch_B, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            
            if np.shape(coords)[0] == n:
                network_apelp_batch_B_prhd_n_vs_n[indx] = 1
            else: pass

        network_auelp_batch_A_sample_config_params_prhd_n_neq_n = (
            network_auelp_batch_A_sample_config_params_arr[np.where(network_auelp_batch_A_prhd_n_vs_n == 0)]
        )
        network_auelp_batch_B_sample_config_params_prhd_n_neq_n = (
            network_auelp_batch_B_sample_config_params_arr[np.where(network_auelp_batch_B_prhd_n_vs_n == 0)]
        )
        network_apelp_batch_A_sample_config_params_prhd_n_neq_n = (
            network_apelp_batch_A_sample_config_params_arr[np.where(network_apelp_batch_A_prhd_n_vs_n == 0)]
        )
        network_apelp_batch_B_sample_config_params_prhd_n_neq_n = (
            network_apelp_batch_B_sample_config_params_arr[np.where(network_apelp_batch_B_prhd_n_vs_n == 0)]
        )
        
        if np.shape(network_auelp_batch_A_sample_config_params_prhd_n_neq_n)[0] == 0:
            print_str = (
                "Success! prhd_n = n  for all ``auelp'' batch A network parameters!"
            )
            print(print_str, flush=True)
        elif np.shape(network_auelp_batch_A_sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of ``auelp'' batch A network "
                + "parameters. Repeat the periodic random hard disk node placement "
                + "procedure for the applicable set of ``auelp'' batch A network "
                + "parameters before continuing on to the topology initialization "
                + "procedure."
            )
            print(print_str, flush=True)
        if np.shape(network_auelp_batch_B_sample_config_params_prhd_n_neq_n)[0] == 0:
            print_str = (
                "Success! prhd_n = n  for all ``auelp'' batch B network parameters!"
            )
            print(print_str, flush=True)
        elif np.shape(network_auelp_batch_B_sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of ``auelp'' batch B network "
                + "parameters. Repeat the periodic random hard disk node placement "
                + "procedure for the applicable set of ``auelp'' batch B network "
                + "parameters before continuing on to the topology initialization "
                + "procedure."
            )
            print(print_str, flush=True)
        if np.shape(network_apelp_batch_A_sample_config_params_prhd_n_neq_n)[0] == 0:
            print_str = (
                "Success! prhd_n = n  for all ``apelp'' batch A network parameters!"
            )
            print(print_str, flush=True)
        elif np.shape(network_apelp_batch_A_sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of ``apelp'' batch A network "
                + "parameters. Repeat the periodic random hard disk node placement "
                + "procedure for the applicable set of ``apelp'' batch A network "
                + "parameters before continuing on to the topology initialization "
                + "procedure."
            )
            print(print_str, flush=True)
        if np.shape(network_apelp_batch_B_sample_config_params_prhd_n_neq_n)[0] == 0:
            print_str = (
                "Success! prhd_n = n  for all ``apelp'' batch B network parameters!"
            )
            print(print_str, flush=True)
        elif np.shape(network_apelp_batch_B_sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of ``apelp'' batch B network "
                + "parameters. Repeat the periodic random hard disk node placement "
                + "procedure for the applicable set of ``apelp'' batch B network "
                + "parameters before continuing on to the topology initialization "
                + "procedure."
            )
            print(print_str, flush=True)
    
    ##### Perform the network topology initialization procedure for each
    ##### artificial end-linked polymer network parameter sample
    print_str = (
        "Performing the artificial end-linked network topology "
        + "initialization procedure"
    )
    print(print_str, flush=True)

    network_auelp_batch_A_topology_params_arr = (
        np.delete(network_auelp_batch_A_sample_config_params_arr, 4, axis=1)
    ) # sample, dim, b, xi, k, n, nu, nu_max, config
    network_auelp_batch_B_topology_params_arr = (
        np.delete(network_auelp_batch_B_sample_config_params_arr, 4, axis=1)
    ) # sample, dim, b, xi, k, n, nu, nu_max, config
    network_apelp_batch_A_topology_params_arr = (
        np.delete(network_apelp_batch_A_sample_config_params_arr, 4, axis=1)
    ) # sample, dim, b, xi, k, n, nu, nu_max, config
    network_apelp_batch_B_topology_params_arr = (
        np.delete(network_apelp_batch_B_sample_config_params_arr, 4, axis=1)
    ) # sample, dim, b, xi, k, n, nu, nu_max, config
    network_auelp_batch_A_topology_params_list = params_list_func(
        network_auelp_batch_A_topology_params_arr)
    network_auelp_batch_B_topology_params_list = params_list_func(
        network_auelp_batch_B_topology_params_arr)
    network_apelp_batch_A_topology_params_list = params_list_func(
        network_apelp_batch_A_topology_params_arr)
    network_apelp_batch_B_topology_params_list = params_list_func(
        network_apelp_batch_B_topology_params_arr)
    network_auelp_batch_A_topology_args = (
        [
            (network_auelp, date, batch_A, int(sample), scheme, int(dim), b, xi, int(k), int(n), int(nu), int(nu_max), int(config), int(max_try))
            for (sample, dim, b, xi, k, n, nu, nu_max, config) in network_auelp_batch_A_topology_params_list
        ]
    )
    network_auelp_batch_B_topology_args = (
        [
            (network_auelp, date, batch_B, int(sample), scheme, int(dim), b, xi, int(k), int(n), int(nu), int(nu_max), int(config), int(max_try))
            for (sample, dim, b, xi, k, n, nu, nu_max, config) in network_auelp_batch_B_topology_params_list
        ]
    )
    network_apelp_batch_A_topology_args = (
        [
            (network_apelp, date, batch_A, int(sample), scheme, int(dim), b, xi, int(k), int(n), int(nu), int(nu_max), int(config), int(max_try))
            for (sample, dim, b, xi, k, n, nu, nu_max, config) in network_apelp_batch_A_topology_params_list
        ]
    )
    network_apelp_batch_B_topology_args = (
        [
            (network_apelp, date, batch_B, int(sample), scheme, int(dim), b, xi, int(k), int(n), int(nu), int(nu_max), int(config), int(max_try))
            for (sample, dim, b, xi, k, n, nu, nu_max, config) in network_apelp_batch_B_topology_params_list
        ]
    )
    random.shuffle(network_auelp_batch_A_topology_args)
    random.shuffle(network_auelp_batch_B_topology_args)
    random.shuffle(network_apelp_batch_A_topology_args)
    random.shuffle(network_apelp_batch_B_topology_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(run_aelp_network_topology, network_auelp_batch_A_topology_args)
        pool.map(run_aelp_network_topology, network_auelp_batch_B_topology_args)
        pool.map(run_aelp_network_topology, network_apelp_batch_A_topology_args)
        pool.map(run_aelp_network_topology, network_apelp_batch_B_topology_args)
    
    ##### Perform the additional node seeding procedure for each
    ##### artificial end-linked polymer network parameter sample
    print("Performing the additional node seeding procedure", flush=True)

    # Initialize an array to store the maximum number of nodes in the
    # initial network for each sample
    network_auelp_batch_A_sample_n_coords_max = np.empty(
        network_auelp_batch_A_sample_num, dtype=int)
    network_auelp_batch_B_sample_n_coords_max = np.empty(
        network_auelp_batch_B_sample_num, dtype=int)
    network_apelp_batch_A_sample_n_coords_max = np.empty(
        network_apelp_batch_A_sample_num, dtype=int)
    network_apelp_batch_B_sample_n_coords_max = np.empty(
        network_apelp_batch_B_sample_num, dtype=int)

    # Calculate maximum number of nodes in the initial network for each
    # sample
    for sample in range(network_auelp_batch_A_sample_num):
        sample_n_coords = np.asarray([], dtype=int)
        for config in range(config_num):
            coords_filename = (
                aelp_filename_str(network_auelp, date, batch_A, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            sample_n_coords = np.concatenate(
                (sample_n_coords, np.asarray([np.shape(coords)[0]])))
        network_auelp_batch_A_sample_n_coords_max[sample] = np.max(sample_n_coords)
    for sample in range(network_auelp_batch_B_sample_num):
        sample_n_coords = np.asarray([], dtype=int)
        for config in range(config_num):
            coords_filename = (
                aelp_filename_str(network_auelp, date, batch_B, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            sample_n_coords = np.concatenate(
                (sample_n_coords, np.asarray([np.shape(coords)[0]])))
        network_auelp_batch_B_sample_n_coords_max[sample] = np.max(sample_n_coords)
    for sample in range(network_apelp_batch_A_sample_num):
        sample_n_coords = np.asarray([], dtype=int)
        for config in range(config_num):
            coords_filename = (
                aelp_filename_str(network_apelp, date, batch_A, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            sample_n_coords = np.concatenate(
                (sample_n_coords, np.asarray([np.shape(coords)[0]])))
        network_apelp_batch_A_sample_n_coords_max[sample] = np.max(sample_n_coords)
    for sample in range(network_apelp_batch_B_sample_num):
        sample_n_coords = np.asarray([], dtype=int)
        for config in range(config_num):
            coords_filename = (
                aelp_filename_str(network_apelp, date, batch_B, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            sample_n_coords = np.concatenate(
                (sample_n_coords, np.asarray([np.shape(coords)[0]])))
        network_apelp_batch_B_sample_n_coords_max[sample] = np.max(sample_n_coords)

    # Populate the network sample configuration parameters array
    network_auelp_batch_A_sample_config_addtnl_n_params_arr = np.empty(
        (network_auelp_batch_A_sample_config_num, 5))
    sample = 0
    indx = 0
    for dim in np.nditer(dim_2_arr):
        for b in np.nditer(b_arr):
            for xi in np.nditer(xi_arr):
                for rho_nu in np.nditer(rho_nu_arr):
                    for k in np.nditer(k_arr):
                        for n in np.nditer(n_arr):
                            for nu in np.nditer(dim_2_nu_arr):
                                for config in np.nditer(config_arr):
                                    network_auelp_batch_A_sample_config_addtnl_n_params_arr[indx, :] = (
                                        np.asarray(
                                            [
                                                sample,
                                                dim,
                                                b,
                                                network_auelp_batch_A_sample_n_coords_max[sample],
                                                config
                                            ]
                                        )
                                    )
                                    indx += 1
                                sample += 1
    network_auelp_batch_B_sample_config_addtnl_n_params_arr = np.empty(
        (network_auelp_batch_B_sample_config_num, 5))
    sample = 0
    indx = 0
    for dim in np.nditer(dim_3_arr):
        for b in np.nditer(b_arr):
            for xi in np.nditer(xi_arr):
                for rho_nu in np.nditer(rho_nu_arr):
                    for k in np.nditer(k_arr):
                        for n in np.nditer(n_arr):
                            for nu in np.nditer(dim_3_nu_arr):
                                for config in np.nditer(config_arr):
                                    network_auelp_batch_B_sample_config_addtnl_n_params_arr[indx, :] = (
                                        np.asarray(
                                            [
                                                sample,
                                                dim,
                                                b,
                                                network_auelp_batch_B_sample_n_coords_max[sample],
                                                config
                                            ]
                                        )
                                    )
                                    indx += 1
                                sample += 1
    network_apelp_batch_A_sample_config_addtnl_n_params_arr = np.empty(
        (network_apelp_batch_A_sample_config_num, 5))
    sample = 0
    indx = 0
    for dim in np.nditer(dim_2_arr):
        for b in np.nditer(b_arr):
            for xi in np.nditer(xi_arr):
                for rho_nu in np.nditer(rho_nu_arr):
                    for k in np.nditer(k_arr):
                        for n in np.nditer(n_arr):
                            for nu in np.nditer(dim_2_nu_arr):
                                for config in np.nditer(config_arr):
                                    network_apelp_batch_A_sample_config_addtnl_n_params_arr[indx, :] = (
                                        np.asarray(
                                            [
                                                sample,
                                                dim,
                                                b,
                                                network_apelp_batch_A_sample_n_coords_max[sample],
                                                config
                                            ]
                                        )
                                    )
                                    indx += 1
                                sample += 1
    network_apelp_batch_B_sample_config_addtnl_n_params_arr = np.empty(
        (network_apelp_batch_B_sample_config_num, 5))
    sample = 0
    indx = 0
    for dim in np.nditer(dim_3_arr):
        for b in np.nditer(b_arr):
            for xi in np.nditer(xi_arr):
                for rho_nu in np.nditer(rho_nu_arr):
                    for k in np.nditer(k_arr):
                        for n in np.nditer(n_arr):
                            for nu in np.nditer(dim_3_nu_arr):
                                for config in np.nditer(config_arr):
                                    network_apelp_batch_B_sample_config_addtnl_n_params_arr[indx, :] = (
                                        np.asarray(
                                            [
                                                sample,
                                                dim,
                                                b,
                                                network_apelp_batch_B_sample_n_coords_max[sample],
                                                config
                                            ]
                                        )
                                    )
                                    indx += 1
                                sample += 1

    network_auelp_batch_A_additional_node_seeding_params_list = params_list_func(
        network_auelp_batch_A_sample_config_addtnl_n_params_arr)
    network_auelp_batch_B_additional_node_seeding_params_list = params_list_func(
        network_auelp_batch_B_sample_config_addtnl_n_params_arr)
    network_apelp_batch_A_additional_node_seeding_params_list = params_list_func(
        network_apelp_batch_A_sample_config_addtnl_n_params_arr)
    network_apelp_batch_B_additional_node_seeding_params_list = params_list_func(
        network_apelp_batch_B_sample_config_addtnl_n_params_arr)
    network_auelp_batch_A_additional_node_seeding_args = (
        [
            (network_auelp, date, batch_A, int(sample), scheme, int(dim), float(b), int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in network_auelp_batch_A_additional_node_seeding_params_list
        ]
    )
    network_auelp_batch_B_additional_node_seeding_args = (
        [
            (network_auelp, date, batch_B, int(sample), scheme, int(dim), float(b), int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in network_auelp_batch_B_additional_node_seeding_params_list
        ]
    )
    network_apelp_batch_A_additional_node_seeding_args = (
        [
            (network_apelp, date, batch_A, int(sample), scheme, int(dim), float(b), int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in network_apelp_batch_A_additional_node_seeding_params_list
        ]
    )
    network_apelp_batch_B_additional_node_seeding_args = (
        [
            (network_apelp, date, batch_B, int(sample), scheme, int(dim), float(b), int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in network_apelp_batch_B_additional_node_seeding_params_list
        ]
    )
    random.shuffle(network_auelp_batch_A_additional_node_seeding_args)
    random.shuffle(network_auelp_batch_B_additional_node_seeding_args)
    random.shuffle(network_apelp_batch_A_additional_node_seeding_args)
    random.shuffle(network_apelp_batch_B_additional_node_seeding_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_aelp_network_additional_node_seeding,
            network_auelp_batch_A_additional_node_seeding_args)
        pool.map(
            run_aelp_network_additional_node_seeding,
            network_auelp_batch_B_additional_node_seeding_args)
        pool.map(
            run_aelp_network_additional_node_seeding,
            network_apelp_batch_A_additional_node_seeding_args)
        pool.map(
            run_aelp_network_additional_node_seeding,
            network_apelp_batch_B_additional_node_seeding_args)
    
    ##### Reassign the node labels using the Hilbert space-filling curve
    ##### for each artificial end-linked polymer network
    print(
        "Reassigning node labels using the Hilbert space-filling curve",
        flush=True)

    network_auelp_batch_A_hilbert_node_label_assignment_params_arr = (
        network_auelp_batch_A_sample_config_params_arr[:, [0, 9]]
    ) # sample, config
    network_auelp_batch_B_hilbert_node_label_assignment_params_arr = (
        network_auelp_batch_B_sample_config_params_arr[:, [0, 9]]
    ) # sample, config
    network_apelp_batch_A_hilbert_node_label_assignment_params_arr = (
        network_apelp_batch_A_sample_config_params_arr[:, [0, 9]]
    ) # sample, config
    network_apelp_batch_B_hilbert_node_label_assignment_params_arr = (
        network_apelp_batch_B_sample_config_params_arr[:, [0, 9]]
    ) # sample, config
    network_auelp_batch_A_hilbert_node_label_assignment_params_list = params_list_func(
        network_auelp_batch_A_hilbert_node_label_assignment_params_arr)
    network_auelp_batch_B_hilbert_node_label_assignment_params_list = params_list_func(
        network_auelp_batch_B_hilbert_node_label_assignment_params_arr)
    network_apelp_batch_A_hilbert_node_label_assignment_params_list = params_list_func(
        network_apelp_batch_A_hilbert_node_label_assignment_params_arr)
    network_apelp_batch_B_hilbert_node_label_assignment_params_list = params_list_func(
        network_apelp_batch_B_hilbert_node_label_assignment_params_arr)
    network_auelp_batch_A_hilbert_node_label_assignment_args = (
        [
            (network_auelp, date, batch_A, int(sample), int(config))
            for (sample, config) in network_auelp_batch_A_hilbert_node_label_assignment_params_list
        ]
    )
    network_auelp_batch_B_hilbert_node_label_assignment_args = (
        [
            (network_auelp, date, batch_B, int(sample), int(config))
            for (sample, config) in network_auelp_batch_B_hilbert_node_label_assignment_params_list
        ]
    )
    network_apelp_batch_A_hilbert_node_label_assignment_args = (
        [
            (network_apelp, date, batch_A, int(sample), int(config))
            for (sample, config) in network_apelp_batch_A_hilbert_node_label_assignment_params_list
        ]
    )
    network_apelp_batch_B_hilbert_node_label_assignment_args = (
        [
            (network_apelp, date, batch_B, int(sample), int(config))
            for (sample, config) in network_apelp_batch_B_hilbert_node_label_assignment_params_list
        ]
    )
    random.shuffle(network_auelp_batch_A_hilbert_node_label_assignment_args)
    random.shuffle(network_auelp_batch_B_hilbert_node_label_assignment_args)
    random.shuffle(network_apelp_batch_A_hilbert_node_label_assignment_args)
    random.shuffle(network_apelp_batch_B_hilbert_node_label_assignment_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_aelp_network_hilbert_node_label_assignment,
            network_auelp_batch_A_hilbert_node_label_assignment_args)
        pool.map(
            run_aelp_network_hilbert_node_label_assignment,
            network_auelp_batch_B_hilbert_node_label_assignment_args)
        pool.map(
            run_aelp_network_hilbert_node_label_assignment,
            network_apelp_batch_A_hilbert_node_label_assignment_args)
        pool.map(
            run_aelp_network_hilbert_node_label_assignment,
            network_apelp_batch_B_hilbert_node_label_assignment_args)

if __name__ == "__main__":
    main()