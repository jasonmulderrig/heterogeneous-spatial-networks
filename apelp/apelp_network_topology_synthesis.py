# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import multiprocessing
import random
import numpy as np
from file_io.file_io import (
    filepath_str,
    config_filename_str
)
from helpers.multiprocessing_utils import (
    run_aelp_L,
    run_initial_node_seeding,
    run_apelp_network_topology
)

def params_list_func(params_arr: np.ndarray) -> list[tuple]:
    if params_arr.ndim == 1:
        return [tuple(params_arr)]
    else:
        return list(map(tuple, params_arr))

def main():
    # This may or may not correspond to the number of cpus for optimal
    # parallelization performance. Feel free to modify if you see fit.
    cpu_num = int(np.floor(multiprocessing.cpu_count()/2))
    
    ##### Load in artificial polydisperse end-linked polymer network
    ##### topology configuration
    print(
        "Loading in artificial polydisperse end-linked polymer network topology configuration",
        flush=True)
    
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

    identifier_arr = np.loadtxt(identifier_filename, dtype=str, usecols=0, ndmin=1)
    dim_arr = np.loadtxt(dim_filename, dtype=int, ndmin=1)
    b_arr = np.loadtxt(b_filename, ndmin=1)
    xi_arr = np.loadtxt(xi_filename, ndmin=1)
    rho_nu_arr = np.loadtxt(rho_nu_filename, ndmin=1)
    k_arr = np.loadtxt(k_filename, dtype=int, ndmin=1)
    n_arr = np.loadtxt(n_filename, dtype=int, ndmin=1)
    nu_arr = np.loadtxt(nu_filename, dtype=int, ndmin=1)
    config_arr = np.loadtxt(config_filename, dtype=int, ndmin=1)
    
    sample_params_arr = np.loadtxt(sample_params_filename, ndmin=1)
    sample_config_params_arr = np.loadtxt(sample_config_params_filename, ndmin=1)
    
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
    sample_config_num = sample_num * config_num

    ##### Calculate and save L for each artificial polydisperse
    ##### end-linked polymer network parameter sample
    print("Calculating simulation box size", flush=True)

    if sample_params_arr.ndim == 1:
        L_params_arr = (
            sample_params_arr[[0, 1, 4, 5, 6, 7]]
        ) # sample, dim, rho_nu, k, n, nu
    else:
        L_params_arr = (
            sample_params_arr[:, [0, 1, 4, 5, 6, 7]]
        ) # sample, dim, rho_nu, k, n, nu
    L_params_list = params_list_func(L_params_arr)
    L_args = (
        [
            (network, date, batch, int(sample), int(dim), rho_nu, int(k), int(n), int(nu))
            for (sample, dim, rho_nu, k, n, nu) in L_params_list
        ]
    )
    random.shuffle(L_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(run_aelp_L, L_args)
    
    ##### Perform the initial node seeding procedure for each artificial
    ##### polydisperse end-linked polymer network parameter sample
    print("Performing the initial node seeding", flush=True)

    max_try = 500

    initial_node_seeding_params_arr = (
        sample_config_params_arr[:, [0, 1, 2, 6, 9]]
    ) # sample, dim, b, n, config
    initial_node_seeding_params_list = params_list_func(
        initial_node_seeding_params_arr)
    initial_node_seeding_args = (
        [
            (network, date, batch, int(sample), scheme, int(dim), b, int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in initial_node_seeding_params_list
        ]
    )
    random.shuffle(initial_node_seeding_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(run_initial_node_seeding, initial_node_seeding_args)
    
    # Check to see if the number of seeded nodes, prhd_n, equals the
    # intended/specified number of nodes to be seeded, n. Continue to
    # the topology initialization procedure ONLY IF prhd_n = n. If
    # prhd_n != n for any specified network, then the code block
    # identifies which particular set(s) of network parameters
    # prhd_n != n occurred for.
    if scheme == "prhd":
        prhd_n_vs_n = np.zeros(sample_config_num)
        
        for indx in range(sample_config_num):
            sample = int(sample_config_params_arr[indx, 0])
            n = int(sample_config_params_arr[indx, 6])
            config = int(sample_config_params_arr[indx, 9])
            
            coords_filename = (
                config_filename_str(network, date, batch, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            
            if np.shape(coords)[0] == n: prhd_n_vs_n[indx] = 1
            else: pass

        sample_config_params_prhd_n_neq_n = (
            sample_config_params_arr[np.where(prhd_n_vs_n == 0)]
        )
        
        if np.shape(sample_config_params_prhd_n_neq_n)[0] == 0:
            print_str = (
                "Success! prhd_n = n  for all apelp network parameters!"
            )
            print(print_str, flush=True)
        elif np.shape(sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of apelp network "
                + "parameters. Repeat the periodic random hard disk node "
                + "placement procedure for the applicable set of apelp "
                + "network parameters before continuing on to the "
                + "topology initialization procedure."
            )
            print(print_str, flush=True)
    
    ##### Perform the network topology initialization procedure for each
    ##### polydisperse artificial end-linked polymer network parameter
    ##### sample
    print_str = (
        "Performing the artificial polydisperse end-linked polymer "
        + "network topology initialization procedure"
    )
    print(print_str, flush=True)

    topology_params_arr = (
        np.delete(sample_config_params_arr, 4, axis=1)
    ) # sample, dim, b, xi, k, n, nu, nu_max, config
    topology_params_list = params_list_func(topology_params_arr)
    topology_args = (
        [
            (network, date, batch, int(sample), scheme, int(dim), b, xi, int(k), int(n), int(nu), int(nu_max), int(config), int(max_try))
            for (sample, dim, b, xi, k, n, nu, nu_max, config) in topology_params_list
        ]
    )
    random.shuffle(topology_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(run_apelp_network_topology, topology_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial polydisperse end-linked polymer network synthesis protocol took {execution_time} seconds to run")