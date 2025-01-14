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
    run_voronoi_L,
    run_initial_node_seeding,
    run_voronoi_network_topology
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

    ##### Load in Voronoi network topology configuration
    print("Loading in Voronoi network topology configuration", flush=True)

    # Initialization of identification information for this particular
    # batch of Voronoi-tessellated networks
    network = "voronoi"
    date = "20250102"
    batch_A = "A"
    batch_B = "B"
    scheme = "prhd"

    network_str = f"network_{network}"
    date_str = f"date_{date}"
    batch_A_str = f"batch_{batch_A}"
    batch_B_str = f"batch_{batch_B}"
    scheme_str = f"scheme_{scheme}"

    dim_str = "dim"
    b_str = "b"
    n_str = "n"
    eta_n_str = "eta_n"
    config_str = "config"

    filepath = filepath_str(network)
    batch_A_filename_prefix = filepath + f"{date}{batch_A}"
    batch_B_filename_prefix = filepath + f"{date}{batch_B}"

    batch_A_identifier_filename = batch_A_filename_prefix + "-identifier" + ".txt"
    batch_B_identifier_filename = batch_B_filename_prefix + "-identifier" + ".txt"
    batch_A_dim_filename = batch_A_filename_prefix + f"-{dim_str}" + ".dat"
    batch_B_dim_filename = batch_B_filename_prefix + f"-{dim_str}" + ".dat"
    batch_A_b_filename = batch_A_filename_prefix + f"-{b_str}" + ".dat"
    batch_B_b_filename = batch_B_filename_prefix + f"-{b_str}" + ".dat"
    batch_A_n_filename = batch_A_filename_prefix + f"-{n_str}" + ".dat"
    batch_B_n_filename = batch_B_filename_prefix + f"-{n_str}" + ".dat"
    batch_A_eta_n_filename = batch_A_filename_prefix + f"-{eta_n_str}" + ".dat"
    batch_B_eta_n_filename = batch_B_filename_prefix + f"-{eta_n_str}" + ".dat"
    batch_A_config_filename = batch_A_filename_prefix + f"-{config_str}" + ".dat"
    batch_B_config_filename = batch_B_filename_prefix + f"-{config_str}" + ".dat"
    batch_A_params_filename = batch_A_filename_prefix + "-params" + ".dat"
    batch_B_params_filename = batch_B_filename_prefix + "-params" + ".dat"
    batch_A_sample_params_filename = (
        batch_A_filename_prefix + "-sample_params" + ".dat"
    )
    batch_B_sample_params_filename = (
        batch_B_filename_prefix + "-sample_params" + ".dat"
    )
    batch_A_sample_config_params_filename = (
        batch_A_filename_prefix + "-sample_config_params" + ".dat"
    )
    batch_B_sample_config_params_filename = (
        batch_B_filename_prefix + "-sample_config_params" + ".dat"
    )

    batch_A_identifier_arr = np.loadtxt(
        batch_A_identifier_filename, dtype=str, usecols=0, ndmin=1)
    batch_B_identifier_arr = np.loadtxt(
        batch_B_identifier_filename, dtype=str, usecols=0, ndmin=1)
    dim_2_arr = np.loadtxt(batch_A_dim_filename, dtype=int, ndmin=1)
    dim_3_arr = np.loadtxt(batch_B_dim_filename, dtype=int, ndmin=1)
    b_arr = np.loadtxt(batch_A_b_filename, ndmin=1)
    dim_2_n_arr = np.loadtxt(batch_A_n_filename, dtype=int, ndmin=1)
    dim_3_n_arr = np.loadtxt(batch_B_n_filename, dtype=int, ndmin=1)
    eta_n_arr = np.loadtxt(batch_A_eta_n_filename, ndmin=1)
    config_arr = np.loadtxt(batch_A_config_filename, dtype=int, ndmin=1)

    batch_A_sample_params_arr = np.loadtxt(
        batch_A_sample_params_filename, ndmin=1)
    batch_B_sample_params_arr = np.loadtxt(
        batch_B_sample_params_filename, ndmin=1)
    batch_A_sample_config_params_arr = np.loadtxt(
        batch_A_sample_config_params_filename, ndmin=1)
    batch_B_sample_config_params_arr = np.loadtxt(
        batch_B_sample_config_params_filename, ndmin=1)

    dim_2_num = np.shape(dim_2_arr)[0]
    dim_3_num = np.shape(dim_3_arr)[0]
    b_num = np.shape(b_arr)[0]
    dim_2_n_num = np.shape(dim_2_n_arr)[0]
    dim_3_n_num = np.shape(dim_3_n_arr)[0]
    eta_n_num = np.shape(eta_n_arr)[0]
    config_num = np.shape(config_arr)[0]

    batch_A_sample_num = dim_2_num * b_num * dim_2_n_num * eta_n_num
    batch_B_sample_num = dim_3_num * b_num * dim_3_n_num * eta_n_num
    batch_A_sample_config_num = batch_A_sample_num * config_num
    batch_B_sample_config_num = batch_B_sample_num * config_num

    ##### Calculate and save L for each Voronoi-tessellated network
    ##### parameter sample
    print("Calculating simulation box size", flush=True)

    batch_A_voronoi_L_params_list = params_list_func(batch_A_sample_params_arr)
    batch_B_voronoi_L_params_list = params_list_func(batch_B_sample_params_arr)
    batch_A_voronoi_L_args = (
        [
            (network, date, batch_A, int(sample), int(dim), b, int(n), eta_n)
            for (sample, dim, b, n, eta_n) in batch_A_voronoi_L_params_list
        ]
    )
    batch_B_voronoi_L_args = (
        [
            (network, date, batch_B, int(sample), int(dim), b, int(n), eta_n)
            for (sample, dim, b, n, eta_n) in batch_B_voronoi_L_params_list
        ]
    )
    random.shuffle(batch_A_voronoi_L_args)
    random.shuffle(batch_B_voronoi_L_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(run_voronoi_L, batch_A_voronoi_L_args)
        pool.map(run_voronoi_L, batch_B_voronoi_L_args)
    
    ##### Perform the initial node seeding procedure for each
    ##### Voronoi-tessellated network parameter sample
    print("Performing the initial node seeding", flush=True)
    
    max_try = 500

    batch_A_initial_node_seeding_params_arr = (
        batch_A_sample_config_params_arr[:, [0, 1, 2, 3, 5]]
    ) # sample, dim, b, n, config
    batch_B_initial_node_seeding_params_arr = (
        batch_B_sample_config_params_arr[:, [0, 1, 2, 3, 5]]
    ) # sample, dim, b, n, config
    batch_A_initial_node_seeding_params_list = params_list_func(
        batch_A_initial_node_seeding_params_arr)
    batch_B_initial_node_seeding_params_list = params_list_func(
        batch_B_initial_node_seeding_params_arr)
    batch_A_initial_node_seeding_args = (
        [
            (network, date, batch_A, int(sample), scheme, int(dim), b, int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in batch_A_initial_node_seeding_params_list
        ]
    )
    batch_B_initial_node_seeding_args = (
        [
            (network, date, batch_B, int(sample), scheme, int(dim), b, int(n), int(config), int(max_try))
            for (sample, dim, b, n, config) in batch_B_initial_node_seeding_params_list
        ]
    )
    random.shuffle(batch_A_initial_node_seeding_args)
    random.shuffle(batch_B_initial_node_seeding_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(run_initial_node_seeding, batch_A_initial_node_seeding_args)
        pool.map(run_initial_node_seeding, batch_B_initial_node_seeding_args)
    
    # Check to see if the number of seeded nodes, prhd_n, equals the
    # intended/specified number of nodes to be seeded, n. Continue to
    # the topology initialization procedure ONLY IF prhd_n = n. If
    # prhd_n != n for any specified network, then the code block
    # identifies which particular set(s) of network parameters
    # prhd_n != n occurred for.
    if scheme == "prhd":
        batch_A_prhd_n_vs_n = np.zeros(batch_A_sample_config_num)
        batch_B_prhd_n_vs_n = np.zeros(batch_B_sample_config_num)

        for indx in range(batch_A_sample_config_num):
            sample = int(batch_A_sample_config_params_arr[indx, 0])
            n = int(batch_A_sample_config_params_arr[indx, 3])
            config = int(batch_A_sample_config_params_arr[indx, 5])
            
            coords_filename = (
                config_filename_str(network, date, batch_A, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            
            if np.shape(coords)[0] == n: batch_A_prhd_n_vs_n[indx] = 1
            else: pass
        for indx in range(batch_B_sample_config_num):
            sample = int(batch_B_sample_config_params_arr[indx, 0])
            n = int(batch_B_sample_config_params_arr[indx, 3])
            config = int(batch_B_sample_config_params_arr[indx, 5])
            
            coords_filename = (
                config_filename_str(network, date, batch_B, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            
            if np.shape(coords)[0] == n: batch_B_prhd_n_vs_n[indx] = 1
            else: pass

        batch_A_sample_config_params_prhd_n_neq_n = (
            batch_A_sample_config_params_arr[np.where(batch_A_prhd_n_vs_n == 0)]
        )
        batch_B_sample_config_params_prhd_n_neq_n = (
            batch_B_sample_config_params_arr[np.where(batch_B_prhd_n_vs_n == 0)]
        )

        if np.shape(batch_A_sample_config_params_prhd_n_neq_n)[0] == 0:
            print_str = "Success! prhd_n = n for batch A network parameters!"
            print(print_str, flush=True)
        elif np.shape(batch_A_sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of batch A network "
                + "parameters. Repeat the periodic random hard disk node "
                + "placement procedure for the applicable set of batch A "
                + "network parameters before continuing on to the topology "
                + "initialization procedure."
            )
            print(print_str, flush=True)
        if np.shape(batch_B_sample_config_params_prhd_n_neq_n)[0] == 0:
            print_str = "Success! prhd_n = n for batch B network parameters!"
            print(print_str, flush=True)
        elif np.shape(batch_B_sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of batch B network "
                + "parameters. Repeat the periodic random hard disk node "
                + "placement procedure for the applicable set of batch B "
                + "network parameters before continuing on to the topology "
                + "initialization procedure."
            )
            print(print_str, flush=True)
    
    ##### Perform the topology initialization procedure for each
    ##### Voronoi-tessellated network parameter sample
    print(
        "Performing the Voronoi network topology initialization procedure",
        flush=True)

    batch_A_voronoi_network_topology_params_arr = (
        batch_A_sample_config_params_arr[:, [0, 1, 3, 5]]
    ) # sample, dim, n, config
    batch_B_voronoi_network_topology_params_arr = (
        batch_B_sample_config_params_arr[:, [0, 1, 3, 5]]
    ) # sample, dim, n, config
    batch_A_voronoi_network_topology_params_list = params_list_func(
        batch_A_voronoi_network_topology_params_arr)
    batch_B_voronoi_network_topology_params_list = params_list_func(
        batch_B_voronoi_network_topology_params_arr)
    batch_A_voronoi_network_topology_args = (
        [
            (network, date, batch_A, int(sample), scheme, int(dim), int(n), int(config))
            for (sample, dim, n, config) in batch_A_voronoi_network_topology_params_list
        ]
    )
    batch_B_voronoi_network_topology_args = (
        [
            (network, date, batch_B, int(sample), scheme, int(dim), int(n), int(config))
            for (sample, dim, n, config) in batch_B_voronoi_network_topology_params_list
        ]
    )
    random.shuffle(batch_A_voronoi_network_topology_args)
    random.shuffle(batch_B_voronoi_network_topology_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_voronoi_network_topology, batch_A_voronoi_network_topology_args)
        pool.map(
            run_voronoi_network_topology, batch_B_voronoi_network_topology_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Voronoi network synthesis protocol took {execution_time} seconds to run")