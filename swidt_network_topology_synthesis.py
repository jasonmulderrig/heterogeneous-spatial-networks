import multiprocessing
import random
import numpy as np
from file_io import (
    filepath_str,
    config_filename_str
)
from multiprocessing_utils import (
    run_swidt_L,
    run_initial_node_seeding,
    run_swidt_network_topology,
    run_swidt_network_edge_pruning_procedure
)

def params_list_func(params_arr: np.ndarray) -> list[tuple]:
    if params_arr.ndim == 1:
        return [tuple(params_arr)]
    else:
        return list(map(tuple, params_arr))

def main():
    ##### Load in spider web-inspired Delaunay network topology
    ##### configuration
    print_str = (
            "Loading in spider web-inspired Delaunay network topology "
            + "configuration"
        )
    print(print_str, flush=True)
    
    # This may or may not correspond to the number of cpus for optimal
    # parallelization performance. Feel free to modify if you see fit.
    cpu_num = int(np.floor(multiprocessing.cpu_count()/2))

    # Initialization of identification information for this particular
    # batch of Delaunay-triangulated networks
    network = "swidt"
    date = "20250102"
    batch = "A"
    scheme = "prhd"

    network_str = f"network_{network}"
    date_str = f"date_{date}"
    batch_str = f"batch_{batch}"
    scheme_str = f"scheme_{scheme}"

    dim_str = "dim"
    b_str = "b"
    n_str = "n"
    k_str = "k"
    eta_n_str = "eta_n"
    config_str = "config"
    pruning_str = "pruning"

    filepath = filepath_str(network)
    filename_prefix = filepath + f"{date}{batch}"

    identifier_filename = filename_prefix + "-identifier" + ".txt"
    dim_filename = filename_prefix + f"-{dim_str}" + ".dat"
    b_filename = filename_prefix + f"-{b_str}" + ".dat"
    n_filename = filename_prefix + f"-{n_str}" + ".dat"
    k_filename = filename_prefix + f"-{k_str}" + ".dat"
    eta_n_filename = filename_prefix + f"-{eta_n_str}" + ".dat"
    config_filename = filename_prefix + f"-{config_str}" + ".dat"
    pruning_filename = filename_prefix + f"-{pruning_str}" + ".dat"
    params_filename = filename_prefix + "-params" + ".dat"
    sample_params_filename = filename_prefix + "-sample_params" + ".dat"
    sample_config_params_filename = (
        filename_prefix + "-sample_config_params" + ".dat"
    )
    sample_config_pruning_params_filename = (
        filename_prefix + "-sample_config_pruning_params" + ".dat"
    )

    identifier_arr = np.loadtxt(
        identifier_filename, dtype=str, usecols=0, ndmin=1)
    dim_arr = np.loadtxt(dim_filename, dtype=int, ndmin=1)
    b_arr = np.loadtxt(b_filename, ndmin=1)
    n_arr = np.loadtxt(n_filename, dtype=int, ndmin=1)
    k_arr = np.loadtxt(k_filename, dtype=int, ndmin=1)
    eta_n_arr = np.loadtxt(eta_n_filename, ndmin=1)
    config_arr = np.loadtxt(config_filename, dtype=int, ndmin=1)
    pruning_arr = np.loadtxt(pruning_filename, dtype=int, ndmin=1)

    sample_params_arr = np.loadtxt(sample_params_filename, ndmin=1)
    sample_config_params_arr = np.loadtxt(sample_config_params_filename, ndmin=1)
    sample_config_pruning_params_arr = np.loadtxt(
        sample_config_pruning_params_filename, ndmin=1)

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    n_num = np.shape(n_arr)[0]
    k_num = np.shape(k_arr)[0]
    eta_n_num = np.shape(eta_n_arr)[0]
    config_num = np.shape(config_arr)[0]
    pruning_num = np.shape(pruning_arr)[0]

    sample_num = dim_num * b_num * n_num * k_num * eta_n_num
    sample_config_num = sample_num * config_num
    sample_config_pruning_num = sample_num * config_num * pruning_num

    ##### Calculate and save L for each spider web-inspired
    ##### Delaunay-triangulated network parameter sample
    print("Calculating simulation box size", flush=True)

    swidt_L_params_arr = (
        sample_params_arr[:, [0, 1, 2, 3, 5]]
    ) # sample, dim, b, n, eta_n
    swidt_L_params_list = params_list_func(swidt_L_params_arr)
    swidt_L_args = (
        [
            (network, date, batch, int(sample), int(dim), b, int(n), eta_n)
            for (sample, dim, b, n, eta_n) in swidt_L_params_list
        ]
    )
    random.shuffle(swidt_L_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(run_swidt_L, swidt_L_args)
    
    ##### Perform the initial node seeding procedure for each spider
    ##### web-inspired Delaunay-triangulated network parameter sample
    print("Performing the initial node seeding", flush=True)

    max_try = 500

    initial_node_seeding_params_arr = (
        sample_config_params_arr[:, [0, 1, 2, 3, 6]]
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
            n = int(sample_config_params_arr[indx, 3])
            config = int(sample_config_params_arr[indx, 6])
            
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
            print_str = "Success! prhd_n = n for all network parameters!"
            print(print_str, flush=True)
        elif np.shape(sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of network parameters. "
                + "Repeat the periodic random hard disk node placement "
                + "procedure for the applicable set of network parameters before "
                + "continuing on to the topology initialization procedure."
            )
            print(print_str, flush=True)
    
    ##### Perform the topology initialization procedure for each spider
    ##### web-inspired Delaunay-triangulated network parameter sample
    print_str = (
            "Performing the spider web-inspired Delaunay network "
            + "topology initialization procedure"
        )
    print(print_str, flush=True)

    swidt_network_topology_params_arr = (
        sample_config_params_arr[:, [0, 1, 3, 6]]
    ) # sample, dim, n, config
    swidt_network_topology_params_list = params_list_func(
        swidt_network_topology_params_arr)
    swidt_network_topology_args = (
        [
            (network, date, batch, int(sample), scheme, int(dim), int(n), int(config))
            for (sample, dim, n, config) in swidt_network_topology_params_list
        ]
    )
    random.shuffle(swidt_network_topology_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(run_swidt_network_topology, swidt_network_topology_args)
    
    ##### Perform the edge pruning procedure for each spider
    ##### web-inspired Delaunay-triangulated network parameter sample
    print_str = (
            "Performing the spider web-inspired Delaunay network "
            + "topology edge pruning procedure"
        )
    print(print_str, flush=True)

    swidt_network_edge_pruning_procedure_params_arr = (
        sample_config_pruning_params_arr[:, [0, 3, 4, 6, 7]]
    ) # sample, n, k, config, pruning
    swidt_network_edge_pruning_procedure_params_list = params_list_func(
        swidt_network_edge_pruning_procedure_params_arr)
    swidt_network_edge_pruning_procedure_args = (
        [
            (network, date, batch, int(sample), int(n), int(k), int(config), int(pruning))
            for (sample, n, k, config, pruning) in swidt_network_edge_pruning_procedure_params_list
        ]
    )
    random.shuffle(swidt_network_edge_pruning_procedure_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_swidt_network_edge_pruning_procedure,
            swidt_network_edge_pruning_procedure_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Spider web-inspired Delaunay network synthesis protocol took {execution_time} seconds to run")