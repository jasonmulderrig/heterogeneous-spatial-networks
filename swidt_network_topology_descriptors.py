import multiprocessing
import random
import numpy as np
from file_io import filepath_str
from multiprocessing_utils import run_swidt_network_topological_descriptor

def main():
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

    length_bound = 25

    ##### Calculate first set of spider web-inspired Delaunay network
    ##### topology descriptors
    print(
        "Calculating first set of spider web-inspired Delaunay network topology descriptors",
        flush=True)

    tplgcl_dscrptr_list = [
        "l", "l_cmpnts", "avrg_nn_k", "avrg_k_diff", "c", "lcl_avrg_kappa", "e",
        "avrg_d", "avrg_e", "n_bc", "m_bc", "cc", "scc"
    ]
    np_oprtn_list = ["", "mean"]
    eeel_ntwrk = True
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    swidt_network_topological_descriptor_args = (
        [
            (network, date, batch, int(sample), int(config), int(pruning), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(sample_num)
            for config in range(config_num)
            for pruning in range(pruning_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    random.shuffle(swidt_network_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_swidt_network_topological_descriptor,
            swidt_network_topological_descriptor_args)
    
    ##### Calculate second set of spider web-inspired Delaunay network
    ##### topology descriptors
    print(
        "Calculating second set of spider web-inspired Delaunay network topology descriptors",
        flush=True)
    
    tplgcl_dscrptr_list = [
        "k", "k_diff", "kappa", "epsilon", "d"
    ]
    np_oprtn_list = ["bincount", "mean"]
    eeel_ntwrk = True
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False
    
    swidt_network_topological_descriptor_args = (
        [
            (network, date, batch, int(sample), int(config), int(pruning), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(sample_num)
            for config in range(config_num)
            for pruning in range(pruning_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    random.shuffle(swidt_network_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_swidt_network_topological_descriptor,
            swidt_network_topological_descriptor_args)
    
    ##### Calculate third set of spider web-inspired Delaunay network
    ##### topology descriptors
    print(
        "Calculating third set of spider web-inspired Delaunay network topology descriptors",
        flush=True)
    
    tplgcl_dscrptr_list = [
        "n", "m", "rho_graph", "glbl_avrg_kappa", "lambda_1", "r_pearson",
        "r", "sigma", "lcl_e", "glbl_e"
    ]
    np_oprtn = ""
    eeel_ntwrk = True
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    swidt_network_topological_descriptor_args = (
        [
            (network, date, batch, int(sample), int(config), int(pruning), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(sample_num)
            for config in range(config_num)
            for pruning in range(pruning_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    random.shuffle(swidt_network_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_swidt_network_topological_descriptor,
            swidt_network_topological_descriptor_args)
    
    ##### Calculate fourth set of spider web-inspired Delaunay network
    ##### topology descriptors
    print(
        "Calculating fourth set of spider web-inspired Delaunay network topology descriptors",
        flush=True)
    
    tplgcl_dscrptr_list = [
        "prop_eeel_n", "prop_eeel_m"
    ]
    np_oprtn = ""
    eeel_ntwrk = False
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    swidt_network_topological_descriptor_args = (
        [
            (network, date, batch, int(sample), int(config), int(pruning), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(sample_num)
            for config in range(config_num)
            for pruning in range(pruning_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    random.shuffle(swidt_network_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_swidt_network_topological_descriptor,
            swidt_network_topological_descriptor_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Spider web-inspired Delaunay network topology descriptors calculation took {execution_time} seconds to run")