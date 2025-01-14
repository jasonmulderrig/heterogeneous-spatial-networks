# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import multiprocessing
import random
import numpy as np
from file_io.file_io import filepath_str
from helpers.multiprocessing_utils import (
    run_aelp_network_topological_descriptor
)

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
    
    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    xi_num = np.shape(xi_arr)[0]
    rho_nu_num = np.shape(rho_nu_arr)[0]
    k_num = np.shape(k_arr)[0]
    n_num = np.shape(n_arr)[0]
    if nu_arr.ndim == 1: nu_num = 1
    else: nu_num = np.shape(nu_arr)[0]
    config_num = np.shape(config_arr)[0]

    sample_num = dim_num * b_num * xi_num * rho_nu_num * k_num * n_num * nu_num

    length_bound = 25

    ##### Calculate first set of artificial polydisperse end-linked
    ##### polymer network topology descriptors
    print(
        "Calculating first set of artificial polydisperse end-linked polymer network topology descriptors",
        flush=True)

    tplgcl_dscrptr_list = [
        "l", "l_cmpnts", "avrg_nn_k", "avrg_k_diff", "c", "lcl_avrg_kappa", "e",
        "avrg_d", "avrg_e", "n_bc", "m_bc", "cc", "scc"
    ]
    np_oprtn_list = ["", "mean"]
    eeel_ntwrk = True
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    topological_descriptor_args = (
        [
            (network, date, batch, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    random.shuffle(topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_aelp_network_topological_descriptor,
            topological_descriptor_args)
    
    ##### Calculate second set of artificial polydisperse end-linked
    ##### polymer network topology descriptors
    print(
        "Calculating second set of artificial polydisperse end-linked polymer network topology descriptors",
        flush=True)
    
    tplgcl_dscrptr_list = [
        "k", "k_diff", "kappa", "epsilon", "d"
    ]
    np_oprtn_list = ["bincount", "mean"]
    eeel_ntwrk = True
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    topological_descriptor_args = (
        [
            (network, date, batch, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    random.shuffle(topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_aelp_network_topological_descriptor,
            topological_descriptor_args)
    
    ##### Calculate third set of artificial polydisperse end-linked
    ##### polymer network topology descriptors
    print(
        "Calculating third set of artificial polydisperse end-linked polymer network topology descriptors",
        flush=True)
    
    tplgcl_dscrptr_list = [
        "n", "m", "rho_graph", "glbl_avrg_kappa", "lambda_1", "r_pearson",
        "r", "sigma", "lcl_e", "glbl_e"
    ]
    np_oprtn = ""
    eeel_ntwrk = True
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    topological_descriptor_args = (
        [
            (network, date, batch, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    random.shuffle(topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_aelp_network_topological_descriptor,
            topological_descriptor_args)
    
    ##### Calculate fourth set of artificial polydisperse end-linked
    ##### polymer network topology descriptors
    print(
        "Calculating fourth set of artificial polydisperse end-linked polymer network topology descriptors",
        flush=True)
    
    tplgcl_dscrptr_list = [
        "prop_eeel_n", "prop_eeel_m"
    ]
    np_oprtn = ""
    eeel_ntwrk = False
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    topological_descriptor_args = (
        [
            (network, date, batch, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    random.shuffle(topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_aelp_network_topological_descriptor,
            topological_descriptor_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial polydisperse end-linked polymer network topology descriptors calculation took {execution_time} seconds to run")