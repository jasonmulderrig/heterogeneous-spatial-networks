import multiprocessing
import random
import numpy as np
from file_io import filepath_str
from multiprocessing_utils import run_voronoi_network_topological_descriptor

def main():
    # This may or may not correspond to the number of cpus for optimal
    # parallelization performance. Feel free to modify if you see fit.
    cpu_num = int(np.floor(multiprocessing.cpu_count()/2))

    # Initialization of identification information for this particular
    # batch of Voronoi-tessellated networks
    network = "voronoi"
    date = "20241210"
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

    length_bound = 12

    ##### Calculate first set of Voronoi network topology descriptors
    print(
        "Calculating first set of Voronoi network topology descriptors",
        flush=True)

    tplgcl_dscrptr_list = [
        "l", "l_cmpnts", "avrg_nn_k", "avrg_k_diff", "c", "lcl_avrg_kappa", "e",
        "avrg_d", "avrg_e", "n_bc", "m_bc", "cc", "scc"
    ]
    np_oprtn_list = ["", "mean"]
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    batch_A_voronoi_network_topological_descriptor_args = (
        [
            (network, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    batch_B_voronoi_network_topological_descriptor_args = (
        [
            (network, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    random.shuffle(batch_A_voronoi_network_topological_descriptor_args)
    random.shuffle(batch_B_voronoi_network_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_voronoi_network_topological_descriptor,
            batch_A_voronoi_network_topological_descriptor_args)
        pool.map(
            run_voronoi_network_topological_descriptor,
            batch_B_voronoi_network_topological_descriptor_args)
    
    ##### Calculate second set of Voronoi network topology descriptors
    print(
        "Calculating second set of Voronoi network topology descriptors",
        flush=True)
    tplgcl_dscrptr_list = [
        "k", "k_diff", "kappa", "epsilon", "d"
    ]
    np_oprtn_list = ["bincount", "mean"]
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    batch_A_voronoi_network_topological_descriptor_args = (
        [
            (network, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    batch_B_voronoi_network_topological_descriptor_args = (
        [
            (network, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    random.shuffle(batch_A_voronoi_network_topological_descriptor_args)
    random.shuffle(batch_B_voronoi_network_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_voronoi_network_topological_descriptor,
            batch_A_voronoi_network_topological_descriptor_args)
        pool.map(
            run_voronoi_network_topological_descriptor,
            batch_B_voronoi_network_topological_descriptor_args)
    
    ##### Calculate third set of Voronoi network topology descriptors
    print(
        "Calculating third set of Voronoi network topology descriptors",
        flush=True)
    tplgcl_dscrptr_list = [
        "n", "m", "rho_graph", "glbl_avrg_kappa", "lambda_1", "r", "sigma",
        "lcl_e", "glbl_e"
    ]
    np_oprtn = ""
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    batch_A_voronoi_network_topological_descriptor_args = (
        [
            (network, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    batch_B_voronoi_network_topological_descriptor_args = (
        [
            (network, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    random.shuffle(batch_A_voronoi_network_topological_descriptor_args)
    random.shuffle(batch_B_voronoi_network_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_voronoi_network_topological_descriptor,
            batch_A_voronoi_network_topological_descriptor_args)
        pool.map(
            run_voronoi_network_topological_descriptor,
            batch_B_voronoi_network_topological_descriptor_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Voronoi network topology descriptors calculation took {execution_time} seconds to run")