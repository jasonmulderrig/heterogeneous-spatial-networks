import multiprocessing
import random
import numpy as np
from file_io import filepath_str
from multiprocessing_utils import run_aelp_network_topological_descriptor

def main():
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

    length_bound = 25

    ##### Calculate first set of artificial end-linked polymer network
    ##### topology descriptors
    print(
        "Calculating first set of artificial end-linked polymer network topology descriptors",
        flush=True)

    tplgcl_dscrptr_list = [
        "l", "l_cmpnts", "avrg_nn_k", "avrg_k_diff", "c", "lcl_avrg_kappa", "e",
        "avrg_d", "avrg_e", "n_bc", "m_bc", "cc", "scc"
    ]
    np_oprtn_list = ["", "mean"]
    eeel_ntwrk = True
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    network_auelp_batch_A_topological_descriptor_args = (
        [
            (network_auelp, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    network_auelp_batch_B_topological_descriptor_args = (
        [
            (network_auelp, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    network_apelp_batch_A_topological_descriptor_args = (
        [
            (network_apelp, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    network_apelp_batch_B_topological_descriptor_args = (
        [
            (network_apelp, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    random.shuffle(network_auelp_batch_A_topological_descriptor_args)
    random.shuffle(network_auelp_batch_B_topological_descriptor_args)
    random.shuffle(network_apelp_batch_A_topological_descriptor_args)
    random.shuffle(network_apelp_batch_B_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_aelp_network_topological_descriptor,
            network_auelp_batch_A_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_auelp_batch_B_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_apelp_batch_A_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_apelp_batch_B_topological_descriptor_args)
    
    ##### Calculate second set of artificial end-linked polymer network
    ##### topology descriptors
    print(
        "Calculating second set of artificial end-linked polymer network topology descriptors",
        flush=True)
    
    tplgcl_dscrptr_list = [
        "k", "k_diff", "kappa", "epsilon", "d"
    ]
    np_oprtn_list = ["bincount", "mean"]
    eeel_ntwrk = True
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    network_auelp_batch_A_topological_descriptor_args = (
        [
            (network_auelp, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    network_auelp_batch_B_topological_descriptor_args = (
        [
            (network_auelp, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    network_apelp_batch_A_topological_descriptor_args = (
        [
            (network_apelp, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    network_apelp_batch_B_topological_descriptor_args = (
        [
            (network_apelp, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
            for np_oprtn in np_oprtn_list
        ]
    )
    random.shuffle(network_auelp_batch_A_topological_descriptor_args)
    random.shuffle(network_auelp_batch_B_topological_descriptor_args)
    random.shuffle(network_apelp_batch_A_topological_descriptor_args)
    random.shuffle(network_apelp_batch_B_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_aelp_network_topological_descriptor,
            network_auelp_batch_A_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_auelp_batch_B_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_apelp_batch_A_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_apelp_batch_B_topological_descriptor_args)
    
    ##### Calculate third set of artificial end-linked polymer network
    ##### topology descriptors
    print(
        "Calculating third set of artificial end-linked polymer network topology descriptors",
        flush=True)
    
    tplgcl_dscrptr_list = [
        "n", "m", "rho_graph", "glbl_avrg_kappa", "lambda_1", "r_pearson",
        "r", "sigma", "lcl_e", "glbl_e"
    ]
    np_oprtn = ""
    eeel_ntwrk = True
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    network_auelp_batch_A_topological_descriptor_args = (
        [
            (network_auelp, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    network_auelp_batch_B_topological_descriptor_args = (
        [
            (network_auelp, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    network_apelp_batch_A_topological_descriptor_args = (
        [
            (network_apelp, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    network_apelp_batch_B_topological_descriptor_args = (
        [
            (network_apelp, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    random.shuffle(network_auelp_batch_A_topological_descriptor_args)
    random.shuffle(network_auelp_batch_B_topological_descriptor_args)
    random.shuffle(network_apelp_batch_A_topological_descriptor_args)
    random.shuffle(network_apelp_batch_B_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_aelp_network_topological_descriptor,
            network_auelp_batch_A_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_auelp_batch_B_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_apelp_batch_A_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_apelp_batch_B_topological_descriptor_args)
    
    ##### Calculate fourth set of artificial end-linked polymer network
    ##### topology descriptors
    print(
        "Calculating fourth set of artificial end-linked polymer network topology descriptors",
        flush=True)
    
    tplgcl_dscrptr_list = [
        "prop_eeel_n", "prop_eeel_m"
    ]
    np_oprtn = ""
    eeel_ntwrk = False
    save_tplgcl_dscrptr_result = True
    return_tplgcl_dscrptr_result = False

    network_auelp_batch_A_topological_descriptor_args = (
        [
            (network_auelp, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    network_auelp_batch_B_topological_descriptor_args = (
        [
            (network_auelp, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    network_apelp_batch_A_topological_descriptor_args = (
        [
            (network_apelp, date, batch_A, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_A_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    network_apelp_batch_B_topological_descriptor_args = (
        [
            (network_apelp, date, batch_B, int(sample), int(config), int(length_bound), tplgcl_dscrptr, np_oprtn, eeel_ntwrk, save_tplgcl_dscrptr_result, return_tplgcl_dscrptr_result)
            for sample in range(network_auelp_batch_B_sample_num)
            for config in range(config_num)
            for tplgcl_dscrptr in tplgcl_dscrptr_list
        ]
    )
    random.shuffle(network_auelp_batch_A_topological_descriptor_args)
    random.shuffle(network_auelp_batch_B_topological_descriptor_args)
    random.shuffle(network_apelp_batch_A_topological_descriptor_args)
    random.shuffle(network_apelp_batch_B_topological_descriptor_args)

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(
            run_aelp_network_topological_descriptor,
            network_auelp_batch_A_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_auelp_batch_B_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_apelp_batch_A_topological_descriptor_args)
        pool.map(
            run_aelp_network_topological_descriptor,
            network_apelp_batch_B_topological_descriptor_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial end-linked polymer network topology descriptors calculation took {execution_time} seconds to run")