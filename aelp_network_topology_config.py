import numpy as np
from file_io import filepath_str

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

    network_auelp_batch_A_identifier_arr = np.asarray(
        [
            network_auelp_str,
            date_str,
            batch_A_str,
            scheme_str,
            dim_str,
            b_str,
            xi_str,
            rho_nu_str,
            k_str,
            n_str,
            nu_str,
            nu_max_str
        ]
    )
    network_auelp_batch_B_identifier_arr = np.asarray(
        [
            network_auelp_str,
            date_str,
            batch_B_str,
            scheme_str,
            dim_str,
            b_str,
            xi_str,
            rho_nu_str,
            k_str,
            n_str,
            nu_str,
            nu_max_str
        ]
    )
    network_apelp_batch_A_identifier_arr = np.asarray(
        [
            network_apelp_str,
            date_str,
            batch_A_str,
            scheme_str,
            dim_str,
            b_str,
            xi_str,
            rho_nu_str,
            k_str,
            n_str,
            nu_str,
            nu_max_str
        ]
    )
    network_apelp_batch_B_identifier_arr = np.asarray(
        [
            network_apelp_str,
            date_str,
            batch_B_str,
            scheme_str,
            dim_str,
            b_str,
            xi_str,
            rho_nu_str,
            k_str,
            n_str,
            nu_str,
            nu_max_str
        ]
    )

    # Initialization of fundamental parameters for artificial end-linked
    # polymer networks
    dim_2_arr = np.asarray([2], dtype=int)
    dim_3_arr = np.asarray([3], dtype=int)
    b_arr = np.asarray([1.0])
    xi_arr = np.asarray([0.98])
    rho_nu_arr = np.asarray([0.85])
    k_arr = np.asarray([4], dtype=int)
    n_arr = np.asarray([100], dtype=int)
    dim_2_nu_arr = np.asarray([50], dtype=int)
    dim_3_nu_arr = np.asarray([50], dtype=int)
    nu_max_multiplier = 10
    dim_2_nu_max_arr = dim_2_nu_arr * nu_max_multiplier
    dim_3_nu_max_arr = dim_3_nu_arr * nu_max_multiplier
    config_arr = np.arange(200, dtype=int) # 0, 1, 2, ..., 198, 199

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

    # Populate the network parameters array
    network_auelp_batch_A_params_arr = np.empty(
        (network_auelp_batch_A_sample_num, 8))
    sample = 0
    for dim in np.nditer(dim_2_arr):
        for b in np.nditer(b_arr):
            for xi in np.nditer(xi_arr):
                for rho_nu in np.nditer(rho_nu_arr):
                    for k in np.nditer(k_arr):
                        for n in np.nditer(n_arr):
                            for nu in np.nditer(dim_2_nu_arr):
                                network_auelp_batch_A_params_arr[sample, :] = (
                                    np.asarray(
                                        [
                                            dim,
                                            b,
                                            xi,
                                            rho_nu,
                                            k,
                                            n,
                                            nu,
                                            nu * nu_max_multiplier
                                        ]
                                    )
                                )
                                sample += 1
    network_auelp_batch_B_params_arr = np.empty(
        (network_auelp_batch_B_sample_num, 8))
    sample = 0
    for dim in np.nditer(dim_3_arr):
        for b in np.nditer(b_arr):
            for xi in np.nditer(xi_arr):
                for rho_nu in np.nditer(rho_nu_arr):
                    for k in np.nditer(k_arr):
                        for n in np.nditer(n_arr):
                            for nu in np.nditer(dim_3_nu_arr):
                                network_auelp_batch_B_params_arr[sample, :] = (
                                    np.asarray(
                                        [
                                            dim,
                                            b,
                                            xi,
                                            rho_nu,
                                            k,
                                            n,
                                            nu,
                                            nu * nu_max_multiplier
                                        ]
                                    )
                                )
                                sample += 1
    network_apelp_batch_A_params_arr = network_auelp_batch_A_params_arr.copy()
    network_apelp_batch_B_params_arr = network_auelp_batch_B_params_arr.copy()

    # Populate the network sample parameters array
    network_auelp_batch_A_sample_params_arr = np.empty(
        (network_auelp_batch_A_sample_num, 9))
    sample = 0
    for dim in np.nditer(dim_2_arr):
        for b in np.nditer(b_arr):
            for xi in np.nditer(xi_arr):
                for rho_nu in np.nditer(rho_nu_arr):
                    for k in np.nditer(k_arr):
                        for n in np.nditer(n_arr):
                            for nu in np.nditer(dim_2_nu_arr):
                                network_auelp_batch_A_sample_params_arr[sample, :] = (
                                    np.asarray(
                                        [
                                            sample,
                                            dim,
                                            b,
                                            xi,
                                            rho_nu,
                                            k,
                                            n,
                                            nu,
                                            nu * nu_max_multiplier
                                        ]
                                    )
                                )
                                sample += 1
    network_auelp_batch_B_sample_params_arr = np.empty(
        (network_auelp_batch_B_sample_num, 9))
    sample = 0
    for dim in np.nditer(dim_3_arr):
        for b in np.nditer(b_arr):
            for xi in np.nditer(xi_arr):
                for rho_nu in np.nditer(rho_nu_arr):
                    for k in np.nditer(k_arr):
                        for n in np.nditer(n_arr):
                            for nu in np.nditer(dim_3_nu_arr):
                                network_auelp_batch_B_sample_params_arr[sample, :] = (
                                    np.asarray(
                                        [
                                            sample,
                                            dim,
                                            b,
                                            xi,
                                            rho_nu,
                                            k,
                                            n,
                                            nu,
                                            nu * nu_max_multiplier
                                        ]
                                    )
                                )
                                sample += 1
    network_apelp_batch_A_sample_params_arr = (
        network_auelp_batch_A_sample_params_arr.copy()
    )
    network_apelp_batch_B_sample_params_arr = (
        network_auelp_batch_B_sample_params_arr.copy()
    )

    # Populate the network sample configuration parameters array
    network_auelp_batch_A_sample_config_params_arr = np.empty(
        (network_auelp_batch_A_sample_config_num, 10))
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
                                    network_auelp_batch_A_sample_config_params_arr[indx, :] = (
                                        np.asarray(
                                            [
                                                sample,
                                                dim,
                                                b,
                                                xi,
                                                rho_nu,
                                                k,
                                                n,
                                                nu,
                                                nu * nu_max_multiplier,
                                                config
                                            ]
                                        )
                                    )
                                    indx += 1
                                sample += 1
    network_auelp_batch_B_sample_config_params_arr = np.empty(
        (network_auelp_batch_B_sample_config_num, 10))
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
                                    network_auelp_batch_B_sample_config_params_arr[indx, :] = (
                                        np.asarray(
                                            [
                                                sample,
                                                dim,
                                                b,
                                                xi,
                                                rho_nu,
                                                k,
                                                n,
                                                nu,
                                                nu * nu_max_multiplier,
                                                config
                                            ]
                                        )
                                    )
                                    indx += 1
                                sample += 1
    network_apelp_batch_A_sample_config_params_arr = (
        network_auelp_batch_A_sample_config_params_arr.copy()
    )
    network_apelp_batch_B_sample_config_params_arr = (
        network_auelp_batch_B_sample_config_params_arr.copy()
    )

    # Save identification information and fundamental network parameters
    np.savetxt(
        network_auelp_batch_A_identifier_filename,
        network_auelp_batch_A_identifier_arr, fmt="%s")
    np.savetxt(
        network_auelp_batch_B_identifier_filename,
        network_auelp_batch_B_identifier_arr, fmt="%s")
    np.savetxt(
        network_apelp_batch_A_identifier_filename,
        network_apelp_batch_A_identifier_arr, fmt="%s")
    np.savetxt(
        network_apelp_batch_B_identifier_filename,
        network_apelp_batch_B_identifier_arr, fmt="%s")
    np.savetxt(network_auelp_batch_A_dim_filename, dim_2_arr, fmt="%d")
    np.savetxt(network_auelp_batch_B_dim_filename, dim_3_arr, fmt="%d")
    np.savetxt(network_apelp_batch_A_dim_filename, dim_2_arr, fmt="%d")
    np.savetxt(network_apelp_batch_B_dim_filename, dim_3_arr, fmt="%d")
    np.savetxt(network_auelp_batch_A_b_filename, b_arr)
    np.savetxt(network_auelp_batch_B_b_filename, b_arr)
    np.savetxt(network_apelp_batch_A_b_filename, b_arr)
    np.savetxt(network_apelp_batch_B_b_filename, b_arr)
    np.savetxt(network_auelp_batch_A_xi_filename, xi_arr)
    np.savetxt(network_auelp_batch_B_xi_filename, xi_arr)
    np.savetxt(network_apelp_batch_A_xi_filename, xi_arr)
    np.savetxt(network_apelp_batch_B_xi_filename, xi_arr)
    np.savetxt(network_auelp_batch_A_rho_nu_filename, rho_nu_arr)
    np.savetxt(network_auelp_batch_B_rho_nu_filename, rho_nu_arr)
    np.savetxt(network_apelp_batch_A_rho_nu_filename, rho_nu_arr)
    np.savetxt(network_apelp_batch_B_rho_nu_filename, rho_nu_arr)
    np.savetxt(network_auelp_batch_A_k_filename, k_arr, fmt="%d")
    np.savetxt(network_auelp_batch_B_k_filename, k_arr, fmt="%d")
    np.savetxt(network_apelp_batch_A_k_filename, k_arr, fmt="%d")
    np.savetxt(network_apelp_batch_B_k_filename, k_arr, fmt="%d")
    np.savetxt(network_auelp_batch_A_n_filename, n_arr, fmt="%d")
    np.savetxt(network_auelp_batch_B_n_filename, n_arr, fmt="%d")
    np.savetxt(network_apelp_batch_A_n_filename, n_arr, fmt="%d")
    np.savetxt(network_apelp_batch_B_n_filename, n_arr, fmt="%d")
    np.savetxt(network_auelp_batch_A_nu_filename, dim_2_nu_arr, fmt="%d")
    np.savetxt(network_auelp_batch_B_nu_filename, dim_3_nu_arr, fmt="%d")
    np.savetxt(network_apelp_batch_A_nu_filename, dim_2_nu_arr, fmt="%d")
    np.savetxt(network_apelp_batch_B_nu_filename, dim_3_nu_arr, fmt="%d")
    np.savetxt(network_auelp_batch_A_nu_max_filename, dim_2_nu_max_arr, fmt="%d")
    np.savetxt(network_auelp_batch_B_nu_max_filename, dim_3_nu_max_arr, fmt="%d")
    np.savetxt(network_apelp_batch_A_nu_max_filename, dim_2_nu_max_arr, fmt="%d")
    np.savetxt(network_apelp_batch_B_nu_max_filename, dim_3_nu_max_arr, fmt="%d")
    np.savetxt(network_auelp_batch_A_config_filename, config_arr, fmt="%d")
    np.savetxt(network_auelp_batch_B_config_filename, config_arr, fmt="%d")
    np.savetxt(network_apelp_batch_A_config_filename, config_arr, fmt="%d")
    np.savetxt(network_apelp_batch_B_config_filename, config_arr, fmt="%d")
    np.savetxt(
        network_auelp_batch_A_params_filename, network_auelp_batch_A_params_arr)
    np.savetxt(
        network_auelp_batch_B_params_filename, network_auelp_batch_B_params_arr)
    np.savetxt(
        network_apelp_batch_A_params_filename, network_apelp_batch_A_params_arr)
    np.savetxt(
        network_apelp_batch_B_params_filename, network_apelp_batch_B_params_arr)
    np.savetxt(
        network_auelp_batch_A_sample_params_filename,
        network_auelp_batch_A_sample_params_arr)
    np.savetxt(
        network_auelp_batch_B_sample_params_filename,
        network_auelp_batch_B_sample_params_arr)
    np.savetxt(
        network_apelp_batch_A_sample_params_filename,
        network_apelp_batch_A_sample_params_arr)
    np.savetxt(
        network_apelp_batch_B_sample_params_filename,
        network_apelp_batch_B_sample_params_arr)
    np.savetxt(
        network_auelp_batch_A_sample_config_params_filename,
        network_auelp_batch_A_sample_config_params_arr)
    np.savetxt(
        network_auelp_batch_B_sample_config_params_filename,
        network_auelp_batch_B_sample_config_params_arr)
    np.savetxt(
        network_apelp_batch_A_sample_config_params_filename,
        network_apelp_batch_A_sample_config_params_arr)
    np.savetxt(
        network_apelp_batch_B_sample_config_params_filename,
        network_apelp_batch_B_sample_config_params_arr)

if __name__ == "__main__":
    main()