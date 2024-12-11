import numpy as np
from file_io import filepath_str

def main():
    # Initialization of identification information for these particular
    # batches of Voronoi-tessellated networks
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

    batch_A_identifier_arr = np.asarray(
        [
            network_str,
            date_str,
            batch_A_str,
            scheme_str,
            dim_str,
            b_str,
            n_str,
            eta_n_str
        ]
    )
    batch_B_identifier_arr = np.asarray(
        [
            network_str,
            date_str,
            batch_B_str,
            scheme_str,
            dim_str,
            b_str,
            n_str,
            eta_n_str
        ]
    )

    # Initialization of fundamental parameters for Voronoi-tessellated
    # networks
    dim_2_arr = np.asarray([2], dtype=int)
    dim_3_arr = np.asarray([3], dtype=int)
    b_arr = np.asarray([1.0])
    dim_2_n_arr = np.asarray([50], dtype=int)
    dim_3_n_arr = np.asarray([15], dtype=int)
    eta_n_arr = np.asarray([0.01])
    config_arr = np.arange(200, dtype=int) # 0, 1, 2, ..., 198, 199

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

    # Populate the network parameters array
    batch_A_params_arr = np.empty((batch_A_sample_num, 4))
    sample = 0
    for dim in np.nditer(dim_2_arr):
        for b in np.nditer(b_arr):
            for n in np.nditer(dim_2_n_arr):
                for eta_n in np.nditer(eta_n_arr):
                    batch_A_params_arr[sample, :] = (
                        np.asarray(
                            [
                                dim,
                                b,
                                n,
                                eta_n
                            ]
                        )
                    )
                    sample += 1
    batch_B_params_arr = np.empty((batch_B_sample_num, 4))
    sample = 0
    for dim in np.nditer(dim_3_arr):
        for b in np.nditer(b_arr):
            for n in np.nditer(dim_3_n_arr):
                for eta_n in np.nditer(eta_n_arr):
                    batch_B_params_arr[sample, :] = (
                        np.asarray(
                            [
                                dim,
                                b,
                                n,
                                eta_n
                            ]
                        )
                    )
                    sample += 1

    # Populate the network sample parameters array
    batch_A_sample_params_arr = np.empty((batch_A_sample_num, 5))
    sample = 0
    for dim in np.nditer(dim_2_arr):
        for b in np.nditer(b_arr):
            for n in np.nditer(dim_2_n_arr):
                for eta_n in np.nditer(eta_n_arr):
                    batch_A_sample_params_arr[sample, :] = (
                        np.asarray(
                            [
                                sample,
                                dim,
                                b,
                                n,
                                eta_n
                            ]
                        )
                    )
                    sample += 1
    batch_B_sample_params_arr = np.empty((batch_B_sample_num, 5))
    sample = 0
    for dim in np.nditer(dim_3_arr):
        for b in np.nditer(b_arr):
            for n in np.nditer(dim_3_n_arr):
                for eta_n in np.nditer(eta_n_arr):
                    batch_B_sample_params_arr[sample, :] = (
                        np.asarray(
                            [
                                sample,
                                dim,
                                b,
                                n,
                                eta_n
                            ]
                        )
                    )
                    sample += 1

    # Populate the network sample configuration parameters array
    batch_A_sample_config_params_arr = np.empty((batch_A_sample_config_num, 6))
    sample = 0
    indx = 0
    for dim in np.nditer(dim_2_arr):
        for b in np.nditer(b_arr):
            for n in np.nditer(dim_2_n_arr):
                for eta_n in np.nditer(eta_n_arr):
                    for config in np.nditer(config_arr):
                        batch_A_sample_config_params_arr[indx, :] = (
                            np.asarray(
                                [
                                    sample,
                                    dim,
                                    b,
                                    n,
                                    eta_n,
                                    config
                                ]
                            )
                        )
                        indx += 1
                    sample += 1
    batch_B_sample_config_params_arr = np.empty((batch_B_sample_config_num, 6))
    sample = 0
    indx = 0
    for dim in np.nditer(dim_3_arr):
        for b in np.nditer(b_arr):
            for n in np.nditer(dim_3_n_arr):
                for eta_n in np.nditer(eta_n_arr):
                    for config in np.nditer(config_arr):
                        batch_B_sample_config_params_arr[indx, :] = (
                            np.asarray(
                                [
                                    sample,
                                    dim,
                                    b,
                                    n,
                                    eta_n,
                                    config
                                ]
                            )
                        )
                        indx += 1
                    sample += 1

    # Save identification information and fundamental network parameters
    np.savetxt(batch_A_identifier_filename, batch_A_identifier_arr, fmt="%s")
    np.savetxt(batch_B_identifier_filename, batch_B_identifier_arr, fmt="%s")
    np.savetxt(batch_A_dim_filename, dim_2_arr, fmt="%d")
    np.savetxt(batch_B_dim_filename, dim_3_arr, fmt="%d")
    np.savetxt(batch_A_b_filename, b_arr)
    np.savetxt(batch_B_b_filename, b_arr)
    np.savetxt(batch_A_n_filename, dim_2_n_arr, fmt="%d")
    np.savetxt(batch_B_n_filename, dim_3_n_arr, fmt="%d")
    np.savetxt(batch_A_eta_n_filename, eta_n_arr)
    np.savetxt(batch_B_eta_n_filename, eta_n_arr)
    np.savetxt(batch_A_config_filename, config_arr, fmt="%d")
    np.savetxt(batch_B_config_filename, config_arr, fmt="%d")
    np.savetxt(batch_A_params_filename, batch_A_params_arr)
    np.savetxt(batch_B_params_filename, batch_B_params_arr)
    np.savetxt(batch_A_sample_params_filename, batch_A_sample_params_arr)
    np.savetxt(batch_B_sample_params_filename, batch_B_sample_params_arr)
    np.savetxt(
        batch_A_sample_config_params_filename, batch_A_sample_config_params_arr)
    np.savetxt(
        batch_B_sample_config_params_filename, batch_B_sample_config_params_arr)

if __name__ == "__main__":
    main()