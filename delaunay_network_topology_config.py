import numpy as np
from file_io import filepath_str

def main():
    # Initialization of identification information for this particular
    # batch of Delaunay-triangulated networks
    network = "delaunay"
    date = "20241210"
    batch = "A"
    scheme = "prhd"

    network_str = f"network_{network}"
    date_str = f"date_{date}"
    batch_str = f"batch_{batch}"
    scheme_str = f"scheme_{scheme}"

    dim_str = "dim"
    b_str = "b"
    n_str = "n"
    eta_n_str = "eta_n"
    config_str = "config"

    filepath = filepath_str(network)
    filename_prefix = filepath + f"{date}{batch}"

    identifier_filename = filename_prefix + "-identifier" + ".txt"
    dim_filename = filename_prefix + f"-{dim_str}" + ".dat"
    b_filename = filename_prefix + f"-{b_str}" + ".dat"
    n_filename = filename_prefix + f"-{n_str}" + ".dat"
    eta_n_filename = filename_prefix + f"-{eta_n_str}" + ".dat"
    config_filename = filename_prefix + f"-{config_str}" + ".dat"
    params_filename = filename_prefix + "-params" + ".dat"
    sample_params_filename = filename_prefix + "-sample_params" + ".dat"
    sample_config_params_filename = (
        filename_prefix + "-sample_config_params" + ".dat"
    )

    identifier_arr = np.asarray(
        [
            network_str,
            date_str,
            batch_str,
            scheme_str,
            dim_str,
            b_str,
            n_str,
            eta_n_str
        ]
    )

    # Initialization of fundamental parameters for Delaunay-triangulated
    # networks
    dim_arr = np.asarray([2, 3], dtype=int)
    b_arr = np.asarray([1.0])
    n_arr = np.asarray([100], dtype=int)
    eta_n_arr = np.asarray([0.01])
    config_arr = np.arange(200, dtype=int) # 0, 1, 2, ..., 198, 199

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    n_num = np.shape(n_arr)[0]
    eta_n_num = np.shape(eta_n_arr)[0]
    config_num = np.shape(config_arr)[0]

    sample_num = dim_num * b_num * n_num * eta_n_num
    sample_config_num = sample_num * config_num

    # Populate the network parameters array
    params_arr = np.empty((sample_num, 4))
    sample = 0
    for dim in np.nditer(dim_arr):
        for b in np.nditer(b_arr):
            for n in np.nditer(n_arr):
                for eta_n in np.nditer(eta_n_arr):
                    params_arr[sample, :] = (
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
    sample_params_arr = np.empty((sample_num, 5))
    sample = 0
    for dim in np.nditer(dim_arr):
        for b in np.nditer(b_arr):
            for n in np.nditer(n_arr):
                for eta_n in np.nditer(eta_n_arr):
                    sample_params_arr[sample, :] = (
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
    sample_config_params_arr = np.empty((sample_config_num, 6))
    sample = 0
    indx = 0
    for dim in np.nditer(dim_arr):
        for b in np.nditer(b_arr):
            for n in np.nditer(n_arr):
                for eta_n in np.nditer(eta_n_arr):
                    for config in np.nditer(config_arr):
                        sample_config_params_arr[indx, :] = (
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
    np.savetxt(identifier_filename, identifier_arr, fmt="%s")
    np.savetxt(dim_filename, dim_arr, fmt="%d")
    np.savetxt(b_filename, b_arr)
    np.savetxt(n_filename, n_arr, fmt="%d")
    np.savetxt(eta_n_filename, eta_n_arr)
    np.savetxt(config_filename, config_arr, fmt="%d")
    np.savetxt(params_filename, params_arr)
    np.savetxt(sample_params_filename, sample_params_arr)
    np.savetxt(sample_config_params_filename, sample_config_params_arr)

if __name__ == "__main__":
    main()