# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import numpy as np
from file_io.file_io import filepath_str

def main():
    # Initialization of identification information for these batches of
    # artificial uniform end-linked polymer networks
    network = "auelp"
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
    rho_en_str = "rho_en"
    k_str = "k"
    n_str = "n"
    en_str = "en"
    config_str = "config"

    filepath = filepath_str(network)
    filename_prefix = filepath + f"{date}{batch}"

    identifier_filename = filename_prefix + "-identifier" + ".txt"
    dim_filename = filename_prefix + f"-{dim_str}" + ".dat"
    b_filename = filename_prefix + f"-{b_str}" + ".dat"
    xi_filename = filename_prefix + f"-{xi_str}" + ".dat"
    rho_en_filename = filename_prefix + f"-{rho_en_str}" + ".dat"
    k_filename = filename_prefix + f"-{k_str}" + ".dat"
    n_filename = filename_prefix + f"-{n_str}" + ".dat"
    en_filename = filename_prefix + f"-{en_str}" + ".dat"
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
            xi_str,
            rho_en_str,
            k_str,
            n_str,
            en_str
        ]
    )

    # Initialization of fundamental parameters for artificial uniform
    # end-linked polymer networks
    dim_arr = np.asarray([2, 3], dtype=int)
    b_arr = np.asarray([1.0])
    xi_arr = np.asarray([0.98])
    rho_en_arr = np.asarray([0.85])
    k_arr = np.asarray([4], dtype=int)
    n_arr = np.asarray([100], dtype=int)
    en_arr = np.asarray([51], dtype=int)
    config_arr = np.arange(200, dtype=int) # 0, 1, 2, ..., 198, 199

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    xi_num = np.shape(xi_arr)[0]
    rho_en_num = np.shape(rho_en_arr)[0]
    k_num = np.shape(k_arr)[0]
    n_num = np.shape(n_arr)[0]
    en_num = np.shape(en_arr)[0]
    config_num = np.shape(config_arr)[0]

    sample_num = dim_num * b_num * xi_num * rho_en_num * k_num * n_num * en_num
    sample_config_num = sample_num * config_num

    # Populate the network parameters array
    params_arr = np.empty((sample_num, 7))
    sample = 0
    for dim in dim_arr:
        for b in b_arr:
            for xi in xi_arr:
                for rho_en in rho_en_arr:
                    for k in k_arr:
                        for n in n_arr:
                            for en in en_arr:
                                params_arr[sample, :] = (
                                    np.asarray(
                                        [
                                            dim,
                                            b,
                                            xi,
                                            rho_en,
                                            k,
                                            n,
                                            en
                                        ]
                                    )
                                )
                                sample += 1

    # Populate the network sample parameters array
    sample_params_arr = np.empty((sample_num, 8))
    sample = 0
    for dim in dim_arr:
        for b in b_arr:
            for xi in xi_arr:
                for rho_en in rho_en_arr:
                    for k in k_arr:
                        for n in n_arr:
                            for en in en_arr:
                                sample_params_arr[sample, :] = (
                                    np.asarray(
                                        [
                                            sample,
                                            dim,
                                            b,
                                            xi,
                                            rho_en,
                                            k,
                                            n,
                                            en
                                        ]
                                    )
                                )
                                sample += 1

    # Populate the network sample configuration parameters array
    sample_config_params_arr = np.empty((sample_config_num, 9))
    sample = 0
    indx = 0
    for dim in dim_arr:
        for b in b_arr:
            for xi in xi_arr:
                for rho_en in rho_en_arr:
                    for k in k_arr:
                        for n in n_arr:
                            for en in en_arr:
                                for config in config_arr:
                                    sample_config_params_arr[indx, :] = (
                                        np.asarray(
                                            [
                                                sample,
                                                dim,
                                                b,
                                                xi,
                                                rho_en,
                                                k,
                                                n,
                                                en,
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
    np.savetxt(xi_filename, xi_arr)
    np.savetxt(rho_en_filename, rho_en_arr)
    np.savetxt(k_filename, k_arr, fmt="%d")
    np.savetxt(n_filename, n_arr, fmt="%d")
    np.savetxt(en_filename, en_arr, fmt="%d")
    np.savetxt(config_filename, config_arr, fmt="%d")
    np.savetxt(params_filename, params_arr)
    np.savetxt(sample_params_filename, sample_params_arr)
    np.savetxt(sample_config_params_filename, sample_config_params_arr)

if __name__ == "__main__":
    main()