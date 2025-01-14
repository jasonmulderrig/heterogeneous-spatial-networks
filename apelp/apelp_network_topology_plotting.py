# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import numpy as np
from file_io.file_io import filepath_str
from networks.aelp_networks_plotting import aelp_network_topology_plotter

def main():
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

    filepath = filepath_str(network)
    filename_prefix = filepath + f"{date}{batch}"

    params_filename = filename_prefix + "-params" + ".dat"
    
    params_arr = np.loadtxt(params_filename, ndmin=1)
    
    # Artificial polydisperse end-linked polymer network plotting
    # parameters
    plt_pad_prefactor = 0.30
    core_tick_inc_prefactor = 0.2

    # Network parameters
    dim_2 = 2
    dim_3 = 3
    b = 1.0
    xi = 0.98
    rho_nu = 0.85
    k = 4
    n = 100
    nu = 50
    nu_max = 500
    config = 0

    # Identification of the sample value for the desired network
    dim_2_sample = int(
        np.where(np.all(params_arr == (dim_2, b, xi, rho_nu, k, n, nu, nu_max), axis=1))[0][0])
    dim_3_sample = int(
        np.where(np.all(params_arr == (dim_3, b, xi, rho_nu, k, n, nu, nu_max), axis=1))[0][0])

    # Artificial polydisperse end-linked polymer network plotting
    aelp_network_topology_plotter(
        plt_pad_prefactor, core_tick_inc_prefactor, network, date, batch,
        dim_2_sample, config)
    aelp_network_topology_plotter(
        plt_pad_prefactor, core_tick_inc_prefactor, network, date, batch,
        dim_3_sample, config)


if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial polydisperse end-linked polymer network topology plotting took {execution_time} seconds to run")