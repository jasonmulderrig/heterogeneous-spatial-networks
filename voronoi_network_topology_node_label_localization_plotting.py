import numpy as np
import matplotlib.pyplot as plt
from file_io import (
    filepath_str,
    L_filename_str
)
from voronoi_networks import voronoi_filename_str

def main():
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

    dim_2 = dim_2_arr[0]
    dim_3 = dim_3_arr[0]

    # Calculate number of nodes and the dimensionality in each network
    # configuration
    batch_A_coords_filename = (
        voronoi_filename_str(network, date, batch_A, sample=0, config=0)
        + ".coords"
    )
    batch_A_n = np.shape(np.loadtxt(batch_A_coords_filename))[0]
    batch_B_coords_filename = (
        voronoi_filename_str(network, date, batch_B, sample=0, config=0)
        + ".coords"
    )
    batch_B_n = np.shape(np.loadtxt(batch_B_coords_filename))[0]

    batch_A_nrmlzd_coords = np.empty(
        (batch_A_sample_num, config_num, batch_A_n, dim_2))
    batch_B_nrmlzd_coords = np.empty(
        (batch_B_sample_num, config_num, batch_B_n, dim_3))

    for sample in range(batch_A_sample_num):
        L = np.loadtxt(L_filename_str(network, date, batch_A, sample))
        for config in range(config_num):
            coords_filename = (
                voronoi_filename_str(network, date, batch_A, sample, config)
                + ".coords"
            )
            nrmlzd_coords = np.loadtxt(coords_filename) / L
            batch_A_nrmlzd_coords[sample, config, :, :] = nrmlzd_coords
    for sample in range(batch_B_sample_num):
        L = np.loadtxt(L_filename_str(network, date, batch_B, sample))
        for config in range(config_num):
            coords_filename = (
                voronoi_filename_str(network, date, batch_B, sample, config)
                + ".coords"
            )
            nrmlzd_coords = np.loadtxt(coords_filename) / L
            batch_B_nrmlzd_coords[sample, config, :, :] = nrmlzd_coords

    batch_A_nrmlzd_coords_min = np.empty((batch_A_sample_num, batch_A_n, dim_2))
    batch_A_nrmlzd_coords_mean_minus_std = np.empty(
        (batch_A_sample_num, batch_A_n, dim_2))
    batch_A_nrmlzd_coords_mean = np.empty((batch_A_sample_num, batch_A_n, dim_2))
    batch_A_nrmlzd_coords_mean_plus_std = np.empty(
        (batch_A_sample_num, batch_A_n, dim_2))
    batch_A_nrmlzd_coords_max = np.empty((batch_A_sample_num, batch_A_n, dim_2))

    batch_B_nrmlzd_coords_min = np.empty((batch_B_sample_num, batch_B_n, dim_3))
    batch_B_nrmlzd_coords_mean_minus_std = np.empty(
        (batch_B_sample_num, batch_B_n, dim_3))
    batch_B_nrmlzd_coords_mean = np.empty((batch_B_sample_num, batch_B_n, dim_3))
    batch_B_nrmlzd_coords_mean_plus_std = np.empty(
        (batch_B_sample_num, batch_B_n, dim_3))
    batch_B_nrmlzd_coords_max = np.empty((batch_B_sample_num, batch_B_n, dim_3))

    for sample in range(batch_A_sample_num):
        for node_label in range(batch_A_n):
            for coord in range(dim_2):
                mean_val = np.mean(
                    batch_A_nrmlzd_coords[sample, :, node_label, coord])
                std_val = np.std(
                    batch_A_nrmlzd_coords[sample, :, node_label, coord])
                batch_A_nrmlzd_coords_min[sample, node_label, coord] = np.min(
                    batch_A_nrmlzd_coords[sample, :, node_label, coord])
                batch_A_nrmlzd_coords_mean_minus_std[sample, node_label, coord] = (
                    mean_val - std_val
                )
                batch_A_nrmlzd_coords_mean[sample, node_label, coord] = mean_val
                batch_A_nrmlzd_coords_mean_plus_std[sample, node_label, coord] = (
                    mean_val + std_val
                )
                batch_A_nrmlzd_coords_max[sample, node_label, coord] = np.max(
                    batch_A_nrmlzd_coords[sample, :, node_label, coord])
    for sample in range(batch_B_sample_num):
        for node_label in range(batch_B_n):
            for coord in range(dim_3):
                mean_val = np.mean(
                    batch_B_nrmlzd_coords[sample, :, node_label, coord])
                std_val = np.std(
                    batch_B_nrmlzd_coords[sample, :, node_label, coord])
                batch_B_nrmlzd_coords_min[sample, node_label, coord] = np.min(
                    batch_B_nrmlzd_coords[sample, :, node_label, coord])
                batch_B_nrmlzd_coords_mean_minus_std[sample, node_label, coord] = (
                    mean_val - std_val
                )
                batch_B_nrmlzd_coords_mean[sample, node_label, coord] = mean_val
                batch_B_nrmlzd_coords_mean_plus_std[sample, node_label, coord] = (
                    mean_val + std_val
                )
                batch_B_nrmlzd_coords_max[sample, node_label, coord] = np.max(
                    batch_B_nrmlzd_coords[sample, :, node_label, coord])
    
    batch_A_node_labels = np.arange(batch_A_n, dtype=int)
    batch_B_node_labels = np.arange(batch_B_n, dtype=int)

    for sample in range(batch_A_sample_num):
        for coord in range(dim_2):
            plt.fill_between(
                batch_A_node_labels,
                batch_A_nrmlzd_coords_min[sample, :, coord],
                batch_A_nrmlzd_coords_max[sample, :, coord],
                color="skyblue", alpha=0.25)
            plt.fill_between(
                batch_A_node_labels,
                batch_A_nrmlzd_coords_mean_minus_std[sample, :, coord],
                batch_A_nrmlzd_coords_mean_plus_std[sample, :, coord],
                color="steelblue", alpha=0.25)
            plt.plot(
                batch_A_node_labels,
                batch_A_nrmlzd_coords_mean[sample, :, coord],
                linestyle="-", color="tab:blue")
            plt.xlabel("Node label", fontsize=16)
            if coord == 0:
                plt.ylabel("x/L", fontsize=16)
                filename_str = "x_nrmlzd_vs_node_label"
            elif coord == 1:
                plt.ylabel("y/L", fontsize=16)
                filename_str = "y_nrmlzd_vs_node_label"
            elif coord == 2:
                plt.ylabel("z/L", fontsize=16)
                filename_str = "z_nrmlzd_vs_node_label"
            plt.title("{}D Voronoi network".format(dim_2), fontsize=16)
            plt.grid(True, alpha=0.25, zorder=0)
            plt.tight_layout()
            filename_str += "_{}d_voronoi_network_finalized.png".format(dim_2)
            plt.savefig(filename_str)
            plt.close()
    for sample in range(batch_B_sample_num):
        for coord in range(dim_3):
            plt.fill_between(
                batch_B_node_labels,
                batch_B_nrmlzd_coords_min[sample, :, coord],
                batch_B_nrmlzd_coords_max[sample, :, coord],
                color="skyblue", alpha=0.25)
            plt.fill_between(
                batch_B_node_labels,
                batch_B_nrmlzd_coords_mean_minus_std[sample, :, coord],
                batch_B_nrmlzd_coords_mean_plus_std[sample, :, coord],
                color="steelblue", alpha=0.25)
            plt.plot(
                batch_B_node_labels,
                batch_B_nrmlzd_coords_mean[sample, :, coord],
                linestyle="-", color="tab:blue")
            plt.xlabel("Node label", fontsize=16)
            if coord == 0:
                plt.ylabel("x/L", fontsize=16)
                filename_str = "x_nrmlzd_vs_node_label"
            elif coord == 1:
                plt.ylabel("y/L", fontsize=16)
                filename_str = "y_nrmlzd_vs_node_label"
            elif coord == 2:
                plt.ylabel("z/L", fontsize=16)
                filename_str = "z_nrmlzd_vs_node_label"
            plt.title("{}D Voronoi network".format(dim_3), fontsize=16)
            plt.grid(True, alpha=0.25, zorder=0)
            plt.tight_layout()
            filename_str += "_{}d_voronoi_network_finalized.png".format(dim_3)
            plt.savefig(filename_str)
            plt.close()

if __name__ == "__main__":
    main()