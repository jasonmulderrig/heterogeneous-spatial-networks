import numpy as np
import matplotlib.pyplot as plt
from file_io import (
    filepath_str,
    L_filename_str
)
from delaunay_networks import delaunay_filename_str

def main():
    # Initialization of identification information for this particular
    # batch of Voronoi-tessellated networks
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

    identifier_arr = np.loadtxt(identifier_filename, dtype=str, usecols=0, ndmin=1)
    dim_arr = np.loadtxt(dim_filename, dtype=int, ndmin=1)
    b_arr = np.loadtxt(b_filename, ndmin=1)
    n_arr = np.loadtxt(n_filename, dtype=int, ndmin=1)
    eta_n_arr = np.loadtxt(eta_n_filename, ndmin=1)
    config_arr = np.loadtxt(config_filename, dtype=int, ndmin=1)
    sample_config_params_arr = np.loadtxt(sample_config_params_filename, ndmin=1)

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    n_num = np.shape(n_arr)[0]
    eta_n_num = np.shape(eta_n_arr)[0]
    config_num = np.shape(config_arr)[0]

    sample_num = dim_num * b_num * n_num * eta_n_num
    sample_config_num = sample_num * config_num

    n = n_arr[0]
    dim_2 = dim_arr[0]
    dim_3 = dim_arr[1]

    dim_2_nrmlzd_coords = np.empty((config_num, n, dim_2))
    dim_3_nrmlzd_coords = np.empty((config_num, n, dim_3))

    for sample in range(sample_num):
        L = np.loadtxt(L_filename_str(network, date, batch, sample))
        for config in range(config_num):
            coords_filename = (
                delaunay_filename_str(network, date, batch, sample, config)
                + ".coords"
            )
            nrmlzd_coords = np.loadtxt(coords_filename) / L
            if sample == 0:
                dim_2_nrmlzd_coords[config, :, :] = nrmlzd_coords
            elif sample == 1:
                dim_3_nrmlzd_coords[config, :, :] = nrmlzd_coords

    dim_2_nrmlzd_coords_min = np.empty((n, dim_2))
    dim_2_nrmlzd_coords_mean_minus_std = np.empty((n, dim_2))
    dim_2_nrmlzd_coords_mean = np.empty((n, dim_2))
    dim_2_nrmlzd_coords_mean_plus_std = np.empty((n, dim_2))
    dim_2_nrmlzd_coords_max = np.empty((n, dim_2))

    dim_3_nrmlzd_coords_min = np.empty((n, dim_3))
    dim_3_nrmlzd_coords_mean_minus_std = np.empty((n, dim_3))
    dim_3_nrmlzd_coords_mean = np.empty((n, dim_3))
    dim_3_nrmlzd_coords_mean_plus_std = np.empty((n, dim_3))
    dim_3_nrmlzd_coords_max = np.empty((n, dim_3))

    for sample in range(sample_num):
        for node_label in range(n):
            if sample == 0:
                for coord in range(dim_2):
                    mean_val = np.mean(dim_2_nrmlzd_coords[:, node_label, coord])
                    std_val = np.std(dim_2_nrmlzd_coords[:, node_label, coord])
                    dim_2_nrmlzd_coords_min[node_label, coord] = np.min(
                        dim_2_nrmlzd_coords[:, node_label, coord])
                    dim_2_nrmlzd_coords_mean_minus_std[node_label, coord] = (
                        mean_val - std_val
                    )
                    dim_2_nrmlzd_coords_mean[node_label, coord] = mean_val
                    dim_2_nrmlzd_coords_mean_plus_std[node_label, coord] = (
                        mean_val + std_val
                    )
                    dim_2_nrmlzd_coords_max[node_label, coord] = np.max(
                        dim_2_nrmlzd_coords[:, node_label, coord])
            elif sample == 1:
                for coord in range(dim_3):
                    mean_val = np.mean(dim_3_nrmlzd_coords[:, node_label, coord])
                    std_val = np.std(dim_3_nrmlzd_coords[:, node_label, coord])
                    dim_3_nrmlzd_coords_min[node_label, coord] = np.min(
                        dim_3_nrmlzd_coords[:, node_label, coord])
                    dim_3_nrmlzd_coords_mean_minus_std[node_label, coord] = (
                        mean_val - std_val
                    )
                    dim_3_nrmlzd_coords_mean[node_label, coord] = mean_val
                    dim_3_nrmlzd_coords_mean_plus_std[node_label, coord] = (
                        mean_val + std_val
                    )
                    dim_3_nrmlzd_coords_max[node_label, coord] = np.max(
                        dim_3_nrmlzd_coords[:, node_label, coord])
    
    node_labels = np.arange(n, dtype=int)

    for sample in range(sample_num):
        if sample == 0:
            for coord in range(dim_2):
                plt.fill_between(
                    node_labels, dim_2_nrmlzd_coords_min[:, coord],
                    dim_2_nrmlzd_coords_max[:, coord], color="skyblue", alpha=0.25)
                plt.fill_between(
                    node_labels, dim_2_nrmlzd_coords_mean_minus_std[:, coord],
                    dim_2_nrmlzd_coords_mean_plus_std[:, coord], color="steelblue",
                    alpha=0.25)
                plt.plot(
                    node_labels, dim_2_nrmlzd_coords_mean[:, coord],
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
                plt.title("{}D Delaunay network".format(dim_2), fontsize=16)
                plt.grid(True, alpha=0.25, zorder=0)
                plt.tight_layout()
                filename_str += "_{}d_delaunay_network_finalized.png".format(dim_2)
                plt.savefig(filename_str)
                plt.close()
        elif sample == 1:
            for coord in range(dim_3):
                plt.fill_between(
                    node_labels, dim_3_nrmlzd_coords_min[:, coord],
                    dim_3_nrmlzd_coords_max[:, coord], color="skyblue", alpha=0.25)
                plt.fill_between(
                    node_labels, dim_3_nrmlzd_coords_mean_minus_std[:, coord],
                    dim_3_nrmlzd_coords_mean_plus_std[:, coord], color="steelblue",
                    alpha=0.25)
                plt.plot(
                    node_labels, dim_3_nrmlzd_coords_mean[:, coord],
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
                plt.title("{}D Delaunay network".format(dim_3), fontsize=16)
                plt.grid(True, alpha=0.25, zorder=0)
                plt.tight_layout()
                filename_str += "_{}d_delaunay_network_finalized.png".format(dim_3)
                plt.savefig(filename_str)
                plt.close()

if __name__ == "__main__":
    main()