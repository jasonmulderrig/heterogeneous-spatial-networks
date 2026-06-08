# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
import numpy as np
from file_io.file_io import L_filename_str
from topological_descriptors.general_topological_descriptors import l_arr_func
from networks.aelp_networks import aelp_filename_str
from networks.apelp_networks_config import (
    apelpConfig,
    params_arr_func
)

# Hydra ConfigStore initialization
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=apelpConfig)

@hydra.main(version_base=None, config_path=".", config_name="apelp_networks_config")
def main(cfg: apelpConfig) -> None:
    _, sample_num = params_arr_func(cfg)
    
    b = cfg.topology.b[0]
    
    mean_perc_errant_edges = 0.
    for sample in range(sample_num):
        for config in range(cfg.topology.config):
            # Generate filenames
            L_filename = L_filename_str(cfg.label.network, cfg.label.date, cfg.label.batch, sample)
            aelp_filename = aelp_filename_str(cfg.label.network, cfg.label.date, cfg.label.batch, sample, config)
            coords_filename = aelp_filename + ".coords"
            conn_edges_filename = aelp_filename + "-conn_edges" + ".dat"
            conn_edges_type_filename = (
                aelp_filename + "-conn_edges_type" + ".dat"
            )
            en_conn_edges_filename = aelp_filename + "-en_conn_edges" + ".dat"

            # Load simulation box size and node coordinates
            L = np.loadtxt(L_filename)
            coords = np.loadtxt(coords_filename)

            # Load fundamental graph constituents
            conn_edges = np.loadtxt(conn_edges_filename, dtype=int)
            conn_edges_type = np.loadtxt(conn_edges_type_filename, dtype=int)
            conn_core_edges = conn_edges[np.where(conn_edges_type==1)[0]]
            conn_pb_edges = conn_edges[np.where(conn_edges_type==0)[0]]
            m = np.shape(conn_edges)[0]
            en_conn_edges = np.loadtxt(en_conn_edges_filename, dtype=int)
            en_conn_core_edges = en_conn_edges[np.where(conn_edges_type==1)[0]]
            en_conn_pb_edges = en_conn_edges[np.where(conn_edges_type==0)[0]]
            nu_conn_core_edges = en_conn_core_edges - 1
            nu_conn_pb_edges = en_conn_pb_edges - 1
            nu_edges = np.concatenate(
                (nu_conn_core_edges, nu_conn_pb_edges), dtype=int)

            # Calculate end-to-end chain length (Euclidean edge length)
            l_core_chn, l_pb_chn = l_arr_func(
                conn_core_edges, conn_pb_edges, coords, L)
            l_edges = np.concatenate((l_core_chn, l_pb_chn))
            
            errant_edges = 0
            for edge in range(m):
                if (b*nu_edges[edge]) < l_edges[edge]:
                    errant_edges += 1
                    print("apelp sample {} config {} edge {} nu = {} < l = {}".format(sample, config, edge, nu_edges[edge], l_edges[edge]))
            perc_errant_edges = errant_edges / m * 100
            if perc_errant_edges > 0:
                print("apelp sample {} config {} has {} errant edges, {} percent of edges in the network".format(sample, config, errant_edges, perc_errant_edges))
            mean_perc_errant_edges += perc_errant_edges
    mean_perc_errant_edges /= cfg.topology.config

    print("{} percent of apelp edges are errant (on average)".format(mean_perc_errant_edges))

if __name__ == "__main__":
    main()