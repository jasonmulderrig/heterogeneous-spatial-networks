##############################
heterogeneous spatial networks
##############################

A repository of research codes that computationally synthesize and analyze heterogeneous spatial networks. The networks of focus include: Voronoi-tessellated networks, Delaunay-triangulated networks, spider web-inspired Delaunay-triangulated networks, artificial uniform end-linked polymer networks, and artificial polydisperse end-linked polymer networks.

*****
Setup
*****

Once the contents of the repository have been cloned or downloaded, the Python virtual environment associated with the project needs to be installed. The installation of this Python virtual environment and some essential packages is handled by the ``virtual-environment-install-master.sh`` Bash script. Before running the script, make sure to change the ``VENV_PATH`` parameter to comply with your personal filetree structure. Alternatively, a Conda environment can be installed with the required packages. All required packages are listed in the ``requirements.txt`` file.

*****
Usage
*****

The foundational functions are all located in the ``heterogeneous_spatial_networks_funcs.py`` code. Before running any of the code, it is required that the user verify the baseline filepath in the first function of the ``heterogeneous_spatial_networks_funcs.py`` code called ``filepath_str()``. Note that filepath string conventions are operating system-sensitive.

Important "worker" functions are located in the ``heterogeneous_spatial_networks_funcs.py`` code. In general, these functions assist in parallelization and plotting.

Finally, there are an assortmant of Jupyter Python notebooks that execute the network synthesis and analysis protocols in parallel via the ``multiprocessing`` Python module.

The following contains some timing benchmarks for the following network synthesis Jupyter Python notebooks on my Dell Inspiron computer with ``cpu_num = 8`` for the ``20241016`` network parameters:
    * ``voronoi-network-topology-synthesis.ipynb`` ~ 4 minutes
    * ``delaunay-network-topology-synthesis.ipynb`` ~ 0.5 minute
    * ``swidt-network-topology-synthesis.ipynb`` ~ 4 minutes
    * ``aelp-network-topology-synthesis.ipynb`` ~ 3 hours