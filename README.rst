##############################
heterogeneous spatial networks
##############################

A repository of research codes that computationally synthesize and analyze heterogeneous spatial networks. The networks of focus include: Voronoi-tessellated networks, Delaunay-triangulated networks, spider web-inspired Delaunay-triangulated networks, artificial uniform end-linked polymer networks, and artificial polydisperse end-linked polymer networks.

*****
Setup
*****

Once the contents of the repository have been cloned or downloaded, the Python virtual environment associated with the project needs to be installed. The installation of this Python virtual environment and some essential packages is handled by the ``virtual-environment-install-master.sh`` Bash script. Before running the script, make sure to change the ``VENV_PATH`` parameter to comply with your personal filetree structure. Alternatively, a Conda environment can be installed with the required packages. All required packages are listed in the ``requirements.txt`` file.

*********
Structure
*********

The core functions in this repository are modularly distributed in the following Python files:

* ``aelp_networks.py``
* ``delaunay_networks.py``
* ``file_io.py``
* ``general_topological_descriptors.py``
* ``graph_utils.py``
* ``multiprocessing_utils.py``
* ``network_topological_descriptors.py``
* ``network_topology_initialization_utils.py``
* ``network_utils.py``
* ``nodal_degree_topological_descriptors.py``
* ``node_placement.py``
* ``plotting_utils.py``
* ``polymer_network_chain_statistics.py``
* ``shortest_path_topological_descriptors.py``
* ``simulation_box_utils.py``
* ``swidt_networks.py``
* ``voronoi_networks.py``

The core functions can then be called upon in Python files or Jupyter notebooks for various tasks. The following Jupyter notebooks synthesize various heterogeneous spatial networks:

* ``voronoi-network-topology-synthesis.ipynb``
* ``delaunay-network-topology-synthesis.ipynb``
* ``swidt-network-topology-synthesis.ipynb``
* ``aelp-network-topology-synthesis.ipynb``

In the near future, additional Python files or Jupyter notebooks will be supplied that can plot the spatially-embedded structure of each network, generate files that store topological descriptor values, and analyze topological descriptor statistics.

*****
Usage
*****

**Before running any of the code, it is required that the user verify the baseline filepath in the ``filepath_str()`` function of the ``file_io.py`` Python file. Note that filepath string conventions are operating system-sensitive.**

*************************
Example timing benchmarks
*************************

The following contains some timing benchmarks for the network synthesis Jupyter notebooks on my Dell Inspiron computer with ``cpu_num = 8`` for the ``20241121`` network parameters:

* ``voronoi-network-topology-synthesis.ipynb`` ~ a few minutes
* ``delaunay-network-topology-synthesis.ipynb`` ~ a few minutes
* ``swidt-network-topology-synthesis.ipynb`` ~ a few minutes
* ``aelp-network-topology-synthesis.ipynb`` ~ 1.75 hours