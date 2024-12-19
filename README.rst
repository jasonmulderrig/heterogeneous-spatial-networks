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

The core functions can then be called upon in Python files or Jupyter notebooks for various tasks. The following Python files synthesize various heterogeneous spatial networks and calculate topological descriptors when run in the order provided:

* ``voronoi_network_topology_config.py`` -> ``voronoi_network_topology_synthesis.py`` -> ``voronoi_network_topology_descriptors.py``
* ``delaunay_network_topology_config.py`` -> ``delaunay_network_topology_synthesis.py`` -> ``delaunay_network_topology_descriptors.py``
* ``swidt_network_topology_config.py`` -> ``swidt_network_topology_synthesis.py`` -> ``swidt_network_topology_descriptors.py``
* ``aelp_network_topology_config.py`` -> ``aelp_network_topology_synthesis.py`` -> ``aelp_network_topology_descriptors.py``

In the near future, additional Python files will be supplied that can plot the spatially-embedded structure of each network and analyze the statistics of topological descriptors.

*****
Usage
*****

**Before running any of the code, it is required that the user verify the baseline filepath in the ``filepath_str()`` function of the ``file_io.py`` Python file. Note that filepath string conventions are operating system-sensitive.**

*************************
Example timing benchmarks
*************************

The following contains some timing benchmarks for the network synthesis and topological descriptor Python files on my Dell Inspiron computer with ``cpu_num = 8`` for the ``20241210`` network parameters:

* ``voronoi_network_topology_config.py`` -> ``voronoi_network_topology_synthesis.py`` -> ``voronoi_network_topology_descriptors.py``: ~ 30 seconds (for ``voronoi_network_topology_synthesis.py``) + ~ 25 minutes (for ``voronoi_network_topology_descriptors.py``)
* ``delaunay_network_topology_config.py`` -> ``delaunay_network_topology_synthesis.py`` -> ``delaunay_network_topology_descriptors.py``: ~ 10 seconds (for ``delaunay_network_topology_synthesis.py``) + ~ 65 minutes (for ``delaunay_network_topology_descriptors.py``)
* ``swidt_network_topology_config.py`` -> ``swidt_network_topology_synthesis.py`` -> ``swidt_network_topology_descriptors.py``: ~ 20 seconds (for ``swidt_network_topology_synthesis.py``) + ~ 20 minutes (for ``swidt_network_topology_descriptors.py``)
* ``aelp_network_topology_config.py`` -> ``aelp_network_topology_synthesis.py`` -> ``aelp_network_topology_descriptors.py``: ~ 115 minutes (for ``aelp_network_topology_synthesis.py``) + ~ 55 minutes (for ``aelp_network_topology_descriptors.py``)

Note that essentially every type of topological descriptor is calculated for each network in the topological descriptor Python files (for the sake of benchmarking).