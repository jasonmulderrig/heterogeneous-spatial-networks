##############################
heterogeneous spatial networks
##############################

A repository of research codes that computationally synthesize and analyze heterogeneous spatial networks. The networks of focus include (in alphabetical order): artificial bimodal end-linked polymer networks, artificial polydisperse end-linked polymer networks, artificial uniform end-linked polymer networks, Delaunay-triangulated networks, spider web-inspired Delaunay-triangulated networks, and Voronoi-tessellated networks.

*****
Setup
*****

Once the contents of the repository have been cloned or downloaded, the Python virtual environment associated with the project needs to be installed. The installation of this Python virtual environment and some essential packages is handled by the ``virtual-environment-install-master.sh`` Bash script. Before running the script, make sure to change the ``VENV_PATH`` parameter to comply with your personal filetree structure. Alternatively, a Conda environment can be installed with the required packages. All required packages are listed in the ``requirements.txt`` file.

*********
Structure
*********

The core functions in this repository are modularly distributed in Python files that reside in the following directories:

* ``file_io``
* ``helpers``
* ``networks``
* ``topological_descriptors``

The core functions can then be called upon in Python files or Jupyter notebooks for various tasks. The following directories contain various Python files that synthesize and analyze the aforementioned types of heterogeneous spatial networks:

* ``abelp``
* ``apelp``
* ``auelp``
* ``delaunay``
* ``swidt``
* ``voronoi``

The following Python files synthesize various heterogeneous spatial networks and calculate topological descriptors when run in the order provided:

* In the ``abelp`` directory: ``abelp_network_topology_config.py`` -> ``abelp_network_topology_synthesis.py`` -> ``abelp_network_topology_descriptors.py``
* In the ``apelp`` directory: ``apelp_network_topology_config.py`` -> ``apelp_network_topology_synthesis.py`` -> ``apelp_network_topology_descriptors.py``
* In the ``auelp`` directory: ``auelp_network_topology_config.py`` -> ``auelp_network_topology_synthesis.py`` -> ``auelp_network_topology_descriptors.py``
* In the ``delaunay`` directory: ``delaunay_network_topology_config.py`` -> ``delaunay_network_topology_synthesis.py`` -> ``delaunay_network_topology_descriptors.py``
* In the ``swidt`` directory: ``swidt_network_topology_config.py`` -> ``swidt_network_topology_synthesis.py`` -> ``swidt_network_topology_descriptors.py``
* In the ``voronoi`` directory: ``voronoi_network_topology_config.py`` -> ``voronoi_network_topology_synthesis.py`` -> ``voronoi_network_topology_descriptors.py``

In addition, several Python files are supplied in the ``abelp``, ``apelp``, and ``auelp`` directories that plot the spatially-embedded structure of a given network and analyze the statistics of various network topological features. Time and bandwidth permitting, such functionality will also be extended to the ``delaunay``, ``swidt``, and ``voronoi`` networks.

*****
Usage
*****

**Before running any of the code, it is required that the user verify the baseline filepath in the ``filepath_str()`` function of the ``file_io.py`` Python file in the ``file_io`` directory. Note that filepath string conventions are operating system-sensitive.**

*************************
Example timing benchmarks
*************************

The following contains some timing benchmarks for the network synthesis and topological descriptor Python files on my Dell Inspiron computer with ``cpu_num = 8`` for the ``20250102`` network parameters:

* ``abelp_network_topology_config.py`` -> ``abelp_network_topology_synthesis.py`` -> ``abelp_network_topology_descriptors.py``: ~ 60 seconds (for ``abelp_network_topology_synthesis.py``) + ~ 25 minutes (for ``abelp_network_topology_descriptors.py``)
* ``apelp_network_topology_config.py`` -> ``apelp_network_topology_synthesis.py`` -> ``apelp_network_topology_descriptors.py``: ~ 105 minutes (for ``apelp_network_topology_synthesis.py``) + ~ 25 minutes (for ``apelp_network_topology_descriptors.py``)
* ``auelp_network_topology_config.py`` -> ``auelp_network_topology_synthesis.py`` -> ``auelp_network_topology_descriptors.py``: ~ 30 seconds (for ``auelp_network_topology_synthesis.py``) + ~ 25 minutes (for ``auelp_network_topology_descriptors.py``)
* ``delaunay_network_topology_config.py`` -> ``delaunay_network_topology_synthesis.py`` -> ``delaunay_network_topology_descriptors.py``: ~ 10 seconds (for ``delaunay_network_topology_synthesis.py``) + ~ 65 minutes (for ``delaunay_network_topology_descriptors.py``)
* ``swidt_network_topology_config.py`` -> ``swidt_network_topology_synthesis.py`` -> ``swidt_network_topology_descriptors.py``: ~ 20 seconds (for ``swidt_network_topology_synthesis.py``) + ~ 20 minutes (for ``swidt_network_topology_descriptors.py``)
* ``voronoi_network_topology_config.py`` -> ``voronoi_network_topology_synthesis.py`` -> ``voronoi_network_topology_descriptors.py``: ~ 30 seconds (for ``voronoi_network_topology_synthesis.py``) + ~ 20 minutes (for ``voronoi_network_topology_descriptors.py``)

Note that essentially every type of topological descriptor is calculated for each network in the topological descriptor Python files (for the sake of benchmarking).