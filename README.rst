##############################
heterogeneous spatial networks
##############################

A repository of research codes that computationally synthesize and analyze heterogeneous spatial networks. The networks of focus include spider web-inspired networks, artificial uniform end-linked polymer networks, and artificial polydisperse end-linked polymer networks.

*****
Setup
*****

Once the contents of the repository have been cloned or downloaded, the Python virtual environment associated with the project needs to be installed. The installation of this Python virtual environment and some essential packages is handled by the ``virtual-environment-install-master.sh`` Bash script. Before running the script, make sure to change the ``VENV_PATH`` parameter to comply with your personal filetree structure.

*****
Usage
*****

The foundational functions are all located in the ``heterogeneous_spatial_networks_funcs.py`` code. Important "worker" functions are located in the ``heterogeneous_spatial_networks_funcs.py`` code. Finally, there are an assortmant of Jupyter Python notebooks that execute the network synthesis and analysis protocols in parallel via the ``multiprocessing`` Python module.