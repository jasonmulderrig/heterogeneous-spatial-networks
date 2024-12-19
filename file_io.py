import os
import pathlib

def filepath_str(network: str) -> str:
    """Filepath generator for heterogeneous spatial networks.

    This function ensures that a baseline filepath for files involved
    with heterogeneous spatial networks exists as a directory, and then
    returns the baseline filepath. The filepath must match the directory
    structure of the local computer. For Windows machines, the backslash
    must be represented as a double backslash. For Linux/Mac machines,
    the forwardslash can be directly represented as a forwardslash.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "apelp", "swidt", "delaunay", or "voronoi" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial polydisperse end-linked polymer networks ("apelp"), spider web-inspired Delaunay-triangulated networks ("swidt"), Delaunay-triangulated networks ("delaunay"), or Voronoi-tessellated networks ("voronoi")).
    
    Returns:
        str: The baseline filepath.
    
    """
    # For MacOS
    # filepath = f"/Users/jasonmulderrig/research/projects/heterogeneous-spatial-networks/{network}/"
    # For Windows OS
    filepath = f"C:\\Users\\mulderjp\\projects\\heterogeneous-spatial-networks\\{network}\\"
    # For Linux
    # filepath = f"/p/home/jpm2225/projects/heterogeneous-spatial-networks/{network}/"
    if os.path.isdir(filepath) == False:
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    return filepath

def filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int) -> str:
    """Baseline filename generator for heterogeneous spatial networks.

    This function returns the baseline filename for files involved with
    heterogeneous spatial networks. The baseline filename is explicitly
    prefixed with the filepath to the directory that the files ought to
    be saved to (and loaded from for future use). This filepath is set
    by the user, and must match the directory structure of the local
    computer. The baseline filename is then appended to the filepath.
    It is incumbent on the user to save a data file that records the
    network parameter values that correspond to each network sample in
    the batch (i.e., a "lookup table").

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "apelp", "swidt", "delaunay", or "voronoi" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial polydisperse end-linked polymer networks ("apelp"), spider web-inspired Delaunay-triangulated networks ("swidt"), Delaunay-triangulated networks ("delaunay"), or Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
    
    Returns:
        str: The baseline filename.
    
    """
    return filepath_str(network) + f"{date}{batch}{sample:d}"

def L_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int) -> str:
    """Filename for simulation box size.

    This function returns the filename for the simulation box size.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "apelp", "swidt", "delaunay", or "voronoi" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial polydisperse end-linked polymer networks ("apelp"), spider web-inspired Delaunay-triangulated networks ("swidt"), Delaunay-triangulated networks ("delaunay"), or Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
    
    Returns:
        str: The simulation box size filename.
    
    """
    return filename_str(network, date, batch, sample) + "-L" + ".dat"

def config_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> str:
    """Configuration filename prefix.

    This function returns the configuration filename prefix.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "apelp", "swidt", "delaunay", or "voronoi" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial polydisperse end-linked polymer networks ("apelp"), spider web-inspired Delaunay-triangulated networks ("swidt"), Delaunay-triangulated networks ("delaunay"), or Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    Returns:
        str: The configuration filename prefix.
    
    """
    return filename_str(network, date, batch, sample) + f"C{config:d}"

def config_pruning_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        pruning: int) -> str:
    """Configuration and pruning filename prefix.

    This function returns the configuration and pruning filename prefix.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "apelp", "swidt", "delaunay", or "voronoi" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial polydisperse end-linked polymer networks ("apelp"), spider web-inspired Delaunay-triangulated networks ("swidt"), Delaunay-triangulated networks ("delaunay"), or Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        pruning (int): Pruning number.
    
    Returns:
        str: The configuration and pruning filename prefix.
    
    """
    return (
        filename_str(network, date, batch, sample)
        + f"C{config:d}" + f"P{pruning:d}"
    )