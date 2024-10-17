import sys
import numpy as np
from numpy.typing import ArrayLike
import networkx as nx

def filepath_str(network: str) -> str:
    """Filepath generator for heterogeneous spatial networks.

    This function ensures that a baseline filepath for files involved
    with heterogeneous spatial networks exists as a directory, and then
    returns the baseline filepath. The filepath must match the directory
    structure of the local computer. For Windows machines, the backslash
    must be represented as a double backslash. For Linux/Mac machines,
    the forwardslash can be directly represented as a forwardslash.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; either "auelp", "apelp", "swidt", "delaunay", or
        "voronoi" (corresponding to artificial uniform end-linked
        polymer networks ("auelp"), artificial polydisperse end-linked
        polymer networks ("apelp"), spider web-inspired
        Delaunay-triangulated networks ("swidt"), Delaunay-triangulated
        networks ("delaunay"), or Voronoi-tessellated networks
        ("voronoi")).
    
    Returns:
        str: The baseline filepath.
    
    """
    import os
    import pathlib

    # For MacOS
    # filepath = f"/Users/jasonmulderrig/research/projects/heterogeneous-spatial-networks/{network}/"
    # For Windows OS
    filepath = f"C:\\Users\\mulderjp\\projects\\polymer-network-topology-graph-design\\modular-sandbox\\heterogeneous-spatial-networks\\{network}\\"
    # For Linux
    # filepath = f"/p/home/jpm2225/projects/polymer-network-topology-graph-design/module-sandbox/heterogeneous-spatial-networks/{network}/"
    if os.path.isdir(filepath) == False:
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    return filepath

def filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int) -> str:
    """Filename generator for heterogeneous spatial networks.

    This function returns the baseline filename for files involved with
    heterogeneous spatial networks. The filename is explicitly prefixed
    with the filepath to the directory that the files ought to be saved
    to (and loaded from for future use). This filepath is set by the
    user, and must match the directory structure of the local computer.
    The baseline filename is then appended to the filepath. It is
    incumbent on the user to save a data file that records the network
    parameter values that correspond to each network sample in the batch
    (i.e., a "lookup table").

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; either "auelp", "apelp", "swidt", "delaunay", or
        "voronoi" (corresponding to artificial uniform end-linked
        polymer networks ("auelp"), artificial polydisperse end-linked
        polymer networks ("apelp"), spider web-inspired
        Delaunay-triangulated networks ("swidt"), Delaunay-triangulated
        networks ("delaunay"), or Voronoi-tessellated networks
        ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
    
    Returns:
        str: The baseline filename.
    
    """
    return filepath_str(network) + f"{date}{batch}{sample:d}"

def p_gaussian_cnfrmtn_func(b: float, nu: ArrayLike, r: ArrayLike) -> ArrayLike:
    """Gaussian end-to-end distance polymer chain conformation
    probability distribution.

    This function calculates the Gaussian end-to-end distance polymer
    chain conformation probability for a chain with a given number of
    segments and a given end-to-end distance.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        nu (ArrayLike): Number of segments in the chain.
        r (ArrayLike): End-to-end chain distance.
    
    Note: If nu is an np.ndarray, then r must be a float. Likewise, if r
    is an np.ndarray, then nu must be a float.

    Returns:
        ArrayLike: Gaussian end-to-end distance polymer chain
        conformation probability (distribution).
    """
    return (
        (np.sqrt(3/(2*np.pi*nu*b**2)))**3 * np.exp(-3*r**2/(2*nu*b**2))
        * 4 * np.pi * r**2
    )

def p_nu_flory_func(nu_mean: float, nu: ArrayLike) -> ArrayLike:
    """Chain segment number probability distribution representative of
    step-growth linear chain polymerization, as per the theory from
    Flory.

    This function calculates the probability of finding a chain with a
    given segment number in a polymer network formed via step-growth
    linear chain polymerization with a given mean segment number.

    Args:
        nu_mean (float): Average number of segments in the polymer
        network.
        nu (ArrayLike): Number of segments in the chain.

    Returns:
        ArrayLike: Chain segment number probability (distribution)
        representative of step-growth linear chain polymerization.
    """
    return (1./nu_mean)*(1.-(1./nu_mean))**(nu-1)

def p_net_gaussian_cnfrmtn_func(
        b: float,
        nu_mean: float,
        nu: ArrayLike,
        r: ArrayLike) -> ArrayLike:
    """Probability distribution of network chains accounting for
    dispersity in chain segment number and Gaussian end-to-end distance
    polymer chain conformation.

    This function calculates the probability of finding a chain with a
    given segment number in a particular conformation, assuming that the
    polymer network was formed via step-growth linear chain
    polymerization and assuming that the end-to-end distance chain
    conformation probability distribution is well captured via the
    classical Gaussian distribution.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        nu_mean (float): Average number of segments in the polymer
        network.
        nu (ArrayLike): Number of segments in the chain.
        r (ArrayLike): End-to-end chain distance.
    
    Note: If nu is an np.ndarray, then r must be a float. Likewise, if r
    is an np.ndarray, then nu must be a float.

    Returns:
        ArrayLike: Probability of finding a chain with a given segment
        number in a particular conformation, assuming step-growth linear
        chain polymerization and a Gaussian end-to-end distance chain
        conformation probability distribution.
    """
    return p_nu_flory_func(nu_mean, nu) * p_gaussian_cnfrmtn_func(b, nu, r)

def p_rel_net_r_float_gaussian_cnfrmtn_func(
        b: float,
        nu_mean: float,
        r: float) -> float:
    """Relative probability distribution of network chains accounting
    for dispersity in chain segment number and Gaussian end-to-end
    distance polymer chain conformation.

    This function calculates the relative probability of finding a chain
    in a particular conformation, assuming that the polymer network was
    formed via step-growth linear chain polymerization and assuming that
    the end-to-end distance chain conformation probability distribution
    is well captured via the classical Gaussian distribution.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        nu_mean (float): Average number of segments in the polymer
        network.
        r (float): End-to-end chain distance.

    Returns:
        float: Relative probability of finding a chain in a particular
        conformation, assuming step-growth linear chain polymerization
        and a Gaussian end-to-end distance chain conformation
        probability distribution.
    """
    # nu = 1 -> n = 100,000 is proscribed by Hanson
    nu_arr = np.arange(100000, dtype=int) + 1
    return np.sum(p_net_gaussian_cnfrmtn_func(b, nu_mean, nu_arr, r))

def p_rel_net_r_arr_gaussian_cnfrmtn_func(
        b: float,
        nu_mean: float,
        r: np.ndarray) -> np.ndarray:
    """Relative probability distribution of network chains accounting
    for dispersity in chain segment number and Gaussian end-to-end
    distance polymer chain conformation.

    This function calculates the relative probability of finding a chain
    in a particular conformation, assuming that the polymer network was
    formed via step-growth linear chain polymerization and assuming that
    the end-to-end distance chain conformation probability distribution
    is well captured via the classical Gaussian distribution.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        nu_mean (float): Average number of segments in the polymer
        network.
        r (np.ndarray): End-to-end chain distance.

    Returns:
        np.ndarray: Relative probability of finding a chain in a
        particular conformation, assuming step-growth linear chain
        polymerization and a Gaussian end-to-end distance chain
        conformation probability distribution.
    """
    # nu = 1 -> n = 100,000 is proscribed by Hanson
    nu_arr = np.arange(100000, dtype=int) + 1
    r_num = np.shape(r)[0]
    p_arr = np.empty(r_num)
    # Calculate relative probability for each value of r
    for r_indx in range(r_num):
        p_arr[r_indx] = np.sum(
            p_net_gaussian_cnfrmtn_func(b, nu_mean, nu_arr, r[r_indx]))
    return p_arr

def a_or_v_func(dim: int, b: float) -> float:
    """Area or volume of a chain segment and/or cross-linker.

    This function calculates the area or volume of a chain segment
    and/or cross-linker, given its diameter.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Chain segment and/or cross-linker diameter.

    Returns:
        float: Chain segment and/or cross-linker area or volume.
    """
    return np.pi * b**dim / (2*dim)

def m_arg_stoich_func(n: float, k: float) -> float:
    """Number of chains.

    This function calculates the number of chains, given the number of
    cross-linkers and the maximum cross-linker degree/functionality.
    This calculation assumes a stoichiometric mixture of cross-linkers
    and chains.

    Args:
        n (float): Number of cross-linkers.
        k (float): Maximum cross-linker degree/functionality.

    Returns:
        float: Number of chains.
    """
    return n * k / 2

def m_arg_nu_func(n_nu: float, nu: float) -> float:
    """Number of chains.

    This function calculates the number of chains, given the number of
    chain segments and the (average) number of segments per chain.

    Args:
        n_nu (float): Number of chain segments.
        nu (float): (Average) Number of segments per chain.

    Returns:
        float: Number of chains.
    """
    return n_nu / nu

def nu_arg_m_func(n_nu: float, m: float) -> float:
    """Number of chains.

    This function calculates the number of chains, given the number of
    chain segments and the (average) number of segments per chain.

    Args:
        n_nu (float): Number of chain segments.
        m (float): Number of chains.

    Returns:
        float: (Average) Number of segments per chain.
    """
    return n_nu / m

def n_arg_stoich_func(m: float, k: float) -> float:
    """Number of cross-linkers.

    This function calculates the number of cross-linkers, given the
    number of chains and the maximum cross-linker degree/functionality.
    This calculation assumes a stoichiometric mixture of cross-linkers
    and chains.

    Args:
        m (float): Number of chains.
        k (float): Maximum cross-linker degree/functionality.

    Returns:
        float: Number of chains.
    """
    return 2 * m / k

def n_nu_arg_m_func(m: float, nu: float) -> float:
    """Number of chain segments.

    This function calculates the number of chain segments, given the
    number of chains and the (average) number of segments per chain.

    Args:
        m (float): Number of chains.
        nu (float): (Average) Number of segments per chain.

    Returns:
        float: Number of chain segments.
    """
    return m * nu

def n_arg_n_tot_func(n_tot: float, n_other: float) -> float:
    """Number of particles (chain segments or cross-linkers).

    This function calculates the number of particles (chain segments or
    cross-linkers), given the number of constituents and the number of
    the other type of particle (cross-linkers or chain segments,
    respectively).

    Args:
        n_tot (float): Number of constituents.
        n_other (float): Number of the other type of particles
        (cross-linkers or chain segments).

    Returns:
        float: Number of particles (chain segments or cross-linkers,
        respectively).
    """
    return n_tot - n_other

def n_arg_f_func(f: float, n_tot: float) -> float:
    """Number of particles (chain segments or cross-linkers).

    This function calculates the number of particles (chain segments or
    cross-linkers), given the particle number fraction and the number of
    constituents.

    Args:
        f (float): Particle (chain segment or cross-linker) number
        fraction.
        n_tot (float): Number of constituents.

    Returns:
        float: Number of particles (chain segments or cross-linkers).
    """
    return f * n_tot

def n_tot_arg_n_func(n_nu: float, n: float) -> float:
    """Number of constituents.

    This function calculates the number of constituents,
    given the number of chain segments and the number of cross-linkers.

    Args:
        n_nu (float): Number of chain segments.
        n (float): Number of cross-linkers.

    Returns:
        float: Number of constituents.
    """
    return n_nu + n

def n_tot_arg_f_func(n: float, f: float) -> float:
    """Number of constituents.
    
    This function calculates the number of constituents, given the
    number of particles (chain segments or cross-linkers) and its number
    fraction.

    Args:
        n (float): Number of particles (chain segments or cross-linkers)
        f (float): Particle (chain segment or cross-linker) number
        fraction.

    Returns:
        float: Number of constituents.
    """
    return n / f

def f_arg_n_func(n: float, n_tot: float) -> float:
    """Particle (chain segment or cross-linker) number fraction.

    This function calculates the particle (chain segment or
    cross-linker) number fraction, given the number of particles (chain
    segments or cross-linkers) and and the number of constituents.

    Args:
        n (float): Number of particles (chain segments or cross-linkers)
        n_tot (float): Number of constituents.

    Returns:
        float: Particle (chain segment or cross-linker) number fraction.
    """
    return n / n_tot

def f_arg_f_func(f_other: float) -> float:
    """Particle (chain segment or cross-linker) number fraction.

    This function calculates the particle (chain segment or
    cross-linker) number fraction, given the other particle
    (cross-linker or chain segment, respectively) number fraction.

    Args:
        f_other (float): Other particle (cross-linker or chain segment,
        respectively) number fraction.

    Returns:
        float: Particle (chain segment or cross-linker) number fraction.
    """
    return 1 - f_other

def rho_func(n: float, A_or_V: float) -> float:
    """Particle number density.

    This function calculates the particle number density given the
    number of particles and the simulation box area or volume in two or
    three dimensions, respectively.

    Args:
        n (float): Number of particles.
        A_or_V (float): Simulation box area or volume.

    Returns:
        float: Particle number density.
    """
    return n / A_or_V

def eta_func(dim: int, b: float, rho: float) -> float:
    """Particle packing density.

    This function calculates the particle packing density given the
    particle number density.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Particle diameter.
        rho (float): Particle number density.

    Returns:
        float: Particle packing density.
    """
    return a_or_v_func(dim, b) * rho

def A_or_V_arg_L_func(dim: int, L: float) -> float:
    """Simulation box area or volume.

    This function calculates the simulation box area or volume in two or
    three dimensions, respectively, given the simulation box size.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size.

    Returns:
        float: Simulation box area or volume.
    """
    return L**dim

def A_or_V_arg_rho_func(n: float, rho: float) -> float:
    """Simulation box area or volume.

    This function calculates the simulation box area or volume in two or
    three dimensions, respectively, given the number of particles and
    the particle number density.

    Args:
        n (float): Number of particles.
        rho (float): Particle number density.

    Returns:
        float: Simulation box area or volume.
    """
    return n / rho

def A_or_V_arg_eta_func(dim: int, b: float, n: float, eta: float) -> float:
    """Simulation box area or volume.
    
    This function calculates the simulation box area or volume in two or
    three dimensions, respectively, given the number of particles and
    the particle packing density.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Particle diameter.
        n (float): Number of particles.
        eta (float): Particle packing density.

    Returns:
        float: Simulation box area or volume.
    """
    return a_or_v_func(dim, b) * n / eta

def L_arg_A_or_V_func(dim: int, A_or_V: float) -> float:
    """Simulation box size.
    
    This function calculates the simulation box size given the
    simulation box area or volume in two or three dimensions,
    respectively.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        A_or_V (float): Simulation box area or volume.

    Returns:
        float: Simulation box size.
    """
    if dim == 2: return np.sqrt(A_or_V)
    elif dim == 3: return np.cbrt(A_or_V)

def L_arg_rho_func(dim: int, n: float, rho: float) -> float:
    """Simulation box size.
    
    This function calculates the simulation box size given the number of
    particles and the particle number density.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        n (float): Number of particles.
        rho (float): Particle number density.

    Returns:
        float: Simulation box size.
    """
    if dim == 2: return np.sqrt(n/rho)
    elif dim == 3: return np.cbrt(n/rho)

def L_arg_eta_func(dim: int, b: float, n: float, eta: float) -> float:
    """Simulation box size.
    
    This function calculates the simulation box size given the number of
    particles and the particle packing density.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Particle diameter.
        n (float): Number of particles.
        eta (float): Particle packing density.

    Returns:
        float: Simulation box size.
    """
    if dim == 2: return np.sqrt(a_or_v_func(dim, b)*n/eta)
    elif dim == 3: return np.cbrt(a_or_v_func(dim, b)*n/eta)

def delaunay_or_voronoi_L(
        dim: int,
        b: float,
        n: int,
        eta_n: float,
        L_filename: str) -> None:
    """Simulation box size for Delaunay-triangulated or
    Voronoi-tessellated networks.

    This function calculates and saves the simulation box size for
    Delaunay-triangulated or Voronoi-tessellated networks.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended number of core nodes.
        eta_n (float): Node packing density.
        L_filename (str): Filename for simulation box size file
    
    """
    # Calculate L
    L = L_arg_eta_func(dim, b, n, eta_n)
    
    # Save L
    np.savetxt(L_filename, [L])

def voronoi_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float) -> None:
    """Simulation box size for Voronoi-tessellated networks.

    This function calculates and saves the simulation box size for
    Voronoi-tessellated networks.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "voronoi" is applicable (corresponding
        to Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended number of core nodes.
        eta_n (float): Node packing density.
    
    """
    # This calculation for L is only applicable for Voronoi-tessellated
    # networks. Exit if a different type of network is passed.
    if network != "voronoi":
        error_str = (
            "This calculation for L is only applicable for "
            + "Voronoi-tessellated networks networks. This calculation "
            + "will only proceed if network = ``voronoi''."
        )
        sys.exit(error_str)
    # Generate filename
    L_filename = filename_str(network, date, batch, sample) + "-L" + ".dat"

    # Calculate and save L
    delaunay_or_voronoi_L(dim, b, n, eta_n, L_filename)

def delaunay_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float) -> None:
    """Simulation box size for Delaunay-triangulated networks.

    This function calculates and saves the simulation box size for
    Delaunay-triangulated networks.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "delaunay" is applicable (corresponding
        to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended number of core nodes.
        eta_n (float): Node packing density.
    
    """
    # This calculation for L is only applicable for
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "delaunay":
        error_str = (
            "This calculation for L is only applicable for "
            + "Delaunay-triangulated networks networks. This calculation "
            + "will only proceed if network = ``delaunay''."
        )
        sys.exit(error_str)
    # Generate filename
    L_filename = filename_str(network, date, batch, sample) + "-L" + ".dat"

    # Calculate and save L
    delaunay_or_voronoi_L(dim, b, n, eta_n, L_filename)

def swidt_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float) -> None:
    """Simulation box size for spider web-inspired Delaunay-triangulated
    networks.

    This function calculates the simulation box size for spider
    web-inspired Delaunay-triangulated networks.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "swidt" is applicable (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended number of core nodes.
        eta_n (float): Node packing density.
    
    """
    # This calculation for L is only applicable for spider web-inspired
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "swidt":
        error_str = (
            "This calculation for L is only applicable for spider "
            + "web-inspired Delaunay-triangulated networks. This "
            + "calculation will only proceed if network = ``swidt''."
        )
        sys.exit(error_str)
    # Generate filename
    L_filename = filename_str(network, date, batch, sample) + "-L" + ".dat"

    # Calculate and save L
    delaunay_or_voronoi_L(dim, b, n, eta_n, L_filename)

def aelp_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        rho_nu: float,
        k: int,
        n: int,
        nu: int) -> None:
    """Simulation box size for artificial end-linked polymer networks.

    This function calculates the simulation box size for artificial
    end-linked polymer networks.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, either "auelp" or "apelp" are applicable
        (corresponding to artificial uniform end-linked polymer networks
        ("auelp") or artificial polydisperse end-linked polymer networks
        ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        rho_nu (float): Segment number density.
        k (int): Maximum cross-linker degree/functionality; either 3, 4,
        5, 6, 7, or 8.
        n (int): Intended number of core cross-linkers.
        nu (int): (Average) Number of segments per chain.
    
    """
    # This calculation for L is only applicable for artificial
    # end-linked polymer networks. Exit if a different type of network
    # is passed.
    if network != "auelp":
        if network != "apelp":
            error_str = (
                "This calculation for L is only applicable for "
                + "artificial end-linked polymer networks. This "
                + "calculation will only proceed if "
                + "network = ``auelp'' or network = ``apelp''."
            )
            sys.exit(error_str)
    # Generate filename
    L_filename = filename_str(network, date, batch, sample) + "-L" + ".dat"

    # Calculate the stoichiometric (average) number of chain segments in
    # the simulation box
    n_nu = n_nu_arg_m_func(m_arg_stoich_func(n, k), nu)

    # Calculate L
    L = L_arg_rho_func(dim, n_nu, rho_nu)
    
    # Save L
    np.savetxt(L_filename, [L])

def tessellation_protocol(dim: int) -> tuple[np.ndarray, int]:
    """Tessellation protocol.

    This function determines the tessellation protocol and the number of
    tessellations involved in that protocol. Each of these are sensitive
    to the physical dimensionality of the network.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
    
    Returns:
        tuple[np.ndarray, int]: Tessellation protocol and the number of
        tessellations involved in that protocol, respectively.
    
    """
    tsslltn = np.asarray([-1, 0, 1], dtype=int)
    if dim == 2:
        tsslltn_protocol = (
            np.asarray(np.meshgrid(tsslltn, tsslltn)).T.reshape(-1, 2)
        )
    elif dim == 3:
        tsslltn_protocol = (
            np.asarray(np.meshgrid(tsslltn, tsslltn, tsslltn)).T.reshape(-1, 3)
        )
    tsslltn_num = np.shape(tsslltn_protocol)[0]
    return tsslltn_protocol, tsslltn_num

def dim_2_tessellation_protocol(
        L: float,
        x: float,
        y: float,
        dim_2_tsslltn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-dimensional tessellation protocol.

    This function fully tessellates (or translates) an (x, y) coordinate
    about the x-y plane in the x- and y-directions via a scaling
    distance L.

    Args:
        L (float): Tessellation scaling distance.
        x (float): x-coordinate to be tessellated.
        y (float): y-coordinate to be tessellated.
        dim_2_tsslltn (np.ndarray): Two-dimensional tessellation
        protocol.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Tessellated x-coordinates and
        y-coordinates, respectively.
    
    """
    x_tsslltn = dim_2_tsslltn[:, 0]
    y_tsslltn = dim_2_tsslltn[:, 1]
    return x + x_tsslltn * L, y + y_tsslltn * L

def dim_3_tessellation_protocol(
        L: float,
        x: float,
        y: float,
        z: float,
        dim_3_tsslltn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-dimensional tessellation protocol.

    This function fully tessellates (or translates) an (x, y, z)
    coordinate about the x-y-z plane in the x-, y-, and z-directions via
    a scaling distance L.

    Args:
        L (float): Tessellation scaling distance.
        x (float): x-coordinate to be tessellated.
        y (float): y-coordinate to be tessellated.
        z (float): z-coordinate to be tessellated.
        dim_3_tsslltn (np.ndarray): Three-dimensional tessellation
        protocol.
    
    Returns:
        tuple[np.ndarray, np.ndarray,np.ndarray]: Tessellated x-, y-,
        and z-coordinates, respectively.
    
    """
    x_tsslltn = dim_3_tsslltn[:, 0]
    y_tsslltn = dim_3_tsslltn[:, 1]
    z_tsslltn = dim_3_tsslltn[:, 2]
    return x + x_tsslltn * L, y + y_tsslltn * L, z + z_tsslltn * L

def dim_2_tessellation(
        L: float,
        x: ArrayLike,
        y: ArrayLike,
        x_tsslltn: int,
        y_tsslltn: int) -> tuple[ArrayLike, ArrayLike]:
    """Two-dimensional tessellation procedure.

    This function tessellates (or translates) an (x, y) coordinate(s)
    about the x-y plane in the x- and y-directions via a scaling
    distance L.

    Args:
        L (float): Tessellation scaling distance.
        x (ArrayLike): x-coordinate(s) to be tessellated.
        y (ArrayLike): y-coordinate(s) to be tessellated.
        x_tsslltn (int): Tessellation action for the
        x-coordinate(s); either -1, 0, or 1.
        y_tsslltn (int): Tessellation action for the
        y-coordinate(s); either -1, 0, or 1.
    
    Returns:
        tuple[ArrayLike, ArrayLike]: Tessellated x-coordinate(s) and
        y-coordinate(s), respectively.
    
    """
    return x + x_tsslltn * L, y + y_tsslltn * L

def dim_3_tessellation(
        L: float,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        x_tsslltn: int,
        y_tsslltn: int,
        z_tsslltn: int) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Three-dimensional tessellation procedure.

    This function tessellates (or translates) an (x, y, z) coordinate(s)
    about the x-y-z plane in the x-, y-, and z-directions via a scaling
    distance L.

    Args:
        L (float): Tessellation scaling distance.
        x (ArrayLike): x-coordinate(s) to be tessellated.
        y (ArrayLike): y-coordinate(s) to be tessellated.
        z (ArrayLike): z-coordinate(s) to be tessellated.
        x_tsslltn (int): Tessellation action for the
        x-coordinate(s); either -1, 0, or 1.
        y_tsslltn (int): Tessellation action for the
        y-coordinate(s); either -1, 0, or 1.
        z_tsslltn (int): Tessellation action for the
        z-coordinate(s); either -1, 0, or 1.
    
    Returns:
        tuple[ArrayLike, ArrayLike, ArrayLike]: Tessellated
        x-coordinate(s), y-coordinate(s), and z-coordinate(s),
        respectively.
    
    """
    return x + x_tsslltn * L, y + y_tsslltn * L, z + z_tsslltn * L

def random_node_placement(
        dim: int,
        L: float,
        n: int,
        filename_prefix: str) -> None:
    """Random node placement procedure.

    This function randomly places/seeds nodes within a pre-defined
    simulation box.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size.
        n (int): Number of nodes.
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Generate data filename
    config_filename = filename_prefix + ".config"

    # Initialize random number generator
    rng = np.random.default_rng()

    # Random node coordinates
    coords = L * rng.random((n, dim))

    # Save coordinates to configuration file
    np.savetxt(config_filename, coords)

def periodic_random_hard_disk_node_placement(
        dim: int,
        L: float,
        b: float,
        n: int,
        max_try: int,
        filename_prefix: str) -> None:
    """Periodic random hard disk node placement procedure.

    This function randomly places/seeds nodes within a pre-defined
    simulation box where each nodes is treated as the center of a hard
    disk, and the simulation box is periodic.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size.
        b (float): Hard disk diameter.
        n (int): Intended number of nodes.
        max_try (int): Maximum number of node placement attempts.
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Generate data filename
    config_filename = filename_prefix + ".config"

    # Initialize random number generator
    rng = np.random.default_rng()

    # Lists for node x- and y-coordinates
    x = []
    y = []

    # np.ndarrays for x- and y-coordinates of tessellated core nodes
    tsslltd_x = np.asarray([])
    tsslltd_y = np.asarray([])

    if dim == 2:
        # Two-dimensional tessellation protocol
        dim_2_tsslltn, dim_2_tsslltn_num = tessellation_protocol(2)
        
        # Periodic random hard disk node placement procedure
        for seed_attmpt in range(n):
            # Accept and tessellate the first node
            if seed_attmpt == 0:
                # Accept the first node
                seed = L * rng.random((2,))
                
                seed_x = seed[0]
                seed_y = seed[1]
                
                x.append(seed_x)
                y.append(seed_y)

                # Use two-dimensional tessellation protocol to
                # tessellate the first node
                seed_tsslltn_x, seed_tsslltn_y = dim_2_tessellation_protocol(
                    L, seed_x, seed_y, dim_2_tsslltn)
                tsslltd_x = np.concatenate((tsslltd_x, seed_tsslltn_x))
                tsslltd_y = np.concatenate((tsslltd_y, seed_tsslltn_y))
                continue
            else:
                # Begin periodic random hard disk node placement
                # procedure
                num_try = 0

                while num_try < max_try:
                    # Generate randomly placed node candidate
                    seed_cnddt = L * rng.random((2,))
                    seed_cnddt_x = seed_cnddt[0]
                    seed_cnddt_y = seed_cnddt[1]

                    # Downselect the previously-accepted tessellated
                    # nodes to those that reside in a local square
                    # neighborhood that is \pm b about the node
                    # candidate. Start by gathering the indices of nodes
                    # that meet this criterion in each separate
                    # coordinate.
                    nghbr_x_lb = seed_cnddt_x - b
                    nghbr_x_ub = seed_cnddt_x + b
                    nghbr_y_lb = seed_cnddt_y - b
                    nghbr_y_ub = seed_cnddt_y + b
                    psbl_nghbr_x_indcs = (
                        np.where(np.logical_and(tsslltd_x>=nghbr_x_lb, tsslltd_x<=nghbr_x_ub))[0]
                    )
                    psbl_nghbr_y_indcs = (
                        np.where(np.logical_and(tsslltd_y>=nghbr_y_lb, tsslltd_y<=nghbr_y_ub))[0]
                    )
                    # Gather the indices from each separate coordinate
                    # together to assess all possible node neighbors.
                    # Retain unique indices corresponding to each
                    # possible node neighbor, and the number of times
                    # each such index value appears.
                    psbl_nghbr_indcs, psbl_nghbr_indcs_counts = np.unique(
                        np.concatenate((psbl_nghbr_x_indcs, psbl_nghbr_y_indcs), dtype=int),
                        return_counts=True)
                    # The true node neighbors are those whose index
                    # value appears twice in the possible node neighbor
                    # array -- equal to the network dimensionality
                    nghbr_indcs_vals_indcs = (
                        np.where(psbl_nghbr_indcs_counts == 2)[0]
                    )
                    nghbr_num = np.shape(nghbr_indcs_vals_indcs)[0]
                    # Continue analysis if a local neighborhood of
                    # tessellated nodes actually exists about the node
                    # candidate
                    if nghbr_num > 0:
                        # Gather the indices of the node neighbors
                        nghbr_indcs = psbl_nghbr_indcs[nghbr_indcs_vals_indcs]
                        # Extract node neighbor coordinates
                        nghbr_tsslltd_x = tsslltd_x[nghbr_indcs]
                        nghbr_tsslltd_y = tsslltd_y[nghbr_indcs]
                        
                        # Calculate the minimum distance between the
                        # node candidate and its neighbors
                        dist = np.empty(nghbr_num)
                        for nghbr_indx in range(nghbr_num):
                            nghbr_tsslltd_rccs = np.asarray(
                                [
                                    nghbr_tsslltd_x[nghbr_indx],
                                    nghbr_tsslltd_y[nghbr_indx]
                                ]
                            )
                            dist[nghbr_indx] = (
                                np.linalg.norm(seed_cnddt-nghbr_tsslltd_rccs)
                            )
                        min_dist = np.min(dist)

                        # Try again if the minimum distance between the
                        # node candidate and its neighbors is less than
                        # b
                        if min_dist < b:
                            num_try += 1
                            continue
                    
                    # Accept and tessellate the node candidate if (1) no
                    # local neighborhood of tessellated nodes exists
                    # about the node candidate, or (2) the minimum
                    # distance between the node candidate and its
                    # neighbors is greater than or equal to b
                    x.append(seed_cnddt_x)
                    y.append(seed_cnddt_y)

                    # Use two-dimensional tessellation protocol to
                    # tessellate the accepted node candidate
                    seed_tsslltn_x, seed_tsslltn_y = (
                        dim_2_tessellation_protocol(
                            L, seed_cnddt_x, seed_cnddt_y, dim_2_tsslltn)
                    )
                    tsslltd_x = np.concatenate((tsslltd_x, seed_tsslltn_x))
                    tsslltd_y = np.concatenate((tsslltd_y, seed_tsslltn_y))
                    break
        
        # Convert x- and y-coordinate lists to np.ndarrays, and stack
        # the x- and y-coordinates next to each other columnwise
        x = np.asarray(x)
        y = np.asarray(y)
        coords = np.column_stack((x, y))
    elif dim == 3:
        # List for z-coordinates
        z = []

        # np.ndarray for z-coordinates of tessellated core nodes
        tsslltd_z = np.asarray([])

        # Three-dimensional tessellation protocol
        dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)
        
        # Periodic random hard disk node placement procedure
        for seed_attmpt in range(n):
            # Accept and tessellate the first node
            if seed_attmpt == 0:
                # Accept the first node
                seed = L * rng.random((3,))
                
                seed_x = seed[0]
                seed_y = seed[1]
                seed_z = seed[2]
                
                x.append(seed_x)
                y.append(seed_y)
                z.append(seed_z)

                # Use three-dimensional tessellation protocol to
                # tessellate the first node
                seed_tsslltn_x, seed_tsslltn_y, seed_tsslltn_z = (
                    dim_3_tessellation_protocol(
                        L, seed_x, seed_y, seed_z, dim_3_tsslltn)
                )
                tsslltd_x = np.concatenate((tsslltd_x, seed_tsslltn_x))
                tsslltd_y = np.concatenate((tsslltd_y, seed_tsslltn_y))
                tsslltd_z = np.concatenate((tsslltd_z, seed_tsslltn_z))
                continue
            else:
                # Begin periodic random hard disk node placement
                # procedure
                num_try = 0

                while num_try < max_try:
                    # Generate randomly placed node candidate
                    seed_cnddt = L * rng.random((3,))
                    seed_cnddt_x = seed_cnddt[0]
                    seed_cnddt_y = seed_cnddt[1]
                    seed_cnddt_z = seed_cnddt[2]

                    # Downselect the previously-accepted tessellated
                    # nodes to those that reside in a local cube
                    # neighborhood that is \pm b about the node
                    # candidate. Start by gathering the indices of nodes
                    # that meet this criterion in each separate
                    # coordinate.
                    nghbr_x_lb = seed_cnddt_x - b
                    nghbr_x_ub = seed_cnddt_x + b
                    nghbr_y_lb = seed_cnddt_y - b
                    nghbr_y_ub = seed_cnddt_y + b
                    nghbr_z_lb = seed_cnddt_z - b
                    nghbr_z_ub = seed_cnddt_z + b
                    psbl_nghbr_x_indcs = (
                        np.where(np.logical_and(tsslltd_x>=nghbr_x_lb, tsslltd_x<=nghbr_x_ub))[0]
                    )
                    psbl_nghbr_y_indcs = (
                        np.where(np.logical_and(tsslltd_y>=nghbr_y_lb, tsslltd_y<=nghbr_y_ub))[0]
                    )
                    psbl_nghbr_z_indcs = (
                        np.where(np.logical_and(tsslltd_z>=nghbr_z_lb, tsslltd_z<=nghbr_z_ub))[0]
                    )
                    # Gather the indices from each separate coordinate
                    # together to assess all possible node neighbors.
                    # Retain unique indices corresponding to each
                    # possible node neighbor, and the number of times
                    # each such index value appears.
                    psbl_nghbr_indcs, psbl_nghbr_indcs_counts = np.unique(
                        np.concatenate((psbl_nghbr_x_indcs, psbl_nghbr_y_indcs, psbl_nghbr_z_indcs), dtype=int),
                        return_counts=True)
                    # The true node neighbors are those whose index
                    # value appears thrice in the possible node neighbor
                    # array -- equal to the network dimensionality
                    nghbr_indcs_vals_indcs = (
                        np.where(psbl_nghbr_indcs_counts == 3)[0]
                    )
                    nghbr_num = np.shape(nghbr_indcs_vals_indcs)[0]
                    # Continue analysis if a local neighborhood of
                    # tessellated nodes actually exists about the node
                    # candidate
                    if nghbr_num > 0:
                        # Gather the indices of the node neighbors
                        nghbr_indcs = psbl_nghbr_indcs[nghbr_indcs_vals_indcs]
                        # Extract node neighbor coordinates
                        nghbr_tsslltd_x = tsslltd_x[nghbr_indcs]
                        nghbr_tsslltd_y = tsslltd_y[nghbr_indcs]
                        nghbr_tsslltd_z = tsslltd_z[nghbr_indcs]
                        
                        # Calculate the minimum distance between the
                        # node candidate and its neighbors
                        dist = np.empty(nghbr_num)
                        for nghbr_indx in range(nghbr_num):
                            nghbr_tsslltd_rccs = np.asarray(
                                [
                                    nghbr_tsslltd_x[nghbr_indx],
                                    nghbr_tsslltd_y[nghbr_indx],
                                    nghbr_tsslltd_z[nghbr_indx]
                                ]
                            )
                            dist[nghbr_indx] = (
                                np.linalg.norm(seed_cnddt-nghbr_tsslltd_rccs)
                            )
                        min_dist = np.min(dist)

                        # Try again if the minimum distance between the
                        # node candidate and its neighbors is
                        # less than b
                        if min_dist < b:
                            num_try += 1
                            continue
                    
                    # Accept and tessellate the node candidate if (1) no
                    # local neighborhood of tessellated nodes exists
                    # about the node candidate, or (2) the minimum
                    # distance between the node candidate and its
                    # neighbors is greater than or equal to b
                    x.append(seed_cnddt_x)
                    y.append(seed_cnddt_y)
                    z.append(seed_cnddt_z)

                    # Use three-dimensional tessellation protocol to
                    # tessellate the accepted node candidate
                    seed_tsslltn_x, seed_tsslltn_y, seed_tsslltn_z = (
                        dim_3_tessellation_protocol(
                            L, seed_cnddt_x, seed_cnddt_y,
                            seed_cnddt_z, dim_3_tsslltn)
                    )
                    tsslltd_x = np.concatenate((tsslltd_x, seed_tsslltn_x))
                    tsslltd_y = np.concatenate((tsslltd_y, seed_tsslltn_y))
                    tsslltd_z = np.concatenate((tsslltd_z, seed_tsslltn_z))
                    break
        
        # Convert x-, y-, and z-coordinate lists to np.ndarrays, and
        # stack the x-, y-, and z-coordinates next to each other
        # columnwise
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        coords = np.column_stack((x, y, z))

    # Save coordinates to configuration file
    np.savetxt(config_filename, coords)

def periodic_disordered_hyperuniform_node_placement(
        dim: int,
        L: float,
        n: int,
        filename_prefix: str) -> None:
    """Periodic disordered hyperuniform node placement procedure.

    This function places/seeds nodes within a pre-defined simulation box
    in a periodic disordered hyperuniform fashion.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size.
        n (int): Intended number of nodes.
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Generate data filename
    config_filename = filename_prefix + ".config"

    # Initialize random number generator
    rng = np.random.default_rng()
    
    error_str = (
        "Periodic disordered hyperuniform node placement procedure has "
        + "not been defined yet!"
    )
    sys.exit(error_str)

def lammps_input_file_generator(
        dim: int,
        L: float,
        b: float,
        n: int,
        filename_prefix: str) -> None:
    """LAMMPS input file generator for soft pushoff and FIRE energy
    minimization of randomly placed nodes.

    This function generates a LAMMPS input file that randomly places
    nodes within a pre-defined simulation box, followed by a soft
    pushoff procedure, and finally followed by a FIRE energy
    minimization procedure.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size.
        b (float): Node diameter.
        n (int): Number of core nodes.
        filename_prefix (str): Baseline filename prefix for LAMMPS data
        files.
    
    """
    # Generate filenames
    lammps_input_filename = filename_prefix + ".in"
    config_input_filename = filename_prefix + ".config"
    
    # Create random integers
    rng = np.random.default_rng()
    create_atoms_random_int = rng.integers(100000, dtype=int) + 1
    velocity_random_int = rng.integers(10000000, dtype=int) + 1
    
    lammps_input_file = open(lammps_input_filename, "w")

    # Specify fundamental parameters and simulation box
    lammps_input_file.write("units         lj\n")
    lammps_input_file.write(f"dimension     {dim:d}\n")
    lammps_input_file.write("boundary      p p p\n")
    lammps_input_file.write(f"region        box block 0 {L:0.4f} 0 {L:0.4f} 0 {L:0.4f}\n")
    lammps_input_file.write("create_box    1 box\n")
    # Randomly place nodes in simulation box
    lammps_input_file.write(f"create_atoms  1 random {n:d} {create_atoms_random_int:d} NULL\n")
    lammps_input_file.write("mass          1 1.0\n\n")

    # Initiate thermodynammics
    lammps_input_file.write(f"velocity all create 1.0 {velocity_random_int:d} mom yes rot yes dist gaussian\n\n")

    lammps_input_file.write("timestep 0.002\n\n")

    lammps_input_file.write("thermo_style custom step temp pe ke ebond pxx pyy pzz lx ly lz vol density\n")
    lammps_input_file.write("thermo 1\n\n")

    # Soft pushoff procedure
    lammps_input_file.write(f"pair_style soft {b:0.1f}\n")
    lammps_input_file.write("pair_coeff * * 10.0\n")
    lammps_input_file.write("pair_modify shift yes\n\n")

    # FIRE energy minimization procedure
    lammps_input_file.write("min_style fire\n")
    lammps_input_file.write("minimize 1.0e-10 1.0e-10 100000 100000\n\n")

    # Write coordinates to configuration file
    lammps_input_file.write(f"write_data {config_input_filename} nocoeff")

    lammps_input_file.close()

def node_seeding(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        b: float,
        n: int,
        config: int,
        max_try: int) -> None:
    """Node placement procedure for heterogeneous spatial networks.

    This function loads the simulation box size for the heterogeneous
    spatial networks. Then, depending on the particular core node
    placement scheme, this function calls upon a corresponding helper
    function to calculate the core node positions.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; either "auelp", "apelp", "swidt", "delaunay", or
        "voronoi" (corresponding to artificial uniform end-linked
        polymer networks ("auelp"), artificial polydisperse end-linked
        polymer networks ("apelp"), spider web-inspired
        Delaunay-triangulated networks ("swidt"), Delaunay-triangulated
        networks ("delaunay"), or Voronoi-tessellated networks
        ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular
        scheme used to generate the positions of the core nodes;
        either "random", "prhd", "pdhu", or "lammps" (corresponding to
        the random node placement procedure ("random"), periodic random
        hard disk node placement procedure ("prhd"), periodic disordered
        hyperuniform node placement procedure ("pdhu"), or nodes
        randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Number of core nodes.
        config (int): Configuration number.
        max_try (int): Maximum number of node placement attempts for the
        random node placement procedure (scheme = "random").
    
    """
    # Generate filename prefix
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    
    # Load L
    L = np.loadtxt(L_filename)

    # Append configuration number to filename prefix
    filename_prefix = filename_prefix + f"C{config:d}"
    
    # Call appropriate node placement helper function
    if scheme == "random":
        random_node_placement(dim, L, n, filename_prefix)
    elif scheme == "prhd":
        periodic_random_hard_disk_node_placement(
            dim, L, b, n, max_try, filename_prefix)
    elif scheme == "pdhu":
        periodic_disordered_hyperuniform_node_placement(
            dim, L, n, filename_prefix)
    elif scheme == "lammps":
        lammps_input_file_generator(dim, L, b, n, filename_prefix)

def unique_sorted_edges(edges: list[tuple[int, int]]) -> np.ndarray:
    """Unique edges.

    This function takes a list of (A, B) nodes specifying edges,
    converts this to an np.ndarray, and retains unique edges. If the
    original edge list contains edges (A, B) and (B, A), then only
    (A, B) will be retained (assuming that A <= B).

    Args:
        edges (list[tuple[int, int]]): List of edges.
    
    Returns:
        np.ndarray: Unique edges.
    
    """
    # Convert list of edges to np.ndarray, sort the order of each (A, B)
    # edge entry so that A <= B for all entries (after sorting), and
    # retain unique edges
    return np.unique(np.sort(np.asarray(edges, dtype=int), axis=1), axis=0)

def core2pb_nodes_func(
        core_nodes: np.ndarray,
        pb2core_nodes: np.ndarray) -> list[np.ndarray]:
    """List of np.ndarrays corresponding to the periodic boundary nodes
    associated with a particular core node.

    This function creates a list of np.ndarrays corresponding to the
    periodic boundary nodes associated with a particular core node such
    that core2pb_nodes[core_node] = pb_nodes.

    Args:
        core_nodes (np.ndarray): np.ndarray of the core node numbers.
        pb2core_nodes (np.ndarray): np.ndarray that returns the core
        node that corresponds to each core and periodic boundary node,
        i.e., pb2core_nodes[core_pb_node] = core_node.
    
    Returns:
        list[np.ndarray]: list of np.ndarrays corresponding to the
        periodic boundary nodes associated with a particular core node.

    """
    core2pb_nodes = []
    for core_node in np.nditer(core_nodes):
        # Isolate core and periodic boundary nodes associated with the
        # core node, and delete the core node
        pb_nodes = np.delete(
            np.where(pb2core_nodes == int(core_node))[0], 0, axis=0)
        core2pb_nodes.append(pb_nodes)
    return core2pb_nodes

def voronoi_dim_2_network_topology_initialization(
        L: float,
        core_x: np.ndarray,
        core_y: np.ndarray,
        filename_prefix: str) -> None:
    """Network topology initialization procedure for two-dimensional
    Voronoi-tessellated networks.

    This function ``tessellates'' the core nodes about themselves,
    applies Voronoi tessellation to the resulting tessellated network
    via the scipy.spatial.Voronoi() function, acquires back the
    periodic network topology of the core nodes, and ascertains
    fundamental graph constituents (node and edge information) from this
    topology.

    Args:
        L (float): Simulation box size.
        core_x (np.ndarray): x-coordinates of the input core nodes.
        core_y (np.ndarray): y-coordinates of the input core nodes.
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Import the scipy.spatial.Voronoi() function
    from scipy.spatial import Voronoi
    
    # Generate filenames
    input_core_x_filename = filename_prefix + "-input_core_x" + ".dat"
    input_core_y_filename = filename_prefix + "-input_core_y" + ".dat"
    conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"

    # Save the input core node x- and y-coordinates
    np.savetxt(input_core_x_filename, core_x)
    np.savetxt(input_core_y_filename, core_y)

    # Copy the core_x and core_y np.ndarrays as the first n entries in
    # the tessellated node x- and y-coordinate np.ndarrays
    tsslltd_core_x = core_x.copy()
    tsslltd_core_y = core_y.copy()

    # Two-dimensional tessellation protocol
    dim_2_tsslltn, dim_2_tsslltn_num = tessellation_protocol(2)

    # Use two-dimensional tessellation protocol to tessellate the core
    # nodes
    for tsslltn in range(dim_2_tsslltn_num):
        x_tsslltn = dim_2_tsslltn[tsslltn, 0]
        y_tsslltn = dim_2_tsslltn[tsslltn, 1]
        # Skip the (hold, hold) tessellation call because the core nodes
        # are being tessellated about themselves
        if (x_tsslltn == 0) and (y_tsslltn == 0): continue
        else:
            # x- and y-coordinates from two-dimensional tessellation
            # protocol
            core_tsslltn_x, core_tsslltn_y = dim_2_tessellation(
                L, core_x, core_y, x_tsslltn, y_tsslltn)
            # Concatenate the tessellated x- and y-coordinates
            tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
            tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))

    del core_tsslltn_x, core_tsslltn_y

    # Stack the tessellated x- and y-coordinates next to each other
    # columnwise
    tsslltd_core = np.column_stack((tsslltd_core_x, tsslltd_core_y))

    # Apply Voronoi tessellation
    tsslltd_core_voronoi = Voronoi(tsslltd_core)

    del tsslltd_core, tsslltd_core_x, tsslltd_core_y

    # Extract vertices from the Voronoi tessellation
    vertices = tsslltd_core_voronoi.vertices

    # Separate vertex x- and y-coordinates
    vertices_x = vertices[:, 0]
    vertices_y = vertices[:, 1]

    # Confirm that each vertex solely occupies a square neighborhood
    # that is \pm tol about itself
    tol = 1e-10
    for vertex in range(np.shape(vertices)[0]):
        # Extract x- and y-coordinates of the vertex
        vertex_x = vertices_x[vertex]
        vertex_y = vertices_y[vertex]

        # Determine if the vertex solely occupies a square neighborhood
        # that is \pm tol about itself
        nghbr_x_lb = vertex_x - tol
        nghbr_x_ub = vertex_x + tol
        nghbr_y_lb = vertex_y - tol
        nghbr_y_ub = vertex_y + tol
        psbl_nghbr_x_indcs = (
            np.where(np.logical_and(vertices_x>=nghbr_x_lb, vertices_x<=nghbr_x_ub))[0]
        )
        psbl_nghbr_y_indcs = (
            np.where(np.logical_and(vertices_y>=nghbr_y_lb, vertices_y<=nghbr_y_ub))[0]
        )
        # Gather the indices from each separate coordinate together to
        # assess all possible vertex neighbors. Retain unique indices
        # corresponding to each possible vertex neighbor, and the number
        # of times each such index value appears.
        psbl_nghbr_indcs, psbl_nghbr_indcs_counts = np.unique(
            np.concatenate((psbl_nghbr_x_indcs, psbl_nghbr_y_indcs), dtype=int),
            return_counts=True)
        # The true vertex neighbors are those whose index value appears
        # twice in the possible vertex neighbor array -- equal to the
        # network dimensionality
        nghbr_indcs_vals_indcs = np.where(psbl_nghbr_indcs_counts == 2)[0]
        nghbr_num = np.shape(nghbr_indcs_vals_indcs)[0]
        # If nghbr_num > 1, then the neighborhood is overpopulated, and
        # therefore this is an invalid set of input node coordinates for
        # Voronoi tessellation
        if nghbr_num > 1:
            error_str = (
                "A vertex neighborhood has more than one vertex! "
                + "Therefore, this is an invalid set of input node "
                + "coordinates for Voronoi tessellation."
            )
            sys.exit(error_str)
    
    # Extract core vertices and label these as core nodes
    psbl_core_x_indcs = np.where(np.logical_and(vertices_x>=0, vertices_x<L))[0]
    psbl_core_y_indcs = np.where(np.logical_and(vertices_y>=0, vertices_y<L))[0]
    # Gather the indices from each separate coordinate together to
    # assess all possible core vertices. Retain unique indices
    # corresponding to each possible core vertex, and the number of
    # times each such index value appears.
    psbl_core_indcs, psbl_core_indcs_counts = np.unique(
        np.concatenate((psbl_core_x_indcs, psbl_core_y_indcs), dtype=int),
        return_counts=True)
    # The true core vertices are those whose index value appears twice
    # in the possible core vertex array -- equal to the network
    # dimensionality
    core_indcs = psbl_core_indcs[np.where(psbl_core_indcs_counts == 2)[0]]

    # Number of core vertices
    n = np.shape(core_indcs)[0]

    # Gather core node x- and y-coordinates
    core_x = vertices_x[core_indcs]
    core_y = vertices_y[core_indcs]

    # Copy the core_x and core_y np.ndarrays as the first n entries in
    # the tessellated node x- and y-coordinate np.ndarrays
    tsslltd_core_x = core_x.copy()
    tsslltd_core_y = core_y.copy()

    # Core nodes
    core_nodes = np.arange(n, dtype=int)

    # Use two-dimensional tessellation protocol to tessellate the core
    # nodes
    for tsslltn in range(dim_2_tsslltn_num):
        x_tsslltn = dim_2_tsslltn[tsslltn, 0]
        y_tsslltn = dim_2_tsslltn[tsslltn, 1]
        # Skip the (hold, hold) tessellation call because the core nodes
        # are being tessellated about themselves
        if (x_tsslltn == 0) and (y_tsslltn == 0): continue
        else:
            # x- and y-coordinates from two-dimensional tessellation
            # protocol
            core_tsslltn_x, core_tsslltn_y = dim_2_tessellation(
                L, core_x, core_y, x_tsslltn, y_tsslltn)
            # Concatenate the tessellated x- and y-coordinates
            tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
            tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))

    del core_tsslltn_x, core_tsslltn_y

    # Construct the pb2core_nodes np.ndarray such that
    # pb2core_nodes[core_pb_node] = core_node
    pb2core_nodes = np.tile(core_nodes, dim_2_tsslltn_num)
        
    del core_nodes

    # Extract the ridge vertices from the Voronoi tessellation
    ridge_vertices = tsslltd_core_voronoi.ridge_vertices

    # List for edges of the core and periodic boundary nodes
    tsslltd_core_pb_edges = []

    for ridge_vertex in ridge_vertices:
        # In two dimensions, each ridge vertex is a line
        vertex_0 = int(ridge_vertex[0])
        vertex_1 = int(ridge_vertex[1])
        # Skip over ridge vertices that extend out to infinity
        if (vertex_0 == -1) or (vertex_1 == -1): continue
        else:
            # Determine the indices of the vertices in the tessellated
            # core node topology 
            # Extract vertex x- and y-coordinates
            vertex_0_x = vertices_x[vertex_0]
            vertex_0_y = vertices_y[vertex_0]
            vertex_1_x = vertices_x[vertex_1]
            vertex_1_y = vertices_y[vertex_1]
            # Define vertex neighborhood
            nghbr_vertex_0_x_lb = vertex_0_x - tol
            nghbr_vertex_0_x_ub = vertex_0_x + tol
            nghbr_vertex_0_y_lb = vertex_0_y - tol
            nghbr_vertex_0_y_ub = vertex_0_y + tol
            nghbr_vertex_1_x_lb = vertex_1_x - tol
            nghbr_vertex_1_x_ub = vertex_1_x + tol
            nghbr_vertex_1_y_lb = vertex_1_y - tol
            nghbr_vertex_1_y_ub = vertex_1_y + tol
            psbl_vertex_0_x_indcs = (
                np.where(np.logical_and(tsslltd_core_x>=nghbr_vertex_0_x_lb, tsslltd_core_x<=nghbr_vertex_0_x_ub))[0]
            )
            psbl_vertex_0_y_indcs = (
                np.where(np.logical_and(tsslltd_core_y>=nghbr_vertex_0_y_lb, tsslltd_core_y<=nghbr_vertex_0_y_ub))[0]
            )
            psbl_vertex_1_x_indcs = (
                np.where(np.logical_and(tsslltd_core_x>=nghbr_vertex_1_x_lb, tsslltd_core_x<=nghbr_vertex_1_x_ub))[0]
            )
            psbl_vertex_1_y_indcs = (
                np.where(np.logical_and(tsslltd_core_y>=nghbr_vertex_1_y_lb, tsslltd_core_y<=nghbr_vertex_1_y_ub))[0]
            )
            # Gather the indices from each separate coordinate together,
            # retain unique indices, and the number of times each such
            # index value appears.
            psbl_vertex_0_indcs, psbl_vertex_0_indcs_counts = np.unique(
                np.concatenate((psbl_vertex_0_x_indcs, psbl_vertex_0_y_indcs), dtype=int),
                return_counts=True)
            psbl_vertex_1_indcs, psbl_vertex_1_indcs_counts = np.unique(
                np.concatenate((psbl_vertex_1_x_indcs, psbl_vertex_1_y_indcs), dtype=int),
                return_counts=True)
            # The true vertex is whose index value appears twice in the
            # possible vertex array -- equal to the network
            # dimensionality
            vertex_0_indcs_vals_indcs = (
                np.where(psbl_vertex_0_indcs_counts == 2)[0]
            )
            vertex_0_indcs_num = np.shape(vertex_0_indcs_vals_indcs)[0]
            vertex_1_indcs_vals_indcs = (
                np.where(psbl_vertex_1_indcs_counts == 2)[0]
            )
            vertex_1_indcs_num = np.shape(vertex_1_indcs_vals_indcs)[0]
            # Skip over situations where a vertex value is not able to
            # be solved for, which does not occur for the core and
            # periodic boundary vertices
            if (vertex_0_indcs_num != 1) or (vertex_1_indcs_num != 1): continue
            else:
                # Extract vertex node index
                node_0 = int(psbl_vertex_0_indcs[vertex_0_indcs_vals_indcs][0])
                node_1 = int(psbl_vertex_1_indcs[vertex_1_indcs_vals_indcs][0])

                # If any of the nodes involved in the ridge vertex
                # correspond to the original core nodes, then add that
                # edge to the edge list. Duplicate entries will arise.
                if (node_0 < n) or (node_1 < n):
                    tsslltd_core_pb_edges.append((node_0, node_1))
                else: pass

    del vertex, vertices, ridge_vertex, ridge_vertices, tsslltd_core_voronoi
    del tsslltd_core_x, tsslltd_core_y

    # Convert edge list to np.ndarray, and retain the unique edges from
    # the core and periodic boundary nodes
    tsslltd_core_pb_edges = unique_sorted_edges(tsslltd_core_pb_edges)

    # Lists for the edges of the graph capturing the periodic
    # connections between the core nodes
    conn_core_edges = []
    conn_pb_edges = []

    for edge in range(np.shape(tsslltd_core_pb_edges)[0]):
        node_0 = int(tsslltd_core_pb_edges[edge, 0])
        node_1 = int(tsslltd_core_pb_edges[edge, 1])

        # Edge is a core edge
        if (node_0 < n) and (node_1 < n):
            conn_core_edges.append((node_0, node_1))
        # Edge is a periodic boundary edge
        else:
            node_0 = int(pb2core_nodes[node_0])
            node_1 = int(pb2core_nodes[node_1])
            conn_pb_edges.append((node_0, node_1))

    # Convert edge lists to np.ndarrays, and retain unique edges
    conn_core_edges = unique_sorted_edges(conn_core_edges)
    conn_pb_edges = unique_sorted_edges(conn_pb_edges)

    # Save fundamental graph constituents from this topology
    np.savetxt(conn_n_filename, [n], fmt="%d")
    np.savetxt(conn_core_edges_filename, conn_core_edges, fmt="%d")
    np.savetxt(conn_pb_edges_filename, conn_pb_edges, fmt="%d")

    # Save the core node x- and y-coordinates
    np.savetxt(core_x_filename, core_x)
    np.savetxt(core_y_filename, core_y)

def voronoi_dim_3_network_topology_initialization(
        L: float,
        core_x: np.ndarray,
        core_y: np.ndarray,
        core_z: np.ndarray,
        filename_prefix: str) -> None:
    """Network topology initialization procedure for three-dimensional
    Voronoi-tessellated networks.

    This function ``tessellates'' the core nodes about themselves,
    applies Voronoi tessellation to the resulting tessellated network
    via the scipy.spatial.Voronoi() function, acquires back the
    periodic network topology of the core nodes, and ascertains
    fundamental graph constituents (node and edge information) from this
    topology.

    Args:
        L (float): Simulation box size.
        core_x (np.ndarray): x-coordinates of the input core nodes.
        core_y (np.ndarray): y-coordinates of the input core nodes.
        core_z (np.ndarray): z-coordinates of the input core nodes.
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Import the scipy.spatial.Voronoi() function
    from scipy.spatial import Voronoi
    
    # Generate filenames
    input_core_x_filename = filename_prefix + "-input_core_x" + ".dat"
    input_core_y_filename = filename_prefix + "-input_core_y" + ".dat"
    input_core_z_filename = filename_prefix + "-input_core_z" + ".dat"
    conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"

    # Save the input core node x-, y-, and z-coordinates
    np.savetxt(input_core_x_filename, core_x)
    np.savetxt(input_core_y_filename, core_y)
    np.savetxt(input_core_z_filename, core_z)

    # Copy the core_x, core_y, and core_z np.ndarrays as the first n
    # entries in the tessellated node x-, y-, and z-coordinate
    # np.ndarrays
    tsslltd_core_x = core_x.copy()
    tsslltd_core_y = core_y.copy()
    tsslltd_core_z = core_z.copy()

    # Three-dimensional tessellation protocol
    dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)

    # Use three-dimensional tessellation protocol to tessellate the core
    # nodes
    for tsslltn in range(dim_3_tsslltn_num):
        x_tsslltn = dim_3_tsslltn[tsslltn, 0]
        y_tsslltn = dim_3_tsslltn[tsslltn, 1]
        z_tsslltn = dim_3_tsslltn[tsslltn, 2]
        # Skip the (hold, hold, hold) tessellation call because the core
        # nodes are being tessellated about themselves
        if (x_tsslltn == 0) and (y_tsslltn == 0) and (z_tsslltn == 0): continue
        else:
            # x-, y-, and z-coordinates from three-dimensional
            # tessellation protocol
            core_tsslltn_x, core_tsslltn_y, core_tsslltn_z = dim_3_tessellation(
                L, core_x, core_y, core_z, x_tsslltn, y_tsslltn, z_tsslltn)
            # Concatenate the tessellated x-, y-, and z-coordinates
            tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
            tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
            tsslltd_core_z = np.concatenate((tsslltd_core_z, core_tsslltn_z))

    del core_tsslltn_x, core_tsslltn_y, core_tsslltn_z

    # Stack the tessellated x-, y-, and z-coordinates next to each other
    # columnwise
    tsslltd_core = np.column_stack(
        (tsslltd_core_x, tsslltd_core_y, tsslltd_core_z))

    # Apply Voronoi tessellation
    tsslltd_core_voronoi = Voronoi(tsslltd_core)

    del tsslltd_core, tsslltd_core_x, tsslltd_core_y, tsslltd_core_z

    # Extract vertices from the Voronoi tessellation
    vertices = tsslltd_core_voronoi.vertices

    # Separate vertex x-, y-, and z-coordinates
    vertices_x = vertices[:, 0]
    vertices_y = vertices[:, 1]
    vertices_z = vertices[:, 2]

    # Confirm that each vertex solely occupies a cube neighborhood
    # that is \pm tol about itself
    tol = 1e-10
    for vertex in range(np.shape(vertices)[0]):
        # Extract x-, y-, and z-coordinates of the vertex
        vertex_x = vertices_x[vertex]
        vertex_y = vertices_y[vertex]
        vertex_z = vertices_z[vertex]

        # Determine if the vertex solely occupies a cube neighborhood
        # that is \pm tol about itself
        nghbr_x_lb = vertex_x - tol
        nghbr_x_ub = vertex_x + tol
        nghbr_y_lb = vertex_y - tol
        nghbr_y_ub = vertex_y + tol
        nghbr_z_lb = vertex_z - tol
        nghbr_z_ub = vertex_z + tol
        psbl_nghbr_x_indcs = (
            np.where(np.logical_and(vertices_x>=nghbr_x_lb, vertices_x<=nghbr_x_ub))[0]
        )
        psbl_nghbr_y_indcs = (
            np.where(np.logical_and(vertices_y>=nghbr_y_lb, vertices_y<=nghbr_y_ub))[0]
        )
        psbl_nghbr_z_indcs = (
            np.where(np.logical_and(vertices_z>=nghbr_z_lb, vertices_z<=nghbr_z_ub))[0]
        )
        # Gather the indices from each separate coordinate together to
        # assess all possible vertex neighbors. Retain unique indices
        # corresponding to each possible vertex neighbor, and the number
        # of times each such index value appears.
        psbl_nghbr_indcs, psbl_nghbr_indcs_counts = np.unique(
            np.concatenate((psbl_nghbr_x_indcs, psbl_nghbr_y_indcs, psbl_nghbr_z_indcs), dtype=int),
            return_counts=True)
        # The true vertex neighbors are those whose index value appears
        # thrice in the possible vertex neighbor array -- equal to the
        # network dimensionality
        nghbr_indcs_vals_indcs = np.where(psbl_nghbr_indcs_counts == 3)[0]
        nghbr_num = np.shape(nghbr_indcs_vals_indcs)[0]
        # If nghbr_num > 1, then the neighborhood is overpopulated, and
        # therefore this is an invalid set of input node coordinates for
        # Voronoi tessellation
        if nghbr_num > 1:
            error_str = (
                "A vertex neighborhood has more than one vertex! "
                + "Therefore, this is an invalid set of input node "
                + "coordinates for Voronoi tessellation."
            )
            sys.exit(error_str)
    
    # Extract core vertices and label these as core nodes
    psbl_core_x_indcs = np.where(np.logical_and(vertices_x>=0, vertices_x<L))[0]
    psbl_core_y_indcs = np.where(np.logical_and(vertices_y>=0, vertices_y<L))[0]
    psbl_core_z_indcs = np.where(np.logical_and(vertices_z>=0, vertices_z<L))[0]
    # Gather the indices from each separate coordinate together to
    # assess all possible core vertices. Retain unique indices
    # corresponding to each possible core vertex, and the number of
    # times each such index value appears.
    psbl_core_indcs, psbl_core_indcs_counts = np.unique(
        np.concatenate((psbl_core_x_indcs, psbl_core_y_indcs, psbl_core_z_indcs), dtype=int),
        return_counts=True)
    # The true core vertices are those whose index value appears thrice
    # in the possible core vertex array -- equal to the network
    # dimensionality
    core_indcs = psbl_core_indcs[np.where(psbl_core_indcs_counts == 3)[0]]

    # Number of core vertices
    n = np.shape(core_indcs)[0]

    # Gather core node x-, y-, and z-coordinates
    core_x = vertices_x[core_indcs]
    core_y = vertices_y[core_indcs]
    core_z = vertices_z[core_indcs]

    # Copy the core_x, core_y, and core_z np.ndarrays as the first n
    # entries in the tessellated node x-, y-, and z-coordinate
    # np.ndarrays
    tsslltd_core_x = core_x.copy()
    tsslltd_core_y = core_y.copy()
    tsslltd_core_z = core_z.copy()

    # Core nodes
    core_nodes = np.arange(n, dtype=int)

    # Use three-dimensional tessellation protocol to tessellate the core
    # nodes
    for tsslltn in range(dim_3_tsslltn_num):
        x_tsslltn = dim_3_tsslltn[tsslltn, 0]
        y_tsslltn = dim_3_tsslltn[tsslltn, 1]
        z_tsslltn = dim_3_tsslltn[tsslltn, 2]
        # Skip the (hold, hold, hold) tessellation call because the core
        # nodes are being tessellated about themselves
        if (x_tsslltn == 0) and (y_tsslltn == 0) and (z_tsslltn == 0): continue
        else:
            # x-, y-, and z-coordinates from three-dimensional
            # tessellation protocol
            core_tsslltn_x, core_tsslltn_y, core_tsslltn_z = dim_3_tessellation(
                L, core_x, core_y, core_z, x_tsslltn, y_tsslltn, z_tsslltn)
            # Concatenate the tessellated x-, y-, and z-coordinates
            tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
            tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
            tsslltd_core_z = np.concatenate((tsslltd_core_z, core_tsslltn_z))

    del core_tsslltn_x, core_tsslltn_y, core_tsslltn_z

    # Construct the pb2core_nodes np.ndarray such that
    # pb2core_nodes[core_pb_node] = core_node
    pb2core_nodes = np.tile(core_nodes, dim_3_tsslltn_num)
        
    del core_nodes

    # Extract the ridge vertices from the Voronoi tessellation
    ridge_vertices = tsslltd_core_voronoi.ridge_vertices

    # List for edges of the core and periodic boundary nodes
    tsslltd_core_pb_edges = []

    for ridge_vertex in ridge_vertices:
        # In three dimensions, each ridge vertex is a facet
        # Extract list of lines that altogether define the facet
        facet_vertices_0 = ridge_vertex[:-1] + [ridge_vertex[-1]]
        facet_vertices_1 = ridge_vertex[1:] + [ridge_vertex[0]]
        for facet_vertex in range(len(ridge_vertex)):
            vertex_0 = int(facet_vertices_0[facet_vertex])
            vertex_1 = int(facet_vertices_1[facet_vertex])
            # Skip over ridge vertices that extend out to infinity
            if (vertex_0 == -1) or (vertex_1 == -1): continue
            else:
                # Determine the indices of the vertices in the
                # tessellated core node topology 
                # Extract vertex x-, y-, and z-coordinates
                vertex_0_x = vertices_x[vertex_0]
                vertex_0_y = vertices_y[vertex_0]
                vertex_0_z = vertices_z[vertex_0]
                vertex_1_x = vertices_x[vertex_1]
                vertex_1_y = vertices_y[vertex_1]
                vertex_1_z = vertices_z[vertex_1]
                # Define vertex neighborhood
                nghbr_vertex_0_x_lb = vertex_0_x - tol
                nghbr_vertex_0_x_ub = vertex_0_x + tol
                nghbr_vertex_0_y_lb = vertex_0_y - tol
                nghbr_vertex_0_y_ub = vertex_0_y + tol
                nghbr_vertex_0_z_lb = vertex_0_z - tol
                nghbr_vertex_0_z_ub = vertex_0_z + tol
                nghbr_vertex_1_x_lb = vertex_1_x - tol
                nghbr_vertex_1_x_ub = vertex_1_x + tol
                nghbr_vertex_1_y_lb = vertex_1_y - tol
                nghbr_vertex_1_y_ub = vertex_1_y + tol
                nghbr_vertex_1_z_lb = vertex_1_z - tol
                nghbr_vertex_1_z_ub = vertex_1_z + tol
                psbl_vertex_0_x_indcs = (
                    np.where(np.logical_and(tsslltd_core_x>=nghbr_vertex_0_x_lb, tsslltd_core_x<=nghbr_vertex_0_x_ub))[0]
                )
                psbl_vertex_0_y_indcs = (
                    np.where(np.logical_and(tsslltd_core_y>=nghbr_vertex_0_y_lb, tsslltd_core_y<=nghbr_vertex_0_y_ub))[0]
                )
                psbl_vertex_0_z_indcs = (
                    np.where(np.logical_and(tsslltd_core_z>=nghbr_vertex_0_z_lb, tsslltd_core_z<=nghbr_vertex_0_z_ub))[0]
                )
                psbl_vertex_1_x_indcs = (
                    np.where(np.logical_and(tsslltd_core_x>=nghbr_vertex_1_x_lb, tsslltd_core_x<=nghbr_vertex_1_x_ub))[0]
                )
                psbl_vertex_1_y_indcs = (
                    np.where(np.logical_and(tsslltd_core_y>=nghbr_vertex_1_y_lb, tsslltd_core_y<=nghbr_vertex_1_y_ub))[0]
                )
                psbl_vertex_1_z_indcs = (
                    np.where(np.logical_and(tsslltd_core_z>=nghbr_vertex_1_z_lb, tsslltd_core_z<=nghbr_vertex_1_z_ub))[0]
                )
                # Gather the indices from each separate coordinate
                # together, retain unique indices, and the number of
                # times each such index value appears.
                psbl_vertex_0_indcs, psbl_vertex_0_indcs_counts = np.unique(
                    np.concatenate((psbl_vertex_0_x_indcs, psbl_vertex_0_y_indcs, psbl_vertex_0_z_indcs), dtype=int),
                    return_counts=True)
                psbl_vertex_1_indcs, psbl_vertex_1_indcs_counts = np.unique(
                    np.concatenate((psbl_vertex_1_x_indcs, psbl_vertex_1_y_indcs, psbl_vertex_1_z_indcs), dtype=int),
                    return_counts=True)
                # The true vertex is whose index value appears thrice in
                # the possible vertex array -- equal to the network
                # dimensionality
                vertex_0_indcs_vals_indcs = (
                    np.where(psbl_vertex_0_indcs_counts == 3)[0]
                )
                vertex_0_indcs_num = np.shape(vertex_0_indcs_vals_indcs)[0]
                vertex_1_indcs_vals_indcs = (
                    np.where(psbl_vertex_1_indcs_counts == 3)[0]
                )
                vertex_1_indcs_num = np.shape(vertex_1_indcs_vals_indcs)[0]
                # Skip over situations where a vertex value is not able
                # to be solved for, which does not occur for the core
                # and periodic boundary vertices
                if (vertex_0_indcs_num != 1) or (vertex_1_indcs_num != 1):
                    continue
                else:
                    # Extract vertex node index
                    node_0 = int(
                        psbl_vertex_0_indcs[vertex_0_indcs_vals_indcs][0])
                    node_1 = int(
                        psbl_vertex_1_indcs[vertex_1_indcs_vals_indcs][0])

                    # If any of the nodes involved in the ridge vertex
                    # correspond to the original core nodes, then add
                    # that edge to the edge list. Duplicate entries will
                    # arise.
                    if (node_0 < n) or (node_1 < n):
                        tsslltd_core_pb_edges.append((node_0, node_1))
                    else: pass

    del vertex, vertices, ridge_vertex, ridge_vertices, tsslltd_core_voronoi
    del tsslltd_core_x, tsslltd_core_y, tsslltd_core_z

    # Convert edge list to np.ndarray, and retain the unique edges from
    # the core and periodic boundary nodes
    tsslltd_core_pb_edges = unique_sorted_edges(tsslltd_core_pb_edges)

    # Lists for the edges of the graph capturing the periodic
    # connections between the core nodes
    conn_core_edges = []
    conn_pb_edges = []

    for edge in range(np.shape(tsslltd_core_pb_edges)[0]):
        node_0 = int(tsslltd_core_pb_edges[edge, 0])
        node_1 = int(tsslltd_core_pb_edges[edge, 1])

        # Edge is a core edge
        if (node_0 < n) and (node_1 < n):
            conn_core_edges.append((node_0, node_1))
        # Edge is a periodic boundary edge
        else:
            node_0 = int(pb2core_nodes[node_0])
            node_1 = int(pb2core_nodes[node_1])
            conn_pb_edges.append((node_0, node_1))

    # Convert edge lists to np.ndarrays, and retain unique edges
    conn_core_edges = unique_sorted_edges(conn_core_edges)
    conn_pb_edges = unique_sorted_edges(conn_pb_edges)

    # Save fundamental graph constituents from this topology
    np.savetxt(conn_n_filename, [n], fmt="%d")
    np.savetxt(conn_core_edges_filename, conn_core_edges, fmt="%d")
    np.savetxt(conn_pb_edges_filename, conn_pb_edges, fmt="%d")

    # Save the core node x-, y-, and z-coordinates
    np.savetxt(core_x_filename, core_x)
    np.savetxt(core_y_filename, core_y)
    np.savetxt(core_z_filename, core_z)

def voronoi_network_topology_initialization(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Network topology initialization procedure for Voronoi-tessellated
    networks.

    This function loads the simulation box size and the core node
    coordinates previously generated by the node_seeding() function.
    Then, depending on the network dimensionality, this function calls
    upon a corresponding helper function to initialize the
    Voronoi-tessellated network topology.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "voronoi" is applicable (corresponding
        to Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular
        scheme used to generate the positions of the core nodes;
        either "random", "prhd", "pdhu", or "lammps" (corresponding to
        the random node placement procedure ("random"), periodic random
        hard disk node placement procedure ("prhd"), periodic disordered
        hyperuniform node placement procedure ("pdhu"), or nodes
        randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Network topology initialization procedure is only applicable for 
    # Voronoi-tessellated networks. Exit if a different type of network
    # is passed.
    if network != "voronoi":
        error_str = (
            "Network topology initialization procedure is only "
            + "applicable for Voronoi-tessellated networks. This procedure "
            + "will only proceed if network = ``voronoi''."
        )
        sys.exit(error_str)
    # Generate filename prefix
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    
    # Load L
    L = np.loadtxt(L_filename)

    # Append configuration number to filename prefix
    filename_prefix = filename_prefix + f"C{config:d}"

    # Generate config filename
    config_filename = filename_prefix + ".config"

    # Call appropriate helper function to initialize network topology
    if (scheme == "random") or (scheme == "prhd") or (scheme == "pdhu"):
        # Load core node coordinates
        coords = np.loadtxt(config_filename)
    elif scheme == "lammps":
        skiprows_num = 15
        # Load core node coordinates
        coords = np.loadtxt(config_filename, skiprows=skiprows_num, max_rows=n)
    # Separate core nodes x- and y-coordinates
    x = coords[:, 0].copy()
    y = coords[:, 1].copy()
    if dim == 2:
        del coords
        voronoi_dim_2_network_topology_initialization(
            L, x, y, filename_prefix)
    elif dim == 3:
        # Separate core node z-coordinates
        z = coords[:, 2].copy()
        del coords
        voronoi_dim_3_network_topology_initialization(
            L, x, y, z, filename_prefix)

def delaunay_dim_2_network_topology_initialization(
        L: float,
        core_x: np.ndarray,
        core_y: np.ndarray,
        n: int,
        filename_prefix: str) -> None:
    """Network topology initialization procedure for two-dimensional
    Delaunay-triangulated networks.

    This function ``tessellates'' the core nodes about themselves,
    applies Delaunay triangulation to the resulting tessellated network
    via the scipy.spatial.Delaunay() function, acquires back the
    periodic network topology of the core nodes, and ascertains
    fundamental graph constituents (node and edge information) from this
    topology.

    Args:
        L (float): Simulation box size.
        core_x (np.ndarray): x-coordinates of the core nodes.
        core_y (np.ndarray): y-coordinates of the core nodes.
        n (int): Number of core nodes.
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Import the scipy.spatial.Delaunay() function
    from scipy.spatial import Delaunay
    
    # Generate filenames
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"

    # Core nodes
    core_nodes = np.arange(n, dtype=int)

    # Copy the core_x and core_y np.ndarrays as the first n entries in
    # the tessellated node x- and y-coordinate np.ndarrays
    tsslltd_core_x = core_x.copy()
    tsslltd_core_y = core_y.copy()

    # Two-dimensional tessellation protocol
    dim_2_tsslltn, dim_2_tsslltn_num = tessellation_protocol(2)
    
    # Use two-dimensional tessellation protocol to tessellate the core
    # nodes
    for tsslltn in range(dim_2_tsslltn_num):
        x_tsslltn = dim_2_tsslltn[tsslltn, 0]
        y_tsslltn = dim_2_tsslltn[tsslltn, 1]
        # Skip the (hold, hold) tessellation call because the core
        # nodes are being tessellated about themselves
        if (x_tsslltn == 0) and (y_tsslltn == 0): continue
        else:
            # x- and y-coordinates from two-dimensional tessellation
            # protocol
            core_tsslltn_x, core_tsslltn_y = dim_2_tessellation(
                L, core_x, core_y, x_tsslltn, y_tsslltn)
            # Concatenate the tessellated x- and y-coordinates
            tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
            tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
    
    del core_tsslltn_x, core_tsslltn_y
    
    # Construct the pb2core_nodes np.ndarray such that
    # pb2core_nodes[core_pb_node] = core_node
    pb2core_nodes = np.tile(core_nodes, dim_2_tsslltn_num)
    
    del core_nodes

    # Stack the tessellated x- and y-coordinates next to each other
    # columnwise
    tsslltd_core = np.column_stack((tsslltd_core_x, tsslltd_core_y))

    # Apply Delaunay triangulation
    tsslltd_core_delaunay = Delaunay(tsslltd_core)

    del tsslltd_core, tsslltd_core_x, tsslltd_core_y

    # Extract the simplices from the Delaunay triangulation
    simplices = tsslltd_core_delaunay.simplices

    # List for edges of the core and periodic boundary nodes
    tsslltd_core_pb_edges = []

    for simplex in simplices:
        # In two-dimensions, each simplex is a triangle
        node_0 = int(simplex[0])
        node_1 = int(simplex[1])
        node_2 = int(simplex[2])

        # If any of the nodes involved in any simplex edge correspond to
        # the original core nodes, then add that edge to the edge list.
        # Duplicate entries will arise.
        if (node_0 < n) or (node_1 < n):
            tsslltd_core_pb_edges.append((node_0, node_1))
        if (node_1 < n) or (node_2 < n):
            tsslltd_core_pb_edges.append((node_1, node_2))
        if (node_2 < n) or (node_0 < n):
            tsslltd_core_pb_edges.append((node_2, node_0))
        else: pass
    
    del simplex, simplices, tsslltd_core_delaunay

    # Convert edge list to np.ndarray, and retain the unique edges from
    # the core and periodic boundary nodes
    tsslltd_core_pb_edges = unique_sorted_edges(tsslltd_core_pb_edges)

    # Lists for the edges of the graph capturing the periodic
    # connections between the core nodes
    conn_core_edges = []
    conn_pb_edges = []

    for edge in range(np.shape(tsslltd_core_pb_edges)[0]):
        node_0 = int(tsslltd_core_pb_edges[edge, 0])
        node_1 = int(tsslltd_core_pb_edges[edge, 1])

        # Edge is a core edge
        if (node_0 < n) and (node_1 < n):
            conn_core_edges.append((node_0, node_1))
        # Edge is a periodic boundary edge
        else:
            node_0 = int(pb2core_nodes[node_0])
            node_1 = int(pb2core_nodes[node_1])
            conn_pb_edges.append((node_0, node_1))
    
    # Convert edge lists to np.ndarrays, and retain unique edges
    conn_core_edges = unique_sorted_edges(conn_core_edges)
    conn_pb_edges = unique_sorted_edges(conn_pb_edges)

    # Save fundamental graph constituents from this topology
    np.savetxt(conn_core_edges_filename, conn_core_edges, fmt="%d")
    np.savetxt(conn_pb_edges_filename, conn_pb_edges, fmt="%d")

    # Save the core node x- and y-coordinates
    np.savetxt(core_x_filename, core_x)
    np.savetxt(core_y_filename, core_y)

def delaunay_dim_3_network_topology_initialization(
        L: float,
        core_x: np.ndarray,
        core_y: np.ndarray,
        core_z: np.ndarray,
        n: int,
        filename_prefix: str) -> None:
    """Network topology initialization procedure for three-dimensional
    Delaunay-triangulated networks.

    This function ``tessellates'' the core nodes about themselves,
    applies Delaunay triangulation to the resulting tessellated network
    via the scipy.spatial.Delaunay() function, acquires back the
    periodic network topology of the core nodes, and ascertains
    fundamental graph constituents (node and edge information) from this
    topology.

    Args:
        L (float): Simulation box size.
        core_x (np.ndarray): x-coordinates of the core nodes.
        core_y (np.ndarray): y-coordinates of the core nodes.
        core_z (np.ndarray): z-coordinates of the core nodes.
        n (int): Number of core nodes.
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Import the scipy.spatial.Delaunay() function
    from scipy.spatial import Delaunay
    
    # Generate filenames
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"

    # Core nodes
    core_nodes = np.arange(n, dtype=int)

    # Copy the core_x, core_y, and core_z np.ndarrays as the first n
    # entries in the tessellated node x-, y-, and z-coordinate
    # np.ndarrays
    tsslltd_core_x = core_x.copy()
    tsslltd_core_y = core_y.copy()
    tsslltd_core_z = core_z.copy()

    # Three-dimensional tessellation protocol
    dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)
    
    # Use three-dimensional tessellation protocol to tessellate the core
    # nodes
    for tsslltn in range(dim_3_tsslltn_num):
        x_tsslltn = dim_3_tsslltn[tsslltn, 0]
        y_tsslltn = dim_3_tsslltn[tsslltn, 1]
        z_tsslltn = dim_3_tsslltn[tsslltn, 2]
        # Skip the (hold, hold, hold) tessellation call because the core
        # nodes are being tessellated about themselves
        if (x_tsslltn == 0) and (y_tsslltn == 0) and (z_tsslltn == 0): continue
        else:
            # x-, y-, and z-coordinates from three-dimensional
            # tessellation protocol
            core_tsslltn_x, core_tsslltn_y, core_tsslltn_z = dim_3_tessellation(
                L, core_x, core_y, core_z, x_tsslltn, y_tsslltn, z_tsslltn)
            # Concatenate the tessellated x-, y-, and z-coordinates
            tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
            tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
            tsslltd_core_z = np.concatenate((tsslltd_core_z, core_tsslltn_z))
    
    del core_tsslltn_x, core_tsslltn_y, core_tsslltn_z
    
    # Construct the pb2core_nodes np.ndarray such that
    # pb2core_nodes[core_pb_node] = core_node
    pb2core_nodes = np.tile(core_nodes, dim_3_tsslltn_num)
    
    del core_nodes

    # Stack the tessellated x-, y-, and z-coordinates next to each other
    # columnwise
    tsslltd_core = np.column_stack(
        (tsslltd_core_x, tsslltd_core_y, tsslltd_core_z))

    # Apply Delaunay triangulation
    tsslltd_core_delaunay = Delaunay(tsslltd_core)

    del tsslltd_core, tsslltd_core_x, tsslltd_core_y, tsslltd_core_z

    # Extract the simplices from the Delaunay triangulation
    simplices = tsslltd_core_delaunay.simplices

    # List for edges of the core and periodic boundary nodes
    tsslltd_core_pb_edges = []

    for simplex in simplices:
        # In three-dimensions, each simplex is a tetrahedron
        node_0 = int(simplex[0])
        node_1 = int(simplex[1])
        node_2 = int(simplex[2])
        node_3 = int(simplex[3])

        # If any of the nodes involved in any simplex edge correspond to
        # the original core nodes, then add those nodes and that edge to
        # the appropriate lists. Duplicate entries will arise.
        if (node_0 < n) or (node_1 < n):
            tsslltd_core_pb_edges.append((node_0, node_1))
        if (node_1 < n) or (node_2 < n):
            tsslltd_core_pb_edges.append((node_1, node_2))
        if (node_2 < n) or (node_0 < n):
            tsslltd_core_pb_edges.append((node_2, node_0))
        if (node_3 < n) or (node_0 < n):
            tsslltd_core_pb_edges.append((node_3, node_0))
        if (node_3 < n) or (node_1 < n):
            tsslltd_core_pb_edges.append((node_3, node_1))
        if (node_3 < n) or (node_2 < n):
            tsslltd_core_pb_edges.append((node_3, node_2))
        else: pass
    
    del simplex, simplices, tsslltd_core_delaunay

    # Convert edge list to np.ndarray, and retain the unique edges from
    # the core and periodic boundary nodes
    tsslltd_core_pb_edges = unique_sorted_edges(tsslltd_core_pb_edges)

    # Lists for the edges of the graph capturing the periodic
    # connections between the core nodes
    conn_core_edges = []
    conn_pb_edges = []

    for edge in range(np.shape(tsslltd_core_pb_edges)[0]):
        node_0 = int(tsslltd_core_pb_edges[edge, 0])
        node_1 = int(tsslltd_core_pb_edges[edge, 1])

        # Edge is a core edge
        if (node_0 < n) and (node_1 < n):
            conn_core_edges.append((node_0, node_1))
        # Edge is a periodic boundary edge
        else:
            node_0 = int(pb2core_nodes[node_0])
            node_1 = int(pb2core_nodes[node_1])
            conn_pb_edges.append((node_0, node_1))
    
    # Convert edge lists to np.ndarrays, and retain unique edges
    conn_core_edges = unique_sorted_edges(conn_core_edges)
    conn_pb_edges = unique_sorted_edges(conn_pb_edges)

    # Save fundamental graph constituents from this topology
    np.savetxt(conn_core_edges_filename, conn_core_edges, fmt="%d")
    np.savetxt(conn_pb_edges_filename, conn_pb_edges, fmt="%d")

    # Save the core node x-, y-, and z-coordinates
    np.savetxt(core_x_filename, core_x)
    np.savetxt(core_y_filename, core_y)
    np.savetxt(core_z_filename, core_z)

def delaunay_network_topology_initialization(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Network topology initialization procedure for
    Delaunay-triangulated networks.

    This function loads the simulation box size and the core node
    coordinates previously generated by the node_seeding() function.
    Then, depending on the network dimensionality, this function calls
    upon a corresponding helper function to initialize the
    Delaunay-triangulated network topology.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "delaunay" is applicable (corresponding
        to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular
        scheme used to generate the positions of the core nodes;
        either "random", "prhd", "pdhu", or "lammps" (corresponding to
        the random node placement procedure ("random"), periodic random
        hard disk node placement procedure ("prhd"), periodic disordered
        hyperuniform node placement procedure ("pdhu"), or nodes
        randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Network topology initialization procedure is only applicable for 
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "delaunay":
        error_str = (
            "Network topology initialization procedure is only "
            + "applicable for Delaunay-triangulated networks. This "
            + "procedure will only proceed if network = ``delaunay''."
        )
        sys.exit(error_str)
    # Generate filename prefix
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    
    # Load L
    L = np.loadtxt(L_filename)

    # Append configuration number to filename prefix
    filename_prefix = filename_prefix + f"C{config:d}"

    # Generate config filename
    config_filename = filename_prefix + ".config"

    # Call appropriate helper function to initialize network topology
    if (scheme == "random") or (scheme == "prhd") or (scheme == "pdhu"):
        # Load core node coordinates
        coords = np.loadtxt(config_filename)
    elif scheme == "lammps":
        skiprows_num = 15
        # Load core node coordinates
        coords = np.loadtxt(config_filename, skiprows=skiprows_num, max_rows=n)
    # Actual number of core nodes
    n = np.shape(coords)[0]
    # Separate core nodes x- and y-coordinates
    x = coords[:, 0].copy()
    y = coords[:, 1].copy()
    if dim == 2:
        del coords
        delaunay_dim_2_network_topology_initialization(
            L, x, y, n, filename_prefix)
    elif dim == 3:
        # Separate core node z-coordinates
        z = coords[:, 2].copy()
        del coords
        delaunay_dim_3_network_topology_initialization(
            L, x, y, z, n, filename_prefix)

def swidt_network_topology_initialization(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Network topology initialization procedure for spider web-inspired
    Delaunay-triangulated networks.

    This function loads the simulation box size and the core node
    coordinates previously generated by the node_seeding() function.
    Then, depending on the network dimensionality, this function calls
    upon a corresponding helper function to initialize the spider
    web-inspired Delaunay-triangulated network topology.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "swidt" is applicable (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular
        scheme used to generate the positions of the core nodes;
        either "random", "prhd", "pdhu", or "lammps" (corresponding to
        the random node placement procedure ("random"), periodic random
        hard disk node placement procedure ("prhd"), periodic disordered
        hyperuniform node placement procedure ("pdhu"), or nodes
        randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Network topology initialization procedure is only applicable for 
    # spider web-inspired Delaunay-triangulated networks. Exit if a
    # different type of network is passed.
    if network != "swidt":
        error_str = (
            "Network topology initialization procedure is only "
            + "applicable for spider web-inspired "
            + "Delaunay-triangulated networks. This procedure will "
            + "only proceed if network = ``swidt''."
        )
        sys.exit(error_str)
    # Generate filename prefix
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    
    # Load L
    L = np.loadtxt(L_filename)

    # Append configuration number to filename prefix
    filename_prefix = filename_prefix + f"C{config:d}"

    # Generate config filename
    config_filename = filename_prefix + ".config"

    # Call appropriate helper function to initialize network topology
    if (scheme == "random") or (scheme == "prhd") or (scheme == "pdhu"):
        # Load core node coordinates
        coords = np.loadtxt(config_filename)
    elif scheme == "lammps":
        skiprows_num = 15
        # Load core node coordinates
        coords = np.loadtxt(config_filename, skiprows=skiprows_num, max_rows=n)
    # Actual number of core nodes
    n = np.shape(coords)[0]
    # Separate core nodes x- and y-coordinates
    x = coords[:, 0].copy()
    y = coords[:, 1].copy()
    if dim == 2:
        del coords
        delaunay_dim_2_network_topology_initialization(
            L, x, y, n, filename_prefix)
    elif dim == 3:
        # Separate core node z-coordinates
        z = coords[:, 2].copy()
        del coords
        delaunay_dim_3_network_topology_initialization(
            L, x, y, z, n, filename_prefix)

def add_nodes_from_numpy_array(graph, nodes: np.ndarray):
    """Add node numbers from a np.ndarray array to an undirected
    NetworkX graph.

    This function adds node numbers from a np.ndarray array to an
    undirected NetworkX graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph
        or nx.MultiGraph.
        nodes (np.ndarray): Node numbers
    
    Returns:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph
        or nx.MultiGraph.
    
    """
    graph.add_nodes_from(nodes.tolist())
    return graph

def add_edges_from_numpy_array(graph, edges: np.ndarray):
    """Add edges from a two-dimensional np.ndarray to an undirected
    NetworkX graph.

    This function adds edges from a two-dimensional np.ndarray to an
    undirected NetworkX graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph
        or nx.MultiGraph.
        edges (np.ndarray): np.ndarray of edges
    
    Returns:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph
        or nx.MultiGraph.
    
    """
    graph.add_edges_from(list(tuple(edge) for edge in edges.tolist()))
    return graph

def swidt_network_edge_pruning_procedure(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        n: int,
        k: int,
        config: int,
        pruning: int) -> None:
    """Edge pruning procedure for the initialized topology of spider
    web-inspired Delaunay-triangulated networks.

    This function loads fundamental graph constituents along with core
    node coordinates, performs a random edge pruning procedure such that
    each node in the network is connected to, at most, k edges, and
    isolates the maximum connected component from the resulting network.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "swidt" is applicable (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        k (int): Maximum node degree/functionality; either 3, 4,
        5, 6, 7, or 8.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
    
    """
    # Edge pruning procedure is only applicable for the initialized
    # topology of spider web-inspired Delaunay-triangulated networks.
    # Exit if a different type of network is passed.
    if network != "swidt":
        error_str = (
            "Edge pruning procedure is only applicable for the "
            + "initialized topology of spider web-inspired "
            + "Delaunay-triangulated networks. This procedure will "
            + "only proceed if network = ``swidt''."
        )
        sys.exit(error_str)
    # Initialize random number generator
    rng = np.random.default_rng()

    # Initialize node number integer constants
    core_node_0 = 0
    core_node_1 = 0

    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample) + f"C{config:d}"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    if dim == 3:
        core_z_filename = filename_prefix + "-core_z" + ".dat"
    filename_prefix = filename_prefix + f"P{pruning:d}"
    mx_cmp_pruned_conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    mx_cmp_pruned_conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    mx_cmp_pruned_conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    mx_cmp_pruned_core_x_filename = filename_prefix + "-core_x" + ".dat"
    mx_cmp_pruned_core_y_filename = filename_prefix + "-core_y" + ".dat"
    if dim == 3:
        mx_cmp_pruned_core_z_filename = filename_prefix + "-core_z" + ".dat"
    
    # Load fundamental graph constituents
    core_nodes = np.arange(n, dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Load core node x- and y-coordinates
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    if dim == 3:
        # Load core node z-coordinates
        core_z = np.loadtxt(core_z_filename)
    
    # Create nx.Graphs, and add nodes before edges
    conn_core_graph = nx.Graph()
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_core_graph = add_edges_from_numpy_array(conn_core_graph, conn_core_edges)

    conn_pb_graph = nx.Graph()
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)

    conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Degree of nodes in the graph
    conn_graph_k = np.asarray(list(conn_graph.degree()), dtype=int)[:, 1]

    if np.any(conn_graph_k > k):
        # Explicit edge pruning procedure
        while np.any(conn_graph_k > k):
            # Identify the nodes connected to more than k edges in the
            # graph, i.e., hyperconnected nodes
            conn_graph_hyprconn_nodes = np.where(conn_graph_k > k)[0]
            # Identify the edges connected to the hyperconnected nodes
            conn_graph_hyprconn_edge_indcs_0 = (
                np.where(np.isin(conn_edges[:, 0], conn_graph_hyprconn_nodes))[0]
            )
            conn_graph_hyprconn_edge_indcs_1 = (
                np.where(np.isin(conn_edges[:, 1], conn_graph_hyprconn_nodes))[0]
            )
            conn_graph_hyprconn_edge_indcs = (
                np.unique(
                    np.concatenate(
                        (conn_graph_hyprconn_edge_indcs_0, conn_graph_hyprconn_edge_indcs_1),
                        dtype=int))
            )
            # Randomly select a hyperconnected edge to remove
            edge_indcs_indx2remove_indx = (
                rng.integers(
                    np.shape(conn_graph_hyprconn_edge_indcs)[0], dtype=int)
            )
            edge_indx2remove = (
                conn_graph_hyprconn_edge_indcs[edge_indcs_indx2remove_indx]
            )
            core_node_0 = int(conn_edges[edge_indx2remove, 0])
            core_node_1 = int(conn_edges[edge_indx2remove, 1])

            # Remove hyperconnected edge in the graphs
            conn_graph.remove_edge(core_node_0, core_node_1)
            conn_edges = np.delete(conn_edges, edge_indx2remove, axis=0)
            if conn_core_graph.has_edge(core_node_0, core_node_1):
                conn_core_graph.remove_edge(core_node_0, core_node_1)
            elif conn_pb_graph.has_edge(core_node_0, core_node_1):
                conn_pb_graph.remove_edge(core_node_0, core_node_1)

            # Update degree of nodes in the graph
            conn_graph_k[core_node_0] -= 1
            conn_graph_k[core_node_1] -= 1
                
        # Isolate largest/maximum connected component in a nodewise
        # fashion
        mx_cmp_pruned_conn_graph_nodes = max(
            nx.connected_components(conn_graph), key=len)
        mx_cmp_pruned_conn_core_graph = (
            conn_core_graph.subgraph(mx_cmp_pruned_conn_graph_nodes).copy()
        )
        mx_cmp_pruned_conn_pb_graph = (
            conn_pb_graph.subgraph(mx_cmp_pruned_conn_graph_nodes).copy()
        )
        mx_cmp_pruned_conn_core_graph_edges = np.asarray(
            list(mx_cmp_pruned_conn_core_graph.edges()), dtype=int)
        mx_cmp_pruned_conn_pb_graph_edges = np.asarray(
            list(mx_cmp_pruned_conn_pb_graph.edges()), dtype=int)
        # Number of edges in the largest/maximum connected component
        mx_cmp_pruned_conn_core_graph_m = (
            np.shape(mx_cmp_pruned_conn_core_graph_edges)[0]
        )
        mx_cmp_pruned_conn_pb_graph_m = (
            np.shape(mx_cmp_pruned_conn_pb_graph_edges)[0]
        )
        # Nodes from the largest/maximum connected component, sorted in
        # ascending order
        mx_cmp_pruned_conn_graph_nodes = (
            np.sort(np.fromiter(mx_cmp_pruned_conn_graph_nodes, dtype=int))
        )
        # Number of nodes in the largest/maximum connected component
        mx_cmp_pruned_conn_graph_n = np.shape(mx_cmp_pruned_conn_graph_nodes)[0]

        # Isolate the node coordinates for the largest/maximum connected
        # component
        # updated_node
        mx_cmp_pruned_core_x = core_x[mx_cmp_pruned_conn_graph_nodes]
        mx_cmp_pruned_core_y = core_y[mx_cmp_pruned_conn_graph_nodes]
        if dim == 3:
            mx_cmp_pruned_core_z = core_z[mx_cmp_pruned_conn_graph_nodes]

        # Update all original_node values with updated_node values
        for edge in range(mx_cmp_pruned_conn_core_graph_m):
            # updated_node
            mx_cmp_pruned_conn_core_graph_edges[edge, 0] = (
                int(np.where(mx_cmp_pruned_conn_graph_nodes == mx_cmp_pruned_conn_core_graph_edges[edge, 0])[0][0])
            )
            mx_cmp_pruned_conn_core_graph_edges[edge, 1] = (
                int(np.where(mx_cmp_pruned_conn_graph_nodes == mx_cmp_pruned_conn_core_graph_edges[edge, 1])[0][0])
            )

        for edge in range(mx_cmp_pruned_conn_pb_graph_m):
            # updated_node
            mx_cmp_pruned_conn_pb_graph_edges[edge, 0] = (
                int(np.where(mx_cmp_pruned_conn_graph_nodes == mx_cmp_pruned_conn_pb_graph_edges[edge, 0])[0][0])
            )
            mx_cmp_pruned_conn_pb_graph_edges[edge, 1] = (
                int(np.where(mx_cmp_pruned_conn_graph_nodes == mx_cmp_pruned_conn_pb_graph_edges[edge, 1])[0][0])
            )
                
        # Save fundamental graph constituents from this topology
        np.savetxt(
            mx_cmp_pruned_conn_n_filename,
            [mx_cmp_pruned_conn_graph_n], fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_core_edges_filename,
            mx_cmp_pruned_conn_core_graph_edges, fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_pb_edges_filename,
            mx_cmp_pruned_conn_pb_graph_edges, fmt="%d")
        
        # Save the core node x- and y-coordinates
        np.savetxt(mx_cmp_pruned_core_x_filename, mx_cmp_pruned_core_x)
        np.savetxt(mx_cmp_pruned_core_y_filename, mx_cmp_pruned_core_y)
        if dim == 3:
            # Save the core node z-coordinates
            np.savetxt(mx_cmp_pruned_core_z_filename, mx_cmp_pruned_core_z)
    else:
        # Save fundamental graph constituents from this topology
        np.savetxt(mx_cmp_pruned_conn_n_filename, [n], fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_core_edges_filename, conn_core_edges, fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_pb_edges_filename, conn_pb_edges, fmt="%d")
        
        # Save the core node x- and y-coordinates
        np.savetxt(mx_cmp_pruned_core_x_filename, core_x)
        np.savetxt(mx_cmp_pruned_core_y_filename, core_y)
        if dim == 3:
            # Save the core node z-coordinates
            np.savetxt(mx_cmp_pruned_core_z_filename, core_z)

def core_node_update_func(
        core_node_updt: int,
        core_node_updt_active_site_indx: int,
        core_nodes_active_sites: np.ndarray,
        core_nodes_active_sites_num: int,
        core_nodes_anti_k: np.ndarray,
        core_nodes: np.ndarray,
        core2pb_nodes: list[np.ndarray],
        core_nodes_nghbrhd: list[np.ndarray],
        r_core_nodes_nghbrhd: list[np.ndarray]) -> tuple[np.ndarray, int, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Core node update protocol.

    This function updates the core node active cross-linking sites,
    anti-degree (the number of remaining active cross-linking sites),
    and core node sampling neighborhood list in light of a particular
    core node active cross-linking site becoming inactive.

    Args:
        core_node_updt (int): Core node number corresponding to where an
        active cross-linking site is becoming inactive.
        core_node_updt_active_site_indx (int): Index corresponding to
        the core node active cross-linking site that is becoming
        inactive.
        core_nodes_active_sites (np.ndarray): np.ndarray of the core
        node numbers that have active cross-linking sites.
        core_nodes_active_sites_num (int): Number of core node active
        cross-linking sites.
        core_nodes_anti_k (np.ndarray): np.ndarray of the number of
        remaining active cross-linking sites for each core node.
        core_nodes (np.ndarray): np.ndarray of the core node numbers.
        core2pb_nodes (list[np.ndarray]): List of np.ndarrays
        corresponding to the periodic boundary nodes associated with
        each core node.
        core_nodes_nghbrhd (list[np.ndarray]): List of np.ndarrays
        corresponding to the core and periodic boundary node numbers in
        the sampling neighborhood of each core node.
        r_core_nodes_nghbrhd (list[np.ndarray]): List of np.ndarrays
        corresponding to the neighbor node-to-core node distances for
        each core node.
    
    Returns:
        tuple[np.ndarray, int, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        Core node numbers that have active cross-linking sites, number
        of core node active cross-linking sites, number of remaining
        active cross-linking sites for each core node, core and periodic
        boundary node numbers in the sampling neighborhood of each core
        node, and the neighbor node-to-core node distances for each core
        node.
    
    """
    # Update active cross-linking sites
    core_nodes_active_sites = np.delete(
        core_nodes_active_sites, core_node_updt_active_site_indx, axis=0)
    core_nodes_active_sites_num -= 1
    # Update anti-degree
    core_nodes_anti_k[core_node_updt] -= 1
    # Update core node sampling neighborhood list if the core node now
    # has no more active cross-linking sites
    if core_nodes_anti_k[core_node_updt] == 0:
        # Gather periodic boundary nodes associated with the inactive
        # core node
        pb_nodes_updt = core2pb_nodes[core_node_updt]
        # Assess and update each core node sampling neighborhood list
        for core_node in np.nditer(core_nodes):
            core_node = int(core_node)
            nghbr_nodes = core_nodes_nghbrhd[core_node]
            r_nghbr_nodes = r_core_nodes_nghbrhd[core_node]
            if np.shape(nghbr_nodes)[0] == 0: pass
            else:
                # Address inactive core node
                core_node_updt_indx_arr = (
                    np.where(nghbr_nodes == core_node_updt)[0]
                )
                if np.shape(core_node_updt_indx_arr)[0] == 0: pass
                else:
                    core_node_updt_indx = int(core_node_updt_indx_arr[0])
                    nghbr_nodes = np.delete(
                        nghbr_nodes, core_node_updt_indx, axis=0)
                    r_nghbr_nodes = np.delete(
                        r_nghbr_nodes, core_node_updt_indx, axis=0)
                # Address all associated inactive periodic boundary
                # nodes
                if np.shape(pb_nodes_updt)[0] == 0: pass
                else:
                    for pb_node_updt in np.nditer(pb_nodes_updt):
                        pb_node_updt = int(pb_node_updt)
                        pb_node_updt_indx_arr = (
                            np.where(nghbr_nodes == pb_node_updt)[0]
                        )
                        if np.shape(pb_node_updt_indx_arr)[0] == 0: pass
                        else:
                            pb_node_updt_indx = int(pb_node_updt_indx_arr[0])
                            nghbr_nodes = np.delete(
                                nghbr_nodes, pb_node_updt_indx, axis=0)
                            r_nghbr_nodes = np.delete(
                                r_nghbr_nodes, pb_node_updt_indx, axis=0)
                core_nodes_nghbrhd[core_node] = nghbr_nodes
                r_core_nodes_nghbrhd[core_node] = r_nghbr_nodes
    
    return (
        core_nodes_active_sites, core_nodes_active_sites_num, core_nodes_anti_k,
        core_nodes_nghbrhd, r_core_nodes_nghbrhd
    )

def dim_2_dangling_chains_update_func(
        core_node_updt: int,
        max_try: int,
        rng: np.random.Generator,
        r_nghbrhd_chns: np.ndarray,
        p_nghbrhd_chns: np.ndarray,
        b: float,
        L: float,
        dnglng_chn_fail: bool,
        dnglng_n: int,
        core_dnglng_m: int,
        pb_dnglng_m: int,
        conn_core_dnglng_edges: list[list[int, int]],
        conn_pb_dnglng_edges: list[list[int, int]],
        core_dnglng_chns_x: list[float],
        core_dnglng_chns_y: list[float],
        core_pb_dnglng_chns_x: np.ndarray,
        core_pb_dnglng_chns_y: np.ndarray) -> tuple[bool, int, int, int, list[list[int, int]], list[list[int, int]], list[float], list[float], np.ndarray, np.ndarray]:
    """Two-dimensional dangling chains update protocol.

    This function instantiates a dangling chain about a particular core
    cross-linker node by randomly placing the dangling chain free end in
    the two-dimensional network and confirming that this free end is at
    least a distance b away from all other nodes.

    Args:
        core_node_updt (int): Core node number corresponding to where an
        active cross-linking site is becoming inactive. This node is
        also where the dangling chain will emanate out from.
        max_try (int): Maximum number of dangling chain instantiation
        attempts.
        rng (np.random.Generator): np.random.Generator object.
        r_nghbrhd_chns (np.ndarray): np.ndarray of the core cross-linker
        sampling neighborhood radius.
        p_nghbrhd_chns (np.ndarray): np.ndarray of the core cross-linker
        sampling neighborhood polymer chain probability distribution.
        b (float): Chain segment and/or cross-linker diameter.
        L (float): Simulation box size.
        dnglng_chn_fail (bool): Tracker for failure of dangling chain
        creation.
        dnglng_n (int): Number of dangling chains, and dangling chain
        core node number.
        core_dnglng_m (int): Number of core dangling chains.
        pb_dnglng_m (int): Number of periodic boundary dangling chains.
        conn_core_dnglng_edges (list[list[int, int]]): Core dangling
        chain edge list, where each edge is represented as
        [core_node_updt, dnglng_n].
        conn_pb_dnglng_edges (list[list[int, int]]): Periodic boundary
        dangling chain edge list, where each edge is represented as
        [core_node_updt, dnglng_n].
        core_dnglng_chns_x (list[float]): List of dangling chain free
        end node x-coordinates in the simulation box.
        core_dnglng_chns_y (list[float]): List of dangling chain free
        end node y-coordinates in the simulation box.
        core_pb_dnglng_chns_x (np.ndarray): np.ndarray of core, periodic
        boundary, and dangling chain free end node x-coordinates.
        core_pb_dnglng_chns_y (np.ndarray): np.ndarray of core, periodic
        boundary, and dangling chain free end node y-coordinates.
    
    Returns:
        tuple[bool, int, int, int, list[list[int, int]], list[list[int, int]], list[float], list[float], np.ndarray, np.ndarray]:
        Tracker for failure of dangling chain creation, number of
        dangling chains, number of core dangling chains, number of
        periodic boundary dangling chains, core dangling chain edge
        list, periodic boundary dangling chain edge list, list of
        dangling chain free end node x-coordinates in the simulation
        box, list of dangling chain free end node y-coordinates in the
        simulation box, np.ndarray of core, periodic boundary, and
        dangling chain free end node x-coordinates, and np.ndarray of
        core, periodic boundary, and dangling chain free end node
        y-coordinates.
    
    """
    # Two-dimensional tessellation protocol
    dim_2_tsslltn, dim_2_tsslltn_num = tessellation_protocol(2)
    
    # Begin dangling chain update procedure
    num_try = 0

    while num_try < max_try:
        # Isotropically place a candidate node representing the free end
        # of the dangling chain
        node_dnglng_chn_cnddt_x = core_pb_dnglng_chns_x[core_node_updt]
        node_dnglng_chn_cnddt_y = core_pb_dnglng_chns_y[core_node_updt]
        # Randomly-selected polar coordinates
        theta = 2 * np.pi * rng.uniform()
        r = rng.choice(r_nghbrhd_chns, size=None, p=p_nghbrhd_chns)
        # Polar-to-Cartesian x- and y-coordinate transformation
        node_dnglng_chn_cnddt_x += r * np.cos(theta)
        node_dnglng_chn_cnddt_y += r * np.sin(theta)
        # Position of candidate node representing the free end of the
        # dangling chain
        node_dnglng_chn_cnddt = np.asarray(
            [
                node_dnglng_chn_cnddt_x,
                node_dnglng_chn_cnddt_y
            ]
        )
        # Downselect the local square neighborhood about the free end of
        # the dangling chain defined by b. Start by gathering the
        # previously-generated nodes that meet this criterion in each x-
        # and y-coordinate.
        nghbr_x_lb = node_dnglng_chn_cnddt_x - b
        nghbr_x_ub = node_dnglng_chn_cnddt_x + b
        nghbr_y_lb = node_dnglng_chn_cnddt_y - b
        nghbr_y_ub = node_dnglng_chn_cnddt_y + b
        psbl_nghbr_x_indcs = (
            np.where(np.logical_and(core_pb_dnglng_chns_x>=nghbr_x_lb, core_pb_dnglng_chns_x<=nghbr_x_ub))[0]
        )
        psbl_nghbr_y_indcs = (
            np.where(np.logical_and(core_pb_dnglng_chns_y>=nghbr_y_lb, core_pb_dnglng_chns_y<=nghbr_y_ub))[0]
        )
        # Gather and retain unique indices corresponding to each
        # possible neighbor, and the number of times each such index
        # value appears
        psbl_nghbr_indcs, psbl_nghbr_indcs_counts = np.unique(
            np.concatenate((psbl_nghbr_x_indcs, psbl_nghbr_y_indcs), dtype=int),
            return_counts=True)
        # The true neighbors are those whose index value appears twice
        # in the possible neighbor array -- equal to the network
        # dimensionality
        nghbr_indcs_vals_indcs = np.where(psbl_nghbr_indcs_counts == 2)[0]
        nghbr_num = np.shape(nghbr_indcs_vals_indcs)[0]
        # Continue analysis if a local neighborhood actually exists
        # about the candidate
        if nghbr_num > 0:
            # Gather the indices of the neighbors
            nghbr_indcs = psbl_nghbr_indcs[nghbr_indcs_vals_indcs]
            # Extract neighbor x- and y-coordinates
            nghbr_x = core_pb_dnglng_chns_x[nghbr_indcs]
            nghbr_y = core_pb_dnglng_chns_y[nghbr_indcs]
            # Calculate the minimum distance between the candidate and
            # its neighbors
            dist = np.empty(nghbr_num)
            for nghbr_indx in range(nghbr_num):
                nghbr = np.asarray(
                    [
                        nghbr_x[nghbr_indx],
                        nghbr_y[nghbr_indx]
                    ]
                )
                dist[nghbr_indx] = np.linalg.norm(node_dnglng_chn_cnddt-nghbr)
            min_dist = np.min(dist)
            # Try again if the minimum distance between the candidate
            # and its neighbors is less than b
            if min_dist < b:
                num_try += 1
                continue
        
        # Accept and tessellate the candidate if (1) no local
        # neighborhood exists about the candidate, or (2) the minimum
        # distance between the candidate and its neighbors is greater
        # than or equal to b
        # Increase dangling chain core node number
        dnglng_n += 1
        # Confirm if the node representing the free end of the dangling
        # chain is a core node or a periodic boundary node
        core_node_dnglng_chn_x = node_dnglng_chn_cnddt_x
        core_node_dnglng_chn_y = node_dnglng_chn_cnddt_y
        # Free end is a core node
        if (0 <= core_node_dnglng_chn_x < L) and (0 <= core_node_dnglng_chn_y < L):
            # Increase core dangling chain number
            core_dnglng_m += 1
            # Add to edge list
            conn_core_dnglng_edges.append([core_node_updt, dnglng_n])
        # Free end is a periodic boundary node
        else:
            # Determine the free end core node x- and y-coordinates that
            # correspond to the free end periodic boundary node via the
            # minimum image criterion
            if core_node_dnglng_chn_x < 0: core_node_dnglng_chn_x += L
            elif core_node_dnglng_chn_x >= L: core_node_dnglng_chn_x -= L
            if core_node_dnglng_chn_y < 0: core_node_dnglng_chn_y += L
            elif core_node_dnglng_chn_y >= L: core_node_dnglng_chn_y -= L
            # Increase periodic boundary dangling chain number
            pb_dnglng_m += 1
            # Add to edge lists
            conn_pb_dnglng_edges.append([core_node_updt, dnglng_n])
        # Add x- and y-coordinates to coordinate lists
        core_dnglng_chns_x.append(core_node_dnglng_chn_x)
        core_dnglng_chns_y.append(core_node_dnglng_chn_y)
        # Use two-dimensional tessellation protocol to tessellate
        # the accepted candidate core node
        node_dnglng_chn_tsslltn_x, node_dnglng_chn_tsslltn_y = (
            dim_2_tessellation_protocol(
                L, core_node_dnglng_chn_x, core_node_dnglng_chn_y,
                dim_2_tsslltn)
        )
        core_pb_dnglng_chns_x = (
            np.concatenate((core_pb_dnglng_chns_x, node_dnglng_chn_tsslltn_x))
        )
        core_pb_dnglng_chns_y = (
            np.concatenate((core_pb_dnglng_chns_y, node_dnglng_chn_tsslltn_y))
        )
        
        break

    # Update the dangling chain creation failure tracker if the number
    # of attempts to instantiate the dangling chain is equal to its
    # maximal value
    if num_try == max_try: dnglng_chn_fail = True

    return (
        dnglng_chn_fail, dnglng_n, core_dnglng_m, pb_dnglng_m,
        conn_core_dnglng_edges, conn_pb_dnglng_edges,
        core_dnglng_chns_x, core_dnglng_chns_y,
        core_pb_dnglng_chns_x, core_pb_dnglng_chns_y
    )

def dim_3_dangling_chains_update_func(
        core_node_updt: int,
        max_try: int,
        rng: np.random.Generator,
        r_nghbrhd_chns: np.ndarray,
        p_nghbrhd_chns: np.ndarray,
        b: float,
        L: float,
        dnglng_chn_fail: bool,
        dnglng_n: int,
        core_dnglng_m: int,
        pb_dnglng_m: int,
        conn_core_dnglng_edges: list[list[int, int]],
        conn_pb_dnglng_edges: list[list[int, int]],
        core_dnglng_chns_x: list[float],
        core_dnglng_chns_y: list[float],
        core_dnglng_chns_z: list[float],
        core_pb_dnglng_chns_x: np.ndarray,
        core_pb_dnglng_chns_y: np.ndarray,
        core_pb_dnglng_chns_z: np.ndarray) -> tuple[bool, int, int, int, list[list[int, int]], list[list[int, int]], list[float], list[float], list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Three-dimensional dangling chains update protocol.

    This function instantiates a dangling chain about a particular core
    cross-linker node by randomly placing the dangling chain free end in
    the three-dimensional network and confirming that this free end is
    at least a distance b away from all other nodes.

    Args:
        core_node_updt (int): Core node number corresponding to where an
        active cross-linking site is becoming inactive. This node is
        also where the dangling chain will emanate out from.
        max_try (int): Maximum number of dangling chain instantiation
        attempts.
        rng (np.random.Generator): np.random.Generator object.
        r_nghbrhd_chns (np.ndarray): np.ndarray of the core cross-linker
        sampling neighborhood radius.
        p_nghbrhd_chns (np.ndarray): np.ndarray of the core cross-linker
        sampling neighborhood polymer chain probability distribution.
        b (float): Chain segment and/or cross-linker diameter.
        L (float): Simulation box size.
        dnglng_chn_fail (bool): Tracker for failure of dangling chain
        creation.
        dnglng_n (int): Number of dangling chains, and dangling chain
        core node number.
        core_dnglng_m (int): Number of core dangling chains.
        pb_dnglng_m (int): Number of periodic boundary dangling chains.
        conn_core_dnglng_edges (list[list[int, int]]): Core dangling
        chain edge list, where each edge is represented as
        [core_node_updt, dnglng_n].
        conn_pb_dnglng_edges (list[list[int, int]]): Periodic boundary
        dangling chain edge list, where each edge is represented as
        [core_node_updt, dnglng_n].
        core_dnglng_chns_x (list[float]): List of dangling chain free
        end node x-coordinates in the simulation box.
        core_dnglng_chns_y (list[float]): List of dangling chain free
        end node y-coordinates in the simulation box.
        core_dnglng_chns_z (list[float]): List of dangling chain free
        end node z-coordinates in the simulation box.
        core_pb_dnglng_chns_x (np.ndarray): np.ndarray of core, periodic
        boundary, and dangling chain free end node x-coordinates.
        core_pb_dnglng_chns_y (np.ndarray): np.ndarray of core, periodic
        boundary, and dangling chain free end node y-coordinates.
        core_pb_dnglng_chns_z (np.ndarray): np.ndarray of core, periodic
        boundary, and dangling chain free end node z-coordinates.
    
    Returns:
        tuple[bool, int, int, int, list[list[int, int]], list[list[int, int]], list[float], list[float], list[float], np.ndarray, np.ndarray, np.ndarray]:
        Tracker for failure of dangling chain creation, number of
        dangling chains, number of core dangling chains, number of
        periodic boundary dangling chains, core dangling chain edge
        list, periodic boundary dangling chain edge list, list of
        dangling chain free end node x-coordinates in the simulation
        box, list of dangling chain free end node y-coordinates in the
        simulation box, list of dangling chain free end node
        z-coordinates in the simulation box, np.ndarray of core,
        periodic boundary, and dangling chain free end node
        x-coordinates, np.ndarray of core, periodic boundary, and
        dangling chain free end node y-coordinates, and np.ndarray of
        core, periodic boundary, and dangling chain free end node
        z-coordinates.
    
    """
    # Three-dimensional tessellation protocol
    dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)
    
    # Begin dangling chain update procedure
    num_try = 0

    while num_try < max_try:
        # Isotropically place a candidate node representing the free end
        # of the dangling chain
        node_dnglng_chn_cnddt_x = core_pb_dnglng_chns_x[core_node_updt]
        node_dnglng_chn_cnddt_y = core_pb_dnglng_chns_y[core_node_updt]
        node_dnglng_chn_cnddt_z = core_pb_dnglng_chns_z[core_node_updt]
        # Randomly-selected spherical coordinates
        theta = np.pi * rng.uniform()
        phi = 2 * np.pi * rng.uniform()
        r = rng.choice(r_nghbrhd_chns, size=None, p=p_nghbrhd_chns)
        # Spherical-to-Cartesian x-, y-, and z-coordinate transformation
        node_dnglng_chn_cnddt_x += r * np.sin(theta) * np.cos(phi)
        node_dnglng_chn_cnddt_y += r * np.sin(theta) * np.sin(phi)
        node_dnglng_chn_cnddt_z += r * np.cos(theta)
        # Position of candidate node representing the free end of the
        # dangling chain
        node_dnglng_chn_cnddt = np.asarray(
            [
                node_dnglng_chn_cnddt_x,
                node_dnglng_chn_cnddt_y,
                node_dnglng_chn_cnddt_z
            ]
        )
        # Downselect the local square neighborhood about the free end of
        # the dangling chain defined by b. Start by gathering the
        # previously-generated nodes that meet this criterion in each
        # x-, y-, and z-coordinate.
        nghbr_x_lb = node_dnglng_chn_cnddt_x - b
        nghbr_x_ub = node_dnglng_chn_cnddt_x + b
        nghbr_y_lb = node_dnglng_chn_cnddt_y - b
        nghbr_y_ub = node_dnglng_chn_cnddt_y + b
        nghbr_z_lb = node_dnglng_chn_cnddt_z - b
        nghbr_z_ub = node_dnglng_chn_cnddt_z + b
        psbl_nghbr_x_indcs = (
            np.where(np.logical_and(core_pb_dnglng_chns_x>=nghbr_x_lb, core_pb_dnglng_chns_x<=nghbr_x_ub))[0]
        )
        psbl_nghbr_y_indcs = (
            np.where(np.logical_and(core_pb_dnglng_chns_y>=nghbr_y_lb, core_pb_dnglng_chns_y<=nghbr_y_ub))[0]
        )
        psbl_nghbr_z_indcs = (
            np.where(np.logical_and(core_pb_dnglng_chns_z>=nghbr_z_lb, core_pb_dnglng_chns_z<=nghbr_z_ub))[0]
        )
        # Gather and retain unique indices corresponding to each
        # possible neighbor, and the number of times each such index
        # value appears
        psbl_nghbr_indcs, psbl_nghbr_indcs_counts = np.unique(
            np.concatenate((psbl_nghbr_x_indcs, psbl_nghbr_y_indcs, psbl_nghbr_z_indcs), dtype=int),
            return_counts=True)
        # The true neighbors are those whose index value appears thrice
        # in the possible neighbor array -- equal to the network
        # dimensionality
        nghbr_indcs_vals_indcs = np.where(psbl_nghbr_indcs_counts == 3)[0]
        nghbr_num = np.shape(nghbr_indcs_vals_indcs)[0]
        # Continue analysis if a local neighborhood actually exists
        # about the candidate
        if nghbr_num > 0:
            # Gather the indices of the neighbors
            nghbr_indcs = psbl_nghbr_indcs[nghbr_indcs_vals_indcs]
            # Extract neighbor x-, y-, and z-coordinates
            nghbr_x = core_pb_dnglng_chns_x[nghbr_indcs]
            nghbr_y = core_pb_dnglng_chns_y[nghbr_indcs]
            nghbr_z = core_pb_dnglng_chns_z[nghbr_indcs]
            # Calculate the minimum distance between the candidate and
            # its neighbors
            dist = np.empty(nghbr_num)
            for nghbr_indx in range(nghbr_num):
                nghbr = np.asarray(
                    [
                        nghbr_x[nghbr_indx],
                        nghbr_y[nghbr_indx],
                        nghbr_z[nghbr_indx]
                    ]
                )
                dist[nghbr_indx] = np.linalg.norm(node_dnglng_chn_cnddt-nghbr)
            min_dist = np.min(dist)
            # Try again if the minimum distance between the candidate
            # and its neighbors is less than b
            if min_dist < b:
                num_try += 1
                continue
        
        # Accept and tessellate the candidate if (1) no local
        # neighborhood exists about the candidate, or (2) the minimum
        # distance between the candidate and its neighbors is greater
        # than or equal to b
        # Increase dangling chain core node number
        dnglng_n += 1
        # Confirm if the node representing the free end of the dangling
        # chain is a core node or a periodic boundary node
        core_node_dnglng_chn_x = node_dnglng_chn_cnddt_x
        core_node_dnglng_chn_y = node_dnglng_chn_cnddt_y
        core_node_dnglng_chn_z = node_dnglng_chn_cnddt_z
        # Free end is a core node
        if (0 <= core_node_dnglng_chn_x < L) and (0 <= core_node_dnglng_chn_y < L) and (0 <= core_node_dnglng_chn_z < L):
            # Increase core dangling chain number
            core_dnglng_m += 1
            # Add to edge list
            conn_core_dnglng_edges.append([core_node_updt, dnglng_n])
        # Free end is a periodic boundary node
        else:
            # Determine the free end core node x-, y-, and z-coordinates
            # that correspond to the free end periodic boundary node via
            # the minimum image criterion
            if core_node_dnglng_chn_x < 0: core_node_dnglng_chn_x += L
            elif core_node_dnglng_chn_x >= L: core_node_dnglng_chn_x -= L
            if core_node_dnglng_chn_y < 0: core_node_dnglng_chn_y += L
            elif core_node_dnglng_chn_y >= L: core_node_dnglng_chn_y -= L
            if core_node_dnglng_chn_z < 0: core_node_dnglng_chn_z += L
            elif core_node_dnglng_chn_z >= L: core_node_dnglng_chn_z -= L
            # Increase periodic boundary dangling chain number
            pb_dnglng_m += 1
            # Add to edge lists
            conn_pb_dnglng_edges.append([core_node_updt, dnglng_n])
        # Add x-, y-, and z-coordinates to coordinate lists
        core_dnglng_chns_x.append(core_node_dnglng_chn_x)
        core_dnglng_chns_y.append(core_node_dnglng_chn_y)
        core_dnglng_chns_z.append(core_node_dnglng_chn_z)
        # Use three-dimensional tessellation protocol to tessellate
        # the accepted candidate core node
        node_dnglng_chn_tsslltn_x, node_dnglng_chn_tsslltn_y, node_dnglng_chn_tsslltn_z = (
            dim_3_tessellation_protocol(
                L, core_node_dnglng_chn_x, core_node_dnglng_chn_y,
                core_node_dnglng_chn_z, dim_3_tsslltn)
        )
        core_pb_dnglng_chns_x = (
            np.concatenate((core_pb_dnglng_chns_x, node_dnglng_chn_tsslltn_x))
        )
        core_pb_dnglng_chns_y = (
            np.concatenate((core_pb_dnglng_chns_y, node_dnglng_chn_tsslltn_y))
        )
        core_pb_dnglng_chns_z = (
            np.concatenate((core_pb_dnglng_chns_z, node_dnglng_chn_tsslltn_z))
        )
        
        break

    # Update the dangling chain creation failure tracker if the number
    # of attempts to instantiate the dangling chain is equal to its
    # maximal value
    if num_try == max_try: dnglng_chn_fail = True

    return (
        dnglng_chn_fail, dnglng_n, core_dnglng_m, pb_dnglng_m,
        conn_core_dnglng_edges, conn_pb_dnglng_edges,
        core_dnglng_chns_x, core_dnglng_chns_y, core_dnglng_chns_z,
        core_pb_dnglng_chns_x, core_pb_dnglng_chns_y, core_pb_dnglng_chns_z
    )

def aelp_dim_2_network_topology_initialization(
        network: str,
        L: float,
        core_x: np.ndarray,
        core_y: np.ndarray,
        b: float,
        xi: float,
        k: int,
        n: int,
        nu: int,
        max_try: int,
        filename_prefix: str) -> None:
    """Network topology initialization procedure for two-dimensional
    artificial end-linked polymer networks.

    This function initializes and saves the topology of a
    two-dimensional artificial end-linked polymer network via a modified
    Gusev-Hanson procedure.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, either "auelp" or "apelp" are applicable
        (corresponding to artificial uniform end-linked polymer networks
        ("auelp") or artificial polydisperse end-linked polymer networks
        ("apelp")).
        L (float): Simulation box size.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
        b (float): Chain segment and/or cross-linker diameter.
        xi (float): Chain-to-cross-link connection probability.
        k (int): Maximum cross-linker degree/functionality; either 3, 4,
        5, 6, 7, or 8.
        n (int): Number of core cross-linkers.
        nu (int): (Average) Number of segments per chain.
        max_try (int): Maximum number of dangling chain instantiation
        attempts.
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Network topology initialization procedure is only applicable for
    # artificial end-linked polymer networks. Exit if a different type
    # of network is passed.
    if network != "auelp":
        if network != "apelp":
            error_str = (
                "Network topology initialization procedure is only "
                + "applicable for artificial end-linked polymer "
                + "networks. This calculation will only proceed if "
                + "network = ``auelp'' or network = ``apelp''."
            )
            sys.exit(error_str)
    
    # Generate filenames
    mx_cmp_conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    mx_cmp_core_node_type_filename = (
        filename_prefix + "-core_node_type" + ".dat"
    )
    mx_cmp_conn_core_edges_filename = (
        filename_prefix + "-conn_core_edges" + ".dat"
    )
    mx_cmp_conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    mx_cmp_core_x_filename = filename_prefix + "-core_x" + ".dat"
    mx_cmp_core_y_filename = filename_prefix + "-core_y" + ".dat"
    
    # Calculate the stoichiometric number of chains
    m = m_arg_stoich_func(n, k)

    # As a fail-safe check, force int-valued parameters to be ints
    k = int(np.floor(k))
    n = int(np.floor(n))
    nu = int(np.floor(nu))
    max_try = int(np.floor(max_try))
    m = int(np.floor(m))

    # Core cross-linker nodes
    core_nodes = np.arange(n, dtype=int)

    # Identify core nodes as cross-linkers
    core_node_type = np.ones(n, dtype=int)

    # Copy the core_x and core_y np.ndarrays as the first n entries in
    # the tessellated cross-linker x- and y-coordinate np.ndarrays
    tsslltd_core_x = core_x.copy()
    tsslltd_core_y = core_y.copy()

    # Two-dimensional tessellation protocol
    dim_2_tsslltn, dim_2_tsslltn_num = tessellation_protocol(2)
        
    # Use two-dimensional tessellation protocol to tessellate the core
    # cross-linkers
    for tsslltn in range(dim_2_tsslltn_num):
        x_tsslltn = dim_2_tsslltn[tsslltn, 0]
        y_tsslltn = dim_2_tsslltn[tsslltn, 1]
        # Skip the (hold, hold) tessellation call because the core
        # cross-linkers are being tessellated about themselves
        if (x_tsslltn == 0) and (y_tsslltn == 0): continue
        else:
            # x- and y-coordinates from two-dimensional tessellation
            # protocol
            core_tsslltn_x, core_tsslltn_y = dim_2_tessellation(
                L, core_x, core_y, x_tsslltn, y_tsslltn)
            # Concatenate the tessellated x- and y-coordinates
            tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
            tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))

    del core_tsslltn_x, core_tsslltn_y

    # Construct the pb2core_nodes np.ndarray such that
    # pb2core_nodes[core_pb_node] = core_node
    pb2core_nodes = np.tile(core_nodes, dim_2_tsslltn_num)

    # Maximal core cross-linker sampling neighborhood radius as per the
    # minimum image criterion 
    r_mic = L / 2.
    # Polymer chain contour length
    l_cntr = nu * b
    # Determine the core cross-linker sampling neighborhood radius
    r_nghbrhd = np.minimum(r_mic, l_cntr)

    # Finely discretize the core cross-linker sampling neighborhood
    # radius
    r_nghbrhd_chns = np.linspace(0, r_nghbrhd, 10001)
    # Depending on the type of artificial end-linked polymer network,
    # calculate the polymer chain probability distribution
    p_nghbrhd_chns = np.asarray([])
    if network == "auelp":
        p_nghbrhd_chns = p_gaussian_cnfrmtn_func(b, nu, r_nghbrhd_chns)
    elif network == "apelp":
        p_nghbrhd_chns = p_rel_net_r_arr_gaussian_cnfrmtn_func(
            b, nu, r_nghbrhd_chns)
    # Normalize the polymer chain probability distribution
    p_nghbrhd_chns /= np.sum(p_nghbrhd_chns, dtype=float)

    # Initialize core cross-linker sampling neighborhood list
    core_nodes_nghbrhd_init = []
    r_core_nodes_nghbrhd_init = []

    # Determine core cross-linker sampling neighborhood
    for core_node in np.nditer(core_nodes):
        core_node = int(core_node)
        # x- and y-coordinates of core cross-linker
        core_node_x = tsslltd_core_x[core_node]
        core_node_y = tsslltd_core_y[core_node]
        # Core cross-linker position
        core_node_pstn = np.asarray(
            [
                core_node_x,
                core_node_y
            ]
        )
        # Downselect the local square neighborhood about the core node
        # defined by r_nghbrhd. Start by gathering the tessellated
        # cross-linker nodes that meet this criterion in each x- and
        # y-coordinate.
        lcl_nghbr_x_lb = core_node_x - r_nghbrhd
        lcl_nghbr_x_ub = core_node_x + r_nghbrhd
        lcl_nghbr_y_lb = core_node_y - r_nghbrhd
        lcl_nghbr_y_ub = core_node_y + r_nghbrhd
        psbl_lcl_nghbr_x_nodes = (
            np.where(np.logical_and(tsslltd_core_x>=lcl_nghbr_x_lb, tsslltd_core_x<=lcl_nghbr_x_ub))[0]
        )
        psbl_lcl_nghbr_y_nodes = (
            np.where(np.logical_and(tsslltd_core_y>=lcl_nghbr_y_lb, tsslltd_core_y<=lcl_nghbr_y_ub))[0]
        )
        # Gather the nodes from each x- and y-coordinate together to
        # assess all possible local cross-linker neighbors. Retain
        # unique possible local cross-linker neighbor nodes, and the
        # number of times each such node appears.
        psbl_lcl_nghbr_nodes, psbl_lcl_nghbr_nodes_counts = np.unique(
            np.concatenate((psbl_lcl_nghbr_x_nodes, psbl_lcl_nghbr_y_nodes), dtype=int),
            return_counts=True)
        # The true local cross-linker neighbor nodes are those who
        # appear twice in the possible cross-linker neighbor node array
        # -- equal to the network dimensionality
        lcl_nghbr_nodes_indcs = np.where(psbl_lcl_nghbr_nodes_counts == 2)[0]
        lcl_nghbr_node_num = np.shape(lcl_nghbr_nodes_indcs)[0]
        # Further downselect to the neighborhood of tessellated
        # cross-linker nodes that are a distance of r_nghbrhd away from
        # the core cross-linker node
        if lcl_nghbr_node_num > 0:
            # Gather the indices of the local cross-linker neighbors
            lcl_nghbr_nodes = psbl_lcl_nghbr_nodes[lcl_nghbr_nodes_indcs]
            # Extract local cross-linker neighbor x- and y-coordinates
            lcl_nghbr_x = tsslltd_core_x[lcl_nghbr_nodes]
            lcl_nghbr_y = tsslltd_core_y[lcl_nghbr_nodes]
            # Calculate the distance between the core node and the local
            # cross-linker neighbor nodes
            r_lcl_nghbr_nodes = np.empty(lcl_nghbr_node_num)
            for lcl_nghbr_node_indx in range(lcl_nghbr_node_num):
                lcl_nghbr_pstn = np.asarray(
                    [
                        lcl_nghbr_x[lcl_nghbr_node_indx],
                        lcl_nghbr_y[lcl_nghbr_node_indx]
                    ]
                )
                r_lcl_nghbr_nodes[lcl_nghbr_node_indx] = (
                    np.linalg.norm(core_node_pstn-lcl_nghbr_pstn)
                )
            # The true cross-linker neighbor nodes are those whose
            # distance to the core node is less than r_nghbrhd
            nghbr_nodes_indcs = np.where(r_lcl_nghbr_nodes <= r_nghbrhd)[0]
            nghbr_nodes = lcl_nghbr_nodes[nghbr_nodes_indcs]
            r_nghbr_nodes = r_lcl_nghbr_nodes[nghbr_nodes_indcs]
            # Add the cross-linker neighbor nodes array to the core
            # cross-linker sampling neighborhood list
            core_nodes_nghbrhd_init.append(nghbr_nodes)
            r_core_nodes_nghbrhd_init.append(r_nghbr_nodes)
        else:
            core_nodes_nghbrhd_init.append(np.asarray([]))
            r_core_nodes_nghbrhd_init.append(np.asarray([]))
    
    # Retain unique nodes from the core and periodic boundary
    # cross-linkers in the core cross-linker sampling neighborhood list
    core_pb_nodes = np.unique(
        np.concatenate(
            tuple(nghbrhd for nghbrhd in core_nodes_nghbrhd_init), dtype=int))

    # Extract the core and periodic boundary cross-linker x- and
    # y-coordinates using the corresponding node numbers
    core_pb_x = tsslltd_core_x[core_pb_nodes].copy()
    core_pb_y = tsslltd_core_y[core_pb_nodes].copy()

    del tsslltd_core_x, tsslltd_core_y

    # Extract the core and periodic boundary cross-linker nodes in the
    # pb2core_nodes np.ndarray
    pb2core_nodes = pb2core_nodes[core_pb_nodes].copy()

    # Construct the core2pb_nodes list
    core2pb_nodes = core2pb_nodes_func(core_nodes, pb2core_nodes)

    # Refactor the node numbers in the core cross-linker sampling
    # neighborhood list
    for core_node in np.nditer(core_nodes):
        core_node = int(core_node)
        nghbr_nodes = core_nodes_nghbrhd_init[core_node]
        for nghbr_node_indx in range(np.shape(nghbr_nodes)[0]):
            nghbr_nodes[nghbr_node_indx] = int(
                np.where(core_pb_nodes == nghbr_nodes[nghbr_node_indx])[0][0])
        core_nodes_nghbrhd_init[core_node] = nghbr_nodes

    # Initialize tracker for failure of dangling chain creation
    dnglng_chn_fail = False

    # Initialize random number generator
    rng = np.random.default_rng()

    # Initialize dangling chain free end node x- and y-coordinates
    core_dnglng_chns_x = []
    core_dnglng_chns_y = []

    # Initialize edge lists
    conn_core_edges = []
    conn_pb_edges = []
    conn_core_dnglng_edges = []
    conn_pb_dnglng_edges = []

    # Initialize dangling chain node and edge numbers
    core_dnglng_m = 0
    pb_dnglng_m = 0
    dnglng_n = -1

    # Network topology initialization
    while True:
        # Initialize anti-degree of the core cross-linker nodes
        core_nodes_anti_k = k*np.ones(n, dtype=int)
        # Initialize core cross-linker node active sites
        core_nodes_active_sites = np.repeat(core_nodes, k)
        core_nodes_active_sites_num = np.shape(core_nodes_active_sites)[0]

        # Initialize core cross-linker sampling neighborhood list
        core_nodes_nghbrhd = core_nodes_nghbrhd_init.copy()
        r_core_nodes_nghbrhd = r_core_nodes_nghbrhd_init.copy()
        
        # Initialize cross-linker node x- and y-coordinates for dangling
        # chains
        core_pb_dnglng_chns_x = core_pb_x.copy()
        core_pb_dnglng_chns_y = core_pb_y.copy()

        # Initialize dangling chain free end node x- and y-coordinates
        core_dnglng_chns_x = []
        core_dnglng_chns_y = []

        # Initialize edge array and lists
        edges = np.full((m, 2), np.inf)
        conn_core_edges = []
        conn_pb_edges = []
        conn_core_dnglng_edges = []
        conn_pb_dnglng_edges = []

        # Initialize dangling chain node and edge numbers
        core_dnglng_m = 0
        pb_dnglng_m = 0
        dnglng_n = -1

        # Initialize node number integer constants
        core_node = 0
        nghbr_node = 0

        # Initialize and randomly shuffle chain end numbers
        chn_ends = np.arange(2*m, dtype=int)
        rng.shuffle(chn_ends)

        # Initialize network topology on a chain end-by-chain end basis
        for chn_end in np.nditer(chn_ends):
            # Chain end is a free chain end
            if rng.random() > xi: continue
            else:
                # Identify the chain and its ends
                chn_end = int(chn_end)
                chn = int(np.floor(chn_end/2))
                chn_end_0 = int(chn_end%2)
                chn_end_1 = 1 - chn_end_0
                # If this chain has not yet been instantiated, then
                # randomly select an active site from the core nodes to
                # instantiate the chain
                if (edges[chn, 0] == np.inf) and (edges[chn, 1] == np.inf):
                    core_node_active_site_indx = rng.integers(
                        np.shape(core_nodes_active_sites)[0], dtype=int)
                    core_node = (
                        core_nodes_active_sites[core_node_active_site_indx]
                    )
                    # Instantiate the chain at one end
                    edges[chn, chn_end_0] = core_node
                    # Update inactive sites, anti-degree, active core
                    # cross-linker nodes, and the sampling neighborhood
                    # list for the parent core cross-linker node
                    (core_nodes_active_sites, core_nodes_active_sites_num, 
                     core_nodes_anti_k, core_nodes_nghbrhd, r_core_nodes_nghbrhd) = core_node_update_func(
                         core_node, core_node_active_site_indx,
                         core_nodes_active_sites, core_nodes_active_sites_num,
                         core_nodes_anti_k, core_nodes, core2pb_nodes,
                         core_nodes_nghbrhd, r_core_nodes_nghbrhd)
                # Otherwise, the chain has previously been instantiated
                else:
                    # Extract sampling neighborhood about the
                    # previously-instantiated chain end
                    core_node = int(edges[chn, chn_end_1])
                    nghbr_nodes = core_nodes_nghbrhd[core_node]
                    nghbr_nodes_num = np.shape(nghbr_nodes)[0]
                    r_nghbr_nodes = r_core_nodes_nghbrhd[core_node]
                    # Check if the sampling neighborhood is empty. If
                    # so, then the chain is a dangling chain.
                    if nghbr_nodes_num == 0:
                        continue
                    else:
                        # Check if there is only one neighbor in the
                        # sampling neighborhood
                        if nghbr_nodes_num == 1:
                            nghbr_node = nghbr_nodes[0]
                        else:
                            # Calculate and normalize weighting factors
                            p_nghbrhd = np.asarray([])
                            if network == "auelp":
                                p_nghbrhd = p_gaussian_cnfrmtn_func(
                                    b, nu, r_nghbr_nodes)
                            elif network == "apelp":
                                p_nghbrhd = np.empty(nghbr_nodes_num)
                                for nghbr_node_indx in range(nghbr_nodes_num):
                                    r_nghbr_node = (
                                        r_nghbr_nodes[nghbr_node_indx]
                                    )
                                    if r_nghbr_node == 0.0:
                                        p_nghbrhd[nghbr_node_indx] = 0.0
                                    else:
                                        # Linear interpolation
                                        r_nghbrhd_chns_diff = (
                                            r_nghbrhd_chns - r_nghbr_node
                                        )
                                        r_nghbrhd_chns_diff[r_nghbrhd_chns_diff < 0] = (
                                            np.inf
                                        )
                                        indx_right = np.argmin(
                                            r_nghbrhd_chns_diff)
                                        r_right = r_nghbrhd_chns[indx_right]
                                        p_right = p_nghbrhd_chns[indx_right]
                                        indx_left = indx_right - 1
                                        r_left = r_nghbrhd_chns[indx_left]
                                        p_left = p_nghbrhd_chns[indx_left]
                                        p_nghbrhd[nghbr_node_indx] = (
                                            p_left + (r_nghbr_node-r_left)
                                            * (p_right-p_left) / (r_right-r_left)
                                        )
                            p_nghbrhd /= np.sum(p_nghbrhd, dtype=float)
                            # Randomly select a neighbor cross-linker
                            # node to host the other end of the chain
                            nghbr_node = int(
                                rng.choice(nghbr_nodes, size=None, p=p_nghbrhd))
                        # Instantiate the other end of the chain
                        edges[chn, chn_end_0] = nghbr_node
                        core_node = int(pb2core_nodes[nghbr_node])
                        core_node_active_site_indx = int(
                            np.where(core_nodes_active_sites == core_node)[0][0])
                        # Update inactive sites, anti-degree, active core
                        # cross-linker nodes, and the sampling neighborhood list
                        # for the neighbor core cross-linker node
                        (core_nodes_active_sites, core_nodes_active_sites_num,
                         core_nodes_anti_k, core_nodes_nghbrhd,
                         r_core_nodes_nghbrhd) = core_node_update_func(
                            core_node, core_node_active_site_indx,
                            core_nodes_active_sites, core_nodes_active_sites_num,
                            core_nodes_anti_k, core_nodes, core2pb_nodes,
                            core_nodes_nghbrhd, r_core_nodes_nghbrhd)
        
        # Post-process the edges array (that was populated during the
        # network topology initialization) on a chain-by-chain basis
        # into edges lists
        for chn in range(m):
            # Chain is a core edge
            if (edges[chn, 0] < n) and (edges[chn, 1] < n):
                conn_core_edges.append(
                    (int(edges[chn, 0]), int(edges[chn, 1])))
            # Chain is a periodic edge
            elif (edges[chn, 0] < n) and (n <= edges[chn, 1] < np.inf):
                conn_pb_edges.append(
                    (int(edges[chn, 0]), int(pb2core_nodes[int(edges[chn, 1])])))
            elif (n <= edges[chn, 0] < np.inf) and (edges[chn, 1] < n):
                conn_pb_edges.append(
                    (int(pb2core_nodes[int(edges[chn, 0])]), int(edges[chn, 1])))
            # Chain is a dangling chain
            elif (edges[chn, 0] < n) and (edges[chn, 1] == np.inf):
                (dnglng_chn_fail, dnglng_n, core_dnglng_m, pb_dnglng_m,
                 conn_core_dnglng_edges, conn_pb_dnglng_edges,
                 core_dnglng_chns_x, core_dnglng_chns_y,
                 core_pb_dnglng_chns_x, core_pb_dnglng_chns_y) = dim_2_dangling_chains_update_func(
                     int(edges[chn, 0]), max_try, rng, r_nghbrhd_chns,
                     p_nghbrhd_chns, b, L, dnglng_chn_fail, dnglng_n,
                     core_dnglng_m, pb_dnglng_m, conn_core_dnglng_edges,
                     conn_pb_dnglng_edges, core_dnglng_chns_x,
                     core_dnglng_chns_y, core_pb_dnglng_chns_x,
                     core_pb_dnglng_chns_y)
                # Break if a dangling chain failed to be instantiated
                if dnglng_chn_fail == True: break
            elif (edges[chn, 0] == np.inf) and (edges[chn, 1] < n):
                (dnglng_chn_fail, dnglng_n, core_dnglng_m, pb_dnglng_m,
                 conn_core_dnglng_edges, conn_pb_dnglng_edges,
                 core_dnglng_chns_x, core_dnglng_chns_y,
                 core_pb_dnglng_chns_x, core_pb_dnglng_chns_y) = dim_2_dangling_chains_update_func(
                     int(edges[chn, 1]), max_try, rng, r_nghbrhd_chns,
                     p_nghbrhd_chns, b, L, dnglng_chn_fail, dnglng_n,
                     core_dnglng_m, pb_dnglng_m, conn_core_dnglng_edges,
                     conn_pb_dnglng_edges, core_dnglng_chns_x,
                     core_dnglng_chns_y, core_pb_dnglng_chns_x,
                     core_pb_dnglng_chns_y)
                # Break if a dangling chain failed to be instantiated
                if dnglng_chn_fail == True: break
            # Chain is a free chain
            elif (edges[chn, 0] == np.inf) and (edges[chn, 1] == np.inf):
                continue

        # Restart the network topology initialization protocol if a
        # dangling chain failed to be instantiated
        if dnglng_chn_fail == True: continue
        
        del core_nodes_anti_k, core_nodes_active_sites
        del core_pb_dnglng_chns_x, core_pb_dnglng_chns_y
        del edges, chn_ends
        # Break out of acceptable initialized topology 
        break

    del core_pb_nodes, pb2core_nodes, core2pb_nodes
    del core_pb_x, core_pb_y
    del core_nodes_nghbrhd_init, core_nodes_nghbrhd
    del r_core_nodes_nghbrhd_init, r_core_nodes_nghbrhd
    del r_nghbrhd_chns, p_nghbrhd_chns

    # Refactor edge lists to np.ndarrays
    conn_core_edges = np.asarray(conn_core_edges, dtype=int)
    conn_pb_edges = np.asarray(conn_pb_edges, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Create nx.MultiGraphs
    conn_core_graph = nx.MultiGraph()
    conn_pb_graph = nx.MultiGraph()
    conn_graph = nx.MultiGraph()
    
    # Recalibrate number of dangling chains
    dnglng_n += 1

    # No dangling chains were added to the network
    if dnglng_n == 0:
        conn_core_graph = add_nodes_from_numpy_array(
            conn_core_graph, core_nodes)
        conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
        conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
        
        conn_core_graph = add_edges_from_numpy_array(
            conn_core_graph, conn_core_edges)
        conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)
        conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)
    # Only core dangling chains were added to the network
    elif core_dnglng_m > 0 and pb_dnglng_m == 0:
        # Update core dangling chain node numbers
        for edge in conn_core_dnglng_edges: edge[1] += n
        # Update edge lists
        conn_core_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_core_dnglng_edges), dtype=int)
        conn_core_edges = np.vstack(
            (conn_core_edges, conn_core_dnglng_edges), dtype=int)
        conn_edges = np.vstack((conn_edges, conn_core_dnglng_edges), dtype=int)
    # Only periodic boundary dangling chains were added to the network
    elif core_dnglng_m == 0 and pb_dnglng_m > 0:
        # Update periodic boundary dangling chain node numbers
        for edge in conn_pb_dnglng_edges: edge[1] += n
        # Update edge lists
        conn_pb_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_pb_dnglng_edges), dtype=int)
        conn_pb_edges = np.vstack(
            (conn_pb_edges, conn_pb_dnglng_edges), dtype=int)
        conn_edges = np.vstack((conn_edges, conn_pb_dnglng_edges), dtype=int)
    # Both core and periodic boundary dangling chains were added to the
    # network
    else:
        # Update core and periodic boundary dangling chain node numbers
        for edge in conn_core_dnglng_edges: edge[1] += n
        for edge in conn_pb_dnglng_edges: edge[1] += n
        # Update edge lists
        conn_core_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_core_dnglng_edges), dtype=int)
        conn_pb_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_pb_dnglng_edges), dtype=int)
        conn_core_edges = np.vstack(
            (conn_core_edges, conn_core_dnglng_edges), dtype=int)
        conn_pb_edges = np.vstack(
            (conn_pb_edges, conn_pb_dnglng_edges), dtype=int)
        conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)
    # Update core_node_type correspondingly to end-linked polymer
    # network code
    core_node_type = np.concatenate(
        (core_node_type, np.repeat(3, dnglng_n)), dtype=int)
    # Add core coordinates from dangling chains
    core_x = np.concatenate((core_x, np.asarray(core_dnglng_chns_x)))
    core_y = np.concatenate((core_y, np.asarray(core_dnglng_chns_y)))
    # Update core_nodes
    core_nodes = np.arange(n+dnglng_n, dtype=int)
    # Update nx.MultiGraph nodes
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    # Update nx.MultiGraph edges
    conn_core_graph = add_edges_from_numpy_array(
        conn_core_graph, conn_core_edges)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Isolate largest/maximum connected component in a nodewise fashion
    mx_cmp_conn_graph_nodes = max(nx.connected_components(conn_graph), key=len)
    # Extract largest/maximum connected component subgraphs
    mx_cmp_conn_core_graph = (
        conn_core_graph.subgraph(mx_cmp_conn_graph_nodes).copy()
    )
    mx_cmp_conn_pb_graph = (
        conn_pb_graph.subgraph(mx_cmp_conn_graph_nodes).copy()
    )
    # Extract edges
    mx_cmp_conn_core_graph_edges = np.asarray(
        list(mx_cmp_conn_core_graph.edges()), dtype=int)
    mx_cmp_conn_pb_graph_edges = np.asarray(
        list(mx_cmp_conn_pb_graph.edges()), dtype=int)
    # Number of edges in the largest/maximum connected component
    mx_cmp_conn_core_graph_m = np.shape(mx_cmp_conn_core_graph_edges)[0]
    mx_cmp_conn_pb_graph_m = np.shape(mx_cmp_conn_pb_graph_edges)[0]
    # Nodes from the largest/maximum connected component, sorted in
    # ascending order
    mx_cmp_conn_graph_nodes = np.sort(
        np.fromiter(mx_cmp_conn_graph_nodes, dtype=int))
    # Number of nodes in the largest/maximum connected component
    mx_cmp_conn_graph_n = np.shape(mx_cmp_conn_graph_nodes)[0]

    # Isolate core_node_type for the largest/maximum connected component
    mx_cmp_core_node_type = core_node_type[mx_cmp_conn_graph_nodes]
    # Isolate the cross-linker coordinates for the largest/maximum
    # connected component
    mx_cmp_core_x = core_x[mx_cmp_conn_graph_nodes]
    mx_cmp_core_y = core_y[mx_cmp_conn_graph_nodes]

    # Update all original node values with updated node values
    for edge in range(mx_cmp_conn_core_graph_m):
        mx_cmp_conn_core_graph_edges[edge, 0] = int(
            np.where(mx_cmp_conn_graph_nodes == mx_cmp_conn_core_graph_edges[edge, 0])[0][0])
        mx_cmp_conn_core_graph_edges[edge, 1] = int(
            np.where(mx_cmp_conn_graph_nodes == mx_cmp_conn_core_graph_edges[edge, 1])[0][0])
    for edge in range(mx_cmp_conn_pb_graph_m):
        mx_cmp_conn_pb_graph_edges[edge, 0] = int(
            np.where(mx_cmp_conn_graph_nodes == mx_cmp_conn_pb_graph_edges[edge, 0])[0][0])
        mx_cmp_conn_pb_graph_edges[edge, 1] = int(
            np.where(mx_cmp_conn_graph_nodes == mx_cmp_conn_pb_graph_edges[edge, 1])[0][0])
    
    # Save fundamental graph constituents
    np.savetxt(mx_cmp_conn_n_filename, [mx_cmp_conn_graph_n], fmt="%d")
    np.savetxt(mx_cmp_core_node_type_filename, mx_cmp_core_node_type, fmt="%d")
    np.savetxt(
        mx_cmp_conn_core_edges_filename, mx_cmp_conn_core_graph_edges, fmt="%d")
    np.savetxt(
        mx_cmp_conn_pb_edges_filename, mx_cmp_conn_pb_graph_edges, fmt="%d")
    
    # Save the core node x- and y-coordinates
    np.savetxt(mx_cmp_core_x_filename, mx_cmp_core_x)
    np.savetxt(mx_cmp_core_y_filename, mx_cmp_core_y)

def aelp_dim_3_network_topology_initialization(
        network: str,
        L: float,
        core_x: np.ndarray,
        core_y: np.ndarray,
        core_z: np.ndarray,
        b: float,
        xi: float,
        k: int,
        n: int,
        nu: int,
        max_try: int,
        filename_prefix: str) -> None:
    """Network topology initialization procedure for three-dimensional
    artificial end-linked polymer networks.

    This function initializes and saves the topology of a
    three-dimensional artificial end-linked polymer network via a
    modified Gusev-Hanson procedure.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, either "auelp" or "apelp" are applicable
        (corresponding to artificial uniform end-linked polymer networks
        ("auelp") or artificial polydisperse end-linked polymer networks
        ("apelp")).
        L (float): Simulation box size.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
        core_z (np.ndarray): z-coordinates of the core cross-linkers.
        b (float): Chain segment and/or cross-linker diameter.
        xi (float): Chain-to-cross-link connection probability.
        k (int): Maximum cross-linker degree/functionality; either 3, 4,
        5, 6, 7, or 8.
        n (int): Number of core cross-linkers.
        nu (int): (Average) Number of segments per chain.
        max_try (int): Maximum number of dangling chain instantiation
        attempts.
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Network topology initialization procedure is only applicable for
    # artificial end-linked polymer networks. Exit if a different type
    # of network is passed.
    if network != "auelp":
        if network != "apelp":
            error_str = (
                "Network topology initialization procedure is only "
                + "applicable for artificial end-linked polymer "
                + "networks. This calculation will only proceed if "
                + "network = ``auelp'' or network = ``apelp''."
            )
            sys.exit(error_str)
    
    # Generate filenames
    mx_cmp_conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    mx_cmp_core_node_type_filename = (
        filename_prefix + "-core_node_type" + ".dat"
    )
    mx_cmp_conn_core_edges_filename = (
        filename_prefix + "-conn_core_edges" + ".dat"
    )
    mx_cmp_conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    mx_cmp_core_x_filename = filename_prefix + "-core_x" + ".dat"
    mx_cmp_core_y_filename = filename_prefix + "-core_y" + ".dat"
    mx_cmp_core_z_filename = filename_prefix + "-core_z" + ".dat"
    
    # Calculate the stoichiometric number of chains
    m = m_arg_stoich_func(n, k)

    # As a fail-safe check, force int-valued parameters to be ints
    k = int(np.floor(k))
    n = int(np.floor(n))
    nu = int(np.floor(nu))
    max_try = int(np.floor(max_try))
    m = int(np.floor(m))

    # Core cross-linker nodes
    core_nodes = np.arange(n, dtype=int)

    # Identify core nodes as cross-linkers
    core_node_type = np.ones(n, dtype=int)

    # Copy the core_x, core_y, and core_z np.ndarrays as the first n
    # entries in the tessellated cross-linker x-, y-, and z-coordinate
    # np.ndarrays
    tsslltd_core_x = core_x.copy()
    tsslltd_core_y = core_y.copy()
    tsslltd_core_z = core_z.copy()

    # Three-dimensional tessellation protocol
    dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)
        
    # Use three-dimensional tessellation protocol to tessellate the core
    # cross-linkers
    for tsslltn in range(dim_3_tsslltn_num):
        x_tsslltn = dim_3_tsslltn[tsslltn, 0]
        y_tsslltn = dim_3_tsslltn[tsslltn, 1]
        z_tsslltn = dim_3_tsslltn[tsslltn, 2]
        # Skip the (hold, hold, hold) tessellation call because the core
        # cross-linkers are being tessellated about themselves
        if (x_tsslltn == 0) and (y_tsslltn == 0) and (z_tsslltn == 0): continue
        else:
            # x-, y-, and z-coordinates from three-dimensional
            # tessellation protocol
            core_tsslltn_x, core_tsslltn_y, core_tsslltn_z = dim_3_tessellation(
                L, core_x, core_y, core_z, x_tsslltn, y_tsslltn, z_tsslltn)
            # Concatenate the tessellated x-, y-, and z-coordinates
            tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
            tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
            tsslltd_core_z = np.concatenate((tsslltd_core_z, core_tsslltn_z))

    del core_tsslltn_x, core_tsslltn_y, core_tsslltn_z

    # Construct the pb2core_nodes np.ndarray such that
    # pb2core_nodes[core_pb_node] = core_node
    pb2core_nodes = np.tile(core_nodes, dim_3_tsslltn_num)

    # Maximal core cross-linker sampling neighborhood radius as per the
    # minimum image criterion 
    r_mic = L / 2.
    # Polymer chain contour length
    l_cntr = nu * b
    # Determine the core cross-linker sampling neighborhood radius
    r_nghbrhd = np.minimum(r_mic, l_cntr)

    # Finely discretize the core cross-linker sampling neighborhood
    # radius
    r_nghbrhd_chns = np.linspace(0, r_nghbrhd, 10001)
    # Depending on the type of artificial end-linked polymer network,
    # calculate the polymer chain probability distribution
    p_nghbrhd_chns = np.asarray([])
    if network == "auelp":
        p_nghbrhd_chns = p_gaussian_cnfrmtn_func(b, nu, r_nghbrhd_chns)
    elif network == "apelp":
        p_nghbrhd_chns = p_rel_net_r_arr_gaussian_cnfrmtn_func(
            b, nu, r_nghbrhd_chns)
    # Normalize the polymer chain probability distribution
    p_nghbrhd_chns /= np.sum(p_nghbrhd_chns, dtype=float)

    # Initialize core cross-linker sampling neighborhood list
    core_nodes_nghbrhd_init = []
    r_core_nodes_nghbrhd_init = []

    # Determine core cross-linker sampling neighborhood
    for core_node in np.nditer(core_nodes):
        core_node = int(core_node)
        # x-, y-, and z-coordinates of core cross-linker
        core_node_x = tsslltd_core_x[core_node]
        core_node_y = tsslltd_core_y[core_node]
        core_node_z = tsslltd_core_z[core_node]
        # Core cross-linker position
        core_node_pstn = np.asarray(
            [
                core_node_x,
                core_node_y,
                core_node_z
            ]
        )
        # Downselect the local square neighborhood about the core node
        # defined by r_nghbrhd. Start by gathering the tessellated
        # cross-linker nodes that meet this criterion in each x-, y-,
        # and z-coordinate.
        lcl_nghbr_x_lb = core_node_x - r_nghbrhd
        lcl_nghbr_x_ub = core_node_x + r_nghbrhd
        lcl_nghbr_y_lb = core_node_y - r_nghbrhd
        lcl_nghbr_y_ub = core_node_y + r_nghbrhd
        lcl_nghbr_z_lb = core_node_z - r_nghbrhd
        lcl_nghbr_z_ub = core_node_z + r_nghbrhd
        psbl_lcl_nghbr_x_nodes = (
            np.where(np.logical_and(tsslltd_core_x>=lcl_nghbr_x_lb, tsslltd_core_x<=lcl_nghbr_x_ub))[0]
        )
        psbl_lcl_nghbr_y_nodes = (
            np.where(np.logical_and(tsslltd_core_y>=lcl_nghbr_y_lb, tsslltd_core_y<=lcl_nghbr_y_ub))[0]
        )
        psbl_lcl_nghbr_z_nodes = (
            np.where(np.logical_and(tsslltd_core_z>=lcl_nghbr_z_lb, tsslltd_core_z<=lcl_nghbr_z_ub))[0]
        )
        # Gather the nodes from each x-, y-, and z-coordinate together
        # to assess all possible local cross-linker neighbors. Retain
        # unique possible local cross-linker neighbor nodes, and the
        # number of times each such node appears.
        psbl_lcl_nghbr_nodes, psbl_lcl_nghbr_nodes_counts = np.unique(
            np.concatenate((psbl_lcl_nghbr_x_nodes, psbl_lcl_nghbr_y_nodes, psbl_lcl_nghbr_z_nodes), dtype=int),
            return_counts=True)
        # The true local cross-linker neighbor nodes are those who
        # appear thrics in the possible cross-linker neighbor node array
        # -- equal to the network dimensionality
        lcl_nghbr_nodes_indcs = np.where(psbl_lcl_nghbr_nodes_counts == 3)[0]
        lcl_nghbr_node_num = np.shape(lcl_nghbr_nodes_indcs)[0]
        # Further downselect to the neighborhood of tessellated
        # cross-linker nodes that are a distance of r_nghbrhd away from
        # the core cross-linker node
        if lcl_nghbr_node_num > 0:
            # Gather the indices of the local cross-linker neighbors
            lcl_nghbr_nodes = psbl_lcl_nghbr_nodes[lcl_nghbr_nodes_indcs]
            # Extract local cross-linker neighbor x-, y-, and
            # z-coordinates
            lcl_nghbr_x = tsslltd_core_x[lcl_nghbr_nodes]
            lcl_nghbr_y = tsslltd_core_y[lcl_nghbr_nodes]
            lcl_nghbr_z = tsslltd_core_z[lcl_nghbr_nodes]
            # Calculate the distance between the core node and the local
            # cross-linker neighbor nodes
            r_lcl_nghbr_nodes = np.empty(lcl_nghbr_node_num)
            for lcl_nghbr_node_indx in range(lcl_nghbr_node_num):
                lcl_nghbr_pstn = np.asarray(
                    [
                        lcl_nghbr_x[lcl_nghbr_node_indx],
                        lcl_nghbr_y[lcl_nghbr_node_indx],
                        lcl_nghbr_z[lcl_nghbr_node_indx]
                    ]
                )
                r_lcl_nghbr_nodes[lcl_nghbr_node_indx] = (
                    np.linalg.norm(core_node_pstn-lcl_nghbr_pstn)
                )
            # The true cross-linker neighbor nodes are those whose
            # distance to the core node is less than r_nghbrhd
            nghbr_nodes_indcs = np.where(r_lcl_nghbr_nodes <= r_nghbrhd)[0]
            nghbr_nodes = lcl_nghbr_nodes[nghbr_nodes_indcs]
            r_nghbr_nodes = r_lcl_nghbr_nodes[nghbr_nodes_indcs]
            # Add the cross-linker neighbor nodes array to the core
            # cross-linker sampling neighborhood list
            core_nodes_nghbrhd_init.append(nghbr_nodes)
            r_core_nodes_nghbrhd_init.append(r_nghbr_nodes)
        else:
            core_nodes_nghbrhd_init.append(np.asarray([]))
            r_core_nodes_nghbrhd_init.append(np.asarray([]))
    
    # Retain unique nodes from the core and periodic boundary
    # cross-linkers in the core cross-linker sampling neighborhood list
    core_pb_nodes = np.unique(
        np.concatenate(
            tuple(nghbrhd for nghbrhd in core_nodes_nghbrhd_init), dtype=int))

    # Extract the core and periodic boundary cross-linker x-, y-, and
    # z-coordinates using the corresponding node numbers
    core_pb_x = tsslltd_core_x[core_pb_nodes].copy()
    core_pb_y = tsslltd_core_y[core_pb_nodes].copy()
    core_pb_z = tsslltd_core_z[core_pb_nodes].copy()

    del tsslltd_core_x, tsslltd_core_y, tsslltd_core_z

    # Extract the core and periodic boundary cross-linker nodes in the
    # pb2core_nodes np.ndarray
    pb2core_nodes = pb2core_nodes[core_pb_nodes].copy()

    # Construct the core2pb_nodes list
    core2pb_nodes = core2pb_nodes_func(core_nodes, pb2core_nodes)

    # Refactor the node numbers in the core cross-linker sampling
    # neighborhood list
    for core_node in np.nditer(core_nodes):
        core_node = int(core_node)
        nghbr_nodes = core_nodes_nghbrhd_init[core_node]
        for nghbr_node_indx in range(np.shape(nghbr_nodes)[0]):
            nghbr_nodes[nghbr_node_indx] = int(
                np.where(core_pb_nodes == nghbr_nodes[nghbr_node_indx])[0][0])
        core_nodes_nghbrhd_init[core_node] = nghbr_nodes

    # Initialize tracker for failure of dangling chain creation
    dnglng_chn_fail = False

    # Initialize random number generator
    rng = np.random.default_rng()

    # Initialize dangling chain free end node x-, y-, and z-coordinates
    core_dnglng_chns_x = []
    core_dnglng_chns_y = []
    core_dnglng_chns_z = []

    # Initialize edge lists
    conn_core_edges = []
    conn_pb_edges = []
    conn_core_dnglng_edges = []
    conn_pb_dnglng_edges = []

    # Initialize dangling chain node and edge numbers
    core_dnglng_m = 0
    pb_dnglng_m = 0
    dnglng_n = -1

    # Network topology initialization
    while True:
        # Initialize anti-degree of the core cross-linker nodes
        core_nodes_anti_k = k*np.ones(n, dtype=int)
        # Initialize core cross-linker node active sites
        core_nodes_active_sites = np.repeat(core_nodes, k)
        core_nodes_active_sites_num = np.shape(core_nodes_active_sites)[0]

        # Initialize core cross-linker sampling neighborhood list
        core_nodes_nghbrhd = core_nodes_nghbrhd_init.copy()
        r_core_nodes_nghbrhd = r_core_nodes_nghbrhd_init.copy()
        
        # Initialize cross-linker node x-, y-, and z-coordinates for
        # dangling chains
        core_pb_dnglng_chns_x = core_pb_x.copy()
        core_pb_dnglng_chns_y = core_pb_y.copy()
        core_pb_dnglng_chns_z = core_pb_z.copy()

        # Initialize dangling chain free end node x-, y-, and
        # z-coordinates
        core_dnglng_chns_x = []
        core_dnglng_chns_y = []
        core_dnglng_chns_z = []

        # Initialize edge array and lists
        edges = np.full((m, 2), np.inf)
        conn_core_edges = []
        conn_pb_edges = []
        conn_core_dnglng_edges = []
        conn_pb_dnglng_edges = []

        # Initialize dangling chain node and edge numbers
        core_dnglng_m = 0
        pb_dnglng_m = 0
        dnglng_n = -1

        # Initialize node number integer constants
        core_node = 0
        nghbr_node = 0

        # Initialize and randomly shuffle chain end numbers
        chn_ends = np.arange(2*m, dtype=int)
        rng.shuffle(chn_ends)

        # Initialize network topology on a chain end-by-chain end basis
        for chn_end in np.nditer(chn_ends):
            # Chain end is a free chain end
            if rng.random() > xi: continue
            else:
                # Identify the chain and its ends
                chn_end = int(chn_end)
                chn = int(np.floor(chn_end/2))
                chn_end_0 = int(chn_end%2)
                chn_end_1 = 1 - chn_end_0
                # If this chain has not yet been instantiated, then
                # randomly select an active site from the core nodes to
                # instantiate the chain
                if (edges[chn, 0] == np.inf) and (edges[chn, 1] == np.inf):
                    core_node_active_site_indx = rng.integers(
                        np.shape(core_nodes_active_sites)[0], dtype=int)
                    core_node = (
                        core_nodes_active_sites[core_node_active_site_indx]
                    )
                    # Instantiate the chain at one end
                    edges[chn, chn_end_0] = core_node
                    # Update inactive sites, anti-degree, active core
                    # cross-linker nodes, and the sampling neighborhood
                    # list for the parent core cross-linker node
                    (core_nodes_active_sites, core_nodes_active_sites_num, 
                     core_nodes_anti_k, core_nodes_nghbrhd, r_core_nodes_nghbrhd) = core_node_update_func(
                         core_node, core_node_active_site_indx,
                         core_nodes_active_sites, core_nodes_active_sites_num,
                         core_nodes_anti_k, core_nodes, core2pb_nodes,
                         core_nodes_nghbrhd, r_core_nodes_nghbrhd)
                # Otherwise, the chain has previously been instantiated
                else:
                    # Extract sampling neighborhood about the
                    # previously-instantiated chain end
                    core_node = int(edges[chn, chn_end_1])
                    nghbr_nodes = core_nodes_nghbrhd[core_node]
                    nghbr_nodes_num = np.shape(nghbr_nodes)[0]
                    r_nghbr_nodes = r_core_nodes_nghbrhd[core_node]
                    # Check if the sampling neighborhood is empty. If
                    # so, then the chain is a dangling chain.
                    if nghbr_nodes_num == 0:
                        continue
                    else:
                        # Check if there is only one neighbor in the
                        # sampling neighborhood
                        if nghbr_nodes_num == 1:
                            nghbr_node = nghbr_nodes[0]
                        else:
                            # Calculate and normalize weighting factors
                            p_nghbrhd = np.asarray([])
                            if network == "auelp":
                                p_nghbrhd = p_gaussian_cnfrmtn_func(
                                    b, nu, r_nghbr_nodes)
                            elif network == "apelp":
                                p_nghbrhd = np.empty(nghbr_nodes_num)
                                for nghbr_node_indx in range(nghbr_nodes_num):
                                    r_nghbr_node = (
                                        r_nghbr_nodes[nghbr_node_indx]
                                    )
                                    if r_nghbr_node == 0.0:
                                        p_nghbrhd[nghbr_node_indx] = 0.0
                                    else:
                                        # Linear interpolation
                                        r_nghbrhd_chns_diff = (
                                            r_nghbrhd_chns - r_nghbr_node
                                        )
                                        r_nghbrhd_chns_diff[r_nghbrhd_chns_diff < 0] = (
                                            np.inf
                                        )
                                        indx_right = np.argmin(
                                            r_nghbrhd_chns_diff)
                                        r_right = r_nghbrhd_chns[indx_right]
                                        p_right = p_nghbrhd_chns[indx_right]
                                        indx_left = indx_right - 1
                                        r_left = r_nghbrhd_chns[indx_left]
                                        p_left = p_nghbrhd_chns[indx_left]
                                        p_nghbrhd[nghbr_node_indx] = (
                                            p_left + (r_nghbr_node-r_left)
                                            * (p_right-p_left) / (r_right-r_left)
                                        )
                            p_nghbrhd /= np.sum(p_nghbrhd, dtype=float)
                            # Randomly select a neighbor cross-linker
                            # node to host the other end of the chain
                            nghbr_node = int(
                                rng.choice(nghbr_nodes, size=None, p=p_nghbrhd))
                        # Instantiate the other end of the chain
                        edges[chn, chn_end_0] = nghbr_node
                        core_node = int(pb2core_nodes[nghbr_node])
                        core_node_active_site_indx = int(
                            np.where(core_nodes_active_sites == core_node)[0][0])
                        # Update inactive sites, anti-degree, active core
                        # cross-linker nodes, and the sampling neighborhood list
                        # for the neighbor core cross-linker node
                        (core_nodes_active_sites, core_nodes_active_sites_num,
                         core_nodes_anti_k, core_nodes_nghbrhd,
                         r_core_nodes_nghbrhd) = core_node_update_func(
                            core_node, core_node_active_site_indx,
                            core_nodes_active_sites, core_nodes_active_sites_num,
                            core_nodes_anti_k, core_nodes, core2pb_nodes,
                            core_nodes_nghbrhd, r_core_nodes_nghbrhd)
        
        # Post-process the edges array (that was populated during the
        # network topology initialization) on a chain-by-chain basis
        # into edges lists
        for chn in range(m):
            # Chain is a core edge
            if (edges[chn, 0] < n) and (edges[chn, 1] < n):
                conn_core_edges.append(
                    (int(edges[chn, 0]), int(edges[chn, 1])))
            # Chain is a periodic edge
            elif (edges[chn, 0] < n) and (n <= edges[chn, 1] < np.inf):
                conn_pb_edges.append(
                    (int(edges[chn, 0]), int(pb2core_nodes[int(edges[chn, 1])])))
            elif (n <= edges[chn, 0] < np.inf) and (edges[chn, 1] < n):
                conn_pb_edges.append(
                    (int(pb2core_nodes[int(edges[chn, 0])]), int(edges[chn, 1])))
            # Chain is a dangling chain
            elif (edges[chn, 0] < n) and (edges[chn, 1] == np.inf):
                (dnglng_chn_fail, dnglng_n, core_dnglng_m, pb_dnglng_m,
                 conn_core_dnglng_edges, conn_pb_dnglng_edges,
                 core_dnglng_chns_x, core_dnglng_chns_y, core_dnglng_chns_z,
                 core_pb_dnglng_chns_x, core_pb_dnglng_chns_y,
                 core_pb_dnglng_chns_z) = dim_3_dangling_chains_update_func(
                     int(edges[chn, 0]), max_try, rng, r_nghbrhd_chns,
                     p_nghbrhd_chns, b, L, dnglng_chn_fail, dnglng_n,
                     core_dnglng_m, pb_dnglng_m, conn_core_dnglng_edges,
                     conn_pb_dnglng_edges, core_dnglng_chns_x,
                     core_dnglng_chns_y, core_dnglng_chns_z,
                     core_pb_dnglng_chns_x, core_pb_dnglng_chns_y,
                     core_pb_dnglng_chns_z)
                # Break if a dangling chain failed to be instantiated
                if dnglng_chn_fail == True: break
            elif (edges[chn, 0] == np.inf) and (edges[chn, 1] < n):
                (dnglng_chn_fail, dnglng_n, core_dnglng_m, pb_dnglng_m,
                 conn_core_dnglng_edges, conn_pb_dnglng_edges,
                 core_dnglng_chns_x, core_dnglng_chns_y, core_dnglng_chns_z,
                 core_pb_dnglng_chns_x, core_pb_dnglng_chns_y,
                 core_pb_dnglng_chns_z) = dim_3_dangling_chains_update_func(
                     int(edges[chn, 1]), max_try, rng, r_nghbrhd_chns,
                     p_nghbrhd_chns, b, L, dnglng_chn_fail, dnglng_n,
                     core_dnglng_m, pb_dnglng_m, conn_core_dnglng_edges,
                     conn_pb_dnglng_edges, core_dnglng_chns_x,
                     core_dnglng_chns_y, core_dnglng_chns_z,
                     core_pb_dnglng_chns_x, core_pb_dnglng_chns_y,
                     core_pb_dnglng_chns_z)
                # Break if a dangling chain failed to be instantiated
                if dnglng_chn_fail == True: break
            # Chain is a free chain
            elif (edges[chn, 0] == np.inf) and (edges[chn, 1] == np.inf):
                continue

        # Restart the network topology initialization protocol if a
        # dangling chain failed to be instantiated
        if dnglng_chn_fail == True: continue
        
        del core_nodes_anti_k, core_nodes_active_sites
        del core_pb_dnglng_chns_x, core_pb_dnglng_chns_y, core_pb_dnglng_chns_z
        del edges, chn_ends
        # Break out of acceptable initialized topology 
        break

    del core_pb_nodes, pb2core_nodes, core2pb_nodes
    del core_pb_x, core_pb_y, core_pb_z
    del core_nodes_nghbrhd_init, core_nodes_nghbrhd
    del r_core_nodes_nghbrhd_init, r_core_nodes_nghbrhd
    del r_nghbrhd_chns, p_nghbrhd_chns

    # Refactor edge lists to np.ndarrays
    conn_core_edges = np.asarray(conn_core_edges, dtype=int)
    conn_pb_edges = np.asarray(conn_pb_edges, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Create nx.MultiGraphs
    conn_core_graph = nx.MultiGraph()
    conn_pb_graph = nx.MultiGraph()
    conn_graph = nx.MultiGraph()
    
    # Recalibrate number of dangling chains
    dnglng_n += 1

    # No dangling chains were added to the network
    if dnglng_n == 0:
        conn_core_graph = add_nodes_from_numpy_array(
            conn_core_graph, core_nodes)
        conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
        conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
        
        conn_core_graph = add_edges_from_numpy_array(
            conn_core_graph, conn_core_edges)
        conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)
        conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)
    # Only core dangling chains were added to the network
    elif core_dnglng_m > 0 and pb_dnglng_m == 0:
        # Update core dangling chain node numbers
        for edge in conn_core_dnglng_edges: edge[1] += n
        # Update edge lists
        conn_core_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_core_dnglng_edges), dtype=int)
        conn_core_edges = np.vstack(
            (conn_core_edges, conn_core_dnglng_edges), dtype=int)
        conn_edges = np.vstack((conn_edges, conn_core_dnglng_edges), dtype=int)
    # Only periodic boundary dangling chains were added to the network
    elif core_dnglng_m == 0 and pb_dnglng_m > 0:
        # Update periodic boundary dangling chain node numbers
        for edge in conn_pb_dnglng_edges: edge[1] += n
        # Update edge lists
        conn_pb_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_pb_dnglng_edges), dtype=int)
        conn_pb_edges = np.vstack(
            (conn_pb_edges, conn_pb_dnglng_edges), dtype=int)
        conn_edges = np.vstack((conn_edges, conn_pb_dnglng_edges), dtype=int)
    # Both core and periodic boundary dangling chains were added to the
    # network
    else:
        # Update core and periodic boundary dangling chain node numbers
        for edge in conn_core_dnglng_edges: edge[1] += n
        for edge in conn_pb_dnglng_edges: edge[1] += n
        # Update edge lists
        conn_core_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_core_dnglng_edges), dtype=int)
        conn_pb_dnglng_edges = np.asarray(
            list(tuple(edge) for edge in conn_pb_dnglng_edges), dtype=int)
        conn_core_edges = np.vstack(
            (conn_core_edges, conn_core_dnglng_edges), dtype=int)
        conn_pb_edges = np.vstack(
            (conn_pb_edges, conn_pb_dnglng_edges), dtype=int)
        conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)
    # Update core_node_type correspondingly to end-linked polymer
    # network code
    core_node_type = np.concatenate(
        (core_node_type, np.repeat(3, dnglng_n)), dtype=int)
    # Add core coordinates from dangling chains
    core_x = np.concatenate((core_x, np.asarray(core_dnglng_chns_x)))
    core_y = np.concatenate((core_y, np.asarray(core_dnglng_chns_y)))
    core_z = np.concatenate((core_z, np.asarray(core_dnglng_chns_z)))
    # Update core_nodes
    core_nodes = np.arange(n+dnglng_n, dtype=int)
    # Update nx.MultiGraph nodes
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    # Update nx.MultiGraph edges
    conn_core_graph = add_edges_from_numpy_array(
        conn_core_graph, conn_core_edges)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Isolate largest/maximum connected component in a nodewise fashion
    mx_cmp_conn_graph_nodes = max(nx.connected_components(conn_graph), key=len)
    # Extract largest/maximum connected component subgraphs
    mx_cmp_conn_core_graph = (
        conn_core_graph.subgraph(mx_cmp_conn_graph_nodes).copy()
    )
    mx_cmp_conn_pb_graph = (
        conn_pb_graph.subgraph(mx_cmp_conn_graph_nodes).copy()
    )
    # Extract edges
    mx_cmp_conn_core_graph_edges = np.asarray(
        list(mx_cmp_conn_core_graph.edges()), dtype=int)
    mx_cmp_conn_pb_graph_edges = np.asarray(
        list(mx_cmp_conn_pb_graph.edges()), dtype=int)
    # Number of edges in the largest/maximum connected component
    mx_cmp_conn_core_graph_m = np.shape(mx_cmp_conn_core_graph_edges)[0]
    mx_cmp_conn_pb_graph_m = np.shape(mx_cmp_conn_pb_graph_edges)[0]
    # Nodes from the largest/maximum connected component, sorted in
    # ascending order
    mx_cmp_conn_graph_nodes = np.sort(
        np.fromiter(mx_cmp_conn_graph_nodes, dtype=int))
    # Number of nodes in the largest/maximum connected component
    mx_cmp_conn_graph_n = np.shape(mx_cmp_conn_graph_nodes)[0]

    # Isolate core_node_type for the largest/maximum connected component
    mx_cmp_core_node_type = core_node_type[mx_cmp_conn_graph_nodes]
    # Isolate the cross-linker coordinates for the largest/maximum
    # connected component
    mx_cmp_core_x = core_x[mx_cmp_conn_graph_nodes]
    mx_cmp_core_y = core_y[mx_cmp_conn_graph_nodes]
    mx_cmp_core_z = core_z[mx_cmp_conn_graph_nodes]

    # Update all original node values with updated node values
    for edge in range(mx_cmp_conn_core_graph_m):
        mx_cmp_conn_core_graph_edges[edge, 0] = int(
            np.where(mx_cmp_conn_graph_nodes == mx_cmp_conn_core_graph_edges[edge, 0])[0][0])
        mx_cmp_conn_core_graph_edges[edge, 1] = int(
            np.where(mx_cmp_conn_graph_nodes == mx_cmp_conn_core_graph_edges[edge, 1])[0][0])
    for edge in range(mx_cmp_conn_pb_graph_m):
        mx_cmp_conn_pb_graph_edges[edge, 0] = int(
            np.where(mx_cmp_conn_graph_nodes == mx_cmp_conn_pb_graph_edges[edge, 0])[0][0])
        mx_cmp_conn_pb_graph_edges[edge, 1] = int(
            np.where(mx_cmp_conn_graph_nodes == mx_cmp_conn_pb_graph_edges[edge, 1])[0][0])
    
    # Save fundamental graph constituents
    np.savetxt(mx_cmp_conn_n_filename, [mx_cmp_conn_graph_n], fmt="%d")
    np.savetxt(mx_cmp_core_node_type_filename, mx_cmp_core_node_type, fmt="%d")
    np.savetxt(
        mx_cmp_conn_core_edges_filename, mx_cmp_conn_core_graph_edges, fmt="%d")
    np.savetxt(
        mx_cmp_conn_pb_edges_filename, mx_cmp_conn_pb_graph_edges, fmt="%d")
    
    # Save the core node x-, y- and z-coordinates
    np.savetxt(mx_cmp_core_x_filename, mx_cmp_core_x)
    np.savetxt(mx_cmp_core_y_filename, mx_cmp_core_y)
    np.savetxt(mx_cmp_core_z_filename, mx_cmp_core_z)

def aelp_network_topology_initialization(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        b: float,
        xi: float,
        k: int,
        n: int,
        nu: int,
        config: int,
        max_try: int,) -> None:
    """Network topology initialization procedure for artificial
    end-linked polymer networks.

    This function loads the simulation box size and the core
    cross-linker coordinates previously generated by the
    node_seeding() function. Then, depending on the network
    dimensionality, this function calls upon a corresponding helper
    function to initialize the artificial end-linked polymer network
    topology.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, either "auelp" or "apelp" are applicable
        (corresponding to artificial uniform end-linked polymer networks
        ("auelp") or artificial polydisperse end-linked polymer networks
        ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular
        scheme used to generate the positions of the core nodes;
        either "random", "prhd", "pdhu", or "lammps" (corresponding to
        the random node placement procedure ("random"), periodic random
        hard disk node placement procedure ("prhd"), periodic disordered
        hyperuniform node placement procedure ("pdhu"), or nodes
        randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Chain segment and/or cross-linker diameter.
        xi (float): Chain-to-cross-link connection probability.
        k (int): Maximum cross-linker degree/functionality; either 3, 4,
        5, 6, 7, or 8.
        n (int): Number of core cross-linkers.
        nu (int): (Average) Number of segments per chain.
        config (int): Configuration number.
        max_try (int): Maximum number of dangling chain instantiation
        attempts.
    
    """
    # Network topology initialization procedure is only applicable for
    # artificial end-linked polymer networks. Exit if a different type
    # of network is passed.
    if network != "auelp":
        if network != "apelp":
            error_str = (
                "Network topology initialization procedure is only "
                + "applicable for artificial end-linked polymer "
                + "networks. This calculation will only proceed if "
                + "network = ``auelp'' or network = ``apelp''."
            )
            sys.exit(error_str)
    
    # Generate filename prefix
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    
    # Load L
    L = np.loadtxt(L_filename)

    # Append configuration number to filename prefix
    filename_prefix = filename_prefix + f"C{config:d}"

    # Generate config filename
    config_filename = filename_prefix + ".config"

    # Call appropriate helper function to initialize network topology
    if (scheme == "random") or (scheme == "prhd") or (scheme == "pdhu"):
        # Load core cross-linker coordinates
        coords = np.loadtxt(config_filename)
    elif scheme == "lammps":
        skiprows_num = 15
        # Load core cross-linker coordinates
        coords = np.loadtxt(config_filename, skiprows=skiprows_num, max_rows=n)
    # Actual number of core cross-linkers
    n = np.shape(coords)[0]
    # Separate core cross-linkers x- and y-coordinates
    x = coords[:, 0].copy()
    y = coords[:, 1].copy()
    if dim == 2:
        del coords
        aelp_dim_2_network_topology_initialization(
            network, L, x, y, b, xi, k, n, nu, max_try, filename_prefix)
    elif dim == 3:
        # Separate core cross-linker z-coordinates
        z = coords[:, 2].copy()
        del coords
        aelp_dim_3_network_topology_initialization(
            network, L, x, y, z, b, xi, k, n, nu, max_try, filename_prefix)

def elastically_effective_graph(graph):
    """Elastically-effective graph.

    This function returns the portion of a given graph that corresponds
    to the elastically-effective network in the graph, i.e., the given
    graph less self-loops, dangling chains, or pending chains.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph
        or nx.MultiGraph.
    
    Returns:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph
        or nx.MultiGraph. This graph corresponds to the
        elastically-effective network of the given graph.
    
    """
    # Self-loop pruning procedure
    if nx.number_of_selfloops(graph) > 0:
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    
    # If the graph is of type nx.MultiGraph, then address multi-edges by
    # pruning redundant edges
    graph_edges = np.asarray([])
    graph_edges_counts = np.asarray([])
    if graph.is_multigraph():
        # Gather edges and edges counts
        graph_edges, graph_edges_counts = (
            np.unique(
                np.sort(np.asarray(list(graph.edges()), dtype=int), axis=1),
                return_counts=True, axis=0)
        )

        # Address multi-edges by pruning redundant edges
        if np.any(graph_edges_counts > 1):
            # Extract multi-edges
            multiedges = np.where(graph_edges_counts > 1)[0]
            for multiedge in np.nditer(multiedges):
                multiedge = int(multiedge)
                # Number of edges in the multiedge
                edge_num = graph_edges_counts[multiedge]
                # Remove redundant edges in the multiedge (thereby
                # leaving one edge)
                graph.remove_edges_from(
                    list((int(graph_edges[multiedge, 0]), int(graph_edges[multiedge, 1])) for _ in range(edge_num-1)))
    
    # Degree of nodes
    graph_nodes_k = np.asarray(list(graph.degree()), dtype=int)
    graph_k = graph_nodes_k[:, 1]

    # Dangling edge pruning procedure
    while np.any(graph_k == 1):
        # Extract dangling nodes (with k = 1)
        dangling_node_indcs = np.where(graph_k == 1)[0]
        for dangling_node_indx in np.nditer(dangling_node_indcs):
            dangling_node_indx = int(dangling_node_indx)
            # Dangling node (with k = 1)
            dangling_node = int(graph_nodes_k[dangling_node_indx, 0])
            # Neighbor of dangling node (with k = 1)
            dangling_node_nghbr_arr = np.asarray(
                list(graph.neighbors(dangling_node)), dtype=int)
            # Check to see if the dangling edge was previously removed
            if np.shape(dangling_node_nghbr_arr)[0] == 0: continue
            else:
                # Remove dangling edge
                graph.remove_edge(dangling_node, int(dangling_node_nghbr_arr[0]))
        # Update degree of nodes
        graph_nodes_k = np.asarray(list(graph.degree()), dtype=int)
        graph_k = graph_nodes_k[:, 1]
    
    # Isolate node pruning procedure
    if nx.number_of_isolates(graph) > 0:
        graph.remove_nodes_from(list(nx.isolates(graph)))
    
    # If the graph is of type nx.MultiGraph, then address multi-edges by
    # adding back elastically-effective redundant edges
    if graph.is_multigraph():    
        # Address multi-edges by adding back elastically-effective
        # redundant edges
        if np.any(graph_edges_counts > 1):
            # Extract multi-edges
            multiedges = np.where(graph_edges_counts > 1)[0]
            for multiedge in np.nditer(multiedges):
                multiedge = int(multiedge)
                # Number of edges in the multiedge
                edge_num = graph_edges_counts[multiedge]
                # Multiedge nodes
                node_0 = int(graph_edges[multiedge, 0])
                node_1 = int(graph_edges[multiedge, 1])
                # Add back elastically-effective redundant edges
                if graph.has_edge(node_0, node_1):
                    graph.add_edges_from(
                        list((node_0, node_1) for _ in range(edge_num-1)))

    return graph

def proportion_elastically_effective_nodes(graph) -> float:
    """Proportion of elastically-effective nodes in a given graph.

    This function calculates and returns the proportion of
    elastically-effective nodes in a given graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph
        or nx.MultiGraph.
    
    Returns:
        float: (Undirected) Proportion of elastically-effective nodes.
    
    """
    # Number of nodes involved in the given graph
    graph_n = np.shape(np.unique(np.asarray(list(graph.edges()), dtype=int)))[0]
    # Elastically-effective graph
    ee_graph = elastically_effective_graph(graph)
    # Number of nodes involved in the elastically-effective graph
    ee_graph_n = (
        np.shape(np.unique(np.asarray(list(ee_graph.edges()), dtype=int)))[0]
    )

    return ee_graph_n / graph_n

def proportion_elastically_effective_edges(graph):
    """Proportion of elastically-effective edges in a given graph.

    This function calculates and returns the proportion of
    elastically-effective edges in a given graph.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph
        or nx.MultiGraph.
    
    Returns:
        float: (Undirected) Proportion of elastically-effective edges.
    
    """
    # Number of edges in given graph
    graph_m = len(list(graph.edges()))
    # Elastically-effective graph
    ee_graph = elastically_effective_graph(graph)
    # Number of edges in elastically-effective graph
    ee_graph_m = len(list(ee_graph.edges()))

    return ee_graph_m / graph_m

def k_counts_calculation(graph) -> np.ndarray:
    """Node degree counts.

    This function calculates the node degree counts in an (undirected)
    graph, where the node degree can be between 1 and 8, inclusive.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph
        or nx.MultiGraph.
    
    Returns:
        np.ndarray: Node degree counts, where the node degree can be
        between 1 and 8, inclusive.
    
    """
    # Initialize the node degree counts, where k\in[1, 8]. Thus,
    # k_counts[k-1] = counts, i.e., k_counts[0] = number of nodes with
    # k = 1, k_counts[1] = number of nodes with k = 2, ...,
    # k_counts[7] = number of nodes with k = 8.
    k_counts = np.zeros(8, dtype=int)

    # Calculate number of occurrences for each node degree
    graph_k, graph_k_counts = np.unique(
        np.asarray(list(graph.degree()), dtype=int)[:, 1], return_counts=True)

    # Store the node degree counts
    for k_indx in range(np.shape(graph_k)[0]):
        if graph_k[k_indx] == 0: continue
        else: k_counts[graph_k[k_indx]-1] = graph_k_counts[k_indx]
    
    return k_counts

def swidt_network_k_counts(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        pruning: int,
        elastically_effective: bool) -> None:
    """Node degree counts in spider web-inspired Delaunay-triangulated
    networks.

    This function generates the filename prefix associated with
    fundamental graph constituents for spider web-inspired
    Delaunay-triangulated networks. This function then calls upon a
    corresponding helper function to calculate and save the node degree
    counts.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "swidt" is applicable (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
        elastically_effective (bool): Marker used to indicate if the
        elastically-effective network should be analyzed.
    
    """
    # Node degree counts calculation is only applicable for spider
    # web-inspired Delaunay-triangulated networks. Exit if a different
    # type of network is passed.
    if network != "swidt":
        error_str = (
            "Node degree count calculation is only applicable for the "
            + "spider web-inspired Delaunay-triangulated networks. This "
            + "procedure will only proceed if network = ``swidt''."
        )
        sys.exit(error_str)
    # Generate filenames
    filename_prefix = (
        filename_str(network, date, batch, sample)
        + f"C{config:d}" + f"P{pruning:d}"
    )
    conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    k_counts_filename = filename_prefix + "-k_counts" + ".dat"

    # Load fundamental graph constituents
    core_nodes = np.arange(np.loadtxt(conn_n_filename, dtype=int), dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Create nx.Graph and add nodes before edges
    conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Yield the elastically-effective network graph, if called for
    if elastically_effective:
        k_counts_filename = filename_prefix + "-ee_k_counts" + ".dat"
        conn_graph = elastically_effective_graph(conn_graph)

    # Calculate and save the node degree counts
    np.savetxt(
        k_counts_filename, k_counts_calculation(conn_graph), fmt="%d")

def h_counts_calculation(graph, l_bound: int) -> np.ndarray:
    """Chordless cycle counts.

    This function calculates the chordless cycle counts in an
    (undirected) graph, where the chordless cycle order can be between 1
    and a maximum value, inclusive.

    Args:
        graph: (Undirected) NetworkX graph that can be of type nx.Graph
        or nx.MultiGraph.
        l_bound (int): Maximal chordless cycle order.
    
    Returns:
        np.ndarray: Chordless cycle counts, where the chordless cycle
        order can be between 1 and l_bound, inclusive.
    
    """
    # Import the scipy.special.comb() function
    from scipy.special import comb

    # Initialize the chordless cycle counts, where h\in[1, l_bound].
    # Thus, h_counts[h-1] = counts, i.e., h_counts[0] = number of
    # self-loops, h_counts[1] = number of second-order cycles induced by
    # redundant multi-edges, h_counts[2] = number of third-order cycles, 
    # h_counts[3] = number of fourth-order cycles, ...,
    # h_counts[l_bound-1] = number of l_bound-order cycles.
    h_counts = np.zeros(l_bound, dtype=int)

    # Calculate and store the number of self-loops
    self_loop_num = int(nx.number_of_selfloops(graph))
    h_counts[0] = self_loop_num

    # Self-loop pruning procedure
    if self_loop_num > 0:
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    
    # If the graph is of type nx.MultiGraph, then address multi-edges,
    # prune redundant edges, and convert resulting graph to type
    # nx.Graph
    if graph.is_multigraph():    
        # Gather edges and edges counts
        graph_edges, graph_edges_counts = (
            np.unique(
                np.sort(np.asarray(list(graph.edges()), dtype=int), axis=1),
                return_counts=True, axis=0)
        )
        
        # Address multi-edges by calculating and storing the number of
        # second-order cycles and by pruning redundant edges
        if np.any(graph_edges_counts > 1):
            # Extract multi-edges
            multiedges = np.where(graph_edges_counts > 1)[0]
            for multiedge in np.nditer(multiedges):
                multiedge = int(multiedge)
                # Number of edges in the multiedge
                edge_num = graph_edges_counts[multiedge]
                # Calculate the number of second-order cycles induced by
                # redundant multi-edges
                h_counts[1] += int(comb(edge_num, 2))
                # Remove redundant edges in the multiedge (thereby
                # leaving one edge)
                graph.remove_edges_from(
                    list((int(graph_edges[multiedge, 0]), int(graph_edges[multiedge, 1])) for _ in range(edge_num-1)))
        
        # Convert graph to type nx.Graph
        graph = nx.Graph(graph)

    # Degree of nodes
    graph_nodes_k = np.asarray(list(graph.degree()), dtype=int)
    graph_k = graph_nodes_k[:, 1]

    # Dangling edge pruning procedure
    while np.any(graph_k == 1):
        # Extract dangling nodes (with k = 1)
        dangling_node_indcs = np.where(graph_k == 1)[0]
        for dangling_node_indx in np.nditer(dangling_node_indcs):
            dangling_node_indx = int(dangling_node_indx)
            # Dangling node (with k = 1)
            dangling_node = int(graph_nodes_k[dangling_node_indx, 0])
            # Neighbor of dangling node (with k = 1)
            dangling_node_nghbr_arr = np.asarray(
                list(graph.neighbors(dangling_node)), dtype=int)
            # Check to see if the dangling edge was previously removed
            if np.shape(dangling_node_nghbr_arr)[0] == 0: continue
            else:
                # Remove dangling edge
                graph.remove_edge(dangling_node, int(dangling_node_nghbr_arr[0]))
        # Update degree of nodes
        graph_nodes_k = np.asarray(list(graph.degree()), dtype=int)
        graph_k = graph_nodes_k[:, 1]
    
    # Remove isolate nodes
    if nx.number_of_isolates(graph) > 0:
        graph.remove_nodes_from(list(nx.isolates(graph)))
    
    # Find chordless cycles
    chrdls_cycls = list(nx.chordless_cycles(graph, length_bound=l_bound))
    
    # Calculate number of occurrences for each chordless cycle order
    graph_h, graph_h_counts = np.unique(
        np.asarray(list(len(cycl) for cycl in chrdls_cycls), dtype=int),
        return_counts=True)

    # Store the chordless cycle counts
    for h_indx in range(np.shape(graph_h)[0]):
        if graph_h[h_indx] == 0: continue
        else: h_counts[graph_h[h_indx]-1] = graph_h_counts[h_indx]
    
    return h_counts

def swidt_network_h_counts(
        network: str,
        date: str,
        batch: str,
        sample: int,
        l_bound: int,
        config: int,
        pruning: int) -> None:
    """Chordless cycle counts in spider web-inspired
    Delaunay-triangulated networks.

    This function generates the filename prefix associated with
    fundamental graph constituents for spider web-inspired
    Delaunay-triangulated networks. This function then calls upon a
    corresponding helper function to calculate and save the chordless
    cycle counts.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "swidt" is applicable (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        l_bound (int): Maximal chordless cycle order.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
    
    """
    # Chordless cycle counts calculation is only applicable for spider
    # web-inspired Delaunay-triangulated networks. Exit if a different
    # type of network is passed.
    if network != "swidt":
        error_str = (
            "Chordless cycle count calculation is only applicable for the "
            + "spider web-inspired Delaunay-triangulated networks. This "
            + "procedure will only proceed if network = ``swidt''."
        )
        sys.exit(error_str)
    # Generate filenames
    filename_prefix = (
        filename_str(network, date, batch, sample)
        + f"C{config:d}" + f"P{pruning:d}"
    )
    conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    h_counts_filename = filename_prefix + "-h_counts" + ".dat"

    # Load fundamental graph constituents
    core_nodes = np.arange(np.loadtxt(conn_n_filename, dtype=int), dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Create nx.Graph, load fundamental graph constituents, and add
    # nodes before edges
    conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Calculate and save the chordless cycle counts counts
    np.savetxt(
        h_counts_filename, h_counts_calculation(conn_graph, l_bound), fmt="%d")

def dim_2_core_pb_edge_identification(
        core_node_0_x: float,
        core_node_0_y: float,
        core_node_1_x: float,
        core_node_1_y: float,
        L: float) -> tuple[float, float, float]:
    """Two-dimensional periodic boundary edge and node identification.

    This function uses the minimum image criterion to determine/identify
    the node coordinates of a particular periodic boundary edge in a
    two-dimensional network.

    Args:
        core_node_0_x (float): x-coordinate of the core node in the
        periodic boundary edge.
        core_node_0_y (float): y-coordinate of the core node in the
        periodic boundary edge.
        core_node_1_x (float): x-coordinate of the core node that
        translates/tessellates to the periodic node in the periodic
        boundary edge.
        core_node_1_y (float): y-coordinate of the core node that
        translates/tessellates to the periodic node in the periodic
        boundary edge.
        L (float): Tessellation scaling distance (i.e., simulation box
        size).

    Returns:
        tuple[float, float, float]: x- and y-coordinates of the periodic
        node in the periodic boundary edge, and the length of the
        periodic boundary edge, respectively.
    
    """
    # core_node_0 position
    core_node_0_pstn = np.asarray(
        [
            core_node_0_x,
            core_node_0_y
        ]
    )
    # Two-dimensional tessellation protocol
    dim_2_tsslltn, dim_2_tsslltn_num = tessellation_protocol(2)
    # Use two-dimensional tessellation protocol to tessellate
    # core_node_1
    core_node_1_tsslltn_x, core_node_1_tsslltn_y = (
        dim_2_tessellation_protocol(
            L, core_node_1_x, core_node_1_y, dim_2_tsslltn)
    )
    # Use minimum image/distance criterion to select the correct
    # periodic boundary node and edge corresponding to core_node_1
    l_pb_nodes_1 = np.empty(dim_2_tsslltn_num)
    for pb_node_1 in range(dim_2_tsslltn_num):
        pb_node_1_pstn = np.asarray(
            [
                core_node_1_tsslltn_x[pb_node_1],
                core_node_1_tsslltn_y[pb_node_1]
            ]
        )
        l_pb_nodes_1[pb_node_1] = np.linalg.norm(pb_node_1_pstn-core_node_0_pstn)
    pb_node_1 = np.argmin(l_pb_nodes_1)
    
    return (
        core_node_1_tsslltn_x[pb_node_1], core_node_1_tsslltn_y[pb_node_1],
        l_pb_nodes_1[pb_node_1]
    )

def dim_3_core_pb_edge_identification(
        core_node_0_x: float,
        core_node_0_y: float,
        core_node_0_z: float,
        core_node_1_x: float,
        core_node_1_y: float,
        core_node_1_z: float,
        L: float) -> tuple[float, float, float, float]:
    """Three-dimensional periodic boundary edge and node identification.

    This function uses the minimum image criterion to determine/identify
    the node coordinates of a particular periodic boundary edge in a
    three-dimensional network.

    Args:
        core_node_0_x (float): x-coordinate of the core node in the
        periodic boundary edge.
        core_node_0_y (float): y-coordinate of the core node in the
        periodic boundary edge.
        core_node_0_z (float): z-coordinate of the core node in the
        periodic boundary edge.
        core_node_1_x (float): x-coordinate of the core node that
        translates/tessellates to the periodic node in the periodic
        boundary edge.
        core_node_1_y (float): y-coordinate of the core node that
        translates/tessellates to the periodic node in the periodic
        boundary edge.
        core_node_1_z (float): z-coordinate of the core node that
        translates/tessellates to the periodic node in the periodic
        boundary edge.
        L (float): Tessellation scaling distance (i.e., simulation box
        size).
    
    Returns:
        tuple[float, float, float, float]: x-, y-, and z-coordinates of
        the periodic node in the periodic boundary edge, and the length
        of the periodic boundary edge, respectively.
    
    """
    # core_node_0 position
    core_node_0_pstn = np.asarray(
        [
            core_node_0_x,
            core_node_0_y,
            core_node_0_z
        ]
    )
    # Three-dimensional tessellation protocol
    dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)
    # Use three-dimensional tessellation protocol to tessellate
    # core_node_1
    core_node_1_tsslltn_x, core_node_1_tsslltn_y, core_node_1_tsslltn_z = (
        dim_3_tessellation_protocol(
            L, core_node_1_x, core_node_1_y, core_node_1_z, dim_3_tsslltn)
    )
    # Use minimum image/distance criterion to select the correct
    # periodic boundary node and edge corresponding to core_node_1
    l_pb_nodes_1 = np.empty(dim_3_tsslltn_num)
    for pb_node_1 in range(dim_3_tsslltn_num):
        pb_node_1_pstn = np.asarray(
            [
                core_node_1_tsslltn_x[pb_node_1],
                core_node_1_tsslltn_y[pb_node_1],
                core_node_1_tsslltn_z[pb_node_1]
            ]
        )
        l_pb_nodes_1[pb_node_1] = np.linalg.norm(pb_node_1_pstn-core_node_0_pstn)
    pb_node_1 = np.argmin(l_pb_nodes_1)
    
    return (
        core_node_1_tsslltn_x[pb_node_1], core_node_1_tsslltn_y[pb_node_1],
        core_node_1_tsslltn_z[pb_node_1], l_pb_nodes_1[pb_node_1]
    )

def dim_2_l_edges_calculation(
        conn_core_graph,
        conn_pb_graph,
        conn_graph,
        core_x: np.ndarray,
        core_y: np.ndarray,
        L: float) -> np.ndarray:
    """Edge length calculation for two-dimensional networks.

    This function calculates the length of each core and periodic
    boundary edge in a two-dimensional network. Note that the length of
    each edge is calculated as the true spatial length, not the naive
    length present in the graph.

    Args:
        conn_core_graph: (Undirected) NetworkX graph that can be of
        type nx.Graph or nx.MultiGraph. This graph represents the core
        edges from the graph capturing the periodic connections between
        the core nodes.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core nodes.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core nodes.
        core_x (np.ndarray): x-coordinates of the core nodes.
        core_y (np.ndarray): y-coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box
        size).
    
    Returns:
        np.ndarray: Edge length.
    
    """
    # Gather edges and initialize l_edges np.ndarray
    conn_edges = list(conn_graph.edges())
    conn_m = len(conn_edges)
    l_edges = np.empty(conn_m)

    # Calculate and store the length of each edge
    for edge_indx, edge in enumerate(conn_edges):
        # Node numbers
        core_node_0 = int(edge[0])
        core_node_1 = int(edge[1])

        # Coordinates of each node
        core_node_0_x = core_x[core_node_0]
        core_node_0_y = core_y[core_node_0]
        core_node_1_x = core_x[core_node_1]
        core_node_1_y = core_y[core_node_1]

        # Edge is a core edge
        if conn_core_graph.has_edge(core_node_0, core_node_1):
            # Position of each core node
            core_node_0_pstn = np.asarray(
                [
                    core_node_0_x,
                    core_node_0_y
                ]
            )
            core_node_1_pstn = np.asarray(
                [
                    core_node_1_x,
                    core_node_1_y
                ]
            )
            # Calculate and store core edge length
            l_edges[edge_indx] = np.linalg.norm(core_node_1_pstn-core_node_0_pstn)
        # Edge is a periodic boundary edge
        elif conn_pb_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store periodic boundary edge length
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            l_edges[edge_indx] = l_pb_edge
        else:
            error_str = (
                "The edge in the overall graph was not detected in "
                + "either the core edge graph or the periodic boundary "
                + "edge graph."
            )
            sys.exit(error_str)
        
    return l_edges

def dim_3_l_edges_calculation(
        conn_core_graph,
        conn_pb_graph,
        conn_graph,
        core_x: np.ndarray,
        core_y: np.ndarray,
        core_z: np.ndarray,
        L: float) -> np.ndarray:
    """Edge length calculation for three-dimensional networks.

    This function calculates the length of each core and periodic
    boundary edge in a three-dimensional network. Note that the length
    of each edge is calculated as the true spatial length, not the naive
    length present in the graph.

    Args:
        conn_core_graph: (Undirected) NetworkX graph that can be of
        type nx.Graph or nx.MultiGraph. This graph represents the core
        edges from the graph capturing the periodic connections between
        the core nodes.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core nodes.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core nodes.
        core_x (np.ndarray): x-coordinates of the core nodes.
        core_y (np.ndarray): y-coordinates of the core nodes.
        core_z (np.ndarray): z-coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box
        size).
    
    Returns:
        np.ndarray: Edge length.
    
    """
    # Gather edges and initialize l_edges np.ndarray
    conn_edges = list(conn_graph.edges())
    conn_m = len(conn_edges)
    l_edges = np.empty(conn_m)

    # Calculate and store the length of each edge
    for edge_indx, edge in enumerate(conn_edges):
        # Node numbers
        core_node_0 = int(edge[0])
        core_node_1 = int(edge[1])

        # Coordinates of each node
        core_node_0_x = core_x[core_node_0]
        core_node_0_y = core_y[core_node_0]
        core_node_0_z = core_z[core_node_0]
        core_node_1_x = core_x[core_node_1]
        core_node_1_y = core_y[core_node_1]
        core_node_1_z = core_z[core_node_1]

        # Edge is a core edge
        if conn_core_graph.has_edge(core_node_0, core_node_1):
            # Position of each core node
            core_node_0_pstn = np.asarray(
                [
                    core_node_0_x,
                    core_node_0_y,
                    core_node_0_z
                ]
            )
            core_node_1_pstn = np.asarray(
                [
                    core_node_1_x,
                    core_node_1_y,
                    core_node_1_z
                ]
            )
            # Calculate and store core edge length
            l_edges[edge_indx] = np.linalg.norm(core_node_1_pstn-core_node_0_pstn)
        # Edge is a periodic boundary edge
        elif conn_pb_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store periodic boundary edge length
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            l_edges[edge_indx] = l_pb_edge
        else:
            error_str = (
                "The edge in the overall graph was not detected in "
                + "either the core edge graph or the periodic boundary "
                + "edge graph."
            )
            sys.exit(error_str)
            
    return l_edges

def swidt_network_l_edges(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        config: int,
        pruning: int,
        elastically_effective: bool) -> None:
    """Length of each edge in spider web-inspired Delaunay-triangulated
    networks.

    This function generates filenames associated with fundamental graph
    constituents for spider web-inspired Delaunay-triangulated networks.
    This function then calls upon a corresponding helper function to
    calculate and save the length of each edge.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "swidt" is applicable (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
        elastically_effective (bool): Marker used to indicate if the
        elastically-effective network should be analyzed.
    
    """
    # Edge length calculation is only applicable for spider web-inspired
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "swidt":
        error_str = (
            "Edge length calculation is only applicable for the spider "
            + "web-inspired Delaunay-triangulated networks. This "
            + "procedure will only proceed if network = ``swidt''."
        )
        sys.exit(error_str)
    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    filename_prefix = filename_prefix + f"C{config:d}" + f"P{pruning:d}"
    conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    l_edges_filename = filename_prefix + "-l_edges" + ".dat"

    # Load fundamental graph constituents
    core_nodes = np.arange(np.loadtxt(conn_n_filename, dtype=int), dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Load L
    L = np.loadtxt(L_filename)

    # Load core node x- and y-coordinates
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    
    # Create nx.Graphs, load fundamental graph constituents, and add
    # nodes before edges
    conn_core_graph = nx.Graph()
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_core_graph = add_edges_from_numpy_array(conn_core_graph, conn_core_edges)

    conn_pb_graph = nx.Graph()
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)

    conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Yield the elastically-effective network graph, if called for
    if elastically_effective:
        l_edges_filename = filename_prefix + "-ee_l_edges" + ".dat"
        conn_graph = elastically_effective_graph(conn_graph)
    
    if dim == 2:
        # Calculate and save edge lengths
        np.savetxt(
            l_edges_filename,
            dim_2_l_edges_calculation(
                conn_core_graph, conn_pb_graph, conn_graph, core_x, core_y, L))
    elif dim == 3:
        # Load core node z-coordinates
        core_z = np.loadtxt(core_z_filename)
        # Calculate and save edge lengths
        np.savetxt(
            l_edges_filename,
            dim_3_l_edges_calculation(
                conn_core_graph, conn_pb_graph, conn_graph,
                core_x, core_y, core_z, L))

def dim_2_l_nrmlzd_edges_calculation(
        conn_core_graph,
        conn_pb_graph,
        conn_graph,
        core_x: np.ndarray,
        core_y: np.ndarray,
        L: float) -> np.ndarray:
    """Normalized edge length calculation for two-dimensional networks.

    This function calculates the normalized length of each core and
    periodic boundary edge in a two-dimensional network. Note that the
    length of each edge is calculated as the true spatial length, not
    the naive length present in the graph.

    Args:
        conn_core_graph: (Undirected) NetworkX graph that can be of
        type nx.Graph or nx.MultiGraph. This graph represents the core
        edges from the graph capturing the periodic connections between
        the core nodes.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core nodes.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core nodes.
        core_x (np.ndarray): x-coordinates of the core nodes.
        core_y (np.ndarray): y-coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box
        size).
    
    Returns:
        np.ndarray: Normalized edge length.
    
    """
    # Edge length
    l_edges = dim_2_l_edges_calculation(
        conn_core_graph, conn_pb_graph, conn_graph, core_x, core_y, L)
    
    # Edge length normalization by L*sqrt(dim)
    return l_edges / (L*np.sqrt(2))

def dim_3_l_nrmlzd_edges_calculation(
        conn_core_graph,
        conn_pb_graph,
        conn_graph,
        core_x: np.ndarray,
        core_y: np.ndarray,
        core_z: np.ndarray,
        L: float) -> np.ndarray:
    """Normalized edge length calculation for three-dimensional
    networks.

    This function calculates the normalized length of each core and
    periodic boundary edge in a three-dimensional network. Note that the
    length of each edge is calculated as the true spatial length, not
    the naive length present in the graph.

    Args:
        conn_core_graph: (Undirected) NetworkX graph that can be of
        type nx.Graph or nx.MultiGraph. This graph represents the core
        edges from the graph capturing the periodic connections between
        the core nodes.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core nodes.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core nodes.
        core_x (np.ndarray): x-coordinates of the core nodes.
        core_y (np.ndarray): y-coordinates of the core nodes.
        core_z (np.ndarray): z-coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box
        size).
    
    Returns:
        np.ndarray: Normalized edge length.
    
    """
    # Edge length
    l_edges = dim_3_l_edges_calculation(
        conn_core_graph, conn_pb_graph, conn_graph, core_x, core_y, core_z, L)
    
    # Edge length normalization by L*sqrt(dim)
    return l_edges / (L*np.sqrt(3))

def swidt_network_l_nrmlzd_edges(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        config: int,
        pruning: int,
        elastically_effective: bool) -> None:
    """Normalized length of each edge in spider web-inspired
    Delaunay-triangulated networks.

    This function generates filenames associated with fundamental graph
    constituents for spider web-inspired Delaunay-triangulated networks.
    This function then calls upon a corresponding helper function to
    calculate and save the normalized length of each edge.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "swidt" is applicable (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
        elastically_effective (bool): Marker used to indicate if the
        elastically-effective network should be analyzed.
    
    """
    # Normalized edge length calculation is only applicable for spider
    # web-inspired Delaunay-triangulated networks. Exit if a different
    # type of network is passed.
    if network != "swidt":
        error_str = (
            "Normalized edge length calculation is only applicable for "
            + "the spider web-inspired Delaunay-triangulated networks. "
            + "This procedure will only proceed if network = ``swidt''."
        )
        sys.exit(error_str)
    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    filename_prefix = filename_prefix + f"C{config:d}" + f"P{pruning:d}"
    conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    l_nrmlzd_edges_filename = filename_prefix + "-l_nrmlzd_edges" + ".dat"

    # Load fundamental graph constituents
    core_nodes = np.arange(np.loadtxt(conn_n_filename, dtype=int), dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Load L
    L = np.loadtxt(L_filename)

    # Load core node x- and y-coordinates
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    
    # Create nx.Graphs, load fundamental graph constituents, and add
    # nodes before edges
    conn_core_graph = nx.Graph()
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_core_graph = add_edges_from_numpy_array(conn_core_graph, conn_core_edges)

    conn_pb_graph = nx.Graph()
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)

    conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Yield the elastically-effective network graph, if called for
    if elastically_effective:
        l_nrmlzd_edges_filename = (
            filename_prefix + "-ee_l_nrmlzd_edges" + ".dat"
        )
        conn_graph = elastically_effective_graph(conn_graph)
    
    if dim == 2:
        # Calculate and save normalized edge lengths
        np.savetxt(
            l_nrmlzd_edges_filename,
            dim_2_l_nrmlzd_edges_calculation(
                conn_core_graph, conn_pb_graph, conn_graph, core_x, core_y, L))
    elif dim == 3:
        # Load core node z-coordinates
        core_z = np.loadtxt(core_z_filename)
        # Calculate and save normalized edge lengths
        np.savetxt(
            l_nrmlzd_edges_filename,
            dim_3_l_nrmlzd_edges_calculation(
                conn_core_graph, conn_pb_graph, conn_graph,
                core_x, core_y, core_z, L))
    
def dim_2_l_cmpnts_edges_calculation(
        conn_core_graph,
        conn_pb_graph,
        conn_graph,
        core_x: np.ndarray,
        core_y: np.ndarray,
        L: float) -> tuple[np.ndarray, np.ndarray]:
    """Edge length component calculation for two-dimensional networks.

    This function calculates the length components of each core and
    periodic boundary edge in a two-dimensional network. Note that the
    length components of each edge are calculated as the true spatial
    length components, not the naive length components present in the
    graph.

    Args:
        conn_core_graph: (Undirected) NetworkX graph that can be of
        type nx.Graph or nx.MultiGraph. This graph represents the core
        edges from the graph capturing the periodic connections between
        the core nodes.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core nodes.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core nodes.
        core_x (np.ndarray): x-coordinates of the core nodes.
        core_y (np.ndarray): y-coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box
        size).
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Edge length x- and y-components,
        respectively.
    
    """
    # Gather edges and initialize l_x_cmpnt_edges and l_y_cmpnt_edges
    # np.ndarrays
    conn_edges = list(conn_graph.edges())
    conn_m = len(conn_edges)
    l_x_cmpnt_edges = np.empty(conn_m)
    l_y_cmpnt_edges = np.empty(conn_m)

    # Calculate and store the length of each edge
    for edge_indx, edge in enumerate(conn_edges):
        # Node numbers
        core_node_0 = int(edge[0])
        core_node_1 = int(edge[1])

        # Coordinates of each node
        core_node_0_x = core_x[core_node_0]
        core_node_0_y = core_y[core_node_0]
        core_node_1_x = core_x[core_node_1]
        core_node_1_y = core_y[core_node_1]

        # Edge is a core edge
        if conn_core_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store core edge length components
            l_x_cmpnt_edges[edge_indx] = core_node_1_x - core_node_0_x
            l_y_cmpnt_edges[edge_indx] = core_node_1_y - core_node_0_y
        # Edge is a periodic boundary edge
        elif conn_pb_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store periodic boundary edge length
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            l_x_cmpnt_edges[edge_indx] = pb_node_1_x - core_node_0_x
            l_y_cmpnt_edges[edge_indx] = pb_node_1_y - core_node_0_y
        else:
            error_str = (
                "The edge in the overall graph was not detected in "
                + "either the core edge graph or the periodic boundary "
                + "edge graph."
            )
            sys.exit(error_str)
        
    return (l_x_cmpnt_edges, l_y_cmpnt_edges)

def dim_3_l_cmpnts_edges_calculation(
        conn_core_graph,
        conn_pb_graph,
        conn_graph,
        core_x: np.ndarray,
        core_y: np.ndarray,
        core_z: np.ndarray,
        L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge length component calculation for three-dimensional networks.

    This function calculates the length components of each core and
    periodic boundary edge in a three-dimensional network. Note that the
    length components of each edge are calculated as the true spatial
    length components, not the naive length components present in the
    graph.

    Args:
        conn_core_graph: (Undirected) NetworkX graph that can be of
        type nx.Graph or nx.MultiGraph. This graph represents the core
        edges from the graph capturing the periodic connections between
        the core nodes.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core nodes.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core nodes.
        core_x (np.ndarray): x-coordinates of the core nodes.
        core_y (np.ndarray): y-coordinates of the core nodes.
        core_z (np.ndarray): z-coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box
        size).
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Edge length x-, y-,
        and z-components, respectively.
    
    """
    # Gather edges and initialize l_x_cmpnt_edges, l_y_cmpnt_edges,
    # and l_z_cmpnt_edges np.ndarrays
    conn_edges = list(conn_graph.edges())
    conn_m = len(conn_edges)
    l_x_cmpnt_edges = np.empty(conn_m)
    l_y_cmpnt_edges = np.empty(conn_m)
    l_z_cmpnt_edges = np.empty(conn_m)

    # Calculate and store the length of each edge
    for edge_indx, edge in enumerate(conn_edges):
        # Node numbers
        core_node_0 = int(edge[0])
        core_node_1 = int(edge[1])

        # Coordinates of each node
        core_node_0_x = core_x[core_node_0]
        core_node_0_y = core_y[core_node_0]
        core_node_0_z = core_z[core_node_0]
        core_node_1_x = core_x[core_node_1]
        core_node_1_y = core_y[core_node_1]
        core_node_1_z = core_z[core_node_1]

        # Edge is a core edge
        if conn_core_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store core edge length components
            l_x_cmpnt_edges[edge_indx] = core_node_1_x - core_node_0_x
            l_y_cmpnt_edges[edge_indx] = core_node_1_y - core_node_0_y
            l_z_cmpnt_edges[edge_indx] = core_node_1_z - core_node_0_z
        # Edge is a periodic boundary edge
        elif conn_pb_graph.has_edge(core_node_0, core_node_1):
            # Calculate and store periodic boundary edge length
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            l_x_cmpnt_edges[edge_indx] = pb_node_1_x - core_node_0_x
            l_y_cmpnt_edges[edge_indx] = pb_node_1_y - core_node_0_y
            l_z_cmpnt_edges[edge_indx] = pb_node_1_z - core_node_0_z
        else:
            error_str = (
                "The edge in the overall graph was not detected in "
                + "either the core edge graph or the periodic boundary "
                + "edge graph."
            )
            sys.exit(error_str)
        
    return (l_x_cmpnt_edges, l_y_cmpnt_edges, l_z_cmpnt_edges)

def swidt_network_l_cmpnts_edges(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        config: int,
        pruning: int,
        elastically_effective: bool) -> None:
    """Length components of each edge in spider web-inspired
    Delaunay-triangulated networks.

    This function generates filenames associated with fundamental graph
    constituents for spider web-inspired Delaunay-triangulated networks.
    This function then calls upon a corresponding helper function to
    calculate and save the length components of each edge.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "swidt" is applicable (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
        elastically_effective (bool): Marker used to indicate if the
        elastically-effective network should be analyzed.
    
    """
    # Edge length component calculation is only applicable for spider
    # web-inspired Delaunay-triangulated networks. Exit if a different
    # type of network is passed.
    if network != "swidt":
        error_str = (
            "Edge length component calculation is only applicable for "
            + "the spider web-inspired Delaunay-triangulated networks. "
            + "This procedure will only proceed if network = ``swidt''."
        )
        sys.exit(error_str)
    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    filename_prefix = filename_prefix + f"C{config:d}" + f"P{pruning:d}"
    conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    l_x_cmpnt_edges_filename = filename_prefix + "-l_x_cmpnt_edges" + ".dat"
    l_y_cmpnt_edges_filename = filename_prefix + "-l_y_cmpnt_edges" + ".dat"
    l_z_cmpnt_edges_filename = filename_prefix + "-l_z_cmpnt_edges" + ".dat"

    # Load fundamental graph constituents
    core_nodes = np.arange(np.loadtxt(conn_n_filename, dtype=int), dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Load L
    L = np.loadtxt(L_filename)

    # Load core node x- and y-coordinates
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    
    # Create nx.Graphs, load fundamental graph constituents, and add
    # nodes before edges
    conn_core_graph = nx.Graph()
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_core_graph = add_edges_from_numpy_array(conn_core_graph, conn_core_edges)

    conn_pb_graph = nx.Graph()
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)

    conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Yield the elastically-effective network graph, if called for
    if elastically_effective:
        l_x_cmpnt_edges_filename = (
            filename_prefix + "-ee_l_x_cmpnt_edges" + ".dat"
        )
        l_y_cmpnt_edges_filename = (
            filename_prefix + "-ee_l_y_cmpnt_edges" + ".dat"
        )
        l_z_cmpnt_edges_filename = (
            filename_prefix + "-ee_l_z_cmpnt_edges" + ".dat"
        )
        conn_graph = elastically_effective_graph(conn_graph)
    
    if dim == 2:
        # Calculate and save edge length components
        l_x_cmpnt_edges, l_y_cmpnt_edges = (
            dim_2_l_cmpnts_edges_calculation(
                conn_core_graph, conn_pb_graph, conn_graph, core_x, core_y, L)
        )
        np.savetxt(l_x_cmpnt_edges_filename, l_x_cmpnt_edges)
        np.savetxt(l_y_cmpnt_edges_filename, l_y_cmpnt_edges)
    elif dim == 3:
        # Load core node z-coordinates
        core_z = np.loadtxt(core_z_filename)
        # Calculate and save edge length components
        l_x_cmpnt_edges, l_y_cmpnt_edges, l_z_cmpnt_edges = (
            dim_3_l_cmpnts_edges_calculation(
                conn_core_graph, conn_pb_graph, conn_graph,
                core_x, core_y, core_z, L)
        )
        np.savetxt(l_x_cmpnt_edges_filename, l_x_cmpnt_edges)
        np.savetxt(l_y_cmpnt_edges_filename, l_y_cmpnt_edges)
        np.savetxt(l_z_cmpnt_edges_filename, l_z_cmpnt_edges)

def dim_2_l_cmpnts_nrmlzd_edges_calculation(
        conn_core_graph,
        conn_pb_graph,
        conn_graph,
        core_x: np.ndarray,
        core_y: np.ndarray,
        L: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalized edge length component calculation for two-dimensional
    networks.

    This function calculates the normalized length components of each
    core and periodic boundary edge in a two-dimensional network. Note
    that the length components of each edge are calculated as the true
    spatial length components, not the naive length components present
    in the graph.

    Args:
        conn_core_graph: (Undirected) NetworkX graph that can be of
        type nx.Graph or nx.MultiGraph. This graph represents the core
        edges from the graph capturing the periodic connections between
        the core nodes.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core nodes.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core nodes.
        core_x (np.ndarray): x-coordinates of the core nodes.
        core_y (np.ndarray): y-coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box
        size).
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Normalized edge length x- and
        y-components, respectively.
    
    """
    # Edge length components
    l_x_cmpnt_edges, l_y_cmpnt_edges = (
        dim_2_l_cmpnts_edges_calculation(
            conn_core_graph, conn_pb_graph, conn_graph, core_x, core_y, L)
    )
    # Edge length normalization by L*sqrt(dim)
    l_x_cmpnt_nrmlzd_edges = l_x_cmpnt_edges / (L*np.sqrt(2))
    l_y_cmpnt_nrmlzd_edges = l_y_cmpnt_edges / (L*np.sqrt(2))
    return (l_x_cmpnt_nrmlzd_edges, l_y_cmpnt_nrmlzd_edges)

def dim_3_l_cmpnts_nrmlzd_edges_calculation(
        conn_core_graph,
        conn_pb_graph,
        conn_graph,
        core_x: np.ndarray,
        core_y: np.ndarray,
        core_z: np.ndarray,
        L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized edge length component calculation for
    three-dimensional networks.

    This function calculates the normalized length components of each
    core and periodic boundary edge in a three-dimensional network. Note
    that the length components of each edge are calculated as the true
    spatial length components, not the naive length components present
    in the graph.

    Args:
        conn_core_graph: (Undirected) NetworkX graph that can be of
        type nx.Graph or nx.MultiGraph. This graph represents the core
        edges from the graph capturing the periodic connections between
        the core nodes.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core nodes.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core nodes.
        core_x (np.ndarray): x-coordinates of the core nodes.
        core_y (np.ndarray): y-coordinates of the core nodes.
        core_z (np.ndarray): z-coordinates of the core nodes.
        L (float): Tessellation scaling distance (i.e., simulation box
        size).
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Normalized edge
        length x-, y-, and z-components, respectively.
    
    """
    # Edge length components
    l_x_cmpnt_edges, l_y_cmpnt_edges, l_z_cmpnt_edges = (
        dim_3_l_cmpnts_edges_calculation(
            conn_core_graph, conn_pb_graph, conn_graph,
            core_x, core_y, core_z, L)
    )# Edge length normalization by L*sqrt(dim)
    l_x_cmpnt_nrmlzd_edges = l_x_cmpnt_edges / (L*np.sqrt(3))
    l_y_cmpnt_nrmlzd_edges = l_y_cmpnt_edges / (L*np.sqrt(3))
    l_z_cmpnt_nrmlzd_edges = l_z_cmpnt_edges / (L*np.sqrt(3))
    return (
        l_x_cmpnt_nrmlzd_edges,
        l_y_cmpnt_nrmlzd_edges,
        l_z_cmpnt_nrmlzd_edges
    )

def swidt_network_l_cmpnts_nrmlzd_edges(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        config: int,
        pruning: int,
        elastically_effective: bool) -> None:
    """Normalized length components of each edge in spider web-inspired
    Delaunay-triangulated networks.

    This function generates filenames associated with fundamental graph
    constituents for spider web-inspired Delaunay-triangulated networks.
    This function then calls upon a corresponding helper function to
    calculate and save the normalized length components of each edge.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; here, only "swidt" is applicable (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
        elastically_effective (bool): Marker used to indicate if the
        elastically-effective network should be analyzed.
    
    """
    # Normalized edge length component calculation is only applicable
    # for spider web-inspired Delaunay-triangulated networks. Exit if a
    # different type of network is passed.
    if network != "swidt":
        error_str = (
            "Normalized edge length component calculation is only "
            + "applicable for the spider web-inspired "
            + "Delaunay-triangulated networks. This procedure will "
            + "only proceed if network = ``swidt''."
        )
        sys.exit(error_str)
    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    filename_prefix = filename_prefix + f"C{config:d}" + f"P{pruning:d}"
    conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    l_x_cmpnt_nrmlzd_edges_filename = (
        filename_prefix + "-l_x_cmpnt_nrmlzd_edges" + ".dat"
    )
    l_y_cmpnt_nrmlzd_edges_filename = (
        filename_prefix + "-l_y_cmpnt_nrmlzd_edges" + ".dat"
    )
    l_z_cmpnt_nrmlzd_edges_filename = (
        filename_prefix + "-l_z_cmpnt_nrmlzd_edges" + ".dat"
    )

    # Load fundamental graph constituents
    core_nodes = np.arange(np.loadtxt(conn_n_filename, dtype=int), dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)

    # Load L
    L = np.loadtxt(L_filename)

    # Load core node x- and y-coordinates
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    
    # Create nx.Graphs, load fundamental graph constituents, and add
    # nodes before edges
    conn_core_graph = nx.Graph()
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_core_graph = add_edges_from_numpy_array(conn_core_graph, conn_core_edges)

    conn_pb_graph = nx.Graph()
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)

    conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Yield the elastically-effective network graph, if called for
    if elastically_effective:
        l_x_cmpnt_nrmlzd_edges_filename = (
            filename_prefix + "-ee_l_x_cmpnt_nrmlzd_edges" + ".dat"
        )
        l_y_cmpnt_nrmlzd_edges_filename = (
            filename_prefix + "-ee_l_y_cmpnt_nrmlzd_edges" + ".dat"
        )
        l_z_cmpnt_nrmlzd_edges_filename = (
            filename_prefix + "-ee_l_z_cmpnt_nrmlzd_edges" + ".dat"
        )
        conn_graph = elastically_effective_graph(conn_graph)
    
    if dim == 2:
        # Calculate and save normalized edge length components
        l_x_cmpnt_nrmlzd_edges, l_y_cmpnt_nrmlzd_edges = (
            dim_2_l_cmpnts_nrmlzd_edges_calculation(
                conn_core_graph, conn_pb_graph, conn_graph, core_x, core_y, L)
        )
        np.savetxt(l_x_cmpnt_nrmlzd_edges_filename, l_x_cmpnt_nrmlzd_edges)
        np.savetxt(l_y_cmpnt_nrmlzd_edges_filename, l_y_cmpnt_nrmlzd_edges)
    elif dim == 3:
        # Load core node z-coordinates
        core_z = np.loadtxt(core_z_filename)
        # Calculate and save normalized edge length components
        l_x_cmpnt_nrmlzd_edges, l_y_cmpnt_nrmlzd_edges, l_z_cmpnt_nrmlzd_edges = (
            dim_3_l_cmpnts_nrmlzd_edges_calculation(
                conn_core_graph, conn_pb_graph, conn_graph,
                core_x, core_y, core_z, L)
        )
        np.savetxt(l_x_cmpnt_nrmlzd_edges_filename, l_x_cmpnt_nrmlzd_edges)
        np.savetxt(l_y_cmpnt_nrmlzd_edges_filename, l_y_cmpnt_nrmlzd_edges)
        np.savetxt(l_z_cmpnt_nrmlzd_edges_filename, l_z_cmpnt_nrmlzd_edges)