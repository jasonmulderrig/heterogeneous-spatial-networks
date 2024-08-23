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
        topology; either "swidt", "auelp", or "apelp" (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt"), artificial uniform end-linked polymer networks
        ("auelp"), or artificial polydisperse end-linked polymer
        networks ("apelp")).
    
    Returns:
        str: The baseline filepath.
    
    """
    import os
    import pathlib

    filepath = f"/Users/jasonmulderrig/research/projects/heterogeneous-spatial-networks/{network}/"
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
        topology; either "swidt", "auelp", or "apelp" (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt"), artificial uniform end-linked polymer networks
        ("auelp"), or artificial polydisperse end-linked polymer
        networks ("apelp")).
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
    """Gaussian polymer chain conformation probability density.

    This function calculates the Gaussian polymer chain conformation
    probability density for a chain with a given number of segments and
    a given end-to-end distance.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        nu (ArrayLike): Number of segments in the chain.
        r (ArrayLike): End-to-end chain distance.
    
    Note: If nu is ArrayLike, then r must be a float. Likewise, if r is
    ArrayLike, then nu must be a float.

    Returns:
        ArrayLike: Gaussian polymer chain conformation probability
        density.
    """
    return (np.sqrt(3/(2*np.pi*nu*b**2)))**3 * np.exp(-3*r**2/(2*nu*b**2))

def ln_p_cnfrmtn_cubic_poly_fit_func(
        r: ArrayLike,
        c_0: float,
        c_1: float,
        c_2: float,
        c_3: float) -> ArrayLike:
    """Cubic polynomial used to curve fit the natural logarithm of a
    polymer chain conformation probability density.

    This function is a cubic polynomial used to curve fit the natural
    logarithm of a polymer chain conformation probability density.

    Args:
        r (ArrayLike): End-to-end chain distance.
        c_0 (float): Constant polynomial term.
        c_1 (float): Linear polynomial term constant.
        c_2 (float): Quadratic polynomial term constant.
        c_3 (float): Cubic polynomial term constant.

    Returns:
        ArrayLike: Cubic polynomial curve fit of the natural logarithm
        of a polymer chain conformation probability density.
    """
    return c_0 + c_1 * r + c_2 * r**2 + c_3 * r**3

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
        b (float): Chain segment and/or cross-linker diameter.
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
        b (float): Chain segment and/or cross-linker diameter.
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
        b (float): Chain segment and/or cross-linker diameter.
        n (float): Number of particles.
        eta (float): Particle packing density.

    Returns:
        float: Simulation box size.
    """
    if dim == 2: return np.sqrt(a_or_v_func(dim, b)*n/eta)
    elif dim == 3: return np.cbrt(a_or_v_func(dim, b)*n/eta)

def swidt_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float) -> None:
    """Core cross-linker simulation box size for spider web-inspired
    Delaunay-triangulated networks.

    This function calculates the core cross-linker simulation box size
    for spider web-inspired Delaunay-triangulated networks.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; either "swidt", "auelp", or "apelp" (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt"), artificial uniform end-linked polymer networks
        ("auelp"), or artificial polydisperse end-linked polymer
        networks ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Cross-linker diameter.
        n (int): Intended number of core cross-linkers.
        eta_n (float): Cross-linker packing density.
    
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

    # Calculate L
    L = L_arg_eta_func(dim, b, n, eta_n)
    
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

def random_core_coords_crosslinker_seeding(
        dim: int,
        L: float,
        b: float,
        n: int,
        max_try: int,
        filename: str) -> None:
    """Random cross-linker seeding procedure for heterogeneous spatial
    networks.

    This function randomly seeds cross-linkers within a pre-defined
    simulation box.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size of the core cross-linkers.
        b (float): Chain segment and/or cross-linker diameter.
        n (int): Intended number of core cross-linkers.
        max_try (int): Maximum number of cross-linker placement attempts.
        filename (str): Baseline filename for data files.
    
    """
    # Generate data filename
    config_filename = filename + ".config"

    # Initialize random number generator
    rng = np.random.default_rng()

    # Lists for x- and y-coordinates of core cross-linkers
    rccs_x = []
    rccs_y = []

    # np.ndarrays for x- and y-coordinates of tessellated core
    # cross-linkers
    tsslltd_rccs_x = np.asarray([])
    tsslltd_rccs_y = np.asarray([])

    if dim == 2:
        # Two-dimensional tessellation protocol
        dim_2_tsslltn, dim_2_tsslltn_num = tessellation_protocol(2)
        
        # Random cross-linker seeding procedure
        for rccs_seed_attmpt in range(n):
            # Accept and tessellate the first cross-linker
            if rccs_seed_attmpt == 0:
                # Accept the first cross-linker
                rccs_seed = L * rng.random((2,))
                
                rccs_seed_x = rccs_seed[0]
                rccs_seed_y = rccs_seed[1]
                
                rccs_x.append(rccs_seed_x)
                rccs_y.append(rccs_seed_y)

                # Use two-dimensional tessellation protocol to
                # tessellate the first cross-linker
                rccs_seed_tsslltn_x, rccs_seed_tsslltn_y = (
                    dim_2_tessellation_protocol(
                        L, rccs_seed_x, rccs_seed_y, dim_2_tsslltn)
                )

                tsslltd_rccs_x = (
                    np.concatenate((tsslltd_rccs_x, rccs_seed_tsslltn_x))
                )
                tsslltd_rccs_y = (
                    np.concatenate((tsslltd_rccs_y, rccs_seed_tsslltn_y))
                )
                continue
            else:
                # Begin random cross-linker seeding procedure
                num_try = 0

                while num_try < max_try:
                    # Generate randomly placed cross-linker candidate
                    rccs_seed_cnddt = L * rng.random((2,))
                    rccs_seed_cnddt_x = rccs_seed_cnddt[0]
                    rccs_seed_cnddt_y = rccs_seed_cnddt[1]

                    # Downselect the previously-accepted tessellated
                    # cross-linkers to those that reside in a local
                    # square neighborhood that is \pm b about the
                    # cross-linker candidate. Start by gathering the
                    # indices of cross-linkers that meet this criterion
                    # in each separate coordinate.
                    nghbr_x_lb = rccs_seed_cnddt_x - b
                    nghbr_x_ub = rccs_seed_cnddt_x + b
                    nghbr_y_lb = rccs_seed_cnddt_y - b
                    nghbr_y_ub = rccs_seed_cnddt_y + b
                    psbl_nghbr_x_indcs = (
                        np.where(np.logical_and(tsslltd_rccs_x>=nghbr_x_lb, tsslltd_rccs_x<=nghbr_x_ub))[0]
                    )
                    psbl_nghbr_y_indcs = (
                        np.where(np.logical_and(tsslltd_rccs_y>=nghbr_y_lb, tsslltd_rccs_y<=nghbr_y_ub))[0]
                    )
                    # Gather the indices from each separate coordinate
                    # together to assess all possible cross-linker
                    # neighbors
                    psbl_nghbr_indcs = (
                        np.concatenate((psbl_nghbr_x_indcs, psbl_nghbr_y_indcs))
                    )
                    # Retain unique indices corresponding to each
                    # possible cross-linker neighbor, and the number of
                    # times each such index value appears
                    psbl_nghbr_indcs, psbl_nghbr_indcs_counts = (
                        np.unique(psbl_nghbr_indcs, return_counts=True)
                    )
                    # The true cross-linker neighbors are those whose
                    # index value appears twice in the possible
                    # cross-linker neighbor array -- equal to the
                    # network dimensionality
                    nghbr_indcs_vals_indcs = (
                        np.where(psbl_nghbr_indcs_counts == 2)[0]
                    )
                    nghbr_num = np.shape(nghbr_indcs_vals_indcs)[0]
                    # Continue analysis if a local neighborhood of
                    # tessellated cross-linkers actually exists about
                    # the cross-linker candidate
                    if nghbr_num > 0:
                        # Gather the indices of the cross-linker
                        # neighbors
                        nghbr_indcs = (
                            psbl_nghbr_indcs[nghbr_indcs_vals_indcs]
                        )
                        # Extract cross-linker neighbor coordinates
                        nghbr_tsslltd_rccs_x = tsslltd_rccs_x[nghbr_indcs]
                        nghbr_tsslltd_rccs_y = tsslltd_rccs_y[nghbr_indcs]
                        
                        # Calculate the minimum distance between the
                        # cross-linker candidate and its neighbors
                        dist = np.empty(nghbr_num)
                        for nghbr_indx in range(nghbr_num):
                            nghbr_tsslltd_rccs = np.asarray(
                                [
                                    nghbr_tsslltd_rccs_x[nghbr_indx],
                                    nghbr_tsslltd_rccs_y[nghbr_indx]
                                ]
                            )
                            dist[nghbr_indx] = (
                                np.linalg.norm(
                                    rccs_seed_cnddt-nghbr_tsslltd_rccs)
                            )
                        min_dist = np.min(dist)

                        # Try again if the minimum distance between the
                        # cross-linker candidate and its neighbors is
                        # less than b
                        if min_dist < b:
                            num_try += 1
                            continue
                    
                    # Accept and tessellate the cross-linker candidate
                    # if (1) no local neighborhood of tessellated
                    # cross-linkers exists about the cross-linker
                    # candidate, or (2) the minimum distance between the
                    # cross-linker candidate and its neighbors is
                    # greater than or equal to b
                    rccs_x.append(rccs_seed_cnddt_x)
                    rccs_y.append(rccs_seed_cnddt_y)

                    # Use two-dimensional tessellation protocol to
                    # tessellate the accepted cross-linker candidate
                    rccs_seed_tsslltn_x, rccs_seed_tsslltn_y = (
                        dim_2_tessellation_protocol(
                            L, rccs_seed_cnddt_x, rccs_seed_cnddt_y,
                            dim_2_tsslltn)
                    )

                    tsslltd_rccs_x = (
                        np.concatenate((tsslltd_rccs_x, rccs_seed_tsslltn_x))
                    )
                    tsslltd_rccs_y = (
                        np.concatenate((tsslltd_rccs_y, rccs_seed_tsslltn_y))
                    )
                    break
        
        # Convert x- and y-coordinate lists to np.ndarrays, and stack
        # the x- and y-coordinates next to each other columnwise
        rccs_x = np.asarray(rccs_x)
        rccs_y = np.asarray(rccs_y)
        rccs = np.column_stack((rccs_x, rccs_y))
    elif dim == 3:
        # List for z-coordinates of core cross-linkers
        rccs_z = []

        # np.ndarray for z-coordinates of tessellated core cross-linkers
        tsslltd_rccs_z = np.asarray([])

        # Three-dimensional tessellation protocol
        dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)
        
        # Random cross-linker seeding procedure
        for rccs_seed_attmpt in range(n):
            # Accept and tessellate the first cross-linker
            if rccs_seed_attmpt == 0:
                # Accept the first cross-linker
                rccs_seed = L * rng.random((3,))
                
                rccs_seed_x = rccs_seed[0]
                rccs_seed_y = rccs_seed[1]
                rccs_seed_z = rccs_seed[2]
                
                rccs_x.append(rccs_seed_x)
                rccs_y.append(rccs_seed_y)
                rccs_z.append(rccs_seed_z)

                # Use three-dimensional tessellation protocol to
                # tessellate the first cross-linker
                rccs_seed_tsslltn_x, rccs_seed_tsslltn_y, rccs_seed_tsslltn_z = (
                    dim_3_tessellation_protocol(
                        L, rccs_seed_x, rccs_seed_y, rccs_seed_z, dim_3_tsslltn)
                )

                tsslltd_rccs_x = (
                    np.concatenate((tsslltd_rccs_x, rccs_seed_tsslltn_x))
                )
                tsslltd_rccs_y = (
                    np.concatenate((tsslltd_rccs_y, rccs_seed_tsslltn_y))
                )
                tsslltd_rccs_z = (
                    np.concatenate((tsslltd_rccs_z, rccs_seed_tsslltn_z))
                )
                continue
            else:
                # Begin random cross-linker seeding procedure
                num_try = 0

                while num_try < max_try:
                    # Generate randomly placed cross-linker candidate
                    rccs_seed_cnddt = L * rng.random((3,))
                    rccs_seed_cnddt_x = rccs_seed_cnddt[0]
                    rccs_seed_cnddt_y = rccs_seed_cnddt[1]
                    rccs_seed_cnddt_z = rccs_seed_cnddt[2]

                    # Downselect the previously-accepted tessellated
                    # cross-linkers to those that reside in a local
                    # cube neighborhood that is \pm b about the
                    # cross-linker candidate. Start by gathering the
                    # indices of cross-linkers that meet this criterion
                    # in each separate coordinate.
                    nghbr_x_lb = rccs_seed_cnddt_x - b
                    nghbr_x_ub = rccs_seed_cnddt_x + b
                    nghbr_y_lb = rccs_seed_cnddt_y - b
                    nghbr_y_ub = rccs_seed_cnddt_y + b
                    nghbr_z_lb = rccs_seed_cnddt_z - b
                    nghbr_z_ub = rccs_seed_cnddt_z + b
                    psbl_nghbr_x_indcs = (
                        np.where(np.logical_and(tsslltd_rccs_x>=nghbr_x_lb, tsslltd_rccs_x<=nghbr_x_ub))[0]
                    )
                    psbl_nghbr_y_indcs = (
                        np.where(np.logical_and(tsslltd_rccs_y>=nghbr_y_lb, tsslltd_rccs_y<=nghbr_y_ub))[0]
                    )
                    psbl_nghbr_z_indcs = (
                        np.where(np.logical_and(tsslltd_rccs_z>=nghbr_z_lb, tsslltd_rccs_z<=nghbr_z_ub))[0]
                    )
                    # Gather the indices from each separate coordinate
                    # together to assess all possible cross-linker
                    # neighbors
                    psbl_nghbr_indcs = (
                        np.concatenate(
                            (psbl_nghbr_x_indcs, psbl_nghbr_y_indcs, psbl_nghbr_z_indcs))
                    )
                    # Retain unique indices corresponding to each
                    # possible cross-linker neighbor, and the number of
                    # times each such index value appears
                    psbl_nghbr_indcs, psbl_nghbr_indcs_counts = (
                        np.unique(psbl_nghbr_indcs, return_counts=True)
                    )
                    # The true cross-linker neighbors are those whose
                    # index value appears thrice in the possible
                    # cross-linker neighbor array -- equal to the
                    # network dimensionality
                    nghbr_indcs_vals_indcs = (
                        np.where(psbl_nghbr_indcs_counts == 3)[0]
                    )
                    nghbr_num = np.shape(nghbr_indcs_vals_indcs)[0]
                    # Continue analysis if a local neighborhood of
                    # tessellated cross-linkers actually exists about
                    # the cross-linker candidate
                    if nghbr_num > 0:
                        # Gather the indices of the cross-linker
                        # neighbors
                        nghbr_indcs = (
                            psbl_nghbr_indcs[nghbr_indcs_vals_indcs]
                        )
                        # Extract cross-linker neighbor coordinates
                        nghbr_tsslltd_rccs_x = tsslltd_rccs_x[nghbr_indcs]
                        nghbr_tsslltd_rccs_y = tsslltd_rccs_y[nghbr_indcs]
                        nghbr_tsslltd_rccs_z = tsslltd_rccs_z[nghbr_indcs]
                        
                        # Calculate the minimum distance between the
                        # cross-linker candidate and its neighbors
                        dist = np.empty(nghbr_num)
                        for nghbr_indx in range(nghbr_num):
                            nghbr_tsslltd_rccs = np.asarray(
                                [
                                    nghbr_tsslltd_rccs_x[nghbr_indx],
                                    nghbr_tsslltd_rccs_y[nghbr_indx],
                                    nghbr_tsslltd_rccs_z[nghbr_indx]
                                ]
                            )
                            dist[nghbr_indx] = (
                                np.linalg.norm(
                                    rccs_seed_cnddt-nghbr_tsslltd_rccs)
                            )
                        min_dist = np.min(dist)

                        # Try again if the minimum distance between the
                        # cross-linker candidate and its neighbors is
                        # less than b
                        if min_dist < b:
                            num_try += 1
                            continue
                    
                    # Accept and tessellate the cross-linker candidate
                    # if (1) no local neighborhood of tessellated
                    # cross-linkers exists about the cross-linker
                    # candidate, or (2) the minimum distance between the
                    # cross-linker candidate and its neighbors is
                    # greater than or equal to b
                    rccs_x.append(rccs_seed_cnddt_x)
                    rccs_y.append(rccs_seed_cnddt_y)
                    rccs_z.append(rccs_seed_cnddt_z)

                    # Use three-dimensional tessellation protocol to
                    # tessellate the accepted cross-linker candidate
                    rccs_seed_tsslltn_x, rccs_seed_tsslltn_y, rccs_seed_tsslltn_z = (
                        dim_3_tessellation_protocol(
                            L, rccs_seed_cnddt_x, rccs_seed_cnddt_y,
                            rccs_seed_cnddt_z, dim_3_tsslltn)
                    )

                    tsslltd_rccs_x = (
                        np.concatenate((tsslltd_rccs_x, rccs_seed_tsslltn_x))
                    )
                    tsslltd_rccs_y = (
                        np.concatenate((tsslltd_rccs_y, rccs_seed_tsslltn_y))
                    )
                    tsslltd_rccs_z = (
                        np.concatenate((tsslltd_rccs_z, rccs_seed_tsslltn_z))
                    )
                    break
        
        # Convert x-, y-, and z-coordinate lists to np.ndarrays, and
        # stack the x-, y-, and z-coordinates next to each other
        # columnwise
        rccs_x = np.asarray(rccs_x)
        rccs_y = np.asarray(rccs_y)
        rccs_z = np.asarray(rccs_z)
        rccs = np.column_stack((rccs_x, rccs_y, rccs_z))

    # Save core cross-linker coordinates to configuration file
    np.savetxt(config_filename, rccs)

def lammps_input_file_generator(
        dim: int,
        L: float,
        b: float,
        n: int,
        filename: str) -> None:
    """LAMMPS input file generator for soft pushoff and FIRE energy
    minimization of randomly placed cross-linkers.

    This function generates a LAMMPS input file that randomly places
    cross-linkers within a pre-defined simulation box, followed by a
    soft pushoff procedure, and finally followed by a FIRE energy
    minimization procedure.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size of the core cross-linkers.
        b (float): Chain segment and/or cross-linker diameter.
        n (int): Number of core cross-linkers.
        filename (str): Baseline filename for LAMMPS data files.
    
    """
    # Generate filenames
    lammps_input_filename = filename + ".in"
    config_input_filename = filename + ".config"
    
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
    # Randomly place cross-linkers in simulation box
    lammps_input_file.write(f"create_atoms  1 random {n:d} {create_atoms_random_int:d} NULL\n")
    lammps_input_file.write("mass          1 1.0\n\n")

    # Initiate cross-linker thermodynammics
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

    # Write core cross-linker coordinates to configuration file
    lammps_input_file.write(f"write_data {config_input_filename} nocoeff")

    lammps_input_file.close()

def crosslinker_seeding(
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
    """Cross-linker seeding procedure for heterogeneous spatial
    networks.

    This function loads the simulation box size for the core
    cross-linkers of the heterogeneous spatial networks. Then, depending
    on the particular core cross-linker placement scheme, this function
    calls upon a corresponding helper function to calculate the core
    cross-linker positions.

    Args:
        network (str): Lower-case acronym indicating the particular type
        of network that is being represented by the eventual network
        topology; either "swidt", "auelp", or "apelp" (corresponding
        to spider web-inspired Delaunay-triangulated networks
        ("swidt"), artificial uniform end-linked polymer networks
        ("auelp"), or artificial polydisperse end-linked polymer
        networks ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which
        the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...)
        indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular
        scheme used to generate the positions of the core cross-linkers;
        either "rccs" or "mccs" (corresponding to random core
        cross-linker coordinates ("rccs") or minimized core
        cross-linker coordinates ("mccs")).
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        b (float): Chain segment and/or cross-linker diameter.
        n (int): Number of core cross-linkers.
        config (int): Configuration number.
        max_try (int): Maximum number of cross-linker placement attempts
        for the random core cross-linker coordinates (scheme = "rccs")
        scheme.
    
    """
    # Generate filename
    filename = filename_str(network, date, batch, sample)
    L_filename = filename + "-L" + ".dat"
    
    # Load L
    L = np.loadtxt(L_filename)

    # Append configuration number to filename
    filename = filename + f"C{config:d}"
    
    # Call appropriate helper function to calculate the core
    # cross-linker coordinates
    if scheme == "rccs": # random core cross-linker coordinates
        random_core_coords_crosslinker_seeding(dim, L, b, n, max_try, filename)
    elif scheme == "mccs": # minimized core cross-linker coordinates
        lammps_input_file_generator(dim, L, b, n, filename)

def unique_sorted_nodes_edges(
        nodes: list[int],
        edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Unique nodes and edges.

    This function takes a list of node numbers and a list of (A, B)
    nodes specifying edges, converts these lists to np.ndarrays, and
    retains unique node numbers and unique edges. If the original edge
    list contains edges (A, B) and (B, A), then only (A, B) will be
    retained (assuming that A <= B).

    Args:
        nodes (list[int]): List of node numbers.
        edges (list[tuple[int, int]]): List of edges.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Unique node numbers and unique
        edges, respectively.
    
    """
    # Convert list of node numbers to np.ndarray, and retain unique node
    # numbers
    nodes = np.unique(np.asarray(nodes, dtype=int))
    # Convert list of edges to np.ndarray, sort the order of each (A, B)
    # edge entry so that A <= B for all entries (after sorting), and
    # retain unique edges
    edges = (
        np.unique(np.sort(np.asarray(edges, dtype=int), axis=1), axis=0)
    )

    return nodes, edges

def core_pb_refactor(
        tsslltd_core_pb_nodes: np.ndarray,
        tsslltd_core_pb_edges: np.ndarray,
        n: int) -> tuple[int, int, np.ndarray]:
    """Number of core and periodic boundary nodes and edges, and
    refactored core and periodic boundary edge np.ndarray.

    This function calculates the number of core and periodic boundary
    nodes and edges. It also refactors the node numbers in the edge
    np.ndarray to be members of the (zero-indexed) set
    {0, 1, ..., core_pb_n-1}.

    Args:
        tsslltd_core_pb_nodes (np.ndarray): np.ndarray of the core and
        periodic boundary node numbers. The total number of nodes in
        this np.ndarray is core_pb_n.
        tsslltd_core_pb_edges (np.ndarray): np.ndarray of the core and
        periodic boundary edges. The total number of edges in this
        np.ndarray is core_pb_m.
        n (int): Number of core nodes.
    
    Note that there exists node numbers in tsslltd_core_pb_nodes and
    tsslltd_core_pb_edges that are >= core_pb_n.
    
    Returns:
        tuple[int, int, np.ndarray]: Number of core and periodic
        boundary nodes, number of core and periodic boundary edges, and
        refactored core and periodic boundary edge np.ndarray,
        respectively. The node numbers in the refactored core and
        periodic boundary edge np.ndarray are members of the
        (zero-indexed) set {0, 1, ..., core_pb_n-1}.
    
    """
    # Refactored node numbers for the core and periodic boundary
    # nodes are to be members of the (zero-indexed) set
    # {0, 1, ..., core_pb_n-1}, i.e.,
    # core_pb_nodes = np.arange(core_pb_n, dtype=int)
    core_pb_edges = np.empty_like(tsslltd_core_pb_edges, dtype=int)
    
    # Number of core and periodic boundary nodes; note that core_pb_n > n
    core_pb_n = np.shape(tsslltd_core_pb_nodes)[0]
    # Number of core and periodic boundary edges
    core_pb_m = np.shape(core_pb_edges)[0]
    
    for edge in range(core_pb_m):
        # Node numbers for each edge in tsslltd_core_pb_edges
        node_0 = tsslltd_core_pb_edges[edge, 0]
        node_1 = tsslltd_core_pb_edges[edge, 1]

        # Note that node numbers corresponding to the core nodes are
        # members of the (zero-indexed) set {0, 1, ..., n-1}, and the
        # node numbers corresponding to the periodic boundary nodes are
        # members of the set {n, n+1, ..., core_pb_n-1}. Thus, only the
        # node numbers from tsslltd_core_pb_edges that are >= n need to
        # be refactored to be members of the set
        # {n, n+1, ..., core_pb_n-1}.
        if node_0 >= n:
            node_0 = int(np.where(tsslltd_core_pb_nodes == node_0)[0][0])
        elif node_1 >= n:
            node_1 = int(np.where(tsslltd_core_pb_nodes == node_1)[0][0])
        else: pass

        # Save (refactored) node numbers for each core and periodic
        # boundary edge
        core_pb_edges[edge, 0] = node_0
        core_pb_edges[edge, 1] = node_1
    
    return core_pb_n, core_pb_m, core_pb_edges

def core_pb_conn_refactor(
        core_pb_m: int,
        core_pb_edges: np.ndarray,
        pb2core_nodes: np.ndarray) -> np.ndarray:
    """Refactored edge np.ndarray corresponding to the periodic
    connections between the core nodes.

    This function creates an edge np.ndarray corresponding to the
    periodic connections between the core nodes.

    Args:
        core_pb_m (int): Number of core and periodic boundary edges.
        core_pb_edges (np.ndarray): np.ndarray of the core and periodic
        boundary edges.
        pb2core_nodes (np.ndarray): np.ndarray that returns the core
        node that corresponds to each core and periodic boundary node,
        i.e., pb2core_nodes[core_pb_node] = core_node.
    
    Returns:
        np.ndarray: Refactored edge np.ndarray corresponding to the
        periodic connections between the core nodes (the node numbers in
        this np.ndarray are members of the (zero-indexed) set
        {0, 1, ..., n-1}.
    
    """
    core_pb_conn_edges = np.empty_like(core_pb_edges, dtype=int)

    for edge in range(core_pb_m):
        # Use pb2core_nodes to convert any periodic boundary node to its
        # corresponding core node for each edge in the network.
        # Duplicate entries will arise.
        core_pb_conn_edges[edge, 0] = pb2core_nodes[core_pb_edges[edge, 0]]
        core_pb_conn_edges[edge, 1] = pb2core_nodes[core_pb_edges[edge, 1]]
    
    # Sort the order of each (A, B) edge entry so that A <= B for all
    # entries (after sorting), and retain unique edges
    core_pb_conn_edges = np.unique(np.sort(core_pb_conn_edges, axis=1), axis=0)

    return core_pb_conn_edges

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

def swidt_dim_2_network_topology_initialization(
        L: float,
        core_x: np.ndarray,
        core_y: np.ndarray,
        n: int,
        filename: str) -> None:
    """Network topology initialization procedure for two-dimensional
    spider web-inspired Delaunay-triangulated networks.

    This function ``tessellates'' the core cross-linkers about
    themselves, applies Delaunay triangulation to the resulting
    tessellated network via the scipy.spatial.Delaunay() function,
    acquires back the periodic network topology of the core
    cross-linkers, and ascertains fundamental graph constituents (node
    and edge information) from this topology.

    Args:
        L (float): Simulation box size of the core cross-linkers.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
        n (int): Number of core cross-linkers.
        filename (str): Baseline filename for data files.
    
    """
    # Import the scipy.spatial.Delaunay() function
    from scipy.spatial import Delaunay
    
    # Generate filenames
    core_pb_n_filename = filename + "-core_pb_n" + ".dat"
    core_pb_edges_filename = filename + "-core_pb_edges" + ".dat"
    pb2core_nodes_filename = filename + "-pb2core_nodes" + ".dat"
    core_pb_conn_edges_filename = filename + "-core_pb_conn_edges" + ".dat"
    
    core_pb_x_filename = filename + "-core_pb_x" + ".dat"
    core_pb_y_filename = filename + "-core_pb_y" + ".dat"

    # Core cross-linker nodes
    core_nodes = np.arange(n, dtype=int)

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
            core_tsslltn_x, core_tsslltn_y = (
                dim_2_tessellation(L, core_x, core_y, x_tsslltn, y_tsslltn)
            )
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

    # Apply Delaunay triangulation to the tessellated network
    tsslltd_core_deltri = Delaunay(tsslltd_core)

    del tsslltd_core

    # Extract the simplices from the Delaunay triangulation
    simplices = tsslltd_core_deltri.simplices

    # Lists for the nodes and edges of the core and periodic boundary
    # cross-linkers
    tsslltd_core_pb_nodes = []
    tsslltd_core_pb_edges = []

    for simplex in simplices:
        # In two-dimensions, each simplex is a triangle
        node_0 = int(simplex[0])
        node_1 = int(simplex[1])
        node_2 = int(simplex[2])

        # If any of the nodes involved in any simplex edge correspond to
        # the original core cross-linkers, then add those nodes and that
        # edge to the appropriate lists. Duplicate entries will arise.
        if (node_0 < n) or (node_1 < n):
            tsslltd_core_pb_nodes.append(node_0)
            tsslltd_core_pb_nodes.append(node_1)
            tsslltd_core_pb_edges.append((node_0, node_1))
        if (node_1 < n) or (node_2 < n):
            tsslltd_core_pb_nodes.append(node_1)
            tsslltd_core_pb_nodes.append(node_2)
            tsslltd_core_pb_edges.append((node_1, node_2))
        if (node_2 < n) or (node_0 < n):
            tsslltd_core_pb_nodes.append(node_2)
            tsslltd_core_pb_nodes.append(node_0)
            tsslltd_core_pb_edges.append((node_2, node_0))
        else: pass
    
    del simplex, simplices, tsslltd_core_deltri

    # Convert node and edge lists to np.ndarrays, and retain the unique
    # nodes and edges from the core and periodic boundary cross-linkers
    tsslltd_core_pb_nodes, tsslltd_core_pb_edges = (
        unique_sorted_nodes_edges(tsslltd_core_pb_nodes, tsslltd_core_pb_edges)
    )

    # Extract the core and periodic boundary cross-linker x- and
    # y-coordinates using the corresponding node numbers
    core_pb_x = tsslltd_core_x[tsslltd_core_pb_nodes].copy()
    core_pb_y = tsslltd_core_y[tsslltd_core_pb_nodes].copy()

    del tsslltd_core_x, tsslltd_core_y

    # Extract the core and periodic boundary cross-linker nodes in the
    # np.ndarray that returns the core cross-linker node that
    # corresponds to each core and periodic boundary cross-linker node
    pb2core_nodes = pb2core_nodes[tsslltd_core_pb_nodes].copy()

    # Extract the number of core and periodic boundary cross-linker
    # nodes and edges. Refactor the node numbers in the edge np.ndarray
    # to be members of the (zero-indexed) set {0, 1, ..., core_pb_n-1}.
    core_pb_n, core_pb_m, core_pb_edges = (
        core_pb_refactor(tsslltd_core_pb_nodes, tsslltd_core_pb_edges, n)
    )
    
    del tsslltd_core_pb_nodes, tsslltd_core_pb_edges

    # Determine the edge np.ndarray corresponding to the periodic
    # connections between the core cross-linkers (without the periodic
    # boundary cross-linkers)
    core_pb_conn_edges = (
        core_pb_conn_refactor(core_pb_m, core_pb_edges, pb2core_nodes)
    )

    # Save fundamental graph constituents from this topology
    np.savetxt(core_pb_n_filename, [core_pb_n], fmt="%d")
    np.savetxt(core_pb_edges_filename, core_pb_edges, fmt="%d")
    np.savetxt(pb2core_nodes_filename, pb2core_nodes, fmt="%d")
    np.savetxt(core_pb_conn_edges_filename, core_pb_conn_edges, fmt="%d")

    # Save the core and periodic boundary cross-linker x- and
    # y-coordinates
    np.savetxt(core_pb_x_filename, core_pb_x)
    np.savetxt(core_pb_y_filename, core_pb_y)

def swidt_dim_3_network_topology_initialization(
        L: float,
        core_x: np.ndarray,
        core_y: np.ndarray,
        core_z: np.ndarray,
        n: int,
        filename: str) -> None:
    """Network topology initialization procedure for three-dimensional
    spider web-inspired Delaunay-triangulated networks.

    This function ``tessellates'' the core cross-linkers about
    themselves, applies Delaunay triangulation to the resulting
    tessellated network via the scipy.spatial.Delaunay() function,
    acquires back the periodic network topology of the core
    cross-linkers, and ascertains fundamental graph constituents (node
    and edge information) from this topology.

    Args:
        L (float): Simulation box size of the core cross-linkers.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
        core_z (np.ndarray): z-coordinates of the core cross-linkers.
        n (int): Number of core cross-linkers.
        filename (str): Baseline filename for data files.

    """
    # Import the scipy.spatial.Delaunay() function
    from scipy.spatial import Delaunay
    
    # Generate filenames
    core_pb_n_filename = filename + "-core_pb_n" + ".dat"
    core_pb_edges_filename = filename + "-core_pb_edges" + ".dat"
    pb2core_nodes_filename = filename + "-pb2core_nodes" + ".dat"
    core_pb_conn_edges_filename = filename + "-core_pb_conn_edges" + ".dat"
    
    core_pb_x_filename = filename + "-core_pb_x" + ".dat"
    core_pb_y_filename = filename + "-core_pb_y" + ".dat"
    core_pb_z_filename = filename + "-core_pb_z" + ".dat"

    # Core cross-linker nodes
    core_nodes = np.arange(n, dtype=int)

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
            core_tsslltn_x, core_tsslltn_y, core_tsslltn_z = (
                dim_3_tessellation(
                    L, core_x, core_y, core_z, x_tsslltn, y_tsslltn, z_tsslltn)
            )
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
    tsslltd_core = (
        np.column_stack((tsslltd_core_x, tsslltd_core_y, tsslltd_core_z))
    )

    # Apply Delaunay triangulation to the tessellated network
    tsslltd_core_deltri = Delaunay(tsslltd_core)

    del tsslltd_core

    # Extract the simplices from the Delaunay triangulation
    simplices = tsslltd_core_deltri.simplices

    # Lists for the nodes and edges of the core and periodic boundary
    # cross-linkers
    tsslltd_core_pb_nodes = []
    tsslltd_core_pb_edges = []

    for simplex in simplices:
        # In three-dimensions, each simplex is a tetrahedron
        node_0 = int(simplex[0])
        node_1 = int(simplex[1])
        node_2 = int(simplex[2])
        node_3 = int(simplex[3])

        # If any of the nodes involved in any simplex edge correspond to
        # the original core cross-linkers, then add those nodes and that
        # edge to the appropriate lists. Duplicate entries will arise.
        if (node_0 < n) or (node_1 < n):
            tsslltd_core_pb_nodes.append(node_0)
            tsslltd_core_pb_nodes.append(node_1)
            tsslltd_core_pb_edges.append((node_0, node_1))
        if (node_1 < n) or (node_2 < n):
            tsslltd_core_pb_nodes.append(node_1)
            tsslltd_core_pb_nodes.append(node_2)
            tsslltd_core_pb_edges.append((node_1, node_2))
        if (node_2 < n) or (node_0 < n):
            tsslltd_core_pb_nodes.append(node_2)
            tsslltd_core_pb_nodes.append(node_0)
            tsslltd_core_pb_edges.append((node_2, node_0))
        if (node_3 < n) or (node_0 < n):
            tsslltd_core_pb_nodes.append(node_3)
            tsslltd_core_pb_nodes.append(node_0)
            tsslltd_core_pb_edges.append((node_3, node_0))
        if (node_3 < n) or (node_1 < n):
            tsslltd_core_pb_nodes.append(node_3)
            tsslltd_core_pb_nodes.append(node_1)
            tsslltd_core_pb_edges.append((node_3, node_1))
        if (node_3 < n) or (node_2 < n):
            tsslltd_core_pb_nodes.append(node_3)
            tsslltd_core_pb_nodes.append(node_2)
            tsslltd_core_pb_edges.append((node_3, node_2))
        else: pass
    
    del simplex, simplices, tsslltd_core_deltri

    # Convert node and edge lists to np.ndarrays, and retain the unique
    # nodes and edges from the core and periodic boundary cross-linkers
    tsslltd_core_pb_nodes, tsslltd_core_pb_edges = (
        unique_sorted_nodes_edges(tsslltd_core_pb_nodes, tsslltd_core_pb_edges)
    )

    # Extract the core and periodic boundary cross-linker x-, y-, and
    # z-coordinates using the corresponding node numbers
    core_pb_x = tsslltd_core_x[tsslltd_core_pb_nodes].copy()
    core_pb_y = tsslltd_core_y[tsslltd_core_pb_nodes].copy()
    core_pb_z = tsslltd_core_z[tsslltd_core_pb_nodes].copy()

    del tsslltd_core_x, tsslltd_core_y, tsslltd_core_z

    # Extract the core and periodic boundary cross-linker nodes in the
    # np.ndarray that returns the core cross-linker node that
    # corresponds to each core and periodic boundary cross-linker node
    pb2core_nodes = pb2core_nodes[tsslltd_core_pb_nodes].copy()

    # Extract the number of core and periodic boundary cross-linker
    # nodes and edges. Refactor the node numbers in the edge np.ndarray
    # to be members of the (zero-indexed) set {0, 1, ..., core_pb_n-1}.
    core_pb_n, core_pb_m, core_pb_edges = (
        core_pb_refactor(tsslltd_core_pb_nodes, tsslltd_core_pb_edges, n)
    )
    
    del tsslltd_core_pb_nodes, tsslltd_core_pb_edges

    # Determine the edge np.ndarray corresponding to the periodic
    # connections between the core cross-linkers (without the periodic
    # boundary cross-linkers)
    core_pb_conn_edges = (
        core_pb_conn_refactor(core_pb_m, core_pb_edges, pb2core_nodes)
    )

    # Save fundamental graph constituents from this topology
    np.savetxt(core_pb_n_filename, [core_pb_n], fmt="%d")
    np.savetxt(core_pb_edges_filename, core_pb_edges, fmt="%d")
    np.savetxt(pb2core_nodes_filename, pb2core_nodes, fmt="%d")
    np.savetxt(core_pb_conn_edges_filename, core_pb_conn_edges, fmt="%d")
    
    # Save the core and periodic boundary cross-linker x-, y-, and
    # z-coordinates
    np.savetxt(core_pb_x_filename, core_pb_x)
    np.savetxt(core_pb_y_filename, core_pb_y)
    np.savetxt(core_pb_z_filename, core_pb_z)

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

    This function loads the simulation box size and the core
    cross-linker coordinates previously generated by the
    crosslinker_seeding() function. Then, depending on the network
    dimensionality, this function calls upon a corresponding helper
    function to initialize the spider web-inspired Delaunay-triangulated
    network topology.

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
        scheme used to generate the positions of the core cross-linkers;
        either "rccs" or "mccs" (corresponding to random core
        cross-linker coordinates ("rccs") or minimized core
        cross-linker coordinates ("mccs")).
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
        n (int): Number of core cross-linkers.
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
    # Generate filename
    filename = filename_str(network, date, batch, sample)
    L_filename = filename + "-L" + ".dat"
    
    # Load L
    L = np.loadtxt(L_filename)

    # Append configuration number to filename
    filename = filename + f"C{config:d}"

    # Generate config filename
    config_filename = filename + ".config"

    # Call appropriate helper function to initialize network topology
    if scheme == "rccs": # random core cross-linker coordinates
        # Load core cross-linker coordinates
        rccs = np.loadtxt(config_filename)
        # Actual number of core cross-linkers
        rccs_n = np.shape(rccs)[0]
        # Separate x- and y-coordinates of core cross-linkers
        rccs_x = rccs[:, 0].copy()
        rccs_y = rccs[:, 1].copy()
        if dim == 2:
            del rccs
            swidt_dim_2_network_topology_initialization(
                L, rccs_x, rccs_y, rccs_n, filename)
        elif dim == 3:
            # Separate z-coordinates of core cross-linkers
            rccs_z = rccs[:, 2].copy()
            del rccs
            swidt_dim_3_network_topology_initialization(
                L, rccs_x, rccs_y, rccs_z, rccs_n, filename)
    elif scheme == "mccs": # minimized core cross-linker coordinates
        skiprows_num = 15
        # Load core cross-linker coordinates
        mccs = np.loadtxt(config_filename, skiprows=skiprows_num, max_rows=n)
        # Separate x- and y-coordinates of core cross-linkers
        mccs_x = mccs[:, 2].copy()
        mccs_y = mccs[:, 3].copy()
        if dim == 2:
            del mccs
            swidt_dim_2_network_topology_initialization(
                L, mccs_x, mccs_y, n, filename)
        elif dim == 3:
            # Separate z-coordinates of core cross-linkers
            mccs_z = mccs[:, 4].copy()
            del mccs
            swidt_dim_3_network_topology_initialization(
                L, mccs_x, mccs_y, mccs_z, n, filename)

def dim_2_core_node2pb_node_mic_identification(
        core_node: int,
        core_node2pb_node: int,
        core2pb_nodes: list[np.ndarray],
        core_pb_x: np.ndarray,
        core_pb_y: np.ndarray) -> int:
    pb_nodes = core2pb_nodes[core_node2pb_node]
    # Select the correct pb_node via the minimum image criterion, i.e.,
    # the minimum distance criterion
    core_node_pstn = np.asarray(
        [
            core_pb_x[core_node],
            core_pb_y[core_node]
        ]
    )
    pb_nodes_num = np.shape(pb_nodes)[0]
    r_pb_nodes = np.empty(pb_nodes_num)
    for pb_node_indx in range(pb_nodes_num):
        pb_node = pb_nodes[pb_node_indx]
        pb_node_pstn = np.asarray(
            [
                core_pb_x[pb_node],
                core_pb_y[pb_node]
            ]
        )
        r_pb_nodes[pb_node_indx] = (
            np.linalg.norm(core_node_pstn-pb_node_pstn)
        )
    return pb_nodes[np.argmin(r_pb_nodes)]

def dim_3_core_node2pb_node_mic_identification(
        core_node: int,
        core_node2pb_node: int,
        core2pb_nodes: list[np.ndarray],
        core_pb_x: np.ndarray,
        core_pb_y: np.ndarray,
        core_pb_z: np.ndarray) -> int:
    pb_nodes = core2pb_nodes[core_node2pb_node]
    # Select the correct pb_node via the minimum image criterion, i.e.,
    # the minimum distance criterion
    core_node_pstn = np.asarray(
        [
            core_pb_x[core_node],
            core_pb_y[core_node],
            core_pb_z[core_node]
        ]
    )
    pb_nodes_num = np.shape(pb_nodes)[0]
    r_pb_nodes = np.empty(pb_nodes_num)
    for pb_node_indx in range(pb_nodes_num):
        pb_node = pb_nodes[pb_node_indx]
        pb_node_pstn = np.asarray(
            [
                core_pb_x[pb_node],
                core_pb_y[pb_node],
                core_pb_z[pb_node]
            ]
        )
        r_pb_nodes[pb_node_indx] = (
            np.linalg.norm(core_node_pstn-pb_node_pstn)
        )
    return pb_nodes[np.argmin(r_pb_nodes)]

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
    and periodic boundary cross-linker coordinates, performs a random
    edge pruning procedure such that each cross-linker in the network is
    connected to, at most, k edges, and isolates the maximum connected
    component from the resulting network.

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
        n (int): Number of core cross-linkers.
        k (int): Maximum cross-linker degree/functionality; either 3, 4,
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
    pb_node_0 = 0
    pb_node_1 = 0

    # Generate filenames
    filename = filename_str(network, date, batch, sample) + f"C{config:d}"
    core_pb_n_filename = filename + "-core_pb_n" + ".dat"
    core_pb_edges_filename = filename + "-core_pb_edges" + ".dat"
    pb2core_nodes_filename = filename + "-pb2core_nodes" + ".dat"
    core_pb_conn_edges_filename = filename + "-core_pb_conn_edges" + ".dat"
    core_pb_x_filename = filename + "-core_pb_x" + ".dat"
    core_pb_y_filename = filename + "-core_pb_y" + ".dat"
    if dim == 3:
        core_pb_z_filename = filename + "-core_pb_z" + ".dat"
    mx_cmp_core_pb_n_filename = (
        filename + f"P{pruning:d}" + "-core_pb_n" + ".dat"
    )
    mx_cmp_core_pb_edges_filename = (
        filename + f"P{pruning:d}" + "-core_pb_edges" + ".dat"
    )
    mx_cmp_pb2core_nodes_filename = (
        filename + f"P{pruning:d}" + "-pb2core_nodes" + ".dat"
    )
    mx_cmp_core_pb_conn_n_filename = (
        filename + f"P{pruning:d}" + "-core_pb_conn_n" + ".dat"
    )
    mx_cmp_core_pb_conn_edges_filename = (
        filename + f"P{pruning:d}" + "-core_pb_conn_edges" + ".dat"
    )
    mx_cmp_core_pb_x_filename = (
        filename + f"P{pruning:d}" + "-core_pb_x" + ".dat"
    )
    mx_cmp_core_pb_y_filename = (
        filename + f"P{pruning:d}" + "-core_pb_y" + ".dat"
    )
    if dim == 3:
        mx_cmp_core_pb_z_filename = (
            filename + f"P{pruning:d}" + "-core_pb_z" + ".dat"
        )
    
    # Load fundamental graph constituents
    core_pb_n = np.loadtxt(core_pb_n_filename, dtype=int)
    core_pb_edges = np.loadtxt(core_pb_edges_filename, dtype=int)
    pb2core_nodes = np.loadtxt(pb2core_nodes_filename, dtype=int)
    core_pb_conn_edges = np.loadtxt(core_pb_conn_edges_filename, dtype=int)

    core_nodes = np.arange(n, dtype=int)
    core_pb_nodes = np.arange(core_pb_n, dtype=int)
    core_pb_conn_nodes = core_nodes.copy()

    # Construct the core2pb_nodes list
    core2pb_nodes = core2pb_nodes_func(core_nodes, pb2core_nodes)

    # Load the core and periodic boundary cross-linker x- and
    # y-coordinates
    core_pb_x = np.loadtxt(core_pb_x_filename)
    core_pb_y = np.loadtxt(core_pb_y_filename)
    if dim == 3:
        # Load the core and periodic boundary cross-linker z-coordinates
        core_pb_z = np.loadtxt(core_pb_z_filename)
    
    # Create nx.Graphs, and add nodes before edges
    core_pb_graph = nx.Graph()
    core_pb_graph.add_nodes_from(core_pb_nodes)
    core_pb_graph.add_edges_from(core_pb_edges)

    core_pb_conn_graph = nx.Graph()
    core_pb_conn_graph.add_nodes_from(core_pb_conn_nodes)
    core_pb_conn_graph.add_edges_from(core_pb_conn_edges)

    # Degree of cross-linker nodes in the core_pb_conn_graph
    core_pb_conn_graph_k = (
        np.asarray(list(core_pb_conn_graph.degree()), dtype=int)[:, 1]
    )

    if np.any(core_pb_conn_graph_k > k):
        # Explicit edge pruning procedure
        while np.any(core_pb_conn_graph_k > k):
            # Identify the cross-linker nodes connected to more than k
            # edges in the core_pb_conn_graph, i.e., hyperconnected
            # cross-linker nodes
            core_pb_conn_graph_hyprconn_nodes = (
                np.where(core_pb_conn_graph_k > k)[0]
            )
            # Identify the edges connected to the hyperconnected
            # cross-linker nodes
            core_pb_conn_graph_hyprconn_edge_indcs_0 = (
                np.where(np.isin(core_pb_conn_edges[:, 0], core_pb_conn_graph_hyprconn_nodes))[0]
            )
            core_pb_conn_graph_hyprconn_edge_indcs_1 = (
                np.where(np.isin(core_pb_conn_edges[:, 1], core_pb_conn_graph_hyprconn_nodes))[0]
            )
            core_pb_conn_graph_hyprconn_edge_indcs = (
                np.unique(
                    np.concatenate(
                        (core_pb_conn_graph_hyprconn_edge_indcs_0, core_pb_conn_graph_hyprconn_edge_indcs_1)))
            )
            # Randomly select a hyperconnected edge to remove
            edge_indcs_indx2remove_indx = (
                rng.integers(
                    np.shape(core_pb_conn_graph_hyprconn_edge_indcs)[0], dtype=int)
            )
            edge_indx2remove = (
                core_pb_conn_graph_hyprconn_edge_indcs[edge_indcs_indx2remove_indx]
            )
            edge2remove = core_pb_conn_edges[edge_indx2remove]
            core_node_0 = edge2remove[0]
            core_node_1 = edge2remove[1]

            # Remove hyperconnected edge in the core_pb_conn_graph
            core_pb_conn_graph.remove_edge(core_node_0, core_node_1)
            core_pb_conn_edges = (
                np.delete(core_pb_conn_edges, edge_indx2remove, axis=0)
            )
            
            # Remove hyperconnected edge(s) in the core_pb_graph
            if core_pb_graph.has_edge(core_node_0, core_node_1):
                # Hyperconnected edge connects two core cross-linkers in
                # the core_pb_graph
                core_pb_graph.remove_edge(core_node_0, core_node_1)
            else:
                # Two hyperconnected edges that each separately yet
                # correspondingly connect a core cross-linker and a
                # periodic boundary cross-linker in the core_pb_graph.
                # Identify all possible periodic boundary cross-linkers
                # that could be involved in each hyperconnected edge.
                pb_nodes_0 = core2pb_nodes[core_node_0]
                pb_nodes_1 = core2pb_nodes[core_node_1]
                # Determine the specific periodic boundary and core
                # cross-linker pair involved in each hyperconnected
                # edge, and remove each edge
                for pb_node_0 in np.nditer(pb_nodes_0):
                    pb_node_0 = int(pb_node_0)
                    if core_pb_graph.has_edge(pb_node_0, core_node_1):
                        core_pb_graph.remove_edge(pb_node_0, core_node_1)
                        break
                    else: pass
                for pb_node_1 in np.nditer(pb_nodes_1):
                    pb_node_1 = int(pb_node_1)
                    if core_pb_graph.has_edge(core_node_0, pb_node_1):
                        core_pb_graph.remove_edge(core_node_0, pb_node_1)
                        break
                    else: pass

            # Update degree of cross-linker nodes in the
            # core_pb_conn_graph
            core_pb_conn_graph_k[core_node_0] -= 1
            core_pb_conn_graph_k[core_node_1] -= 1
                
        # Isolate largest/maximum connected component from the
        # core_pb_conn_graph in a nodewise fashion. Note that
        # mx_cmp_core_pb_conn_graph_nodes[updated_node] = original_node
        mx_cmp_core_pb_conn_graph_nodes = max(
            nx.connected_components(core_pb_conn_graph), key=len)
        mx_cmp_core_pb_conn_graph = (
            core_pb_conn_graph.subgraph(mx_cmp_core_pb_conn_graph_nodes).copy()
        )
        # Nodes from the core_pb_conn_graph largest/maximum connected
        # component, sorted in ascending order
        mx_cmp_core_pb_conn_graph_nodes = (
            np.sort(np.fromiter(mx_cmp_core_pb_conn_graph_nodes, dtype=int))
        )
        # Edges from the core_pb_conn_graph largest/maximum connected
        # component
        mx_cmp_core_pb_conn_graph_edges = (
            np.asarray(list(mx_cmp_core_pb_conn_graph.edges()), dtype=int)
        )
        # Number of nodes in the core_pb_conn_graph largest/maximum
        # connected component
        mx_cmp_core_pb_conn_graph_n = (
            np.shape(mx_cmp_core_pb_conn_graph_nodes)[0]
        )
        # Number of edges in the core_pb_conn_graph largest/maximum
        # connected component
        mx_cmp_core_pb_conn_graph_m = (
            np.shape(mx_cmp_core_pb_conn_graph_edges)[0]
        )
        
        # Isolate largest/maximum connected component from the
        # core_pb_graph via the core_pb_conn_graph largest/maximum
        # connected component
        mx_cmp_core_pb_graph_nodes = []
        mx_cmp_core_pb_graph_edges = []

        for edge in range(mx_cmp_core_pb_conn_graph_m):
            # original_node
            core_node_0 = mx_cmp_core_pb_conn_graph_edges[edge, 0]
            core_node_1 = mx_cmp_core_pb_conn_graph_edges[edge, 1]

            # Add edge(s) in the core_pb_graph
            if core_pb_graph.has_edge(core_node_0, core_node_1):
                # Add edge that connects two core cross-linkers in the
                # core_pb_graph, and add the core cross-linkers
                mx_cmp_core_pb_graph_nodes.append(core_node_0)
                mx_cmp_core_pb_graph_nodes.append(core_node_1)
                mx_cmp_core_pb_graph_edges.append((core_node_0, core_node_1))
            else:
                # Add two edges that each separately yet correspondingly
                # connect a core cross-linker and a periodic boundary
                # cross-linker in the core_pb_graph. Also add the core
                # and periodic boundary cross-linkers. Identify all
                # possible periodic boundary cross-linkers that could be
                # involved in each edge.
                pb_nodes_0 = core2pb_nodes[core_node_0]
                pb_nodes_1 = core2pb_nodes[core_node_1]
                # Determine the specific periodic boundary and core
                # cross-linker pair involved in each edge, and then add
                # the edge and cross-linkers
                for pb_node_0 in np.nditer(pb_nodes_0):
                    pb_node_0 = int(pb_node_0)
                    if core_pb_graph.has_edge(pb_node_0, core_node_1):
                        mx_cmp_core_pb_graph_nodes.append(pb_node_0)
                        mx_cmp_core_pb_graph_nodes.append(core_node_1)
                        mx_cmp_core_pb_graph_edges.append((pb_node_0, core_node_1))
                        break
                    else: pass
                for pb_node_1 in np.nditer(pb_nodes_1):
                    pb_node_1 = int(pb_node_1)
                    if core_pb_graph.has_edge(core_node_0, pb_node_1):
                        mx_cmp_core_pb_graph_nodes.append(core_node_0)
                        mx_cmp_core_pb_graph_nodes.append(pb_node_1)
                        mx_cmp_core_pb_graph_edges.append((core_node_0, pb_node_1))
                        break
                    else: pass
        
        # Convert to np.ndarrays and retain unique values
        mx_cmp_core_pb_graph_nodes = (
            np.unique(np.asarray(mx_cmp_core_pb_graph_nodes, dtype=int))
        )
        mx_cmp_core_pb_graph_edges = (
            np.unique(np.asarray(mx_cmp_core_pb_graph_edges, dtype=int), axis=0)
        )
        # Number of nodes in the core_pb_graph largest/maximum connected
        # component
        mx_cmp_core_pb_graph_n = np.shape(mx_cmp_core_pb_graph_nodes)[0]
        # Number of edges in the core_pb_graph largest/maximum connected
        # component
        mx_cmp_core_pb_graph_m = np.shape(mx_cmp_core_pb_graph_edges)[0]

        # Isolate the cross-linker coordinates for the largest/maximum
        # connected component
        # updated_node
        mx_cmp_core_pb_x = core_pb_x[mx_cmp_core_pb_graph_nodes]
        mx_cmp_core_pb_y = core_pb_y[mx_cmp_core_pb_graph_nodes]
        if dim == 3:
            mx_cmp_core_pb_z = core_pb_z[mx_cmp_core_pb_graph_nodes]
        
        # Isolate pb2core_nodes for the largest/maximum connected
        # component. Note that
        # mx_cmp_pb2core_nodes[updated_node] = original_node
        mx_cmp_pb2core_nodes = pb2core_nodes[mx_cmp_core_pb_graph_nodes]

        # Update all original_node values with updated_node values for
        # mx_cmp_core_pb_conn_graph_edges
        for edge in range(mx_cmp_core_pb_conn_graph_m):
            # updated_node
            mx_cmp_core_pb_conn_graph_edges[edge, 0] = (
                int(np.where(mx_cmp_core_pb_conn_graph_nodes == mx_cmp_core_pb_conn_graph_edges[edge, 0])[0][0])
            )
            mx_cmp_core_pb_conn_graph_edges[edge, 1] = (
                int(np.where(mx_cmp_core_pb_conn_graph_nodes == mx_cmp_core_pb_conn_graph_edges[edge, 1])[0][0])
            )
        
        # Update all original_node values with updated_node values for
        # mx_cmp_pb2core_nodes
        for node in range(mx_cmp_core_pb_graph_n):
            # updated_node
            mx_cmp_pb2core_nodes[node] = (
                int(np.where(mx_cmp_core_pb_conn_graph_nodes == mx_cmp_pb2core_nodes[node])[0][0])
            )
        
        # Update all original_node values with updated_node values for
        # mx_cmp_core_pb_graph_edges
        for edge in range(mx_cmp_core_pb_graph_m):
            # updated_node
            mx_cmp_core_pb_graph_edges[edge, 0] = (
                int(np.where(mx_cmp_core_pb_graph_nodes == mx_cmp_core_pb_graph_edges[edge, 0])[0][0])
            )
            mx_cmp_core_pb_graph_edges[edge, 1] = (
                int(np.where(mx_cmp_core_pb_graph_nodes == mx_cmp_core_pb_graph_edges[edge, 1])[0][0])
            )
                
        # Save fundamental graph constituents from this topology
        np.savetxt(
            mx_cmp_core_pb_n_filename, [mx_cmp_core_pb_graph_n], fmt="%d")
        np.savetxt(
            mx_cmp_core_pb_edges_filename, mx_cmp_core_pb_graph_edges, fmt="%d")
        np.savetxt(
            mx_cmp_pb2core_nodes_filename, mx_cmp_pb2core_nodes, fmt="%d")
        np.savetxt(
            mx_cmp_core_pb_conn_n_filename, [mx_cmp_core_pb_conn_graph_n],
            fmt="%d")
        np.savetxt(
            mx_cmp_core_pb_conn_edges_filename, mx_cmp_core_pb_conn_graph_edges,
            fmt="%d")
        
        # Save the core and periodic boundary cross-linker x- and
        # y-coordinates
        np.savetxt(mx_cmp_core_pb_x_filename, mx_cmp_core_pb_x)
        np.savetxt(mx_cmp_core_pb_y_filename, mx_cmp_core_pb_y)
        if dim == 3:
            # Save the core and periodic boundary cross-linker
            # z-coordinates
            np.savetxt(mx_cmp_core_pb_z_filename, mx_cmp_core_pb_z)
    else:
        # Save fundamental graph constituents from this topology
        np.savetxt(mx_cmp_core_pb_n_filename, [core_pb_n], fmt="%d")
        np.savetxt(mx_cmp_core_pb_edges_filename, core_pb_edges, fmt="%d")
        np.savetxt(mx_cmp_pb2core_nodes_filename, pb2core_nodes, fmt="%d")
        np.savetxt(mx_cmp_core_pb_conn_n_filename, [n], fmt="%d")
        np.savetxt(
            mx_cmp_core_pb_conn_edges_filename, core_pb_conn_edges, fmt="%d")
        
        # Save the core and periodic boundary cross-linker x- and
        # y-coordinates
        np.savetxt(mx_cmp_core_pb_x_filename, core_pb_x)
        np.savetxt(mx_cmp_core_pb_y_filename, core_pb_y)
        if dim == 3:
            # Save the core and periodic boundary cross-linker
            # z-coordinates
            np.savetxt(mx_cmp_core_pb_z_filename, core_pb_z)

def graph_k_counts_calculation(graph) -> np.ndarray:
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
        k_counts[graph_k[k_indx]-1] = graph_k_counts[k_indx]
    
    return k_counts

def swidt_network_graph_k_counts(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        pruning: int,
        graph: str) -> None:
    """Cross-linker node degree counts in spider web-inspired
    Delaunay-triangulated networks.

    This function generates the filename prefix associated with
    fundamental graph constituents for spider web-inspired
    Delaunay-triangulated networks. This function then calls upon a
    corresponding helper function to calculate and save the cross-linker
    node degree counts.

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
        graph (str): Graph type in which the cross-linker node degree
        counts will be calculated; here, either "core_pb" (corresponding
        to the graph capturing the spatial topology of the core and
        periodic boundary nodes and edges) or "core_pb_conn"
        (corresponding to the graph capturing the periodic connections
        between the core nodes) are applicable.
    
    """
    # Cross-linker node degree counts calculation is only applicable for
    # spider web-inspired Delaunay-triangulated networks. Exit if a
    # different type of network is passed.
    if network != "swidt":
        error_str = (
            "Cross-linker node degree count calculation is only applicable for "
            + "the spider web-inspired Delaunay-triangulated networks. This "
            + "procedure will only proceed if network = ``swidt''."
        )
        sys.exit(error_str)
    # Generate filenames
    filename_prefix = (
        filename_str(network, date, batch, sample)
        + f"C{config:d}" + f"P{pruning:d}"
    )
    n_filename = filename_prefix + "-" + graph + "_n" + ".dat"
    edges_filename = filename_prefix + "-" + graph + "_edges" + ".dat"
    k_counts_filename = filename_prefix + "-" + graph + "_k_counts" + ".dat"

    # Create nx.Graph, load fundamental graph constituents, and add
    # nodes before edges
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(np.loadtxt(n_filename, dtype=int), dtype=int))
    graph.add_edges_from(np.loadtxt(edges_filename, dtype=int))

    # Calculate and save the node degree counts
    np.savetxt(k_counts_filename, graph_k_counts_calculation(graph), fmt="%d")

def graph_h_counts_calculation(graph, l_bound: int) -> np.ndarray:
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
    self_loop_num = nx.number_of_selfloops(graph)
    h_counts[0] = self_loop_num

    # Self-loop pruning procedure
    if self_loop_num > 0:
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    
    # If the graph is of type nx.MultiGraph, then address multi-edges,
    # prune redundant edges, and convert resulting graph to type
    # nx.Graph
    if graph.is_multigraph() == True:    
        # Gather edges and edges counts
        graph_edges, graph_edges_counts = (
            np.unique(np.sort(np.asarray(list(graph.edges()), dtype=int), axis=1), return_counts=True, axis=0)
        )
        
        # Address multi-edges by calculating and storing the number of
        # second-order cycles and by pruning redundant edges
        if np.any(graph_edges_counts > 1):
            # Extract multi-edges
            multiedges = np.where(graph_edges_counts > 1)[0]
            for multiedge in np.nditer(multiedges):
                multiedge = int(multiedge)
                # Multiedge
                edge = graph_edges[multiedge]
                # Number of edges in the multiedge
                edge_num = graph_edges_counts[multiedge]
                # Calculate the number of second-order cycles induced by
                # redundant multi-edges
                h_counts[1] += int(comb(edge_num, 2))
                # Remove redundant edges in the multiedge (thereby
                # leaving one edge)
                graph.remove_edges_from(
                    list((edge[0], edge[1]) for _ in range(edge_num-1)))
        
        # Convert graph to type nx.Graph
        graph = nx.Graph(graph)

    # Degree of nodes
    graph_nodes_k = np.asarray(list(graph.degree()), dtype=int)
    graph_k = graph_nodes_k[:, 1]

    # Dangling edge pruning procedure
    while np.any(graph_k == 1):
        # Extract nodes with k = 1
        k1_nodes = np.where(graph_k == 1)[0]
        for k1_node in np.nditer(k1_nodes):
            k1_node = int(k1_node)
            # Node with k = 1
            node = graph_nodes_k[k1_node, 0]
            # Neighbor of node with k = 1
            node_nghbr_arr = np.asarray(list(graph.neighbors(node)), dtype=int)
            # Check to see if the dangling edge was previously removed
            if np.shape(node_nghbr_arr)[0] == 0: continue
            else:
                # Remove dangling edge
                graph.remove_edge(node, node_nghbr_arr[0])
        # Update degree of nodes
        graph_nodes_k = np.asarray(list(graph.degree()), dtype=int)
        graph_k = graph_nodes_k[:, 1]
    
    # Find chordless cycles
    chrdls_cycls = list(nx.chordless_cycles(graph, length_bound=l_bound))
    
    # Calculate number of occurrences for each chordless cycle order
    graph_h, graph_h_counts = np.unique(
        np.asarray(list(len(cycl) for cycl in chrdls_cycls), dtype=int),
        return_counts=True)

    # Store the chordless cycle counts
    for h_indx in range(np.shape(graph_h)[0]):
        h_counts[graph_h[h_indx]-1] = graph_h_counts[h_indx]
    
    return h_counts

def swidt_network_graph_h_counts(
        network: str,
        date: str,
        batch: str,
        sample: int,
        l_bound: int,
        config: int,
        pruning: int,
        graph: str) -> None:
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
        graph (str): Graph type in which the chordless cycle counts will
        be calculated; here, either "core_pb" (corresponding to the
        graph capturing the spatial topology of the core and periodic
        boundary nodes and edges) or "core_pb_conn" (corresponding to
        the graph capturing the periodic connections between the core
        nodes) are applicable.
    
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
    n_filename = filename_prefix + "-" + graph + "_n" + ".dat"
    edges_filename = filename_prefix + "-" + graph + "_edges" + ".dat"
    h_counts_filename = filename_prefix + "-" + graph + "_h_counts" + ".dat"

    # Create nx.Graph, load fundamental graph constituents, and add
    # nodes before edges
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(np.loadtxt(n_filename, dtype=int), dtype=int))
    graph.add_edges_from(np.loadtxt(edges_filename, dtype=int))

    # Calculate and save the chordless cycle counts counts
    np.savetxt(
        h_counts_filename, graph_h_counts_calculation(graph, l_bound), fmt="%d")

############ Split up the functionality for l_edges and l_edges_cmpnts in an analogous manner as was done above for k and h, where the graph type is made 
# Could normalization be activated with a boolean variable? And thus reduce the code size?

def core_pb_graph_l_edges_calculation(
        filename_prefix: str,
        dim: int) -> np.ndarray:
    """Length of each edge calculated in the graph capturing the spatial
    topology of the core and periodic boundary nodes and edges.

    This function loads fundamental graph constituents, calculates the
    length of each edge in the graph capturing the spatial topology of
    the core and periodic boundary nodes and edges, and returns this
    information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
    
    Returns:
        np.ndarray: Length of each edge in the graph capturing the
        spatial topology of the core and periodic boundary nodes and
        edges.
    
    """
    # Generate filenames
    core_pb_n_filename = filename_prefix + "-core_pb_n" + ".dat"
    core_pb_edges_filename = filename_prefix + "-core_pb_edges" + ".dat"
    pb2core_nodes_filename = filename_prefix + "-pb2core_nodes" + ".dat"
    core_pb_conn_n_filename = filename_prefix + "-core_pb_conn_n" + ".dat"
    core_pb_conn_edges_filename = (
        filename_prefix + "-core_pb_conn_edges" + ".dat"
    )
    core_pb_x_filename = filename_prefix + "-core_pb_x" + ".dat"
    core_pb_y_filename = filename_prefix + "-core_pb_y" + ".dat"
    if dim == 3:
        core_pb_z_filename = filename_prefix + "-core_pb_z" + ".dat"

    # Load fundamental graph constituents
    core_pb_n = np.loadtxt(core_pb_n_filename, dtype=int)
    core_pb_edges = np.loadtxt(core_pb_edges_filename, dtype=int)
    pb2core_nodes = np.loadtxt(pb2core_nodes_filename, dtype=int)
    core_pb_conn_n = np.loadtxt(core_pb_conn_n_filename, dtype=int)
    core_pb_conn_edges = np.loadtxt(core_pb_conn_edges_filename, dtype=int)

    core_pb_nodes = np.arange(core_pb_n, dtype=int)
    core_pb_conn_nodes = np.arange(core_pb_conn_n, dtype=int)
    core_pb_conn_m = np.shape(core_pb_conn_edges)[0]

    # Construct the core2pb_nodes list
    core2pb_nodes = core2pb_nodes_func(core_pb_conn_nodes, pb2core_nodes)

    # Load the core and periodic boundary node x- and y-coordinates
    core_pb_x = np.loadtxt(core_pb_x_filename)
    core_pb_y = np.loadtxt(core_pb_y_filename)
    if dim == 3:
        # Load the core and periodic boundary node z-coordinates
        core_pb_z = np.loadtxt(core_pb_z_filename)
    
    # Create nx.Graph, and add nodes before edges
    core_pb_graph = nx.Graph()
    core_pb_graph.add_nodes_from(core_pb_nodes)
    core_pb_graph.add_edges_from(core_pb_edges)

    # Length of each distinct edge in the core_pb_graph. Do note that
    # the number of distinct edges in the core_pb_graph is equal to the
    # number of edges from the core_pb_conn_graph.
    core_pb_l_edges = np.empty(core_pb_conn_m)

    # Initialize other essential parameters
    l_edge = 0.0
    core_node_0 = 0
    core_node_1 = 0
    pb_node_1 = 0
    core_node_0_pstn = np.asarray([])
    core_node_1_pstn = np.asarray([])
    pb_node_1_pstn = np.asarray([])

    # Calculate and store the length of each distinct edge in the
    # core_pb_graph
    for edge in range(core_pb_conn_m):
        core_node_0 = core_pb_conn_edges[edge, 0]
        core_node_1 = core_pb_conn_edges[edge, 1]

        # Edge is a distinct core edge
        if core_pb_graph.has_edge(core_node_0, core_node_1):
            # Determine coordinates for the location of each edge node
            if dim == 2:
                core_node_0_pstn = np.asarray(
                    [
                        core_pb_x[core_node_0],
                        core_pb_y[core_node_0]
                    ]
                )
                core_node_1_pstn = np.asarray(
                    [
                        core_pb_x[core_node_1],
                        core_pb_y[core_node_1]
                    ]
                )
            elif dim == 3:
                core_node_0_pstn = np.asarray(
                    [
                        core_pb_x[core_node_0],
                        core_pb_y[core_node_0],
                        core_pb_z[core_node_0]
                    ]
                )
                core_node_1_pstn = np.asarray(
                    [
                        core_pb_x[core_node_1],
                        core_pb_y[core_node_1],
                        core_pb_z[core_node_1]
                    ]
                )
            # Calculate edge length
            l_edge = np.linalg.norm(core_node_1_pstn-core_node_0_pstn)
            
        # Edge is a periodic edge, which is represented by two edges in
        # the core_pb_graph. Only one of these edges needs to be
        # interrogated in order to calculate the periodic edge length.
        else:
            # Identify all possible periodic boundary nodes that could
            # be involved in the periodic edge
            pb_nodes_1 = core2pb_nodes[core_node_1]
            # Determine the specific periodic boundary and core node
            # pair involved in the periodic edge
            for pb_node_1 in np.nditer(pb_nodes_1):
                pb_node_1 = int(pb_node_1)
                if core_pb_graph.has_edge(core_node_0, pb_node_1):
                    # Determine coordinates for the location of each
                    # edge node
                    if dim == 2:
                        node_0_pstn = np.asarray(
                            [
                                core_pb_x[core_node_0],
                                core_pb_y[core_node_0]
                            ]
                        )
                        pb_node_1_pstn = np.asarray(
                            [
                                core_pb_x[pb_node_1],
                                core_pb_y[pb_node_1]
                            ]
                        )
                    elif dim == 3:
                        node_0_pstn = np.asarray(
                            [
                                core_pb_x[core_node_0],
                                core_pb_y[core_node_0],
                                core_pb_z[core_node_0]
                            ]
                        )
                        pb_node_1_pstn = np.asarray(
                            [
                                core_pb_x[pb_node_1],
                                core_pb_y[pb_node_1],
                                core_pb_z[pb_node_1]
                            ]
                        )
                    # Calculate edge length
                    l_edge = np.linalg.norm(pb_node_1_pstn-node_0_pstn)
                    break
                else: pass
        
        # Store edge length
        core_pb_l_edges[edge] = l_edge
    
    # Return the length of distinct edges in the core_pb_graph
    return core_pb_l_edges

def core_pb_conn_graph_l_edges_calculation(
        filename_prefix: str,
        dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Length of each edge calculated in the graph capturing the
    periodic connections between the core nodes.

    This function loads fundamental graph constituents, calculates the
    length of each edge in the graph capturing the periodic connections
    between the core nodes, and returns this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Respective lengths of the core
        and periodic boundary edges in the graph capturing the periodic
        connections between the core nodes.
    
    """
    # Generate filenames
    core_pb_n_filename = filename_prefix + "-core_pb_n" + ".dat"
    core_pb_edges_filename = filename_prefix + "-core_pb_edges" + ".dat"
    core_pb_conn_edges_filename = (
        filename_prefix + "-core_pb_conn_edges" + ".dat"
    )
    core_pb_x_filename = filename_prefix + "-core_pb_x" + ".dat"
    core_pb_y_filename = filename_prefix + "-core_pb_y" + ".dat"
    if dim == 3:
        core_pb_z_filename = filename_prefix + "-core_pb_z" + ".dat"

    # Load fundamental graph constituents
    core_pb_n = np.loadtxt(core_pb_n_filename, dtype=int)
    core_pb_edges = np.loadtxt(core_pb_edges_filename, dtype=int)
    core_pb_conn_edges = np.loadtxt(core_pb_conn_edges_filename, dtype=int)

    core_pb_nodes = np.arange(core_pb_n, dtype=int)
    core_pb_conn_m = np.shape(core_pb_conn_edges)[0]

    # Load the core and periodic boundary node x- and y-coordinates
    core_pb_x = np.loadtxt(core_pb_x_filename)
    core_pb_y = np.loadtxt(core_pb_y_filename)
    if dim == 3:
        # Load the core and periodic boundary node z-coordinates
        core_pb_z = np.loadtxt(core_pb_z_filename)
    
    # Create nx.Graphs, and add nodes before edges
    core_pb_graph = nx.Graph()
    core_pb_graph.add_nodes_from(core_pb_nodes)
    core_pb_graph.add_edges_from(core_pb_edges)

    # Initialize lists for the length of the core edges and the length
    # of the periodic boundary edges in the core_pb_conn_graph
    core_pb_conn_l_core_edges = []
    core_pb_conn_l_pb_edges = []

    # Initialize other essential parameters
    core_node_0_pstn = np.asarray([])
    core_node_1_pstn = np.asarray([])

    # Calculate and store the length of each edge in the
    # core_pb_conn_graph
    for edge in range(core_pb_conn_m):
        core_node_0 = core_pb_conn_edges[edge, 0]
        core_node_1 = core_pb_conn_edges[edge, 1]

        # Determine coordinates for the location of each edge node
        if dim == 2:
            core_node_0_pstn = np.asarray(
                [
                    core_pb_x[core_node_0],
                    core_pb_y[core_node_0]
                ]
            )
            core_node_1_pstn = np.asarray(
                [
                    core_pb_x[core_node_1],
                    core_pb_y[core_node_1]
                ]
            )
        elif dim == 3:
            core_node_0_pstn = np.asarray(
                [
                    core_pb_x[core_node_0],
                    core_pb_y[core_node_0],
                    core_pb_z[core_node_0]
                ]
            )
            core_node_1_pstn = np.asarray(
                [
                    core_pb_x[core_node_1],
                    core_pb_y[core_node_1],
                    core_pb_z[core_node_1]
                ]
            )
        # Calculate edge length
        l_edge = np.linalg.norm(core_node_1_pstn-core_node_0_pstn)
        # Store edge length
        if core_pb_graph.has_edge(core_node_0, core_node_1): # Core edge
            core_pb_conn_l_core_edges.append(l_edge)
        else: # Periodic edge
            core_pb_conn_l_pb_edges.append(l_edge)
    
    # Convert lists for the length of the core edges and the length of
    # the periodic boundary edges in the core_pb_conn_graph to
    # np.ndarrays
    core_pb_conn_l_core_edges = np.asarray(core_pb_conn_l_core_edges)
    core_pb_conn_l_pb_edges = np.asarray(core_pb_conn_l_pb_edges)
    
    # Return the length of the core edges and the length of the periodic
    # boundary edges in the core_pb_conn_graph
    return core_pb_conn_l_core_edges, core_pb_conn_l_pb_edges

def core_pb_graph_l_edges(filename_prefix: str, dim: int) -> None:
    """Length of each edge calculated in the graph capturing the spatial
    topology of the core and periodic boundary nodes and edges.

    This function calls upon a helper function to calculate the length
    of each edge in the graph capturing the spatial topology of the core
    and periodic boundary nodes and edges, and saves this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
    
    """
    # Generate filename
    core_pb_l_edges_filename = filename_prefix + "-core_pb_l_edges" + ".dat"
    
    # Length of each distinct edge in the core_pb_graph
    core_pb_l_edges = core_pb_graph_l_edges_calculation(filename_prefix, dim)
    
    # Save length of distinct edges in the core_pb_graph
    np.savetxt(core_pb_l_edges_filename, core_pb_l_edges)

def core_pb_conn_graph_l_edges(filename_prefix: str, dim: int) -> None:
    """Length of each edge calculated in the graph capturing the
    periodic connections between the core nodes.

    This function calls upon a helper function to calculate the length
    of each edge in the graph capturing the periodic connections between
    the core nodes, and saves this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
    
    """
    # Generate filenames
    core_pb_conn_l_core_edges_filename = (
        filename_prefix + "-core_pb_conn_l_core_edges" + ".dat"
    )
    core_pb_conn_l_pb_edges_filename = (
        filename_prefix + "-core_pb_conn_l_pb_edges" + ".dat"
    )
    
    # Length of the core edges and length of the periodic boundary edges
    # in the core_pb_conn_graph
    core_pb_conn_l_core_edges, core_pb_conn_l_pb_edges = (
        core_pb_conn_graph_l_edges_calculation(filename_prefix, dim)
    )
    
    # Save the length of the core edges and the length of the periodic
    # boundary edges in the core_pb_conn_graph
    np.savetxt(core_pb_conn_l_core_edges_filename, core_pb_conn_l_core_edges)
    np.savetxt(core_pb_conn_l_pb_edges_filename, core_pb_conn_l_pb_edges)

def swidt_network_graph_l_edges(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        config: int,
        pruning: int,
        graph: str) -> None:
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
        graph (str): Graph type in which the length of each edge will be
        calculated; here, either "core_pb" (corresponding to the graph
        capturing the spatial topology of the core and periodic boundary
        nodes and edges) or "core_pb_conn" (corresponding to the graph
        capturing the periodic connections between the core nodes) are
        applicable.
    
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
    swidt_filename_prefix = (
        filename_str(network, date, batch, sample)
        + f"C{config:d}" + f"P{pruning:d}"
    )
    # Calculate and save the length of each edge in the spider
    # web-inspired Delaunay-triangulated networks.
    if graph == "core_pb":
        # Graph capturing the spatial topology of the core and periodic
        # boundary nodes and edges
        core_pb_graph_l_edges(swidt_filename_prefix, dim)
    elif graph == "core_pb_conn":
        # Graph capturing the periodic connections between the core
        # cross-linker nodes.
        core_pb_conn_graph_l_edges(swidt_filename_prefix, dim)

def core_pb_graph_l_nrmlzd_edges(
        filename: str,
        filename_prefix: str,
        dim: int) -> None:
    """Normalized length of each edge calculated in the graph capturing
    the spatial topology of the core and periodic boundary nodes and
    edges.

    This function loads the simulation box size, calls upon a helper
    function to calculate the length of each edge in the graph capturing
    the spatial topology of the core and periodic boundary nodes and
    edges, normalizes the edge lengths by the simulation box size, and
    saves this information.

    Args:
        filename (str): Baseline filename for the simulation box size.
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
    
    """
    # Generate filename
    L_filename = filename + "-L" + ".dat"
    core_pb_l_nrmlzd_edges_filename = (
        filename_prefix + "-core_pb_l_nrmlzd_edges" + ".dat"
    )
    
    # Load L
    L = np.loadtxt(L_filename)
    
    # Length of each distinct edge in the core_pb_graph
    core_pb_l_edges = core_pb_graph_l_edges_calculation(filename_prefix, dim)
    
    # Edge length normalization by L*sqrt(dim)
    core_pb_l_nrmlzd_edges = core_pb_l_edges / (L*np.sqrt(dim))

    # Save normalized length of distinct edges in the core_pb_graph
    np.savetxt(core_pb_l_nrmlzd_edges_filename, core_pb_l_nrmlzd_edges)

def core_pb_conn_graph_l_nrmlzd_edges(
        filename: str,
        filename_prefix: str,
        dim: int) -> None:
    """Normalized length of each edge calculated in the graph capturing
    the periodic connections between the core nodes.

    This function loads the simulation box size, calls upon a helper
    function to calculate the length of each edge calculated in the
    graph capturing the periodic connections between the core nodes,
    normalizes the edge lengths by the simulation box size, and saves
    this information.

    Args:
        filename (str): Baseline filename for the simulation box size.
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
        dim (int): Physical dimensionality of the network; either 2 or 3
        (for two-dimensional or three-dimensional networks).
    
    """
    # Generate filename
    L_filename = filename + "-L" + ".dat"
    core_pb_conn_l_nrmlzd_core_edges_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_core_edges" + ".dat"
    )
    core_pb_conn_l_nrmlzd_pb_edges_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_pb_edges" + ".dat"
    )
    
    # Load L
    L = np.loadtxt(L_filename)
    
    # Length of the core edges and the length of the periodic boundary
    # edges in the core_pb_conn_graph
    core_pb_conn_l_core_edges, core_pb_conn_l_pb_edges = (
        core_pb_conn_graph_l_edges_calculation(filename_prefix, dim)
    )
    
    # Edge length normalization by L*sqrt(dim)
    core_pb_conn_l_nrmlzd_core_edges = (
        core_pb_conn_l_core_edges / (L*np.sqrt(dim))
    )
    core_pb_conn_l_nrmlzd_pb_edges = core_pb_conn_l_pb_edges / (L*np.sqrt(dim))

    # Save the normalized length of the core edges and the normalized
    # length of the periodic boundary edges in the core_pb_conn_graph
    np.savetxt(
        core_pb_conn_l_nrmlzd_core_edges_filename, core_pb_conn_l_nrmlzd_core_edges)
    np.savetxt(
        core_pb_conn_l_nrmlzd_pb_edges_filename, core_pb_conn_l_nrmlzd_pb_edges)

def swidt_network_graph_l_nrmlzd_edges(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        config: int,
        pruning: int,
        graph: str) -> None:
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
        graph (str): Graph type in which the normalized length of each
        edge will be calculated; here, either "core_pb" (corresponding
        to the graph capturing the spatial topology of the core and
        periodic boundary nodes and edges) or "core_pb_conn"
        (corresponding to the graph capturing the periodic connections
        between the core nodes) are applicable.
    
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
    filename = filename_str(network, date, batch, sample)
    swidt_filename_prefix = filename + f"C{config:d}" + f"P{pruning:d}"
    # Calculate and save the normalized length of each edge in the
    # spider web-inspired Delaunay-triangulated networks.
    if graph == "core_pb":
        # Graph capturing the spatial topology of the core and periodic
        # boundary nodes and edges
        core_pb_graph_l_nrmlzd_edges(filename, swidt_filename_prefix, dim)
    elif graph == "core_pb_conn":
        # Graph capturing the periodic connections between the core
        # cross-linker nodes.
        core_pb_conn_graph_l_nrmlzd_edges(filename, swidt_filename_prefix, dim)

def core_pb_graph_l_edges_dim_2_cmpnts_calculation(
        filename_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Two-dimensional length components of each edge calculated in the
    graph capturing the spatial topology of the core and periodic
    boundary nodes and edges.

    This function loads fundamental graph constituents, calculates the
    two-dimensional length components of each edge in the graph
    capturing the spatial topology of the core and periodic boundary
    nodes and edges, and returns this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Respective two-dimensional x- and
        y-length components of each edge calculated in the graph
        capturing the spatial topology of the core and periodic boundary
        nodes and edges.
    
    """
    # Initialize random number generator
    rng = np.random.default_rng()

    # Generate filenames
    core_pb_n_filename = filename_prefix + "-core_pb_n" + ".dat"
    core_pb_edges_filename = filename_prefix + "-core_pb_edges" + ".dat"
    pb2core_nodes_filename = filename_prefix + "-pb2core_nodes" + ".dat"
    core_pb_conn_n_filename = filename_prefix + "-core_pb_conn_n" + ".dat"
    core_pb_conn_edges_filename = (
        filename_prefix + "-core_pb_conn_edges" + ".dat"
    )
    core_pb_x_filename = filename_prefix + "-core_pb_x" + ".dat"
    core_pb_y_filename = filename_prefix + "-core_pb_y" + ".dat"

    # Load fundamental graph constituents
    core_pb_n = np.loadtxt(core_pb_n_filename, dtype=int)
    core_pb_edges = np.loadtxt(core_pb_edges_filename, dtype=int)
    pb2core_nodes = np.loadtxt(pb2core_nodes_filename, dtype=int)
    core_pb_conn_n = np.loadtxt(core_pb_conn_n_filename, dtype=int)
    core_pb_conn_edges = np.loadtxt(core_pb_conn_edges_filename, dtype=int)

    core_pb_nodes = np.arange(core_pb_n, dtype=int)
    core_pb_conn_nodes = np.arange(core_pb_conn_n, dtype=int)
    core_pb_conn_m = np.shape(core_pb_conn_edges)[0]

    # Construct the core2pb_nodes list
    core2pb_nodes = core2pb_nodes_func(core_pb_conn_nodes, pb2core_nodes)

    # Load the core and periodic boundary node x- and y-coordinates
    core_pb_x = np.loadtxt(core_pb_x_filename)
    core_pb_y = np.loadtxt(core_pb_y_filename)
    
    # Create nx.Graphs, and add nodes before edges
    core_pb_graph = nx.Graph()
    core_pb_graph.add_nodes_from(core_pb_nodes)
    core_pb_graph.add_edges_from(core_pb_edges)

    # Length components of each distinct edge in the core_pb_graph. Do
    # note that the number of distinct edges in the core_pb_graph is
    # equal to the number of edges from the core_pb_conn_graph.
    core_pb_l_edges_x_cmpnt = np.empty(core_pb_conn_m)
    core_pb_l_edges_y_cmpnt = np.empty(core_pb_conn_m)

    # Initialize other essential parameters
    l_edge_x_cmpnt = 0.0
    l_edge_y_cmpnt = 0.0
    l_edge_x_cmpnt_opt_0 = 0.0
    l_edge_y_cmpnt_opt_0 = 0.0
    l_edge_x_cmpnt_opt_1 = 0.0
    l_edge_y_cmpnt_opt_1 = 0.0

    # Calculate and store the length components of each distinct edge in
    # the core_pb_graph
    for edge in range(core_pb_conn_m):
        core_node_0 = core_pb_conn_edges[edge, 0]
        core_node_1 = core_pb_conn_edges[edge, 1]

        # Edge is a distinct core edge
        if core_pb_graph.has_edge(core_node_0, core_node_1):
            core_node_0_x = core_pb_x[core_node_0]
            core_node_1_x = core_pb_x[core_node_1]
            l_edge_x_cmpnt = core_node_1_x - core_node_0_x

            core_node_0_y = core_pb_y[core_node_0]
            core_node_1_y = core_pb_y[core_node_1]
            l_edge_y_cmpnt = core_node_1_y - core_node_0_y
        # Edge is a periodic edge, which is represented by two edges in
        # the core_pb_graph. Each of these edges needs to be
        # interrogated in order to calculate the periodic edge length
        # components.
        else:
            # Identify all possible periodic boundary nodes that could
            # be involved in each periodic edge
            pb_nodes_0 = core2pb_nodes[core_node_0]
            pb_nodes_1 = core2pb_nodes[core_node_1]
            # Determine the specific periodic boundary and core node
            # pair involved in each periodic edge
            for pb_node_0 in np.nditer(pb_nodes_0):
                pb_node_0 = int(pb_node_0)
                if core_pb_graph.has_edge(pb_node_0, core_node_1):
                    pb_node_0_x = core_pb_x[pb_node_0]
                    core_node_1_x = core_pb_x[core_node_1]
                    l_edge_x_cmpnt_opt_0 = core_node_1_x - pb_node_0_x

                    pb_node_0_y = core_pb_y[pb_node_0]
                    core_node_1_y = core_pb_y[core_node_1]
                    l_edge_y_cmpnt_opt_0 = core_node_1_y - pb_node_0_y
                    
                    break
                else: pass
            for pb_node_1 in np.nditer(pb_nodes_1):
                pb_node_1 = int(pb_node_1)
                if core_pb_graph.has_edge(core_node_0, pb_node_1):
                    core_node_0_x = core_pb_x[core_node_0]
                    pb_node_1_x = core_pb_x[pb_node_1]
                    l_edge_x_cmpnt_opt_1 = pb_node_1_x - core_node_0_x

                    core_node_0_y = core_pb_y[core_node_0]
                    pb_node_1_y = core_pb_y[pb_node_1]
                    l_edge_y_cmpnt_opt_1 = pb_node_1_y - core_node_0_y
                    
                    break
                else: pass
            # Interrogate each periodic edge in order to determine the
            # periodic edge length components
            if l_edge_x_cmpnt_opt_0 == l_edge_x_cmpnt_opt_1:
                l_edge_x_cmpnt = l_edge_x_cmpnt_opt_0
            else:
                coin_flip = rng.integers(2, dtype=int)
                if coin_flip == 0:
                    l_edge_x_cmpnt = l_edge_x_cmpnt_opt_0
                else:
                    l_edge_x_cmpnt = l_edge_x_cmpnt_opt_1
            
            if l_edge_y_cmpnt_opt_0 == l_edge_y_cmpnt_opt_1:
                l_edge_y_cmpnt = l_edge_y_cmpnt_opt_0
            else:
                coin_flip = rng.integers(2, dtype=int)
                if coin_flip == 0:
                    l_edge_y_cmpnt = l_edge_y_cmpnt_opt_0
                else:
                    l_edge_y_cmpnt = l_edge_y_cmpnt_opt_1
        # Store edge length components
        core_pb_l_edges_x_cmpnt[edge] = l_edge_x_cmpnt
        core_pb_l_edges_y_cmpnt[edge] = l_edge_y_cmpnt
    
    # Return the length components of distinct edges in the
    # core_pb_graph
    return core_pb_l_edges_x_cmpnt, core_pb_l_edges_y_cmpnt

def core_pb_graph_l_edges_dim_3_cmpnts_calculation(
        filename_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-dimensional length components of each edge calculated in
    the graph capturing the spatial topology of the core and periodic
    boundary nodes and edges.

    This function loads fundamental graph constituents, calculates the
    three-dimensional length components of each edge in the graph
    capturing the spatial topology of the core and periodic boundary
    nodes and edges, and returns this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Respective
        three-dimensional x-, y-, and z-length components of each edge
        calculated in the graph capturing the spatial topology of the
        core and periodic boundary nodes and edges.
    
    """
    # Initialize random number generator
    rng = np.random.default_rng()

    # Generate filenames
    core_pb_n_filename = filename_prefix + "-core_pb_n" + ".dat"
    core_pb_edges_filename = filename_prefix + "-core_pb_edges" + ".dat"
    pb2core_nodes_filename = filename_prefix + "-pb2core_nodes" + ".dat"
    core_pb_conn_n_filename = filename_prefix + "-core_pb_conn_n" + ".dat"
    core_pb_conn_edges_filename = (
        filename_prefix + "-core_pb_conn_edges" + ".dat"
    )
    core_pb_x_filename = filename_prefix + "-core_pb_x" + ".dat"
    core_pb_y_filename = filename_prefix + "-core_pb_y" + ".dat"
    core_pb_z_filename = filename_prefix + "-core_pb_z" + ".dat"

    # Load fundamental graph constituents
    core_pb_n = np.loadtxt(core_pb_n_filename, dtype=int)
    core_pb_edges = np.loadtxt(core_pb_edges_filename, dtype=int)
    pb2core_nodes = np.loadtxt(pb2core_nodes_filename, dtype=int)
    core_pb_conn_n = np.loadtxt(core_pb_conn_n_filename, dtype=int)
    core_pb_conn_edges = np.loadtxt(core_pb_conn_edges_filename, dtype=int)

    core_pb_nodes = np.arange(core_pb_n, dtype=int)
    core_pb_conn_nodes = np.arange(core_pb_conn_n, dtype=int)
    core_pb_conn_m = np.shape(core_pb_conn_edges)[0]

    # Construct the core2pb_nodes list
    core2pb_nodes = core2pb_nodes_func(core_pb_conn_nodes, pb2core_nodes)

    # Load the core and periodic boundary node x-, y-, and z-coordinates
    core_pb_x = np.loadtxt(core_pb_x_filename)
    core_pb_y = np.loadtxt(core_pb_y_filename)
    core_pb_z = np.loadtxt(core_pb_z_filename)
    
    # Create nx.Graphs, and add nodes before edges
    core_pb_graph = nx.Graph()
    core_pb_graph.add_nodes_from(core_pb_nodes)
    core_pb_graph.add_edges_from(core_pb_edges)

    # Length components of each distinct edge in the core_pb_graph. Do
    # note that the number of distinct edges in the core_pb_graph is
    # equal to the number of edges from the core_pb_conn_graph.
    core_pb_l_edges_x_cmpnt = np.empty(core_pb_conn_m)
    core_pb_l_edges_y_cmpnt = np.empty(core_pb_conn_m)
    core_pb_l_edges_z_cmpnt = np.empty(core_pb_conn_m)

    # Initialize other essential parameters
    l_edge_x_cmpnt = 0.0
    l_edge_y_cmpnt = 0.0
    l_edge_z_cmpnt = 0.0
    l_edge_x_cmpnt_opt_0 = 0.0
    l_edge_y_cmpnt_opt_0 = 0.0
    l_edge_z_cmpnt_opt_0 = 0.0
    l_edge_x_cmpnt_opt_1 = 0.0
    l_edge_y_cmpnt_opt_1 = 0.0
    l_edge_z_cmpnt_opt_1 = 0.0

    # Calculate and store the length components of each distinct edge in
    # the core_pb_graph
    for edge in range(core_pb_conn_m):
        core_node_0 = core_pb_conn_edges[edge, 0]
        core_node_1 = core_pb_conn_edges[edge, 1]

        # Edge is a distinct core edge
        if core_pb_graph.has_edge(core_node_0, core_node_1):
            core_node_0_x = core_pb_x[core_node_0]
            core_node_1_x = core_pb_x[core_node_1]
            l_edge_x_cmpnt = core_node_1_x - core_node_0_x

            core_node_0_y = core_pb_y[core_node_0]
            core_node_1_y = core_pb_y[core_node_1]
            l_edge_y_cmpnt = core_node_1_y - core_node_0_y

            core_node_0_z = core_pb_z[core_node_0]
            core_node_1_z = core_pb_z[core_node_1]
            l_edge_z_cmpnt = core_node_1_z - core_node_0_z
        # Edge is a periodic edge, which is represented by two edges in
        # the core_pb_graph. Each of these edges needs to be
        # interrogated in order to calculate the periodic edge length
        # components.
        else:
            # Identify all possible periodic boundary nodes that could
            # be involved in each periodic edge
            pb_nodes_0 = core2pb_nodes[core_node_0]
            pb_nodes_1 = core2pb_nodes[core_node_1]
            # Determine the specific periodic boundary and core node
            # pair involved in each periodic edge
            for pb_node_0 in np.nditer(pb_nodes_0):
                pb_node_0 = int(pb_node_0)
                if core_pb_graph.has_edge(pb_node_0, core_node_1):
                    pb_node_0_x = core_pb_x[pb_node_0]
                    core_node_1_x = core_pb_x[core_node_1]
                    l_edge_x_cmpnt_opt_0 = core_node_1_x - pb_node_0_x

                    pb_node_0_y = core_pb_y[pb_node_0]
                    core_node_1_y = core_pb_y[core_node_1]
                    l_edge_y_cmpnt_opt_0 = core_node_1_y - pb_node_0_y

                    pb_node_0_z = core_pb_z[pb_node_0]
                    core_node_1_z = core_pb_z[core_node_1]
                    l_edge_z_cmpnt_opt_0 = core_node_1_z - pb_node_0_z
                    
                    break
                else: pass
            for pb_node_1 in np.nditer(pb_nodes_1):
                pb_node_1 = int(pb_node_1)
                if core_pb_graph.has_edge(core_node_0, pb_node_1):
                    core_node_0_x = core_pb_x[core_node_0]
                    pb_node_1_x = core_pb_x[pb_node_1]
                    l_edge_x_cmpnt_opt_1 = pb_node_1_x - core_node_0_x

                    core_node_0_y = core_pb_y[core_node_0]
                    pb_node_1_y = core_pb_y[pb_node_1]
                    l_edge_y_cmpnt_opt_1 = pb_node_1_y - core_node_0_y

                    core_node_0_z = core_pb_z[core_node_0]
                    pb_node_1_z = core_pb_z[pb_node_1]
                    l_edge_z_cmpnt_opt_1 = pb_node_1_z - core_node_0_z
                    
                    break
                else: pass
            # Interrogate each periodic edge in order to determine the
            # periodic edge length components
            if l_edge_x_cmpnt_opt_0 == l_edge_x_cmpnt_opt_1:
                l_edge_x_cmpnt = l_edge_x_cmpnt_opt_0
            else:
                coin_flip = rng.integers(2, dtype=int)
                if coin_flip == 0:
                    l_edge_x_cmpnt = l_edge_x_cmpnt_opt_0
                else:
                    l_edge_x_cmpnt = l_edge_x_cmpnt_opt_1
            
            if l_edge_y_cmpnt_opt_0 == l_edge_y_cmpnt_opt_1:
                l_edge_y_cmpnt = l_edge_y_cmpnt_opt_0
            else:
                coin_flip = rng.integers(2, dtype=int)
                if coin_flip == 0:
                    l_edge_y_cmpnt = l_edge_y_cmpnt_opt_0
                else:
                    l_edge_y_cmpnt = l_edge_y_cmpnt_opt_1
            
            if l_edge_z_cmpnt_opt_0 == l_edge_z_cmpnt_opt_1:
                l_edge_z_cmpnt = l_edge_z_cmpnt_opt_0
            else:
                coin_flip = rng.integers(2, dtype=int)
                if coin_flip == 0:
                    l_edge_z_cmpnt = l_edge_z_cmpnt_opt_0
                else:
                    l_edge_z_cmpnt = l_edge_z_cmpnt_opt_1
        # Store edge length components
        core_pb_l_edges_x_cmpnt[edge] = l_edge_x_cmpnt
        core_pb_l_edges_y_cmpnt[edge] = l_edge_y_cmpnt
        core_pb_l_edges_z_cmpnt[edge] = l_edge_z_cmpnt
    
    # Return the length components of distinct edges in the
    # core_pb_graph
    return (
        core_pb_l_edges_x_cmpnt,
        core_pb_l_edges_y_cmpnt,
        core_pb_l_edges_z_cmpnt
    )

def core_pb_conn_graph_l_edges_dim_2_cmpnts_calculation(
        filename_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-dimensional length components of each edge calculated in the
    graph capturing the periodic connections between the core nodes.

    This function loads fundamental graph constituents, calculates the
    two-dimensional length components of each edge in the graph
    capturing the periodic connections between the core nodes, and
    returns this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Respective two-dimensional x- and y-length components of each
        core and periodic edge calculated in the graph capturing the
        periodic connections between the core nodes.
    
    """
    # Generate filenames
    core_pb_n_filename = filename_prefix + "-core_pb_n" + ".dat"
    core_pb_edges_filename = filename_prefix + "-core_pb_edges" + ".dat"
    core_pb_conn_edges_filename = (
        filename_prefix + "-core_pb_conn_edges" + ".dat"
    )
    core_pb_x_filename = filename_prefix + "-core_pb_x" + ".dat"
    core_pb_y_filename = filename_prefix + "-core_pb_y" + ".dat"

    # Load fundamental graph constituents
    core_pb_n = np.loadtxt(core_pb_n_filename, dtype=int)
    core_pb_edges = np.loadtxt(core_pb_edges_filename, dtype=int)
    core_pb_conn_edges = np.loadtxt(core_pb_conn_edges_filename, dtype=int)

    core_pb_nodes = np.arange(core_pb_n, dtype=int)
    core_pb_conn_m = np.shape(core_pb_conn_edges)[0]

    # Load the core and periodic boundary node x- and y-coordinates
    core_pb_x = np.loadtxt(core_pb_x_filename)
    core_pb_y = np.loadtxt(core_pb_y_filename)
    
    # Create nx.Graphs, and add nodes before edges
    core_pb_graph = nx.Graph()
    core_pb_graph.add_nodes_from(core_pb_nodes)
    core_pb_graph.add_edges_from(core_pb_edges)

    # Initialize lists for the length components of the core edges and
    # the length components of the periodic boundary edges in the
    # core_pb_conn_graph
    core_pb_conn_l_core_edges_x_cmpnt = []
    core_pb_conn_l_core_edges_y_cmpnt = []
    core_pb_conn_l_pb_edges_x_cmpnt = []
    core_pb_conn_l_pb_edges_y_cmpnt = []

    # Calculate and store the length components of each edge in the
    # core_pb_conn_graph
    for edge in range(core_pb_conn_m):
        core_node_0 = core_pb_conn_edges[edge, 0]
        core_node_1 = core_pb_conn_edges[edge, 1]

        core_node_0_x = core_pb_x[core_node_0]
        core_node_1_x = core_pb_x[core_node_1]
        l_edge_x_cmpnt = core_node_1_x - core_node_0_x

        core_node_0_y = core_pb_y[core_node_0]
        core_node_1_y = core_pb_y[core_node_1]
        l_edge_y_cmpnt = core_node_1_y - core_node_0_y

        if core_pb_graph.has_edge(core_node_0, core_node_1): # Core edge
            core_pb_conn_l_core_edges_x_cmpnt.append(l_edge_x_cmpnt)
            core_pb_conn_l_core_edges_y_cmpnt.append(l_edge_y_cmpnt)
        else: # Periodic edge
            core_pb_conn_l_pb_edges_x_cmpnt.append(l_edge_x_cmpnt)
            core_pb_conn_l_pb_edges_y_cmpnt.append(l_edge_y_cmpnt)
    
    # Convert lists for the length components of the core edges and the
    # length components of the periodic boundary edges in the
    # core_pb_conn_graph to np.ndarrays
    core_pb_conn_l_core_edges_x_cmpnt = (
        np.asarray(core_pb_conn_l_core_edges_x_cmpnt)
    )
    core_pb_conn_l_core_edges_y_cmpnt = (
        np.asarray(core_pb_conn_l_core_edges_y_cmpnt)
    )
    core_pb_conn_l_pb_edges_x_cmpnt = (
        np.asarray(core_pb_conn_l_pb_edges_x_cmpnt)
    )
    core_pb_conn_l_pb_edges_y_cmpnt = (
        np.asarray(core_pb_conn_l_pb_edges_y_cmpnt)
    )

    # Return the length components of the core edges and the length
    # components of the periodic boundary edges in the
    # core_pb_conn_graph
    return (
        core_pb_conn_l_core_edges_x_cmpnt,
        core_pb_conn_l_core_edges_y_cmpnt,
        core_pb_conn_l_pb_edges_x_cmpnt,
        core_pb_conn_l_pb_edges_y_cmpnt
    )

def core_pb_conn_graph_l_edges_dim_3_cmpnts_calculation(
        filename_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three-dimensional length components of each edge calculated in
    the graph capturing the periodic connections between the core nodes.

    This function loads fundamental graph constituents, calculates the
    three-dimensional length components of each edge in the graph
    capturing the periodic connections between the core nodes, and
    returns this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Respective three-dimensional x-, y-, and z-length components of
        each core and periodic edge calculated in the graph capturing
        the periodic connections between the core nodes.
    
    """
    # Generate filenames
    core_pb_n_filename = filename_prefix + "-core_pb_n" + ".dat"
    core_pb_edges_filename = filename_prefix + "-core_pb_edges" + ".dat"
    core_pb_conn_edges_filename = (
        filename_prefix + "-core_pb_conn_edges" + ".dat"
    )
    core_pb_x_filename = filename_prefix + "-core_pb_x" + ".dat"
    core_pb_y_filename = filename_prefix + "-core_pb_y" + ".dat"
    core_pb_z_filename = filename_prefix + "-core_pb_z" + ".dat"

    # Load fundamental graph constituents
    core_pb_n = np.loadtxt(core_pb_n_filename, dtype=int)
    core_pb_edges = np.loadtxt(core_pb_edges_filename, dtype=int)
    core_pb_conn_edges = np.loadtxt(core_pb_conn_edges_filename, dtype=int)

    core_pb_nodes = np.arange(core_pb_n, dtype=int)
    core_pb_conn_m = np.shape(core_pb_conn_edges)[0]

    # Load the core and periodic boundary node x-, y-, and z-coordinates
    core_pb_x = np.loadtxt(core_pb_x_filename)
    core_pb_y = np.loadtxt(core_pb_y_filename)
    core_pb_z = np.loadtxt(core_pb_z_filename)
    
    # Create nx.Graphs, and add nodes before edges
    core_pb_graph = nx.Graph()
    core_pb_graph.add_nodes_from(core_pb_nodes)
    core_pb_graph.add_edges_from(core_pb_edges)

    # Initialize lists for the length components of the core edges and
    # the length components of the periodic boundary edges in the
    # core_pb_conn_graph
    core_pb_conn_l_core_edges_x_cmpnt = []
    core_pb_conn_l_core_edges_y_cmpnt = []
    core_pb_conn_l_core_edges_z_cmpnt = []
    core_pb_conn_l_pb_edges_x_cmpnt = []
    core_pb_conn_l_pb_edges_y_cmpnt = []
    core_pb_conn_l_pb_edges_z_cmpnt = []

    # Calculate and store the length components of each edge in the
    # core_pb_conn_graph
    for edge in range(core_pb_conn_m):
        core_node_0 = core_pb_conn_edges[edge, 0]
        core_node_1 = core_pb_conn_edges[edge, 1]

        core_node_0_x = core_pb_x[core_node_0]
        core_node_1_x = core_pb_x[core_node_1]
        l_edge_x_cmpnt = core_node_1_x - core_node_0_x

        core_node_0_y = core_pb_y[core_node_0]
        core_node_1_y = core_pb_y[core_node_1]
        l_edge_y_cmpnt = core_node_1_y - core_node_0_y

        core_node_0_z = core_pb_z[core_node_0]
        core_node_1_z = core_pb_z[core_node_1]
        l_edge_z_cmpnt = core_node_1_z - core_node_0_z

        if core_pb_graph.has_edge(core_node_0, core_node_1): # Core edge
            core_pb_conn_l_core_edges_x_cmpnt.append(l_edge_x_cmpnt)
            core_pb_conn_l_core_edges_y_cmpnt.append(l_edge_y_cmpnt)
            core_pb_conn_l_core_edges_z_cmpnt.append(l_edge_z_cmpnt)
        else: # Periodic edge
            core_pb_conn_l_pb_edges_x_cmpnt.append(l_edge_x_cmpnt)
            core_pb_conn_l_pb_edges_y_cmpnt.append(l_edge_y_cmpnt)
            core_pb_conn_l_pb_edges_z_cmpnt.append(l_edge_z_cmpnt)
    
    # Convert lists for the length components of the core edges and the
    # length components of the periodic boundary edges in the
    # core_pb_conn_graph to np.ndarrays
    core_pb_conn_l_core_edges_x_cmpnt = (
        np.asarray(core_pb_conn_l_core_edges_x_cmpnt)
    )
    core_pb_conn_l_core_edges_y_cmpnt = (
        np.asarray(core_pb_conn_l_core_edges_y_cmpnt)
    )
    core_pb_conn_l_core_edges_z_cmpnt = (
        np.asarray(core_pb_conn_l_core_edges_z_cmpnt)
    )
    core_pb_conn_l_pb_edges_x_cmpnt = (
        np.asarray(core_pb_conn_l_pb_edges_x_cmpnt)
    )
    core_pb_conn_l_pb_edges_y_cmpnt = (
        np.asarray(core_pb_conn_l_pb_edges_y_cmpnt)
    )
    core_pb_conn_l_pb_edges_z_cmpnt = (
        np.asarray(core_pb_conn_l_pb_edges_z_cmpnt)
    )

    # Return the length components of the core edges and the length
    # components of the periodic boundary edges in the
    # core_pb_conn_graph
    return (
        core_pb_conn_l_core_edges_x_cmpnt,
        core_pb_conn_l_core_edges_y_cmpnt,
        core_pb_conn_l_core_edges_z_cmpnt,
        core_pb_conn_l_pb_edges_x_cmpnt,
        core_pb_conn_l_pb_edges_y_cmpnt,
        core_pb_conn_l_pb_edges_z_cmpnt
    )

def core_pb_graph_l_edges_dim_2_cmpnts(filename_prefix: str) -> None:
    """Two-dimensional length components of each edge calculated in the
    graph capturing the spatial topology of the core and periodic
    boundary nodes and edges.

    This function calls upon a helper function to calculate the
    two-dimensional length components of each edge in the graph
    capturing the spatial topology of the core and periodic boundary
    nodes and edges, and saves this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    """
    # Generate filenames
    core_pb_l_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_l_edges_x_cmpnt" + ".dat"
    )
    core_pb_l_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_l_edges_y_cmpnt" + ".dat"
    )

    # Length components of each distinct edge in the core_pb_graph
    core_pb_l_edges_x_cmpnt, core_pb_l_edges_y_cmpnt = (
        core_pb_graph_l_edges_dim_2_cmpnts_calculation(filename_prefix)
    )

    # Save length components of distinct edges in the core_pb_graph
    np.savetxt(core_pb_l_edges_x_cmpnt_filename, core_pb_l_edges_x_cmpnt)
    np.savetxt(core_pb_l_edges_y_cmpnt_filename, core_pb_l_edges_y_cmpnt)

def core_pb_graph_l_edges_dim_3_cmpnts(filename_prefix: str) -> None:
    """Three-dimensional length components of each edge calculated in
    the graph capturing the spatial topology of the core and periodic
    boundary nodes and edges.

    This function calls upon a helper function to calculate the
    three-dimensional length components of each edge in the graph
    capturing the spatial topology of the core and periodic boundary
    nodes and edges, and saves this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    """
    # Generate filenames
    core_pb_l_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_l_edges_x_cmpnt" + ".dat"
    )
    core_pb_l_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_l_edges_y_cmpnt" + ".dat"
    )
    core_pb_l_edges_z_cmpnt_filename = (
        filename_prefix + "-core_pb_l_edges_z_cmpnt" + ".dat"
    )

    # Length components of each distinct edge in the core_pb_graph
    core_pb_l_edges_x_cmpnt, core_pb_l_edges_y_cmpnt, core_pb_l_edges_z_cmpnt = (
        core_pb_graph_l_edges_dim_3_cmpnts_calculation(filename_prefix)
    )

    # Save length components of distinct edges in the core_pb_graph
    np.savetxt(core_pb_l_edges_x_cmpnt_filename, core_pb_l_edges_x_cmpnt)
    np.savetxt(core_pb_l_edges_y_cmpnt_filename, core_pb_l_edges_y_cmpnt)
    np.savetxt(core_pb_l_edges_z_cmpnt_filename, core_pb_l_edges_z_cmpnt)

def core_pb_conn_graph_l_edges_dim_2_cmpnts(filename_prefix: str) -> None:
    """Two-dimensional length components of each edge calculated in the
    graph capturing the periodic connections between the core nodes.

    This function calls upon a helper function to calculate the
    two-dimensional length components of each edge in the graph
    capturing the periodic connections between the core nodes, and saves
    this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    """
    # Generate filenames
    core_pb_conn_l_core_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_core_edges_x_cmpnt" + ".dat"
    )
    core_pb_conn_l_core_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_core_edges_y_cmpnt" + ".dat"
    )
    core_pb_conn_l_pb_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_pb_edges_x_cmpnt" + ".dat"
    )
    core_pb_conn_l_pb_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_pb_edges_y_cmpnt" + ".dat"
    )

    # Length components of the core edges and length components of the
    # periodic boundary edges in the core_pb_conn_graph
    (core_pb_conn_l_core_edges_x_cmpnt, core_pb_conn_l_core_edges_y_cmpnt,
     core_pb_conn_l_pb_edges_x_cmpnt, core_pb_conn_l_pb_edges_y_cmpnt) = (
         core_pb_conn_graph_l_edges_dim_2_cmpnts_calculation(filename_prefix)
    )

    # Save the length components of the core edges and the length
    # components of the periodic boundary edges in the
    # core_pb_conn_graph
    np.savetxt(
        core_pb_conn_l_core_edges_x_cmpnt_filename,
        core_pb_conn_l_core_edges_x_cmpnt)
    np.savetxt(
        core_pb_conn_l_core_edges_y_cmpnt_filename,
        core_pb_conn_l_core_edges_y_cmpnt)
    np.savetxt(
        core_pb_conn_l_pb_edges_x_cmpnt_filename,
        core_pb_conn_l_pb_edges_x_cmpnt)
    np.savetxt(
        core_pb_conn_l_pb_edges_y_cmpnt_filename,
        core_pb_conn_l_pb_edges_y_cmpnt)

def core_pb_conn_graph_l_edges_dim_3_cmpnts(filename_prefix: str) -> None:
    """Three-dimensional length components of each edge calculated in
    the graph capturing the periodic connections between the core nodes.

    This function calls upon a helper function to calculate the
    three-dimensional length components of each edge in the graph
    capturing the periodic connections between the core nodes, and saves
    this information.

    Args:
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    """
    # Generate filenames
    core_pb_conn_l_core_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_core_edges_x_cmpnt" + ".dat"
    )
    core_pb_conn_l_core_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_core_edges_y_cmpnt" + ".dat"
    )
    core_pb_conn_l_core_edges_z_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_core_edges_z_cmpnt" + ".dat"
    )
    core_pb_conn_l_pb_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_pb_edges_x_cmpnt" + ".dat"
    )
    core_pb_conn_l_pb_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_pb_edges_y_cmpnt" + ".dat"
    )
    core_pb_conn_l_pb_edges_z_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_pb_edges_z_cmpnt" + ".dat"
    )

    # Length components of the core edges and length components of the
    # periodic boundary edges in the core_pb_conn_graph
    (core_pb_conn_l_core_edges_x_cmpnt, core_pb_conn_l_core_edges_y_cmpnt,
     core_pb_conn_l_core_edges_z_cmpnt, core_pb_conn_l_pb_edges_x_cmpnt,
     core_pb_conn_l_pb_edges_y_cmpnt, core_pb_conn_l_pb_edges_z_cmpnt) = (
         core_pb_conn_graph_l_edges_dim_3_cmpnts_calculation(filename_prefix)
    )

    # Save the length components of the core edges and the length
    # components of the periodic boundary edges in the
    # core_pb_conn_graph
    np.savetxt(
        core_pb_conn_l_core_edges_x_cmpnt_filename,
        core_pb_conn_l_core_edges_x_cmpnt)
    np.savetxt(
        core_pb_conn_l_core_edges_y_cmpnt_filename,
        core_pb_conn_l_core_edges_y_cmpnt)
    np.savetxt(
        core_pb_conn_l_core_edges_z_cmpnt_filename,
        core_pb_conn_l_core_edges_z_cmpnt)
    np.savetxt(
        core_pb_conn_l_pb_edges_x_cmpnt_filename,
        core_pb_conn_l_pb_edges_x_cmpnt)
    np.savetxt(
        core_pb_conn_l_pb_edges_y_cmpnt_filename,
        core_pb_conn_l_pb_edges_y_cmpnt)
    np.savetxt(
        core_pb_conn_l_pb_edges_z_cmpnt_filename,
        core_pb_conn_l_pb_edges_z_cmpnt)

def swidt_network_graph_l_edges_cmpnts(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        config: int,
        pruning: int,
        graph: str) -> None:
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
        graph (str): Graph type in which the length components of each
        edge will be calculated; here, either "core_pb" (corresponding
        to the graph capturing the spatial topology of the core and
        periodic boundary nodes and edges) or "core_pb_conn"
        (corresponding to the graph capturing the periodic connections
        between the core nodes) are applicable.
    
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
    swidt_filename_prefix = (
        filename_str(network, date, batch, sample)
        + f"C{config:d}" + f"P{pruning:d}"
    )
    # Calculate and save the length components of each edge in the
    # spider web-inspired Delaunay-triangulated networks.
    if graph == "core_pb":
        # Graph capturing the spatial topology of the core and periodic
        # boundary nodes and edges
        if dim == 2:
            core_pb_graph_l_edges_dim_2_cmpnts(swidt_filename_prefix)
        elif dim == 3:
            core_pb_graph_l_edges_dim_3_cmpnts(swidt_filename_prefix)
    elif graph == "core_pb_conn":
        # Graph capturing the periodic connections between the core
        # cross-linker nodes.
        if dim == 2:
            core_pb_conn_graph_l_edges_dim_2_cmpnts(swidt_filename_prefix)
        elif dim == 3:
            core_pb_conn_graph_l_edges_dim_3_cmpnts(swidt_filename_prefix)

def core_pb_graph_l_nrmlzd_edges_dim_2_cmpnts(
        filename: str,
        filename_prefix: str) -> None:
    """Two-dimensional normalized length components of each edge
    calculated in the graph capturing the spatial topology of the core
    and periodic boundary nodes and edges.

    This function loads the simulation box size, calls upon a helper
    function to calculate the two-dimensional length components of each
    edge in the graph capturing the spatial topology of the core and
    periodic boundary nodes and edges, normalizes the edge length
    components by the simulation box size, and saves this information.

    Args:
        filename (str): Baseline filename for the simulation box size.
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    """
    # Generate filenames
    L_filename = filename + "-L" + ".dat"
    core_pb_l_nrmlzd_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_l_nrmlzd_edges_x_cmpnt" + ".dat"
    )
    core_pb_l_nrmlzd_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_l_nrmlzd_edges_y_cmpnt" + ".dat"
    )

    # Load L
    L = np.loadtxt(L_filename)

    # Length components of each distinct edge in the core_pb_graph
    core_pb_l_edges_x_cmpnt, core_pb_l_edges_y_cmpnt = (
        core_pb_graph_l_edges_dim_2_cmpnts_calculation(filename_prefix)
    )

    # Edge length component normalization by L*sqrt(2)
    core_pb_l_nrmlzd_edges_x_cmpnt = core_pb_l_edges_x_cmpnt / (L*np.sqrt(2))
    core_pb_l_nrmlzd_edges_y_cmpnt = core_pb_l_edges_y_cmpnt / (L*np.sqrt(2))

    # Save the normalized length components of distinct edges in the
    # core_pb_graph
    np.savetxt(
        core_pb_l_nrmlzd_edges_x_cmpnt_filename, core_pb_l_nrmlzd_edges_x_cmpnt)
    np.savetxt(
        core_pb_l_nrmlzd_edges_y_cmpnt_filename, core_pb_l_nrmlzd_edges_y_cmpnt)

def core_pb_graph_l_nrmlzd_edges_dim_3_cmpnts(
        filename: str,
        filename_prefix: str) -> None:
    """Three-dimensional normalized length components of each edge
    calculated in the graph capturing the spatial topology of the core
    and periodic boundary nodes and edges.

    This function loads the simulation box size, calls upon a helper
    function to calculate the three-dimensional length components of
    each edge in the graph capturing the spatial topology of the core
    and periodic boundary nodes and edges, normalizes the edge length
    components by the simulation box size, and saves this information.

    Args:
        filename (str): Baseline filename for the simulation box size.
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    """
    # Generate filenames
    L_filename = filename + "-L" + ".dat"
    core_pb_l_nrmlzd_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_l_nrmlzd_edges_x_cmpnt" + ".dat"
    )
    core_pb_l_nrmlzd_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_l_nrmlzd_edges_y_cmpnt" + ".dat"
    )
    core_pb_l_nrmlzd_edges_z_cmpnt_filename = (
        filename_prefix + "-core_pb_l_nrmlzd_edges_z_cmpnt" + ".dat"
    )

    # Load L
    L = np.loadtxt(L_filename)

    # Length components of each distinct edge in the core_pb_graph
    core_pb_l_edges_x_cmpnt, core_pb_l_edges_y_cmpnt, core_pb_l_edges_z_cmpnt = (
        core_pb_graph_l_edges_dim_3_cmpnts_calculation(filename_prefix)
    )

    # Edge length component normalization by L*sqrt(3)
    core_pb_l_nrmlzd_edges_x_cmpnt = core_pb_l_edges_x_cmpnt / (L*np.sqrt(3))
    core_pb_l_nrmlzd_edges_y_cmpnt = core_pb_l_edges_y_cmpnt / (L*np.sqrt(3))
    core_pb_l_nrmlzd_edges_z_cmpnt = core_pb_l_edges_z_cmpnt / (L*np.sqrt(3))

    # Save the normalized length components of distinct edges in the
    # core_pb_graph
    np.savetxt(
        core_pb_l_nrmlzd_edges_x_cmpnt_filename, core_pb_l_nrmlzd_edges_x_cmpnt)
    np.savetxt(
        core_pb_l_nrmlzd_edges_y_cmpnt_filename, core_pb_l_nrmlzd_edges_y_cmpnt)
    np.savetxt(
        core_pb_l_nrmlzd_edges_z_cmpnt_filename, core_pb_l_nrmlzd_edges_z_cmpnt)

def core_pb_conn_graph_l_nrmlzd_edges_dim_2_cmpnts(
        filename: str,
        filename_prefix: str) -> None:
    """Two-dimensional normalized length components of each edge
    calculated in the graph capturing the periodic connections between
    the core nodes.

    This function loads the simulation box size, calls upon a helper
    function to calculate the two-dimensional length components of each
    edge in the graph capturing the periodic connections between the
    core nodes, normalizes the edge length components by the simulation
    box size, and saves this information.

    Args:
        filename (str): Baseline filename for the simulation box size.
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    """
    # Generate filenames
    L_filename = filename + "-L" + ".dat"
    core_pb_conn_l_nrmlzd_core_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_core_edges_x_cmpnt" + ".dat"
    )
    core_pb_conn_l_nrmlzd_core_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_core_edges_y_cmpnt" + ".dat"
    )
    core_pb_conn_l_nrmlzd_pb_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_pb_edges_x_cmpnt" + ".dat"
    )
    core_pb_conn_l_nrmlzd_pb_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_pb_edges_y_cmpnt" + ".dat"
    )

    # Load L
    L = np.loadtxt(L_filename)

    # Length components of the core edges and length components of the
    # periodic boundary edges in the core_pb_conn_graph
    (core_pb_conn_l_core_edges_x_cmpnt, core_pb_conn_l_core_edges_y_cmpnt,
     core_pb_conn_l_pb_edges_x_cmpnt, core_pb_conn_l_pb_edges_y_cmpnt) = (
         core_pb_conn_graph_l_edges_dim_2_cmpnts_calculation(filename_prefix)
    )

    # Edge length component normalization by L*sqrt(2)
    core_pb_conn_l_nrmlzd_core_edges_x_cmpnt = (
        core_pb_conn_l_core_edges_x_cmpnt / (L*np.sqrt(2))
    )
    core_pb_conn_l_nrmlzd_core_edges_y_cmpnt = (
        core_pb_conn_l_core_edges_y_cmpnt / (L*np.sqrt(2))
    )
    core_pb_conn_l_nrmlzd_pb_edges_x_cmpnt = (
        core_pb_conn_l_pb_edges_x_cmpnt / (L*np.sqrt(2))
    )
    core_pb_conn_l_nrmlzd_pb_edges_y_cmpnt = (
        core_pb_conn_l_pb_edges_y_cmpnt / (L*np.sqrt(2))
    )

    # Save the normalized length components of the core edges and the
    # normalized length components of the periodic boundary edges in the
    # core_pb_conn_graph
    np.savetxt(
        core_pb_conn_l_nrmlzd_core_edges_x_cmpnt_filename,
        core_pb_conn_l_nrmlzd_core_edges_x_cmpnt)
    np.savetxt(
        core_pb_conn_l_nrmlzd_core_edges_y_cmpnt_filename,
        core_pb_conn_l_nrmlzd_core_edges_y_cmpnt)
    np.savetxt(
        core_pb_conn_l_nrmlzd_pb_edges_x_cmpnt_filename,
        core_pb_conn_l_nrmlzd_pb_edges_x_cmpnt)
    np.savetxt(
        core_pb_conn_l_nrmlzd_pb_edges_y_cmpnt_filename,
        core_pb_conn_l_nrmlzd_pb_edges_y_cmpnt)

def core_pb_conn_graph_l_nrmlzd_edges_dim_3_cmpnts(
        filename: str,
        filename_prefix: str) -> None:
    """Three-dimensional normalized length components of each edge
    calculated in the graph capturing the periodic connections between
    the core nodes.

    This function loads the simulation box size, calls upon a helper
    function to calculate the three-dimensional length components of
    each edge in the graph capturing the periodic connections between
    the core nodes, normalizes the edge length components by the
    simulation box size, and saves this information.

    Args:
        filename (str): Baseline filename for the simulation box size.
        filename_prefix (str): Filename prefix for the files associated
        with the fundamental graph constituents.
    
    """
    # Generate filenames
    L_filename = filename + "-L" + ".dat"
    core_pb_conn_l_nrmlzd_core_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_core_edges_x_cmpnt" + ".dat"
    )
    core_pb_conn_l_nrmlzd_core_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_core_edges_y_cmpnt" + ".dat"
    )
    core_pb_conn_l_nrmlzd_core_edges_z_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_core_edges_z_cmpnt" + ".dat"
    )
    core_pb_conn_l_nrmlzd_pb_edges_x_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_pb_edges_x_cmpnt" + ".dat"
    )
    core_pb_conn_l_nrmlzd_pb_edges_y_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_pb_edges_y_cmpnt" + ".dat"
    )
    core_pb_conn_l_nrmlzd_pb_edges_z_cmpnt_filename = (
        filename_prefix + "-core_pb_conn_l_nrmlzd_pb_edges_z_cmpnt" + ".dat"
    )

    # Load L
    L = np.loadtxt(L_filename)

    # Length components of the core edges and length components of the
    # periodic boundary edges in the core_pb_conn_graph
    (core_pb_conn_l_core_edges_x_cmpnt, core_pb_conn_l_core_edges_y_cmpnt,
     core_pb_conn_l_core_edges_z_cmpnt, core_pb_conn_l_pb_edges_x_cmpnt,
     core_pb_conn_l_pb_edges_y_cmpnt, core_pb_conn_l_pb_edges_z_cmpnt) = (
         core_pb_conn_graph_l_edges_dim_3_cmpnts_calculation(filename_prefix)
    )

    # Edge length component normalization by L*sqrt(3)
    core_pb_conn_l_nrmlzd_core_edges_x_cmpnt = (
        core_pb_conn_l_core_edges_x_cmpnt / (L*np.sqrt(3))
    )
    core_pb_conn_l_nrmlzd_core_edges_y_cmpnt = (
        core_pb_conn_l_core_edges_y_cmpnt / (L*np.sqrt(3))
    )
    core_pb_conn_l_nrmlzd_core_edges_z_cmpnt = (
        core_pb_conn_l_core_edges_z_cmpnt / (L*np.sqrt(3))
    )
    core_pb_conn_l_nrmlzd_pb_edges_x_cmpnt = (
        core_pb_conn_l_pb_edges_x_cmpnt / (L*np.sqrt(3))
    )
    core_pb_conn_l_nrmlzd_pb_edges_y_cmpnt = (
        core_pb_conn_l_pb_edges_y_cmpnt / (L*np.sqrt(3))
    )
    core_pb_conn_l_nrmlzd_pb_edges_z_cmpnt = (
        core_pb_conn_l_pb_edges_z_cmpnt / (L*np.sqrt(3))
    )

    # Save the normalized length components of the core edges and the
    # normalized length components of the periodic boundary edges in the
    # core_pb_conn_graph
    np.savetxt(
        core_pb_conn_l_nrmlzd_core_edges_x_cmpnt_filename,
        core_pb_conn_l_nrmlzd_core_edges_x_cmpnt)
    np.savetxt(
        core_pb_conn_l_nrmlzd_core_edges_y_cmpnt_filename,
        core_pb_conn_l_nrmlzd_core_edges_y_cmpnt)
    np.savetxt(
        core_pb_conn_l_nrmlzd_core_edges_z_cmpnt_filename,
        core_pb_conn_l_nrmlzd_core_edges_z_cmpnt)
    np.savetxt(
        core_pb_conn_l_nrmlzd_pb_edges_x_cmpnt_filename,
        core_pb_conn_l_nrmlzd_pb_edges_x_cmpnt)
    np.savetxt(
        core_pb_conn_l_nrmlzd_pb_edges_y_cmpnt_filename,
        core_pb_conn_l_nrmlzd_pb_edges_y_cmpnt)
    np.savetxt(
        core_pb_conn_l_nrmlzd_pb_edges_z_cmpnt_filename,
        core_pb_conn_l_nrmlzd_pb_edges_z_cmpnt)

def swidt_network_graph_l_nrmlzd_edges_cmpnts(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        config: int,
        pruning: int,
        graph: str) -> None:
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
        graph (str): Graph type in which the normalized length
        components of each edge will be calculated; here, either
        "core_pb" (corresponding to the graph capturing the spatial
        topology of the core and periodic boundary nodes and edges) or
        "core_pb_conn" (corresponding to the graph capturing the
        periodic connections between the core nodes) are applicable.
    
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
    filename = filename_str(network, date, batch, sample)
    swidt_filename_prefix = (
        filename_str(network, date, batch, sample)
        + f"C{config:d}" + f"P{pruning:d}"
    )
    # Calculate and save the normalized length components of each edge
    # in the spider web-inspired Delaunay-triangulated networks.
    if graph == "core_pb":
        # Graph capturing the spatial topology of the core and periodic
        # boundary nodes and edges
        if dim == 2:
            core_pb_graph_l_nrmlzd_edges_dim_2_cmpnts(
                filename, swidt_filename_prefix)
        elif dim == 3:
            core_pb_graph_l_nrmlzd_edges_dim_3_cmpnts(
                filename, swidt_filename_prefix)
    elif graph == "core_pb_conn":
        # Graph capturing the periodic connections between the core
        # cross-linker nodes.
        if dim == 2:
            core_pb_conn_graph_l_nrmlzd_edges_dim_2_cmpnts(
                filename, swidt_filename_prefix)
        elif dim == 3:
            core_pb_conn_graph_l_nrmlzd_edges_dim_3_cmpnts(
                filename, swidt_filename_prefix)