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
        filename_prefix: str) -> None:
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
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Generate data filename
    config_filename = filename_prefix + ".config"

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
                    # neighbors. Retain unique indices corresponding to
                    # each possible cross-linker neighbor, and the
                    # number of times each such index value appears
                    psbl_nghbr_indcs, psbl_nghbr_indcs_counts = (
                        np.unique(np.concatenate((psbl_nghbr_x_indcs, psbl_nghbr_y_indcs), dtype=int), return_counts=True)
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
                        nghbr_indcs = psbl_nghbr_indcs[nghbr_indcs_vals_indcs]
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
                    # neighbors. Retain unique indices corresponding to
                    # each possible cross-linker neighbor, and the
                    # number of times each such index value appears
                    psbl_nghbr_indcs, psbl_nghbr_indcs_counts = (
                        np.unique(np.concatenate((psbl_nghbr_x_indcs, psbl_nghbr_y_indcs, psbl_nghbr_z_indcs), dtype=int), return_counts=True)
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
                        nghbr_indcs = psbl_nghbr_indcs[nghbr_indcs_vals_indcs]
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
        filename_prefix: str) -> None:
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
    # Generate filename prefix
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    
    # Load L
    L = np.loadtxt(L_filename)

    # Append configuration number to filename prefix
    filename_prefix = filename_prefix + f"C{config:d}"
    
    # Call appropriate helper function to calculate the core
    # cross-linker coordinates
    if scheme == "rccs": # random core cross-linker coordinates
        random_core_coords_crosslinker_seeding(
            dim, L, b, n, max_try, filename_prefix)
    elif scheme == "mccs": # minimized core cross-linker coordinates
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

def swidt_dim_2_network_topology_initialization(
        L: float,
        core_x: np.ndarray,
        core_y: np.ndarray,
        n: int,
        filename_prefix: str) -> None:
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
        filename_prefix (str): Baseline filename prefix for data files.
    
    """
    # Import the scipy.spatial.Delaunay() function
    from scipy.spatial import Delaunay
    
    # Generate filenames
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"

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

    # List for edges of the core and periodic boundary cross-linkers
    tsslltd_core_pb_edges = []

    for simplex in simplices:
        # In two-dimensions, each simplex is a triangle
        node_0 = int(simplex[0])
        node_1 = int(simplex[1])
        node_2 = int(simplex[2])

        # If any of the nodes involved in any simplex edge correspond to
        # the original core cross-linkers, then add that edge to the
        # edge list. Duplicate entries will arise.
        if (node_0 < n) or (node_1 < n):
            tsslltd_core_pb_edges.append((node_0, node_1))
        if (node_1 < n) or (node_2 < n):
            tsslltd_core_pb_edges.append((node_1, node_2))
        if (node_2 < n) or (node_0 < n):
            tsslltd_core_pb_edges.append((node_2, node_0))
        else: pass
    
    del simplex, simplices, tsslltd_core_deltri

    # Convert edge list to np.ndarray, and retain the unique edges from
    # the core and periodic boundary cross-linkers
    tsslltd_core_pb_edges = unique_sorted_edges(tsslltd_core_pb_edges)

    # Lists for the edges of the graph capturing the periodic
    # connections between the core cross-linkers
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

    # Save the core cross-linker x- and y-coordinates
    np.savetxt(core_x_filename, core_x)
    np.savetxt(core_y_filename, core_y)

def swidt_dim_3_network_topology_initialization(
        L: float,
        core_x: np.ndarray,
        core_y: np.ndarray,
        core_z: np.ndarray,
        n: int,
        filename_prefix: str) -> None:
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
        # Skip the (hold, hold) tessellation call because the core
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

    # List for edges of the core and periodic boundary cross-linkers
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
    
    del simplex, simplices, tsslltd_core_deltri

    # Convert edge list to np.ndarray, and retain the unique edges from
    # the core and periodic boundary cross-linkers
    tsslltd_core_pb_edges = unique_sorted_edges(tsslltd_core_pb_edges)

    # Lists for the edges of the graph capturing the periodic
    # connections between the core cross-linkers
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

    # Save the core cross-linker x-, y-, and z-coordinates
    np.savetxt(core_x_filename, core_x)
    np.savetxt(core_y_filename, core_y)
    np.savetxt(core_z_filename, core_z)

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
                L, rccs_x, rccs_y, rccs_n, filename_prefix)
        elif dim == 3:
            # Separate z-coordinates of core cross-linkers
            rccs_z = rccs[:, 2].copy()
            del rccs
            swidt_dim_3_network_topology_initialization(
                L, rccs_x, rccs_y, rccs_z, rccs_n, filename_prefix)
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
                L, mccs_x, mccs_y, n, filename_prefix)
        elif dim == 3:
            # Separate z-coordinates of core cross-linkers
            mccs_z = mccs[:, 4].copy()
            del mccs
            swidt_dim_3_network_topology_initialization(
                L, mccs_x, mccs_y, mccs_z, n, filename_prefix)

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
    cross-linker coordinates, performs a random edge pruning procedure
    such that each cross-linker in the network is connected to, at most,
    k edges, and isolates the maximum connected component from the
    resulting network.

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

    # Load core cross-linker x- and y-coordinates
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    if dim == 3:
        # Load core cross-linker z-coordinates
        core_z = np.loadtxt(core_z_filename)
    
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

    # Degree of cross-linker nodes in the graph
    conn_graph_k = np.asarray(list(conn_graph.degree()), dtype=int)[:, 1]

    if np.any(conn_graph_k > k):
        # Explicit edge pruning procedure
        while np.any(conn_graph_k > k):
            # Identify the cross-linker nodes connected to more than k
            # edges in the graph, i.e., hyperconnected cross-linker
            # nodes
            conn_graph_hyprconn_nodes = np.where(conn_graph_k > k)[0]
            # Identify the edges connected to the hyperconnected
            # cross-linker nodes
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

            # Update degree of cross-linker nodes in the graph
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

        # Isolate the cross-linker coordinates for the largest/maximum
        # connected component
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
        
        # Save the core cross-linker x- and y-coordinates
        np.savetxt(mx_cmp_pruned_core_x_filename, mx_cmp_pruned_core_x)
        np.savetxt(mx_cmp_pruned_core_y_filename, mx_cmp_pruned_core_y)
        if dim == 3:
            # Save the core cross-linker z-coordinates
            np.savetxt(mx_cmp_pruned_core_z_filename, mx_cmp_pruned_core_z)
    else:
        # Save fundamental graph constituents from this topology
        np.savetxt(mx_cmp_pruned_conn_n_filename, [n], fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_core_edges_filename, conn_core_edges, fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_pb_edges_filename, conn_pb_edges, fmt="%d")
        
        # Save the core cross-linker x- and y-coordinates
        np.savetxt(mx_cmp_pruned_core_x_filename, core_x)
        np.savetxt(mx_cmp_pruned_core_y_filename, core_y)
        if dim == 3:
            # Save the core cross-linker z-coordinates
            np.savetxt(mx_cmp_pruned_core_z_filename, core_z)

# To realize the elastically-effective network from the original
# network, I need to remove self-loops, and dangling chains repeatedly
# (e.g., to remove pending loops). However, one must deal with the case
# where a second-order loop, i.e., multiedge, is connected to a free
# node. This is a case of a dangling multiedge. Such an edge needs to be
# removed. Check this case by viewing the neighbor list for the both
# nodes holding the multiedge. If one of them only has one neighbor
# (being the other node), then remove the multiedge entirely. This needs
# to be performed in the same protocol as the dangling edge removal
# (where this is repeatedly checked until no dangling edges exist).

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
        elastically_effective (bool): Marker used to indicate if the
        elastically-effective network should be analyzed.
    
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
    conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    k_counts_filename = filename_prefix + "-k_counts" + ".dat"

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
    the nodal coordinates of a particular periodic boundary edge in a
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
    the nodal coordinates of a particular periodic boundary edge in a
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
        the core cross-linkers.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core cross-linkers.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core cross-linkers.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
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
        the core cross-linkers.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core cross-linkers.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core cross-linkers.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
        core_z (np.ndarray): z-coordinates of the core cross-linkers.
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
        the core cross-linkers.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core cross-linkers.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core cross-linkers.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
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
        the core cross-linkers.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core cross-linkers.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core cross-linkers.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
        core_z (np.ndarray): z-coordinates of the core cross-linkers.
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
        the core cross-linkers.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core cross-linkers.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core cross-linkers.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
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
        the core cross-linkers.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core cross-linkers.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core cross-linkers.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
        core_z (np.ndarray): z-coordinates of the core cross-linkers.
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
        the core cross-linkers.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core cross-linkers.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core cross-linkers.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
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
        the core cross-linkers.
        conn_pb_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph represents the periodic
        boundary edges from the graph capturing the periodic connections
        between the core cross-linkers.
        conn_graph: (Undirected) NetworkX graph that can be of type
        nx.Graph or nx.MultiGraph. This graph captures the periodic
        connections between the core cross-linkers.
        core_x (np.ndarray): x-coordinates of the core cross-linkers.
        core_y (np.ndarray): y-coordinates of the core cross-linkers.
        core_z (np.ndarray): z-coordinates of the core cross-linkers.
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