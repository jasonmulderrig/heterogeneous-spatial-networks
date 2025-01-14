import numpy as np

def a_or_v_func(dim: int, b: float) -> float:
    """Area or volume of a chain segment and/or cross-linker.

    This function calculates the area or volume of a chain segment
    and/or cross-linker, given its diameter.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
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
        n_other (float): Number of the other type of particles (cross-linkers or chain segments).

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
        f (float): Particle (chain segment or cross-linker) number fraction.
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
        f (float): Particle (chain segment or cross-linker) number fraction.

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
        f_other (float): Other particle (cross-linker or chain segment, respectively) number fraction.

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
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Particle diameter.
        rho (float): Particle number density.

    Returns:
        float: Particle packing density.
    """
    return a_or_v_func(dim, b) * rho