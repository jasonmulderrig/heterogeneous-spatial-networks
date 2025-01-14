import numpy as np
from helpers.network_utils import a_or_v_func

def A_or_V_arg_L_func(dim: int, L: float) -> float:
    """Simulation box area or volume.

    This function calculates the simulation box area or volume in two or
    three dimensions, respectively, given the simulation box size.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
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
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
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
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
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
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
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
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Particle diameter.
        n (float): Number of particles.
        eta (float): Particle packing density.

    Returns:
        float: Simulation box size.
    """
    if dim == 2: return np.sqrt(a_or_v_func(dim, b)*n/eta)
    elif dim == 3: return np.cbrt(a_or_v_func(dim, b)*n/eta)