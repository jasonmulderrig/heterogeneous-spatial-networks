import numpy as np

def p_gaussian_cnfrmtn_func(
        b: float, nu: np.ndarray | float | int,
        r: np.ndarray | float | int) -> np.ndarray | float:
    """Gaussian end-to-end distance polymer chain conformation
    probability distribution.

    This function calculates the Gaussian end-to-end distance polymer
    chain conformation probability for a chain with a given number of
    segments and a given end-to-end distance.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        nu (np.ndarray | float | int): Number of segments in the chain.
        r (np.ndarray | float | int): End-to-end chain distance.
    
    Note: If nu is an np.ndarray, then r must be a float or int.
    Likewise, if r is an np.ndarray, then nu must be a float or int.

    Returns:
        np.ndarray | float: Gaussian end-to-end distance polymer chain
        conformation probability (distribution).
    """
    return (
        (np.sqrt(3/(2*np.pi*nu*b**2)))**3 * np.exp(-3*r**2/(2*nu*b**2))
        * 4 * np.pi * r**2
    )

def p_nu_flory_func(
        nu_mean: float,
        nu: np.ndarray | float | int) -> np.ndarray | float:
    """Chain segment number probability distribution representative of
    step-growth linear chain polymerization, as per the theory from
    Flory.

    This function calculates the probability of finding a chain with a
    given segment number in a polymer network formed via step-growth
    linear chain polymerization with a given mean segment number.

    Args:
        nu_mean (float): Average number of segments in the polymer network.
        nu (np.ndarray | float | int): Number of segments in the chain.

    Returns:
        np.ndarray | float: Chain segment number probability
        (distribution) representative of step-growth linear chain
        polymerization.
    """
    return (1./nu_mean)*(1.-(1./nu_mean))**(nu-1)

def p_net_gaussian_cnfrmtn_func(
        b: float,
        nu_mean: float,
        nu: np.ndarray | float | int,
        r: np.ndarray | float | int) -> np.ndarray | float:
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
        nu_mean (float): Average number of segments in the polymer network.
        nu (np.ndarray | float | int): Number of segments in the chain.
        r (np.ndarray | float | int): End-to-end chain distance.
    
    Note: If nu is an np.ndarray, then r must be a float or int.
    Likewise, if r is an np.ndarray, then nu must be a float or int.

    Returns:
        np.ndarray | float: Probability of finding a chain with a given
        segment number in a particular conformation, assuming
        step-growth linear chain polymerization and a Gaussian
        end-to-end distance chain conformation probability distribution.
    """
    return p_nu_flory_func(nu_mean, nu) * p_gaussian_cnfrmtn_func(b, nu, r)

def p_rel_net_gaussian_cnfrmtn_func(
        b: float,
        nu_mean: float,
        r: np.ndarray | float | int) -> np.ndarray | float:
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
        nu_mean (float): Average number of segments in the polymer network.
        r (np.ndarray | float | int): End-to-end chain distance.

    Returns:
        np.ndarray | float: Relative probability of finding a chain in a
        particular conformation, assuming step-growth linear chain
        polymerization and a Gaussian end-to-end distance chain
        conformation probability distribution.
    """
    # nu = 1 -> n = 100,000 is proscribed by Hanson
    nu_arr = np.arange(100000, dtype=int) + 1
    if isinstance(r, float | int):
        return np.sum(p_net_gaussian_cnfrmtn_func(b, nu_mean, nu_arr, r))
    elif isinstance(r, np.ndarray):
        r_num = np.shape(r)[0]
        p_arr = np.empty(r_num)
        # Calculate relative probability for each value of r
        for r_indx in range(r_num):
            p_arr[r_indx] = np.sum(
                p_net_gaussian_cnfrmtn_func(b, nu_mean, nu_arr, r[r_indx]))
        return p_arr
    else:
        error_str = (
            "The end-to-end distance r must be provided as either a "
            + "float, int, or an np.ndarray of np.floats or np.ints."
        )
        print(error_str)
        return None