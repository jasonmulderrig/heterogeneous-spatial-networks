import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from network_topology_initialization_utils import (
    tessellation_protocol,
    tessellation,
    orb_neighborhood_id
)
from file_io import (
    L_filename_str,
    config_filename_str
)

def initial_random_node_placement(
        dim: int,
        L: float,
        n: int,
        config_filename_prefix: str) -> None:
    """Initial random node placement procedure.

    This function randomly places/seeds nodes within an empty simulation
    box.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size.
        n (int): Number of nodes.
        config_filename_prefix (str): Configuration filename prefix.
    
    """
    # Coordinates filename
    coords_filename = config_filename_prefix + ".coords"

    # Initialize random number generator
    rng = np.random.default_rng()

    # Save random node coordinates
    np.savetxt(coords_filename, L*rng.random((n, dim)))

def periodic_random_hard_disk_node_placement(
        dim: int,
        L: float,
        b: float,
        max_try: int,
        rng: np.random.Generator,
        tsslltn: np.ndarray,
        coords: np.ndarray,
        tsslltd_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Periodic random hard disk node placement procedure.

    This function randomly places/seeds nodes within a simulation box
    where each node is treated as the center of a hard disk, and the
    simulation box is periodic.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size.
        b (float): Hard disk diameter.
        max_try (int): Maximum number of node placement attempts.
        rng (np.random.Generator): np.random.Generator object.
        tsslltn (np.ndarray): Tessellation protocol.
        coords (np.ndarray): Node coordinates.
        tsslltd_coords (np.ndarray): Tessellated node coordinates.
    
    Return:
        tuple[np.ndarray, np.ndarray]: Node coordinates and tessellated
        node coordinates.
    
    """
    # Begin periodic random hard disk node placement procedure
    num_try = 0
    
    while num_try < max_try:
        # Generate randomly placed node candidate
        seed_cnddt = L * rng.random((dim,))

        # Downselect the previously-accepted tessellated nodes to those
        # that reside in a local orb neighborhood with radius b about
        # the node candidate
        _, orb_nghbr_num = orb_neighborhood_id(
            dim, tsslltd_coords, seed_cnddt, b, inclusive=False, indices=True)
        
        # Try again if the local orb neighborhood has at least one
        # neighbor in it
        if orb_nghbr_num > 0:
            num_try += 1
            continue
        
        # Accept and tessellate the node candidate if no local orb
        # neighborhood of tessellated nodes exists about the node
        # candidate
        coords = np.vstack((coords, seed_cnddt))
        tsslltd_coords = np.vstack(
            (tsslltd_coords, tessellation(seed_cnddt, tsslltn, L)))
        break

    return coords, tsslltd_coords

def initial_periodic_random_hard_disk_node_placement(
        dim: int,
        L: float,
        b: float,
        n: int,
        max_try: int,
        config_filename_prefix: str) -> None:
    """Initial periodic random hard disk node placement procedure.

    This function randomly places/seeds nodes within an empty simulation
    box where each node is treated as the center of a hard disk, and the
    simulation box is periodic.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size.
        b (float): Hard disk diameter.
        n (int): Intended number of nodes.
        max_try (int): Maximum number of node placement attempts.
        config_filename_prefix (str): Configuration filename prefix.
    
    """
    # Coordinates filename
    coords_filename = config_filename_prefix + ".coords"

    # Initialize random number generator
    rng = np.random.default_rng()

    # Tessellation protocol
    tsslltn, _ = tessellation_protocol(dim)

    # Periodic random hard disk node placement procedure
    for seed_attmpt in range(n):
        # Accept and tessellate the first node
        if seed_attmpt == 0:
            # Accept the first node
            seed = L * rng.random((dim,))
            coords = seed.copy()

            # Tessellate the first node
            seed_tsslltn = tessellation(seed, tsslltn, L)
            tsslltd_coords = seed_tsslltn.copy()
        else:
            # Begin periodic random hard disk node placement procedure
            coords, tsslltd_coords = periodic_random_hard_disk_node_placement(
                dim, L, b, max_try, rng, tsslltn, coords, tsslltd_coords)

    # Save coordinates
    np.savetxt(coords_filename, coords)

def initial_periodic_disordered_hyperuniform_node_placement(
        dim: int,
        L: float,
        n: int,
        config_filename_prefix: str) -> None:
    """Initial periodic disordered hyperuniform node placement
    procedure.

    This function places/seeds nodes within an empty simulation box in a
    periodic disordered hyperuniform fashion.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size.
        n (int): Intended number of nodes.
        config_filename_prefix (str): Configuration filename prefix.
    
    """
    error_str = (
        "Periodic disordered hyperuniform node placement procedure has "
        + "not been defined yet!"
    )
    print(error_str)
    return None

def initial_lammps_input_file_generator(
        dim: int,
        L: float,
        b: float,
        n: int,
        config_filename_prefix: str) -> None:
    """LAMMPS input file generator for soft pushoff and FIRE energy
    minimization of randomly placed nodes.

    This function generates a LAMMPS input file that randomly places
    nodes within an empty simulation box, followed by a soft pushoff
    procedure, and finally followed by a FIRE energy minimization
    procedure.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (float): Simulation box size.
        b (float): Node diameter.
        n (int): Number of core nodes.
        config_filename_prefix (str): Configuration filename prefix.
    
    """
    # Generate filenames
    lammps_input_filename = config_filename_prefix + ".in"
    coords_filename = config_filename_prefix + ".coords"
    
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

    # Write coordinates to coordinates file
    lammps_input_file.write(f"write_data {coords_filename} nocoeff")

    lammps_input_file.close()

def initial_node_seeding(
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
    """Initial node placement procedure for heterogeneous spatial
    networks.

    This function loads the simulation box size for the heterogeneous
    spatial networks. Then, depending on the particular core node
    placement scheme, this function calls upon a corresponding helper
    function to calculate the initial core node positions.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "apelp", "swidt", "delaunay", or "voronoi" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial polydisperse end-linked polymer networks ("apelp"), spider web-inspired Delaunay-triangulated networks ("swidt"), Delaunay-triangulated networks ("delaunay"), or Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Number of core nodes.
        config (int): Configuration number.
        max_try (int): Maximum number of node placement attempts for the periodic random hard disk node placement procedure ("prhd").
    
    """
    # Load L
    L = np.loadtxt(L_filename_str(network, date, batch, sample))

    # Configuration filename prefix
    config_filename_prefix = config_filename_str(
        network, date, batch, sample, config)

    # Call appropriate initial node placement helper function
    if scheme == "random":
        initial_random_node_placement(dim, L, n, config_filename_prefix)
    elif scheme == "prhd":
        initial_periodic_random_hard_disk_node_placement(
            dim, L, b, n, max_try, config_filename_prefix)
    elif scheme == "pdhu":
        initial_periodic_disordered_hyperuniform_node_placement(
            dim, L, n, config_filename_prefix)
    elif scheme == "lammps":
        initial_lammps_input_file_generator(dim, L, b, n, config_filename_prefix)

def additional_random_node_placement(
        L_filename: str,
        coords_filename: str,
        dim: int,
        n: int) -> None:
    """Additional random node placement procedure.

    This function randomly places/seeds nodes within a simulation box
    that already contains nodes.

    Args:
        L_filename (str): Filename for simulation box size.
        coords_filename (str): Filename for the node coordinates.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Intended number of nodes.
    
    """
    # Load L and node coordinates
    L = np.loadtxt(L_filename)
    coords = np.loadtxt(coords_filename)

    # Calculate the number of nodes presently in the simulation box
    n_coords = np.shape(coords)[0]

    # Pass if the number of nodes presently in the simulation box is
    # greater than or equal to the intended number of nodes
    if n_coords > n:
        print_str = (
            "The number of nodes presently in the simulation box is "
            + "greater than the intended number of nodes. You may want "
            + "to modify the intended number of nodes accordingly."
        )
        print(print_str)
    elif n_coords == n: pass
    else:
        # Calculate number of nodes to add to the simulation box
        n_addtnl = n - n_coords
        
        # Initialize random number generator
        rng = np.random.default_rng()

        # Save additional random node coordinates
        np.savetxt(
            coords_filename, np.vstack((coords, L*rng.random((n_addtnl, dim)))))

def additional_periodic_random_hard_disk_node_placement(
        L_filename: str,
        coords_filename: str,
        dim: int,
        b: float,
        n: int,
        max_try: int) -> None:
    """Additional periodic random hard disk node placement procedure.

    This function randomly places/seeds nodes within a simulation box
    that already contains nodes. Here, each node is treated as the
    center of a hard disk, and the simulation box is periodic.

    Args:
        L_filename (str): Filename for simulation box size.
        coords_filename (str): Filename for the node coordinates.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Hard disk diameter.
        n (int): Intended number of nodes.
        max_try (int): Maximum number of node placement attempts.
    
    """
    # Load L and node coordinates
    L = np.loadtxt(L_filename)
    coords = np.loadtxt(coords_filename)

    # Calculate the number of nodes presently in the simulation box
    n_coords = np.shape(coords)[0]

    # Pass if the number of nodes presently in the simulation box is
    # greater than or equal to the intended number of nodes
    if n_coords > n:
        print_str = (
            "The number of nodes presently in the simulation box is "
            + "greater than the intended number of nodes. You may want "
            + "to modify the intended number of nodes accordingly."
        )
        print(print_str)
    elif n_coords == n: pass
    else:
        # Calculate number of nodes to add to the simulation box
        n_addtnl = n - n_coords
        
        # Initialize random number generator
        rng = np.random.default_rng()

        # Tessellation protocol
        tsslltn, _ = tessellation_protocol(dim)

        # Tessellate nodes presently in the simulation box (as an
        # initialization step)
        for node in range(n_coords):
            node_coords = coords[node]
            if node == 0:
                node_tsslltn = tessellation(node_coords, tsslltn, L)
                tsslltd_coords = node_tsslltn.copy()
            else:
                tsslltd_coords = np.vstack(
                    (tsslltd_coords, tessellation(node_coords, tsslltn, L)))

        # Periodic random hard disk additional node placement procedure
        for _ in range(n_addtnl):
            # Begin periodic random hard disk node placement procedure
            coords, tsslltd_coords = periodic_random_hard_disk_node_placement(
                dim, L, b, max_try, rng, tsslltn, coords, tsslltd_coords)

        # Save coordinates
        np.savetxt(coords_filename, coords)

def additional_periodic_disordered_hyperuniform_node_placement(
        L_filename: str,
        coords_filename: str,
        dim: int,
        n: int) -> None:
    """Additional periodic disordered hyperuniform node placement
    procedure.

    This function places/seeds nodes within a simulation box that
    already contains nodes in a periodic disordered hyperuniform
    fashion.

    Args:
        L_filename (str): Filename for simulation box size.
        coords_filename (str): Filename for the node coordinates.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Intended number of nodes.
    
    """
    error_str = (
        "Periodic disordered hyperuniform node placement procedure has "
        + "not been defined yet!"
    )
    print(error_str)
    return None

def additional_lammps_input_file_generator(
        L_filename: str,
        coords_filename: str,
        dim: int,
        b: float,
        n: int) -> None:
    """LAMMPS input file generator for soft pushoff and FIRE energy
    minimization of randomly placed nodes.

    This function generates a LAMMPS input file that randomly places
    nodes within a simulation box that already contains nodes, followed
    by a soft pushoff procedure, and finally followed by a FIRE energy
    minimization procedure.

    Args:
        L_filename (str): Filename for simulation box size.
        coords_filename (str): Filename for the node coordinates.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Number of core nodes.
    
    """
    error_str = (
        "Adding additional nodes to a LAMMPS input file capturing a "
        + "simulation box that already contains nodes has not been "
        + "defined yet!"
    )
    print(error_str)
    return None

def additional_node_seeding(
        L_filename: str,
        coords_filename: str,
        scheme: str,
        dim: int,
        b: float,
        n: int,
        max_try: int) -> None:
    """Additional node placement procedure for heterogeneous spatial
    networks.

    Depending on the particular node placement scheme, this function
    calls upon a corresponding helper function to calculate and place
    additional nodes in the simulation box.

    Args:
        L_filename (str): Filename for simulation box size.
        coords_filename (str): Filename for the node coordinates.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended total number of nodes.
        max_try (int): Maximum number of node placement attempts for the periodic random hard disk node placement procedure ("prhd").
    
    """
    # Call appropriate additional node placement helper function
    if scheme == "random":
        additional_random_node_placement(
            L_filename, coords_filename, dim, n)
    elif scheme == "prhd":
        additional_periodic_random_hard_disk_node_placement(
            L_filename, coords_filename, dim, b, n, max_try)
    elif scheme == "pdhu":
        additional_periodic_disordered_hyperuniform_node_placement(
            L_filename, coords_filename, dim, n)
    elif scheme == "lammps":
        additional_lammps_input_file_generator(
            L_filename, coords_filename, dim, b, n)

def hilbert_node_label_assignment(
        coords: np.ndarray,
        L: float,
        dim: int,
        hilbert_order: int=6):
    """Hilbert space-filling curve-based node label assignment.

    This function labels nodes in a simulation box based upon where the
    coordinates of the nodes fall on the Hilbert space-filling curve.

    Args:
        coords (np.ndarray): Node coordinates.
        L (float): Simulation box size.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        hilbert_order (int): Order of the Hilbert space-filling curve.
        Default (and ideal) value is 6.
    
    Returns:
        np.ndarray: Indices of the nodes that sort the node coordinates
        along the Hilbert space-filling curve.

    """
    # Create HilbertCurve object
    hilbert_curve = HilbertCurve(p=hilbert_order, n=dim)

    # Normalize coordinates by the simulation box size
    nrmlzd_coords = coords / L
    
    # Map normalized coordinates to integer grid
    scld_nrmlzd_coords = (nrmlzd_coords*((2**hilbert_order)-1)).astype(int)
    
    # Calculate Hilbert curve distances of the normalized coordinates
    # mapped to the integer grid
    hilbert_indcs = hilbert_curve.distances_from_points(scld_nrmlzd_coords)

    # Return the indices that sort the Hilbert curve distances
    return np.argsort(hilbert_indcs)