from voronoi_networks import (
    voronoi_L,
    voronoi_network_topology,
    voronoi_network_topological_descriptor
)
from delaunay_networks import (
    delaunay_L,
    delaunay_network_topology,
    delaunay_network_topological_descriptor
)
from swidt_networks import (
    swidt_L,
    swidt_network_topology,
    swidt_network_edge_pruning_procedure,
    swidt_network_topological_descriptor
)
from aelp_networks import (
    aelp_L,
    aelp_network_topology,
    aelp_network_topological_descriptor
)
from node_placement import initial_node_seeding

def run_voronoi_L(args):
    voronoi_L(*args)

def run_delaunay_L(args):
    delaunay_L(*args)

def run_swidt_L(args):
    swidt_L(*args)

def run_aelp_L(args):
    aelp_L(*args)

def run_initial_node_seeding(args):
    initial_node_seeding(*args)

def run_voronoi_network_topology(args):
    voronoi_network_topology(*args)

def run_delaunay_network_topology(args):
    delaunay_network_topology(*args)

def run_swidt_network_topology(args):
    swidt_network_topology(*args)

def run_aelp_network_topology(args):
    aelp_network_topology(*args)

def run_swidt_network_edge_pruning_procedure(args):
    swidt_network_edge_pruning_procedure(*args)

def run_voronoi_network_topological_descriptor(args):
    voronoi_network_topological_descriptor(*args)

def run_delaunay_network_topological_descriptor(args):
    delaunay_network_topological_descriptor(*args)

def run_swidt_network_topological_descriptor(args):
    swidt_network_topological_descriptor(*args)

def run_aelp_network_topological_descriptor(args):
    aelp_network_topological_descriptor(*args)