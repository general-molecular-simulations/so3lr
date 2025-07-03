from collections import namedtuple

_GraphBase = namedtuple(
    "_GraphBase",
    (
        "positions",
        "edges",
        "nodes",
        "centers",
        "others",
        "mask",
        "total_charge",
        "num_unpaired_electrons",
        "edges_lr",
        "idx_i_lr",
        "idx_j_lr",
        "cell",
        "k_grid",
        "k_smearing",
        "theory_mask",
        "residue_segments",
        "residue_charge",
    )
)

class Graph(_GraphBase):
    """Graph class with optional fields for theory_mask, residue_segments, and residue_charge."""
    
    def __new__(cls, positions, edges, nodes, centers, others, mask, 
                total_charge, num_unpaired_electrons, edges_lr, idx_i_lr, 
                idx_j_lr, cell, k_grid=None, k_smearing=None, theory_mask=None, residue_segments=None, 
                residue_charge=None):
        return super().__new__(cls, positions, edges, nodes, centers, others, 
                             mask, total_charge, num_unpaired_electrons, 
                             edges_lr, idx_i_lr, idx_j_lr, cell, k_grid, k_smearing, theory_mask, 
                             residue_segments, residue_charge)