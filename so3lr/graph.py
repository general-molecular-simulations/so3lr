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
        "k_grid_disp",
        "k_smearing_disp",
        "theory_mask",
        "residue_segments",
        "residue_charge",
    )
)

class Graph(_GraphBase):
    """Graph class with optional fields for theory_mask, residue_segments, and residue_charge.

    ``cell`` holds the lattice vectors as COLUMNS (glp's ``System.cell`` is
    ``atoms.cell.T``; jax-md maps fractional to Cartesian as ``box @ s``), or None
    in vacuum. ``PotentialSparse`` transposes it to rows for the model's k-space
    solvers, so a hand-built Graph must follow the column convention too.
    """

    def __new__(cls, positions, edges, nodes, centers, others, mask,
                total_charge, num_unpaired_electrons, edges_lr, idx_i_lr,
                idx_j_lr, cell, k_grid=None, k_smearing=None,
                k_grid_disp=None, k_smearing_disp=None,
                theory_mask=None, residue_segments=None,
                residue_charge=None):
        return super().__new__(cls, positions, edges, nodes, centers, others,
                             mask, total_charge, num_unpaired_electrons,
                             edges_lr, idx_i_lr, idx_j_lr, cell, k_grid, k_smearing,
                             k_grid_disp, k_smearing_disp,
                             theory_mask,
                             residue_segments, residue_charge)
