from collections import namedtuple


Graph = namedtuple(
    "Graph",
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
        "residue_segments",
        "residue_charge",
    )
)
