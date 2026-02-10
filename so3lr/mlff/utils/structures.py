from collections import namedtuple


System = namedtuple(
    "System",
    (
        "R",
        "Z",
        "cell",
        "total_charge",
        "num_unpaired_electrons"
    )
)

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
        "cell",
    )
)

Neighbors = namedtuple(
    "Neighbors",
    (
        "centers",
        "others",
        "overflow",
        "reference_positions"
    )
)

PrimitiveNeighbors = namedtuple(
    "PrimitiveNeighbors",
    (
        "idx_i",
        "idx_j",
        "shifts",
        "overflow"
    )
)
