import jax.numpy as jnp
import jraph
import numpy as np
import pytest
from ase import Atoms

from so3lr.mlff.data.dataloader_sparse_ase import ASE_to_jraph
from so3lr.mlff.data.dataloader_sparse_npz import entry_to_jraph
from so3lr.mlff.utils.jraph_utils import graph_to_batch_fn
from so3lr.pbc import is_fully_periodic


@pytest.mark.parametrize(
    ("pbc", "expected"),
    [
        ([False, False, False], False),
        ([True, True, True], True),
        (np.zeros((1, 3), dtype=bool), False),
        (np.ones((1, 3), dtype=bool), True),
    ],
)
def test_is_fully_periodic_accepts_supported_modes(pbc, expected):
    assert is_fully_periodic(pbc, context="test structure") is expected


@pytest.mark.parametrize("pbc", [[True, False, False], [True, True, False]])
def test_is_fully_periodic_rejects_mixed_axes(pbc):
    with pytest.raises(ValueError, match="mixed-axis"):
        is_fully_periodic(pbc, context="test structure")


def test_is_fully_periodic_rejects_malformed_flags():
    with pytest.raises(ValueError, match="three PBC flags"):
        is_fully_periodic([True, False], context="test structure")


def test_ase_loader_preserves_axis_specific_pbc():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]],
        cell=np.diag([6.0, 7.0, 8.0]),
        pbc=[True, False, True],
    )
    graph, _ = ASE_to_jraph(atoms, cutoff=2.0, calculate_neighbors_lr=False)

    np.testing.assert_array_equal(graph.globals["pbc"], [[True, False, True]])


def test_npz_loader_records_pbc_flags():
    entry = {
        "atomic_numbers": [1, 1],
        "positions": np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]]),
        "cell": np.diag([6.0, 7.0, 8.0]),
        "pbc": [True, False, True],
        "energy": np.array(0.0),
    }
    graph = entry_to_jraph(entry, cutoff=2.0)
    graph = graph[0] if isinstance(graph, tuple) else graph

    np.testing.assert_array_equal(graph.globals["pbc"], [[True, False, True]])


def test_batch_conversion_preserves_pbc_metadata():
    graph = jraph.GraphsTuple(
        nodes={
            "positions": jnp.zeros((1, 3)),
            "atomic_numbers": jnp.ones((1,), dtype=jnp.int32),
            "forces": jnp.zeros((1, 3)),
            "hirshfeld_ratios": jnp.ones((1,)),
            "c6_ratios": jnp.ones((1,)),
        },
        edges={"cell_offset": jnp.zeros((0, 3), dtype=jnp.int32)},
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        globals={
            "energy": jnp.zeros((1,)),
            "stress": jnp.zeros((1, 6)),
            "dipole_vec": jnp.zeros((1, 3)),
            "total_charge": jnp.zeros((1,)),
            "num_unpaired_electrons": jnp.zeros((1,)),
            "theory_level": jnp.zeros((1,), dtype=jnp.int32),
            "theory_mask": jnp.zeros((1, 16)),
            "cell": jnp.eye(3).reshape(1, 3, 3),
            "pbc": jnp.array([[True, True, True]]),
        },
        n_node=jnp.array([1]),
        n_edge=jnp.array([0]),
    )
    graph = jraph.pad_with_graphs(graph, n_node=2, n_edge=1, n_graph=2)
    lr_batch = {
        "idx_i_lr": jnp.zeros((0,), dtype=jnp.int32),
        "idx_j_lr": jnp.zeros((0,), dtype=jnp.int32),
        "cell_offsets_lr": jnp.zeros((0, 3), dtype=jnp.int32),
    }

    batch = graph_to_batch_fn(graph, lr_batch)

    np.testing.assert_array_equal(batch["pbc"][0], [True, True, True])
