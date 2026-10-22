"""Symmetry, permutation and batch independence of a shipped model.

The other rotation, translation and batching tests
(tests/mlff_tests/test_observable_fn.py, test_so3krates_layer_sparse.py) run on
a hand-built toy network. It has no charge or spin embedding, no
electrostatics, no dispersion, no NLH repulsion and no ratio heads, and those
are the parts where such a bug would hide: the long-range terms use their own
pair list, and the total charge and spin are per-graph values spread onto atoms.
Nothing else in the suite tests permutation invariance.

These tests run the shipped so3lr-tauri-s on a charged, open-shell molecule:

* rotation + translation: the energy and per-atom outputs stay the same, and the
  forces rotate with the molecule (fails, e.g., when a long-range term uses
  absolute rather than relative positions);
* permutation of the atoms: the energy stays the same, and the forces and
  per-atom outputs are permuted the same way (fails, e.g., when a pair term
  assumes an ordered neighbour list, or charge equilibration depends on atom
  order);
* a padded batch of the cation between two neutral molecules, run through the
  batching, model apply and unbatching of ``so3lr eval``: each structure gets
  exactly what it gets when evaluated alone, and the cation what
  So3lrCalculator gives it (fails when a charge or spin goes to the wrong
  graph, when padding atoms end up in a real graph's sum, or when pairs cross
  between graphs).

Each entry point (So3lrCalculator, and make_so3lr as used by ``so3lr eval``)
loads the model once for the whole module. Everything runs in float64, so the
tolerances are set by rounding error (see TOLERANCES below), not by model
accuracy.
"""
import json

import jax
import numpy as np
import numpy.testing as npt
import pytest
from ase.build import molecule
from ase.io import read, write

from so3lr.cli.so3lr_eval import process_predictions
from so3lr.mlff.data import AseDataLoaderSparse
from so3lr.mlff.utils import jraph_utils
from so3lr.model_registry import resolve_model

# The cheapest shipped v2 model. It has every part under test: charge and spin
# embedding, electrostatics, dispersion, NLH repulsion, a0/c6 ratio heads. A
# floating alias, because these are properties every release must have.
MODEL = 'so3lr-tauri-s'
# The So3lrCalculator and `so3lr eval` defaults. All pairs of these molecules lie
# within it, and the cation's longest pair falls in the electrostatics'
# smoothing window (0.45 * LR_CUTOFF to LR_CUTOFF).
LR_CUTOFF = 12.0
DISPERSION_DAMPING = 2.0

# The per-atom outputs compared below: the charge-equilibrated partial charges
# (the per-graph total charge reaches the atoms through them) and both ratio heads.
PER_ATOM_KEYS = ('partial_charges', 'a0_ratios', 'c6_ratios')

# TOLERANCES. In each test the model is compared with itself on an input related
# by an exact symmetry, or by an exact rearrangement into a batch, so in exact
# arithmetic the differences are zero. In float64 they come from rounding only.
# A rotated, translated, permuted or batched input changes the last bits of
# every distance, and the order of the floating-point sums (segment sums over
# edges, pairs and atoms, the dense layers, the neighbour-list order of
# So3lrCalculator against so3lr eval). Each of those operations is exact to the
# unit roundoff u = 1.1e-16, relative to the magnitudes involved. With O(1)
# features, a few dozen layers and sums of at most a few hundred terms, the
# expected discrepancy is ~1e-14 relative. That is ~1e-14 eV/A on these O(1-10)
# eV/A forces (the largest pair terms they cancel from are ~10 eV/A), ~1e-15 on
# the charges and ratios, and ~1e-14 |E| on the energy.
#
# The bounds below are 2 (energy) to 4-5 (forces, charges, dipoles) orders of
# magnitude above those estimates, which leaves room for GPU reductions in a
# nondeterministic order. They are also >= 5 orders below the smallest effect
# of the bugs the tests target (meV, meV/A, 1e-4 e and up). A run that silently
# fell back to float32 (~1e-7 relative) fails them too, which is intended: these
# tests assume float64.
ENERGY_RTOL = 1e-12   # times |E|
ENERGY_ATOL = 1e-10   # eV
FORCE_ATOL = 1e-9     # eV/A
PER_ATOM_ATOL = 1e-11  # e (partial charges); the ratio heads are dimensionless
DIPOLE_ATOL = 1e-10   # e*A


def _local_cutoff():
    """The model's short-range cutoff, read the way `so3lr eval` reads it."""
    with open(resolve_model(MODEL).dir / 'hyperparameters.json') as fp:
        return json.load(fp)['model']['cutoff']


def _nudged(name, seed):
    """An ase.build molecule moved off its symmetric geometry (0.05 A noise),
    so that no point-group symmetry can make a wrong rotation or permutation
    look right."""
    atoms = molecule(name)
    rng = np.random.default_rng(seed)
    atoms.positions = atoms.positions + rng.normal(0.0, 0.05, atoms.positions.shape)
    return atoms


def _cation():
    """The trans-butane radical cation C4H10(.+): charge +1 and 33 electrons,
    so a doublet with one unpaired electron. Both the charge and the spin
    embedding are active. Its terminal H...H pair (~5.6 A) lies beyond the
    model's 4.5 A local cutoff, so no edge joins those two atoms. They interact
    directly only through the long-range pair list."""
    atoms = _nudged('trans-butane', seed=1)
    atoms.info.update(charge=1, multiplicity=2)
    return atoms


def _random_rotation(rng):
    """A proper rotation about a random axis, by a random angle in
    [pi/4, 3pi/4], so that it is never close to the identity."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(np.pi / 4, 3 * np.pi / 4)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def _ase_evaluate(calc, atoms):
    """(energy, forces, per-atom outputs) of one fresh So3lrCalculator call."""
    atoms = atoms.copy()
    # ASE reuses cached results unless positions, numbers, cell or pbc change.
    # It never checks info['charge'], so clear the cache before every call.
    calc.reset()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    per_atom = {key: np.asarray(calc.results['aux'][key]) for key in PER_ATOM_KEYS}
    return energy, forces, per_atom


def _run_eval(so3lr_fn, items, budget):
    """The inner loop of `so3lr eval` (evaluate_so3lr_on) on `items`, which
    must fit in one padded batch. The items are batched, converted to model
    inputs, run through the model, and split back into structures as
    `--save-to` writes them. Returns the model inputs (for checking the
    batch itself) and one predicted graph per real structure."""
    batches = list(jraph_utils.dynamically_batch_with_lr(items, **budget))
    assert len(batches) == 1, f'expected one padded batch, got {len(batches)}'
    graph_batch = batches[0]
    inputs = jraph_utils.graph_to_batch_fn(*graph_batch)
    output = jax.device_get(so3lr_fn(inputs))
    inputs = jax.device_get(inputs)
    # The tolerances assume float64 all the way through.
    assert inputs['positions'].dtype == np.float64
    assert output['energy'].dtype == np.float64
    # process_predictions only checks save_to against None. It writes nothing.
    predicted = process_predictions('unused.extxyz', graph_batch, inputs, output)
    return inputs, predicted


@pytest.fixture(scope='module', autouse=True)
def x64():
    """float64 for the whole module. Without x64, So3lrCalculator's
    dtype=np.float64 silently runs in float32. The flag is global to the
    process, so restore it afterwards."""
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', old)


@pytest.fixture(scope='module')
def calc(x64):
    """So3lrCalculator, with the per-atom outputs returned as aux."""
    from so3lr import So3lrCalculator
    return So3lrCalculator(
        model=MODEL,
        lr_cutoff=LR_CUTOFF,
        dispersion_energy_cutoff_lr_damping=DISPERSION_DAMPING,
        dtype=np.float64,
        has_aux=True,
        observables=['partial_charges'],
    )


@pytest.fixture(scope='module')
def so3lr_eval_fn(x64):
    """The model apply of `so3lr eval`: make_so3lr, jitted as
    evaluate_so3lr_on jits it, asked for partial charges as `--targets
    partial_charges` asks for them."""
    from so3lr.base_calculator import make_so3lr
    return jax.jit(make_so3lr(
        model=MODEL,
        lr_cutoff=LR_CUTOFF,
        dispersion_energy_cutoff_lr_damping=DISPERSION_DAMPING,
        calculate_forces=True,
        output_intermediate_quantities=['partial_charges'],
    ))


@pytest.fixture(scope='module')
def cation_reference(calc):
    """The cation and its So3lrCalculator result, shared by the symmetry tests."""
    atoms = _cation()
    return atoms, _ase_evaluate(calc, atoms)


def test_rotation_and_translation(calc, cation_reference):
    """A rigid rotation followed by a translation leaves the energy and the
    per-atom outputs unchanged, and rotates the forces with the molecule."""
    atoms, (energy, forces, per_atom) = cation_reference

    # The setup is not trivial: some pairs are long-range only, and the forces
    # are large enough that "rotate with the molecule" means something.
    assert atoms.get_all_distances().max() > _local_cutoff() + 0.5
    assert np.abs(forces).max() > 1e-2

    rng = np.random.default_rng(2)
    rotation = _random_rotation(rng)
    translation = rng.uniform(-10.0, 10.0, 3)
    npt.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    npt.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-14)

    moved = atoms.copy()
    moved.positions = atoms.positions @ rotation.T + translation
    energy_moved, forces_moved, per_atom_moved = _ase_evaluate(calc, moved)

    npt.assert_allclose(energy_moved, energy, rtol=ENERGY_RTOL, atol=ENERGY_ATOL)
    npt.assert_allclose(forces_moved, forces @ rotation.T, rtol=0, atol=FORCE_ATOL)
    for key in PER_ATOM_KEYS:
        npt.assert_allclose(per_atom_moved[key], per_atom[key],
                            rtol=0, atol=PER_ATOM_ATOL, err_msg=key)

    # The rotation must actually move the forces, or the check above is vacuous.
    with npt.assert_raises(AssertionError):
        npt.assert_allclose(forces_moved, forces, rtol=0, atol=FORCE_ATOL)


def test_permutation(calc, cation_reference):
    """Reordering the atoms leaves the energy unchanged. The forces and
    per-atom outputs are reordered the same way."""
    atoms, (energy, forces, per_atom) = cation_reference

    rng = np.random.default_rng(3)
    permutation = rng.permutation(len(atoms))
    # Atoms of different elements change places, not just equivalent ones.
    assert np.any(atoms.numbers[permutation] != atoms.numbers)

    permuted = atoms[permutation]
    permuted.info.update(charge=atoms.info['charge'],
                         multiplicity=atoms.info['multiplicity'])
    energy_p, forces_p, per_atom_p = _ase_evaluate(calc, permuted)

    npt.assert_allclose(energy_p, energy, rtol=ENERGY_RTOL, atol=ENERGY_ATOL)
    npt.assert_allclose(forces_p, forces[permutation], rtol=0, atol=FORCE_ATOL)
    for key in PER_ATOM_KEYS:
        npt.assert_allclose(per_atom_p[key], per_atom[key][permutation],
                            rtol=0, atol=PER_ATOM_ATOL, err_msg=key)

    # Without the reordering the forces must not match, or the check is vacuous.
    with npt.assert_raises(AssertionError):
        npt.assert_allclose(forces_p, forces, rtol=0, atol=FORCE_ATOL)


def test_padded_batch_matches_each_structure_alone(calc, so3lr_eval_fn, tmp_path):
    """A charged, open-shell molecule between two neutral ones in one padded
    batch, through the `so3lr eval` path. Each structure gets the energy,
    forces, per-atom outputs and dipole it gets when evaluated alone.

    With the charged graph in the middle, its charge and spin are wrong if they
    go to every graph, or to the graph before or after it. ase.build centres
    all three molecules near the origin, so they overlap in space: a pair or
    edge that crossed from one graph to another would not go unnoticed.
    """
    frames = [_nudged('H2O', seed=4), _cation(), _nudged('CH3COCH3', seed=5)]
    expected_charge = [0, 1, 0]
    expected_unpaired = [0, 1, 0]

    # The data come in as `so3lr eval --datafile` reads them: from an extxyz,
    # with the charge and multiplicity in its info. Read the frames back so
    # that So3lrCalculator below sees the same rounded positions.
    datafile = tmp_path / 'mixed_charges.extxyz'
    write(datafile, frames, format='extxyz')
    frames = read(datafile, ':')
    data, stats = AseDataLoaderSparse(str(datafile)).load(
        cutoff=_local_cutoff(), cutoff_lr=LR_CUTOFF, calculate_neighbors_lr=True)
    npt.assert_array_equal([g.globals['total_charge'][0] for g, _ in data], expected_charge)
    npt.assert_array_equal([g.globals['num_unpaired_electrons'][0] for g, _ in data],
                           expected_unpaired)
    # The cation has pairs that only the long-range list contains.
    cation_graph, cation_lr = data[1]
    assert cation_lr['n_pairs'][0] > cation_graph.n_edge[0]

    # Budgets as evaluate_so3lr_on sets them for a batch of batch_size
    # structures. The single-structure runs below use the same budget. The
    # model then compiles once, but each run has a different amount of padding.
    batch_size = len(data)
    budget = dict(
        n_node=stats['max_num_of_nodes'] * batch_size + 1,
        n_edge=stats['max_num_of_edges'] * batch_size + 1,
        n_graph=batch_size + 1,
        n_pairs=stats['max_num_of_pairs'] * batch_size + 1,
    )

    inputs, together = _run_eval(so3lr_eval_fn, data, budget)

    # The batch really holds three graphs plus padding, and the model sees each
    # graph's own charge and spin. The padding graph is neutral.
    n_atoms = sum(len(frame) for frame in frames)
    assert int(inputs['num_of_non_padded_graphs']) == batch_size
    npt.assert_array_equal(inputs['graph_mask'], [True] * batch_size + [False])
    npt.assert_array_equal(inputs['total_charge'], expected_charge + [0])
    npt.assert_array_equal(inputs['num_unpaired_electrons'], expected_unpaired + [0])
    assert inputs['node_mask'].sum() == n_atoms < len(inputs['node_mask'])
    assert len(together) == batch_size

    for index, (item, frame) in enumerate(zip(data, frames)):
        alone_inputs, (alone,) = _run_eval(so3lr_eval_fn, [item], budget)
        # Same array shapes, hence the same compiled model, but n_node - len(frame)
        # padding atoms (plus empty padding graphs) instead of n_node - n_atoms. A
        # contribution from padding atoms would therefore not cancel.
        assert alone_inputs['positions'].shape == inputs['positions'].shape
        assert int(alone_inputs['num_of_non_padded_graphs']) == 1
        assert alone_inputs['node_mask'].sum() == len(frame)

        batched = together[index]
        # The unbatched prediction belongs to the right structure...
        npt.assert_array_equal(batched.nodes['positions'], frame.positions)
        npt.assert_array_equal(batched.nodes['atomic_numbers'], frame.numbers)
        # ...and the other graphs in the batch do not change it.
        npt.assert_allclose(batched.globals['energy_so3lr'], alone.globals['energy_so3lr'],
                            rtol=ENERGY_RTOL, atol=ENERGY_ATOL, err_msg=f'structure {index}')
        npt.assert_allclose(batched.nodes['forces_so3lr'], alone.nodes['forces_so3lr'],
                            rtol=0, atol=FORCE_ATOL, err_msg=f'structure {index}')
        for key in PER_ATOM_KEYS:
            npt.assert_allclose(batched.nodes[f'{key}_so3lr'], alone.nodes[f'{key}_so3lr'],
                                rtol=0, atol=PER_ATOM_ATOL, err_msg=f'{key}, structure {index}')
        npt.assert_allclose(batched.globals['dipole_vec_so3lr'], alone.globals['dipole_vec_so3lr'],
                            rtol=0, atol=DIPOLE_ATOL, err_msg=f'structure {index}')
        # Charge equilibration conserved each graph's own total charge.
        npt.assert_allclose(batched.nodes['partial_charges_so3lr'].sum(), expected_charge[index],
                            rtol=0, atol=PER_ATOM_ATOL, err_msg=f'structure {index}')

    # The cation from the batch is what So3lrCalculator gives with no padding at
    # all, so a bug that is the same in batched and single runs (e.g. the charge
    # dropped for every graph) is caught too.
    cation = together[1]
    energy, forces, per_atom = _ase_evaluate(calc, frames[1])
    npt.assert_allclose(cation.globals['energy_so3lr'][0], energy,
                        rtol=ENERGY_RTOL, atol=ENERGY_ATOL)
    npt.assert_allclose(cation.nodes['forces_so3lr'], forces, rtol=0, atol=FORCE_ATOL)
    for key in PER_ATOM_KEYS:
        npt.assert_allclose(cation.nodes[f'{key}_so3lr'], per_atom[key],
                            rtol=0, atol=PER_ATOM_ATOL, err_msg=key)

    # The checks above can fail. Evaluate the cation as a neutral singlet, which
    # is what it gets if its charge and spin go to another graph. Its energy
    # then moves by far more than the tolerance.
    neutral_graph = cation_graph._replace(globals={
        **cation_graph.globals,
        'total_charge': np.zeros_like(cation_graph.globals['total_charge']),
        'num_unpaired_electrons': np.zeros_like(cation_graph.globals['num_unpaired_electrons']),
    })
    _, (as_neutral,) = _run_eval(so3lr_eval_fn, [(neutral_graph, cation_lr)], budget)
    cation_energy = cation.globals['energy_so3lr'][0]
    shift = abs(as_neutral.globals['energy_so3lr'][0] - cation_energy)
    assert shift > 1e3 * (ENERGY_ATOL + ENERGY_RTOL * abs(cation_energy)), shift
