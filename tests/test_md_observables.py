"""The `--observable` surface must agree with itself end to end.

Three hand-maintained lists had drifted: the CLI's click.Choice, the gate that
decides whether the model is even asked for intermediate quantities, and the
HDF5 dataset whitelist. The result was that 8 of the 9 documented choices
crashed -- either with KeyError after the first MD cycle, or at write time with
"object '<name>' doesn't exist".
"""
import numpy as np
import pytest

from so3lr.md_observables import (
    MD_OBSERVABLES,
    PER_ATOM_OBSERVABLES,
    is_per_atom_observable,
)


def _observable_option():
    from so3lr.cli import so3lr_cli
    nvt = so3lr_cli.cli.commands['nvt']
    return next(p for p in nvt.params if p.name == 'observable')


def test_cli_choices_match_the_observable_table():
    assert tuple(_observable_option().type.choices) == MD_OBSERVABLES


def test_every_per_atom_observable_gets_an_hdf5_dataset(tmp_path):
    """init_hdf5_store must create a dataset for each per-atom observable.

    It used to whitelist a different eight names, omitting nn_energy,
    nlh_repulsion and both *_undamped variants -- all selectable from the CLI.
    """
    import h5py
    from so3lr.cli.so3lr_md import init_hdf5_store

    out = tmp_path / 'traj.hdf5'
    init_hdf5_store(
        save_to=out, batch_size=2, num_atoms=5, num_box_entries=3,
        exist_ok=True, observables=list(PER_ATOM_OBSERVABLES),
    )
    with h5py.File(out, 'r') as f:
        missing = [o for o in PER_ATOM_OBSERVABLES if o not in f]
        assert missing == [], f'no HDF5 dataset created for {missing}'
        for o in PER_ATOM_OBSERVABLES:
            assert f[o].shape[1:] == (5,), f'{o} has shape {f[o].shape}'


def test_embeddings_get_a_dataset_when_the_feature_width_is_known(tmp_path):
    """'x' is (num_atoms, num_features), not one scalar per atom, so its dataset
    needs the model's feature width. Without it the store silently had no 'x'
    dataset and the run died at write time with "object 'x' doesn't exist"."""
    import h5py
    from so3lr.cli.so3lr_md import init_hdf5_store

    assert not is_per_atom_observable('x')
    out = tmp_path / 'traj.hdf5'
    init_hdf5_store(
        save_to=out, batch_size=2, num_atoms=5, num_box_entries=3,
        exist_ok=True, observables=['x'], num_features=128,
    )
    with h5py.File(out, 'r') as f:
        assert f['x'].shape[1:] == (5, 128)


def test_embeddings_are_reported_when_the_width_is_unknown(tmp_path):
    """No width -> no dataset, but say so rather than failing later."""
    import h5py
    from so3lr.cli.so3lr_md import init_hdf5_store

    out = tmp_path / 'traj.hdf5'
    init_hdf5_store(
        save_to=out, batch_size=2, num_atoms=5, num_box_entries=3,
        exist_ok=True, observables=['x'], num_features=None,
    )
    with h5py.File(out, 'r') as f:
        assert 'x' not in f


def test_model_feature_width_is_readable_for_bundled_models():
    from so3lr.cli.so3lr_md import _model_num_features

    assert _model_num_features('so3lr-m-2606') == 128
    assert _model_num_features(None) is not None  # default model


@pytest.mark.parametrize('observables', [
    (),
    ('nn_energy',),
    ('partial_charges',),
    ('dispersion_energy', 'nlh_repulsion'),
    MD_OBSERVABLES,
])
def test_intermediate_quantities_requested_for_any_observable(observables):
    """The model must be asked to emit intermediate quantities for ANY
    observable. This used to be gated on 'partial_charges' being present, so
    every other choice raised KeyError after the first MD cycle."""
    from so3lr.cli.so3lr_md import intermediate_quantity_kwargs

    kwargs = intermediate_quantity_kwargs(observables)
    if observables:
        assert kwargs == {'output_intermediate_quantities': observables}
    else:
        assert kwargs == {}


@pytest.mark.parametrize('cell,should_raise', [
    (np.zeros((3, 3)), True),                                        # no cell
    (None, True),                                                    # no cell at all
    (np.array([[10., 1., 2.], [1., 11., 3.], [2., 3., 12.]]), False),  # triclinic
    (np.diag([10., 11., 12.]), False),                                # orthorhombic
])
def test_tune_ewald_cell_guard(cell, should_raise):
    """`not np.all(cell != 0)` was inverted: it accepted an all-zero cell and
    rejected a fully dense triclinic one. Only orthorhombic cells, whose
    off-diagonal zeros make the test False, behaved correctly. This calls the
    real predicate `tune_ewald.cell_is_present`, not a re-typed expression."""
    from so3lr.cli.tune_ewald import cell_is_present

    assert cell_is_present(cell) == (not should_raise)


def test_tune_ewald_settings_keys_are_known_to_the_validator():
    """AGENTS.md: new settings keys must be registered so the did-you-mean
    validator does not flag them as typos."""
    from so3lr.cli.so3lr_cli import known_yaml_keys

    known = known_yaml_keys()
    tune_keys = [
        'accuracy', 'cutoff_lo', 'cutoff_it', 'cutoff_hi',
        'smear_lo', 'smear_hi', 'smear_it', 'ns_lo', 'ns_hi',
        'nodes_lo', 'nodes_hi', 'mesh_lo', 'mesh_hi',
        'dispersion_damping_lo', 'dispersion_damping_hi', 'dispersion_damping_it',
    ]
    assert [k for k in tune_keys if k not in known] == []


def test_tune_ewald_reads_either_ratio_name():
    """so3lr-s/m/l output 'a0_ratios', the legacy model 'hirshfeld_ratios' and no
    C6 head; tune-ewald read 'hirshfeld_ratios' and 'c6_ratios' unconditionally
    and raised KeyError for every bundled model."""
    from so3lr.cli.tune_ewald import charges_and_ratios
    assert charges_and_ratios({'partial_charges': 1, 'a0_ratios': 2, 'c6_ratios': 3}) == (1, 2, 3)
    assert charges_and_ratios({'partial_charges': 1, 'hirshfeld_ratios': 2}) == (1, 2, None)
