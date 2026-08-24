"""Which elements the model may be asked about, and the check that enforces it.

Two separate limits. The reference tables in ``dispersion_ref_data`` are indexed
with ``Z - 1`` inside jitted code, so an element past the end of a table is
resolved by JAX's out-of-bounds policy rather than raising -- a plausible number
belonging to some other element. And the models were TRAINED only on Z = 1..85,
which is stricter than any table (86, 92, 102): holding parameters for an element
does not mean the network ever saw it. Training coverage is what binds.
"""
import numpy as np
import pytest

from so3lr.mlff.nn.observable.dispersion_ref_data import (
    MAX_Z_DISPERSION,
    MAX_Z_TRAINED,
    MAX_Z_DISPERSION_LEGACY,
    MAX_Z_NLH,
    alphas,
    alphas_legacy,
    check_atomic_numbers_supported,
    max_supported_atomic_number,
    C6_coef,
    C6_coef_legacy,
    NLH_AA,
    NLH_BB,
)


def test_bounds_match_the_tables():
    # Derived from the arrays, so a refit at a different length cannot desync them.
    assert MAX_Z_NLH == NLH_AA.shape[0] == NLH_BB.shape[0] == 92
    assert MAX_Z_DISPERSION == len(alphas) == len(C6_coef) == 86
    assert MAX_Z_DISPERSION_LEGACY == len(alphas_legacy) == len(C6_coef_legacy) == 102


def test_nlh_table_is_square_in_z():
    # Indexed as [min(z_i, z_j), max(z_i, z_j)], so both axes must span the same Z.
    assert NLH_AA.shape[:2] == (MAX_Z_NLH, MAX_Z_NLH)
    assert NLH_BB.shape[:2] == (MAX_Z_NLH, MAX_Z_NLH)


@pytest.mark.parametrize(
    'legacy, dispersion',
    [(False, True), (False, False), (True, True), (True, False)],
)
def test_training_coverage_binds_before_any_table(legacy, dispersion):
    """The models were trained on Z = 1..85. Every reference table reaches
    further (86 for the refitted alpha_0/C6, 92 for NLH, 102 for the legacy
    references), so having parameters for an element says nothing about whether
    the network ever saw it -- training coverage is what must bind."""
    assert max_supported_atomic_number(legacy=legacy, dispersion=dispersion) == MAX_Z_TRAINED
    assert MAX_Z_TRAINED == 85


def test_tables_all_reach_at_least_the_trained_range():
    """If a table were ever refitted shorter than the trained range, the limit
    must fall back to the table rather than promising coverage that is absent."""
    assert min(MAX_Z_NLH, MAX_Z_DISPERSION) >= MAX_Z_TRAINED
    assert MAX_Z_DISPERSION_LEGACY >= MAX_Z_TRAINED


def test_accepts_the_trained_elements():
    # H, C, N, O, F, P, S, Cl -- what SO3LR is actually trained on.
    check_atomic_numbers_supported([1, 6, 7, 8, 9, 15, 16, 17], max_supported_atomic_number())


def test_accepts_the_boundary_and_rejects_one_past_it():
    max_z = max_supported_atomic_number()
    check_atomic_numbers_supported([max_z], max_z)
    with pytest.raises(ValueError, match=r'Z = 1\.\.85'):
        check_atomic_numbers_supported([max_z + 1], max_z)


def test_rejects_heavy_element_by_name():
    with pytest.raises(ValueError, match=r'Pu \(Z=94\)'):
        check_atomic_numbers_supported([1, 6, 94], max_supported_atomic_number())


def test_rejects_padding_and_negative_z():
    # These boundaries see real atoms only; Z=0 padding here means malformed input.
    with pytest.raises(ValueError, match='Z=0'):
        check_atomic_numbers_supported([1, 0, 8], max_supported_atomic_number())
    with pytest.raises(ValueError, match='Z=-1'):
        check_atomic_numbers_supported([-1], max_supported_atomic_number())


def test_message_names_every_offender_once():
    with pytest.raises(ValueError) as exc:
        check_atomic_numbers_supported([88, 88, 94, 1], max_supported_atomic_number())
    msg = str(exc.value)
    assert msg.count('Ra (Z=88)') == 1
    assert 'Pu (Z=94)' in msg


def test_context_is_prefixed():
    with pytest.raises(ValueError, match='^my_structure.xyz: '):
        check_atomic_numbers_supported([94], max_supported_atomic_number(),
                                       context='my_structure.xyz')


def test_untrained_elements_are_rejected_even_where_tables_cover_them():
    """Ra (88) has legacy reference data and Rn (86) has refitted data, but
    neither was in the training set, so both must be refused."""
    for z in (86, 88):
        for legacy in (False, True):
            with pytest.raises(ValueError):
                check_atomic_numbers_supported(
                    [z], max_supported_atomic_number(legacy=legacy))
    # At (85) is the heaviest trained element and must still pass
    check_atomic_numbers_supported([85], max_supported_atomic_number())


def test_empty_and_array_inputs_are_accepted():
    check_atomic_numbers_supported([], max_supported_atomic_number())
    check_atomic_numbers_supported(np.array([1, 6, 8], dtype=np.int32),
                                   max_supported_atomic_number())


@pytest.mark.parametrize('model_name, bad_z, bad_match', [
    # Rn (86) has refitted reference data and Ra (88) has legacy data, but the
    # training set stops at At (85), so both must be refused by both families.
    ('so3lr-m-2606', 86, r'Rn \(Z=86\)'),
    ('so3lr-m-2606', 94, r'Pu \(Z=94\)'),
    ('so3lr', 88, r'Ra \(Z=88\)'),
])
def test_end_to_end_wiring_rejects_unsupported_elements(
        model_name, bad_z, bad_match):
    """The limit must survive the whole chain: checkpoint cfg ->
    PotentialSparse.create_from_workdir -> AseCalculatorSparse.calculate.
    The helpers above are unit-tested; this pins the wiring."""
    from ase import Atoms

    from so3lr.mlff.calculators.ase_calculator import AseCalculatorSparse
    from so3lr.model_registry import resolve_model

    spec = resolve_model(model_name)
    calc = AseCalculatorSparse.create_from_workdir(
        workdir=spec.dir,
        from_file=spec.from_file,
        head_index=spec.head_index,
        dtype=np.float32,
    )
    assert calc.max_atomic_number == MAX_Z_TRAINED

    # The check fires before any graph construction or jit tracing, so this is
    # cheap: no model forward happens for the rejected structure.
    bad_atoms = Atoms(numbers=[bad_z], positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match=bad_match):
        calc.calculate(bad_atoms)
