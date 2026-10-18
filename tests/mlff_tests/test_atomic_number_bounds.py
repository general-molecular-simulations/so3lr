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
    MAX_Z_DISPERSION_LEGACY,
    MAX_Z_NLH,
    MAX_Z_TRAINED,
    SO3LR_V1_ELEMENTS,
    alphas,
    alphas_legacy,
    check_atomic_numbers_supported,
    supported_atomic_numbers,
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


@pytest.mark.parametrize('dispersion', [True, False])
def test_v2_is_bounded_by_training_not_by_the_tables(dispersion):
    """The v2 models were trained on Z = 1..85 while every table reaches further
    (86 refitted, 92 NLH, 102 legacy), so having parameters for an element says
    nothing about whether the network ever saw it."""
    sup = supported_atomic_numbers(legacy=False, dispersion=dispersion)
    assert max(sup) == MAX_Z_TRAINED == 85
    assert min(sup) == 1


@pytest.mark.parametrize('dispersion', [True, False])
def test_v1_supports_only_the_eight_biomolecular_elements(dispersion):
    """v1 SO3LR is a SET, not a range: a scalar Z <= 17 bound would wrongly
    admit Ne, Na, Mg, Al and Si, none of which it was trained on."""
    sup = supported_atomic_numbers(legacy=True, dispersion=dispersion)
    assert sup == SO3LR_V1_ELEMENTS == {1, 6, 7, 8, 9, 15, 16, 17}
    for z in (10, 11, 12, 13, 14):
        assert z not in sup


def test_tables_all_reach_at_least_the_trained_range():
    """If a table were ever refitted shorter than the trained range, the limit
    must fall back to the table rather than promising coverage that is absent."""
    assert min(MAX_Z_NLH, MAX_Z_DISPERSION) >= MAX_Z_TRAINED
    assert MAX_Z_DISPERSION_LEGACY >= max(SO3LR_V1_ELEMENTS)


def test_accepts_the_trained_elements():
    # H, C, N, O, F, P, S, Cl -- what SO3LR is actually trained on.
    check_atomic_numbers_supported([1, 6, 7, 8, 9, 15, 16, 17], supported_atomic_numbers())


def test_accepts_the_boundary_and_rejects_one_past_it():
    sup = supported_atomic_numbers()
    check_atomic_numbers_supported([max(sup)], sup)
    with pytest.raises(ValueError, match=r'Z = 1\.\.85'):
        check_atomic_numbers_supported([max(sup) + 1], sup)


def test_rejects_heavy_element_by_name():
    with pytest.raises(ValueError, match=r'Pu \(Z=94\)'):
        check_atomic_numbers_supported([1, 6, 94], supported_atomic_numbers())


def test_rejects_padding_and_negative_z():
    # These boundaries see real atoms only; Z=0 padding here means malformed input.
    with pytest.raises(ValueError, match='Z=0'):
        check_atomic_numbers_supported([1, 0, 8], supported_atomic_numbers())
    with pytest.raises(ValueError, match='Z=-1'):
        check_atomic_numbers_supported([-1], supported_atomic_numbers())


def test_message_names_every_offender_once():
    with pytest.raises(ValueError) as exc:
        check_atomic_numbers_supported([88, 88, 94, 1], supported_atomic_numbers())
    msg = str(exc.value)
    assert msg.count('Ra (Z=88)') == 1
    assert 'Pu (Z=94)' in msg


def test_context_is_prefixed():
    with pytest.raises(ValueError, match='^my_structure.xyz: '):
        check_atomic_numbers_supported([94], supported_atomic_numbers(),
                                       context='my_structure.xyz')


def test_untrained_elements_are_rejected_even_where_tables_cover_them():
    """Ra (88) has legacy reference data and Rn (86) has refitted data, but
    neither was in the training set, so both must be refused."""
    for z in (86, 88):
        for legacy in (False, True):
            with pytest.raises(ValueError):
                check_atomic_numbers_supported(
                    [z], supported_atomic_numbers(legacy=legacy))
    # At (85) is the heaviest trained v2 element and must still pass
    check_atomic_numbers_supported([85], supported_atomic_numbers())


def test_legacy_rejects_elements_the_v2_models_accept():
    """Na and Si are inside the v2 range but were never in v1's training set."""
    v1, v2 = supported_atomic_numbers(legacy=True), supported_atomic_numbers()
    for z in (11, 14, 26):
        check_atomic_numbers_supported([z], v2)
        with pytest.raises(ValueError, match=r'supports only H, C, N, O, F, P, S, Cl'):
            check_atomic_numbers_supported([z], v1)


def test_empty_and_array_inputs_are_accepted():
    check_atomic_numbers_supported([], supported_atomic_numbers())
    check_atomic_numbers_supported(np.array([1, 6, 8], dtype=np.int32),
                                   supported_atomic_numbers())


def test_unknown_coverage_skips_the_check():
    """A potential built by an older code path carries no element set. Better to
    behave as before than to fail on every structure."""
    check_atomic_numbers_supported([94], None)


def test_md_cli_rejects_before_building_neighbor_lists():
    """so3lr md and so3lr min both funnel through process_model, so the check
    lives there. A stub potential is enough: the point is that it raises before
    reaching to_jax_md_custom, which would need a real model."""
    from so3lr.cli.so3lr_md import process_model

    class _StubPotential:
        supported_atomic_numbers = SO3LR_V1_ELEMENTS

    with pytest.raises(ValueError, match=r'Na \(Z=11\)'):
        process_model(potential=_StubPotential(),
                      species=np.array([8, 1, 1, 11], dtype=np.int32),
                      displacement=None, box=None)


@pytest.mark.parametrize('model_name, bad_z, bad_match', [
    # Rn (86) has refitted reference data and Ra (88) has legacy data, but the
    # training set stops at At (85), so both must be refused by both families.
    ('so3lr-tauri-m-2606', 86, r'Rn \(Z=86\)'),
    ('so3lr-tauri-m-2606', 94, r'Pu \(Z=94\)'),
    ('so3lr', 11, r'Na \(Z=11\)'),
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
    expected = supported_atomic_numbers(legacy=(model_name == 'so3lr'))
    assert calc.supported_atomic_numbers == expected

    # The check fires before any graph construction or jit tracing, so this is
    # cheap: no model forward happens for the rejected structure.
    bad_atoms = Atoms(numbers=[bad_z], positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match=bad_match):
        calc.calculate(bad_atoms)
