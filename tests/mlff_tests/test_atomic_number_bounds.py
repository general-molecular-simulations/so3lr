"""Bounds on the free-atom reference tables, and the check that enforces them.

The tables in ``dispersion_ref_data`` are indexed with ``Z - 1`` inside jitted
code, so an out-of-range element is resolved by JAX's out-of-bounds policy rather
than raising. These tests pin the table lengths (they are not all equal, and the
smallest one binds) and the eager check that turns a bad Z into an error.
"""
import numpy as np
import pytest

from so3lr.mlff.nn.observable.dispersion_ref_data import (
    MAX_Z_DISPERSION,
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
    'legacy, dispersion, expected',
    [
        (False, True, 86),    # so3lr-s/m/l: the refitted alpha_0/C6 tables bind
        (False, False, 92),   # dispersion off: only NLH repulsion binds
        (True, True, 102),    # v1 SO3LR: legacy references
    ],
)
def test_max_supported_atomic_number(legacy, dispersion, expected):
    assert max_supported_atomic_number(legacy=legacy, dispersion=dispersion) == expected


def test_accepts_the_trained_elements():
    # H, C, N, O, F, P, S, Cl -- what SO3LR is actually trained on.
    check_atomic_numbers_supported([1, 6, 7, 8, 9, 15, 16, 17], max_supported_atomic_number())


def test_accepts_the_boundary_and_rejects_one_past_it():
    max_z = max_supported_atomic_number()
    check_atomic_numbers_supported([max_z], max_z)
    with pytest.raises(ValueError, match=r'Z = 1\.\.86'):
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


def test_legacy_model_accepts_what_the_new_tables_cannot():
    # Ra (88) is outside the refitted tables but inside the legacy ones.
    with pytest.raises(ValueError):
        check_atomic_numbers_supported([88], max_supported_atomic_number())
    check_atomic_numbers_supported([88], max_supported_atomic_number(legacy=True))


def test_empty_and_array_inputs_are_accepted():
    check_atomic_numbers_supported([], max_supported_atomic_number())
    check_atomic_numbers_supported(np.array([1, 6, 8], dtype=np.int32),
                                   max_supported_atomic_number())


@pytest.mark.parametrize('model_name, expected_max_z, bad_z, bad_match', [
    # Non-legacy: the refitted alpha_0/C6 tables bind at 86, so Pu (94) is out.
    ('so3lr-m-2606', 86, 94, r'Pu \(Z=94\)'),
    # Legacy: v1 references reach Fm (102) -- Pu is fine, Z=104 is not (and has
    # no symbol in the 102-row species table, so it is reported numerically).
    ('so3lr', 102, 104, r'Z=104'),
])
def test_end_to_end_wiring_rejects_unsupported_elements(
        model_name, expected_max_z, bad_z, bad_match):
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
    assert calc.max_atomic_number == expected_max_z

    # The check fires before any graph construction or jit tracing, so this is
    # cheap: no model forward happens for the rejected structure.
    bad_atoms = Atoms(numbers=[bad_z], positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match=bad_match):
        calc.calculate(bad_atoms)
