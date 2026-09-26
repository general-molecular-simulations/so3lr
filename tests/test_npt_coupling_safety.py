"""Safety guards for NPT cell-coupling modes in the SO3LR MD driver."""

import types

import jax.numpy as jnp
import pytest

from so3lr.cli import so3lr_md


@pytest.mark.parametrize('value, expected', [
    ('isotropic', 'isotropic'),
    (' IsOtRoPiC ', 'isotropic'),
    ('semi-isotropic', 'semi-isotropic'),
    ('SEMI_ISOTROPIC', 'semi-isotropic'),
    ('semi isotropic', 'semi-isotropic'),
])
def test_orthorhombic_couplings_are_accepted(value, expected):
    assert so3lr_md.normalize_npt_coupling(value) == expected


def test_anisotropic_and_unknown_couplings_are_rejected():
    with pytest.raises(ValueError, match="npt_coupling='anisotropic' is experimental"):
        so3lr_md.normalize_npt_coupling('anisotropic')
    with pytest.raises(ValueError, match='Unknown npt_coupling'):
        so3lr_md.normalize_npt_coupling('independent')
    # The integrator factory is public API and must not bypass the guard.
    with pytest.raises(ValueError, match="npt_coupling='anisotropic'"):
        so3lr_md.create_npt_nhc_fn(None, None, 0.001, 0.1, 0.0, {}, {}, False,
                                   npt_coupling='anisotropic')


def test_perform_md_rejects_anisotropic_before_reading_input(monkeypatch):
    monkeypatch.setattr(so3lr_md, 'read', lambda *a, **k: pytest.fail('input was read'))
    with pytest.raises(ValueError, match="npt_coupling='anisotropic'"):
        so3lr_md.perform_md({'ensemble': 'npt', 'npt_coupling': 'anisotropic'})


def _flex_restart_state(*, degrees_of_freedom, box):
    return so3lr_md.jax_md.simulate.NPTNoseHooverFlexState(
        position=jnp.zeros((1, 3)), velocity=jnp.zeros((1, 3)),
        force=jnp.zeros((1, 3)), mass=jnp.ones((1, 1)),
        reference_box=box, box_position=box, box_velocity=jnp.zeros((3, 3)),
        box_mass=jnp.asarray(1.0),
        barostat=types.SimpleNamespace(degrees_of_freedom=jnp.asarray(degrees_of_freedom)),
        thermostat=None, stress_tensor=jnp.zeros((3, 3)),
    )


DIAGONAL_BOX = jnp.diag(jnp.asarray([10.0, 11.0, 12.0]))


def test_semi_isotropic_restart_accepts_two_dof_state():
    state = _flex_restart_state(degrees_of_freedom=2, box=DIAGONAL_BOX)
    so3lr_md.validate_npt_restart_coupling(state, DIAGONAL_BOX, 'semi_isotropic')


def test_semi_isotropic_restart_rejects_anisotropic_checkpoint():
    state = _flex_restart_state(degrees_of_freedom=6, box=DIAGONAL_BOX)
    with pytest.raises(ValueError, match='anisotropic barostat'):
        so3lr_md.validate_npt_restart_coupling(state, DIAGONAL_BOX, 'semi-isotropic')


def test_isotropic_restart_rejects_flexible_checkpoint():
    state = _flex_restart_state(degrees_of_freedom=2, box=DIAGONAL_BOX)
    with pytest.raises(ValueError, match='flexible barostat state'):
        so3lr_md.validate_npt_restart_coupling(state, DIAGONAL_BOX, 'isotropic')


def test_restart_rejects_sheared_cell():
    box = jnp.asarray([[10.0, 0.2, 0.0], [0.1, 11.0, 0.0], [0.0, 0.0, 12.0]])
    state = _flex_restart_state(degrees_of_freedom=2, box=box)
    with pytest.raises(ValueError, match='restart_reset cannot remove cell shear'):
        so3lr_md.validate_npt_restart_coupling(state, box, 'semi-isotropic')
    with pytest.raises(ValueError, match='restart_reset cannot remove cell shear'):
        so3lr_md.validate_orthorhombic_restart_box(box)
