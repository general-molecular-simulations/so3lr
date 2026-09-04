"""Safety guards for NPT cell-coupling modes in the SO3LR MD driver."""

import types

import jax.numpy as jnp
import pytest

from so3lr.cli import so3lr_md


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('isotropic', 'isotropic'),
        (' IsOtRoPiC ', 'isotropic'),
        ('semi-isotropic', 'semi-isotropic'),
        ('SEMI_ISOTROPIC', 'semi-isotropic'),
        ('semi isotropic', 'semi-isotropic'),
        (' semi   isotropic ', 'semi-isotropic'),
    ],
)
def test_normalize_npt_coupling_accepts_only_orthorhombic_modes(value, expected):
    assert so3lr_md.normalize_npt_coupling(value) == expected


def test_normalize_npt_coupling_rejects_fully_flexible_cell():
    with pytest.raises(ValueError, match='silently use inconsistent cells'):
        so3lr_md.normalize_npt_coupling('anisotropic')


def test_normalize_npt_coupling_rejects_unknown_mode():
    with pytest.raises(ValueError, match='Unknown npt_coupling'):
        so3lr_md.normalize_npt_coupling('independent')


def test_integrator_factory_cannot_bypass_anisotropic_guard(monkeypatch):
    called = False

    def forbidden_flexible_integrator(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError('unsupported integrator must not be constructed')

    monkeypatch.setattr(
        so3lr_md.jax_md.simulate,
        'npt_nose_hoover_flex',
        forbidden_flexible_integrator,
    )

    with pytest.raises(ValueError, match="npt_coupling='anisotropic'"):
        so3lr_md.create_npt_nhc_fn(
            energy_fn=None,
            shift=None,
            dt=0.001,
            T=0.1,
            P=0.0,
            nhc_kwargs={},
            barostat_kwargs={},
            lr=False,
            npt_coupling='anisotropic',
        )

    assert called is False


def test_integrator_factory_canonicalizes_semi_isotropic_alias(monkeypatch):
    captured = {}

    def flexible_integrator(*args, **kwargs):
        captured['coupling'] = kwargs['coupling']
        return (lambda *a, **kw: None), (lambda *a, **kw: None)

    monkeypatch.setattr(
        so3lr_md.jax_md.simulate,
        'npt_nose_hoover_flex',
        flexible_integrator,
    )

    so3lr_md.create_npt_nhc_fn(
        energy_fn=None,
        shift=None,
        dt=0.001,
        T=0.1,
        P=0.0,
        nhc_kwargs={},
        barostat_kwargs={},
        lr=False,
        npt_coupling='SEMI_ISOTROPIC',
    )

    assert captured['coupling'] == 'semi-isotropic'


def _flex_restart_state(*, degrees_of_freedom, box, box_velocity=None):
    if box_velocity is None:
        box_velocity = jnp.zeros((3, 3))
    return so3lr_md.jax_md.simulate.NPTNoseHooverFlexState(
        position=jnp.zeros((1, 3)),
        velocity=jnp.zeros((1, 3)),
        force=jnp.zeros((1, 3)),
        mass=jnp.ones((1, 1)),
        reference_box=box,
        box_position=box,
        box_velocity=box_velocity,
        box_mass=jnp.asarray(1.0),
        barostat=types.SimpleNamespace(
            degrees_of_freedom=jnp.asarray(degrees_of_freedom)
        ),
        thermostat=None,
        stress_tensor=jnp.zeros((3, 3)),
    )


def test_semi_isotropic_restart_accepts_diagonal_two_dof_state():
    box = jnp.diag(jnp.asarray([10.0, 11.0, 12.0]))
    state = _flex_restart_state(degrees_of_freedom=2, box=box)
    so3lr_md.validate_npt_restart_coupling(state, box, 'semi_isotropic')


def test_semi_isotropic_restart_rejects_legacy_anisotropic_dof():
    box = jnp.diag(jnp.asarray([10.0, 11.0, 12.0]))
    state = _flex_restart_state(degrees_of_freedom=6, box=box)
    with pytest.raises(ValueError, match='fully flexible/anisotropic barostat'):
        so3lr_md.validate_npt_restart_coupling(state, box, 'semi-isotropic')


def test_semi_isotropic_restart_rejects_sheared_cell_even_with_two_dof():
    box = jnp.asarray([
        [10.0, 0.2, 0.0],
        [0.1, 11.0, 0.0],
        [0.0, 0.0, 12.0],
    ])
    state = _flex_restart_state(degrees_of_freedom=2, box=box)
    with pytest.raises(ValueError, match='restart_reset cannot remove cell shear'):
        so3lr_md.validate_npt_restart_coupling(state, box, 'semi-isotropic')


def test_orthorhombic_restart_box_rejects_shear_for_reset_path():
    box = jnp.asarray([
        [10.0, 0.1, 0.0],
        [0.0, 11.0, 0.0],
        [0.0, 0.0, 12.0],
    ])
    with pytest.raises(ValueError, match='start a fresh run'):
        so3lr_md.validate_orthorhombic_restart_box(box)


def test_perform_md_rejects_anisotropic_before_input_or_model_work(monkeypatch):
    def forbidden_read(*args, **kwargs):
        raise AssertionError('input must not be read for an unsupported mode')

    monkeypatch.setattr(so3lr_md, 'read', forbidden_read)

    with pytest.raises(ValueError, match="npt_coupling='anisotropic'"):
        so3lr_md.perform_md({
            'ensemble': 'npt',
            'npt_coupling': 'anisotropic',
        })
