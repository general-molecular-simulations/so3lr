"""tune-ewald's error estimates: the Ewald k-space bound must predict the actual
reciprocal-space force error, and the estimates keep their values."""

import functools

import jax.numpy as jnp
import numpy as np
import pytest


COULOMB_SCALE = 14.399645351950548


def _reciprocal_forces(positions, charges, box, smearing, spacing):
    """Plain-numpy Ewald reciprocal-space forces in a cubic box (eV/A)."""
    kcut = 2 * np.pi / spacing
    nmax = int(np.ceil(kcut * box / (2 * np.pi)))
    grid = np.arange(-nmax, nmax + 1)
    n = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    k = 2 * np.pi * n / box
    k2 = (k**2).sum(1)
    keep = (k2 > 0) & (k2 <= kcut**2)
    k, k2 = k[keep], k2[keep]
    phase = np.exp(1j * positions @ k.T)
    structure_factor = (charges[:, None] * phase).sum(0)
    green = 4 * np.pi / k2 * np.exp(-k2 * smearing**2 / 2)
    im = np.imag(np.conj(structure_factor)[None, :] * phase)
    return COULOMB_SCALE / box**3 * charges[:, None] * ((im * green) @ k)


def _bound(positions, charges, box, smearing, spacing):
    from so3lr.cli.tune_ewald import EwaldErrorBounds

    bounds = EwaldErrorBounds(
        charges=jnp.asarray(charges[:, None]),
        cell=jnp.asarray(np.eye(3) * box),
        positions=jnp.asarray(positions),
    )
    return float(bounds.err_kspace(smearing=smearing, spacing=spacing))


BOX, SMEARING, N_ATOMS = 15.0, 1.5, 100


@functools.lru_cache(maxsize=None)
def _random_case(charge_scale):
    """Random neutral charges in a cubic box, with converged reciprocal forces.

    The converged reference (spacing 0.75, ~34k k-vectors) is the expensive part
    and does not depend on the spacing under test, so it is computed once per
    charge scale.
    """
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, BOX, (N_ATOMS, 3))
    charges = rng.uniform(-0.5, 0.5, N_ATOMS)
    charges = charge_scale * (charges - charges.mean())
    reference = _reciprocal_forces(positions, charges, BOX, SMEARING, 0.75)
    return positions, charges, reference


@pytest.mark.parametrize("charge_scale", [1.0, 3.0])
@pytest.mark.parametrize("spacing", [2.2, 3.0])
def test_ewald_kspace_bound_matches_measured_force_error(charge_scale, spacing):
    box, smearing = BOX, SMEARING
    positions, charges, reference = _random_case(charge_scale)

    truncated = _reciprocal_forces(positions, charges, box, smearing, spacing)
    measured = np.sqrt(np.mean(np.sum((truncated - reference) ** 2, axis=1)))

    ratio = measured / _bound(positions, charges, box, smearing, spacing)
    assert 0.5 < ratio < 2.0, ratio


def _neutral_box():
    return dict(charges=jnp.asarray(np.array([0.5, -0.5] * 4)[:, None]),
                cell=jnp.diag(jnp.array([20.0, 25.0, 30.0])), positions=jnp.zeros((8, 3)))


def test_estimates_keep_their_values():
    """The Kolafa-Perram and Petersen estimates give what the earlier implementation gave
    for the same structure and parameters."""
    from so3lr.cli.tune_ewald import EwaldErrorBounds, PMEErrorBounds

    ewald = EwaldErrorBounds(**_neutral_box())
    assert ewald.err_kspace(smearing=2.0, spacing=3.0) == pytest.approx(8.89619561939042e-06, rel=1e-6)
    assert ewald.err_rspace(smearing=2.0, cutoff=10.0) == pytest.approx(1.959476866247133e-07, rel=1e-6)
    pme = PMEErrorBounds(**_neutral_box())
    for nodes, value in ((3, 6.177889820238487e-05), (4, 1.5689216429297842e-05),
                         (5, 5.3114413847518695e-06)):
        assert pme.err_kspace(smearing=2.0, spacing=1.55, interpolation_nodes=nodes) == \
            pytest.approx(value, rel=1e-6)


@pytest.mark.parametrize("method,nodes", [("pme", (3, 3)), ("pme", (4, 5)), ("p3m", (2, 2)),
                                          ("p3m", (5, 4))])
def test_unsupported_interpolation_orders_are_refused_up_front(method, nodes, monkeypatch):
    """Refused before the geometry is read (the input file does not exist) or the model
    is built."""
    from so3lr.cli import so3lr_md, tune_ewald

    monkeypatch.setattr(so3lr_md, "setup_jax_globals", lambda **_: None)
    monkeypatch.setattr(tune_ewald, "setup_logger", lambda *_: None)
    settings = dict(kspace_electrostatics=method, nodes_lo=nodes[0], nodes_hi=nodes[1],
                    input_file="missing.xyz")
    with pytest.raises(ValueError, match="interpolation nodes"):
        tune_ewald.tune(settings)
