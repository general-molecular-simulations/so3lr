"""The Ewald k-space error bound must predict the actual reciprocal-space force error."""

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


@pytest.mark.parametrize("charge_scale", [1.0, 3.0])
@pytest.mark.parametrize("spacing", [2.2, 3.0])
def test_ewald_kspace_bound_matches_measured_force_error(charge_scale, spacing):
    rng = np.random.default_rng(0)
    box, smearing, n_atoms = 15.0, 1.5, 100
    positions = rng.uniform(0, box, (n_atoms, 3))
    charges = rng.uniform(-0.5, 0.5, n_atoms)
    charges = charge_scale * (charges - charges.mean())

    reference = _reciprocal_forces(positions, charges, box, smearing, 0.75)
    truncated = _reciprocal_forces(positions, charges, box, smearing, spacing)
    measured = np.sqrt(np.mean(np.sum((truncated - reference) ** 2, axis=1)))

    ratio = measured / _bound(positions, charges, box, smearing, spacing)
    assert 0.5 < ratio < 2.0, ratio
