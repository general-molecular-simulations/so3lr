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
    positions, charges, reference = _random_case(charge_scale)

    truncated = _reciprocal_forces(positions, charges, BOX, SMEARING, spacing)
    measured = np.sqrt(np.mean(np.sum((truncated - reference) ** 2, axis=1)))

    ratio = measured / _bound(positions, charges, BOX, SMEARING, spacing)
    assert 0.5 < ratio < 2.0, ratio


def _neutral_box():
    return dict(charges=jnp.asarray(np.array([0.5, -0.5] * 4)[:, None]),
                cell=jnp.diag(jnp.array([20.0, 25.0, 30.0])), positions=jnp.zeros((8, 3)))


def test_estimates_keep_their_values():
    """The Kolafa-Perram and Petersen estimates keep their values for the same structure
    and parameters. PME's take h from the mesh jax-pme builds (32 x 64 x 64 points here),
    not from the spacing: the earlier h = L / (2 L / spacing + 1) gave 6.18e-5, 1.57e-5
    and 5.31e-6, 3.7-8.9x higher."""
    from so3lr.cli.tune_ewald import EwaldErrorBounds, PMEErrorBounds

    ewald = EwaldErrorBounds(**_neutral_box())
    assert ewald.err_kspace(smearing=2.0, spacing=3.0) == pytest.approx(8.89619561939042e-06, rel=1e-6)
    assert ewald.err_rspace(smearing=2.0, cutoff=10.0) == pytest.approx(1.959476866247133e-07, rel=1e-6)
    pme = PMEErrorBounds(**_neutral_box())
    for nodes, value in ((3, 1.6686409324646276e-05), (4, 2.739249344040009e-06),
                         (5, 5.994462993709993e-07)):
        assert pme.err_kspace(smearing=2.0, spacing=1.55, interpolation_nodes=nodes) == \
            pytest.approx(value, rel=1e-6)


@pytest.mark.parametrize("side", [20.0, 31.7, 64.5, 68.03])
def test_pme_candidate_spacings_give_their_mesh(side):
    """A candidate of m gets 2**m mesh points along the shortest side from jax-pme, which
    pads its mesh to a power of two (the earlier 2 L / (2**m - 1) gave 2**(m+1))."""
    from so3lr.cli.tune_ewald import pme_mesh_spacing
    from so3lr.mlff.utils.kspace_utils import kspace_grid_shape

    for m in range(2, 9):
        shape = kspace_grid_shape("pme", pme_mesh_spacing(side, m), np.diag([side, 1.5 * side, 2 * side]))
        assert shape[0] == 2 ** m, (m, shape)


@pytest.mark.parametrize("start,stop,step,expected", [
    (12, 7, -2, [12, 10, 8]), (12, 8, -2, [12, 10, 8]), (2, 6, 2, [2, 4, 6]),
    (2.0, 6.0, 1.5, [2.0, 3.5, 5.0]), (15, 15, -2, [15])])
def test_candidate_ranges_stay_within_their_bounds(start, stop, step, expected):
    """cutoff_lo 7 with cutoff_hi 12, step 2 tried 6 A as well (#42 item 4)."""
    from so3lr.cli.tune_ewald import _inclusive
    assert list(_inclusive(start, stop, step)) == pytest.approx(expected)


def test_recommendation_breaks_timing_ties_by_error():
    """Of the candidates within the accuracy and as fast as the fastest (a shared timing,
    or within 3 %), the smallest error; none within the accuracy gives None."""
    from so3lr.cli.tune_ewald import recommended_index
    errors = [5e-4, 2e-4, 3e-3, 1e-4, 9.7e-4, 3.8e-4]
    timings = [1.0, 1.0, float("inf"), 2.0, 0.99, 1.02]
    assert recommended_index(errors, timings, 1e-3) == 1
    assert recommended_index(errors, timings, 1e-3, tolerance=0.0) == 4
    assert recommended_index(errors, timings, 1e-5) is None


def _charged_box(n=60, box=14.0, seed=1):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, box, (n, 3))
    charges = rng.uniform(-0.6, 0.6, n)
    return positions, charges - charges.mean(), box


@pytest.mark.parametrize("kernel", ["erf", "pme"])
def test_sampled_pair_forces_match_autograd(kernel, x64):
    """SamplePairs' forces (F_i = -2 de/dr along r_i - r_j, the pairs of the sampled atoms
    only) equal the gradient of the energy summed over every pair, on all atoms and on a
    sample of them."""
    from jax_md import space
    from so3lr.cli.tune_ewald import COULOMB_CONSTANT, SamplePairs, get_forces_rs
    from so3lr.mlff.nn.observable.observable_sparse import (coulomb_erf,
                                                            coulomb_erf_shifted_force_smooth_pme)

    positions, charges, box = _charged_box()
    R, q = jnp.asarray(positions, jnp.float64), jnp.asarray(charges, jnp.float64)
    displacement, _ = space.periodic(box)
    cutoff = 6.5
    # every ordered pair within the cutoff (minimum image), as so3lr's sparse lists hold them
    d = positions[:, None] - positions[None]
    r = np.linalg.norm(d - box * np.round(d / box), axis=-1)
    idx = jnp.asarray(np.stack(np.nonzero((r < cutoff) & (r > 0))))
    params = (dict(cutoff=2.0) if kernel == "erf"
              else dict(cutoff=cutoff, cuton=4.5, smearing=2.0))
    energy = coulomb_erf if kernel == "erf" else coulomb_erf_shifted_force_smooth_pme
    reference = get_forces_rs(energy, R, q, displacement, idx_i=idx[0], idx_j=idx[1],
                              ke=COULOMB_CONSTANT, sigma=4.0, **params)

    every = SamplePairs(R, displacement, cutoff)
    assert every.n == len(R)
    forces = every.forces(energy, q, cutoff, ke=COULOMB_CONSTANT, sigma=4.0, **params)
    np.testing.assert_allclose(forces, reference, rtol=1e-9, atol=1e-11)

    some = SamplePairs(R, displacement, cutoff, n_sample=17, chunk=8)
    assert some.n == 17
    forces = some.forces(energy, q, cutoff, ke=COULOMB_CONSTANT, sigma=4.0, **params)
    np.testing.assert_allclose(forces, np.asarray(reference)[some.sample], rtol=1e-9, atol=1e-11)


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


@pytest.mark.parametrize("model,disp_scale,legacy", [
    ("so3lr-tauri-s-2602", 1.2, False), ("so3lr-tauri-m-2602", 1.0, False),
    ("so3lr-tauri-l-2602", 1.2, False), ("so3lr-tauri-s-2606", 1.0, False),
    ("so3lr-tauri-m-2606", 1.0, False), ("so3lr-tauri-l-2606", 1.0, False),
    ("so3lr", 1.2, True)])
def test_error_estimates_take_the_models_pair_settings(model, disp_scale, legacy):
    """The error estimates evaluate the model's own pair functions, so they take its
    electrostatic and dispersion widths (the dispersion width differs between models,
    even within one Tauri generation) and the legacy so3lr's free-atom tables."""
    from so3lr.cli.tune_ewald import model_pair_settings
    assert model_pair_settings(model) == dict(electrostatic_energy_scale=4.0,
                                              dispersion_energy_scale=disp_scale, legacy=legacy)
