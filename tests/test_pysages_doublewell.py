"""Accuracy test for the PySAGES JaxMD backend on a known free-energy profile.

The PySAGES JaxMD backend wraps jax-md's velocity-Verlet integrator and
adds a CV-dependent bias each step. A bug in that wrapper (missing the
2nd-half momentum kick from the bias) makes the bias act at half its
intended magnitude, which drives the reconstructed free energy to be
2x too steep. This test catches such regressions by running ABF on a
1D quartic double-well V(x) = K (x^2 - x0^2)^2 in jax-md NVT-NHC and
checking that the reconstructed FE matches the analytic profile.

Marked ``slow``; run with ``pytest --full``.
"""

import os
import pytest

pytest.importorskip("pysages", reason="PySAGES is an optional dependency")
pytest.importorskip("jax_md", reason="jax-md is required for this test")

import numpy as np

# ---- physical setup (metal-style units: A, eV, amu) ----
KB = 8.617333262145e-5
T = 300.0
KT = KB * T

K_DW = 0.0106    # eV / A^4  -> barrier = K_DW * x0^4 ~ 6.5 kT
X0 = 2.0
K_YZ = 5.0       # eV / A^2 stiff yz confinement
MASS = 12.0
DT = 0.05
N_STEPS = 200_000
GRID_LO, GRID_HI, NBINS = -3.0, 3.0, 64
BOX = 100.0


def _energy_fn(R, **kwargs):
    import jax.numpy as jnp
    x = R[0, 0]; y = R[0, 1]; z = R[0, 2]
    return K_DW * (x * x - X0 * X0) ** 2 + 0.5 * K_YZ * (y * y + z * z)


def _build_context(seed=0):
    import jax
    import jax.numpy as jnp
    from jax_md import space, simulate
    from pysages.backends.core import JaxMDContext, JaxMDContextState

    _, shift = space.free()
    init_jaxmd, apply_jaxmd = simulate.nvt_nose_hoover(
        _energy_fn, shift, DT, KT, chain_length=3, chain_steps=3, tau=DT * 100
    )

    def init_fn(**kwargs):
        key = jax.random.PRNGKey(seed)
        # start at the barrier top so both wells fill symmetrically
        R0 = jnp.array([[0.0, 0.0, 0.0]])
        st = init_jaxmd(key, R0, mass=MASS, kT=KT)
        return JaxMDContextState(state=st, extras={})

    def step_fn(s):
        return JaxMDContextState(state=apply_jaxmd(s.state), extras={})

    box = jnp.eye(3) * BOX
    return JaxMDContext(init_fn=init_fn, step_fn=step_fn, box=box, dt=DT)


def _reconstruct_fe(state):
    Fsum = np.array(state.Fsum).squeeze()
    Ncount = np.array(state.hist)
    mean_force = np.zeros(NBINS)
    nz = Ncount > 0
    mean_force[nz] = Fsum[nz] / Ncount[nz]
    dx = (GRID_HI - GRID_LO) / NBINS
    fe = -np.cumsum(mean_force) * dx
    fe -= fe.min()
    bin_edges = np.linspace(GRID_LO, GRID_HI, NBINS + 1)
    grid_x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return grid_x, fe, Ncount


@pytest.mark.slow
def test_pysages_jaxmd_backend_recovers_double_well_fe(tmp_path, monkeypatch):
    """ABF + jax-md NVT-NHC on a 1D double-well must recover V(x) within ~30%.

    The 2x-too-steep bug (missing 2nd-half momentum kick from the bias) is
    caught immediately: it drives the median ratio to ~2.0, which fails the
    upper tolerance of 1.4.
    """
    # PySAGES' jax-md backend writes a `cv_vs_timestep.txt` log; isolate it.
    monkeypatch.chdir(tmp_path)

    # Float64 is required for stable cumulative integration of <F> across
    # the grid (float32 accumulates ~0.4 kT noise per bin -> spurious slope).
    import jax
    prior_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)

    import importlib
    import pysages
    from pysages.backends.core import SamplingContext
    from pysages.colvars import Component
    from pysages.methods import ABF

    # Reset module-level JIT cache so this test compiles its own step_fn at
    # the current precision (otherwise a stale fp32 run_fn from another test
    # could be reused). The backend module name has a hyphen, so use
    # importlib.
    jaxmd_backend = importlib.import_module("pysages.backends.jax-md")
    jaxmd_backend.jax_fn_container["is_defined"] = False
    jaxmd_backend.jax_fn_container["run_fn"] = None

    cv = Component([0], 0)
    grid = pysages.Grid(lower=(GRID_LO,), upper=(GRID_HI,), shape=(NBINS,))
    method = ABF([cv], grid, N=50)

    try:
        sc = SamplingContext(method, _build_context)
        sc.run(N_STEPS)
    finally:
        jax.config.update("jax_enable_x64", prior_x64)

    grid_x, fe, hist = _reconstruct_fe(sc.sampler.state)
    fe_true = K_DW * (grid_x ** 2 - X0 ** 2) ** 2
    fe_true -= fe_true.min()

    # Restrict to bins that are (a) well-sampled and (b) above the noise floor
    mask = (hist > 200) & (fe_true > 0.5 * KT)
    assert mask.sum() >= 30, (
        f"only {mask.sum()} bins meet sampling/FE-floor criteria; "
        f"hist_min={hist.min()}, hist_med={int(np.median(hist))}"
    )
    ratios = fe[mask] / fe_true[mask]
    median = float(np.median(ratios))
    mean = float(ratios.mean())

    # The half-bias bug (missing 2nd-half momentum kick) makes the
    # reconstructed FE 2x too steep -> median ratio ~1.9-2.0. A correctly
    # implemented bias gives median ~1.0 +/- statistical noise. The
    # acceptance band 0.7-1.4 catches the 2x bug clearly while tolerating
    # the residual ABF noise at this short run length (200k steps).
    assert 0.7 < median < 1.4, (
        f"median FE_est/FE_true={median:.3f} outside acceptance band "
        f"(mean={mean:.3f}, n_bins={int(mask.sum())}). "
        f"A median ~2.0 indicates the bias half-kick bug in the JaxMD backend."
    )

    # Sanity check: max absolute error should be within ~3 kT (overall scale
    # of the profile is ~6.5 kT, so a 50% miss anywhere is suspicious).
    abs_err = np.abs(fe[mask] - fe_true[mask])
    assert abs_err.max() < 3.0 * KT, (
        f"max |FE_est - FE_true| = {abs_err.max():.4f} eV exceeds 3 kT "
        f"({3.0 * KT:.4f} eV); profile is far from analytic."
    )
