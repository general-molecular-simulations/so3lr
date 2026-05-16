"""Accuracy test for the PySAGES JaxMD backend on a known free-energy profile.

The PySAGES JaxMD backend wraps jax-md's velocity-Verlet integrator and
adds a CV-dependent bias each step. A bug in that wrapper (missing the
2nd-half momentum kick from the bias) makes the bias act at half its
intended magnitude, which drives the reconstructed free energy to be
2x too steep. This test catches such regressions by running ABF on a
1D quartic double-well V(x) = K (x^2 - x0^2)^2 in jax-md NVT-NHC and
checking that the reconstructed FE matches the analytic profile.

At 5M steps (~40 s wall on CPU) the recovery is quantitatively tight:
median ratio fe_est/fe_true ~ 1.00 +/- 0.005, max |fe_est - fe_true|
~ 1.1 kT, barrier height matches the analytic value within ~0.1 kT,
and left-right well asymmetry stays below ~0.05 kT. See the
convergence study at tests/.slow_outputs/.../convergence_study.py
and the multi-seed characterisation at .../asymmetry_study.py.

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

K_DW = 0.0106    # eV / A^4  -> barrier = K_DW * x0^4 ~ 6.56 kT
X0 = 2.0
K_YZ = 5.0       # eV / A^2 stiff yz confinement
MASS = 12.0
DT = 0.05
N_STEPS = 5_000_000   # ~42 s wall; converged within a few % of analytic.
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
    """ABF + jax-md NVT-NHC on a 1D double-well must recover V(x) within a
    few percent at 5M steps.

    Pass criteria (all five must hold):
      1. >= 30 bins meet the sampling/FE-floor mask (sanity).
      2. 0.95 < median(fe_est/fe_true) < 1.05    (per-bin shape, well above noise floor)
      3. max |fe_est - fe_true|  <  1.5 kT       (worst-bin absolute error)
      4. |barrier_est - K_DW*X0^4| < 0.5 kT      (direct physical comparison)
      5. |fe_left_well - fe_right_well| < 0.10 kT (symmetry of recovery)

    The 2x-steep half-bias bug fails (2), (3) and (4) by very wide margins
    (median ~ 2.0; max abs err ~ 6 kT; barrier ~ 13 kT vs analytic 6.56 kT).

    Empirical baseline numbers on the unmodified PySAGES jax-md backend
    (5M steps, 8 seeds):
      median ratio        : 0.992 - 1.003  -- inside (0.95, 1.05)
      max abs err         : 1.08 - 1.12 kT -- inside 1.5 kT
      barrier_dev         : -0.068..+0.005 kT -- inside +/- 0.5 kT
      |fe_L - fe_R|       :  0.001 - 0.041 kT -- inside 0.10 kT
    Margins: bounds (2)-(4) sit 12-24 sigma above the worst observed seed;
    bound (5) sits ~4.5 sigma above (worst |asym| = 0.041 kT, std ~ 0.013).
    The half-bias bug fails (2)-(4) at >= 270 sigma -- detection is
    essentially deterministic.
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

    # (1) sanity: enough bins survive masking
    mask = (hist > 200) & (fe_true > 0.5 * KT)
    assert mask.sum() >= 30, (
        f"only {mask.sum()} bins meet sampling/FE-floor criteria; "
        f"hist_min={hist.min()}, hist_med={int(np.median(hist))}"
    )
    ratios = fe[mask] / fe_true[mask]
    median = float(np.median(ratios))
    mean = float(ratios.mean())

    # (2) Per-bin median ratio within 5% of unity. Half-bias bug -> ~2.0.
    assert 0.95 < median < 1.05, (
        f"median FE_est/FE_true={median:.4f} outside (0.95, 1.05) band "
        f"(mean={mean:.4f}, n_bins={int(mask.sum())}). "
        f"A median ~2.0 indicates the bias half-kick bug in the JaxMD backend."
    )

    # (3) Worst-bin absolute error in kT.
    abs_err = np.abs(fe[mask] - fe_true[mask])
    assert abs_err.max() < 1.5 * KT, (
        f"max |FE_est - FE_true| = {abs_err.max():.5f} eV = "
        f"{abs_err.max() / KT:.3f} kT exceeds 1.5 kT "
        f"({1.5 * KT:.5f} eV); profile is far from analytic."
    )

    # (4) Direct barrier-height comparison, in physical units.
    #   Barrier = peak of fe_est near x=0, minus the mean of the two well
    #   minima. This is the physical observable an experimentalist would
    #   quote and is independent of the (arbitrary) global zero-of-energy
    #   chosen by fe -= fe.min().
    near_top = np.abs(grid_x) < 0.5
    i_top = int(np.argmax(fe * near_top))
    i_left = int(np.argmin(np.abs(grid_x + X0)))
    i_right = int(np.argmin(np.abs(grid_x - X0)))
    barrier_est = fe[i_top] - 0.5 * (fe[i_left] + fe[i_right])
    barrier_true = K_DW * X0 ** 4  # = 0.1696 eV exactly
    assert abs(barrier_est - barrier_true) < 0.5 * KT, (
        f"barrier_est = {barrier_est:.5f} eV ({barrier_est / KT:.3f} kT) "
        f"differs from analytic {barrier_true:.5f} eV "
        f"({barrier_true / KT:.3f} kT) by more than 0.5 kT "
        f"({0.5 * KT:.5f} eV). Half-bias bug doubles this to ~6.6 kT."
    )

    # (5) Left-right well symmetry. V(+X0) = V(-X0) = 0 analytically; any
    #   asymmetry comes from cumulative-integration drift in <F>. Empirically
    #   (8 seeds at 5M steps) |asym| <= 0.041 kT with std ~ 0.013 kT; bound
    #   set at 0.10 kT for ~4.5 sigma margin (false-positive prob ~ 3e-6).
    #   This bound is a secondary symmetry check; the primary bug-detection
    #   signal is the median-ratio assertion (2), which catches the half-bias
    #   bug at >= 270 sigma.
    well_asym = abs(fe[i_left] - fe[i_right])
    assert well_asym < 0.10 * KT, (
        f"fe(left well) - fe(right well) = {well_asym:.5f} eV = "
        f"{well_asym / KT:.4f} kT exceeds 0.10 kT "
        f"({0.10 * KT:.5f} eV); recovered FE is asymmetric beyond "
        f"the typical cumulative-integration drift."
    )
