"""End-to-end check: the SO3LR cylindrical restraint composes with real PySAGES ABF.

This runs ABF through the actual PySAGES jax-md backend (same bias injection and
momentum-kick path used by production runs) on a 1D quartic double-well, but supplies
the transverse (y,z) confinement via ``cylindrical_restraint_energy`` with its axis
along the biased coordinate x. Because the restraint is flat along x, it must NOT
distort the reconstructed free energy: ABF must recover the analytic double-well to the
same accuracy as the plain run. This simultaneously exercises:

  * composition of the restraint term with a live PySAGES method (jit, autodiff force,
    per-step bias injection),
  * the "no PMF contamination where the restraint is flat" property, and
  * the on-axis safe-norm path (the particle sits at y=z~0 throughout).

Marked ``slow``; run with ``pytest --full``.
"""

import importlib
import pytest

pytest.importorskip("pysages", reason="PySAGES is an optional dependency")
pytest.importorskip("jax_md", reason="jax-md is required for this test")

import numpy as np

from so3lr.cli.so3lr_md import cylindrical_restraint_energy

# physical setup mirrors tests/test_pysages_doublewell.py (metal units: A, eV, amu)
KB = 8.617333262145e-5
T = 300.0
KT = KB * T
K_DW = 0.0106      # eV / A^4  -> barrier ~ 6.56 kT
X0 = 2.0
K_YZ = 5.0         # eV / A^2 transverse stiffness (now supplied by the cylinder)
MASS = 12.0
DT = 0.05
N_STEPS = 5_000_000
GRID_LO, GRID_HI, NBINS = -3.0, 3.0, 64
BOX = 100.0

# cylinder axis along x (the biased CV); radius 0 so the flat-bottom radial term
# reduces to 0.5*K_YZ*(y^2+z^2) -- an *active* transverse confinement that is
# nonetheless flat along x.
import jax.numpy as jnp
_IDX = jnp.array([0])
_W = jnp.ones((1, 1))
_AXIS = jnp.array([1.0, 0.0, 0.0])
_ORIGIN = jnp.zeros(3)


def _energy_fn(R, **kwargs):
    x = R[0, 0]
    e_dw = K_DW * (x * x - X0 * X0) ** 2
    e_cyl = cylindrical_restraint_energy(
        R, _IDX, _W, axis_hat=_AXIS, origin=_ORIGIN, radius=0.0, k_r=K_YZ,
    )
    return e_dw + e_cyl


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
def test_cylinder_restraint_composes_with_abf(tmp_path, monkeypatch):
    """ABF (real PySAGES backend) recovers the double-well FE with the cylindrical
    restraint supplying the transverse confinement -- proving the restraint composes
    and does not corrupt the PMF where it is flat."""
    # PySAGES' jax-md backend writes cv_vs_timestep.txt to cwd; isolate it.
    monkeypatch.chdir(tmp_path)

    import jax
    prior_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)

    import pysages
    from pysages.backends.core import SamplingContext
    from pysages.colvars import Component
    from pysages.methods import ABF

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

    # no NaNs anywhere (safe-norm path is live at y=z~0)
    assert np.all(np.isfinite(fe)), "reconstructed FE contains non-finite values"

    mask = (hist > 200) & (fe_true > 0.5 * KT)
    assert mask.sum() >= 30, f"only {mask.sum()} bins meet sampling/FE-floor criteria"

    ratios = fe[mask] / fe_true[mask]
    median = float(np.median(ratios))
    assert 0.95 < median < 1.05, (
        f"median FE_est/FE_true={median:.4f} outside (0.95,1.05); the restraint "
        f"appears to contaminate the PMF or break bias composition."
    )
    abs_err = np.abs(fe[mask] - fe_true[mask])
    assert abs_err.max() < 1.5 * KT, (
        f"max |FE_est-FE_true| = {abs_err.max()/KT:.3f} kT exceeds 1.5 kT"
    )
    near_top = np.abs(grid_x) < 0.5
    i_top = int(np.argmax(fe * near_top))
    i_left = int(np.argmin(np.abs(grid_x + X0)))
    i_right = int(np.argmin(np.abs(grid_x - X0)))
    barrier_est = fe[i_top] - 0.5 * (fe[i_left] + fe[i_right])
    barrier_true = K_DW * X0 ** 4
    assert abs(barrier_est - barrier_true) < 0.5 * KT, (
        f"barrier_est={barrier_est/KT:.3f} kT vs analytic {barrier_true/KT:.3f} kT"
    )
