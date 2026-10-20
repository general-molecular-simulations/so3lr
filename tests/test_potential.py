import jax
import numpy as np
import pytest


@pytest.fixture
def x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.mark.parametrize("model", ["so3lr-tauri-s", "so3lr"])
def test_float64_runs_in_float64(x64, monkeypatch, model):
    """A float64 run computes in float64 though the checkpoint's parameters are float32: a
    charged, open-shell molecule (its charge and spin embedding is made of parameters alone,
    as are the legacy so3lr's ZBL coefficients) gives what parameters stored in float64 give."""
    from ase.build import molecule
    from so3lr import So3lrCalculator
    import so3lr.mlff.calculators.potential as P

    def energy_forces():
        atoms = molecule("CH3CH2OH")
        atoms.info.update(charge=1, multiplicity=2)
        atoms.calc = So3lrCalculator(model=model, dtype=np.float64, lr_cutoff=100.0)
        return atoms.get_potential_energy(), atoms.get_forces()
    e, f = energy_forces()
    load = P.load_model_from_workdir

    def load_float64(*args, **kwargs):
        net, params = load(*args, **kwargs)
        return net, jax.tree_util.tree_map(lambda v: np.asarray(v, np.float64), params)
    monkeypatch.setattr(P, "load_model_from_workdir", load_float64)
    e64, f64 = energy_forces()
    np.testing.assert_allclose(e, e64, rtol=1e-14)
    np.testing.assert_allclose(f, f64, rtol=1e-12, atol=1e-12)
