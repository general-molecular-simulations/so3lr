"""SO3LR-Flash (``so3lr/flash``): the parts that run without a GPU here, and on an NVIDIA GPU
the engine against the neighbour-list potential.

Without a GPU: PME's mesh part and its strain derivative against jaxpme, the real-space
tables against so3lr_dev's pair functions, the reasons the engine choice gives (every
settings key and model hyperparameter classified), and ``so3lr nvt`` falling back to
the neighbour-list potential. With one:
``python -m so3lr.flash.verify`` on 12,288 waters (energy, forces, dE/d(strain), with and
without PME).
"""
import json
import pathlib
import subprocess
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)     # as setup_jax_globals does for every MD run
import jax.numpy as jnp  # noqa: E402

from so3lr.flash import check  # noqa: E402

try:
    from so3lr.flash import engine as E
except Exception as exc:                       # a JAX without Pallas-Triton
    pytest.skip(f"so3lr.flash does not import: {exc}", allow_module_level=True)

DATA = pathlib.Path(__file__).parent / "test_data"
KE = 14.399645351950548


def _cloud(n=300, seed=0):
    rng = np.random.default_rng(seed)
    L = np.array([13.1, 11.7, 14.9])
    R = rng.random((n, 3)) * L - 0.2 * L       # some outside the box, as in MD
    return rng, L, R


def _check(name, a, b, tol=1e-10):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = max(float(np.abs(b).max()), 1e-30)
    assert float(np.abs(a - b).max()) <= tol * scale, f"{name}: max|diff| {np.abs(a - b).max():.2e} of {scale:.2e}"


@pytest.mark.parametrize("smearing,spacing,chunk", [(2.0, 1.55, 70), (4.0, 2.0, 1 << 20)])
def test_pme_electrostatics_matches_jaxpme(smearing, spacing, chunk):
    """Energy per atom, dE/dq, forces and dE/d(strain) of the mesh part, float64."""
    import jaxpme
    rng, L, R = _cloud()
    q = rng.normal(size=len(R))
    mesh = E.pme_mesh_shape(list(L), spacing)
    pbc = jnp.array([True, True, True])

    def e_ref(R, q, s=1.0):
        # box and positions scaled by s together, as jax-md's NPT strain
        cell = jnp.diag(jnp.asarray(L)) * s
        recip = jaxpme.kspace.get_reciprocal(cell)
        kgrid = jaxpme.kspace.get_kgrid_mesh(jnp.diag(jnp.asarray(L)), mesh_spacing=jnp.array([spacing]))
        kvec = jaxpme.kspace.generate_kvectors(recip, kgrid.shape, dtype=jnp.float64, for_ewald=False)
        solver = jaxpme.solvers.pme(jaxpme.potentials.potential(), interpolation_nodes=4)
        pot = solver.kspace(jnp.array([smearing]), q, recip, kgrid, kvec, R * s, jnp.prod(jnp.asarray(L) * s),
                            cell, pbc)
        return KE * q * pot

    Rj, qj = jnp.asarray(R), jnp.asarray(q)
    assert tuple(jaxpme.kspace.get_kgrid_mesh(jnp.diag(jnp.asarray(L)), mesh_spacing=jnp.array([spacing])).shape) == mesh
    gR, gq = jax.grad(lambda R, q: e_ref(R, q).sum(), argnums=(0, 1))(Rj, qj)
    w_ref = jax.grad(lambda s: e_ref(Rj, qj, s).sum())(1.0)
    e, g, f, w = E.pme_electrostatics(tuple(Rj[:, a] for a in range(3)), qj, list(L), mesh, smearing, KE,
                                      jnp.float64, chunk)
    _check("energy", e, e_ref(Rj, qj))
    _check("dE/dq", g, gq)
    _check("forces", jnp.stack(f, -1), -gR)
    _check("dE/dstrain", w, w_ref, 1e-9)


def test_pme_dispersion_matches_jaxpme():
    import jaxpme
    from ase.units import Bohr, Hartree
    from so3lr.mlff.nn.observable.observable_sparse import c6_raw_potential
    rng, L, R = _cloud(seed=1)
    qd = np.abs(rng.normal(size=len(R))) * 3.0 + 0.5
    smearing, spacing = 2.0, 1.55
    mesh = E.pme_mesh_shape(list(L), spacing)
    pbc = jnp.array([True, True, True])

    def e_ref(R, qd, s=1.0):
        cell_b = jnp.diag(jnp.asarray(L)) * s / Bohr
        recip = jaxpme.kspace.get_reciprocal(cell_b)
        kgrid = jaxpme.kspace.get_kgrid_mesh(jnp.diag(jnp.asarray(L)), mesh_spacing=jnp.array([spacing]))
        kvec = jaxpme.kspace.generate_kvectors(recip, kgrid.shape, dtype=jnp.float64, for_ewald=False)
        solver = jaxpme.solvers.pme(jaxpme.potentials.potential(custom_potential=c6_raw_potential()),
                                    interpolation_nodes=4)
        pot = solver.kspace(jnp.array([smearing / Bohr]), qd, recip, kgrid, kvec, R * s / Bohr,
                            jnp.prod(jnp.asarray(L) * s / Bohr), cell_b, pbc)
        return -qd * pot * Hartree

    Rj, qj = jnp.asarray(R), jnp.asarray(qd)
    gR, gq = jax.grad(lambda R, q: e_ref(R, q).sum(), argnums=(0, 1))(Rj, qj)
    w_ref = jax.grad(lambda s: e_ref(Rj, qj, s).sum())(1.0)
    e, g, f, w = E.pme_dispersion(tuple(Rj[:, a] for a in range(3)), qj, list(L), mesh, smearing, Bohr,
                                  Hartree, jnp.float64, 70)
    _check("energy", e, e_ref(Rj, qj))
    _check("dE/dqd", g, gq)
    _check("forces", jnp.stack(f, -1), -gR)
    _check("dE/dstrain", w, w_ref, 1e-9)


@pytest.mark.parametrize("smearing", [None, 2.0])
def test_electrostatic_table(smearing):
    """The tabulated pair function against so3lr_dev's, with and without PME."""
    from so3lr.mlff.nn.observable import observable_sparse as O
    rc, sigma = 10.0, 4.0
    phi, dphi, _ = E._phi_table(rc, sigma, 16384, smearing=smearing)
    grid = np.linspace(0.0, rc, 16385)
    r = np.linspace(0.3, rc - 1e-6, 2001)
    kw = dict(ke=1.0, sigma=sigma, cutoff=rc, cuton=0.45 * rc, neighborlist_format="ordered_sparse")
    one, i0, i1 = jnp.ones(2), jnp.zeros(len(r), int), jnp.ones(len(r), int)
    if smearing is None:
        ref = O.coulomb_erf_shifted_force_smooth(one, jnp.asarray(r), i0, i1, **kw)
    else:
        ref = O.coulomb_erf_shifted_force_smooth_pme(one, jnp.asarray(r), i0, i1, smearing=smearing, **kw)
    _check("phi", np.interp(r, grid, phi), ref, 1e-6)
    _check("dphi", dphi[2:-2], np.gradient(phi, grid)[2:-2], 1e-5)


def test_c6_mesh_table():
    """lr(R) of the dispersion residual: the series and the closed form agree."""
    from ase.units import Bohr
    s = 2.0 / Bohr
    lr6, dlr6 = E._c6_lr_table(10.0, s, 16384, Bohr)
    grid = np.linspace(0.0, 10.0, 16385)
    R = grid[1:] / Bohr
    x = R ** 2 / (2.0 * s * s)
    ref = (1.0 - np.exp(-x) * (1.0 + x + 0.5 * x * x)) / R ** 6
    far = grid[1:] > 1.0
    _check("lr", lr6[1:][far], ref[far], 1e-9)
    _check("dlr/dr", dlr6[2:-2], np.gradient(lr6, grid)[2:-2], 1e-5)
    assert np.isclose(lr6[0], 1.0 / (6.0 * (2.0 * s * s) ** 3))


def _water(reps):
    from ase.io import read
    return read(DATA / "water_64.xyz").repeat(reps)


SETTINGS = dict(ensemble="nvt", model_path=None, lr_cutoff=12.0)


@pytest.mark.parametrize("change,reason", [
    (dict(), None),
    (dict(ensemble="npt"), None),
    (dict(ensemble="npt", npt_coupling="semi-isotropic"), "isotropic"),
    (dict(ensemble="npt", dispersion_tail_correction=True), "tail correction"),
    (dict(kspace_electrostatics="pme"), None),
    (dict(kspace_electrostatics="ewald"), "PME but not ewald"),
    (dict(kspace_dispersion="pme", kspace_dispersion_interp_nodes=6), "4 nodes"),
    (dict(observables=["dipole"]), "observables"),
    (dict(observables=[]), None),
    (dict(fixed_atom_indices=[0, 1]), "fixed atoms"),
    (dict(kspace_npt_cycles=10), None),
    (dict(ensemble="npt", kspace_npt_cycles=10), None),
    (dict(ensemble="npt", kspace_npt_cycles=10, kspace_electrostatics="pme"), "kspace_npt_cycles"),
    (dict(fixed_atom_constraint="cartesian_anchor", pysages_input="p.yaml"), None),
    (dict(buffer_size_multiplier_sr=1.5), None),
    (dict(some_new_setting=True), "does not know the setting 'some_new_setting'"),
    (dict(some_new_setting=None), None),
    (dict(model_path="so3lr-s-2602"), None),
    (dict(model_path="so3lr-l-2606"), None),
    (dict(model_path="so3lr-l-2602"), None),
    (dict(model_path="so3lr"), "legacy"),
])
def test_engine_choice(monkeypatch, change, reason):
    """What auto picks for a GPU, given the settings (37 A of water)."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    problem = check.flash_problem(dict(SETTINGS, **change), _water((3, 3, 3)))
    assert (problem is None) if reason is None else (reason in problem), problem


def test_engine_choice_small_box(monkeypatch):
    monkeypatch.setattr(check, "device_problem", lambda: None)
    assert "narrower than twice the cutoff" in check.flash_problem(SETTINGS, _water((1, 1, 1)))


def test_every_settings_key_classified():
    """A settings key added to the CLI must be sorted into what flash handles, ignores
    or declines; until then auto declines whenever it is set."""
    from so3lr.cli.so3lr_cli import known_yaml_keys
    known = check.HANDLED | check.IRRELEVANT | set(check.UNSUPPORTED)
    assert not known_yaml_keys() - known, sorted(known_yaml_keys() - known)


@pytest.mark.parametrize("edit,reason", [
    (None, None),
    (("num_heads", 8), "degrees, heads and features"),
    (("degrees", [1, 2, 3, 5]), "degrees, heads and features"),
    (("activation_fn", "gelu"), "activation_fn"),
    (("some_new_hyperparameter", True), "does not know the model hyperparameter"),
    (("qk_norm", None), "does not set 'qk_norm'"),
])
def test_model_architecture_check(tmp_path, edit, reason):
    """so3lr-m with one hyperparameter changed (None: removed) is declined."""
    from so3lr.model_registry import resolve_model
    src = resolve_model("so3lr-m-2606").dir
    for f in src.iterdir():
        if f.name != "hyperparameters.json":
            (tmp_path / f.name).symlink_to(f)
    cfg = json.loads((src / "hyperparameters.json").read_text())
    if edit is not None:
        key, value = edit
        if value is None:
            del cfg["model"][key]
        else:
            cfg["model"][key] = value
    (tmp_path / "hyperparameters.json").write_text(json.dumps(cfg))
    problem = check.model_problem(str(tmp_path))
    assert (problem is None) if reason is None else (reason in problem), problem


@pytest.mark.skipif(check.device_problem() is None, reason="this test is for machines without an NVIDIA GPU")
def test_auto_falls_back_without_gpu(tmp_path):
    """Without an NVIDIA GPU, `so3lr nvt` runs the neighbour-list potential and says why."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from subprocess_env import so3lr_subprocess_env
    (tmp_path / "in.xyz").write_text((DATA / "acetone.xyz").read_text())
    result = subprocess.run(
        ["so3lr", "nvt", "--input", "in.xyz", "--no-relax", "--md-cycles", "1", "--md-steps", "2",
         "--output", "out.hdf5"],
        cwd=tmp_path, capture_output=True, text=True, env=so3lr_subprocess_env())
    assert result.returncode == 0, result.stdout + result.stderr
    log = (tmp_path / "out.log").read_text()
    assert "Engine: neighbour-list potential (flash cannot:" in log, log[-2000:]


@pytest.mark.skipif(check.device_problem() is not None, reason="SO3LR-Flash needs an NVIDIA GPU")
@pytest.mark.parametrize("model,pme", [("so3lr-m-2606", ""), ("so3lr-m-2606", "electrostatics,dispersion"),
                                       ("so3lr-l-2606", "electrostatics,dispersion")])
def test_flash_matches_neighbour_lists(tmp_path, model, pme):
    """Energy, forces and dE/d(strain) on 12,288 waters (49.6 A), float64."""
    from ase.io import write
    from so3lr.flash.verify import main
    write(tmp_path / "water.xyz", _water((4, 4, 4)))
    assert main(["--input", str(tmp_path / "water.xyz"), "--model", model, "--precision", "float64",
                 "--lr-cutoff", "10", "--pme", pme])
