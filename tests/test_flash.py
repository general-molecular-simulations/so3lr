"""SO3LR-Flash (``so3lr/flash``): the parts that run without a GPU here, and on an NVIDIA GPU
the engine against the neighbour-list potential.

Without a GPU: PME's mesh part and its strain derivative against jaxpme, the real-space
tables against so3lr_dev's pair functions, the reasons the engine choice gives (every
settings key and model hyperparameter classified), ``so3lr nvt`` falling back to the
neighbour-list potential, and the MD driver around a stand-in for the engine (PySAGES
runs as perform_md's, the repeat of a cycle that overflowed). With one:
``python -m so3lr.flash.verify`` on 12,288 waters (energy, forces, dE/d(strain), with and
without PME). With two or more: the same on several GPUs as on one, and MD on two.
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
    (dict(cylinder_restraint_enabled=True, cylinder_restraint_radius=5.0), "cylindrical restraint"),
    (dict(cylinder_restraint_enabled=False, cylinder_restraint_axis=[0.0, 0.0, 1.0]), None),
    (dict(do_enhanced_sampling=True), None),
    (dict(do_enhanced_sampling=True, ensemble="nve", flash_mode="fast"), None),
    (dict(do_enhanced_sampling=True, ensemble="npt"), "enhanced sampling runs nvt and nve, not npt"),
    (dict(do_enhanced_sampling=True, ensemble="nvt-langevin"), "enhanced sampling runs nvt and nve"),
    (dict(do_enhanced_sampling=True, flash_mode="lean"), "enhanced sampling runs flash_mode fast"),
    (dict(do_enhanced_sampling=True, flash_devices=2), "enhanced sampling runs on one GPU"),
    (dict(do_enhanced_sampling=False, ensemble="npt"), None),
    (dict(buffer_size_multiplier_sr=1.5), None),
    (dict(some_new_setting=True), "does not know the setting 'some_new_setting'"),
    (dict(some_new_setting=None), None),
    (dict(model_path="so3lr-tauri-s-2602"), None),
    (dict(model_path="so3lr-tauri-l-2606"), None),
    (dict(model_path="so3lr-tauri-l-2602"), None),
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
    """so3lr-tauri-m with one hyperparameter changed (None: removed) is declined."""
    from so3lr.model_registry import resolve_model
    src = resolve_model("so3lr-tauri-m-2606").dir
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


def test_cell_grid_takes_every_box_from_the_smallest():
    """Every cubic box from the smallest the grid takes on is accepted (cells rc wide where
    they leave enough per side, wider ones where not); a large box keeps rc-wide cells."""
    for rc, lr, smallest in [(4.5, 10.0, 25.0), (4.5, 12.0, 30.0), (5.0, 12.0, 30.0)]:
        assert check.box_problem([smallest - 0.1] * 3, rc, lr) is not None
        for L in np.arange(smallest, 60.0, 0.1):
            assert check.box_problem([L] * 3, rc, lr) is None, (rc, lr, L)
    assert check.cell_grid([49.6] * 3, 4.5, 12.0) == ([11, 11, 11], 3)


def test_restart_box_is_checked(tmp_path, monkeypatch):
    """A restart starts in its checkpoint's box, so that is the box auto checks."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    atoms = _water((3, 3, 3))
    np.savez(tmp_path / "small.npz", box=np.array([26.0, 26.0, 26.0]))
    np.savez(tmp_path / "same.npz", box=np.diag(atoms.cell[:]).copy())
    problem = check.flash_problem(dict(SETTINGS, restart_load_path=str(tmp_path / "small.npz")), atoms)
    assert "too small for the cell grid" in problem, problem
    assert check.flash_problem(dict(SETTINGS, restart_load_path=str(tmp_path / "same.npz")), atoms) is None


@pytest.mark.parametrize("bad", [dict(engine="Flash"), dict(engine="neighbour-list"), dict(flash_mode="Fast"),
                                 dict(flash_mixed="false"), dict(flash_devices=0), dict(flash_devices="2"),
                                 dict(flash_devices=2, flash_mode="lean"), dict(flash_devices=4096)])
def test_bad_engine_options_raise(bad):
    """Settings files bypass the CLI's choices: a value flash does not take is an error."""
    with pytest.raises(ValueError):
        check.validate_options(bad)
    check.validate_options(dict(engine="flash", flash_mode="lean", flash_mixed=False))
    check.validate_options(dict(engine="standard", flash_mode="lean", flash_devices=4096))


def test_flash_modes_match_the_engine():
    assert tuple(E.MODES) == check.FLASH_MODES[1:]


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


def test_standard_engine_is_logged(tmp_path, monkeypatch):
    """`--engine standard` names the engine in the log, as auto and flash do (the line
    comes before perform_md, so no MD runs)."""
    from click.testing import CliRunner
    from so3lr.cli import so3lr_md
    from so3lr.cli.so3lr_cli import cli
    monkeypatch.setattr(so3lr_md, "setup_jax_globals", lambda **kwargs: None)
    monkeypatch.setattr(so3lr_md, "perform_md", lambda *args, **kwargs: None)
    log = tmp_path / "out.log"
    result = CliRunner().invoke(cli, ["nvt", "--input", str(DATA / "acetone.xyz"), "--no-relax",
                                      "--output", str(tmp_path / "out.hdf5"), "--log-file", str(log),
                                      "--engine", "standard"])
    assert result.exit_code == 0, result.output
    assert "Engine: neighbour-list potential (--engine standard)" in log.read_text()


def test_flash_refuses_untrained_elements(monkeypatch):
    """Flash runs the neighbour-list engine's element check before anything else (so no
    GPU is needed here): Rn is past what so3lr-tauri-m was trained on, an error rather than a
    value gathered from another element's row of the reference tables."""
    from so3lr.flash import md as flash_md
    monkeypatch.setattr(flash_md, "setup_jax_globals", lambda **kwargs: None)
    atoms = _water((3, 3, 3))
    atoms.numbers[0] = 86
    with pytest.raises(ValueError, match="Z=86"):
        flash_md.perform_md_flash(SETTINGS, atoms=atoms)


class _StandInEngine:
    """FlashSO3LR as perform_md_flash uses it, around the neighbour-list potential set up as
    perform_md sets it up for PySAGES (Cartesian positions, dense lists), so the flash MD
    driver runs without a GPU. The jitted force reports a table overflow (NaN forces) until
    the engine has been widened ``overflows`` times."""
    P, device_mesh, K, maxh = 1, None, 0, 0

    def __init__(self, atoms, settings, overflows=0):
        from functools import partial
        from so3lr.cli import so3lr_md as M
        precision = settings["precision"]
        dtype = jnp.float64 if precision == "float64" else jnp.float32
        geo = M.atoms_to_jnp(atoms, precision=precision)
        R, box, disp, _, _ = M.handle_box("periodic_cart", geo["positions"], np.asarray(atoms.cell[:]),
                                          precision=dtype)
        pot = M.load_model(settings["model_path"], precision=dtype, lr_cutoff=settings["lr_cutoff"])
        nfn, nfn_lr, efn = M.process_model(pot, geo["species"], disp, box, precision=precision,
                                           buffer_size_multiplier_sr=settings["buffer_size_multiplier_sr"],
                                           buffer_size_multiplier_lr=settings["buffer_size_multiplier_lr"])
        nb, nb_lr = nfn.allocate(R, box=box), nfn_lr.allocate(R, box=box)
        energy = partial(efn, k_smearing=None, k_smearing_disp=None, has_aux=False)

        def ef(R):
            R = R.astype(dtype)                 # a float64 state (flash_mixed) around a float32 model
            n = nb.update(R, neighbor=nb.idx, box=box)
            n_lr = nb_lr.update(R, neighbor=nb_lr.idx, box=box)
            e, g = jax.value_and_grad(energy)(R, neighbor=n.idx, neighbor_lr=n_lr.idx, box=box,
                                              k_grid=None, k_grid_disp=None)
            return e, -g
        self._ef, self._jit = ef, jax.jit(ef)
        self.rc, self.loops, self.overflows = float(pot.cutoff), {}, overflows

    def _energy_forces(self, R):
        e, f = self._ef(R)
        return e, f, jnp.asarray(self.overflows > 0), jnp.zeros((), R.dtype)

    def widen(self):
        self.overflows -= 1
        self.loops = {}

    def energy_forces_strain(self, R):
        return (*self._jit(R), jnp.zeros((), R.dtype))

    def energy_forces(self, R):
        return self._jit(R)


PYSAGES_ABF = {"method": "abf", "cv": [
    {"type": "coordinationnumber", "indices": [4, 7], "cn_exps": [6, 30], "species": 1,
     "grid_min": 0.0, "grid_max": 1.5, "grid_stride": 32, "periodic": False},
    {"type": "distance", "grp1": [4], "grp2": [7], "grid_min": 1.0, "grid_max": 4.0,
     "grid_stride": 24, "periodic": False}]}


PYSAGES_HARMONIC = {"method": "harmonic", "cv": [PYSAGES_ABF["cv"][1]],
                    "method_args": {"kspring": 10.0, "center": [2.3]}}


def _pysages_run(tmp_path, name, engine, monkeypatch, overflows=0, pysages=PYSAGES_ABF,
                 input_file=DATA / "benzoic_acid_aq_1.xyz", staged=False, **change):
    """A PySAGES run (by default ABF of benzoic acid, 2 cycles of 5 steps, float64) in its
    own directory (PySAGES writes cv_vs_timestep.txt there) by perform_md, by
    perform_md_flash, or (engine "run") by so3lr_md.run with flash allowed."""
    import yaml
    from so3lr.cli import so3lr_md
    from so3lr.cli.so3lr_cli import YAML_DEFAULTS
    from so3lr.flash import md as flash_md
    d = tmp_path / name
    d.mkdir()
    (d / "pysages.yaml").write_text(yaml.safe_dump(pysages))
    s = dict(YAML_DEFAULTS, input_file=str(input_file), output_file=str(d / "out.hdf5"),
             output_format="hdf5", log_file=str(d / "out.log"), ensemble="nvt", precision="float64",
             flash_mixed=False, model_path="so3lr-tauri-m-2606", lr_cutoff=12.0, relax_before_run=False,
             md_cycles=2, md_steps=5, save_buffer=1, do_enhanced_sampling=True,
             pysages_input=str(d / "pysages.yaml"), restart_save_path=str(d / "r.npz"),
             restart_pysages_save_path=str(d / "r.pkl"))
    s.update(change)
    monkeypatch.chdir(d)
    if engine == "standard":
        so3lr_md.perform_md(s, restart=s.get("restart_load_path") is not None)
        return d
    monkeypatch.setattr(flash_md, "flash_problem", lambda settings, atoms: None)
    monkeypatch.setattr(flash_md, "build_engine",
                        lambda atoms, settings, dtype: (_StandInEngine(atoms, settings, overflows), staged))
    if engine == "run":
        monkeypatch.setattr(check, "flash_problem", lambda settings, atoms: None)
        so3lr_md.run(s)
    else:
        flash_md.perform_md_flash(s, restart=s.get("restart_load_path") is not None)
    return d


def _pysages_outputs(d):
    import pickle
    import h5py
    with h5py.File(d / "out.hdf5") as f:
        traj = {k: f[k][:] for k in ("positions", "momenta")}
    with open(d / "r.pkl", "rb") as f:
        method_state = pickle.load(f).states[0]
    state = np.load(d / "r.npz") if (d / "r.npz").exists() else None
    return traj, state, method_state, np.loadtxt(d / "cv_vs_timestep.txt")


def _same_run(out, ref, tol):
    """Per-step CVs, trajectory, final MD state and every array of the method's state."""
    (traj, state, method_state, cv), (traj_ref, state_ref, method_ref, cv_ref) = out, ref
    _check("CVs", cv, cv_ref, tol)
    for k in ("positions", "momenta"):
        _check(k, traj[k], traj_ref[k], tol)
    if state_ref is not None:
        for k in ("position", "momentum", "force"):
            _check(f"final {k}", state[k], state_ref[k], tol)
    for k in method_ref._fields:
        if getattr(method_ref, k) is not None:
            _check(f"method state {k}", getattr(method_state, k), getattr(method_ref, k), 10 * tol)


def test_flash_runs_pysages_as_perform_md(tmp_path, monkeypatch):
    """perform_md_flash's PySAGES path (here around the neighbour-list potential) gives
    perform_md's ABF run: per-step CVs, trajectory, final state, the ABF accumulators, and
    across the cycle boundary. A table overflow in the first cycle repeats it from the last
    accepted state and PySAGES result, with PySAGES' loop compiled again."""
    pytest.importorskip("pysages")
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard", "standard", monkeypatch))
    for overflows in (0, 1):
        out = _pysages_outputs(_pysages_run(tmp_path, f"flash{overflows}", "flash", monkeypatch, overflows))
        log = (tmp_path / f"flash{overflows}" / "out.log").read_text()
        assert ("cycle repeated" in log) == bool(overflows)
        (traj, state, abf, cv), (traj_ref, state_ref, abf_ref, cv_ref) = out, ref
        # the overflowing attempt also wrote its CVs, as perform_md's neighbour-list retries do
        _check("CVs", cv[-len(cv_ref):], cv_ref, 1e-9)
        for k in ("positions", "momenta"):
            _check(k, traj[k], traj_ref[k], 1e-9)
        for k in ("position", "momentum", "force"):
            _check(f"final {k}", state[k], state_ref[k], 1e-9)
        assert not state["fractional_coordinates"] and not state_ref["fractional_coordinates"]
        np.testing.assert_array_equal(np.asarray(abf.hist), np.asarray(abf_ref.hist))
        for k in ("Fsum", "force", "bias"):
            _check(f"ABF {k}", getattr(abf, k), getattr(abf_ref, k), 1e-8)


def test_flash_continues_a_perform_md_pysages_run(tmp_path, monkeypatch):
    """A PySAGES restart of perform_md continued by perform_md_flash, and by perform_md:
    the same run (the ABF state and the MD state carry over between the engines)."""
    pytest.importorskip("pysages")
    first = _pysages_run(tmp_path, "first", "standard", monkeypatch, md_cycles=1)
    restart = dict(restart_load_path=str(first / "r.npz"), restart_pysages_load_path=str(first / "r.pkl"))
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard", "standard", monkeypatch, **restart))
    out = _pysages_outputs(_pysages_run(tmp_path, "flash", "flash", monkeypatch, **restart))
    (traj, state, abf, cv), (traj_ref, state_ref, abf_ref, cv_ref) = out, ref
    _check("CVs", cv, cv_ref, 1e-9)
    _check("positions", traj["positions"], traj_ref["positions"], 1e-9)
    _check("final momentum", state["momentum"], state_ref["momentum"], 1e-9)
    assert int(np.asarray(abf.hist).sum()) == int(np.asarray(abf_ref.hist).sum()) == 15
    np.testing.assert_array_equal(np.asarray(abf.hist), np.asarray(abf_ref.hist))
    _check("ABF Fsum", abf.Fsum, abf_ref.Fsum, 1e-8)


def test_flash_runs_pysages_in_nve(tmp_path, monkeypatch):
    """A harmonic bias in NVE on flash (the NVE state rebuilt after each cycle) gives
    perform_md's run."""
    pytest.importorskip("pysages")
    kw = dict(pysages=PYSAGES_HARMONIC, ensemble="nve")
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard", "standard", monkeypatch, **kw))
    out = _pysages_outputs(_pysages_run(tmp_path, "flash", "flash", monkeypatch, **kw))
    _same_run(out, ref, 1e-9)


def test_flash_runs_pysages_in_float32(tmp_path, monkeypatch):
    """float32: flash with a float32 MD state gives perform_md's run; flash's default float64
    state (flash_mixed) continues a float32 PySAGES restart of perform_md as perform_md does,
    to float32 accuracy; perform_md continues flash's float64 PySAGES restart."""
    pytest.importorskip("pysages")
    f32 = dict(precision="float32")
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard", "standard", monkeypatch, **f32))
    out = _pysages_outputs(_pysages_run(tmp_path, "flash", "flash", monkeypatch, flash_mixed=False, **f32))
    _same_run(out, ref, 1e-4)
    first = _pysages_run(tmp_path, "first", "standard", monkeypatch, md_cycles=1, **f32)
    restart = dict(restart_load_path=str(first / "r.npz"), restart_pysages_load_path=str(first / "r.pkl"), **f32)
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard2", "standard", monkeypatch, **restart))
    mixed = _pysages_run(tmp_path, "mixed", "flash", monkeypatch, flash_mixed=True, **restart)
    out = _pysages_outputs(mixed)
    assert out[1]["momentum"].dtype == np.float64 and ref[1]["momentum"].dtype == np.float32
    _same_run(out, ref, 1e-4)
    back = dict(restart_load_path=str(mixed / "r.npz"), restart_pysages_load_path=str(mixed / "r.pkl"), **f32)
    _, _, method_state, cv = _pysages_outputs(_pysages_run(tmp_path, "back", "standard", monkeypatch, **back))
    assert len(cv) == 10 and int(np.asarray(method_state.hist).sum()) == 25


def test_flash_runs_pysages_without_a_coordination_number(tmp_path, monkeypatch):
    """CVs that need no neighbour list (ionsitedistance finds the waters in the starting
    positions; distance_pbc) on 512 waters: flash gives perform_md's run."""
    pytest.importorskip("pysages")
    from ase.io import write
    write(tmp_path / "w.xyz", _water((2, 2, 2)))
    pysages = {"method": "abf", "cv": [
        {"type": "ionsitedistance", "indices": [0], "mode": "acid", "grid_min": 1.875, "grid_max": 6.125,
         "grid_stride": 17, "periodic": False},
        {"type": "distance_pbc", "grp1": [0], "grp2": [3], "grid_min": 1.0, "grid_max": 9.0,
         "grid_stride": 32, "periodic": False}]}
    # no MD checkpoint: its force is one step's, where the lag below shows in full
    kw = dict(pysages=pysages, input_file=tmp_path / "w.xyz", md_steps=3, restart_save_path=None)
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard", "standard", monkeypatch, **kw))
    out = _pysages_outputs(_pysages_run(tmp_path, "flash", "flash", monkeypatch, **kw))
    assert int(np.asarray(ref[2].hist).sum()) == 6
    # perform_md's step takes the force with the neighbour lists of the step before (a pair
    # entering the cutoff joins a step late), flash's with those of the step: here ~1e-5 in
    # the forces and momenta, ~1e-8 in the CVs (a wrong set of waters moves them by Angstroms)
    _same_run(out, ref, 1e-4)


def test_flash_pysages_falls_back_from_a_staged_mode(tmp_path, monkeypatch):
    """A system that only a staged memory mode fits: auto runs the PySAGES run on the
    neighbour-list potential, having written nothing on flash."""
    pytest.importorskip("pysages")
    d = _pysages_run(tmp_path, "run", "run", monkeypatch, staged=True)
    log = (d / "out.log").read_text()
    assert "flash cannot: it failed to start: ValueError: enhanced sampling runs the jitted fast mode" in log
    assert "Starting MD simulation (flash engine)" not in log
    _, _, method_state, cv = _pysages_outputs(d)
    assert len(cv) == 10 and int(np.asarray(method_state.hist).sum()) == 10


def test_flash_saves_pysages_state_without_an_md_checkpoint(tmp_path, monkeypatch):
    """restart_pysages_save_path without restart_save_path: flash saves the PySAGES state
    alone, as perform_md does."""
    pytest.importorskip("pysages")
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard", "standard", monkeypatch, restart_save_path=None))
    out = _pysages_outputs(_pysages_run(tmp_path, "flash", "flash", monkeypatch, restart_save_path=None))
    assert out[1] is None and ref[1] is None
    assert int(np.asarray(out[2].hist).sum()) == 10
    _same_run(out, ref, 1e-9)


def test_flash_widens_and_repeats_a_jitted_cycle(tmp_path, monkeypatch):
    """A table overflow in the jitted mode, without PySAGES: the cycle is repeated with the
    widened tables traced again (jax-md's step is jitted, and jit keeps the trace of the
    old ones) and gives the run without an overflow."""
    import h5py
    out = []
    for overflows in (0, 1):
        d = _pysages_run(tmp_path, f"plain{overflows}", "flash", monkeypatch, overflows,
                         do_enhanced_sampling=False, restart_pysages_save_path=None)
        assert ("cycle repeated" in (d / "out.log").read_text()) == bool(overflows)
        with h5py.File(d / "out.hdf5") as f:
            out.append({k: f[k][:] for k in ("positions", "momenta")})
    for k in ("positions", "momenta"):
        _check(k, out[1][k], out[0][k], 1e-12)


GPU = pytest.mark.skipif(check.device_problem() is not None, reason="SO3LR-Flash needs an NVIDIA GPU")


@GPU
@pytest.mark.parametrize("model,pme", [("so3lr-tauri-m-2606", ""), ("so3lr-tauri-m-2606", "electrostatics,dispersion"),
                                       ("so3lr-tauri-l-2606", "electrostatics,dispersion")])
def test_flash_matches_neighbour_lists(tmp_path, model, pme):
    """Energy, forces and dE/d(strain) on 12,288 waters (49.6 A), float64."""
    from ase.io import write
    from so3lr.flash.verify import main
    write(tmp_path / "water.xyz", _water((4, 4, 4)))
    assert main(["--input", str(tmp_path / "water.xyz"), "--model", model, "--precision", "float64",
                 "--lr-cutoff", "10", "--pme", pme])


@GPU
@pytest.mark.parametrize("model", ["so3lr-tauri-m-2606", "so3lr-tauri-l-2606"])
def test_float32_follows_float64(model):
    """float32 (each model's attention launch; the backward's filter products as TF32x3 on
    tensor cores, which float64 never runs) gives float64's energy and forces to float32
    accuracy."""
    from so3lr.flash.loader import load_params
    atoms = _water((3, 3, 3))
    out = {}
    for precision in ("float64", "float32"):
        params, statics = load_params(model, precision=precision, lr_cutoff=12.0)
        eng = E.FlashSO3LR(params, statics, atoms.get_atomic_numbers(), np.asarray(atoms.cell[:]))
        out[precision] = eng.energy_forces(jnp.asarray(atoms.get_positions(), statics["dtype"]))
    (e, f), (e_ref, f_ref) = out["float32"], out["float64"]
    _check("energy", e, e_ref, 1e-6)
    _check("forces", f, f_ref, 1e-4)


@GPU
def test_narrow_neighbour_rows_widen():
    """Neighbour rows too short for water: the engine widens them and gives what wide ones
    give, jitted and staged (the kernels never read past a row that overflowed)."""
    from so3lr.flash.loader import load_params
    atoms = _water((3, 3, 3))
    params, statics = load_params(None, precision="float64", lr_cutoff=12.0)
    args = (params, statics, atoms.get_atomic_numbers(), np.asarray(atoms.cell[:]))
    R = jnp.asarray(atoms.get_positions())
    e_ref, f_ref = E.FlashSO3LR(*args).energy_forces(R)
    for staged in (False, True):
        eng = E.FlashSO3LR(*args, max_neighbours=16)
        e, f = (eng.energy_forces_staged if staged else eng.energy_forces)(R)
        assert eng.K > 16
        _check("energy", e, e_ref, 1e-12)
        _check("forces", f, f_ref, 1e-10)


@GPU
def test_box_shrinks_to_the_smallest_grid():
    """NPT near the smallest box: a box that leaves the margin downwards is laid out again
    with no room left to shrink rather than refused, and gives what an engine built for it
    gives."""
    from so3lr.flash.loader import load_params
    atoms = _water((3, 3, 3))
    atoms.set_cell(atoms.cell[:] * (31.0 / atoms.cell[0, 0]), scale_atoms=True)
    params, statics = load_params(None, precision="float64", lr_cutoff=12.0)
    z = atoms.get_atomic_numbers()
    eng = E.FlashSO3LR(params, statics, z, np.asarray(atoms.cell[:]), box_margin=0.03)
    L = 30.05
    R = jnp.asarray(atoms.get_positions() * (L / 31.0))
    eng.set_box([L, L, L])
    e, f = eng.energy_forces(R)
    e_ref, f_ref = E.FlashSO3LR(params, statics, z, np.eye(3) * L).energy_forces(R)
    _check("energy", e, e_ref, 1e-12)
    _check("forces", f, f_ref, 1e-10)


def _so3lr(args, cwd, env=None):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from subprocess_env import so3lr_subprocess_env
    result = subprocess.run(["so3lr", *args], cwd=cwd, capture_output=True, text=True,
                            env=so3lr_subprocess_env(env))
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


@GPU
def test_md_restarts_on_flash(tmp_path):
    """A float32 NVT checkpoint of the neighbour-list potential continued as NVE on flash in
    the jitted mode (an ensemble change: masses and momenta come from the checkpoint);
    NPT with a restart file, continued from it."""
    from ase.io import write
    write(tmp_path / "w.xyz", _water((3, 3, 3)))
    run = ["--input", "w.xyz", "--no-relax", "--md-cycles", "2", "--md-steps", "5"]
    _so3lr(["nvt", *run, "--engine", "standard", "--output", "a.hdf5", "--restart-save", "a.npz"], tmp_path)
    _so3lr(["nve", *run, "--flash-mode", "fast", "--output", "b.hdf5", "--restart-load", "a.npz"], tmp_path)
    _so3lr(["npt", *run, "--pressure", "1", "--output", "c.hdf5", "--restart-save", "c.npz"], tmp_path)
    _so3lr(["npt", *run, "--pressure", "1", "--output", "d.hdf5", "--restart-load", "c.npz"], tmp_path)
    for name in "bcd":
        log = (tmp_path / f"{name}.log").read_text()
        assert "Engine: flash" in log, log[-2000:]


@GPU
def test_auto_falls_back_when_flash_fails_to_start(tmp_path, monkeypatch):
    """A flash engine that cannot be built (say, a kernel that does not compile on this
    GPU): auto runs the neighbour-list potential and says why."""
    from ase.io import write
    from so3lr.cli.so3lr_cli import YAML_DEFAULTS
    from so3lr.cli.so3lr_md import run
    from so3lr.flash import md as flash_md

    def broken(*args, **kwargs):
        raise RuntimeError("no kernels here")
    monkeypatch.setattr(flash_md, "build_engine", broken)
    write(tmp_path / "w.xyz", _water((3, 3, 3)))
    run(dict(YAML_DEFAULTS, input_file=str(tmp_path / "w.xyz"), output_file=str(tmp_path / "o.hdf5"),
             output_format="hdf5", log_file=str(tmp_path / "o.log"), ensemble="nvt", relax_before_run=False,
             md_cycles=1, md_steps=2, save_buffer=1))
    log = (tmp_path / "o.log").read_text()
    assert "flash cannot: it failed to start: RuntimeError: no kernels here" in log, log[-2000:]


@GPU
def test_flash_runs_pysages(tmp_path):
    """ABF (a coordination number and a distance) on 5,184 waters, continued from one float32
    checkpoint: auto picks flash, which follows --engine standard to float32 accuracy (flash
    keeps the MD state in float64) in the per-step CVs, the trajectory and the samples."""
    import h5py
    import pickle
    import yaml
    from ase.io import write
    write(tmp_path / "w.xyz", _water((3, 3, 3)))
    _so3lr(["nvt", "--input", "w.xyz", "--no-relax", "--md-cycles", "1", "--md-steps", "2", "--engine",
            "standard", "--output", "eq.hdf5", "--restart-save", "eq.npz"], tmp_path)
    (tmp_path / "pysages.yaml").write_text(yaml.safe_dump({"method": "abf", "cv": [
        {"type": "coordinationnumber", "indices": [0, 3], "cn_exps": [6, 24], "species": 1,
         "grid_min": 0.0, "grid_max": 6.0, "grid_stride": 30, "periodic": False},
        {"type": "distance_pbc", "grp1": [0], "grp2": [3], "grid_min": 1.0, "grid_max": 9.0,
         "grid_stride": 32, "periodic": False}]}))
    for engine in ("standard", "auto"):
        d = tmp_path / engine
        d.mkdir()
        (d / "md.yaml").write_text(yaml.safe_dump(dict(
            input_file="../w.xyz", output_file="out.hdf5", log_file="out.log", ensemble="nvt", md_cycles=2,
            md_steps=10, relax_before_run=False, engine=engine, do_enhanced_sampling=True, pysages_input="../pysages.yaml",
            restart_load_path="../eq.npz", restart_pysages_save_path="r.pkl")))
        _so3lr(["--settings", "md.yaml"], d)
    assert "Engine: flash (auto)" in (tmp_path / "auto" / "out.log").read_text()
    cv, cv_ref = (np.loadtxt(tmp_path / e / "cv_vs_timestep.txt") for e in ("auto", "standard"))
    assert cv.shape == cv_ref.shape == (20, 2)
    np.testing.assert_allclose(cv, cv_ref, atol=1e-3)
    with h5py.File(tmp_path / "auto" / "out.hdf5") as a, h5py.File(tmp_path / "standard" / "out.hdf5") as b:
        np.testing.assert_allclose(a["positions"][:], b["positions"][:], atol=1e-3)
    hist = []
    for e in ("auto", "standard"):
        with open(tmp_path / e / "r.pkl", "rb") as f:
            hist.append(np.asarray(pickle.load(f).states[0].hist))
    assert hist[0].sum() == hist[1].sum() == 20


GPUS = pytest.mark.skipif(check.device_problem() is not None or len(jax.local_devices()) < 2,
                          reason="several GPUs: needs two NVIDIA GPUs or more")


@GPUS
@pytest.mark.parametrize("precision,tol", [("float64", 1e-10), ("float32", 1e-5)])
def test_several_gpus_give_one_gpus_forces(precision, tol):
    """Energy, forces and dE/d(strain) on two and on all visible GPUs as on one: PME for
    both, an atom count the GPUs do not divide (the last one shares rows); and with
    neighbour rows too short, widened as on one GPU."""
    from so3lr.flash.loader import load_params
    atoms = _water((3, 3, 3))[:-3]
    params, statics = load_params(None, precision=precision, lr_cutoff=10.0)
    args = (params, statics, atoms.get_atomic_numbers(), np.asarray(atoms.cell[:]))
    kw = dict(pme_electrostatics=(2.0, 1.55), pme_dispersion=(2.0, 1.55), **E.MODES["fast"][0])
    R = jnp.asarray(atoms.get_positions(), statics["dtype"])
    e_ref, f_ref, w_ref = E.FlashSO3LR(*args, **kw).energy_forces_strain(R)
    for P in sorted({2, len(jax.local_devices())}):
        e, f, w = E.FlashSO3LR(*args, devices=P, **kw).energy_forces_strain(R)
        _check(f"energy on {P} GPUs", e, e_ref, tol / 100)
        _check(f"forces on {P} GPUs", f, f_ref, tol)
        _check(f"dE/d(strain) on {P} GPUs", w, w_ref, tol)
    if precision == "float64":
        eng = E.FlashSO3LR(*args, devices=2, max_neighbours=16, **kw)
        e, f, w = eng.energy_forces_strain(R)
        assert eng.K > 16
        _check("energy, rows widened on 2 GPUs", e, e_ref, 1e-12)
        _check("forces, rows widened on 2 GPUs", f, f_ref, 1e-10)


@GPUS
def test_md_on_two_gpus(tmp_path):
    """`so3lr nvt --flash-devices 2` follows the one-GPU trajectory (float64, PME)."""
    import h5py
    from ase.io import write
    write(tmp_path / "w.xyz", _water((3, 3, 3)))
    run = ["--input", "w.xyz", "--no-relax", "--md-cycles", "2", "--md-steps", "5", "--precision", "float64",
           "--lr-cutoff", "10", "--kspace-electrostatics", "pme"]
    # this process holds most of each GPU (its own tests): the runs allocate as they go,
    # leaving NCCL and cuSolver room
    env = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    _so3lr(["nvt", *run, "--output", "a.hdf5"], tmp_path, env)
    _so3lr(["nvt", *run, "--flash-devices", "2", "--output", "b.hdf5"], tmp_path, env)
    assert "Flash: 2 GPUs" in (tmp_path / "b.log").read_text()
    with h5py.File(tmp_path / "a.hdf5") as a, h5py.File(tmp_path / "b.hdf5") as b:
        _check("positions", b["positions"][:], a["positions"][:], 1e-9)
