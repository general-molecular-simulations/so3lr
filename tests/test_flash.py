"""SO3LR-Flash (``so3lr/flash``): the parts that run without a GPU here, and on a GPU it runs on
(NVIDIA, AMD Instinct) the engine against the neighbour-list potential.

Without a GPU: PME's mesh part and its strain derivative against jaxpme, the real-space
tables against so3lr_dev's pair functions, the reasons the engine choice gives (every
settings key and model hyperparameter classified), and ``so3lr nvt`` falling back to
the neighbour-list potential. With one:
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


@pytest.mark.skipif(check.device_problem() is None, reason="this test is for machines without a GPU flash runs on")
def test_auto_falls_back_without_gpu(tmp_path):
    """Without a GPU flash runs on, `so3lr nvt` runs the neighbour-list potential and says why."""
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


GPU = pytest.mark.skipif(check.device_problem() is not None, reason="SO3LR-Flash needs an NVIDIA or AMD Instinct GPU")


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


GPUS = pytest.mark.skipif(check.device_problem() is not None or len(jax.local_devices()) < 2,
                          reason="several GPUs: needs two GPUs flash runs on or more")


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
