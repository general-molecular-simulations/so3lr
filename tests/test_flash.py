"""SO3LR-Flash (``so3lr/flash``): the parts that run without a GPU here, and on a GPU it runs on
(NVIDIA, AMD Instinct) the engine against the neighbour-list potential.

Without a GPU: PME's mesh part and its strain derivative against jaxpme, the real-space
tables against so3lr_dev's pair functions, the reasons the engine choice gives (every
settings key and model hyperparameter classified), the box free space runs in, and
``so3lr nvt`` logging the engine it picked and, falling back to the neighbour-list
potential, why. With one: ``python -m so3lr.flash.verify`` on 12,288 waters (energy,
forces, dE/d(strain), with and without PME); ``so3lr opt``, ``so3lr eval``, MD and the ASE
calculator against the neighbour-list potential, periodic and with no box (12 and 1000 A
cutoffs). With two or more: the same on several GPUs as on one, and MD on two.
"""
import itertools
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


def test_electrostatic_far_field():
    """A 1000 A cutoff (the gas phase's): the table out to 8 sigma, and past it the closed
    form the long-range kernel takes (_far_field), give so3lr_dev's pair function and its
    derivative, and meet at the table's end; below 0.45 rc the closed form without its
    switching function is the same."""
    from so3lr.mlff.nn.observable import observable_sparse as O
    rc, sigma = 1000.0, 4.0
    end = 8.0 * sigma
    phi, dphi, h = E._phi_table(rc, sigma, 16384, end=end)
    assert h == end / 16384
    r = np.linspace(end, rc - 1e-6, 4001)
    far, dfar = (np.asarray(v) for v in E._far_field(jnp.asarray(r), rc))
    below = r < 0.45 * rc
    for a, b in zip(E._far_field(jnp.asarray(r[below]), rc, switch=False), (far[below], dfar[below])):
        _check("unswitched", a, b, 1e-15)
    kw = dict(ke=1.0, sigma=sigma, cutoff=rc, cuton=0.45 * rc, neighborlist_format="ordered_sparse")

    def ref(x):
        i0, i1 = jnp.zeros(x.shape, int), jnp.ones(x.shape, int)
        return O.coulomb_erf_shifted_force_smooth(jnp.ones(2), x, i0, i1, **kw)
    _check("phi", far, ref(jnp.asarray(r)), 1e-12)
    _check("dphi", dfar, jax.vmap(jax.grad(lambda x: ref(x[None])[0]))(jnp.asarray(r)), 1e-10)
    assert abs(far[0] - phi[-1]) < 1e-15 and abs(dfar[0] - dphi[-1]) < 1e-15


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


def test_legacy_pieces_match_so3lr():
    """The legacy so3lr's own pieces against so3lr_dev's modules (float64): LayerNorm, the
    Hirshfeld-ratio head, and the learnable ZBL repulsion from the repulsion kernel's tables."""
    import flax.linen as nn
    from so3lr.flash import blocks
    from so3lr.flash.loader import load_params
    from so3lr.mlff.nn.observable.observable_sparse import HirshfeldSparse, ZBLRepulsionSparse
    params, statics = load_params("so3lr", precision="float64")
    assert statics["legacy"] and not (statics["rms_norm"] or statics["qk_norm"])
    rng = np.random.default_rng(0)
    n = 40
    x = jnp.asarray(rng.normal(size=(n, 128)) * 2.0 + 0.5)
    z = jnp.asarray(rng.choice([1, 6, 7, 8, 9, 15, 16, 17], size=n))
    mask = jnp.ones(n)
    p = params["layers_1"]["layer_normalization_2"]
    _check("LayerNorm", blocks.layer_norm(p, x), nn.LayerNorm().apply({"params": p}, x), 1e-12)
    head = HirshfeldSparse(prop_keys=None, regression_dim=128, activation_fn=jax.nn.silu,
                           output_is_zero_at_init=False, legacy=True)
    ref = head.apply({"params": params["observables_2"]}, dict(x=x, atomic_numbers=z, node_mask=mask))
    _check("hirshfeld ratios", blocks.legacy_hirshfeld_head(params["observables_2"], x, z),
           ref["hirshfeld_ratios"], 1e-12)
    # ZBL, a pair per atom (float64 parameters, as the neighbour-list potential casts them)
    i, j = jnp.arange(n), jnp.asarray((np.arange(n) + rng.integers(1, n, n)) % n)
    d = jnp.asarray(rng.uniform(0.3, 1.6, n))
    cut = jnp.asarray(rng.uniform(0.1, 1.0, n))
    inputs = dict(atomic_numbers=z.astype(jnp.int32), node_mask=mask, cut=cut, idx_i=i, idx_j=j, d_ij=d)
    ref = ZBLRepulsionSparse(prop_keys=None).apply({"params": params["observables_0"]["zbl_repulsion"]},
                                                   inputs)["zbl_repulsion"]
    # as _sr_kernel takes the tables: [min(Z) - 1, max(Z) - 1]
    lo, hi = jnp.minimum(z[i], z[j]) - 1, jnp.maximum(z[i], z[j]) - 1
    y = (params["repulsion_a"][lo, hi] * jnp.exp(-params["repulsion_b"][lo, hi] * d[:, None])).sum(-1)
    e_pair = E._switch(d, 0.0, 1.5)[0] * cut * KE * z[i] * z[j] / d * y / 2.0
    np.testing.assert_allclose(e_pair, ref, rtol=1e-12)


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
    (dict(model_path="so3lr"), None),
    (dict(model_path="so3lr", kspace_electrostatics="pme", kspace_dispersion="pme"), None),
])
def test_engine_choice(monkeypatch, change, reason):
    """What auto picks for a GPU, given the settings (37 A of water)."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    problem = check.flash_problem(dict(SETTINGS, **change), _water((3, 3, 3)))
    assert (problem is None) if reason is None else (reason in problem), problem


def test_engine_choice_small_box(monkeypatch):
    monkeypatch.setattr(check, "device_problem", lambda: None)
    assert "narrower than twice the cutoff" in check.flash_problem(SETTINGS, _water((1, 1, 1)))


@pytest.mark.parametrize("task,change,reason", [
    ("opt", dict(), None),
    ("ase", dict(kspace_electrostatics="pme", kspace_dispersion="pme"), None),
    ("eval", dict(targets="energy,forces"), None),
    ("eval", dict(targets=["forces"]), None),
    ("eval", dict(targets="forces,dipole_vec,hirshfeld_ratios"), "not dipole_vec, hirshfeld_ratios"),
    ("eval", dict(targets="energy", kspace_electrostatics="ewald"), "PME but not ewald"),
    ("opt", dict(lr_cutoff=None), "long-range cutoff"),
    ("opt", dict(model_path="so3lr"), None),
])
def test_engine_choice_other_tasks(monkeypatch, task, change, reason):
    """so3lr opt, so3lr eval and the ASE calculator take what MD takes, with no ensemble;
    eval only for energies and forces."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    problem = check.flash_problem(dict(dict(model_path=None, lr_cutoff=12.0), **change), _water((3, 3, 3)), task=task)
    assert (problem is None) if reason is None else (reason in problem), problem


def test_engine_choice_every_structure(monkeypatch):
    """so3lr eval: flash takes a dataset only if it takes every structure, and names the
    first it cannot take."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    settings = dict(model_path=None, lr_cutoff=12.0, targets="energy,forces")
    water = _water((3, 3, 3))
    slab = water.copy()
    slab.pbc = [True, True, False]
    assert check.flash_problem(settings, [water, water.copy()], task="eval") is None
    problem = check.flash_problem(settings, [water, water, slab], task="eval")
    assert problem.startswith("structure 2: it needs a box periodic"), problem


def test_eval_auto_takes_a_few_engines(monkeypatch):
    """so3lr eval builds an engine per set of atoms: auto declines more than
    EVAL_AUTO_ENGINES of them, engine flash takes any number."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    water = _water((3, 3, 3))
    frames = []
    for k in range(check.EVAL_AUTO_ENGINES + 1):
        a = water.copy()
        a.numbers[3 * k + 1] = 9                # a different hydrogen each time
        frames.append(a)
    settings = dict(model_path=None, lr_cutoff=12.0, targets="energy,forces")
    assert check.flash_problem(settings, frames[:-1], task="eval") is None
    assert "different sets of atoms" in check.flash_problem(settings, frames, task="eval")
    assert check.flash_problem(dict(settings, engine="flash"), frames, task="eval") is None


def test_every_settings_key_classified():
    """A settings key added to the CLI must be sorted into what flash handles, ignores
    or declines; until then auto declines whenever it is set."""
    from so3lr.cli.so3lr_cli import known_yaml_keys
    known = check.HANDLED | check.IRRELEVANT | set(check.UNSUPPORTED)
    assert not known_yaml_keys() - known, sorted(known_yaml_keys() - known)


@pytest.mark.parametrize("model,edit,reason", [
    ("so3lr-tauri-m-2606", None, None),
    ("so3lr-tauri-m-2606", ("num_heads", 8), "degrees, heads and features"),
    ("so3lr-tauri-m-2606", ("degrees", [1, 2, 3, 5]), "degrees, heads and features"),
    ("so3lr-tauri-m-2606", ("activation_fn", "gelu"), "activation_fn"),
    ("so3lr-tauri-m-2606", ("some_new_hyperparameter", True), "does not know the model hyperparameter"),
    ("so3lr-tauri-m-2606", ("qk_norm", None), "does not set 'qk_norm'"),
    ("so3lr-tauri-m-2606", ("legacy_so3lr_bool", False), None),
    # declined (the first mismatch it names follows the file's key order)
    ("so3lr-tauri-m-2606", ("legacy_so3lr_bool", True), ""),
    ("so3lr", None, None),
    ("so3lr", ("qk_norm", False), None),
    ("so3lr", ("use_rms_norm", True), "use_rms_norm = False"),
    ("so3lr", ("layer_normalization_2", False), "layer_normalization_2 = True"),
    # use_final_bias_bool (like legacy_so3lr_bool) is read from the checkpoint when the
    # config leaves it out (so3lr.model_config), so removing it changes nothing here
    ("so3lr", ("use_final_bias_bool", None), None),
    ("so3lr", ("layer_normalization_2", None), "does not set 'layer_normalization_2'"),
    ("so3lr", ("electrostatic_energy_kspace_do_ewald_bool", False), "does not know the model hyperparameter"),
])
def test_model_architecture_check(tmp_path, model, edit, reason):
    """A model with one hyperparameter changed (None: removed) is declined."""
    from so3lr.model_registry import resolve_model
    src = resolve_model(model).dir
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
    assert (problem is None) if reason is None else (problem is not None and reason in problem), problem


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


def _acetone():
    from ase.io import read
    return read(DATA / "acetone.xyz")


@pytest.mark.parametrize("task,change,reason", [
    ("md", dict(), None),
    ("md", dict(lr_cutoff=1000.0), None),
    ("md", dict(ensemble="nvt-langevin"), None),
    ("md", dict(ensemble="nve"), None),
    ("md", dict(ensemble="npt"), "NPT needs a box"),
    ("md", dict(kspace_electrostatics="pme"), "k-space (kspace_electrostatics) needs a box"),
    ("md", dict(kspace_dispersion="pme"), "k-space (kspace_dispersion) needs a box"),
    ("opt", dict(lr_cutoff=1000.0), None),
    ("eval", dict(targets="energy,forces", lr_cutoff=1000.0), None),
    ("ase", dict(model_path="so3lr"), None),
])
def test_engine_choice_in_free_space(monkeypatch, task, change, reason):
    """A structure with no box (acetone) runs on flash at any long-range cutoff, but not in
    NPT or with k-space."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    settings = {"model_path": None, "lr_cutoff": 12.0, **({"ensemble": "nvt"} if task == "md" else {}), **change}
    problem = check.flash_problem(settings, _acetone(), task=task)
    assert (problem is None) if reason is None else (reason in problem), problem


def test_free_space_structures(tmp_path, monkeypatch):
    """No box as perform_md takes it: not periodic, or periodic with no cell; a slab (some
    directions periodic) is neither; a restart with a box cannot continue in free space."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    atoms = _acetone()
    assert check.in_vacuum(atoms)
    atoms.pbc = True
    assert check.in_vacuum(atoms) and check.flash_problem(SETTINGS, atoms) is None
    atoms.pbc = [True, True, False]
    assert not check.in_vacuum(atoms)
    assert "periodic in all three directions, or no box" in check.flash_problem(SETTINGS, atoms)
    np.savez(tmp_path / "box.npz", box=np.array([30.0, 30.0, 30.0]))
    np.savez(tmp_path / "none.npz", box=np.array([0.0]))
    problem = check.flash_problem(dict(SETTINGS, restart_load_path=str(tmp_path / "box.npz")), _acetone())
    assert "the restart has a box" in problem, problem
    assert check.flash_problem(dict(SETTINGS, restart_load_path=str(tmp_path / "none.npz")), _acetone()) is None


def _dimer(dx=0.0):
    """Two acetones 7 A apart, the second moved dx A further out (as binding benchmarks move
    a monomer away for the interaction energy)."""
    a, b = _acetone(), _acetone()
    b.translate([7.0 + dx, 0.0, 0.0])
    return a + b


@pytest.mark.parametrize("dx", [1e4, 1e14])
def test_free_space_too_spread_out(monkeypatch, dx):
    """A monomer moved 1 um or 10 km away: the box free space lays out around the dimer is too
    large for the cell grid (MAX_CELLS), so auto runs the neighbour-list potential and names
    the structure (rather than the engine raising), and engine flash raises."""
    from so3lr.cli import so3lr_md
    monkeypatch.setattr(check, "device_problem", lambda: None)
    lines = []
    monkeypatch.setattr(so3lr_md.logger, "info", lines.append)
    settings = dict(model_path=None, lr_cutoff=1000.0, targets="energy")
    near, far = _dimer(), _dimer(dx)
    assert check.flash_problem(settings, [near, near.copy()], task="eval") is None
    problem = check.flash_problem(settings, [near, far, near.copy()], task="eval")
    assert problem.startswith("structure 1: its atoms span") and "too large for the cell grid" in problem, problem

    def start(flash_md):
        raise AssertionError("flash started")
    assert so3lr_md.run_on_flash(dict(settings, engine="auto"), [near, far], "eval", start) == (False, None)
    assert lines[-1] == f"Engine: neighbour-list potential (flash cannot: {problem})", lines
    with pytest.raises(ValueError, match="--engine flash: structure 1: .* too large for the cell grid"):
        so3lr_md.run_on_flash(dict(settings, engine="flash"), [near, far], "eval", start)
    # a dimer spanning 1501 A still fits (free space's box: 2876 A, 639^3 cells)
    assert check.flash_problem(settings, _dimer(1490.0), task="eval") is None


def test_engine_choice_large_box(monkeypatch):
    """A periodic box wider than the cell grid holds (MAX_CELLS cells at 4.5 A each)."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    atoms = _acetone()
    atoms.pbc, atoms.cell = True, [3000.0] * 3
    assert "too large for the cell grid (666 x 666 x 666 cells" in check.flash_problem(SETTINGS, atoms)
    atoms.cell = [2900.0] * 3
    assert check.flash_problem(SETTINGS, atoms) is None


@pytest.mark.parametrize("rc_lr", [3.0, 12.0, 100.0, 1000.0])
def test_free_space_box_has_no_images_within_reach(rc_lr):
    """The box free space runs in (vacuum_box), for atoms as laid out and grown into all of
    its room: every pair within the cutoff is within the long range's reach (never past the
    cutoff, also one below the model's) and is its own minimum image, and no periodic image
    of an atom is within either cutoff of another; vacuum_fits says when the atoms grew past
    that."""
    rng = np.random.default_rng(0)
    rc = 4.5
    for n, spread in ((30, 2.0), (300, 8.0)):
        R = rng.normal(size=(n, 3)) * spread
        size = check.vacuum_size(R)
        L, reach = check.vacuum_box(size, rc, rc_lr)
        assert check.box_problem([L] * 3, rc, reach) is None
        assert reach <= rc_lr
        c = R.mean(0)
        room = min(L - max(rc, reach), reach if reach < rc_lr else np.inf)      # the size vacuum_fits takes
        for grown in (R, c + (R - c) * 0.999 * room / size):
            assert check.vacuum_fits(check.vacuum_size(grown), L, rc, rc_lr, reach)
            d = grown[None] - grown[:, None]
            r = np.linalg.norm(d, axis=-1)
            assert (r[(r > 0) & (r < rc_lr)] < reach).all()
            near = r < reach
            assert np.allclose(np.asarray(E._min_image(d, L))[near], d[near])
            for k in itertools.product((-1, 0, 1), repeat=3):
                if any(k):
                    assert np.linalg.norm(d + L * np.asarray(k), axis=-1).min() >= max(rc, reach), k
        assert not check.vacuum_fits(check.vacuum_size(c + (R - c) * L / size), L, rc, rc_lr, reach)


@pytest.mark.parametrize("bad", [dict(engine="Flash"), dict(engine="neighbour-list"), dict(flash_mode="Fast"),
                                 dict(flash_mixed="false"), dict(flash_devices=0), dict(flash_devices="2"),
                                 dict(flash_devices=2, flash_mode="lean"), dict(flash_devices=4096)])
def test_bad_engine_options_raise(bad):
    """Settings files bypass the CLI's choices: a value flash does not take is an error."""
    with pytest.raises(ValueError):
        check.validate_options(bad)
    check.validate_options(dict(engine="flash", flash_mode="lean", flash_mixed=False))
    check.validate_options(dict(engine="standard", flash_mode="lean", flash_devices=4096))


def test_rocm_launches(monkeypatch):
    """The launches measured on the MI355X on ROCm, the NVIDIA ones elsewhere."""
    from types import SimpleNamespace

    def rows(n, warps):
        return E.FlashSO3LR._row_grid(SimpleNamespace(nl=n, atoms_per_program=None), warps)
    monkeypatch.setattr(E, "is_rocm", lambda: True)
    assert E._att_launch(32) == {"fwd": (16, 1), "bwd_heads": (16, 1), "bwd_degrees": (8, 1)}
    assert E._att_launch(64) == {"fwd": (16, 2), "bwd_heads": (32, 2), "bwd_degrees": (16, 2)}
    assert E._warps(E.NUM_WARPS) == 1
    assert rows(12_288, 1) == (12_288, 1)
    assert rows(84_283_392, 1) == (42_141_696, 2)
    assert rows(84_283_392, 2) == (28_094_464, 3)
    monkeypatch.setattr(E, "is_rocm", lambda: False)
    assert E._att_launch(32) == E.ATT_LAUNCH[32] and E._att_launch(64) == E.ATT_LAUNCH[64]
    assert E._warps(E.NUM_WARPS) == E.NUM_WARPS
    assert rows(84_283_392, 1) == (84_283_392, 1)


@pytest.mark.parametrize("rocm,cc,flags,reason", [
    (True, "gfx950", "--xla_gpu_enable_command_buffer=", None),
    (True, "gfx950", "", "XLA_FLAGS"),
    (True, "gfx942", "--xla_gpu_enable_command_buffer=", "tested on gfx950"),
    (False, "9.0", "", None),
    (False, "7.5", "", "compute capability 8.0"),
    (False, "gfx950", "", "does not know this GPU"),
])
def test_device_check(monkeypatch, rocm, cc, flags, reason):
    """NVIDIA from compute capability 8.0; on ROCm the MI355X with XLA's command buffers off."""
    from types import SimpleNamespace
    monkeypatch.setattr(check, "is_rocm", lambda: rocm)
    monkeypatch.setattr(jax, "devices", lambda *args: [SimpleNamespace(platform="gpu", compute_capability=cc)])
    monkeypatch.setenv("XLA_FLAGS", flags)
    problem = check.device_problem()
    assert (problem is None) if reason is None else (reason in problem), problem


def test_flash_modes_match_the_engine():
    assert tuple(E.MODES) == check.FLASH_MODES[1:]


@pytest.mark.parametrize("engine,line", [
    ("standard", "Engine: neighbour-list potential (--engine standard)"),
    # 64 waters' 12.4 A box is narrower than twice the cutoff, so auto runs the
    # neighbour-list potential on any machine
    ("auto", "Engine: neighbour-list potential (flash cannot:"),
])
def test_md_logs_its_engine(tmp_path, monkeypatch, engine, line):
    """`so3lr nvt` names the engine in the log, and when auto runs the neighbour-list
    potential it says why (the line comes before perform_md, so no MD runs)."""
    from click.testing import CliRunner
    from so3lr.cli import so3lr_cli, so3lr_md
    monkeypatch.setattr(so3lr_md, "setup_jax_globals", lambda **kwargs: None)
    monkeypatch.setattr(so3lr_md, "perform_md", lambda *args, **kwargs: None)
    # auto would set XLA_PYTHON_CLIENT_MEM_FRACTION in this process, and so in every
    # subprocess the session starts after this test
    monkeypatch.setattr(so3lr_cli, "reserve_gpu", lambda *args, **kwargs: None)
    log = tmp_path / "out.log"
    result = CliRunner().invoke(so3lr_cli.cli, ["nvt", "--input", str(DATA / "water_64.xyz"), "--no-relax",
                                                "--output", str(tmp_path / "out.hdf5"), "--log-file", str(log),
                                                "--engine", engine])
    assert result.exit_code == 0, result.output
    assert line in log.read_text(), log.read_text()[-2000:]


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
                                       ("so3lr-tauri-s-2606", "electrostatics,dispersion"), ("so3lr", ""),
                                       ("so3lr", "electrostatics,dispersion")])
def test_flash_matches_neighbour_lists(tmp_path, model, pme):
    """Energy, forces and dE/d(strain) on 12,288 waters (49.6 A), float64."""
    from ase.io import write
    from so3lr.flash.verify import main
    write(tmp_path / "water.xyz", _water((4, 4, 4)))
    assert main(["--input", str(tmp_path / "water.xyz"), "--model", model, "--precision", "float64",
                 "--lr-cutoff", "10", "--pme", pme])


@GPU
@pytest.mark.parametrize("model", ["so3lr-tauri-m-2606", "so3lr-tauri-l-2606", "so3lr"])
def test_float32_follows_float64(model):
    """float32 (each model's attention launch; the backward's filter products as TF32x3 on
    tensor cores, which float64 never runs; plain float32 on ROCm) gives float64's energy and
    forces to float32 accuracy."""
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
@pytest.mark.parametrize("app", [2, 5])
def test_atoms_per_program(app):
    """Several atoms per program, as on HIP for the largest systems (5 leaves a tail), give one
    atom per program's energy and forces (float64)."""
    from so3lr.flash.loader import load_params
    atoms = _water((3, 3, 3))
    params, statics = load_params(None, precision="float64", lr_cutoff=12.0)
    args = (params, statics, atoms.get_atomic_numbers(), np.asarray(atoms.cell[:]))
    R = jnp.asarray(atoms.get_positions())
    e_ref, f_ref = E.FlashSO3LR(*args).energy_forces(R)
    e, f = E.FlashSO3LR(*args, atoms_per_program=app).energy_forces(R)
    _check("energy", e, e_ref, 1e-13)
    _check("forces", f, f_ref, 1e-12)


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


def _scaled(atoms, f):
    out = atoms.copy()
    out.set_cell(atoms.cell[:] * f, scale_atoms=True)
    return out


@GPU
@pytest.mark.parametrize("pme", [None, "pme"])
def test_ase_calculator_matches_neighbour_lists(pme):
    """FlashCalculator gives So3lrCalculator's energy and forces (float64, 37 A of water,
    with and without PME electrostatics), again after the cell grows (the engine's box
    moves) and after a water is removed (a new engine)."""
    from so3lr import FlashCalculator, So3lrCalculator
    from so3lr.mlff.utils.kspace_utils import DEFAULT_KSPACE_SMEARING, DEFAULT_KSPACE_SPACING
    kspace = dict(kspace_electrostatics=pme, kspace_smearing=DEFAULT_KSPACE_SMEARING,
                  kspace_spacing=DEFAULT_KSPACE_SPACING)
    calc = FlashCalculator(model="so3lr-tauri-m-2606", lr_cutoff=12.0, precision="float64", **kspace)
    flash = _water((3, 3, 3))
    flash.calc = calc
    for step in ("first", "cell", "atoms"):
        if step == "cell":
            flash.set_cell(flash.cell[:] * 1.01, scale_atoms=True)
        elif step == "atoms":
            del flash[[0, 1, 2]]
        other = flash.copy()
        # a new reference each time: So3lrCalculator does not take other atoms after its first
        other.calc = So3lrCalculator(model="so3lr-tauri-m-2606", lr_cutoff=12.0, dtype=np.float64, **kspace)
        e_ref, f_ref = other.get_potential_energy(), other.get_forces()
        e, f = flash.get_potential_energy(), flash.get_forces()
        assert abs(e - e_ref) / len(flash) < 1e-9, (step, e, e_ref)
        assert np.abs(f - f_ref).max() < 1e-5, (step, np.abs(f - f_ref).max())


@GPU
def test_ase_calculator_refuses_what_flash_cannot_take():
    from so3lr import FlashCalculator
    atoms = _water((3, 3, 3))
    atoms.pbc = [True, True, False]
    atoms.calc = FlashCalculator(precision="float64")
    with pytest.raises(ValueError, match="periodic"):
        atoms.get_potential_energy()


@GPU
@pytest.mark.parametrize("pme", [None, "pme"])
def test_eval_on_flash_matches_neighbour_lists(tmp_path, pme):
    """so3lr eval on flash (float64): three labelled structures, the second in a larger box
    (the engine's box moves), the third with a water fewer and a charge of +1 (a new engine;
    the charge embedding, made of parameters alone, in float64 on both engines). The energies are
    So3lrCalculator's in float64 and the force metrics and forces so3lr eval's on the
    neighbour-list model."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read, write
    from so3lr import So3lrCalculator
    from so3lr.cli.so3lr_eval import evaluate_so3lr_on
    from so3lr.mlff.utils.kspace_utils import DEFAULT_KSPACE_SMEARING, DEFAULT_KSPACE_SPACING
    rng = np.random.default_rng(0)
    water = _water((3, 3, 3))
    frames = [water, _scaled(water, 1.01), water[3:]]
    frames[2].info["charge"] = 1
    for a in frames:
        a.calc = SinglePointCalculator(a, energy=float(rng.normal()), forces=rng.normal(size=(len(a), 3)))
    write(tmp_path / "data.extxyz", frames)
    out = {}
    for engine in ("flash", "standard"):
        out[engine] = evaluate_so3lr_on(
            str(tmp_path / "data.extxyz"), targets="energy,forces", precision="float64",
            model_path="so3lr-tauri-m-2606", kspace_electrostatics=pme, engine=engine,
            save_to=str(tmp_path / f"{engine}.extxyz"), log_file=str(tmp_path / f"{engine}.log"))
    assert "Engine: flash" in (tmp_path / "flash.log").read_text()
    # the forces agree to 1e-5 eV/A (verify's bound): a force metric moves by at most the mean
    # |difference|
    for key in ("forces_mae", "forces_rmse"):
        assert abs(out["flash"][key] - out["standard"][key]) <= 1e-6, (key, out["flash"][key], out["standard"][key])
    kspace = dict(kspace_electrostatics=pme, kspace_smearing=DEFAULT_KSPACE_SMEARING,
                  kspace_spacing=DEFAULT_KSPACE_SPACING)
    for a, b in zip(read(tmp_path / "flash.extxyz", ":"), read(tmp_path / "standard.extxyz", ":")):
        ref = a.copy()
        ref.calc = So3lrCalculator(model="so3lr-tauri-m-2606", lr_cutoff=12.0, dtype=np.float64, **kspace)
        assert abs(a.info["energy_so3lr"] - ref.get_potential_energy()) / len(a) < 1e-9
        assert np.abs(a.arrays["forces_so3lr"] - b.arrays["forces_so3lr"]).max() < 1e-5
        assert np.allclose(a.positions, b.positions)


@GPU
def test_eval_falls_back_for_other_targets(tmp_path):
    """The default targets include dipoles and ratios, which flash does not compute: auto
    runs the neighbour-list model and says why."""
    from so3lr.cli.so3lr_eval import evaluate_so3lr_on
    from ase.io import write
    write(tmp_path / "data.extxyz", [_water((3, 3, 3))])
    evaluate_so3lr_on(str(tmp_path / "data.extxyz"), log_file=str(tmp_path / "o.log"))
    log = (tmp_path / "o.log").read_text()
    assert "flash cannot: it evaluates energy and forces, not dipole_vec, hirshfeld_ratios" in log, log[-2000:]


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
@pytest.mark.parametrize("model", [None, "so3lr"])
def test_opt_on_flash_follows_neighbour_lists(tmp_path, model):
    """`so3lr opt` on flash (auto) follows the neighbour-list potential's FIRE trajectory
    (float64, PME; the default model and the legacy so3lr), and writes the same trajectory
    file."""
    from ase.io import read, write
    write(tmp_path / "w.xyz", _water((3, 3, 3)))
    run = ["opt", "--input", "w.xyz", "--precision", "float64", "--min-cycles", "3", "--min-steps", "5",
           "--kspace-electrostatics", "pme"] + ([] if model is None else ["--model", model])
    _so3lr([*run, "--output", "a.xyz"], tmp_path)
    _so3lr([*run, "--engine", "standard", "--output", "b.xyz"], tmp_path)
    assert "Engine: flash (auto)" in (tmp_path / "a.log").read_text()
    a, b = read(tmp_path / "a.xyz", ":"), read(tmp_path / "b.xyz", ":")
    assert len(a) == len(b) == 4
    for fa, fb in zip(a, b):
        d = fa.positions - fb.positions
        d -= np.round(d / np.diag(fa.cell[:])) * np.diag(fa.cell[:])
        assert np.abs(d).max() < 1e-6, np.abs(d).max()


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


def _cluster():
    """1536 waters with no box (43 A across)."""
    from ase.io import read
    atoms = read(DATA / "water_1536.xyz")
    atoms.pbc = False
    atoms.cell = np.zeros((3, 3))
    return atoms


def _moved(atoms, dx=60.0):
    """atoms with the first three (a water) moved dx A out: past the room their box has."""
    out = atoms.copy()
    p = out.get_positions()
    p[:3, 0] += dx
    out.set_positions(p)
    return out


@GPU
@pytest.mark.parametrize("model,lr_cutoff,charge", [
    ("so3lr-tauri-m-2606", 12.0, 0), ("so3lr-tauri-m-2606", 1000.0, 0), ("so3lr-tauri-m-2606", 1000.0, -1),
    ("so3lr", 1000.0, 0)])
def test_free_space_matches_neighbour_lists(model, lr_cutoff, charge):
    """FlashCalculator gives So3lrCalculator's energy and forces (float64) for 1536 waters
    with no box, at the 12 A cutoff and at 1000 A (every pair; the electrostatics past the
    table in closed form), neutral and as OH- (a hydrogen fewer), the default model and the
    legacy so3lr; and again after a water moves 60 A out (the engine's box laid out again)."""
    from so3lr import FlashCalculator, So3lrCalculator
    atoms = _cluster()
    if charge:
        del atoms[-1]
        atoms.info["charge"] = charge
    calc = FlashCalculator(model=model, lr_cutoff=lr_cutoff, precision="float64")
    for step in ("first", "moved"):
        flash = atoms if step == "first" else _moved(atoms)
        flash.calc = calc
        e, f = flash.get_potential_energy(), flash.get_forces()
        other = flash.copy()
        other.calc = So3lrCalculator(model=model, lr_cutoff=lr_cutoff, dtype=np.float64)
        e_ref, f_ref = other.get_potential_energy(), other.get_forces()
        assert abs(e - e_ref) / len(flash) < 1e-9, (step, e, e_ref)
        assert np.abs(f - f_ref).max() < 1e-5, (step, np.abs(f - f_ref).max())
    assert calc.points.engine.L[0] > 100.0          # laid out again for the moved water


@GPU
@pytest.mark.parametrize("lr_cutoff", [12.0, 1000.0])
def test_free_space_lays_out_again(lr_cutoff):
    """An engine laid out for the cluster, given it with a water moved out past its room,
    lays out its box again and gives what an engine laid out for that gives, jitted and
    staged (float64)."""
    from so3lr.flash.loader import load_params
    params, statics = load_params(None, precision="float64", lr_cutoff=lr_cutoff)
    atoms = _cluster()
    z, R0, R1 = atoms.get_atomic_numbers(), atoms.get_positions(), _moved(atoms).get_positions()
    e_ref, f_ref = E.FlashSO3LR(params, statics, z, None, size=check.vacuum_size(R1)).energy_forces(jnp.asarray(R1))
    for staged in (False, True):
        eng = E.FlashSO3LR(params, statics, z, None, size=check.vacuum_size(R0))
        L0 = eng.L[0]
        e, f = (eng.energy_forces_staged if staged else eng.energy_forces)(jnp.asarray(R1))
        assert eng.L[0] > L0
        _check("energy", e, e_ref, 1e-12)
        _check("forces", f, f_ref, 1e-10)


def _peptide():
    from ase.io import read
    atoms = read(DATA / "ac.xyz")
    atoms.calc = None
    return atoms


def _verlet(atoms, calc, steps, dt_fs):
    """Positions of `steps` velocity-Verlet steps (dt_fs) from atoms' positions and momenta,
    on calc's forces: the reference NVE trajectory."""
    from ase import units
    b = atoms.copy()
    b.calc = calc
    m = b.get_masses()[:, None]
    R, P, F = b.get_positions(), atoms.get_momenta(), b.get_forces()
    h = dt_fs * units.fs
    out = []
    for _ in range(steps):
        P = P + 0.5 * h * F
        R = R + h * P / m
        b.set_positions(R)
        F = b.get_forces()
        P = P + 0.5 * h * F
        out.append(R)
    return np.asarray(out)


def _nve_no_box(tmp_path, atoms, cycles, steps, lr_cutoff, *args):
    """`so3lr nve` with no box (float64, 0.5 fs steps) from atoms' positions and momenta (whose
    centre-of-mass momentum perform_md takes out, so it should be zero), on flash (auto):
    (its log, its trajectory's positions, velocity Verlet on So3lrCalculator's forces at the
    same steps). Its trajectory has no box: [0]."""
    import h5py
    from ase.io import write
    from so3lr import So3lrCalculator
    write(tmp_path / "p.xyz", atoms)
    _so3lr(["nve", "--input", "p.xyz", "--no-relax", "--md-cycles", str(cycles), "--md-steps", str(steps),
            "--dt", "0.5", "--precision", "float64", "--lr-cutoff", str(lr_cutoff), "--output", "a.hdf5", *args],
           tmp_path)
    log = (tmp_path / "a.log").read_text()
    assert "Engine: flash (auto)" in log and "no box: computed in a" in log, log[-2000:]
    ref = _verlet(atoms, So3lrCalculator(lr_cutoff=float(lr_cutoff), dtype=np.float64), cycles * steps, 0.5)
    with h5py.File(tmp_path / "a.hdf5") as a:
        assert a["box"].shape[1:] == (1,) and not np.any(a["box"][:])
        return log, a["positions"][:], ref[steps - 1::steps]


@GPU
@pytest.mark.parametrize("lr_cutoff", [12.0, 1000.0])
def test_md_in_free_space_is_velocity_verlet(tmp_path, lr_cutoff):
    """`so3lr nve` with no box on flash (a 42-atom peptide at 300 K, 12 and 1000 A cutoffs)
    is velocity Verlet on So3lrCalculator's forces, step by step, and its restart has
    Cartesian positions and no box. (perform_md's own trajectory parts from it by ~1e-6 A in
    10 steps.)"""
    from ase import units
    atoms = _peptide()
    m = atoms.get_masses()
    p = np.random.default_rng(3).normal(size=(len(atoms), 3)) * np.sqrt(300.0 * units.kB * m)[:, None]
    atoms.set_momenta(p - p.mean(0))
    _, x, ref = _nve_no_box(tmp_path, atoms, 10, 1, lr_cutoff, "--restart-save", "a.npz")
    assert np.abs(x - ref).max() < 1e-7, np.abs(x - ref).max(axis=(1, 2))
    with np.load(tmp_path / "a.npz") as a:
        assert not np.any(a["box"]) and not bool(a["fractional_coordinates"])
        assert np.abs(a["position"] - ref[-1]).max() < 1e-7


@GPU
def test_md_in_free_space_lays_out_again(tmp_path):
    """NVE with no box and a hydrogen leaving at 0.6 A/fs: a step inside the jitted loop
    outgrows the engine's box, the cycle is repeated a step at a time in a box laid out
    again, and the trajectory is velocity Verlet on So3lrCalculator's forces."""
    from ase import units
    atoms = _peptide()
    out = atoms.positions[0] - atoms.positions.mean(0)
    p = np.zeros((len(atoms), 3))
    p[0] = atoms.get_masses()[0] * 0.6 / units.fs * out / np.linalg.norm(out)
    atoms.set_momenta(p - p.mean(0))
    log, x, ref = _nve_no_box(tmp_path, atoms, 3, 20, 12.0)
    assert "cycle repeated" in log, log[-2000:]
    assert np.linalg.norm(x[-1, 0] - x[-1, 1:].mean(0)) > 12.0
    assert np.abs(x - ref).max() < 1e-6, np.abs(x - ref).max(axis=(1, 2))


@GPU
@pytest.mark.parametrize("integrator", ["nve", "fire"])
def test_md_loop_widens_its_tables(integrator):
    """A jitted loop of MD (NVE) or FIRE steps whose neighbour rows overflow runs again a
    step at a time on wider ones, and the jitted loop made after that takes the wider rows
    (jax-md jits its steps, FIRE's through its NVE step: the integrator is made again):
    both give the loop on wide rows (float64, periodic water)."""
    import jax_md
    from so3lr.flash import md as flash_md
    from so3lr.flash.loader import load_params
    atoms = _water((3, 3, 3))
    params, statics = load_params(None, precision="float64", lr_cutoff=12.0)
    args = (params, statics, atoms.get_atomic_numbers(), np.asarray(atoms.cell[:]))
    R, mass = jnp.asarray(atoms.get_positions()), jnp.asarray(atoms.get_masses())
    _, shift = jax_md.space.periodic(jnp.asarray(np.diag(atoms.cell[:])))
    F0 = E.FlashSO3LR(*args).energy_forces(R)[1]
    out = []
    for kw in (dict(), dict(max_neighbours=16)):
        eng = E.FlashSO3LR(*args, **kw)

        def make(eng=eng):
            if integrator == "nve":
                return jax_md.simulate.nve(flash_md._force_fn(eng), shift, 1e-2)
            return jax_md.minimize.fire_descent(flash_md._force_fn(eng), shift, dt_start=0.05, dt_max=0.1)
        init, step = make()
        predict, finish = flash_md._two_call_step(step)
        state = (init(jax.random.PRNGKey(0), R, kT=0.025, mass=mass, F=F0) if integrator == "nve"
                 else init(R, F=F0))
        state = flash_md._run_jitted(eng, make, state, 5, predict, finish)
        out.append(flash_md._run_jitted(eng, make, state, 5, predict, finish))
        assert flash_md._finite(eng.loops[make, 5](state))        # the jitted loop, on the rows it has now
    assert eng.K > 16
    _check("positions", out[1].position, out[0].position, 1e-12)
    _check("forces", out[1].force, out[0].force, 1e-10)


@GPU
def test_opt_in_free_space_follows_neighbour_lists(tmp_path):
    """`so3lr opt` with no box on flash (auto, 1000 A cutoff) follows the neighbour-list
    potential's FIRE trajectory (float64)."""
    from ase.io import read, write
    write(tmp_path / "p.xyz", _peptide())
    run = ["opt", "--input", "p.xyz", "--precision", "float64", "--min-cycles", "3", "--min-steps", "5",
           "--lr-cutoff", "1000"]
    _so3lr([*run, "--output", "a.xyz"], tmp_path)
    _so3lr([*run, "--engine", "standard", "--output", "b.xyz"], tmp_path)
    assert "Engine: flash (auto)" in (tmp_path / "a.log").read_text()
    a, b = read(tmp_path / "a.xyz", ":"), read(tmp_path / "b.xyz", ":")
    assert len(a) == len(b) == 4
    for fa, fb in zip(a, b):
        assert not fa.pbc.any() and not fb.pbc.any()
        assert np.abs(fa.positions - fb.positions).max() < 1e-6, np.abs(fa.positions - fb.positions).max()


@GPU
def test_eval_in_free_space_matches_neighbour_lists(tmp_path):
    """so3lr eval on flash with no box (float64, 1000 A): the peptide, a rattled copy (the
    same engine) and a cation (+1, a doublet: an engine of its own) give so3lr eval's force
    metrics and forces on the neighbour-list model, and So3lrCalculator's energies (as in
    test_eval_on_flash_matches_neighbour_lists)."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read, write
    from so3lr import So3lrCalculator
    from so3lr.cli.so3lr_eval import evaluate_so3lr_on
    rng = np.random.default_rng(0)
    frames = [_peptide(), _peptide(), _peptide()]
    frames[1].rattle(0.05, seed=1)
    frames[2].info.update(charge=1, multiplicity=2)
    for a in frames:
        a.calc = SinglePointCalculator(a, energy=float(rng.normal()), forces=rng.normal(size=(len(a), 3)))
    write(tmp_path / "data.extxyz", frames)
    out = {}
    for engine in ("flash", "standard"):
        out[engine] = evaluate_so3lr_on(
            str(tmp_path / "data.extxyz"), targets="energy,forces", precision="float64", lr_cutoff=1000.0,
            model_path="so3lr-tauri-m-2606", engine=engine, save_to=str(tmp_path / f"{engine}.extxyz"),
            log_file=str(tmp_path / f"{engine}.log"))
    assert "Engine: flash" in (tmp_path / "flash.log").read_text()
    for key in ("forces_mae", "forces_rmse"):
        assert abs(out["flash"][key] - out["standard"][key]) <= 1e-6, (key, out["flash"][key], out["standard"][key])
    for frame, a, b in zip(frames, read(tmp_path / "flash.extxyz", ":"), read(tmp_path / "standard.extxyz", ":")):
        ref = a.copy()
        ref.info = dict(frame.info)                  # the doublet's multiplicity too
        ref.calc = So3lrCalculator(model="so3lr-tauri-m-2606", lr_cutoff=1000.0, dtype=np.float64)
        assert abs(a.info["energy_so3lr"] - ref.get_potential_energy()) / len(a) < 1e-9
        assert np.abs(a.arrays["forces_so3lr"] - b.arrays["forces_so3lr"]).max() < 1e-5


@GPU
def test_eval_falls_back_for_a_monomer_moved_away(tmp_path):
    """so3lr eval (auto, 1000 A) on a dimer and on the dimer with a monomer moved 1 um away, as
    binding benchmarks lay them out: free space's box around the second is too large for the
    cell grid, so auto runs the neighbour-list potential and says why (flash raised here)."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read, write
    from so3lr.cli.so3lr_eval import evaluate_so3lr_on
    frames = [_dimer(), _dimer(1e4)]
    for a in frames:
        a.calc = SinglePointCalculator(a, energy=0.0, forces=np.zeros((len(a), 3)))
    write(tmp_path / "data.extxyz", frames)
    evaluate_so3lr_on(str(tmp_path / "data.extxyz"), targets="energy", lr_cutoff=1000.0,
                      model_path="so3lr-tauri-m-2606", save_to=str(tmp_path / "out.extxyz"),
                      log_file=str(tmp_path / "o.log"))
    log = (tmp_path / "o.log").read_text()
    assert "flash cannot: structure 1: its atoms span" in log, log[-2000:]
    energies = [a.info["energy_so3lr"] for a in read(tmp_path / "out.extxyz", ":")]
    assert len(energies) == 2 and np.isfinite(energies).all(), energies


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
def test_ase_calculator_on_two_gpus():
    """FlashCalculator(flash_devices=2) gives one GPU's energy and forces (float64)."""
    from so3lr import FlashCalculator
    out = []
    for devices in (1, 2):
        atoms = _water((3, 3, 3))
        atoms.calc = FlashCalculator(precision="float64", flash_devices=devices)
        out.append((atoms.get_potential_energy(), atoms.get_forces()))
    _check("energy on 2 GPUs", out[1][0], out[0][0], 1e-12)
    _check("forces on 2 GPUs", out[1][1], out[0][1], 1e-10)


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


def _domain_inputs(model=None, precision="float64"):
    """8,000 waters (24,000 atoms, 62 A) with a 10 A long-range cutoff: (args of
    FlashSO3LR, positions)."""
    from so3lr.flash.loader import load_params
    atoms = _water((5, 5, 5))
    params, statics = load_params(model, precision=precision, lr_cutoff=10.0)
    args = (params, statics, atoms.get_atomic_numbers(), np.asarray(atoms.cell[:]))
    return args, jnp.asarray(atoms.get_positions())


@GPU
@pytest.mark.parametrize("model", [None, "so3lr"])
def test_domains_match_one_call(model):
    """A domain at a time (domains.py) gives one call's energy and forces in float64, in
    2x2x2 and 3x2x1 domains; no PME with domains."""
    from so3lr.flash.domains import FlashDomains
    args, R = _domain_inputs(model)
    e_ref, f_ref = E.FlashSO3LR(*args).energy_forces(R)
    for splits in ((2, 2, 2), (3, 2, 1)):
        e, f = FlashDomains(*args, splits=splits).energy_forces(R)
        _check(f"energy, {splits}", e, e_ref, 1e-12)
        _check(f"forces, {splits}", f, f_ref, 1e-10)
    with pytest.raises(NotImplementedError):
        FlashDomains(*args, splits=(2, 2, 2), pme_electrostatics=(2.0, 1.55))
    with pytest.raises(ValueError, match="narrower than the halo"):
        FlashDomains(*args, splits=(8, 1, 1))
    with pytest.raises(ValueError, match="at least one above 1"):
        FlashDomains(*args, splits=(1, 1, 1))


@GPU
def test_domains_match_one_call_in_float32():
    """In float32 too, at the matmul precision FlashDomains sets (with TF32 the energy
    would depend on the split)."""
    from so3lr.flash.domains import FlashDomains
    args, R = _domain_inputs(precision="float32")
    with jax.default_matmul_precision("highest"):
        e_ref, f_ref = E.FlashSO3LR(*args).energy_forces(R)
    e, f = FlashDomains(*args, splits=(2, 2, 2)).energy_forces(R)
    _check("energy", e, e_ref, 1e-6)
    _check("forces", f, f_ref, 1e-4)


@GPUS
@pytest.mark.parametrize("precision,tol_e,tol_f", [("float64", 1e-12, 1e-10), ("float32", 1e-6, 1e-4)])
def test_domains_on_several_gpus(precision, tol_e, tol_f):
    """Domains dealt over two GPUs give one call's energy and forces, 3x2x1 (in float32
    at the matmul precision FlashDomains sets, also in its worker threads)."""
    from so3lr.flash.domains import FlashDomains
    args, R = _domain_inputs(precision=precision)
    with jax.default_matmul_precision("highest"):
        e_ref, f_ref = E.FlashSO3LR(*args).energy_forces(R)
    e, f = FlashDomains(*args, splits=(3, 2, 1), devices=2).energy_forces(R)
    _check("energy", e, e_ref, tol_e)
    _check("forces", f, f_ref, tol_f)


def test_pick_splits_takes_only_allowed_splits():
    """pick_splits returns (1, 1, 1) when the system fits and otherwise only splits
    FlashDomains accepts (domains at least the larger halo wide, not overlapping
    themselves), or raises."""
    from so3lr.flash.domains import pick_splits
    statics = dict(num_layers=3, cutoff=4.5, long_range_cutoff=12.0, num_features=128, degrees=[1, 2, 3, 4])
    assert pick_splits(10 ** 5, np.diag([100.0] * 3), statics, 64 << 30) == (1, 1, 1)
    k = pick_splits(10 ** 9, np.diag([2160.0] * 3), statics, 64 << 30)[0]
    assert k > 1 and 2160.0 / k >= 12.1 and 2160.0 / k + 2 * 12.1 + 0.1 <= 2160.0
    with pytest.raises(ValueError, match="no domain split fits"):
        pick_splits(10 ** 9, np.diag([40.0] * 3), statics, 4 << 30)
