"""SO3LR-Flash (``so3lr/flash``): the parts that run without a GPU here, and on a GPU it runs on
(NVIDIA, AMD Instinct) the engine against the neighbour-list potential.

Without a GPU: PME's mesh part and its strain derivative against jaxpme, the real-space
tables against so3lr's pair functions, the reasons the engine choice gives (every
settings key and model hyperparameter classified), the box free space runs in,
``so3lr nvt`` logging the engine it picked and, falling back to the neighbour-list
potential, why, and the MD driver around a stand-in for the engine (PySAGES runs as
perform_md's, the repeat of a cycle that overflowed). With one: ``python -m
so3lr.flash.verify`` on 12,288 waters (energy, forces, dE/d(strain), with and without
PME); ``so3lr opt``, ``so3lr eval``, MD and the ASE calculator against the neighbour-list
potential, periodic and with no box (12 and 1000 A cutoffs); PySAGES on flash against the
neighbour-list engine. With two or more: the same on several GPUs as on one, and MD on two.
"""
import functools
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
        # box and positions scaled by s together, as jax-md's NPT strain (s per axis or one)
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
    # per axis: side a and positions along a scaled by s_a (semi-isotropic NPT); the sum is
    # the isotropic strain derivative
    w_ref = jax.grad(lambda s: e_ref(Rj, qj, s).sum())(jnp.ones(3))
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
    w_ref = jax.grad(lambda s: e_ref(Rj, qj, s).sum())(jnp.ones(3))
    e, g, f, w = E.pme_dispersion(tuple(Rj[:, a] for a in range(3)), qj, list(L), mesh, smearing, Bohr,
                                  Hartree, jnp.float64, 70)
    _check("energy", e, e_ref(Rj, qj))
    _check("dE/dqd", g, gq)
    _check("forces", jnp.stack(f, -1), -gR)
    _check("dE/dstrain", w, w_ref, 1e-9)


@pytest.mark.parametrize("smearing", [None, 2.0])
def test_electrostatic_table(smearing):
    """The tabulated pair function against so3lr's, with and without PME."""
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
    form the long-range kernel takes (_far_field), give so3lr's pair function and its
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
    """The legacy so3lr's own pieces against the current so3lr modules (float64): LayerNorm, the
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
    (dict(ensemble="npt", npt_coupling="semi-isotropic"), None),
    (dict(ensemble="npt", npt_coupling="semi_isotropic"), None),
    (dict(ensemble="npt", npt_coupling="anisotropic"), "not anisotropic"),
    (dict(ensemble="npt", dispersion_tail_correction=True), "tail correction"),
    (dict(kspace_electrostatics="pme"), None),
    (dict(kspace_electrostatics="ewald"), "PME but not ewald"),
    (dict(kspace_dispersion="pme", kspace_dispersion_interp_nodes=6), "4 nodes"),
    (dict(observables=["dipole"]), "observables"),
    (dict(observables=[]), None),
    (dict(fixed_atom_indices=[0, 1]), "relax_before_run"),
    (dict(fixed_atom_indices=[0, 1], relax_before_run=False), None),
    (dict(fixed_atom_indices=[0, 1], relax_before_run=False, fixed_atom_constraint="Cartesian-Anchor"), None),
    (dict(fixed_atom_indices=[0, 1], relax_before_run=False, fixed_atom_constraint="cartesain"),
     "fixed_atom_constraint"),
    (dict(ensemble="npt", fixed_atom_indices=[0, 1], relax_before_run=False), "NPT box moves them"),
    (dict(mobile_atom_indices=[0, 1]), "mobile_atom_indices"),
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
    (dict(model_path="so3lr"), None),
    (dict(model_path="so3lr", kspace_electrostatics="pme", kspace_dispersion="pme"), None),
])
def test_engine_choice(monkeypatch, change, reason):
    """What auto picks for a GPU, given the settings (37 A of water)."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    problem = check.flash_problem(dict(SETTINGS, **change), _water((3, 3, 3)))
    assert (problem is None) if reason is None else (reason in problem), problem


def test_fixed_atoms_after_a_restart():
    """A restart skips the relaxation before MD, so flash then takes fixed atoms with the default relax_before_run."""
    assert check.settings_problem(dict(SETTINGS, fixed_atom_indices=[0, 1], restart_load_path="r.npz")) is None


def test_engine_choice_small_box(monkeypatch):
    """A box under 2 x the cutoffs runs on flash (over the periodic images); one a
    long-range cutoff far beyond it would make the images' walk too long for does not."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    tiny = _water((1, 1, 1))[:3]
    tiny.set_cell(np.eye(3) * 4.0)
    assert check.flash_problem(SETTINGS, tiny) is None
    assert "too small for its cutoffs" in check.flash_problem(dict(SETTINGS, lr_cutoff=1000.0), _water((1, 1, 1)))


@pytest.mark.parametrize("task,change,reason", [
    ("opt", dict(), None),
    ("ase", dict(kspace_electrostatics="pme", kspace_dispersion="pme"), None),
    ("eval", dict(targets="energy,forces"), None),
    ("eval", dict(targets=["forces"]), None),
    ("eval", dict(targets="forces,dipole_vec,hirshfeld_ratios"), "not dipole_vec, hirshfeld_ratios"),
    ("eval", dict(targets="energy", kspace_electrostatics="ewald"), "PME but not ewald"),
    ("opt", dict(lr_cutoff=None), "long-range cutoff"),
    ("opt", dict(model_path="so3lr"), None),
    ("opt", dict(fixed_atom_indices=[0]), "only in MD"),
    # enhanced sampling is an MD setting: the other tasks decline it whatever the MD rules
    ("opt", dict(do_enhanced_sampling=True, ensemble="nvt"), "it has no enhanced sampling"),
    ("opt", dict(do_enhanced_sampling=True, ensemble="npt"), "it has no enhanced sampling"),
    ("eval", dict(targets="energy", do_enhanced_sampling=True), "it has no enhanced sampling"),
    ("opt", dict(do_enhanced_sampling=False, pysages_input="p.yaml"), None),
])
def test_engine_choice_other_tasks(monkeypatch, task, change, reason):
    """so3lr opt, so3lr eval and the ASE calculator take what MD takes, with no ensemble;
    eval only for energies and forces; none of them enhanced sampling."""
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
    """Every cubic box from the smallest the minimum image's grid takes on is accepted
    there (cells rc wide where they leave enough per side, wider ones where not); a large
    box keeps rc-wide cells. Below it the periodic images' grid takes every box, the short
    range over the images too under 2 rc, and only where the minimum image's has none."""
    for rc, lr, smallest in [(4.5, 10.0, 25.0), (4.5, 12.0, 30.0), (5.0, 12.0, 30.0)]:
        assert check.box_problem([smallest - 0.1] * 3, rc, lr, images=False) is not None
        for L in np.arange(smallest, 60.0, 0.1):
            assert check.box_problem([L] * 3, rc, lr, images=False) is None, (rc, lr, L)
            assert check.layout([L] * 3, rc, lr)[2:] == (False, 0)
        for L in np.arange(1.0, smallest, 0.1):
            assert check.box_problem([L] * 3, rc, lr) is None, (rc, lr, L)
            nc, kc, images, ks = check.layout([L] * 3, rc, lr)
            w = L / nc[0]
            assert images and kc * w >= lr and (ks * w >= rc if ks else min(nc) >= 2), (rc, lr, L, nc, kc, ks)
            assert (ks > 0) == (L < 2 * rc), (rc, lr, L, ks)
    assert check.cell_grid([49.6] * 3, 4.5, 12.0) == ([11, 11, 11], 3)
    assert check.layout([12.4] * 3, 4.5, 12.0) == ([2, 2, 2], 2, True, 0)
    # a slab: the thin side alone takes the images
    assert check.layout([49.6, 49.6, 12.4], 4.5, 12.0) == ([11, 11, 2], 3, True, 0)
    assert check.layout([3.567, 3.567, 7.134], 4.5, 12.0) == ([1, 1, 1], 4, True, 2)
    # the images' walk is bounded: a 1000 A cutoff needs a box of the minimum image
    assert check.box_problem([12.4] * 3, 4.5, 1000.0) is not None


def test_restart_box_is_checked(tmp_path, monkeypatch):
    """A restart starts in its checkpoint's box, so that is the box auto checks."""
    monkeypatch.setattr(check, "device_problem", lambda: None)
    atoms = _water((3, 3, 3))
    np.savez(tmp_path / "small.npz", box=np.array([0.5, 0.5, 0.5]))
    np.savez(tmp_path / "same.npz", box=np.diag(atoms.cell[:]).copy())
    problem = check.flash_problem(dict(SETTINGS, restart_load_path=str(tmp_path / "small.npz")), atoms)
    assert "too small for its cutoffs" in problem, problem
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
    ("md", dict(do_enhanced_sampling=True), "its enhanced sampling needs a box"),
    ("opt", dict(lr_cutoff=1000.0), None),
    ("eval", dict(targets="energy,forces", lr_cutoff=1000.0), None),
    ("ase", dict(model_path="so3lr"), None),
])
def test_engine_choice_in_free_space(monkeypatch, task, change, reason):
    """A structure with no box (acetone) runs on flash at any long-range cutoff, but not in
    NPT, with k-space or with enhanced sampling."""
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


def test_free_space_too_spread_out(monkeypatch):
    """A monomer moved 1 um away: the box free space lays out around the dimer is too large
    for the cell grid (MAX_CELLS), so auto runs the neighbour-list potential and names the
    structure (rather than the engine raising), and engine flash raises."""
    from so3lr.cli import so3lr_md
    monkeypatch.setattr(check, "device_problem", lambda: None)
    lines = []
    monkeypatch.setattr(so3lr_md.logger, "info", lines.append)
    settings = dict(model_path=None, lr_cutoff=1000.0, targets="energy")
    near, far = _dimer(), _dimer(1e4)
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


def test_free_space_not_finite_or_empty(monkeypatch):
    """A free-space structure with a position not finite, or with no atoms: flash declines it,
    rather than the check raising when it measures the atoms."""
    from ase import Atoms
    monkeypatch.setattr(check, "device_problem", lambda: None)
    bad = _acetone()
    bad.positions[0, 0] = np.nan
    assert check.flash_problem(SETTINGS, [_acetone(), bad]) == "structure 1: its positions are not all finite"
    assert check.flash_problem(SETTINGS, Atoms()) == "it has no atoms"


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
        assert check.box_problem([L] * 3, rc, reach, images=False) is None
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
                                 dict(flash_kernels="CUDA"), dict(flash_devices=0), dict(flash_devices="2"),
                                 dict(flash_devices=2, flash_mode="lean"), dict(flash_devices=4096)])
def test_bad_engine_options_raise(bad):
    """Settings files bypass the CLI's choices: a value flash does not take is an error."""
    with pytest.raises(ValueError):
        check.validate_options(bad)
    check.validate_options(dict(engine="flash", flash_mode="lean"))
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
    # a 1000 A long-range cutoff makes 64 waters' 12.4 A box's walk over the periodic
    # images too long for flash, so auto runs the neighbour-list potential on any machine
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
                                                "--lr-cutoff", "1000", "--engine", engine])
    assert result.exit_code == 0, result.output
    assert line in log.read_text(), log.read_text()[-2000:]


def test_flash_refuses_untrained_elements(monkeypatch):
    """Flash runs the neighbour-list engine's element check before anything else (so no
    GPU is needed here): Rn is past what so3lr-tauri-m was trained on, an error rather than a
    value gathered from another element's row of the reference tables."""
    from so3lr.cli import so3lr_md
    from so3lr.flash import md as flash_md
    monkeypatch.setattr(so3lr_md, "setup_jax_globals", lambda **kwargs: None)  # via setup_run
    atoms = _water((3, 3, 3))
    atoms.numbers[0] = 86
    with pytest.raises(ValueError, match="Z=86"):
        flash_md.perform_md_flash(SETTINGS, atoms=atoms)


class _StandInEngine:
    """FlashSO3LR as perform_md_flash uses it, around the neighbour-list potential set up as
    perform_md sets it up for PySAGES (Cartesian positions, dense lists; positions and pair
    vectors in the MD state's precision, the model in the model's: run_dtypes), so the flash
    MD driver runs without a GPU. A table overflow, until the engine has been widened
    ``overflows`` times: the force traced into a jitted loop is NaN, and once a loop has
    overflowed, the engine's own call (energy_forces, the step-at-a-time path) widens the
    tables first, as FlashSO3LR rebuilds them after an overflow."""
    P, device_mesh, K, maxh, vacuum = 1, None, 0, 0, False

    def __init__(self, atoms, settings, overflows=0):
        from functools import partial
        from so3lr.cli import so3lr_md as M
        model_dtype, dtype = M.run_dtypes(settings)
        geo = M.atoms_to_jnp(atoms, precision=dtype)
        R, box, disp, _, _ = M.handle_box("periodic_cart", geo["positions"], np.asarray(atoms.cell[:]),
                                          precision=dtype)
        pot = M.load_model(settings["model_path"], precision=model_dtype, lr_cutoff=settings["lr_cutoff"])
        nfn, nfn_lr, efn = M.process_model(pot, geo["species"], disp, box, precision=model_dtype,
                                           buffer_size_multiplier_sr=settings["buffer_size_multiplier_sr"],
                                           buffer_size_multiplier_lr=settings["buffer_size_multiplier_lr"])
        nb, nb_lr = nfn.allocate(R, box=box), nfn_lr.allocate(R, box=box)
        energy = partial(efn, k_smearing=None, k_smearing_disp=None, has_aux=False)

        def ef(R):
            n = nb.update(R, neighbor=nb.idx, box=box)
            n_lr = nb_lr.update(R, neighbor=nb_lr.idx, box=box)
            e, g = jax.value_and_grad(energy)(R, neighbor=n.idx, neighbor_lr=n_lr.idx, box=box,
                                              k_grid=None, k_grid_disp=None)
            return e, -g
        self._ef, self._jit = ef, jax.jit(ef)
        self.rc, self.loops, self.overflows, self._overflowed = float(pot.cutoff), {}, overflows, False

    def _energy_forces(self, R):
        e, f = self._ef(R)
        # traced when a jitted loop compiles: that loop overflows
        self._overflowed |= self.overflows > 0
        return e, f, jnp.asarray(self.overflows > 0), jnp.zeros((), R.dtype)

    def widen(self):
        self.overflows -= 1
        self._overflowed = False
        self.loops = {}

    def energy_forces_strain(self, R, diag=False):
        if self._overflowed and self.overflows > 0:
            self.widen()
        return (*self._jit(R), jnp.zeros((), R.dtype))

    def energy_forces(self, R):
        return self.energy_forces_strain(R)[:2]


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
             model_path="so3lr-tauri-m-2606", lr_cutoff=12.0, relax_before_run=False,
             md_cycles=2, md_steps=5, save_buffer=1, do_enhanced_sampling=True,
             pysages_input=str(d / "pysages.yaml"), restart_save_path=str(d / "r.npz"),
             restart_pysages_save_path=str(d / "r.pkl"))
    s.update(change)
    monkeypatch.chdir(d)
    if engine == "standard":
        so3lr_md.perform_md(s, restart=s.get("restart_load_path") is not None)
        return d
    monkeypatch.setattr(flash_md, "flash_problem", lambda settings, atoms, task="md": None)
    monkeypatch.setattr(flash_md, "build_engine",
                        lambda atoms, settings, dtype, **kw: (_StandInEngine(atoms, settings, overflows), staged))
    if engine == "run":
        monkeypatch.setattr(check, "flash_problem", lambda settings, atoms, task="md": None)
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
    accepted state and PySAGES result, with PySAGES' loop compiled again; the discarded
    attempt's CV rows go to cv_vs_timestep_discarded.txt."""
    pytest.importorskip("pysages")
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard", "standard", monkeypatch))
    for overflows in (0, 1):
        d = _pysages_run(tmp_path, f"flash{overflows}", "flash", monkeypatch, overflows)
        out = _pysages_outputs(d)
        log = (d / "out.log").read_text()
        assert ("cycle repeated" in log) == bool(overflows)
        (traj, state, abf, cv), (traj_ref, state_ref, abf_ref, cv_ref) = out, ref
        _check("CVs", cv, cv_ref, 1e-9)
        discarded = d / "cv_vs_timestep_discarded.txt"
        assert discarded.exists() == bool(overflows)
        if overflows:
            assert discarded.read_text().startswith(
                "# 5 steps from step 0, attempt 1: discarded (non-finite forces)\n")
            assert np.loadtxt(discarded).shape == (5, cv_ref.shape[1])
        for k in ("positions", "momenta"):
            _check(k, traj[k], traj_ref[k], 1e-9)
        for k in ("position", "momentum", "force"):
            _check(f"final {k}", state[k], state_ref[k], 1e-9)
        assert not state["fractional_coordinates"] and not state_ref["fractional_coordinates"]
        np.testing.assert_array_equal(np.asarray(abf.hist), np.asarray(abf_ref.hist))
        for k in ("Fsum", "force", "bias"):
            _check(f"ABF {k}", getattr(abf, k), getattr(abf_ref, k), 1e-8)


def test_perform_md_repeats_an_overflowed_pysages_cycle(tmp_path, monkeypatch):
    """A neighbour-list overflow in perform_md's first PySAGES cycle repeats it from the last
    accepted state and PySAGES result: the same run as without one. The discarded attempt's
    CV rows go to cv_vs_timestep_discarded.txt, so cv_vs_timestep.txt keeps one row per step."""
    pytest.importorskip("pysages")
    from so3lr.cli import so3lr_md
    ref = _pysages_outputs(_pysages_run(tmp_path, "ref", "standard", monkeypatch))
    real, calls = so3lr_md.check_overflow, []

    def overflow_once(*args, **kwargs):
        nbrs, nbrs_lr, sr, lr = real(*args, **kwargs)
        calls.append(sr)
        return nbrs, nbrs_lr, sr or len(calls) == 1, lr

    monkeypatch.setattr(so3lr_md, "check_overflow", overflow_once)
    d = _pysages_run(tmp_path, "repeat", "standard", monkeypatch)
    assert len(calls) == 3
    _same_run(_pysages_outputs(d), ref, 1e-9)
    discarded = d / "cv_vs_timestep_discarded.txt"
    assert discarded.read_text().startswith(
        "# 5 steps from step 0, attempt 1: discarded (neighbour lists overflowed)\n")
    _check("discarded CVs", np.loadtxt(discarded), ref[3][:5], 1e-9)


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


def test_flash_runs_pysages_in_float32_and_mixed(tmp_path, monkeypatch):
    """float32: flash gives perform_md's run. mixed (a float64 MD state around a float32 model,
    the default precision): both engines continue a float32 PySAGES restart of perform_md
    the same way, the restart carried into the float64 state; perform_md continues flash's
    mixed PySAGES restart in float32."""
    pytest.importorskip("pysages")
    f32 = dict(precision="float32")
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard", "standard", monkeypatch, **f32))
    out = _pysages_outputs(_pysages_run(tmp_path, "flash", "flash", monkeypatch, **f32))
    assert out[1]["momentum"].dtype == ref[1]["momentum"].dtype == np.float32
    _same_run(out, ref, 1e-4)
    first = _pysages_run(tmp_path, "first", "standard", monkeypatch, md_cycles=1, **f32)
    restart = dict(restart_load_path=str(first / "r.npz"), restart_pysages_load_path=str(first / "r.pkl"),
                   precision="mixed")
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard_mixed", "standard", monkeypatch, **restart))
    mixed = _pysages_run(tmp_path, "flash_mixed", "flash", monkeypatch, **restart)
    out = _pysages_outputs(mixed)
    assert out[1]["momentum"].dtype == ref[1]["momentum"].dtype == np.float64
    assert int(np.asarray(out[2].hist).sum()) == int(np.asarray(ref[2].hist).sum()) == 15
    _same_run(out, ref, 1e-6)
    back = dict(restart_load_path=str(mixed / "r.npz"), restart_pysages_load_path=str(mixed / "r.pkl"), **f32)
    _, state, method_state, cv = _pysages_outputs(_pysages_run(tmp_path, "back", "standard", monkeypatch, **back))
    assert state["momentum"].dtype == np.float32
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


def test_flash_runs_chargeseparation_charge_sites_as_perform_md(tmp_path, monkeypatch):
    """chargeseparation with charge_sites: both engines take the neutral H counts from the
    input structure (same CVs); a PySAGES restart whose pysages_input differs from the
    restart file's method (here charge_sites) warns in both engines."""
    pytest.importorskip("pysages")
    cv = {"type": "chargeseparation", "reference": [4, 7], "charge_sites": "all",
          "grid_min": -1.0, "grid_max": 1.0, "grid_stride": 20, "periodic": False}
    kw = dict(pysages={"method": "unbiased", "cv": [cv]})
    ref = _pysages_run(tmp_path, "standard", "standard", monkeypatch, **kw)
    out = _pysages_run(tmp_path, "flash", "flash", monkeypatch, **kw)
    for d in (ref, out):
        assert "neutral H counts from the input structure" in (d / "out.log").read_text()
    _check("CVs", _pysages_outputs(out)[3], _pysages_outputs(ref)[3], 1e-9)
    restart = dict(restart_load_path=str(ref / "r.npz"), restart_pysages_load_path=str(ref / "r.pkl"),
                   pysages={"method": "unbiased", "cv": [dict(cv, charge_sites="water")]})
    for engine in ("standard", "flash"):
        log = (_pysages_run(tmp_path, f"restart_{engine}", engine, monkeypatch, **restart) / "out.log").read_text()
        assert "PySAGES restart: cv[0].charge_sites in pysages_input is None" in log


def test_flash_saves_pysages_state_without_an_md_checkpoint(tmp_path, monkeypatch):
    """restart_pysages_save_path without restart_save_path: flash saves the PySAGES state
    alone, as perform_md does."""
    pytest.importorskip("pysages")
    ref = _pysages_outputs(_pysages_run(tmp_path, "standard", "standard", monkeypatch, restart_save_path=None))
    out = _pysages_outputs(_pysages_run(tmp_path, "flash", "flash", monkeypatch, restart_save_path=None))
    assert out[1] is None and ref[1] is None
    assert int(np.asarray(out[2].hist).sum()) == 10
    _same_run(out, ref, 1e-9)


def test_flash_repeats_an_overflowed_jitted_cycle_a_step_at_a_time(tmp_path, monkeypatch):
    """A table overflow in the jitted mode, without PySAGES (on a CPU, around the stand-in;
    test_md_loop_widens_its_tables does it on a GPU): the first cycle is repeated a step at a
    time, where the engine widens its tables, the next cycle runs in a jitted loop made
    again on the wider tables (jax-md's step is jitted, and jit keeps the trace of the old
    ones), and the run is the run without an overflow."""
    import h5py
    out = []
    for overflows in (0, 1):
        d = _pysages_run(tmp_path, f"plain{overflows}", "flash", monkeypatch, overflows, md_cycles=3,
                         do_enhanced_sampling=False, restart_pysages_save_path=None)
        # once: the loop made after the widening does not overflow again
        assert (d / "out.log").read_text().count("cycle repeated a step at a time") == overflows
        with h5py.File(d / "out.hdf5") as f:
            out.append({k: f[k][:] for k in ("positions", "momenta")})
    for k in ("positions", "momenta"):
        _check(k, out[1][k], out[0][k], 1e-12)


def test_flash_declines_pysages_with_no_box(monkeypatch):
    """Free space: flash runs MD with no box, but not PySAGES (a cycle runs inside PySAGES'
    loop, where the box laid out for the atoms cannot be laid out again, and the
    coordinationnumber CV's list needs a periodic box): auto runs the neighbour-list engine,
    as before flash took free space."""
    from so3lr.cli import so3lr_md
    monkeypatch.setattr(check, "device_problem", lambda: None)
    atoms = _water((1, 1, 1))
    atoms.pbc = False                                # no box, as _cluster
    assert check.flash_problem(dict(SETTINGS), atoms) is None
    s = dict(SETTINGS, do_enhanced_sampling=True)
    assert check.flash_problem(s, atoms) == "its enhanced sampling needs a box"
    ran, _ = so3lr_md.run_on_flash(s, atoms, "md", lambda flash_md: pytest.fail("flash ran"))
    assert not ran
    with pytest.raises(ValueError, match="--engine flash: its enhanced sampling needs a box"):
        so3lr_md.run_on_flash(dict(s, engine="flash"), atoms, "md", lambda flash_md: pytest.fail("flash ran"))


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
def test_atoms_in_any_order():
    """Each call runs with the atoms sorted in space: shuffled atoms give the energy, the
    dE/d(strain) per axis and each atom's force (in the input's order) of the atoms in their
    own order without the sort, jitted and staged (float64, PME)."""
    atoms = _water((3, 3, 3))
    perm = np.random.default_rng(0).permutation(len(atoms))
    pme = dict(pme_electrostatics=(2.0, 1.55))
    e0, f0, w0 = _engine(None, atoms, "float64", sort_atoms=False, **pme).energy_forces_strain(
        jnp.asarray(atoms.get_positions(), jnp.float64), diag=True)
    shuffled = atoms[perm]
    R = jnp.asarray(shuffled.get_positions(), jnp.float64)
    eng = _engine(None, shuffled, "float64", **pme)
    for call in (eng.energy_forces_strain, eng.energy_forces_strain_staged):
        e, f, w = call(R, diag=True)
        _check("energy", e, e0, 1e-10)
        _check("forces", f, np.asarray(f0)[perm], 1e-9)
        _check("dE/d(strain) per axis", w, w0, 1e-9)


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


def _so3lr(args, cwd):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from subprocess_env import so3lr_subprocess_env
    # this process holds most of each GPU (its own tests): the runs allocate as they go,
    # leaving them room on smaller GPUs, and NCCL and cuSolver room
    result = subprocess.run(["so3lr", *args], cwd=cwd, capture_output=True, text=True,
                            env=so3lr_subprocess_env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}))
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
def test_semi_isotropic_npt_on_flash(tmp_path):
    """Semi-isotropic NPT on flash (npt_coupling: semi-isotropic, jax-md's flexible cell, the
    engine's per-axis dE/d(strain)): it runs, x and y stay equal, the log has the pressure
    tensor's diagonal, and a restart file continues it."""
    import yaml
    from ase.io import write
    write(tmp_path / "w.xyz", _water((3, 3, 3)))
    base = dict(input_file="w.xyz", ensemble="npt", npt_coupling="semi-isotropic", md_P=1.0, md_T=300.0,
                md_dt=0.5, md_cycles=2, md_steps=5, relax_before_run=False)
    for name, extra in (("a", dict(restart_save_path="a.npz")), ("b", dict(restart_load_path="a.npz"))):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(
            dict(base, output_file=f"{name}.hdf5", log_file=f"{name}.log", **extra)))
        _so3lr(["--settings", f"{name}.yaml"], tmp_path)
        log = (tmp_path / f"{name}.log").read_text()
        assert "Engine: flash" in log and "P [bar] (xx,yy,zz)" in log, log[-2000:]
        rows = [r.split(" - ")[-1].split("\t") for r in log.splitlines()]
        rows = [r for r in rows if len(r) > 6 and r[0].isdigit()]
        sides = [float(v) for v in rows[-1][-2].strip("()").split(",")]
        assert sides[0] == sides[1] and np.isfinite(sides).all(), rows[-1]


def test_surrogate_takes_the_strain_derivative_per_axis():
    """The NPT step's surrogate energy (CPU): its gradient by a flexible-cell perturbation
    (1 + diag(strain)) is the per-axis dE/d(strain) given, by the isotropic one (1 + strain)
    the number given; a per-axis W needs three numbers."""
    from so3lr.flash.md import _surrogate
    R = jnp.zeros((2, 3))
    W3 = jnp.asarray([1.0, 2.0, 3.0])
    g = jax.grad(lambda e: _surrogate(R, perturbation=jnp.eye(3) + jnp.diag(e), W=W3))(jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(g), np.asarray(W3))
    assert float(jax.grad(lambda e: _surrogate(R, perturbation=1.0 + e, W=6.0))(0.0)) == 6.0
    with pytest.raises(ValueError, match="per axis"):
        _surrogate(R, perturbation=jnp.eye(3), W=6.0)


def test_npt_start_motion():
    """The NPT init's starting motion (CPU): momenta for the isotropic state, velocities
    (momenta / masses) for the flexible cell, None where none are given."""
    from so3lr.cli.so3lr_md import npt_start_motion
    p, m = jnp.ones((2, 3)), jnp.asarray([2.0, 4.0])
    assert npt_start_motion(p, m, "isotropic")["momenta"] is p
    v = npt_start_motion(p, m, "semi_isotropic")["velocities"]
    np.testing.assert_allclose(np.asarray(v), np.asarray(p / m[:, None]))
    assert npt_start_motion(None, m, "semi-isotropic") == dict(velocities=None)

@GPU
@pytest.mark.parametrize("ensemble", ["nvt-langevin", "nvt"])
def test_md_holds_fixed_atoms_on_flash(tmp_path, ensemble):
    """fixed_atom_indices on flash (Langevin and Nose-Hoover): the held atoms end exactly where
    they started, with zero momentum, while the rest of the box moves."""
    import h5py
    import yaml
    from ase.io import write
    atoms = _water((3, 3, 3))
    atoms.wrap()
    write(tmp_path / "w.xyz", atoms)
    held = [0, 1, 2, 3, 4, 5]
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(dict(
        input_file="w.xyz", output_file="a.hdf5", log_file="a.log", ensemble=ensemble, engine="flash",
        md_cycles=2, md_steps=10, md_dt=0.5, md_T=300.0, langevin_thermo=200.0, relax_before_run=False,
        output_cartesian_positions=True, fixed_atom_indices=held, save_buffer=1)))
    _so3lr(["--settings", "s.yaml"], tmp_path)
    assert "Engine: flash" in (tmp_path / "a.log").read_text()
    L = np.diag(atoms.cell[:])
    with h5py.File(tmp_path / "a.hdf5") as f:
        x = np.asarray(f["positions"]).reshape(-1, len(atoms), 3)
        p = np.asarray(f["momenta"]).reshape(-1, len(atoms), 3)
    d = x - atoms.positions[None]
    d -= L * np.round(d / L)
    assert np.abs(d[:, held]).max() < 1e-5, np.abs(d[:, held]).max()
    assert np.abs(p[:, held]).max() == 0.0
    free = np.setdiff1d(np.arange(len(atoms)), held)
    assert np.abs(d[:, free]).max() > 1e-3


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


@GPU
def test_flash_runs_pysages(tmp_path):
    """ABF (a coordination number and a distance) on 5,184 waters, continued from one
    checkpoint, both engines in the default precision (mixed: a float64 MD state around a
    float32 model): auto picks flash, which follows --engine standard to float32 accuracy in
    the per-step CVs, the trajectory and the samples."""
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
                      save_to=str(tmp_path / "out.extxyz"), log_file=str(tmp_path / "o.log"))
    log = (tmp_path / "o.log").read_text()
    assert "flash cannot: structure 1: its atoms span" in log, log[-2000:]
    energies = [a.info["energy_so3lr"] for a in read(tmp_path / "out.extxyz", ":")]
    assert len(energies) == 2 and np.isfinite(energies).all(), energies


GPUS = pytest.mark.skipif(check.device_problem() is not None or len(jax.local_devices()) < 2,
                          reason="several GPUs: needs two GPUs flash runs on or more")


@GPUS
@pytest.mark.parametrize("precision,tol", [("float64", 1e-10), ("float32", 1e-5)])
def test_several_gpus_give_one_gpus_forces(precision, tol):
    """Energy, forces and dE/d(strain) per axis on two and on all visible GPUs as on one: PME
    for both, an atom count the GPUs do not divide (the last one shares rows); and with
    neighbour rows too short, widened as on one GPU."""
    from so3lr.flash.loader import load_params
    atoms = _water((3, 3, 3))[:-3]
    params, statics = load_params(None, precision=precision, lr_cutoff=10.0)
    args = (params, statics, atoms.get_atomic_numbers(), np.asarray(atoms.cell[:]))
    kw = dict(pme_electrostatics=(2.0, 1.55), pme_dispersion=(2.0, 1.55), **E.MODES["fast"][0])
    R = jnp.asarray(atoms.get_positions(), statics["dtype"])
    e_ref, f_ref, w_ref = E.FlashSO3LR(*args, **kw).energy_forces_strain(R, diag=True)
    for P in sorted({2, len(jax.local_devices())}):
        e, f, w = E.FlashSO3LR(*args, devices=P, **kw).energy_forces_strain(R, diag=True)
        _check(f"energy on {P} GPUs", e, e_ref, tol / 100)
        _check(f"forces on {P} GPUs", f, f_ref, tol)
        _check(f"dE/d(strain) per axis on {P} GPUs", w, w_ref, tol)
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
    _so3lr(["nvt", *run, "--output", "a.hdf5"], tmp_path)
    _so3lr(["nvt", *run, "--flash-devices", "2", "--output", "b.hdf5"], tmp_path)
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


# ---- the CUDA kernels (so3lr/flash/cuda) -----------------------------------------------
def _cuda_kernels_problem():
    """Why the CUDA kernels cannot run here (None: they can): no GPU, one of compute capability
    below 8.0, no nvcc."""
    from so3lr.flash.cuda import build
    return build.problem()


CUDA = pytest.mark.skipif(_cuda_kernels_problem() is not None,
                          reason="the CUDA kernels need a GPU of compute capability 8.0 or newer and nvcc")


# shared memory in bytes: per block (opt-in), per SM, reserved per block
SMEM_227K, SMEM_163K, SMEM_99K = (232448, 233472, 1024), (166912, 167936, 1024), (101376, 102400, 1024)
# the kernels' build variants of so3lr-tauri-m and -l (cuda.variant)
M_VARIANT, L_VARIANT = (128, 4, 24, 32, 0x4321, 48), (256, 8, 48, 64, 0x44332211, 64)


@pytest.mark.parametrize("limits,m,l", [
    (SMEM_227K, ((64, 2, True, 2), (32, 2, False, 1)), ((64, 2, True, 1), (32, 2, True, 1))),
    (SMEM_163K, ((32, 2, True, 2), (32, 2, False, 1)), ((16, 2, True, 1), (16, 1, True, 1))),
    (SMEM_99K, ((64, 2, True, 1), (32, 2, True, 1)), None),
])
def test_cuda_kernel_tiles(monkeypatch, limits, m, l):
    """The attention kernels' tiles (pairs, stages, a CTA per channel, CTAs per SM) follow the
    GPU's shared memory per block and per SM: the largest that fit, two CTAs of four heads on an
    SM where they fit; none for so3lr-tauri-l where a block has less than its 16-pair tiles'."""
    from so3lr.flash.cuda import build
    monkeypatch.setattr(build, "smem", lambda: limits)
    for var, want in ((M_VARIANT, m), (L_VARIANT, l)):
        t = build.tiles(var)
        assert (None if t is None else (t["fwd"][:4], t["bwd"][:4])) == want
        assert t is None or max(t["fwd"].smem, t["bwd"].smem) <= limits[0]


def _engine(model, atoms, precision, **kw):
    from so3lr.flash.loader import load_params
    params, statics = load_params(model, precision=precision, lr_cutoff=12.0)
    return E.FlashSO3LR(params, statics, atoms.get_atomic_numbers(), np.asarray(atoms.cell[:]), **kw)


def _skip_without_tiles(model):
    """Skip on a GPU whose shared memory per block so3lr-tauri-l's kernels do not fit."""
    from so3lr.flash.cuda import build
    if "tauri-l" in model and build.tiles(L_VARIANT) is None:
        pytest.skip("so3lr-tauri-l's CUDA kernels need more shared memory per block than this GPU has")


@functools.cache
def _cuda_ref(model):
    """Energy, forces and dE/d(strain) per axis of 27 water boxes on the CUDA kernels as built for
    this GPU, every layer's per-pair data kept: the reference of the kernels' other paths."""
    _skip_without_tiles(model)
    atoms = _water((3, 3, 3))
    eng = _engine(model, atoms, "float32", keep_attention=True, kernels="cuda")
    assert eng.cuda_kernels is not None and eng.cuda_kernels.chunk is None
    return tuple(np.asarray(x) for x in eng.energy_forces_strain(jnp.asarray(atoms.get_positions(), jnp.float32),
                                                                   diag=True))


@CUDA
@pytest.mark.parametrize("model", ["so3lr-tauri-m-2606", "so3lr-tauri-s-2606", "so3lr-tauri-l-2606", "so3lr"])
def test_cuda_kernels_follow_float64(model):
    """The CUDA kernels (asked for explicitly, in the fast memory mode they serve) give the
    float64 engine's energy, forces and dE/d(strain) per axis (the angular forces' part, which
    cancels in the isotropic sum) to float32 accuracy, in the jitted program and in the
    staged one."""
    _skip_without_tiles(model)
    atoms = _water((4, 4, 4))
    R = atoms.get_positions()
    e_ref, f_ref, w_ref = _engine(model, atoms, "float64").energy_forces_strain(jnp.asarray(R, jnp.float64), diag=True)
    eng = _engine(model, atoms, "float32", keep_attention=True, kernels="cuda")
    e, f, w = eng.energy_forces_strain(jnp.asarray(R, jnp.float32), diag=True)
    assert eng.cuda_kernels is not None
    _check("energy", e, e_ref, 1e-6)
    _check("forces", f, f_ref, 1e-4)
    _check("dE/d(strain) per axis", w, w_ref, 1e-4)
    e2, f2, w2 = eng.energy_forces_strain_staged(jnp.asarray(R, jnp.float32), diag=True)
    _check("staged energy", e2, e, 1e-6)
    _check("staged forces", f2, f, 1e-5)
    _check("staged dE/d(strain) per axis", w2, w, 1e-5)


@CUDA
def test_cuda_kernels_fall_back_where_they_do_not_apply():
    """auto takes the Pallas kernels for a float64 model and for the memory modes that split
    the attention into passes (the Pallas kernels' own); cuda asked for explicitly raises
    there instead."""
    atoms = _water((3, 3, 3))
    for precision, kw in (("float64", dict(keep_attention=True)),
                          ("float32", dict(keep_layer_inputs=False, split_channels=True))):
        R = jnp.asarray(atoms.get_positions(), jnp.float64 if precision == "float64" else jnp.float32)
        eng = _engine("so3lr-tauri-m-2606", atoms, precision, **kw)
        eng.energy_forces(R)
        assert eng.cuda_kernels is None
        with pytest.raises(RuntimeError, match="do not serve"):
            _engine("so3lr-tauri-m-2606", atoms, precision, kernels="cuda", **kw).energy_forces(R)


@CUDA
def test_cuda_kernels_widen_rows(monkeypatch):
    """Neighbour rows too short for water on the CUDA kernels: the engine widens them (its
    pair records and filters sized for the wider rows at the next program) and gives what
    wide ones give, jitted and staged; where the wider rows' per-pair data no longer fits the
    memory the kernels were planned in, a chunk of rows at a time."""
    from so3lr.flash import cuda
    from so3lr.flash import md as flash_md
    atoms = _water((3, 3, 3))
    R = jnp.asarray(atoms.get_positions(), jnp.float32)
    e_ref, f_ref, _ = _cuda_ref("so3lr-tauri-m-2606")

    def widened(staged):
        eng = _engine("so3lr-tauri-m-2606", atoms, "float32", keep_attention=True, kernels="cuda", max_neighbours=16)
        e, f = (eng.energy_forces_staged if staged else eng.energy_forces)(R)
        assert eng.K > 16
        _check("energy", e, e_ref, 1e-6)
        _check("forces", f, f_ref, 1e-5)
        return eng

    for staged in (False, True):
        eng = widened(staged)
        assert eng.cuda_kernels.chunk is None
    need = cuda.kept_bytes(eng)          # the wider rows' per-pair data, kept
    monkeypatch.setattr(flash_md, "_free_bytes", lambda: need - 1)
    assert widened(False).cuda_kernels.chunk is not None


@CUDA
@pytest.mark.parametrize("model,kw", [
    ("so3lr-tauri-m-2606", dict(keep_attention=True)),
    ("so3lr-tauri-m-2606", dict()),                              # the attention outputs made again
    ("so3lr-tauri-m-2606", dict(keep_layer_inputs=False)),       # the layer inputs too
    ("so3lr-tauri-s-2606", dict(keep_attention=True)),
    ("so3lr-tauri-l-2606", dict(keep_attention=True)),
    ("so3lr-tauri-l-2606", dict(keep_layer_inputs=False)),
])
def test_cuda_kernels_in_chunks(model, kw):
    """The CUDA kernels a chunk of rows at a time (each chunk's per-pair data made in the
    forward and again in the backward; chunks forced small: several, the last one shifted back
    over the one before it), in the engine's memory modes, give what the kernels keeping every
    layer's per-pair data give: energy, forces and dE/d(strain) per axis, jitted and staged,
    and on two GPUs."""
    from so3lr.flash import cuda
    atoms = _water((3, 3, 3))
    R = jnp.asarray(atoms.get_positions(), jnp.float32)
    e_ref, f_ref, w_ref = _cuda_ref(model)
    runs = [(1, False, 1100), (1, True, 1100)]
    if len(jax.local_devices()) >= 2 and kw.get("keep_attention"):
        runs.append((2, False, 550))                               # several GPUs run the jitted fast mode
    for devices, staged, rows in runs:
        eng = _engine(model, atoms, "float32", kernels="pallas", devices=devices, **kw)
        ck = cuda.attach(eng, strict=True, chunk=rows)
        nchunks = -(-eng.nl // ck.chunk)
        assert nchunks > 1 and nchunks * ck.chunk > eng.nl          # the last chunk shifted back
        e, f, w = (eng.energy_forces_strain_staged if staged else eng.energy_forces_strain)(R, diag=True)
        _check("energy", e, e_ref, 1e-6)
        _check("forces", f, f_ref, 1e-5)
        _check("dE/d(strain) per axis", w, w_ref, 1e-5)


@CUDA
@pytest.mark.parametrize("limits,fits,declines", [
    (SMEM_163K, ["so3lr-tauri-m-2606", "so3lr-tauri-l-2606"], []),
    (SMEM_99K, ["so3lr-tauri-m-2606"], ["so3lr-tauri-l-2606"]),
])
def test_cuda_kernels_for_less_shared_memory(monkeypatch, limits, fits, declines):
    """The kernels as built for compute capability 8.0 (this GPU's driver compiles their PTX)
    and a GPU of less shared memory (smaller tiles, one stage, a CTA per channel) give what this
    GPU's build gives, the per-pair data kept and in chunks; a model whose kernels need more
    shared memory than a block has takes the Pallas kernels."""
    from so3lr.flash import cuda
    from so3lr.flash.cuda import build
    if build.arch() == "sm_80" and build.smem() == limits:
        pytest.skip("this GPU's own build")
    atoms = _water((3, 3, 3))
    R = jnp.asarray(atoms.get_positions(), jnp.float32)
    refs = {m: _cuda_ref(m) for m in fits}                         # this GPU's build, before the patch
    monkeypatch.setattr(build, "arch", lambda: "sm_80")
    monkeypatch.setattr(build, "smem", lambda: limits)
    for m in fits:
        e_ref, f_ref, w_ref = refs[m]
        for chunk in (None, 1100):
            eng = _engine(m, atoms, "float32", keep_attention=True, kernels="pallas")
            assert "_sm_80_" in cuda.attach(eng, strict=True, chunk=chunk).prefix
            e, f, w = eng.energy_forces_strain(R, diag=True)
            _check("energy", e, e_ref, 1e-6)
            _check("forces", f, f_ref, 1e-5)
            _check("dE/d(strain) per axis", w, w_ref, 1e-5)
    for m in declines:
        eng = _engine(m, atoms, "float32", keep_attention=True, kernels="pallas")
        assert cuda.attach(eng) is None
        with pytest.raises(RuntimeError, match="shared memory"):
            cuda.attach(eng, strict=True)


@CUDA
def test_cuda_kernels_asked_for_say_why_they_do_not_fit(monkeypatch):
    """flash_kernels cuda where the CUDA kernels fit none of their modes and the mode that fits
    is one they do not serve (the attention in passes): the MD's engine says so, with the memory
    they need."""
    from so3lr.flash import md as flash_md
    monkeypatch.setattr(flash_md, "_free_bytes", lambda: 1 << 29)
    with pytest.raises(RuntimeError, match="flash_kernels cuda.*GiB of the 0 free.*minimal"):
        flash_md.build_engine(_water((3, 3, 3)), dict(model_path="so3lr-tauri-m-2606", flash_kernels="cuda"),
                              jnp.float32)


# ---- boxes under 2 x a cutoff: that range over the periodic images ----
SMALL_PME = dict(pme_electrostatics=(2.0, 1.55), pme_dispersion=(2.0, 1.55))


def _diamond():
    """Rattled diamond in a 3.57 x 3.57 x 7.13 A box: every side under the short-range cutoff."""
    from ase.build import bulk
    atoms = bulk("C", "diamond", a=3.567, cubic=True).repeat((1, 1, 2))
    atoms.rattle(0.05, seed=0)
    return atoms


# 64 waters' 12.4 A box (two cells a side; PME meshes that scale with the box) and diamond
SMALL_BOXES = pytest.mark.parametrize("make,reps,kw", [(lambda: _water((1, 1, 1)), (4, 4, 4), SMALL_PME),
                                                       (_diamond, (9, 9, 5), {})],
                                      ids=["water-12A-pme", "diamond-3.6A"])


@GPU
@SMALL_BOXES
def test_small_box_follows_its_supercell(make, reps, kw):
    """A box under 2 x the cutoffs gives per copy what its supercell, wide enough for the
    minimum image, gives: the energy per atom, each atom's force and dE/d(strain) per axis
    (float64)."""
    atoms = make()
    big = atoms.repeat(reps)
    n = int(np.prod(reps))
    eng = _engine("so3lr-tauri-m-2606", atoms, "float64", **kw)
    ref = _engine("so3lr-tauri-m-2606", big, "float64", **kw)
    assert eng.images and not ref.images
    if kw:
        for m, m_ref in ((eng.mesh_el, ref.mesh_el), (eng.mesh_disp, ref.mesh_disp)):
            assert [a * r for a, r in zip(m, reps)] == list(m_ref), (m, m_ref)
    e, f, w = eng.energy_forces_strain(jnp.asarray(atoms.get_positions(), jnp.float64), diag=True)
    e_ref, f_ref, w_ref = ref.energy_forces_strain(jnp.asarray(big.get_positions(), jnp.float64), diag=True)
    _check("energy", float(e) * n, e_ref, 1e-10)
    _check("forces", f, np.asarray(f_ref)[:len(atoms)], 1e-9)       # ase's repeat: the unit cell first
    _check("dE/d(strain) per axis", np.asarray(w) * n, w_ref, 1e-9)


@CUDA
@SMALL_BOXES
def test_cuda_small_box(make, reps, kw):
    """The CUDA kernels in a box under 2 x the cutoffs give the float64 engine's energy,
    forces and dE/d(strain) per axis to float32 accuracy."""
    atoms = make()
    R = atoms.get_positions()
    e_ref, f_ref, w_ref = _engine("so3lr-tauri-m-2606", atoms, "float64", **kw).energy_forces_strain(
        jnp.asarray(R, jnp.float64), diag=True)
    eng = _engine("so3lr-tauri-m-2606", atoms, "float32", keep_attention=True, kernels="cuda", **kw)
    assert eng.cuda_kernels is not None and eng.images
    e, f, w = eng.energy_forces_strain(jnp.asarray(R, jnp.float32), diag=True)
    _check("energy", e, e_ref, 1e-6)
    _check("forces", f, f_ref, 1e-4)
    _check("dE/d(strain) per axis", w, w_ref, 1e-4)


@GPU
def test_md_in_a_small_box_on_flash(tmp_path):
    """`so3lr npt` on 64 waters (12.4 A, under 2 x the 12 A cutoff: the neighbour-list
    engine refuses the box) runs on flash."""
    from ase.io import write
    write(tmp_path / "w.xyz", _water((1, 1, 1)))
    _so3lr(["npt", "--input", "w.xyz", "--no-relax", "--md-cycles", "2", "--md-steps", "5", "--pressure", "1",
            "--output", "a.hdf5"], tmp_path)
    log = (tmp_path / "a.log").read_text()
    assert "Engine: flash" in log, log[-2000:]
