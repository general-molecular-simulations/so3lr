"""SO3LR-Flash in ``so3lr nve|nvt|nvt-langevin|npt`` and ``so3lr opt``: jax-md dynamics and
FIRE with the flash engine.

``so3lr_md.run`` runs the flash engine whenever it computes what the settings ask for
(``check.flash_problem`` is None: an NVIDIA GPU or an AMD MI355X, a periodic orthorhombic
box or none (the gas phase), nve / nvt / nvt-langevin / isotropic npt, no k-space other than PME,
so3lr-tauri-s, -m, -l or the legacy so3lr), else the neighbour-list potential of ``perform_md``;
``perform_min`` (``so3lr opt``) likewise runs ``perform_min_flash``. ``--engine flash`` or
``--engine standard`` forces one.

Same settings, units, log lines, trajectory and restart files as ``perform_md``; the
forces come from ``FlashSO3LR`` instead of the neighbour-list potential. NVE and NVT
integrate Cartesian positions (restart files are converted on load either way); NPT
integrates fractional ones, as ``perform_md`` does. With no box the positions are free (never
wrapped, as ``perform_md``'s) and the engine computes them in a box with room for the atoms
and the cutoffs, laid out again if they outgrow it. With ``flash_mixed`` the MD state is
float64 around a float32 model; trajectories are written in the model's precision.

A step's new positions (and NPT box) depend only on the previous step's force (and
strain derivative), so a step can run in two calls: the jax-md step with zero force
gives the positions, the engine evaluates there, and the step runs again with the
result. The staged memory modes (``engine.MODES``), whose force cannot be traced into a
jax-md step, run every step this way; the jitted ones run ``md_steps - 1`` steps in one
jitted loop and only the last in two calls, which also gives the energy for the log.
NPT runs every step in two calls: the engine evaluates in the box the step reaches
(``set_box``), and the step takes the force and dE/d(strain) through an energy whose
gradients they are (``_surrogate``).
"""
import gc
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
from ase.io import read
from jax_md import units

from so3lr.cli.so3lr_md import (
    RestartInNewEnsembleError,
    _existing_regular_output_mode,
    _validate_md_output_write_policy,
    atoms_to_jnp,
    box_to_diag,
    cell_from_atoms_boundary_conditions,
    handle_units,
    init_hdf5_store,
    load_state,
    save_restart_checkpoints,
    setup_jax_globals,
    setup_logger,
    validate_and_log_input,
    validate_npt_restart_coupling,
    validate_orthorhombic_restart_box,
    warn_legacy_kspace,
    write_to_extxyz,
    write_to_hdf5,
)
from so3lr.md_settings_validation import validate_md_counts
from so3lr.mlff.nn.observable.dispersion_ref_data import (
    check_atomic_numbers_supported, supported_atomic_numbers_for_model)

from .check import NPT_BOX_MARGIN, flash_problem, in_vacuum, model_cutoff, structure_problem, vacuum_box, vacuum_size
from .engine import MODES, FlashSO3LR, lr_scratch_bytes, mode_kib, pick_mode, pme_bytes
from .loader import load_params

logger = logging.getLogger("SO3LR")


def _pme(settings):
    """(smearing, spacing) for electrostatics and for dispersion PME, None where off."""
    from so3lr.mlff.utils import kspace_utils as K
    s = settings

    def get(key, default):
        return float(s.get(key) if s.get(key) is not None else default)
    el = ((get("kspace_smearing", K.DEFAULT_KSPACE_SMEARING), get("kspace_spacing", K.DEFAULT_KSPACE_SPACING))
          if s.get("kspace_electrostatics") == "pme" else None)
    disp = ((get("kspace_dispersion_smearing", K.DEFAULT_KSPACE_DISPERSION_SMEARING),
             get("kspace_dispersion_spacing", K.DEFAULT_KSPACE_DISPERSION_SPACING))
            if s.get("kspace_dispersion") == "pme" else None)
    return el, disp


def dtypes(settings):
    """(the model's dtype, the positions'): float64 positions (and MD state) around a float32
    model unless flash_mixed is off or x64 is (float32 coordinates of a 500 A box round to
    6e-5 A)."""
    dtype = jnp.float64 if settings.get("precision", "float32") == "float64" else jnp.float32
    mixed = dtype == jnp.float32 and bool(settings.get("flash_mixed", True)) and jax.config.jax_enable_x64
    return dtype, (jnp.float64 if mixed else dtype)


def build_engine(atoms, settings, dtype, box_margin=None, positions=None):
    """FlashSO3LR for ``atoms`` with the model, charges, PME and memory mode of
    ``settings``; the cell grid laid out for a box ``box_margin`` smaller (default: NPT's
    margin in NPT, else none). Atoms in free space (``check.in_vacuum``): the engine's box
    laid out for them at ``positions`` (default the atoms')."""
    if box_margin is None:
        box_margin = NPT_BOX_MARGIN if settings.get("ensemble") == "npt" else 0.0
    precision = "float64" if dtype == jnp.float64 else "float32"
    params, statics = load_params(settings.get("model_path"), precision=precision,
                                  lr_cutoff=settings.get("lr_cutoff", 12.0),
                                  dispersion_damping=settings.get("dispersion_damping", 2.0))
    n = len(atoms)
    L = [float(v) for v in np.diag(np.asarray(atoms.cell[:]))]
    pme_el, pme_disp = _pme(settings)
    vacuum = in_vacuum(atoms)
    size = vacuum_size(np.asarray(atoms.get_positions() if positions is None else positions, np.float64)) \
        if vacuum else None
    # what the modes' KiB per atom leave out: PME's meshes, and in free space where the reach
    # takes every pair, the long range's scratch (a row of every atom per program)
    reserve = pme_bytes(L, (pme_el, pme_disp), jnp.dtype(dtype).itemsize)
    if vacuum and vacuum_box(size, float(statics["cutoff"]), float(statics["long_range_cutoff"]))[1] > size:
        reserve += lr_scratch_bytes(n, n)
    devices = int(settings.get("flash_devices") or 1)
    # several GPUs run the jitted fast mode (validate_options refuses another)
    mode = "fast" if devices > 1 else (settings.get("flash_mode") or "auto")
    if mode == "auto":
        mode = pick_mode(n, _free_bytes() - reserve, statics)
    total_charge = float(settings.get("total_charge") or 0)
    embed = settings.get("charge_embed_total_charge")
    equil = settings.get("charge_equilibration_total_charge")
    kw, staged, _ = MODES[mode]
    # the sides only: structure_problem takes a cell with off-diagonals up to 1e-9 as orthorhombic
    eng = FlashSO3LR(params, statics, atoms.get_atomic_numbers(), None if vacuum else np.diag(L),
                     total_charge=total_charge if embed is None else float(embed),
                     total_spin=float(settings.get("num_unpaired_electrons") or 0),
                     equilibration_charge=total_charge if equil is None else float(equil),
                     pme_electrostatics=pme_el, pme_dispersion=pme_disp,
                     box_margin=box_margin, devices=devices,
                     size=size, **kw)
    logger.info(f"Flash: {n} atoms, memory mode {mode} ({'staged' if staged else 'jitted'}), "
                f"~{mode_kib(mode, statics):.1f} KiB/atom at peak")
    if vacuum:
        logger.info(f"Flash: {_free_space(eng)}")
    if eng.P > 1:
        logger.info(f"Flash: {eng.P} GPUs, {eng.nl} atoms' rows each")
        # each GPU holds every atom's arrays of the fast mode
        alone = pick_mode(n, _free_bytes() - reserve, statics)
        if alone != "fast":
            logger.warning(f"Flash: {n} atoms may not fit the fast mode each of the {eng.P} GPUs runs "
                           f"(one GPU would run {alone})")
    for what, p, mesh in (("electrostatics", pme_el, eng.mesh_el), ("dispersion", pme_disp, eng.mesh_disp)):
        if p:
            logger.info(f"Flash: PME {what}, smearing {p[0]} A, mesh {mesh}")
    if statics["legacy"] and pme_el:
        warn_legacy_kspace("pme")
    return eng, staged


def replicated(eng, tree):
    """tree on every device of a several-GPU engine (each runs the MD step on all atoms;
    the force call splits the atoms), as is on one."""
    if eng.device_mesh is None:
        return tree
    from jax.sharding import NamedSharding, PartitionSpec
    return jax.device_put(tree, NamedSharding(eng.device_mesh, PartitionSpec()))


def _free_space(eng):
    """The box and long-range reach free space is computed in, for the log."""
    return f"no box: computed in a {eng.L[0]:.1f} A box, long-range partners within {eng.reach:.1f} A"


def _start_box(atoms, vacuum, dtype):
    """perform_md's box: the sides of atoms' cell (the atoms wrapped into it), or [0] with
    no box."""
    if vacuum:
        return jnp.zeros((1,), dtype)
    atoms.wrap()
    return jnp.asarray(np.diag(np.asarray(atoms.cell[:])), dtype)


def _free_bytes():
    try:
        st = jax.local_devices()[0].memory_stats()
        return st.get("bytes_limit", 0) - st.get("bytes_in_use", 0)
    except Exception:
        return 80 * 2 ** 30


def _force_fn(eng):
    """Forces for jax-md, in the positions' dtype: ``F`` if passed (a step run in two
    calls), else the jitted engine with NaN on an overflow (a table, or free space's box),
    which ``_run_jitted`` catches and runs again a step at a time."""
    def force_fn(R, F=None, **kwargs):
        if F is None:
            _, F, over, _ = eng._energy_forces(R)
            F = jnp.where(over > 0, jnp.nan, F)
        return F.astype(R.dtype)
    return force_fn


def _surrogate(R, box=None, perturbation=1.0, F=None, W=None, E=None, **kwargs):
    """An energy with given value E and gradients: -F with respect to the positions, W
    with respect to the strain (perturbation = 1 + strain). jax-md's NPT step takes its
    force and dE/d(strain) from it, so they can come from the flash engine."""
    e = jnp.zeros((), R.dtype) if E is None else jnp.asarray(E, R.dtype)
    if F is not None:
        e = e - jnp.sum(F.astype(R.dtype) * (R - jax.lax.stop_gradient(R)))
    if W is not None:
        e = e + jnp.asarray(W, R.dtype) * (jnp.asarray(perturbation, R.dtype) - 1.0)
    return e


def _finite(state):
    return bool(jnp.isfinite(state.force).all() & jnp.isfinite(state.position).all())


def _run_jitted(eng, make, state, k, predict, finish):
    """k steps of make()'s integrator (jax-md's (init, apply) on the engine as it is) in one
    jitted loop. jax-md jits its steps (FIRE through its NVE step), and a jitted step keeps
    the engine's programs of its first trace, so the integrator is made again after the
    engine is rebuilt (which empties eng.loops). If a step overflowed (its force is NaN,
    and so are the positions of any step after it), the cycle runs again a step at a time
    in two calls (predict, finish: _two_call_step), where the engine widens its tables or
    lays out its box again (free space) at the step that needs it."""
    if (make, k) not in eng.loops:
        step = make()[1]
        eng.loops[make, k] = jax.jit(lambda s: jax.lax.fori_loop(0, k, lambda i, x: step(x), s))
    new = eng.loops[make, k](state)
    if _finite(new):
        return new
    for _ in range(k):
        R = predict(state)
        F = eng.energy_forces(R)[1]
        if not (bool(jnp.isfinite(R).all()) and bool(jnp.isfinite(F).all())):
            raise RuntimeError("Flash: non-finite positions or forces: the dynamics diverged. Check the "
                               "timestep, temperature and structure.")
        state = finish(state, F)
    logger.info(f"Flash: a step overflowed; cycle repeated a step at a time (neighbour rows {eng.K}, "
                f"scratch {eng.maxh}{'; ' + _free_space(eng) if eng.vacuum else ''})")
    return state


def _two_call_step(apply_fn):
    """(predict, finish): the positions a step reaches, and the step with given forces."""
    predict = jax.jit(lambda st: apply_fn(st, F=jnp.zeros_like(st.position)).position)
    finish = jax.jit(lambda st, F: apply_fn(st, F=F))
    return predict, finish


def relax_flash(eng, staged, position, shift, box, settings, atoms, start, opt_file):
    """FIRE as ``perform_min`` runs it (same settings and log, the trajectory to
    ``opt_file`` unless None) from ``position``, where (energy, forces) are ``start``;
    returns the relaxed positions."""
    s = settings
    overwrite = bool(s.get("overwrite", False))
    if opt_file is not None and _existing_regular_output_mode(opt_file) is not None and not overwrite:
        raise RuntimeError(f"Output file already exists: {opt_file}. Refusing to overwrite. "
                           f"Pass --overwrite to overwrite, or choose a different --output.")
    def fire():
        """jax-md's FIRE on the engine as it is (made again after a rebuild: _run_jitted)."""
        return jax_md.minimize.fire_descent(
            _force_fn(eng), shift, dt_start=s.get("min_start_dt", 0.05), dt_max=s.get("min_max_dt", 0.1),
            n_min=s.get("min_n_min", 2))
    init, apply = fire()
    min_cycles, min_steps, fconv = s.get("min_cycles", 10), s.get("min_steps", 10), s.get("force_convergence")
    ef = eng.energy_forces_staged if staged else eng.energy_forces
    predict, finish = _two_call_step(apply)
    e, F = start
    state = jax.jit(lambda R, F: init(R, F=F))(position, F)
    f_max = float(jnp.abs(F).max())
    logger.info("Starting minimization (flash engine)...")
    logger.info("Step\tE [eV]\tF_max [eV/Å]")
    logger.info(f"0\t {float(e):.3f}\t {f_max:.3f}")
    frames = [np.asarray(position)]
    for c in range(1, min_cycles + 1):
        if staged:
            for _ in range(min_steps):
                state = finish(state, ef(predict(state))[1])
        else:
            state = _run_jitted(eng, fire, state, min_steps, predict, finish)
        e, F = ef(state.position)
        f_max = float(jnp.abs(F).max())
        logger.info(f"{c * min_steps}\t {float(e):.3f}\t {f_max:.3f}")
        frames.append(np.asarray(state.position))
        if fconv is not None and f_max <= fconv:
            logger.info(f"Convergence criterion met at step {c * min_steps}: Fmax = {f_max:.3f} eV/Å")
            break
    else:
        if fconv is not None:
            logger.warning(f"FIRE minimization hit the min_cycles ceiling ({min_cycles} cycles * {min_steps} "
                           f"steps) without reaching force_convergence={fconv:.3f} eV/Å (last Fmax = "
                           f"{f_max:.3f} eV/Å). Increase --min-cycles or relax --force-conv if tighter "
                           f"convergence is required.")
    if opt_file is not None:
        write_to_extxyz(opt_file, atoms, boxes=np.asarray(box), momenta=None, positions=frames,
                        fractional_coordinates=False, append=False, overwrite=overwrite)
        logger.info(f"Optimization trajectory with {len(frames)} frames saved to {opt_file}")
    return state.position


class FlashUnavailable(RuntimeError):
    """The flash engine could not be built or evaluated at the start (before anything is
    written): ``--engine auto`` runs the neighbour-list potential instead."""


def perform_min_flash(all_settings, atoms):
    """``perform_min`` with the flash engine: FIRE from ``atoms`` (the input structure, read
    and logged by perform_min), the trajectory to ``output_file``; returns the relaxed
    positions in A. Raises ``FlashUnavailable`` if the engine fails before the run starts."""
    s = all_settings
    check_atomic_numbers_supported(atoms.numbers, supported_atomic_numbers_for_model(s.get("model_path")))
    problem = flash_problem(s, atoms, task="opt")
    if problem:
        raise ValueError(f"flash: {problem}")
    # perform_min's cell checks (None: no box)
    vacuum = cell_from_atoms_boundary_conditions(atoms, s.get("lr_cutoff", 12.0), s.get("kspace_electrostatics"),
                                                 s.get("kspace_smearing")) is None
    atoms = atoms.copy()
    dtype, sdt = dtypes(s)
    box = _start_box(atoms, vacuum, sdt)
    position = jnp.asarray(atoms.get_positions(), sdt)
    try:
        eng, staged = build_engine(atoms, s, dtype)
        position, box = replicated(eng, (position, box))
        start = (eng.energy_forces_staged if staged else eng.energy_forces)(position)
    except Exception as exc:
        raise FlashUnavailable(f"{type(exc).__name__}: {exc}") from exc
    logger.info(f"Flash precision: model {jnp.dtype(dtype).name}, positions {jnp.dtype(sdt).name}")
    _, shift = jax_md.space.free() if vacuum else jax_md.space.periodic(box)
    # as perform_min: a trajectory only for an extxyz (or hdf5, renamed .xyz) output
    out = s.get("output_file")
    opt_file = Path(out) if out and s.get("output_format") in ("extxyz", "hdf5") else None
    return relax_flash(eng, staged, position, shift, box, s, atoms, start=start, opt_file=opt_file)


class SinglePoints:
    """Energy and forces of one structure after another (so3lr eval, the ASE calculator). The
    engine is built for the atoms, charge and spin of the structure and kept while they stay
    the same; a new box is laid out again only outside the cell grid's margin or for another
    PME mesh (so each box gets the mesh a single point of it builds). The settings, model and
    device are checked here, a structure when its atoms or box are new. box_margin: as for
    build_engine (the cell grid for a box that much smaller, so boxes within it compile
    nothing; none for a box that stays the same, which keeps every cell)."""

    def __init__(self, settings, task, matmul_precision="highest", box_margin=0.0):
        problem = flash_problem(settings, [], task=task)
        if problem:
            raise ValueError(f"flash: {problem}")
        self.settings, self.matmul_precision, self.box_margin = settings, matmul_precision, box_margin
        self.dtype, self.position_dtype = dtypes(settings)
        self.supported = supported_atomic_numbers_for_model(settings.get("model_path"))
        self.rc = model_cutoff(settings.get("model_path"))
        self.engine = self._ef = self._key = self._box = None
        self._builds = 0

    def __call__(self, atoms, charge=0.0, spin=0.0):
        """(energy, forces, compiled): compiled if the call built or laid out the engine again."""
        vacuum = in_vacuum(atoms)
        # free space: an engine of its own, which lays out its box again as the atoms need
        key = (atoms.numbers.tobytes(), float(charge), float(spin), vacuum)
        cell = np.asarray(atoms.cell[:])
        box = (cell.tobytes(), tuple(bool(p) for p in atoms.pbc))
        if key != self._key or box != self._box:
            problem = structure_problem(self.settings, atoms, self.rc)
            if problem:
                raise ValueError(f"flash: {problem}")
        with jax.default_matmul_precision(self.matmul_precision):
            if key != self._key:
                check_atomic_numbers_supported(atoms.numbers, self.supported)
                self.engine = self._ef = self._key = None    # until the new engine exists
                gc.collect()                 # the last engine's arrays, before the next one's
                self.engine, staged = build_engine(
                    atoms, dict(self.settings, total_charge=key[1], num_unpaired_electrons=key[2]), self.dtype,
                    box_margin=self.box_margin)
                self._ef = self.engine.energy_forces_staged if staged else self.engine.energy_forces
                self._key, self._builds = key, 0
            elif box != self._box and not vacuum:
                self.engine.set_box(np.diag(cell), exact_mesh=True)
            self._box = box
            R = jnp.asarray(atoms.positions if vacuum else np.mod(atoms.positions, np.diag(cell)),
                            self.position_dtype)
            e, F = self._ef(replicated(self.engine, R))
        compiled, self._builds = self.engine.builds != self._builds, self.engine.builds
        return float(e), np.asarray(F, np.float64), compiled


def perform_md_flash(all_settings, restart=False, atoms=None):
    """``perform_md`` with the flash engine (``atoms``: the input structure, if read).
    Raises ``FlashUnavailable`` if the engine fails before the run starts."""
    s = all_settings
    setup_jax_globals(precision=s.get("precision", "float32"),
                      matmul_precision=s.get("matmul_precision", "highest"))
    setup_logger(s.get("log_file"))
    validate_md_counts(s)
    atoms = read(s["input_file"]) if atoms is None else atoms
    # process_model's element check: past the trained elements the tables are gathered
    # out of bounds under jit
    check_atomic_numbers_supported(atoms.numbers, supported_atomic_numbers_for_model(s.get("model_path")))
    problem = flash_problem(s, atoms)
    if problem:
        raise ValueError(f"flash: {problem}")
    ensemble = s.get("ensemble")
    npt = ensemble == "npt"

    output_file, output_format = s.get("output_file"), s.get("output_format")
    overwrite = bool(s.get("overwrite", False))
    restart_load_path, restart_save_path = s.get("restart_load_path"), s.get("restart_save_path")
    append_extxyz = _validate_md_output_write_policy(
        output_file=output_file, output_format=output_format, overwrite=overwrite,
        restart_load_path=restart_load_path if restart else None)

    validate_and_log_input(atoms, s["input_file"], total_charge=s.get("total_charge"),
                           num_unpaired_electrons=s.get("num_unpaired_electrons", 0), verbose=False)
    # perform_md's cell checks (e.g. a long-range cutoff below 10 A needs k-space; None: no box)
    vacuum = cell_from_atoms_boundary_conditions(atoms, s.get("lr_cutoff", 12.0), s.get("kspace_electrostatics"),
                                                 s.get("kspace_smearing")) is None
    precision = s.get("precision", "float32")
    dtype, sdt = dtypes(s)
    mixed = sdt != dtype
    box = _start_box(atoms, vacuum, sdt)              # [0] in the trajectory and restart files
    geo = atoms_to_jnp(atoms, precision=sdt)
    position = geo["positions"]

    current_cycle, state = 0, None
    if restart:
        logger.info(f"Loading state from {restart_load_path}.")
        try:
            state, rbox, current_cycle = load_state(path_to_load=restart_load_path, ensemble=ensemble,
                                                    restart_reset=s.get("restart_reset", False),
                                                    fractional_coordinates=npt, precision=sdt)
            if npt:
                validate_npt_restart_coupling(state, rbox, s.get("npt_coupling") or "isotropic")
            if not vacuum:                   # (structure_problem: a restart in free space has no box)
                box = jnp.asarray(box_to_diag(rbox), sdt)
        except RestartInNewEnsembleError as e:
            logger.warning(e)
            loaded = e.loaded_state
            if loaded.get("box") is not None and np.any(np.asarray(loaded["box"]) != 0):
                validate_orthorhombic_restart_box(loaded["box"])
                box = jnp.asarray(box_to_diag(jnp.asarray(loaded["box"])), sdt)
            position = jnp.asarray(loaded["position"], sdt)
            if bool(loaded.get("fractional_coordinates", True)) and not vacuum:
                position = position * box
            geo["masses"] = jnp.asarray(loaded["mass"], sdt)
            geo["momenta"] = jnp.asarray(loaded["momentum"], sdt)
        if not vacuum:
            atoms = atoms.copy()
            atoms.set_cell(np.diag(np.asarray(box, np.float64)))      # the engine starts in this box

    # the engine and its first evaluation, where the run starts: a failure up to here (a
    # kernel that does not compile on this GPU, too little memory) has written nothing
    start = position if state is None else (state.position * box if npt else state.position)
    try:
        eng, staged = build_engine(atoms, s, dtype, positions=np.asarray(start))
        # several GPUs: the MD state on all of them
        position, box, geo, state, start = replicated(eng, (position, box, geo, state, start))
        strain = eng.energy_forces_strain_staged if staged else eng.energy_forces_strain
        e0, F0, W0 = strain(start)
    except Exception as exc:
        raise FlashUnavailable(f"{type(exc).__name__}: {exc}") from exc
    del start
    logger.info(f"Flash precision: model {precision}, MD state {'float64 (mixed)' if mixed else precision}")
    ef = eng.energy_forces_staged if staged else eng.energy_forces
    _, shift = jax_md.space.free() if vacuum else jax_md.space.periodic(box)
    if not restart and s.get("relax_before_run", True):
        out = Path(s["output_file"])
        position = relax_flash(eng, staged, position, shift, box, s, atoms, start=(e0, F0),
                               opt_file=out.with_name(f"{out.stem}_opt.xyz"))
        e0, F0, W0 = strain(position)

    md_dt_ps = s.get("md_dt")
    md_T = s.get("md_T")
    if ensemble == "nve" and s.get("init_T") is not None:
        md_T = s.get("init_T")
    unit = handle_units(units.metal_unit_system, md_dt_ps, md_T)
    md_dt, kT = sdt(unit["dt"]), sdt(unit["T"])
    md_cycles, md_steps = s.get("md_cycles"), s.get("md_steps")
    save_buffer = min(md_cycles, s.get("save_buffer"))
    thermo = s.get("langevin_thermo")
    if ensemble == "nvt-langevin" and (thermo is None or thermo <= 0):
        raise ValueError(f"--langevin-thermo must be > 0 (received {thermo})")

    def integrator():
        """jax-md's (init_fn, apply_fn) on the engine as it is (made again after a rebuild:
        _run_jitted)."""
        force_fn = _force_fn(eng)
        if ensemble == "nve":
            return jax_md.simulate.nve(force_fn, shift, md_dt)
        if ensemble == "nvt":
            return jax_md.simulate.nvt_nose_hoover(
                force_fn, shift, md_dt, kT, chain_length=s.get("nhc_chain_length"),
                chain_steps=s.get("nhc_steps"), sy_steps=s.get("nhc_sy_steps"),
                tau=md_dt * s.get("nhc_thermo"))
        return jax_md.simulate.nvt_langevin(
            force_fn, shift, md_dt, kT, gamma=1.0 / (md_dt * thermo),
            center_velocity=s.get("langevin_removecmmotion"))
    if not npt:
        init_fn, apply_fn = integrator()
    else:
        # as perform_md: fractional positions, the box a vector of its sides
        if s.get("md_P") is None:
            raise ValueError("npt needs a pressure (md_P / --pressure)")
        P = sdt(handle_units(units.metal_unit_system, md_dt_ps, md_T, s.get("md_P"))["P"])
        _, shift = jax_md.space.periodic_general(box, fractional_coordinates=True)
        thermostat = dict(chain_length=s.get("nhc_chain_length"), chain_steps=s.get("nhc_steps"),
                          sy_steps=s.get("nhc_sy_steps"), tau=md_dt * s.get("nhc_thermo"))
        barostat = dict(thermostat, tau=md_dt * s.get("nhc_baro"))
        init_fn, apply_fn = jax_md.simulate.npt_nose_hoover(
            _surrogate, shift, md_dt, pressure=P, kT=kT, barostat_kwargs=barostat,
            thermostat_kwargs=thermostat)

    def cartesian(st):
        """Positions in A and the box sides of a state."""
        if npt:
            L1 = box_to_diag(jax_md.simulate.npt_box(st))
            return st.position * L1, L1
        return st.position, box

    key = jax.random.PRNGKey(s.get("seed", 0))
    if npt:
        zero = jnp.zeros((), sdt)
        predict = jax.jit(lambda st: cartesian(apply_fn(st, F=jnp.zeros_like(st.position), W=zero, E=zero)))
        finish = jax.jit(lambda st, F, W, E: apply_fn(st, F=F, W=W, E=E))

        def advance(st, k):
            """k steps; the energy at the last positions too."""
            for _ in range(k):
                R1, L1 = predict(st)
                eng.set_box(L1)
                e, F, W = strain(R1)
                st = finish(st, F, jnp.asarray(W, sdt), jnp.asarray(e, sdt))
            return st, e
        if state is None:
            state = init_fn(key, position / box, box, mass=geo["masses"], momenta=geo.get("momenta"), kT=kT,
                            F=F0, W=jnp.asarray(W0, sdt), E=jnp.asarray(e0, sdt))
    else:
        predict, finish = _two_call_step(apply_fn)

        def advance(st, k):
            """k steps; the energy at the last positions too."""
            if not staged and k > 1:
                st, k = _run_jitted(eng, integrator, st, k - 1, predict, finish), 1
            for _ in range(k):
                e, F = ef(predict(st))
                st = finish(st, F)
            return st, e
        if state is None:
            state = jax.jit(lambda R, F, m, p: init_fn(key, R, kT=kT, mass=m, momenta=p, F=F))(
                position, F0, geo["masses"], geo.get("momenta"))
    del position, F0, W0

    unit_sys = units.metal_unit_system()

    def quantities(st, pe):
        pe = float(pe)
        ke = float(jax_md.quantity.kinetic_energy(momentum=st.momentum, mass=st.mass))
        temp = float(jax_md.quantity.temperature(momentum=st.momentum, mass=st.mass)) / unit_sys["temperature"]
        if ensemble == "nvt":
            h = float(jax_md.simulate.nvt_nose_hoover_invariant(lambda R, **k: pe, st, kT))
        elif npt:
            h = float(jax_md.simulate.npt_nose_hoover_invariant(_surrogate, st, P, kT, E=pe))
        else:
            h = ke + pe
        return ke, pe, h, temp

    def line(step, ke, pe, h, temp, per_step):
        """perform_md's log line (NPT with the box sides)."""
        b = ""
        if npt:
            L1 = np.asarray(cartesian(state)[1])
            b = f"\t({L1[0]:.3f},{L1[1]:.3f},{L1[2]:.3f})"
        return f"{step}\t{ke + pe:.3f}\t{ke:.3f}\t{pe:.3f}\t{h:.3f}\t{temp:.1f}{b}\t{per_step:.2e}"

    # trajectory positions as perform_md writes them: fractional unless asked otherwise (or
    # with no box)
    frac_out = not (vacuum or s.get("output_cartesian_positions") or s.get("enforce_cartesian_coordinates"))
    odt = np.dtype(dtype)                       # in the model's precision, as perform_md writes them
    out_idx = s.get("output_atom_indices")
    hdf5_store = None
    if output_format == "hdf5":
        from so3lr import __version__
        numbers = np.asarray(atoms.get_atomic_numbers(), np.int32)
        hdf5_store = init_hdf5_store(
            save_to=output_file, batch_size=save_buffer,
            num_atoms=len(atoms) if out_idx is None else len(out_idx),
            num_box_entries=tuple(box.shape), exist_ok=overwrite, observables=[], num_features=None,
            so3lr_version=__version__, ensemble=ensemble, md_dt_fs=md_dt_ps * 1000.0, md_steps=md_steps,
            atomic_numbers=numbers if out_idx is None else numbers[out_idx],
            cell_a0=(np.zeros((3, 3)) if vacuum else np.diag(np.asarray(box))).astype(odt),
            input_file=s.get("input_file"),
            model_path=s.get("model_path"), dtype=odt, fractional_coordinates=frac_out)

    # perform_md's watchdog on the drift rate of the conserved quantity (meV/atom/ps),
    # measured from drift_warmup_ps on over at least drift_warmup_ps; warns once
    drift_threshold = float(s.get("energy_drift_threshold", 0.1))
    drift_warmup = float(s.get("drift_warmup_ps", 5.0))
    drift_ref, drift_warned = None, False

    logger.info("Starting MD simulation (flash engine)...")
    logger.info("Step\tE [eV]\tKE\tPE\tH\tTemp [K]" + ("\tBox [A]" if npt else "") + "\ttime/step [s]")
    logger.info(line(current_cycle * md_steps, *quantities(state, e0), 0.0))
    positions, momenta, boxes, obs = [], [], [], {}
    total_time, timed = 0.0, 0
    last_checkpoint = None
    for cycle in range(md_cycles):
        t0 = time.time()
        state, pe = advance(state, md_steps)
        jax.block_until_ready(state.position)
        per_step = (time.time() - t0) / md_steps
        if cycle > 0:
            total_time, timed = total_time + per_step, timed + 1
        current_cycle += 1
        ke, pe, h, temp = quantities(state, pe)
        if not np.all(np.isfinite([ke, pe, temp])):
            raise RuntimeError("MD simulation diverged: NaN/Inf detected in energy or temperature.")
        logger.info(line(current_cycle * md_steps, ke, pe, h, temp, per_step))
        if ensemble != "nvt-langevin" and not drift_warned and np.isfinite(h) and h != 0.0:
            t_ps = current_cycle * md_steps * md_dt_ps
            if drift_ref is None:
                if t_ps >= drift_warmup:
                    drift_ref = (h, t_ps)
            elif t_ps - drift_ref[1] >= drift_warmup:
                window = t_ps - drift_ref[1]
                rate = (h - drift_ref[0]) / len(atoms) / window * 1000.0
                if abs(rate) > drift_threshold:
                    logger.warning(
                        f"Conserved-quantity drift {rate:+.3f} meV/atom/ps exceeds {drift_threshold:.3f} "
                        f"meV/atom/ps at step {current_cycle * md_steps} (measured over {window:.1f} ps after a "
                        f"{drift_warmup:.1f} ps warmup). This usually means the integration is unstable; "
                        f"consider a smaller timestep or float64 precision. (Set energy_drift_threshold "
                        f"[meV/atom/ps] to tune, or drift_warmup_ps to change the equilibration window; some "
                        f"drift early in the run is normal.)")
                    drift_warned = True
        R, L1 = cartesian(state)
        positions.append(np.asarray(R / L1 if frac_out else R, odt))
        momenta.append(np.asarray(state.momentum, odt))
        boxes.append(np.asarray(L1, odt))            # orthorhombic: the box is its sides ([0]: none)
        if len(positions) >= save_buffer or cycle == md_cycles - 1:
            if output_format == "hdf5":
                positions, momenta, boxes, obs = write_to_hdf5(hdf5_store, momenta, positions, boxes, obs, out_idx)
            elif output_format == "extxyz":
                positions, momenta, boxes, obs = write_to_extxyz(
                    output_file, atoms, boxes, momenta, positions, obs, out_idx,
                    fractional_coordinates=frac_out, append=append_extxyz, overwrite=overwrite)
                append_extxyz = True
            if restart_save_path is not None:
                save_restart_checkpoints(state, L1, current_cycle, ensemble=ensemble,
                                         fractional_coordinates=npt, restart_save_path=restart_save_path)
                last_checkpoint = current_cycle
    if restart_save_path is not None and last_checkpoint != current_cycle:
        save_restart_checkpoints(state, cartesian(state)[1], current_cycle, ensemble=ensemble,
                                 fractional_coordinates=npt, restart_save_path=restart_save_path)
    logger.info("Results saved to: " + str(Path(output_file)))
    if timed:
        logger.info(f"Average time per step: {total_time / timed:.2e} seconds")
        logger.info(f"Average time per atom per step: {total_time / timed / len(atoms):.2e} seconds")
