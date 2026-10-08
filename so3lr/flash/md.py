"""SO3LR-Flash in ``so3lr nve|nvt|nvt-langevin|npt``: jax-md dynamics with the flash engine.

``so3lr_md.run`` runs the flash engine whenever it computes what the settings ask for (``flash_problem``
is None: an NVIDIA GPU, a periodic orthorhombic box, nve / nvt / nvt-langevin /
isotropic npt, no k-space other than PME, a model with 4 heads), else the
neighbour-list potential of ``perform_md``. ``--engine flash`` or ``--engine standard``
forces one.

NPT is jax-md's npt_nose_hoover as ``perform_md`` sets it up (fractional positions, the
same barostat and thermostat chains), with the force and dE/d(strain) from the flash engine: the box
and positions a step reaches depend only on the previous step's force and strain
derivative, so each step runs the jax-md step with zeros to find them, evaluates the flash engine
there (in that box: ``set_box``), and runs the step again with the results, passed
through an energy whose gradients they are (``_surrogate``).

Same settings, units, log lines, trajectory and restart files as ``perform_md``; the
forces come from ``FlashSO3LR`` instead of the neighbour-list potential. Positions are
Cartesian (restart files from ``perform_md`` are converted on load).

The engine runs in one of the memory modes of ``engine.MODES``. A jitted mode runs
``md_steps`` steps per jitted cycle like ``perform_md``. A staged mode runs the force
as separately jitted stages, which cannot be traced into a jax-md step, so each step
runs the jax-md step twice: once with zero forces to get the new positions (they do
not depend on the new forces), then with the forces at those positions.
"""
import json
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
    _validate_md_output_write_policy,
    atoms_to_jnp,
    box_for_trajectory,
    handle_units,
    init_hdf5_store,
    load_state,
    save_restart_checkpoints,
    setup_jax_globals,
    setup_logger,
    validate_and_log_input,
    write_to_extxyz,
    write_to_hdf5,
)
from .engine import ASE_UNITS, MODES, FlashSO3LR, box_problem, pick_mode, pme_bytes
from .loader import load_params

logger = logging.getLogger("SO3LR")

# NPT: the cell grid is laid out for a box this much smaller, so the box moves without
# a compile (and is laid out again if it leaves the margin)
NPT_BOX_MARGIN = 0.03

_UNSUPPORTED = {
    "do_enhanced_sampling": "enhanced sampling",
    "observables": "observables",
    "atom_mask_path": "fixed atoms",
    "mobile_atom_indices": "fixed atoms",
    "fixed_atom_indices": "fixed atoms",
}


def _device_problem():
    """Why this process has no GPU the flash kernels (Pallas-Triton) run on, or None."""
    try:
        dev = jax.devices()[0]
    except Exception as exc:                     # no backend at all
        return f"no JAX device ({exc})"
    if dev.platform != "gpu":
        return f"its kernels run on NVIDIA GPUs, and this process has {dev.platform}"
    try:
        version = str(dev.client.platform_version).lower()
    except Exception:
        version = ""
    if "rocm" in version or "hip" in version:
        return "its kernels are tested on NVIDIA GPUs only, and this is ROCm"
    return None


def flash_problem(settings, atoms):
    """Why the flash engine cannot run this simulation (None if it can)."""
    s = settings
    device = _device_problem()
    if device:
        return device
    ens = s.get("ensemble")
    if ens not in ("nve", "nvt", "nvt-langevin", "npt"):
        return f"it runs nve, nvt, nvt-langevin and npt, not {ens}"
    if ens == "npt":
        if (s.get("npt_coupling") or "isotropic") != "isotropic":
            return f"its NPT is isotropic, not {s.get('npt_coupling')}"
        if s.get("dispersion_tail_correction"):
            return "it has no dispersion tail correction"
    for key, what in _UNSUPPORTED.items():
        if s.get(key):
            return f"it has no {what}"
    for key, nodes in (("kspace_electrostatics", "kspace_interp_nodes"),
                       ("kspace_dispersion", "kspace_dispersion_interp_nodes")):
        if s.get(key) not in (None, "pme"):
            return f"it has PME but not {s.get(key)} ({key})"
        if s.get(key) and int(s.get(nodes) or 4) != 4:
            return f"its PME interpolates with 4 nodes, not {s.get(nodes)}"
    from so3lr.model_registry import resolve_model
    spec = resolve_model(s.get("model_path"))
    with open(spec.dir / "hyperparameters.json") as fp:
        m = json.load(fp)["model"]
    if m.get("legacy_so3lr_bool", False):
        return "it runs the v2 models (so3lr-s, so3lr-m), not the legacy so3lr"
    if int(m["num_heads"]) != 4 or tuple(m["degrees"]) != (1, 2, 3, 4):
        return f"its kernels have 4 heads and degrees 1-4 ({spec.dir.name} has {m['num_heads']} heads)"
    if not spec.from_file:
        return f"it reads params.pkl, which {spec.dir} does not have"
    if not atoms.pbc.all() or atoms.cell.rank < 3:
        return "it needs a box periodic in all three directions"
    cell = np.asarray(atoms.cell[:])
    if np.abs(cell - np.diag(np.diag(cell))).max() > 1e-9:
        return "it needs an orthorhombic box"
    shrink = 1.0 - (NPT_BOX_MARGIN if ens == "npt" else 0.0)
    return box_problem([float(v) * shrink for v in np.diag(cell)], float(m["cutoff"]),
                       float(s.get("lr_cutoff") or 12.0))


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


def build_engine(atoms, settings, dtype):
    """FlashSO3LR for ``atoms`` with the model, charges, PME and memory mode of
    ``settings``."""
    precision = "float64" if dtype == jnp.float64 else "float32"
    params, statics = load_params(settings.get("model_path"), precision=precision,
                                  lr_cutoff=settings.get("lr_cutoff", 12.0),
                                  dispersion_damping=settings.get("dispersion_damping", 2.0))
    n = len(atoms)
    L = [float(v) for v in np.diag(np.asarray(atoms.cell[:]))]
    pme_el, pme_disp = _pme(settings)
    mode = settings.get("flash_mode") or "auto"
    if mode == "auto":
        free, _ = _free_and_total()
        mode = pick_mode(n, free - pme_bytes(L, (pme_el, pme_disp), jnp.dtype(dtype).itemsize))
    total_charge = float(settings.get("total_charge") or 0)
    embed = settings.get("charge_embed_total_charge")
    equil = settings.get("charge_equilibration_total_charge")
    kw, staged = MODES[mode][0], MODES[mode][1]
    common = dict(clash_guard="floor", units=ASE_UNITS,
                  total_charge=total_charge if embed is None else float(embed),
                  total_spin=float(settings.get("num_unpaired_electrons") or 0),
                  equilibration_charge=total_charge if equil is None else float(equil))
    eng = FlashSO3LR(params, statics, atoms.get_atomic_numbers(), np.asarray(atoms.cell[:]),
                    pme_electrostatics=pme_el, pme_dispersion=pme_disp,
                    box_margin=NPT_BOX_MARGIN if settings.get("ensemble") == "npt" else 0.0, **common, **kw)
    logger.info(f"Flash: {n} atoms, memory mode {mode} ({'staged' if staged else 'jitted'}), "
                f"~{MODES[mode][2]:.1f} KiB/atom at peak")
    for what, p, mesh in (("electrostatics", pme_el, eng.mesh_el), ("dispersion", pme_disp, eng.mesh_disp)):
        if p:
            logger.info(f"Flash: PME {what}, smearing {p[0]} A, mesh {mesh}")
    return eng, staged


def _free_and_total():
    try:
        st = jax.local_devices()[0].memory_stats()
        total = st.get("bytes_limit", 0)
        return total - st.get("bytes_in_use", 0), total
    except Exception:
        return 80 * 2 ** 30, 80 * 2 ** 30


def _force_fn(eng):
    """Forces for jax-md: ``F`` passed in (staged), or the jitted engine with NaN on an
    overflowed table, which the cycle loop catches and repeats with wider tables."""
    def force_fn(R, F=None, **kwargs):
        if F is not None:
            return F
        _, f, over, _ = eng._energy_forces_in_box(R)
        return jnp.where(over, jnp.nan, f)
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


def relax_flash(eng, staged, position, shift, settings):
    """FIRE with the settings of ``perform_min``; returns the relaxed positions."""
    force_fn = _force_fn(eng)
    init, apply = jax_md.minimize.fire_descent(
        force_fn, shift, dt_start=settings.get("min_start_dt", 0.05),
        dt_max=settings.get("min_max_dt", 0.1), n_min=settings.get("min_n_min", 2))
    min_cycles, min_steps = settings.get("min_cycles", 10), settings.get("min_steps", 10)
    fconv = settings.get("force_convergence")
    if staged:
        F = eng.energy_forces_staged(position)[1]
        state = jax.jit(lambda R, F: init(R, F=F))(position, F)
        apply_j = jax.jit(lambda s, F: apply(s, F=F))
    else:
        state = jax.jit(init)(position)
    logger.info("Starting minimization (flash engine)...")
    logger.info("Step\tE [eV]\tF_max [eV/Å]")
    for c in range(min_cycles + 1):
        e, f = (eng.energy_forces_staged if staged else eng.energy_forces)(state.position)
        f_max = float(jnp.max(jnp.linalg.norm(f, axis=-1)))
        logger.info(f"{c * min_steps}\t {float(e):.3f}\t {f_max:.3f}")
        if c == min_cycles or (fconv is not None and f_max <= fconv):
            break
        if staged:
            for _ in range(min_steps):
                R1 = apply_j(state, jnp.zeros_like(state.position)).position
                state = apply_j(state, eng.energy_forces_staged(R1)[1])
        else:
            state = _run_jitted(eng, apply, state, min_steps)
    return state.position


def _run_jitted(eng, step, state, k):
    """k jitted steps; repeated with wider tables if a step overflowed."""
    cache = eng.__dict__.setdefault("_md_cycles", {})
    while True:
        key = (step, k, eng.K, eng.maxh)
        if key not in cache:
            cache[key] = jax.jit(lambda s: jax.lax.fori_loop(0, k, lambda i, x: step(x), s))
        new = jax.block_until_ready(cache[key](state))
        if bool(jnp.all(jnp.isfinite(new.position))):
            return new
        eng.K = int(np.ceil(eng.K * 1.25 / 16) * 16)
        eng.maxh = int(eng.maxh * 1.5)
        eng._build_jit()
        logger.info(f"Flash: tables widened (row {eng.K}, scratch {eng.maxh}); cycle repeated")


def perform_md_flash(all_settings, restart=False, atoms=None):
    """``perform_md`` with the flash engine (``atoms``: the input structure, if read)."""
    s = all_settings
    setup_jax_globals(precision=s.get("precision", "float32"),
                      matmul_precision=s.get("matmul_precision", "highest"))
    setup_logger(s.get("log_file"))
    atoms = read(s["input_file"]) if atoms is None else atoms
    problem = flash_problem(s, atoms)
    if problem:
        raise ValueError(f"flash: {problem}")
    ensemble = s.get("ensemble")

    output_file, output_format = s.get("output_file"), s.get("output_format")
    overwrite = bool(s.get("overwrite", False))
    restart_load_path, restart_save_path = s.get("restart_load_path"), s.get("restart_save_path")
    append_extxyz = _validate_md_output_write_policy(
        output_file=output_file, output_format=output_format, overwrite=overwrite,
        restart_load_path=restart_load_path if restart else None)

    validate_and_log_input(atoms, s["input_file"], total_charge=s.get("total_charge"),
                           num_unpaired_electrons=s.get("num_unpaired_electrons", 0), verbose=False)
    atoms.wrap()
    precision = s.get("precision", "float32")
    dtype = jnp.float64 if precision == "float64" else jnp.float32
    # the MD state (positions, momenta, energy sums) in float64 around a float32 model
    # unless flash_mixed is off: float32 coordinates of a 500 A box round to 6e-5 A
    mixed = dtype == jnp.float32 and bool(s.get("flash_mixed", True))
    sdt = jnp.float64 if mixed else dtype
    geo = atoms_to_jnp(atoms, precision=sdt)
    L = np.diag(np.asarray(atoms.cell[:]))
    box = jnp.asarray(L, sdt)
    _, shift = jax_md.space.periodic(box)
    eng, staged = build_engine(atoms, s, dtype)
    logger.info(f"Flash precision: model {precision}, MD state {'float64 (mixed)' if mixed else precision}")
    position = geo["positions"]

    md_dt_ps = s.get("md_dt")
    md_T = s.get("md_T")
    if ensemble == "nve" and s.get("init_T") is not None:
        md_T = s.get("init_T")
    unit = handle_units(units.metal_unit_system, md_dt_ps, md_T)
    md_dt, kT = sdt(unit["dt"]), sdt(unit["T"])
    md_cycles, md_steps = s.get("md_cycles"), s.get("md_steps")
    save_buffer = min(md_cycles, s.get("save_buffer"))
    force_fn = _force_fn(eng)
    if ensemble == "nve":
        init_fn, apply_fn = jax_md.simulate.nve(force_fn, shift, md_dt)
    elif ensemble == "nvt":
        init_fn, apply_fn = jax_md.simulate.nvt_nose_hoover(
            force_fn, shift, md_dt, kT, chain_length=s.get("nhc_chain_length"),
            chain_steps=s.get("nhc_steps"), sy_steps=s.get("nhc_sy_steps"),
            tau=md_dt * s.get("nhc_thermo"))
    elif ensemble == "nvt-langevin":
        thermo = s.get("langevin_thermo")
        if thermo is None or thermo <= 0:
            raise ValueError(f"--langevin-thermo must be > 0 (received {thermo})")
        init_fn, apply_fn = jax_md.simulate.nvt_langevin(
            force_fn, shift, md_dt, kT, gamma=1.0 / (md_dt * thermo),
            center_velocity=s.get("langevin_removecmmotion"))
    npt = ensemble == "npt"
    if npt:
        # as perform_md: fractional positions, the box a vector of its sides
        from so3lr.cli.so3lr_md import box_to_diag
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

    current_cycle = 0
    state = None
    if restart:
        logger.info(f"Loading state from {restart_load_path}.")
        try:
            state, rbox, current_cycle = load_state(path_to_load=restart_load_path, ensemble=ensemble,
                                                    restart_reset=s.get("restart_reset", False),
                                                    fractional_coordinates=npt, precision=sdt)
            if npt:
                box = jnp.asarray(box_to_diag(rbox), sdt)
                eng.set_box(box)
        except RestartInNewEnsembleError as e:
            logger.warning(e)
            position = jnp.asarray(e.loaded_state["position"], sdt)
            if bool(e.loaded_state.get("fractional_coordinates", True)):
                position = position * box
            geo["masses"], geo["momenta"] = e.loaded_state["mass"], e.loaded_state["momentum"]
    elif s.get("relax_before_run"):
        position = relax_flash(eng, staged, position, shift, s)

    key = jax.random.PRNGKey(s.get("seed", 0))
    if npt:
        strain = eng.energy_forces_strain_staged if staged else eng.energy_forces_strain
        apply_j = jax.jit(lambda st, F, W, E: apply_fn(st, F=F, W=W, E=E))

        def advance(st, k):
            """k steps; the energy at the last positions too. The flash engine in the box each step
            reaches, at the positions it reaches (both from the previous force alone)."""
            zero = jnp.zeros((), sdt)
            for _ in range(k):
                s1 = apply_j(st, jnp.zeros_like(st.position), zero, zero)
                L1 = box_to_diag(jax_md.simulate.npt_box(s1))
                eng.set_box(L1)
                e, F, W = strain(s1.position * L1)
                st = apply_j(st, F.astype(sdt), jnp.asarray(W, sdt), jnp.asarray(e, sdt))
            return st, e
        if state is None:
            frac = position / box
            e0, F0, W0 = strain(position)
            state = init_fn(key, frac, box, mass=geo["masses"], momenta=geo.get("momenta"), kT=kT,
                            F=F0.astype(sdt), W=jnp.asarray(W0, sdt), E=jnp.asarray(e0, sdt))
    elif staged:
        apply_j = jax.jit(lambda st, F: apply_fn(st, F=F))

        def advance(st, k):
            """k steps; also returns the energy at the last positions."""
            for _ in range(k):
                R1 = apply_j(st, jnp.zeros_like(st.position)).position
                e, F = eng.energy_forces_staged(R1)
                st = apply_j(st, F)
            return st, e
        if state is None:
            F0 = eng.energy_forces_staged(position)[1]
            state = init_fn(key, position, kT=kT, mass=geo["masses"], momenta=geo.get("momenta"), F=F0)
    else:
        def advance(st, k):
            return _run_jitted(eng, apply_fn, st, k), None
        if state is None:
            eng.energy_forces(position)                     # sizes the tables for the start
            state = jax.jit(lambda R, m, p: init_fn(key, R, kT=kT, mass=m, momenta=p))(
                position, geo["masses"], geo.get("momenta"))

    unit_sys = units.metal_unit_system()

    def cartesian(st):
        """Positions in A and the box sides of a state."""
        if npt:
            L1 = box_to_diag(jax_md.simulate.npt_box(st))
            return st.position * L1, L1
        return st.position, box

    def quantities(st, pe=None):
        if pe is None:
            R, L1 = cartesian(st)
            if npt:
                eng.set_box(L1)
            pe = (eng.energy_forces_staged if staged else eng.energy_forces)(R)[0]
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
    hdf5_store = None
    if output_format == "hdf5":
        from so3lr import __version__
        hdf5_store = init_hdf5_store(
            save_to=output_file, batch_size=save_buffer, num_atoms=len(atoms),
            num_box_entries=(3,), exist_ok=overwrite, observables=[], num_features=None,
            so3lr_version=__version__, ensemble=ensemble, md_dt_fs=md_dt_ps * 1000.0, md_steps=md_steps,
            atomic_numbers=np.asarray(atoms.get_atomic_numbers(), np.int32),
            cell_a0=np.diag(L).astype(np.dtype(sdt)), input_file=s.get("input_file"),
            model_path=s.get("model_path"), dtype=np.dtype(sdt), fractional_coordinates=False)

    logger.info("Starting MD simulation (flash engine)...")
    logger.info("Step\tE [eV]\tKE\tPE\tH\tTemp [K]" + ("\tBox [A]" if npt else "") + "\ttime/step [s]")
    ke, pe, h, temp = quantities(state)
    logger.info(line(current_cycle * md_steps, ke, pe, h, temp, 0.0))
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
        R, L1 = cartesian(state)
        positions.append(np.asarray(R))
        momenta.append(np.asarray(state.momentum))
        boxes.append(np.asarray(box_for_trajectory(L1)))
        if len(positions) >= save_buffer or cycle == md_cycles - 1:
            if output_format == "hdf5":
                positions, momenta, boxes, obs = write_to_hdf5(hdf5_store, momenta, positions, boxes, obs,
                                                               s.get("output_atom_indices"))
            elif output_format == "extxyz":
                positions, momenta, boxes, obs = write_to_extxyz(
                    output_file, atoms, boxes, momenta, positions, obs, s.get("output_atom_indices"),
                    fractional_coordinates=False, append=append_extxyz, overwrite=overwrite)
                append_extxyz = True
            if restart_save_path is not None:
                save_restart_checkpoints(state, cartesian(state)[1], current_cycle, ensemble=ensemble,
                                         fractional_coordinates=npt, restart_save_path=restart_save_path)
                last_checkpoint = current_cycle
    if restart_save_path is not None and last_checkpoint != current_cycle:
        save_restart_checkpoints(state, cartesian(state)[1], current_cycle, ensemble=ensemble,
                                 fractional_coordinates=npt, restart_save_path=restart_save_path)
    logger.info("Results saved to: " + str(Path(output_file)))
    if timed:
        logger.info(f"Average time per step: {total_time / timed:.2e} seconds")
        logger.info(f"Average time per atom per step: {total_time / timed / len(atoms):.2e} seconds")
