"""Whether the flash engine can run a task: settings, model, box (or none), device.

Imports no kernels, so ``so3lr_md.run_on_flash`` can decide cheaply and fall back without
touching Pallas. Every check fails closed: a setting or a model hyperparameter flash
does not know makes it decline (with the reason), rather than run something it might
not compute.

The tasks (``TASKS``): ``md`` (``so3lr nve|nvt|nvt-langevin|npt``), ``opt`` (``so3lr opt``),
``eval`` (``so3lr eval``: energies and forces only) and ``ase`` (``calculator.FlashCalculator``).
"""
import dataclasses
import functools
import math
import os

import numpy as np

# NPT: the cell grid is laid out for a box this much smaller, so the box moves without
# a compile (and is laid out again if it leaves the margin)
NPT_BOX_MARGIN = 0.03
# free space (no box): the least room (A) the atoms' size may grow by before the engine's box
# and long-range reach are laid out again; a quarter of the size where that is more
# (``vacuum_box``), so atoms that keep growing are laid out again a few times only
VACUUM_PAD = 10.0
# cells of the engine's grid, at most (its cell_start: 1 GiB): rc-wide cells fill a box up to
# 645 rc wide (2900 A at 4.5 A), the box free space lays out around atoms that span about
# 1500 A (1000 A long range) to 2300 A (12 A)
MAX_CELLS = 1 << 28
ENGINES = ("auto", "flash", "standard")
# the AMD GPUs (gfx targets) flash is tested on: the MI355X
ROCM_TARGETS = ("gfx950",)
FLASH_MODES = ("auto", "fast", "default", "lean", "leanest", "minimal")     # engine.MODES and auto
TASKS = ("md", "opt", "eval", "ase")
# what so3lr eval can take from the engine
EVAL_TARGETS = ("energy", "forces")
# so3lr eval builds (and compiles) an engine per set of atoms: auto takes this many at most
# (the neighbour-list model compiles once for the whole dataset); engine flash any number
EVAL_AUTO_ENGINES = 16

# settings flash honours (aliases included)
HANDLED = {
    "input_file", "output_file", "log_file", "output_format", "overwrite", "output_atom_indices",
    "output_cartesian_positions", "enforce_cartesian_coordinates", "model", "model_path", "precision",
    "matmul_precision", "lr_cutoff", "dispersion_damping", "total_charge", "num_unpaired_electrons", "spin",
    "charge_embed_total_charge", "charge_equilibration_total_charge", "seed", "save_buffer", "md_dt", "dt",
    "md_T", "temperature", "init_T", "md_P", "pressure", "md_cycles", "md_steps", "ensemble",
    "nhc_chain_length", "nhc_chain", "nhc_steps", "nhc_thermo", "nhc_baro", "nhc_sy_steps", "npt_coupling",
    "langevin_removecmmotion", "langevin_thermo", "relax_before_run", "min_cycles", "min_steps",
    "min_start_dt", "min_max_dt", "min_n_min", "force_convergence", "force_conv", "kspace_electrostatics",
    "kspace_smearing", "kspace_spacing", "kspace_interp_nodes", "kspace_dispersion",
    "kspace_dispersion_smearing", "kspace_dispersion_spacing", "kspace_dispersion_interp_nodes",
    "restart_load_path", "restart_save_path", "restart_reset", "energy_drift_threshold", "drift_warmup_ps",
    "engine", "flash_mode", "flash_mixed", "flash_devices", "md_precision",
}
# settings that only concern the neighbour-list engine, other commands sharing a settings
# file (eval, tune-ewald), or act only together with a declined setting below (the
# PySAGES files with do_enhanced_sampling, the constraint mode with fixed atoms)
IRRELEVANT = {
    "buffer_size_multiplier_sr", "buffer_sr", "buffer_size_multiplier_lr", "buffer_lr",
    "minimum_cell_size_multiplier", "jit_compile", "datafile", "save_to", "targets", "batch_size",
    "accuracy", "cutoff_lo", "cutoff_hi", "cutoff_it", "dispersion_damping_lo", "dispersion_damping_hi",
    "dispersion_damping_it", "mesh_lo", "mesh_hi", "nodes_lo", "nodes_hi", "ns_lo", "ns_hi", "smear_lo",
    "smear_hi", "smear_it", "help",
    "pysages_input", "restart_pysages_load_path", "restart_pysages_save_path", "fixed_atom_constraint",
}
# settings flash does not implement: it declines when one is active
UNSUPPORTED = {
    "do_enhanced_sampling": "enhanced sampling",
    "observables": "observables", "observable": "observables",
    "atom_mask_path": "fixed atoms", "mobile_atom_indices": "fixed atoms", "fixed_atom_indices": "fixed atoms",
    "dispersion_tail_correction": "the dispersion tail correction",
    "kspace_npt_cycles": "re-tuning the PME mesh every so many cycles (kspace_npt_cycles)",
}

# the model hyperparameters the kernels are written for (so3lr-tauri-s/m/l 2602 and 2606); the
# sizes below may vary; any other key, or any other value, makes flash decline (a key a model
# leaves out counts as the model builder's default, UNSET, or as missing)
ARCHITECTURE = {
    "legacy_so3lr_bool": False,
    "activation_fn": "silu", "energy_activation_fn": "silu", "cutoff_fn": "phys",
    "radial_basis_fn": "bernstein", "num_features_head": 32,
    "qk_non_linearity": "identity", "qk_norm": True, "layer_normalization_1": True,
    "layer_normalization_2": False, "residual_mlp_1": True, "residual_mlp_2": False,
    "message_normalization": "avg_num_neighbors", "use_charge_embed": True, "use_spin_embed": True,
    "use_final_bias_bool": False, "use_residual_scalars": True, "use_rms_norm": True,
    "zbl_repulsion_bool": True, "input_convention": "positions", "energy_learn_atomic_type_scales": True,
    "energy_learn_atomic_type_shifts": True, "electrostatic_energy_kspace_do_ewald_bool": False,
    "electrostatic_energy_kspace_interp_nodes": None,
}
# the legacy so3lr (legacy_so3lr_bool): the same attention; LayerNorm after the attention and
# the exchange, no q/k norm or residual scalars, a bias on the energy head (ZBL repulsion
# and one ratio head for both dispersion ratios come with the flag)
LEGACY_ARCHITECTURE = dict(
    {k: v for k, v in ARCHITECTURE.items() if not k.startswith("electrostatic_energy_kspace")},
    legacy_so3lr_bool=True, qk_norm=False, use_rms_norm=False, use_residual_scalars=False,
    layer_normalization_2=True, use_final_bias_bool=True)
# what so3lr's model builder takes for these when a model does not set them
UNSET = {"legacy_so3lr_bool": False, "qk_norm": False, "use_rms_norm": False, "use_residual_scalars": False}
# (degrees, heads, features) checked against the neighbour-list potential: one head per
# degree entry, 32 features each (so3lr-tauri-s/m, and so3lr-tauri-l with each degree twice)
SHAPES = {((1, 2, 3, 4), 4, 128), ((1, 1, 2, 2, 3, 3, 4, 4), 8, 256)}
SIZES = {
    "cutoff", "num_layers", "num_radial_basis_fn", "energy_regression_dim", "electrostatic_energy_bool",
    "electrostatic_energy_scale", "dispersion_energy_bool", "dispersion_energy_scale", "cutoff_lr",
    "cutoff_lr_damping", "dispersion_energy_cutoff_lr_damping",
    # initialisation only
    "layers_behave_like_identity_fn_at_init", "output_is_zero_at_init",
}


def _off(value):
    return value is None or value is False or (hasattr(value, "__len__") and len(value) == 0)


def validate_options(settings):
    """Raise on an engine, flash_mode, flash_mixed or flash_devices value flash does not take
    (a settings file bypasses the CLI's choices)."""
    for key, allowed in (("engine", ENGINES), ("flash_mode", FLASH_MODES)):
        if settings.get(key) is not None and settings[key] not in allowed:
            raise ValueError(f"{key} must be one of {', '.join(allowed)} (received {settings[key]!r})")
    if settings.get("flash_mixed") is not None and not isinstance(settings["flash_mixed"], bool):
        raise ValueError(f"flash_mixed must be true or false (received {settings['flash_mixed']!r})")
    n = settings.get("flash_devices")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
        raise ValueError(f"flash_devices must be a positive integer (received {n!r})")
    if (n or 1) > 1 and (settings.get("engine") or "auto") != "standard":
        # several GPUs run the jitted fast mode (the staged modes are for one GPU's most atoms)
        if (settings.get("flash_mode") or "auto") not in ("auto", "fast"):
            raise ValueError(f"flash_devices {n} runs flash_mode fast, not {settings['flash_mode']}")
        # asked for: an error here, not a fall back to the neighbour-list engine on one
        import jax
        if n > len(jax.local_devices()):
            raise ValueError(f"flash_devices {n}: only {len(jax.local_devices())} visible")


def reserve_gpu(settings, task="md"):
    """Let XLA take 97% of the GPU (unless set) when flash may run these settings and this
    model: its largest systems need nearly all of it. Called before JAX starts, so before
    the structure is read: a run that falls back for its structure still reserves it."""
    if ((settings.get("engine") or "auto") != "standard" and settings_problem(settings, task) is None
            and model_problem(settings.get("model_path")) is None):
        # several GPUs: room outside XLA's pool for NCCL's buffers
        n = settings.get("flash_devices")
        several = isinstance(n, int) and n > 1          # validate_options rejects other values
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90" if several else "0.97")


def settings_problem(settings, task="md"):
    """Why flash cannot run these settings for task (one of TASKS; None if it can)."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {', '.join(TASKS)} (received {task!r})")
    s = settings
    ens = s.get("ensemble")
    if task == "md" and ens not in ("nve", "nvt", "nvt-langevin", "npt"):
        return f"it runs nve, nvt, nvt-langevin and npt, not {ens}"
    if task == "eval":
        other = [t for t in _targets(s.get("targets")) if t not in EVAL_TARGETS]
        if other:
            return f"it evaluates {' and '.join(EVAL_TARGETS)}, not {', '.join(other)}"
    if s.get("lr_cutoff", 12.0) is None:
        return "it needs a long-range cutoff"
    kspace = s.get("kspace_electrostatics") or s.get("kspace_dispersion")
    for key, value in s.items():
        if key in UNSUPPORTED:
            # the tail correction and the mesh re-tuning act only in NPT (the re-tuning
            # only with k-space); perform_md re-tunes every 100 cycles when the key is
            # absent, flash whenever the box leaves its margin
            npt_only = key in ("dispersion_tail_correction", "kspace_npt_cycles")
            idle = (npt_only and ens != "npt") or (key == "kspace_npt_cycles" and not kspace)
            if not _off(value) and not idle:
                return f"it has no {UNSUPPORTED[key]}"
        elif key not in HANDLED and key not in IRRELEVANT and not _off(value):
            return f"it does not know the setting {key!r}"
    if ens == "npt" and (s.get("npt_coupling") or "isotropic") != "isotropic":
        return f"its NPT is isotropic, not {s.get('npt_coupling')}"
    for key, nodes in (("kspace_electrostatics", "kspace_interp_nodes"),
                       ("kspace_dispersion", "kspace_dispersion_interp_nodes")):
        if s.get(key) not in (None, "pme"):
            return f"it has PME but not {s.get(key)} ({key})"
        if s.get(key) and int(s.get(nodes) or 4) != 4:
            return f"its PME interpolates with 4 nodes, not {s.get(nodes)}"
    if ens == "npt" and not _jax_md_keeps_dudv():
        return "its NPT needs a jax-md whose NPT state carries dUdV from the previous step"
    return None


def _targets(targets):
    """so3lr eval's targets, a comma-separated string or a sequence, as a list."""
    if isinstance(targets, str):
        targets = targets.split(",")
    return [t.strip() for t in targets or () if t.strip()]


def _jax_md_keeps_dudv():
    """Flash's NPT finds each step's box and positions by running the jax-md step with
    zero force and strain derivative, which holds only if the step's first half uses
    the previous step's dUdV from the state (the jax-md pinned by so3lr does)."""
    import jax_md
    return "dUdV" in {f.name for f in dataclasses.fields(jax_md.simulate.NPTNoseHooverState)}


def model_problem(model_path):
    """Why flash cannot evaluate this model (None if it can)."""
    from so3lr.model_registry import resolve_model
    spec = resolve_model(model_path)
    from so3lr.model_config import load_model_config
    model = load_model_config(spec.dir)["model"]
    arch = LEGACY_ARCHITECTURE if model.get("legacy_so3lr_bool") is True else ARCHITECTURE
    shape = (tuple(model.get("degrees") or ()), model.get("num_heads"), model.get("num_features"))
    if shape not in SHAPES:
        return (f"its kernels are written for degrees, heads and features {sorted(SHAPES)} "
                f"({spec.dir.name} has {shape})")
    for key, value in model.items():
        if key in ("degrees", "num_heads", "num_features"):
            continue
        if key in arch:
            if value != arch[key]:
                return f"its kernels are written for {key} = {arch[key]!r} ({spec.dir.name} has {value!r})"
        elif key not in SIZES:
            return f"it does not know the model hyperparameter {key!r} ({spec.dir.name})"
    missing = [k for k in arch if k not in model and (k not in UNSET or UNSET[k] != arch[k])]
    if missing:
        return f"{spec.dir.name} does not set {missing[0]!r}"
    return None


def cell_grid(L, rc, rc_lr):
    """The finest cell grid for an orthorhombic box of sides L, as (cells per side, kc),
    or None: cells at least rc wide (the short-range search looks one cell out), and
    2 kc + 1 cells per side, kc the cells the long range reaches each way (so its walk
    visits no cell twice). Where rc-wide cells leave too few per side, wider ones that
    need a shorter reach are taken."""
    for k in range(max(int(math.ceil(rc_lr / rc)), 1), 0, -1):
        nc = [int(math.floor(v / max(rc, rc_lr / k))) for v in L]
        if min(nc) < 3:
            continue
        kc = max(int(math.ceil(rc_lr / (v / n))) for v, n in zip(L, nc))
        if min(nc) >= 2 * kc + 1:
            return nc, kc
    return None


def grid_problem(L, rc, rc_lr):
    """Why the engine cannot hold the cell grid (``cell_grid``) of an orthorhombic box of
    sides L that ``box_problem`` takes (None if it can): fewer than MAX_CELLS cells."""
    grid = cell_grid(L, rc, rc_lr)
    if grid is not None and math.prod(grid[0]) >= MAX_CELLS:
        return (f"box {[round(v) for v in L]} A too large for the cell grid "
                f"({' x '.join(map(str, grid[0]))} cells, 2^28 or more)")
    return None


def box_problem(L, rc, rc_lr):
    """Why the engine cannot take an orthorhombic box of sides L (None if it can): the
    minimum image needs 2 x the cutoffs, and the cell grid (``cell_grid``) must exist."""
    if min(L) < 2.0 * max(rc, rc_lr):
        return f"box {[round(v, 2) for v in L]} A narrower than twice the cutoff {max(rc, rc_lr)} A"
    if cell_grid(L, rc, rc_lr) is None:
        return f"box {[round(v, 2) for v in L]} A too small for the cell grid"
    return None


def in_vacuum(atoms):
    """Whether atoms are in free space, as perform_md takes them: not periodic, or periodic
    with no cell."""
    return not atoms.pbc.any() or (atoms.pbc.all() and not np.any(atoms.cell[:]))


def vacuum_size(R, xp=np):
    """Twice the largest distance of positions R from their centroid: no two atoms are
    farther apart, and it stays the same as they move together (xp: numpy or jax.numpy)."""
    d = R - R.mean(0)
    return 2.0 * xp.sqrt(xp.max(xp.sum(d * d, -1)))


def vacuum_box(size, rc, rc_lr):
    """(cube side, long-range reach) the engine takes free space in, for atoms of
    ``vacuum_size`` size with room to grow by (``VACUUM_PAD``): the reach takes every pair
    within rc_lr (every pair, when the atoms are smaller than that), and the periodic box
    is so wide that no atom is within the reach of another's image (``vacuum_fits``), and
    wide enough for the cell grid."""
    pad = max(VACUUM_PAD, 0.25 * size)
    reach = min(rc_lr, size + pad)
    L = size + pad + max(rc, reach)
    while box_problem([L] * 3, rc, reach):
        L += 0.25 * rc
    return L, reach


def vacuum_fits(size, L, rc, rc_lr, reach):
    """Whether atoms of size (``vacuum_size``) compute as in free space in the box L with
    the long-range reach (``vacuum_box``): every image is farther than both cutoffs (rc,
    the reach), and the reach takes every pair within rc_lr."""
    ok = size <= L - max(rc, reach)
    return ok if reach >= rc_lr else ok & (size <= reach)


def vacuum_problem(settings):
    """Why flash cannot run these settings in free space (None if it can)."""
    if settings.get("ensemble") == "npt":
        return "its NPT needs a box"
    for key in ("kspace_electrostatics", "kspace_dispersion"):
        if settings.get(key):
            return f"its k-space ({key}) needs a box"
    if _restart_box(settings) is not None:
        return "the restart has a box and the structure none"
    return None


def _restart_box(settings):
    """The box a restart starts in (its checkpoint's), None with no restart or no box."""
    if not settings.get("restart_load_path"):
        return None
    with np.load(settings["restart_load_path"]) as saved:
        box = np.asarray(saved["box"]) if "box" in saved.files else None
    return box if box is not None and np.any(box) else None


def structure_problem(settings, atoms, rc):
    """Why flash cannot take this structure, or the restart's box (None if it can); rc: the
    model's cutoff (``model_cutoff``). A structure in free space (``in_vacuum``) runs in a
    box the engine lays out around it (``vacuum_box``), whose cell grid one atom far from
    the rest (a dimer translated apart) makes too large."""
    rc_lr = float(settings.get("lr_cutoff") or 12.0)
    if in_vacuum(atoms):
        problem = vacuum_problem(settings)
        if problem is None:
            size = vacuum_size(np.asarray(atoms.get_positions(), np.float64))
            L, reach = vacuum_box(size, rc, rc_lr)
            problem = grid_problem([L] * 3, rc, reach)
            if problem:
                problem = f"its atoms span {size:.0f} A (one far from the rest?), a free-space {problem}"
        return problem
    if not atoms.pbc.all() or atoms.cell.rank < 3:
        return "it needs a box periodic in all three directions, or no box"
    cell = np.asarray(atoms.cell[:])
    if np.abs(cell - np.diag(np.diag(cell))).max() > 1e-9:
        return "it needs an orthorhombic box"
    sides = np.diag(cell)
    box = _restart_box(settings)
    if box is not None:                                  # a restart starts in its own box
        sides = np.diag(box) if box.ndim == 2 else np.broadcast_to(box.reshape(-1), (3,))
    shrink = 1.0 - (NPT_BOX_MARGIN if settings.get("ensemble") == "npt" else 0.0)
    L = [float(v) * shrink for v in sides]
    return box_problem(L, rc, rc_lr) or grid_problem(L, rc, rc_lr)


@functools.cache
def is_rocm():
    """Whether JAX runs on AMD's ROCm (HIP) here."""
    import jax
    try:
        version = str(jax.devices()[0].client.platform_version).lower()
    except Exception:
        return False
    return "rocm" in version or "hip" in version


def device_problem():
    """Why this process has no GPU the kernels (Pallas-Triton) run on, or None."""
    import jax
    try:
        dev = jax.devices()[0]
    except Exception as exc:                     # no backend at all
        return f"no JAX device ({exc})"
    if dev.platform != "gpu":
        return f"its kernels run on NVIDIA and AMD GPUs, and this process has {dev.platform}"
    cc = getattr(dev, "compute_capability", None)
    if is_rocm():
        # as measured on the MI355X: XLA's command buffers off (HIP graph capture fails)
        if str(cc).split(":")[0] not in ROCM_TARGETS:
            return f"on AMD its kernels are tested on {', '.join(ROCM_TARGETS)} (MI355X), and this GPU is {cc}"
        if "--xla_gpu_enable_command_buffer=" not in os.environ.get("XLA_FLAGS", ""):
            return 'on AMD it needs XLA_FLAGS="--xla_gpu_enable_command_buffer=" (set before JAX starts)'
        return None
    try:
        if cc is not None and float(cc) < 8.0:
            return f"its kernels are tested on compute capability 8.0 and newer (A100, H100), and this GPU is {cc}"
    except ValueError:
        return f"it does not know this GPU (compute capability {cc!r})"
    return None


def model_cutoff(model_path):
    """The model's short-range cutoff in A."""
    from so3lr.model_config import load_model_config
    from so3lr.model_registry import resolve_model
    return float(load_model_config(resolve_model(model_path).dir)["model"]["cutoff"])


def frames_problem(settings, frames):
    """Why flash cannot take one of these structures (None if it can take them all): each
    distinct box checked once, and in free space each structure that spans more than those
    before it (the box laid out around the atoms grows with their span)."""
    rc, seen, span = model_cutoff(settings.get("model_path")), set(), -1.0
    for i, atoms in enumerate(frames):
        if in_vacuum(atoms):
            size = vacuum_size(np.asarray(atoms.get_positions(), np.float64))
            if size <= span:
                continue
            span = size
        else:
            key = (tuple(bool(p) for p in atoms.pbc), np.asarray(atoms.cell[:]).tobytes())
            if key in seen:
                continue
            seen.add(key)
        problem = structure_problem(settings, atoms, rc)
        if problem:
            return problem if len(frames) == 1 else f"structure {i}: {problem}"
    return None


def flash_problem(settings, atoms, task="md"):
    """Why the flash engine cannot run task (one of TASKS) on atoms, a structure or a list of
    them (None if it can): the settings, the model and the structures first (no device
    needed), then the device."""
    frames = atoms if isinstance(atoms, (list, tuple)) else [atoms]
    return (settings_problem(settings, task) or model_problem(settings.get("model_path"))
            or frames_problem(settings, frames) or _engines_problem(settings, frames, task) or device_problem())


def _engines_problem(settings, frames, task):
    """Why auto does not run so3lr eval on flash for these structures: too many engines."""
    if task != "eval" or (settings.get("engine") or "auto") != "auto":
        return None
    n = len({atoms.numbers.tobytes() for atoms in frames})
    if n > EVAL_AUTO_ENGINES:
        return (f"the structures have more than {EVAL_AUTO_ENGINES} different sets of atoms, an engine "
                f"to build and compile for each (auto takes up to {EVAL_AUTO_ENGINES}; --engine flash "
                f"runs any number)")
    return None
