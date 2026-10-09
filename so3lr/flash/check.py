"""Whether the flash engine can run a simulation: settings, model, box, device.

Imports no kernels, so ``so3lr_md.run`` can decide cheaply and fall back without
touching Pallas. Every check fails closed: a setting or a model hyperparameter flash
does not know makes it decline (with the reason), rather than run something it might
not compute.
"""
import dataclasses
import json
import math
import os

import numpy as np

# NPT: the cell grid is laid out for a box this much smaller, so the box moves without
# a compile (and is laid out again if it leaves the margin)
NPT_BOX_MARGIN = 0.03
ENGINES = ("auto", "flash", "standard")
FLASH_MODES = ("auto", "fast", "default", "lean", "leanest", "minimal")     # engine.MODES and auto

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
    "engine", "flash_mode", "flash_mixed",
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

# the model hyperparameters the kernels are written for (so3lr-s/m/l 2602 and 2606); the
# sizes below may vary; any other key, or any other value, makes flash decline
ARCHITECTURE = {
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
# (degrees, heads, features) checked against the neighbour-list potential: one head per
# degree entry, 32 features each (so3lr-s/m, and so3lr-l with each degree twice)
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
    """Raise on an engine, flash_mode or flash_mixed value flash does not take (a settings
    file bypasses the CLI's choices)."""
    for key, allowed in (("engine", ENGINES), ("flash_mode", FLASH_MODES)):
        if settings.get(key) is not None and settings[key] not in allowed:
            raise ValueError(f"{key} must be one of {', '.join(allowed)} (received {settings[key]!r})")
    if settings.get("flash_mixed") is not None and not isinstance(settings["flash_mixed"], bool):
        raise ValueError(f"flash_mixed must be true or false (received {settings['flash_mixed']!r})")


def reserve_gpu(settings):
    """Let XLA take 97% of the GPU (unless set) when flash may run these settings and this
    model: its largest systems need nearly all of it. Called before JAX starts, so before
    the structure is read: a run that falls back for its structure still reserves it."""
    if ((settings.get("engine") or "auto") != "standard" and settings_problem(settings) is None
            and model_problem(settings.get("model_path")) is None):
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.97")


def settings_problem(settings):
    """Why flash cannot run these settings (None if it can)."""
    s = settings
    ens = s.get("ensemble")
    if ens not in ("nve", "nvt", "nvt-langevin", "npt"):
        return f"it runs nve, nvt, nvt-langevin and npt, not {ens}"
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
    with open(spec.dir / "hyperparameters.json") as fp:
        model = json.load(fp)["model"]
    if model.get("legacy_so3lr_bool", False):
        return "it runs the v2 models (so3lr-s, -m, -l), not the legacy so3lr"
    shape = (tuple(model.get("degrees") or ()), model.get("num_heads"), model.get("num_features"))
    if shape not in SHAPES:
        return (f"its kernels are written for degrees, heads and features {sorted(SHAPES)} "
                f"({spec.dir.name} has {shape})")
    for key, value in model.items():
        if key in ("legacy_so3lr_bool", "degrees", "num_heads", "num_features"):
            continue
        if key in ARCHITECTURE:
            if value != ARCHITECTURE[key]:
                return (f"its kernels are written for {key} = {ARCHITECTURE[key]!r} "
                        f"({spec.dir.name} has {value!r})")
        elif key not in SIZES:
            return f"it does not know the model hyperparameter {key!r} ({spec.dir.name})"
    missing = [k for k in ARCHITECTURE if k not in model]
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


def box_problem(L, rc, rc_lr):
    """Why the engine cannot take an orthorhombic box of sides L (None if it can): the
    minimum image needs 2 x the cutoffs, and the cell grid (``cell_grid``) must exist."""
    if min(L) < 2.0 * max(rc, rc_lr):
        return f"box {[round(v, 2) for v in L]} A narrower than twice the cutoff {max(rc, rc_lr)} A"
    if cell_grid(L, rc, rc_lr) is None:
        return f"box {[round(v, 2) for v in L]} A too small for the cell grid"
    return None


def structure_problem(settings, atoms):
    """Why flash cannot take this structure, or the restart's box (None if it can)."""
    if not atoms.pbc.all() or atoms.cell.rank < 3:
        return "it needs a box periodic in all three directions"
    cell = np.asarray(atoms.cell[:])
    if np.abs(cell - np.diag(np.diag(cell))).max() > 1e-9:
        return "it needs an orthorhombic box"
    sides = np.diag(cell)
    if settings.get("restart_load_path"):                # a restart starts in its own box
        with np.load(settings["restart_load_path"]) as saved:
            box = np.asarray(saved["box"]) if "box" in saved.files else None
        if box is not None and np.any(box):
            sides = np.diag(box) if box.ndim == 2 else np.broadcast_to(box.reshape(-1), (3,))
    from so3lr.model_registry import resolve_model
    with open(resolve_model(settings.get("model_path")).dir / "hyperparameters.json") as fp:
        rc = float(json.load(fp)["model"]["cutoff"])
    shrink = 1.0 - (NPT_BOX_MARGIN if settings.get("ensemble") == "npt" else 0.0)
    return box_problem([float(v) * shrink for v in sides], rc, float(settings.get("lr_cutoff") or 12.0))


def device_problem():
    """Why this process has no GPU the kernels (Pallas-Triton) run on, or None."""
    import jax
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
    cc = getattr(dev, "compute_capability", None)
    if cc is not None and float(cc) < 8.0:
        return f"its kernels are tested on compute capability 8.0 and newer (A100, H100), and this GPU is {cc}"
    return None


def flash_problem(settings, atoms):
    """Why the flash engine cannot run this simulation (None if it can): the settings,
    the model and the structure first (no device needed), then the device."""
    return (settings_problem(settings) or model_problem(settings.get("model_path"))
            or structure_problem(settings, atoms) or device_problem())
