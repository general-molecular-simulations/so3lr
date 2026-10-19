"""Parameters and statics for the flash engine from a model in the registry (or a path)."""
import json

import jax
import jax.numpy as jnp
import numpy as np
from ase.units import alpha as FINE_STRUCTURE

from so3lr.mlff.nn.observable.dispersion_ref_data import C6_coef, NLH_AA, NLH_BB, alphas
from so3lr.mlff.utils.checkpoint_utils import load_params_from_workdir
from so3lr.mlff.utils.training_utils import expand_observables_to_theory_levels
from so3lr.model_registry import resolve_model


def load_params(model=None, *, precision="float32", lr_cutoff=12.0, dispersion_damping=2.0):
    """(params, statics) for ``FlashSO3LR``, read the way ``load_model`` reads them:
    params.pkl (or the checkpoint) expanded to all theory levels, the energy from
    ``spec.head_index``, the NLH and free-atom dispersion tables of ``dispersion_ref_data``."""
    spec = resolve_model(model)
    with open(spec.dir / "hyperparameters.json") as fp:
        cfg = json.load(fp)
    m = cfg["model"]
    if m.get("legacy_so3lr_bool", False):
        raise ValueError("the flash engine runs the v2 models (so3lr-tauri-s/m/l), not the legacy so3lr")
    raw = expand_observables_to_theory_levels(load_params_from_workdir(spec.dir))["params"]
    dt = jnp.float64 if precision == "float64" else jnp.float32

    def cast(x):
        a = np.asarray(x)
        return jnp.asarray(a, dt) if np.issubdtype(a.dtype, np.floating) else jnp.asarray(a)
    params = jax.tree_util.tree_map(cast, raw)
    params["nlh_aa"] = jnp.asarray(NLH_AA, dt)
    params["nlh_bb"] = jnp.asarray(NLH_BB, dt)
    statics = dict(
        cutoff=float(m["cutoff"]),
        long_range_cutoff=float(lr_cutoff),
        dispersion_energy_cutoff_lr_damping=float(dispersion_damping),
        num_layers=int(m["num_layers"]),
        num_features=int(m["num_features"]),
        num_radial_basis_fn=int(m["num_radial_basis_fn"]),
        degrees=tuple(int(d) for d in m["degrees"]),
        avg_num_neighbors=float(cfg["data"]["avg_num_neighbors"]),
        electrostatic_energy_scale=float(m["electrostatic_energy_scale"]) if m["electrostatic_energy_bool"] else 0.0,
        dispersion_energy_scale=float(m["dispersion_energy_scale"]) if m["dispersion_energy_bool"] else 0.0,
        fine_structure=float(FINE_STRUCTURE),
        num_heads=int(m["num_heads"]),
        qk_norm=bool(m.get("qk_norm", False)),
        ln1=bool(m.get("layer_normalization_1", False)),
        ln2=bool(m.get("layer_normalization_2", False)),
        head_index=int(spec.head_index),
        dispersion_alphas=jnp.asarray(alphas, dt),
        dispersion_c6=jnp.asarray(C6_coef, dt),
        dtype=dt,
    )
    return params, statics
