"""Parameters and statics for the flash engine from a model in the registry (or a path)."""
import jax.numpy as jnp
import numpy as np
from ase.units import alpha as FINE_STRUCTURE

from so3lr.mlff.nn.observable.dispersion_ref_data import (
    C6_coef, C6_coef_legacy, NLH_AA, NLH_BB, alphas, alphas_legacy)
from so3lr.mlff.calculators.potential import cast_float_params
from so3lr.mlff.utils.checkpoint_utils import load_params_from_workdir
from so3lr.mlff.utils.training_utils import expand_observables_to_theory_levels
from so3lr.model_config import load_model_config
from so3lr.model_registry import resolve_model


def zbl_tables(p, n):
    """The legacy so3lr's learnable ZBL screening in the NLH tables' form, A and B of
    sum_k A_k exp(-B_k r) for the element pairs [Z_i - 1, Z_j - 1] up to n, float64 (``p``:
    its zbl_repulsion parameters, each through a softplus as ``ZBLRepulsionSparse`` takes
    them): A_k = c_k / sum(c), B_k = a_k d (Z_i^p + Z_j^p)."""
    sp = {k: float(np.logaddexp(np.asarray(v, np.float64).reshape(-1)[0], 0.0)) for k, v in p.items()}
    a = np.array([sp[f"a{k}"] for k in range(1, 5)])
    c = np.array([sp[f"c{k}"] for k in range(1, 5)])
    zp = np.arange(1, n + 1, dtype=np.float64) ** sp["p"]
    b = (zp[:, None] + zp[None, :])[..., None] * (a * sp["d"])
    return np.broadcast_to(c / c.sum(), b.shape), b


def load_params(model=None, *, precision="float32", lr_cutoff=12.0, dispersion_damping=2.0):
    """(params, statics) for ``FlashSO3LR``, read the way ``load_model`` reads them:
    params.pkl (or the checkpoint) expanded to all theory levels, the energy from
    ``spec.head_index``, the repulsion's tables (NLH, or the legacy so3lr's ZBL in their
    form) and the free-atom dispersion tables of ``dispersion_ref_data`` (the legacy
    so3lr's own for it)."""
    spec = resolve_model(model)
    cfg = load_model_config(spec.dir)
    m = cfg["model"]
    legacy = m.get("legacy_so3lr_bool", False) is True
    raw = expand_observables_to_theory_levels(load_params_from_workdir(spec.dir))["params"]
    dt = jnp.float64 if precision == "float64" else jnp.float32
    params = cast_float_params(raw, dt)
    # the repulsion's tables, over the elements of the descriptor floor's (NLH's)
    rep = zbl_tables(raw["observables_0"]["zbl_repulsion"], NLH_AA.shape[0]) if legacy else (NLH_AA, NLH_BB)
    params["repulsion_a"], params["repulsion_b"] = (jnp.asarray(t, dt) for t in rep)
    if not m.get("use_residual_scalars", False):
        # the identity (the legacy so3lr has none): x_in = 1 x + 0 x0 = x
        params["resid_lambdas"] = jnp.ones((m["num_layers"],), dt)
        params["x0_lambdas"] = jnp.zeros((m["num_layers"],), dt)
    if not m.get("use_final_bias_bool", False):
        # as the model builder: no bias on the energy head, whatever the checkpoint holds
        params["observables_0"]["energy_dense_final"].pop("bias", None)
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
        # RMSNorm without a scale (so3lr-tauri-s/m/l) or LayerNorm (the legacy so3lr)
        rms_norm=bool(m.get("use_rms_norm", False)),
        legacy=legacy,
        head_index=int(spec.head_index),
        dispersion_alphas=jnp.asarray(alphas_legacy if legacy else alphas, dt),
        dispersion_c6=jnp.asarray(C6_coef_legacy if legacy else C6_coef, dt),
        dtype=dt,
    )
    return params, statics
