"""The per-atom pieces of SO3LR the flash engine applies in atom tiles, written on plain
parameter dicts (the tree ``load_params`` returns). They match the corresponding
modules of ``so3lr.mlff``; the engine's kernels do the per-edge work."""
import jax
import jax.numpy as jnp


def functional_rms_norm(x, *, eps: float = 1.0e-6):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + jnp.asarray(eps, dtype=x.dtype))


def degree_repeats(degrees):
    sizes = [2 * int(d) + 1 for d in degrees]
    return jnp.asarray(sizes), int(sum(sizes))


def l0_contraction(ev, degrees):
    out = []
    offset = 0
    for d in degrees:
        n = 2 * int(d) + 1
        block = ev[..., offset:offset + n]
        out.append(jnp.sum(block ** 2, axis=-1) / jnp.sqrt(jnp.asarray(float(n), dtype=ev.dtype)))
        offset += n
    return jnp.stack(out, axis=-1)


def linear(p, x, *, use_bias=True):
    y = x @ p["kernel"]
    if use_bias and ("bias" in p) and (p["bias"] is not None):
        y = y + p["bias"]
    return y


def mlp(p, names, x, *, use_bias=True, final_use_bias=True, last_linear=True):
    n = len(names)
    for i, name in enumerate(names):
        is_last = i == n - 1
        x = linear(p[name], x, use_bias=final_use_bias if is_last else use_bias)
        if not is_last or not last_linear:
            x = jax.nn.silu(x)
    return x


def scalar_equivariant_exchange(p, scalar_features, equivariant_features, *, degrees):
    repeats, total = degree_repeats(degrees)
    features = jnp.concatenate([scalar_features, l0_contraction(equivariant_features, degrees)], axis=-1)
    updates = mlp(p, ("mlp_layer_2",), features, last_linear=True)  # a single linear layer
    scalar_update, equivariant_update = jnp.split(updates, [scalar_features.shape[-1]], axis=-1)
    equivariant_update = jnp.repeat(equivariant_update, repeats=repeats, axis=-1, total_repeat_length=total)
    return scalar_update, equivariant_update * equivariant_features


def local_node_energy(p, x, atomic_numbers, gate, *, head_index):
    energy_offset = p["energy_offset"][atomic_numbers]
    atomic_scales = p["atomic_scales"][atomic_numbers]
    atomic_energy = mlp(p, ("energy_dense_regression", "energy_dense_final"), x, final_use_bias=False)
    atomic_energy = atomic_energy * atomic_scales
    atomic_energy = atomic_energy * gate[:, None]
    atomic_energy = atomic_energy + energy_offset
    return atomic_energy[:, head_index]


def ratio_head(p, names, x, atomic_numbers, gate):
    shift = p["Embed_0"]["embedding"][atomic_numbers].squeeze(axis=-1)
    ratio = jnp.abs(mlp(p, names, x).squeeze(axis=-1) + shift)
    return gate * ratio + (1.0 - gate)
