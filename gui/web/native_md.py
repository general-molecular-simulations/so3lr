"""Isolated Langevin dynamics with SO3LR forces, JAX-MD neighbors and JAX BAOAB."""
from functools import lru_cache, partial
import math

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from ase.data import atomic_masses
def _remove_com(velocity, masses):
    return velocity - jnp.sum(masses * velocity, axis=0) / jnp.sum(masses)


@lru_cache(maxsize=8)
def _thermostat(numbers, precision, device):
    dtype = jnp.float32 if precision == "float32" else jnp.float64
    masses = jnp.asarray(atomic_masses[np.asarray(numbers)], dtype=dtype)[:, None]
    dof = 3 * len(numbers) - 3
    def kinetic_temperature(v):
        return jnp.sum(masses * v**2) / (units.kB * dof)

    def velocities(r, key, kT):
        key, velocity_key = jax.random.split(key)
        v = _remove_com(jax.random.normal(velocity_key, r.shape, dtype) * jnp.sqrt(kT / masses), masses)
        return v, key

    def drift(r, v, force, key, dt, kT, decay, noise_factor):
        key, noise_key = jax.random.split(key)
        # B/A/O/A/B in ASE's eV, Å, u time units. Removing the
        # mass-weighted mean preserves the three constrained COM modes.
        v = _remove_com(v + (dt / 2) * force / masses, masses)
        r = r + (dt / 2) * v
        noise = jax.random.normal(noise_key, r.shape, dtype) * jnp.sqrt(kT / masses)
        v = decay * v + noise_factor * _remove_com(noise, masses)
        return r + (dt / 2) * v, v, key

    def kick(v, force, dt):
        v = _remove_com(v + (dt / 2) * force / masses, masses)
        return v, kinetic_temperature(v)

    return velocities, kinetic_temperature, drift, kick


@lru_cache(maxsize=4)
def _dynamics_kernel(numbers, model, charge, precision, device):
    # Create parameters on the selected device. Cache keys include species, charge
    # and placement because those are constant for an entire trajectory.
    from jax_md import space
    from so3lr import So3lrPotential, to_jax_md

    potential = So3lrPotential(model=model, lr_cutoff=100.0, dtype=jnp.float32,
                               output_intermediate_quantities=["partial_charges", "dipole_vec"])
    if not set(numbers).issubset(potential.supported_atomic_numbers):
        raise ValueError("The selected SO3LR model does not support all supplied elements.")
    neighbors, long_neighbors, energy = to_jax_md(
        potential, space.free()[0], box_size=None, species=jnp.asarray(numbers, jnp.int32),
        total_charge=charge, dr_threshold=0.5, disable_cell_list=True, fractional_coordinates=False)

    def potential_energy(r, short_idx, long_idx):
        per_atom, aux = energy(r, short_idx, long_idx, has_aux=True)
        return per_atom.astype(r.dtype).sum(), (aux["partial_charges"].reshape(-1), aux["dipole_vec"].reshape(3))

    def model_values(r, short_idx, long_idx):
        (e, (q, dipole)), grad = jax.value_and_grad(potential_energy, has_aux=True)(r, short_idx, long_idx)
        return -grad, e, q, dipole / units.Debye

    velocities, kinetic_temperature, drift, kick = _thermostat(numbers, precision, device)

    @jax.jit
    def initialize(r, key, kT, small, large):
        v, key = velocities(r, key, kT)
        f, e, q, d = model_values(r, small.idx, large.idx)
        return (r, v, f, key, small, large), (r, f, e, q, d, kinetic_temperature(v))

    @partial(jax.jit, static_argnames=("count",))
    def advance(state, dt, kT, decay, noise_factor, count):
        def step(carry, _):
            r, v, force, key, small, large = carry
            r, v, key = drift(r, v, force, key, dt, kT, decay, noise_factor)
            small = small.update(r)
            large = large.update(r)
            force, e, q, dipole = model_values(r, small.idx, large.idx)
            v, temperature = kick(v, force, dt)
            return (r, v, force, key, small, large), (r, force, e, q, dipole, temperature,
                                                       small.did_buffer_overflow, large.did_buffer_overflow)
        state, history = jax.lax.scan(step, state, xs=None, length=count)
        overflow = jnp.stack([history[-2].any(), history[-1].any()])
        return state, history[:-2], overflow

    return neighbors, long_neighbors, initialize, advance


def _snapshot(step, data, charge, force_backend="SO3LR JAX-MD"):
    r, force, energy, q, dipole, temperature = data
    for label, value in zip(("coordinates", "forces", "energy", "charges", "dipole", "temperature"), data):
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite {label} at step {step}. Check the timestep and geometry.")
    if abs(float(np.sum(q, dtype=np.float64)) - charge) > max(1e-4, 128 * len(q) * np.finfo(np.float32).eps):
        raise ValueError(f"Predicted atomic charges do not sum to the system charge at step {step}.")
    # The thermostat measures T from the instantaneous velocities with three fixed COM modes.
    kinetic_energy = 0.5 * (3 * len(r) - 3) * units.kB * float(temperature)
    return dict(step=step, positions=r, forces=force, energy=float(energy), kinetic_energy=kinetic_energy,
                force_backend=force_backend, partial_charges=q,
                dipole_debye=dipole if charge == 0 else None, temperature=float(temperature))


def run_dynamics(numbers, positions, *, model, charge, precision, temperature, timestep, seed, steps,
                 progress=None, block_size=16, sample_stride=None):
    """Yield initial, sampled and final snapshots on the active JAX device.

    Input/output geometry is in Å, forces in eV/Å, time in fs and temperature
    in K. CPU/CUDA integrate compiled blocks. Returning true from progress
    at a block boundary saves the last completed frame and stops.
    """
    if precision not in ("float32", "mixed") or (precision == "mixed" and not jax.config.x64_enabled):
        raise ValueError("Choose float32 or enable JAX x64 for mixed precision.")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("Use a positive integration block size.")
    stride = sample_stride if sample_stride is not None else max(1, math.ceil(steps / (100 if steps <= 2000 else 1000)))
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("Use a positive frame sampling interval.")
    numbers = tuple(int(number) for number in numbers)
    r = jnp.asarray(positions, dtype=jnp.float32 if precision == "float32" else jnp.float64)
    kT = jnp.asarray(units.kB * temperature, r.dtype)
    dt = jnp.asarray(timestep * units.fs, r.dtype)
    # OU friction is 0.01 per fs; expm1 preserves accuracy at small timesteps.
    decay = jnp.asarray(math.exp(-0.01 * timestep), r.dtype)
    noise_factor = jnp.asarray(math.sqrt(-math.expm1(-0.02 * timestep)), r.dtype)
    neighbors, long_neighbors, initialize, advance = _dynamics_kernel(numbers, model, charge, precision, r.device)
    small, large = neighbors.allocate(r), long_neighbors.allocate(r)
    state, initial = initialize(r, jax.random.PRNGKey(seed), kT, small, large)
    yield _snapshot(0, jax.device_get(initial), charge)
    step = 0
    if progress and progress(0, steps):
        return
    while step < steps:
        count = min(block_size, steps - step)
        next_state, history, overflow = advance(state, dt, kT, decay, noise_factor, count)
        sr_full, lr_full = jax.device_get(overflow)
        if sr_full or lr_full:
            # Discard the entire tentative block including its random keys.
            # At <=256 atoms full sparse capacity is bounded and guarantees a
            # complete list even for a structure that rapidly compacts.
            r, v, f, key, small, large = state
            if sr_full:
                small = neighbors.allocate(r, extra_capacity=len(numbers))
            if lr_full:
                large = long_neighbors.allocate(r, extra_capacity=len(numbers))
            state = r, v, f, key, small, large
            next_state, history, overflow = advance(state, dt, kT, decay, noise_factor, count)
            if jax.device_get(overflow).any():
                raise RuntimeError("Unable to allocate complete molecular neighbor lists.")
        stop = bool(progress and progress(step + count, steps))
        saved = [i for i in range(count) if (step + i + 1) % stride == 0 or step + i + 1 == steps]
        if stop and count - 1 not in saved:
            saved.append(count - 1)
        if saved:
            indices = jnp.asarray(saved)
            host = jax.device_get(tuple(values[indices] for values in history))
            for i, values in zip(saved, zip(*host)):
                yield _snapshot(step + i + 1, values, charge)
        state = next_state
        step += count
        if stop:
            break
