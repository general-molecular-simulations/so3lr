import jax
import jax.numpy as jnp

from so3lr.mlff.masking.mask import safe_scale

# Optional StackNetSparse inputs. Callers pass them by keyword; others are ignored.
_OPTIONAL_INPUTS = (
    'cell', 'cell_offset', 'cell_offset_lr', 'batch_segments', 'node_mask', 'graph_mask',
    'displacements', 'displacements_lr', 'total_charge', 'num_unpaired_electrons',
    'idx_i_lr', 'idx_j_lr', 'theory_mask', 'k_grid', 'k_smearing', 'k_grid_disp',
    'k_smearing_disp', 'residue_segments', 'residue_charge',
)


def _model_inputs(positions, atomic_numbers, idx_i, idx_j, **kwargs):
    """The StackNetSparse input dict. A single graph (no batch_segments) gets one
    graph segment and all-true masks."""
    inputs = {k: kwargs.get(k) for k in _OPTIONAL_INPUTS}
    inputs.update(positions=positions, atomic_numbers=atomic_numbers, idx_i=idx_i, idx_j=idx_j)
    if inputs['batch_segments'] is None:
        assert inputs['graph_mask'] is None
        assert inputs['node_mask'] is None
        inputs['graph_mask'] = jnp.ones((1,)).astype(jnp.bool_)  # (1)
        inputs['node_mask'] = jnp.ones((len(atomic_numbers),)).astype(jnp.bool_)  # (num_nodes)
        inputs['batch_segments'] = jnp.zeros_like(atomic_numbers)  # (num_nodes)
    return inputs


def get_observable_fn_sparse(model):
    """All observables of `model` (a StackNetSparse) as a function of (params, **inputs)."""
    def observable_fn(params, positions, atomic_numbers, idx_i, idx_j, **kwargs):
        return model.apply(params, _model_inputs(positions, atomic_numbers, idx_i, idx_j, **kwargs))

    return observable_fn


def get_energy_and_force_fn_sparse(model):
    def energy_fn(params, positions, atomic_numbers, idx_i, idx_j, **kwargs):
        inputs = _model_inputs(positions, atomic_numbers, idx_i, idx_j, **kwargs)
        # Return the WHOLE model output as aux, not just the energy: the caller
        # needs dipole_vec / the ratio heads too, and used to obtain them with a
        # second full model.apply -- doubling the cost of every training and
        # validation step. aux is not differentiated, so the gradient is
        # unchanged. energy_fn is closure-internal; only
        # energy_and_force_and_dipole_and_hirsh_fn is exported.
        model_output = model.apply(params, inputs)
        energy = safe_scale(model_output['energy'], inputs['graph_mask'])  # (num_graphs)
        return -jnp.sum(energy), (energy, model_output)  # (), ((num_graphs), dict)

    def energy_and_force_and_dipole_and_hirsh_fn(params, positions, atomic_numbers, idx_i, idx_j, **kwargs):
        (_, (energy, model_output)), forces = jax.value_and_grad(
            energy_fn,
            argnums=1,
            has_aux=True)(params, positions, atomic_numbers, idx_i, idx_j, **kwargs)
        inputs = _model_inputs(positions, atomic_numbers, idx_i, idx_j, **kwargs)
        graph_mask, node_mask = inputs['graph_mask'], inputs['node_mask']

        result = dict(
            energy=energy,
            forces=forces
        )

        if 'dipole_vec' in model_output:
            dipole_vec = model_output['dipole_vec']  # (num_graphs)
            dipole_vec = safe_scale(dipole_vec, graph_mask[:, None])  # (num_graphs)
            result['dipole_vec'] = dipole_vec

        # Per-atom ratio heads. 'a0_ratios' is what the current so3lr-s/m/l models
        # emit; the legacy `so3lr` emits 'hirshfeld_ratios' for the same quantity.
        # Both must be forwarded: omitting 'a0_ratios' dropped the head entirely,
        # which also made the _RATIO_ALIAS remap in make_loss_fn/make_val_fn dead
        # code (it can only rename a key that actually arrives) -- so training or
        # evaluating a ratio target on a current model raised KeyError.
        for ratio_key in ('a0_ratios', 'hirshfeld_ratios', 'c6_ratios'):
            if ratio_key in model_output:
                result[ratio_key] = safe_scale(model_output[ratio_key], node_mask)

        return result

    return energy_and_force_and_dipole_and_hirsh_fn
