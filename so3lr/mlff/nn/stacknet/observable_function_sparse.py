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
    def energy_and_force_and_dipole_and_hirsh_fn(params, positions, atomic_numbers, idx_i, idx_j, **kwargs):
        inputs = _model_inputs(positions, atomic_numbers, idx_i, idx_j, **kwargs)
        graph_mask, node_mask = inputs['graph_mask'], inputs['node_mask']

        def energy_fn(positions):
            # The whole model output is the aux, so dipole_vec and the ratio heads
            # come from the same model.apply as the forces.
            model_output = model.apply(params, {**inputs, 'positions': positions})
            energy = safe_scale(model_output['energy'], graph_mask)  # (num_graphs)
            return -jnp.sum(energy), (energy, model_output)  # (), ((num_graphs), dict)

        (_, (energy, model_output)), forces = jax.value_and_grad(energy_fn, has_aux=True)(positions)

        result = dict(
            energy=energy,
            forces=forces
        )

        if 'dipole_vec' in model_output:
            dipole_vec = model_output['dipole_vec']  # (num_graphs)
            dipole_vec = safe_scale(dipole_vec, graph_mask[:, None])  # (num_graphs)
            result['dipole_vec'] = dipole_vec

        # Per-atom ratio heads: 'a0_ratios' (so3lr-s/m/l) or 'hirshfeld_ratios'
        # (legacy so3lr) for the same quantity, and 'c6_ratios'. Forward whichever
        # the model has; make_loss_fn's _RATIO_ALIAS maps between the two names.
        for ratio_key in ('a0_ratios', 'hirshfeld_ratios', 'c6_ratios'):
            if ratio_key in model_output:
                result[ratio_key] = safe_scale(model_output[ratio_key], node_mask)

        return result

    return energy_and_force_and_dipole_and_hirsh_fn
