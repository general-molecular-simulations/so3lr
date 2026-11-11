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


# Per-atom outputs forwarded next to energy, forces and dipole: the ratio heads
# ('a0_ratios' for so3lr-tauri-s/m/l, 'hirshfeld_ratios' for legacy so3lr, mapped
# onto each other by make_loss_fn's _RATIO_ALIAS) and the partial charges, which
# the model only emits when built with output_intermediate_quantities=['partial_charges'].
_NODE_OUTPUTS = ('a0_ratios', 'hirshfeld_ratios', 'c6_ratios', 'partial_charges')


def _masked_outputs(model_output, inputs, keys):
    """The `keys` of `model_output` that the model emits, zeroed on padding."""
    masks = {'dipole_vec': inputs['graph_mask'][:, None]}
    return {
        k: safe_scale(model_output[k], masks.get(k, inputs['node_mask']))
        for k in keys if k in model_output
    }


def get_observable_fn_sparse(model):
    """All observables of `model` (a StackNetSparse) as a function of (params, **inputs)."""
    def observable_fn(params, positions, atomic_numbers, idx_i, idx_j, **kwargs):
        return model.apply(params, _model_inputs(positions, atomic_numbers, idx_i, idx_j, **kwargs))

    return observable_fn


def get_energy_and_force_fn_sparse(model):
    def energy_and_force_and_dipole_and_hirsh_fn(params, positions, atomic_numbers, idx_i, idx_j, **kwargs):
        inputs = _model_inputs(positions, atomic_numbers, idx_i, idx_j, **kwargs)
        graph_mask = inputs['graph_mask']

        def energy_fn(positions):
            # The whole model output is the aux, so dipole_vec and the ratio heads
            # come from the same model.apply as the forces.
            model_output = model.apply(params, {**inputs, 'positions': positions})
            energy = safe_scale(model_output['energy'], graph_mask)  # (num_graphs)
            return -jnp.sum(energy), (energy, model_output)  # (), ((num_graphs), dict)

        (_, (energy, model_output)), forces = jax.value_and_grad(energy_fn, has_aux=True)(positions)

        return dict(
            energy=energy,
            forces=forces,
            **_masked_outputs(model_output, inputs, ('dipole_vec',) + _NODE_OUTPUTS),
        )

    return energy_and_force_and_dipole_and_hirsh_fn


def get_long_range_anchor_fn_sparse(model, keys):
    """Forward-only long-range quantities of `model` as a function of (params, **inputs).

    Used as the frozen teacher when anchoring the long-range heads during
    fine-tuning: partial charges and the a0/C6 ratio heads need no positional
    derivative, so this skips the force pass that get_energy_and_force_fn_sparse
    does. `model` must emit 'partial_charges' (output_intermediate_quantities) if
    it is among `keys`.
    """
    def anchor_fn(params, positions, atomic_numbers, idx_i, idx_j, **kwargs):
        inputs = _model_inputs(positions, atomic_numbers, idx_i, idx_j, **kwargs)
        return _masked_outputs(model.apply(params, inputs), inputs, keys)

    return anchor_fn
