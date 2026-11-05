import jax
import jax.numpy as jnp

from so3lr.mlff.calculators.potential import (
    build_long_range_kwargs, cast_float_params, load_model_from_workdir, require_x64)
from so3lr.mlff.nn.embed.embed_sparse import pair_displacements
from so3lr.model_registry import resolve_model


def make_so3lr(
    lr_cutoff=12.,
    dispersion_energy_cutoff_lr_damping=2.,
    calculate_forces=True,
    model=None,
    workdir=None,
    output_intermediate_quantities=None,
    charge_embed_total_charge=None,
    charge_equilibration_total_charge=None,
    kspace_electrostatics=None,
    kspace_interp_nodes=4,
    kspace_dispersion=None,
    kspace_dispersion_interp_nodes=4,
    precision=None,
):
    """precision: the model's dtype (None: the positions'). The displacements are formed in the
    positions' dtype before the cast, as in ``jaxmd_utils.neighbor_list_featurizer``."""
    # ``workdir`` (legacy arg) takes precedence over ``model``; either may be a
    # bundled model name or a filesystem path. None -> DEFAULT_MODEL.
    spec = resolve_model(workdir if workdir is not None else model)
    require_x64(precision)  # before the model loads

    long_range_kwargs = build_long_range_kwargs(
        lr_cutoff, dispersion_energy_cutoff_lr_damping, 'sparse',
        kspace_electrostatics=kspace_electrostatics,
        kspace_interp_nodes=kspace_interp_nodes,
        kspace_dispersion=kspace_dispersion,
        kspace_dispersion_interp_nodes=kspace_dispersion_interp_nodes,
    )

    model, params = load_model_from_workdir(
        spec.dir,
        model='so3krates', # or 'itp_net'
        from_file=spec.from_file,
        long_range_kwargs=long_range_kwargs,
        output_intermediate_quantities=output_intermediate_quantities,
        charge_embed_total_charge=charge_embed_total_charge,
        charge_equilibration_total_charge=charge_equilibration_total_charge
    )

    model.reset_input_convention('displacements')
    # per-atom energies, summed in forward
    model.reset_output_convention('per_atom')
    if precision is not None:
        params = cast_float_params(params, precision)  # once; forward's cast is then a no-op

    # No theory-level expansion here: load_model_from_workdir above already did
    # it. This used to hold a copy of the tiling that could never fire (the
    # params are 2-D by then) and which omitted the final-bias case.

    def forward(
            positions,
            other_inputs
    ):
        batch_segments = other_inputs.get('batch_segments')
        graph_mask = other_inputs.get('graph_mask')

        if batch_segments is None:
            assert graph_mask is None

            num_atoms = len(other_inputs['atomic_numbers'])
            batch_segments = jnp.zeros((num_atoms, ), dtype=jnp.int32)
            graph_mask = jnp.array([True])
            other_inputs['batch_segments'] = batch_segments
            other_inputs['graph_mask'] = graph_mask

        inputs = dict(
            positions=positions,
            **other_inputs
        )
        inputs['displacements'], inputs['displacements_lr'] = pair_displacements(inputs)

        # the parameters too: the checkpoint's are float32
        dtype = positions.dtype if precision is None else precision
        output = model.apply(cast_float_params(params, dtype), cast_float_params(inputs, dtype))

        # the structures' energies, summed over their atoms in the positions' dtype
        energy = jax.ops.segment_sum(output['energy'].astype(positions.dtype), batch_segments,
                                     num_segments=len(graph_mask))
        output['energy'] = jnp.where(graph_mask, energy, 0.)
        summed_energy = jnp.sum(output['energy'])

        return -summed_energy, output

    def so3lr_fn(inputs):
        positions = inputs['positions']

        other_inputs = {k: v for k, v in inputs.items() if k != 'positions'}

        if calculate_forces is True:

            (_, output), forces = jax.value_and_grad(
                forward,
                has_aux=True,
                argnums=0
            )(
                positions,
                other_inputs
            )

            output['forces'] = forces

            return output

        else:

            _, output = forward(
                positions,
                other_inputs
            )

            return output

    return so3lr_fn
