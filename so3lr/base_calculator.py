import jax
import jax.numpy as jnp

from so3lr.mlff.calculators.potential import build_long_range_kwargs, cast_float_params, load_model_from_workdir
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
):
    # ``workdir`` (legacy arg) takes precedence over ``model``; either may be a
    # bundled model name or a filesystem path. None -> DEFAULT_MODEL.
    spec = resolve_model(workdir if workdir is not None else model)

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

        # Handle residue information for dimer calculations
        residue_segments = other_inputs.get('residue_segments', None)
        residue_charge = other_inputs.get('residue_charge', None)
        
        inputs = dict(
            positions=positions,
            **other_inputs
        )

        # Add residue information if provided
        if residue_segments is not None:
            inputs['residue_segments'] = residue_segments
        if residue_charge is not None:
            inputs['residue_charge'] = residue_charge

        # in the positions' precision (the checkpoint's parameters are float32)
        output = model.apply(cast_float_params(params, positions.dtype), inputs)

        # We sum over all energies
        summed_energy = jnp.sum(jnp.where(graph_mask, output['energy'], 0.))

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
