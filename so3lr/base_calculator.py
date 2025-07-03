import jax
import jax.numpy as jnp
import pathlib

from mlff.mdx.potential.mlff_potential_sparse import load_model_from_workdir


def make_so3lr(
    lr_cutoff=12.,
    dispersion_energy_cutoff_lr_damping=2.,
    calculate_forces=True,
    workdir=None
):
    if workdir is None:
        # Use default SO3LR params directory
        package_dir = pathlib.Path(__file__).parent.parent.resolve()
        workdir_path = package_dir / 'so3lr' / 'params'
        from_file = True
    else:
        # Use provided workdir
        workdir_path = pathlib.Path(workdir).expanduser().resolve()
        from_file = False

    model, params = load_model_from_workdir(
        workdir_path,
        model='so3krates', # or 'itp_net'
        from_file=from_file,
        long_range_kwargs=dict(
            cutoff_lr=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
            neighborlist_format_lr='sparse'
        )
    )

    # Change shapes to allow for multiple theory levels, only needed for https://github.com/kabylda/mlff/tree/v1.0-tfds-pme version
    if 'energy_offset' in params['params']['observables_0']:
        num_theory_levels=16
        old_energy_offset = params['params']['observables_0']['energy_offset']
        if len(old_energy_offset.shape) == 1:
            # print("\nOriginal energy_offset:")
            # print("Shape:", params['params']['observables_0']['energy_offset'].shape)
            new_energy_offset = jnp.tile(old_energy_offset[:, None], (1, num_theory_levels))
            params['params']['observables_0']['energy_offset'] = new_energy_offset

            old_atomic_scales = params['params']['observables_0']['atomic_scales']
            new_atomic_scales = jnp.tile(old_atomic_scales[:, None], (1, num_theory_levels))
            params['params']['observables_0']['atomic_scales'] = new_atomic_scales

            old_kernel = params['params']['observables_0']['energy_dense_final']['kernel']
            new_kernel = jnp.tile(old_kernel, (1, num_theory_levels))
            params['params']['observables_0']['energy_dense_final']['kernel'] = new_kernel

            # print("\nNew energy_offset:")
            # print("Shape:", params['params']['observables_0']['energy_offset'].shape)
            # print("Values:", params['params']['observables_0']['energy_offset'])

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

        output = model.apply(params, inputs)

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
