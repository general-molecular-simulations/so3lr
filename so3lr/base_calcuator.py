import jax
import jax.numpy as jnp
import pathlib

from mlff.mdx.potential.mlff_potential_sparse import load_model_from_workdir


def make_so3lr(
    lr_cutoff=12.,
    dispersion_energy_lr_cutoff_damping=2.,
    calculate_forces=True
):
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    model, params = load_model_from_workdir(
        package_dir / 'so3lr' / 'params',
        from_file=True,
        long_range_kwargs=dict(
            cutoff_lr=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_lr_cutoff_damping,
            neighborlist_format_lr='sparse'
        )
    )

    def forward(
            positions,
            other_inputs
    ):
        batch_segments = other_inputs.get('batch_segments')
        graph_mask = other_inputs.get('graph_mask')

        if batch_segments is None:
            assert graph_mask is None

            num_atoms = len(other_inputs['atomic_numbers'])
            batch_segments = jnp.ones((num_atoms,), dtype=jnp.int32)
            graph_mask = jnp.array([True])
            other_inputs['batch_segments'] = batch_segments
            other_inputs['graph_mask'] = graph_mask

        inputs = dict(
            positions=positions,
            **other_inputs
        )

        output = model.apply(params, inputs)

        # We sum over all energies
        summed_energy = jnp.sum(jnp.where(graph_mask, output['energy'], 0.))

        return - summed_energy, output

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
