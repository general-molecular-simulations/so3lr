"""Base calculator."""
import jax
import jax.numpy as jnp

from functools import partial
from typing import Optional, Dict, Sequence

from ..utils import calculator_utils


def make_base_calculator(
        workdir,
        model: str = 'so3krates',
        long_range_kwargs: Optional[Dict] = None,
        calculate_forces=True,
        input_convention: str = 'positions',  # 'displacements'
        output_convention: str = 'per_structure',  # 'per_atom'
        output_intermediate_quantities: Optional[Sequence[str]] = None,
        from_file: bool = False,
):
    """
    Make a base calculator for neural network models.

    Args:
        workdir (str): The workdir. Can be any directory which contains `hyperparameters.json` and
            `hyperparameters.yaml` and `checkpoints` as a subdirectory.
        model (str): Model name.
        long_range_kwargs (): Dictionary with the following keys:
            ['cutoff_lr', 'dispersion_energy_cutoff_lr_damping', 'neighborlist_format_lr']. Can not be `None` when
            long-range electrostatic and/or long-range dispersion modules are used.  'cutoff_lr = None' will be treated
            like `cutoff_lr = np.inf`, i.e. infinite long-range cutoff. Neighborlist format can be `sparse` or
            `ordered_sparse`. `dispersion_energy_cutoff_lr_damping` specifies where to start the damping for the
            dispersion interactions. For `dispersion_energy_cutoff_lr_damping = 2.` damping starts at `lr_cutoff - 2.`.
        calculate_forces (bool): Calculate forces.
        input_convention (str): Input convention. Can be either `positions` or `displacements`. In the case of the
            latter, displacement vectors have already been calculated properly, i.e. taking PBCs into account if
            present.
        output_convention (str): Output convention. Can be either `per_structure` or `per_atom`. Some properties, i.e.
            the energy are obtained as U = sum_i U_i, where U_i are atomic energies. For this properties, `per_atom` will
            return U_i whereas `per_structure` will return U.
        output_intermediate_quantities (Optional[Sequence[str]]): If not None, the model will return intermediate quantities for the keys given in the sequence.
        from_file (bool): Load parameters from file not from checkpoint directory.

    Returns:

    """
    if calculate_forces is True:
        if input_convention == 'displacements':
            raise ValueError(
                f'Input convention `displacements` can not be used when forces should be calculated directly, since'
                f'this forbids taking gradients wrt the positions. Received {input_convention=} and '
                f'{calculate_forces=}.'
            )
        if output_convention == 'per_atom':
            raise ValueError(
                f'Ouput convention `per_atom` can not be used when forces should be calculated directly, since'
                f'this would mean taking computing gradients wrt all positions for each atomic energy. This does not '
                f'correspond to proper forces. Received {output_convention=} and {calculate_forces=}.'
            )

    net, params = calculator_utils.load_model_from_workdir(
        workdir=workdir,
        model=model,
        long_range_kwargs=long_range_kwargs,
        output_intermediate_quantities=output_intermediate_quantities,
        from_file=from_file,
    )

    net.reset_input_convention(
        input_convention=input_convention
    )
    net.reset_output_convention(
        output_convention=output_convention
    )

    def forward(
            positions,
            other_inputs
    ):
        batch_segments = other_inputs.get('batch_segments')
        graph_mask = other_inputs.get('graph_mask')
        node_mask = other_inputs.get('node_mask')

        if batch_segments is None:
            assert graph_mask is None
            assert node_mask is None

            num_atoms = len(other_inputs['atomic_numbers'])
            batch_segments = jnp.zeros((num_atoms, ), dtype=jnp.int32)
            graph_mask = jnp.array([True])
            node_mask = jnp.ones((num_atoms,)).astype(jnp.bool_)  # (num_nodes)

            other_inputs['batch_segments'] = batch_segments
            other_inputs['graph_mask'] = graph_mask
            other_inputs['node_mask'] = node_mask

        inputs = dict(
            positions=positions,
            **other_inputs
        )

        output = net.apply(params, inputs)

        # We sum over all energies
        summed_energy = jnp.sum(jnp.where(graph_mask, output['energy'], 0.))

        return - summed_energy, output

    def base_calculator_fn(inputs):
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

    return base_calculator_fn


make_so3krates_calculator = partial(make_base_calculator, model='so3krates')