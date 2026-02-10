"""Potential calculator."""
import jax
import jax.numpy as jnp
import logging
import numpy as np
import pathlib
import json

from abc import abstractmethod
from flax import struct
from typing import Any, Callable, Type, Dict, Sequence, Optional

from glp.graph import Graph

from ..utils import calculator_utils
from .base_calculator import make_base_calculator
from ..nn.stacknet.observable_function_sparse import get_observable_fn_sparse
from ml_collections import config_dict
from ..config import from_config
from orbax import checkpoint


def load_hyperparameters(workdir: str):
    """Load hyperparameters from workdir."""
    with open(pathlib.Path(workdir) / "hyperparameters.json", "r") as fp:
        cfg = json.load(fp)
    cfg = config_dict.ConfigDict(cfg)
    return cfg


def load_model_from_workdir(
        workdir: str,
        model='so3krates',
        long_range_kwargs: Dict[str, Any] = None,
        from_file: bool = False,
        output_intermediate_quantities: Optional[Sequence[str]] = None
):
    """Load model and parameters from workdir with advanced features support."""
    cfg = load_hyperparameters(workdir)

    dispersion_energy_bool = cfg.model.dispersion_energy_bool
    electrostatic_energy_bool = cfg.model.electrostatic_energy_bool

    # For local model both are false.
    if (electrostatic_energy_bool is True) or (dispersion_energy_bool is True):
        if long_range_kwargs is None:
            raise ValueError(
                "For a potential with long-range electrostatic and/or dispersion corrections, long_range_kwargs must "
                f"be specified. Received {long_range_kwargs=}."
            )

        cutoff_lr = long_range_kwargs['cutoff_lr']
        neighborlist_format = long_range_kwargs['neighborlist_format_lr']
        if cutoff_lr is not None:
            if cutoff_lr < 0:
                raise ValueError(
                    f"For a potential with long range components the long range cutoff value must be greater "
                    f"than zero. received {cutoff_lr=}."
                )

        cfg.model.cutoff_lr = cutoff_lr
        cfg.neighborlist_format_lr = neighborlist_format

        cfg.electrostatic_energy_kspace_do_ewald_bool = long_range_kwargs.get('coulomb_kspace_do_ewald', False)
        cfg.electrostatic_energy_kspace_interp_nodes = int(long_range_kwargs.get('coulomb_kspace_interp_nodes', 4))

        if dispersion_energy_bool is True:
            dispersion_energy_cutoff_lr_damping = long_range_kwargs['dispersion_energy_cutoff_lr_damping']
            if cutoff_lr is not None:
                if dispersion_energy_cutoff_lr_damping is None:
                    raise ValueError(
                        f"dispersion_energy_cutoff_lr_damping must not be None if dispersion_energy_bool is True and "
                        f"cutoff_lr has a finite value. received {dispersion_energy_bool=}, {cutoff_lr=} and "
                        f"{dispersion_energy_cutoff_lr_damping=}."
                    )
            if cutoff_lr is None:
                if dispersion_energy_cutoff_lr_damping is not None:
                    raise ValueError(
                        f"dispersion_energy_cutoff_lr_damping must be None if dispersion_energy_bool is True and "
                        f"cutoff_lr is infinite (specified via lr_cutoff=None). received {dispersion_energy_bool=}, "
                        f"{dispersion_energy_cutoff_lr_damping=} and {cutoff_lr=}"
                    )
            cfg.model.dispersion_energy_cutoff_lr_damping = dispersion_energy_cutoff_lr_damping

    if from_file is True:
        import pickle

        with open(pathlib.Path(workdir) / 'params.pkl', 'rb') as f:
            params = pickle.load(f)
    else:
        loaded_mngr = checkpoint.CheckpointManager(
            pathlib.Path(workdir) / "checkpoints",
            item_names=('params',),
            item_handlers={'params': checkpoint.StandardCheckpointHandler()},
            options=checkpoint.CheckpointManagerOptions(step_prefix="ckpt"),
        )

        mngr_state = loaded_mngr.restore(
            loaded_mngr.latest_step()
        )

        params = mngr_state.get('params')

    if model == 'so3krates':
        net = from_config.make_so3krates_sparse_from_config(
            cfg,
            output_intermediate_quantities=output_intermediate_quantities
        )
    else:
        raise ValueError(
            f'{model=} is not a valid model.'
        )

    if 'energy_offset' in params['params']['observables_0']:
        # Change shapes to allow for multiple theory levels
        num_theory_levels = 16
        old_energy_offset = params['params']['observables_0']['energy_offset']
        if len(old_energy_offset.shape) == 1:
            new_energy_offset = jnp.tile(old_energy_offset[:, None], (1, num_theory_levels))
            params['params']['observables_0']['energy_offset'] = new_energy_offset

            old_atomic_scales = params['params']['observables_0']['atomic_scales']
            new_atomic_scales = jnp.tile(old_atomic_scales[:, None], (1, num_theory_levels))
            params['params']['observables_0']['atomic_scales'] = new_atomic_scales

            old_kernel = params['params']['observables_0']['energy_dense_final']['kernel']
            new_kernel = jnp.tile(old_kernel, (1, num_theory_levels))
            params['params']['observables_0']['energy_dense_final']['kernel'] = new_kernel

    return net, params


@struct.dataclass
class MachineLearningPotential:
    cutoff: float = struct.field(pytree_node=False)
    effective_cutoff: float = struct.field(pytree_node=False)

    potential_fn: Callable[[Graph], jnp.ndarray] = struct.field(pytree_node=False)
    dtype: Type = struct.field(pytree_node=False)

    @classmethod
    @abstractmethod
    def create_from_workdir(cls, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, inputs: Dict):
        pass


@struct.dataclass
class PotentialSparse(MachineLearningPotential):
    cutoff: float = struct.field(pytree_node=False)
    effective_cutoff: float = struct.field(pytree_node=False)

    long_range_bool: bool = struct.field(pytree_node=False)
    long_range_cutoff: float = struct.field(pytree_node=False)

    potential_fn: Callable[[Graph, bool], jnp.ndarray] = struct.field(pytree_node=False)
    dtype: Type = struct.field(pytree_node=False)

    @classmethod
    def create_from_ckpt_dir(
            cls,
            ckpt_dir: str,
            from_file: bool = False,
            add_shift: bool = False,
            long_range_kwargs: Dict[str, Any] = None,
            dtype=jnp.float32,
            model: str = 'so3krates',
            output_intermediate_quantities: Optional[Sequence[str]] = None
    ):
        """
        Deprecated: Use create_from_workdir instead.

        This method is kept for backward compatibility with documented API.
        """
        logging.warning(
            '`create_from_ckpt_dir` is deprecated and replaced by `create_from_workdir`, please use this method in '
            'the future. For now this calls `create_from_workdir` but will raise an error in the future.'
        )
        return cls.create_from_workdir(
            ckpt_dir,
            from_file,
            add_shift,
            long_range_kwargs,
            dtype,
            model,
            output_intermediate_quantities
        )

    @classmethod
    def create_from_workdir(
            cls,
            workdir: str,
            from_file: bool = False,
            add_shift: bool = False,
            long_range_kwargs: Dict[str, Any] = None,
            dtype=jnp.float32,
            model: str = 'so3krates',
            output_intermediate_quantities: Optional[Sequence[str]] = None
    ):
        """
        Create potential from workdir with full feature support.

        Args:
            workdir: Path to checkpoint directory
            from_file: Whether to load from pickle file
            add_shift: Whether to add energy shift
            long_range_kwargs: Dictionary with keyword arguments for the long-range modules
            dtype: Data type for computation
            model: Model type ('so3krates')
            output_intermediate_quantities: Optional sequence of intermediate quantities to output

        Returns:
            PotentialSparse instance
        """

        if add_shift is True and (dtype == np.float32 or dtype == jnp.float32):
            logging.warning(
                'Energy shift is enabled but float32 precision is used.'
                ' For large absolute energy values, this can lead to floating point errors in the energies.'
                ' If you do not need the absolute energy values since only relative ones are important, we'
                ' suggest to disable the energy shift since increasing the precision slows down'
                ' computation.'
            )

        # Load model with advanced features support
        net, params = load_model_from_workdir(
            workdir=workdir,
            from_file=from_file,
            model=model,
            long_range_kwargs=long_range_kwargs,
            output_intermediate_quantities=output_intermediate_quantities
        )

        # Handle multi-theory level support
        if 'energy_offset' in params['params']['observables_0']:
            num_theory_levels = 16
            old_energy_offset = params['params']['observables_0']['energy_offset']
            if len(old_energy_offset.shape) == 1:
                new_energy_offset = jnp.tile(old_energy_offset[:, None], (1, num_theory_levels))
                params['params']['observables_0']['energy_offset'] = new_energy_offset

                old_atomic_scales = params['params']['observables_0']['atomic_scales']
                new_atomic_scales = jnp.tile(old_atomic_scales[:, None], (1, num_theory_levels))
                params['params']['observables_0']['atomic_scales'] = new_atomic_scales

                old_kernel = params['params']['observables_0']['energy_dense_final']['kernel']
                new_kernel = jnp.tile(old_kernel, (1, num_theory_levels))
                params['params']['observables_0']['energy_dense_final']['kernel'] = new_kernel

        cfg = load_hyperparameters(workdir=workdir)

        net.reset_input_convention('displacements')
        net.reset_output_convention('per_atom')

        long_range_bool = (cfg.model.electrostatic_energy_bool is True) or (cfg.model.dispersion_energy_bool is True)

        cutoff = cfg.model.cutoff
        steps = cfg.model.num_layers

        effective_cutoff = steps * cutoff

        if add_shift:
            shifts = {int(k): float(v) for k, v in dict(cfg.data.energy_shifts).items()}

            def shift(v, z):
                return v + jnp.asarray(shifts, dtype=dtype)[z][:, None]
        else:
            def shift(v, z):
                return v

        def shift_fn(x: jnp.ndarray, z: jnp.ndarray):
            return shift(x, z)

        obs_fn = get_observable_fn_sparse(net)

        def graph_to_mlff_input(graph: Graph):
            # Enhanced graph conversion with advanced field support
            x = {
                'positions': graph.positions,
                'displacements': graph.edges,
                'atomic_numbers': graph.nodes,
                'idx_i': graph.centers,
                'idx_j': graph.others,
                'total_charge': graph.total_charge,
                'num_unpaired_electrons': graph.num_unpaired_electrons,
                'cell': getattr(graph, 'cell', None),
                'theory_mask': getattr(graph, 'theory_mask', None),
                'k_grid': getattr(graph, 'k_grid', None),
                'k_smearing': getattr(graph, 'k_smearing', None),
                'residue_charge': getattr(graph, 'residue_charge', None),
                'residue_segments': getattr(graph, 'residue_segments', None),
            }
            if long_range_bool is True:
                x_lr = {
                    'displacements_lr': graph.edges_lr,
                    'idx_i_lr': graph.idx_i_lr,
                    'idx_j_lr': graph.idx_j_lr,
                }
                x.update(x_lr)

            return x

        def potential_fn(graph: Graph, has_aux: bool = False):
            x = graph_to_mlff_input(graph)
            y = obs_fn(params, **x)

            shifted_energy = shift_fn(y['energy'], x['atomic_numbers']).reshape(-1).astype(dtype)

            if has_aux:
                aux = jax.tree_util.tree_map(lambda u: u.astype(dtype), y)
                return shifted_energy, aux
            else:
                return shifted_energy

        return cls(
            cutoff=cutoff,
            effective_cutoff=effective_cutoff,
            long_range_bool=long_range_bool,
            long_range_cutoff=long_range_kwargs['cutoff_lr'] if long_range_bool is True else None,
            potential_fn=potential_fn,
            dtype=dtype
        )

    def __call__(
            self,
            graph: Graph,
            has_aux: bool = False
    ) -> jnp.ndarray:
        """Call the potential function on a graph."""
        return self.potential_fn(graph, has_aux)