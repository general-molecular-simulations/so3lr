import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxpme

from ase.units import Bohr, Hartree
from ase.units import alpha as fine_structure
from functools import partial
from jax.ops import segment_sum
from jax.nn.initializers import constant
from typing import Any, Callable, Dict, Tuple, Optional, Sequence

from so3lr.mlff.nn.sub_module import BaseSubModule
from so3lr.mlff.masking.mask import safe_mask
from so3lr.mlff.nn.observable.dispersion_ref_data import alphas, C6_coef, alphas_legacy, C6_coef_legacy, NLH_AA, NLH_BB
from so3lr.mlff.masking.mask import safe_scale
from so3lr.mlff.nn.activation_function import softplus_inverse, softplus


def _flat_to_padded(flat_array, batch_segments, num_graphs, max_atoms_per_graph, local_idx, pad_value=0.0):
    """Reshape a flat (N,) or (N, D) array into (num_graphs, max_atoms_per_graph, ...) padded array."""
    if flat_array.ndim == 1:
        padded = jnp.full((num_graphs, max_atoms_per_graph), pad_value, dtype=flat_array.dtype)
    else:
        padded = jnp.full((num_graphs, max_atoms_per_graph) + flat_array.shape[1:], pad_value, dtype=flat_array.dtype)
    padded = padded.at[batch_segments, local_idx].set(flat_array)
    return padded


def _padded_to_flat(padded_array, batch_segments, node_mask, local_idx):
    """Reshape a (num_graphs, max_atoms_per_graph) padded array back to flat (N,)."""
    return padded_array[batch_segments, local_idx] * node_mask


def _vmap_kspace_per_graph(solve_fn, inputs, per_atom_arrays, output_key):
    """Generic per-graph kspace processing: pad, vmap, mask PBC, unpad.

    Args:
        solve_fn: Callable(cell_g, *per_graph_arrays) -> (max_atoms,) energies for one graph.
        inputs: Full batched inputs dict (for batch_segments, node_mask, cell, etc.).
        per_atom_arrays: List of flat (N,) or (N,D) arrays to pad per-graph and pass to solve_fn.
        output_key: Key name for the result dict.
    """
    batch_segments = inputs['batch_segments']
    node_mask = inputs['node_mask']
    num_graphs = inputs['graph_mask'].shape[0]
    num_atoms_total = node_mask.shape[0]
    cell = inputs['cell']  # (num_graphs, 3, 3)

    # Use num_atoms_total as the static padded dimension (safe upper bound)
    max_atoms = num_atoms_total

    # Compute local indices once
    counts = jnp.bincount(batch_segments, length=num_graphs)
    graph_starts = jnp.zeros(num_graphs + 1, dtype=jnp.int32).at[1:].set(jnp.cumsum(counts))
    local_idx = jnp.arange(num_atoms_total) - graph_starts[batch_segments]

    # Pad all per-atom arrays to (num_graphs, max_atoms, ...)
    padded_arrays = [
        _flat_to_padded(a, batch_segments, num_graphs, max_atoms, local_idx)
        for a in per_atom_arrays
    ]

    # Reciprocal-space solvers invert the cell before doing any useful work.
    # A batched neighbor list represents non-periodic and padding graphs with a
    # zero cell, so evaluating those entries and masking afterwards produces
    # NaNs (NaN * 0 is still NaN).  Only full-rank cells belonging to real
    # graphs are eligible for the solver; all other entries get a harmless
    # finite stand-in before vmap.
    graph_mask = inputs['graph_mask'].astype(bool)
    cell_is_finite = jnp.all(jnp.isfinite(cell), axis=(-2, -1))
    cell_is_full_rank = jnp.abs(jnp.linalg.det(cell)) > 0
    has_pbc = graph_mask & cell_is_finite & cell_is_full_rank
    # A vacuum structure may retain a finite simulation/visualization cell.
    # Explicit boundary flags take precedence over that cell geometry.
    pbc = inputs.get('pbc')
    if pbc is not None:
        has_pbc = has_pbc & jnp.all(pbc, axis=-1)
    identity_cell = jnp.broadcast_to(jnp.eye(3, dtype=cell.dtype), cell.shape)
    solver_cell = jnp.where(has_pbc[:, None, None], cell, identity_cell)

    # vmap the solver over graphs — solver must be pure JAX (no Flax state)
    energies_padded = jax.vmap(solve_fn)(solver_cell, *padded_arrays)

    # Zero out non-PBC graphs and scatter back to flat array
    energies_padded = jnp.where(
        has_pbc[:, None], energies_padded, jnp.zeros_like(energies_padded)
    )
    energies_flat = _padded_to_flat(energies_padded, batch_segments, node_mask, local_idx)

    return {output_key: energies_flat}


def _batched_kspace_electrostatic(kspace_module, inputs):
    """Run electrostatic kspace per-graph using vmap on the solver, then scatter back."""
    k_grid = inputs['k_grid']
    k_smearing = inputs['k_smearing']
    solver = kspace_module.solver
    ke = kspace_module.ke
    is_ewald = (kspace_module.kspace_method == 'ewald')
    pbc = jnp.array([True, True, True])

    def _solve_single(cell_g, pos_g, charges_g, mask_g):
        charges_g = charges_g * mask_g
        volume = jnp.abs(jnp.linalg.det(cell_g))
        reciprocal_cell = jaxpme.kspace.get_reciprocal(cell_g)
        kvectors = jaxpme.kspace.generate_kvectors(
            reciprocal_cell, k_grid.shape, dtype=pos_g.dtype, for_ewald=is_ewald
        )
        if is_ewald:
            potentials = solver.kspace(k_smearing, charges_g, kvectors, pos_g, volume, cell_g, pbc)
        else:
            potentials = solver.kspace(k_smearing, charges_g, reciprocal_cell, k_grid, kvectors, pos_g, volume, cell_g, pbc)
        return safe_scale(charges_g * potentials * ke, mask_g)

    node_mask = inputs['node_mask']
    return _vmap_kspace_per_graph(
        _solve_single, inputs,
        per_atom_arrays=[
            inputs['positions'],
            inputs['partial_charges'],
            node_mask.astype(inputs['positions'].dtype),
        ],
        output_key='electrostatic_energy_kspace',
    )


def _effective_atomic_c6(atomic_numbers, c6_ratios, legacy=False, dtype=None):
    """Return per-atom C6 values from the model generation's reference table.

    Real-space mixing, reciprocal-space dispersion, and the real-space PME
    subtraction must use the same free-atom coefficients.  Keeping this lookup
    in one place prevents a legacy model from mixing its original real-space
    coefficients with the refitted SO3LR-2 table in the PME terms.
    """
    if dtype is None:
        dtype = c6_ratios.dtype
    reference = C6_coef_legacy if legacy else C6_coef
    reference = jnp.asarray(reference, dtype=dtype)
    c6_ratios = jnp.asarray(c6_ratios, dtype=dtype)
    return jnp.take(reference, atomic_numbers - 1, axis=0) * c6_ratios


def _dispersion_charges(atomic_c6, node_mask):
    """Zero padding before differentiating the square root of atomic C6.

    Multiplying sqrt(0) by a padding mask afterwards leaves its infinite
    derivative in the backward pass, producing NaN C6 gradients. Use a finite
    stand-in inside the square root for absent atoms, then zero their charges.
    """
    active = node_mask != 0
    safe_c6 = jnp.where(active, jnp.maximum(atomic_c6, 0.0), 1.0)
    return jnp.where(active, jnp.sqrt(safe_c6), 0.0)


def _batched_kspace_dispersion(kspace_module, inputs):
    """Run dispersion kspace per-graph using vmap on the solver, then scatter back."""
    k_grid_disp = inputs['k_grid_disp']
    k_smearing_disp = inputs['k_smearing_disp']
    solver = kspace_module.solver
    is_ewald = kspace_module.do_ewald
    pbc = jnp.array([True, True, True])

    _Bohr = jnp.asarray(Bohr, dtype=inputs['positions'].dtype)
    _Hartree = jnp.asarray(Hartree, dtype=inputs['positions'].dtype)

    def _solve_single(cell_g, pos_g, z_g, mask_g, c6_g):
        C6_i = _effective_atomic_c6(
            z_g, c6_g, legacy=kspace_module.legacy, dtype=pos_g.dtype
        )
        q_disp = _dispersion_charges(C6_i, mask_g)

        pos_bohr = pos_g / _Bohr
        cell_bohr = cell_g / _Bohr
        smearing_bohr = k_smearing_disp / _Bohr

        volume_bohr = jnp.abs(jnp.linalg.det(cell_bohr))
        reciprocal_cell_bohr = jaxpme.kspace.get_reciprocal(cell_bohr)
        kvectors = jaxpme.kspace.generate_kvectors(
            reciprocal_cell_bohr, k_grid_disp.shape,
            dtype=pos_g.dtype, for_ewald=is_ewald,
        )
        if is_ewald:
            potentials = solver.kspace(smearing_bohr, q_disp, kvectors, pos_bohr, volume_bohr, cell_bohr, pbc)
        else:
            potentials = solver.kspace(smearing_bohr, q_disp, reciprocal_cell_bohr, k_grid_disp, kvectors, pos_bohr, volume_bohr, cell_bohr, pbc)
        return safe_scale(-q_disp * potentials * _Hartree, mask_g)

    node_mask = inputs['node_mask']
    # Resolve per-atom C6 ratios exactly like the non-batched
    # DispersionEnergyKspace.__call__ does. The previous silent jnp.zeros
    # fallback zeroed reciprocal-space dispersion for every batched/PBC
    # structure: EnergySparse never writes 'c6_ratios' into ``inputs``, so the
    # .get() always missed and the dispersion charges collapsed to zero.
    if kspace_module.legacy:
        hirshfeld_ratios = inputs.get('hirshfeld_ratios')
        if hirshfeld_ratios is None:
            hirshfeld_ratios = kspace_module.hirshfeld_ratios(inputs)['hirshfeld_ratios']
        c6_ratios = hirshfeld_ratios ** 2
    else:
        c6_ratios = kspace_module.c6_ratios(inputs)['c6_ratios']

    return _vmap_kspace_per_graph(
        _solve_single, inputs,
        per_atom_arrays=[
            inputs['positions'],
            inputs['atomic_numbers'],
            node_mask.astype(inputs['positions'].dtype),
            c6_ratios,
        ],
        output_key='dispersion_energy_kspace',
    )


class EnergySparse(BaseSubModule):
    prop_keys: Dict
    zmax: int = 118
    regression_dim: int = None
    # Optional config-driven deeper readout MLP. When set (e.g. [128, 128]), it
    # defines the full stack of hidden widths for the per-atom energy head; the
    # first hidden layer keeps the legacy name 'energy_dense_regression' so
    # pretrained weights still load, and any extra layers ('energy_dense_hidden_i')
    # initialise fresh. When None, falls back to the single `regression_dim`
    # layer (or no hidden layer), preserving the original architecture exactly.
    final_mlp_widths: Optional[Sequence[int]] = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    learn_atomic_type_scales: bool = False
    learn_atomic_type_shifts: bool = False
    output_is_zero_at_init: bool = True
    output_convention: str = 'per_structure'
    output_intermediate_quantities: Optional[Sequence[str]] = None
    module_name: str = 'energy_sparse'
    electrostatic_energy_bool: bool = False
    electrostatic_energy: Optional[Any] = None
    electrostatic_energy_kspace: Optional[Any] = None
    dispersion_energy_bool: bool = False
    dispersion_energy: Optional[Any] = None
    dispersion_energy_kspace: Optional[Any] = None
    partial_charges: Optional[Any] = None
    hirshfeld_ratios: Optional[Any] = None
    zbl_repulsion_bool: bool = False
    zbl_repulsion: Optional[Any] = None
    use_final_bias_bool: bool = False

    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        """Compute total energy from node features, including optional physical corrections.

        Predicts per-atom energies from learned node features, with optional atomic-type
        scales and shifts controlled by a theory mask. Optionally adds ZBL repulsion,
        electrostatic energy (real-space and k-space), and dispersion energy contributions.

        Args:
            inputs (Dict):
                x (Array): Node features, shape: (num_nodes, num_features)
                atomic_numbers (Array): Atomic numbers, shape: (num_nodes)
                batch_segments (Array): Batch segments, shape: (num_nodes)
                node_mask (Array): Node mask, shape: (num_nodes)
                graph_mask (Array): Graph mask, shape: (num_graphs)
                theory_mask (Array): Theory level mask, shape: (num_graphs, num_theory_levels)

        Returns:
            Dict: Contains 'energy' as per-structure (num_graphs) or per-atom (num_nodes)
                array depending on ``output_convention``. May also include intermediate
                quantities if ``output_intermediate_quantities`` is set.
        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        theory_mask = inputs['theory_mask'] # (num_graphs, num_theory_levels)

        num_theory_levels = theory_mask.shape[-1]
        theory_mask = theory_mask[batch_segments] # (num_nodes, num_theory_levels)
        num_graphs = len(graph_mask)

        if self.learn_atomic_type_shifts:
            energy_offset = jnp.take(
                self.param(
                    'energy_offset',
                    nn.initializers.zeros_init(),
                    (self.zmax + 1, num_theory_levels)
                ),
                atomic_numbers,
                axis=0
            )  # (num_nodes, num_levels_of_theory)
        else:
            energy_offset = jnp.zeros((1,), dtype=x.dtype)

        if self.learn_atomic_type_scales:
            atomic_scales = jnp.take(
                self.param(
                    'atomic_scales',
                    nn.initializers.ones_init(),
                    (self.zmax + 1, num_theory_levels)
                ), atomic_numbers, axis=0)  # (num_nodes, num_levels_of_theory)
        else:
            atomic_scales = jnp.ones((1,), dtype=x.dtype)

        # Resolve the hidden-layer widths of the readout MLP. `final_mlp_widths`
        # (new, flexible) takes precedence over the legacy single `regression_dim`.
        if self.final_mlp_widths is not None:
            hidden_widths = list(self.final_mlp_widths)
        elif self.regression_dim is not None:
            hidden_widths = [self.regression_dim]
        else:
            hidden_widths = []

        if hidden_widths:
            y = x
            for i, width in enumerate(hidden_widths):
                # Keep the legacy name for the first hidden layer so pretrained
                # 'energy_dense_regression' weights still load; extra layers are
                # named 'energy_dense_hidden_{i}' and initialise fresh.
                layer_name = 'energy_dense_regression' if i == 0 else f'energy_dense_hidden_{i}'
                y = nn.Dense(
                    width,
                    kernel_init=nn.initializers.lecun_normal(),
                    name=layer_name
                )(y)  # (num_nodes, width)
                y = self.activation_fn(y)  # (num_nodes, width)
            atomic_energy = nn.Dense(
                num_theory_levels,
                kernel_init=self.kernel_init,
                use_bias=self.use_final_bias_bool,
                name='energy_dense_final'
            )(y)  # (num_nodes, num_levels_of_theory)
        else:
            atomic_energy = nn.Dense(
                num_theory_levels,
                use_bias=self.use_final_bias_bool,
                kernel_init=self.kernel_init,
                name='energy_dense_final'
            )(x) # (num_nodes, num_levels_of_theory)

        atomic_energy = atomic_energy * atomic_scales
        atomic_energy += energy_offset  # (num_nodes, num_levels_of_theory)

        atomic_energy = jnp.where(
            theory_mask,
            atomic_energy,
            jnp.zeros_like(atomic_energy)
        ).sum(axis=-1)  # (num_nodes)

        atomic_energy = safe_scale(atomic_energy, node_mask)
        inputs.update({'nn_energy': atomic_energy})

        if self.zbl_repulsion_bool:
            inputs.update(**self.zbl_repulsion(inputs))
            atomic_energy += inputs['zbl_repulsion']

        if self.electrostatic_energy_bool:
            inputs.update(**self.partial_charges(inputs))
            inputs.update({'no_sigma':True})
            inputs.update({'electrostatic_energy_undamped':self.electrostatic_energy(inputs)['electrostatic_energy']})
            del inputs['no_sigma']
            inputs.update(**self.electrostatic_energy(inputs))
            atomic_energy += inputs['electrostatic_energy']
            if inputs.get('k_smearing', None) is not None:
                cell = inputs['cell']
                if cell.ndim == 3:
                    # Batched training: cell is (num_graphs, 3, 3). Process kspace per-graph.
                    kspace_e = _batched_kspace_electrostatic(
                        self.electrostatic_energy_kspace,
                        inputs,
                    )
                    inputs.update(kspace_e)
                else:
                    inputs.update(**self.electrostatic_energy_kspace(inputs))
                atomic_energy += inputs['electrostatic_energy_kspace']

        if self.dispersion_energy_bool:
            inputs.update(**self.hirshfeld_ratios(inputs))
            inputs.update({'no_sigma':True})
            inputs.update({'dispersion_energy_undamped':self.dispersion_energy(inputs)['dispersion_energy']})
            del inputs['no_sigma']
            inputs.update(**self.dispersion_energy(inputs))
            atomic_energy += inputs['dispersion_energy']
            if inputs.get('k_smearing_disp', None) is not None:
                cell = inputs['cell']
                if cell.ndim == 3:
                    # Batched training: cell is (num_graphs, 3, 3). Process kspace per-graph.
                    kspace_d = _batched_kspace_dispersion(
                        self.dispersion_energy_kspace,
                        inputs,
                    )
                    inputs.update(kspace_d)
                else:
                    inputs.update(**self.dispersion_energy_kspace(inputs))
                atomic_energy += inputs['dispersion_energy_kspace']

        if self.output_convention == 'per_structure':
            energy = segment_sum(
                atomic_energy,
                segment_ids=batch_segments,
                num_segments=num_graphs
            )  # (num_graphs)
            energy = safe_scale(energy, graph_mask)

            result = dict(energy=energy)
            if self.output_intermediate_quantities is not None:
                result.update(self.get_intermediate_quantities(inputs))
            return result

        elif self.output_convention == 'per_atom':
            energy = safe_scale(atomic_energy, node_mask)  # (num_nodes)

            result = dict(energy=energy)
            if self.output_intermediate_quantities is not None:
                result.update(self.get_intermediate_quantities(inputs))
            return result

        else:
            raise ValueError(
                f'{self.output_convention} is invalid argument for attribute `output_convention`.'
            )

    def get_intermediate_quantities(self, inputs: Dict) -> Dict[str, Any]:
        """
        Returns the intermediate quantities used in the energy calculation.
        """
        intermediate_quantities = {}

        for imq_key in self.output_intermediate_quantities:
            # Skip the intermediate quantities that are already defined as observables
            if imq_key in ['dipole_vec', 'energy', 'hirshfeld_ratios', 'a0_ratios', 'c6_ratios']:
                continue
            imq = inputs.get(imq_key)
            if imq is None:
                raise ValueError(f"The requested intermediate quantity {imq_key} could not be generated.")
            intermediate_quantities.update({imq_key:imq})

        return intermediate_quantities

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'zmax': self.zmax,
                                   'output_is_zero_at_init': self.output_is_zero_at_init,
                                   'output_convention': self.output_convention,
                                   'zbl_repulsion_bool': self.zbl_repulsion_bool,
                                   'electrostatic_energy_bool': self.electrostatic_energy_bool,
                                   'dispersion_energy_bool': self.dispersion_energy_bool,
                                   'prop_keys': self.prop_keys}
                }


class HirshfeldSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'hirshfeld_sparse'
    legacy: bool = False  # Use the original vanilla-SO3LR attention head architecture.

    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """Predict Hirshfeld volume ratios from atom-wise features and atomic types.

        Uses an attention-like mechanism with element-dependent queries and learned keys
        to predict per-atom Hirshfeld ratios, plus an element-dependent shift.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (num_nodes, num_features)
                atomic_numbers (Array): Atomic types, shape: (num_nodes)
                node_mask (Array): Node mask, shape: (num_nodes)

        Returns:
            Dict: ``{'hirshfeld_ratios': Array}`` with predicted Hirshfeld ratios,
                shape: (num_nodes).
        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)

        if self.legacy:
            # Original vanilla-SO3LR head: attention-like mechanism with
            # element-dependent query/shift embeddings. Matches the parameter
            # shapes of the legacy checkpoint (Embed_0 (.,1), Embed_1 (.,F/2),
            # dense_final (.,F/2)).
            num_features = x.shape[-1]

            v_shift = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # (num_nodes)
            q = nn.Embed(num_embeddings=100, features=int(num_features / 2))(atomic_numbers)  # (n, F/2)

            if self.regression_dim is not None:
                y = nn.Dense(
                    int(self.regression_dim / 2),
                    kernel_init=nn.initializers.lecun_normal(),
                    name='hirshfeld_ratios_dense_regression'
                )(x)
                y = self.activation_fn(y)
                k = nn.Dense(
                    int(num_features / 2),
                    kernel_init=self.kernel_init,
                    name='hirshfeld_ratios_dense_final'
                )(y)
            else:
                k = nn.Dense(
                    int(num_features / 2),
                    kernel_init=self.kernel_init,
                    name='hirshfeld_ratios_dense_final'
                )(x)

            qk = (q * k / jnp.sqrt(k.shape[-1])).sum(axis=-1)
            v_eff = v_shift + qk  # (n)
            hirshfeld_ratios = safe_scale(jnp.abs(v_eff), node_mask)
            return dict(hirshfeld_ratios=hirshfeld_ratios)

        q = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)

        if self.regression_dim is not None:
            y = nn.Dense(
                int(self.regression_dim / 2),
                kernel_init=nn.initializers.lecun_normal(),
                name='hirshfeld_ratios_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='hirshfeld_ratios_dense_final'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='hirshfeld_ratios_dense_final'
            )(x).squeeze(axis=-1)  # (num_nodes)

        a0_ratios = safe_scale(jnp.abs(x_ + q), node_mask)

        # New models predict static-polarizability (alpha_0) ratios; surface them
        # under 'a0_ratios'. The legacy branch (above) keeps 'hirshfeld_ratios'.
        return dict(a0_ratios=a0_ratios)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention


class C6RatiosSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name = 'c6_ratios'


    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        #TODO: Update docstring
        Predict Hirshfeld volumes from atom-wise features `x` and atomic types `z`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (num_nodes, num_features)
                atomic_numbers (Array): Atomic types, shape: (num_nodes)
                node_mask (Array): Node mask, (num_nodes)
                theory_mask (Array): Theory mask, (num_nodes, num_theory_levels)
            *args ():
            **kwargs ():

        Returns: Dictionary of form {'v_eff': Array}, where Array are the predicted Hirshfeld ratios

        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        q = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)

        if self.regression_dim is not None:
            y = nn.Dense(
#                self.regression_dim,
                int(self.regression_dim / 2),
                kernel_init=nn.initializers.lecun_normal(),
                name='c6_ratios_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='c6_ratios_dense_final'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='c6_ratios_dense_final'
            )(x).squeeze(axis=-1)  # (num_nodes)

        # qk = (q * k / jnp.sqrt(k.shape[-1])).sum(axis=-1)

        # v_eff = v_shift + qk  # shape: (n)
        # c6_ratios = safe_scale(jnp.abs(v_eff), node_mask)

        # return dict(c6_ratios=c6_ratios)
        c6_ratios = safe_scale(jnp.abs(x_ + q), node_mask)

        return dict(c6_ratios=c6_ratios)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

class PartialChargesSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    # Inference-time override for the charge that equilibration conserves to.
    # None -> use the graph's ``total_charge`` (the default; behaviour
    # unchanged). A float -> conserve to that value instead.
    #
    # Note this is a stronger statement than the embedding's counterpart: the
    # embedding is a feature input, whereas this term *enforces* sum(q) = Q. An
    # override here deliberately conserves to a charge the system does not have,
    # which is a research knob rather than a correction. Left at None, nothing
    # changes.
    #
    # Does not apply to the ``residue_charge`` path: when per-residue charges are
    # supplied they keep taking precedence, as before.
    total_charge_override: Optional[float] = None
    module_name: str = 'partial_charges_sparse'

    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """Predict partial atomic charges with total charge conservation.

        Computes unconstrained per-atom charges from node features with an
        element-dependent bias, then adjusts them so that the sum over each
        structure (or residue, if ``residue_charge`` is provided) equals the
        target total charge.

        Args:
            inputs (Dict):
                x (Array): Node features, shape: (num_nodes, num_features)
                atomic_numbers (Array): Atomic numbers, shape: (num_nodes)
                batch_segments (Array): Batch segments, shape: (num_nodes)
                node_mask (Array): Node mask, shape: (num_nodes)
                graph_mask (Array): Graph mask, shape: (num_graphs)
                total_charge (Array): Target total charge per structure, shape: (num_graphs)
                residue_charge (Array, optional): Target charge per residue for dimer calculations.
                residue_segments (Array, optional): Residue assignment per atom.

        Returns:
            Dict: ``{'partial_charges': Array}`` with charge-conserving partial charges,
                shape: (num_nodes).
        """

        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        total_charge = inputs['total_charge']  # (num_graphs)

        if self.total_charge_override is not None:
            # Applied before the residue rebinding below, so residue_charge
            # still wins when it is provided.
            total_charge = jnp.full_like(total_charge, self.total_charge_override)

        num_graphs = len(graph_mask)
        num_nodes = len(node_mask)

        # Element-dependent bias
        q_ = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (num_nodes)

        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='charge_dense_regression_vec'
            )(x)
            y = self.activation_fn(y)
            partial_charges = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                use_bias=True,
                name='charge_dense_final_vec'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            partial_charges = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                use_bias=True,
                name='charge_dense_final_vec'
            )(x).squeeze(axis=-1)  # (num_nodes)

        x_q = safe_scale(partial_charges + q_, node_mask)

        # If residue_charge/segments is provided, use it for charge conservation per residue/monomer
        residue_charge = inputs.get('residue_charge')
        if residue_charge is not None:
            # Residue-based charge conservation
            residue_charge = jnp.asarray(residue_charge, dtype=jnp.float32)
            residue_segments = jnp.pad(
                jnp.asarray(inputs['residue_segments'], dtype=jnp.int32).at[-1].set(2),
                (0, num_nodes - len(inputs['residue_segments'])),
                constant_values=2
            )
            batch_segments, total_charge, num_graphs = residue_segments, residue_charge, residue_charge.shape[0]
        
        # Unified charge conservation calculation
        predicted_charge = segment_sum(x_q, segment_ids=batch_segments, num_segments=num_graphs) # (num_graphs)
        atom_counts = jnp.bincount(batch_segments, length=num_graphs).astype(x_q.dtype)
        safe_atom_counts = jnp.where(atom_counts > 0, atom_counts, jnp.ones_like(atom_counts))
        charge_conservation = (total_charge - predicted_charge) / safe_atom_counts
        partial_charges = x_q + charge_conservation[batch_segments] # (num_nodes)

        return dict(partial_charges=partial_charges)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention


class DipoleVecSparse(BaseSubModule):
    prop_keys: Dict
    partial_charges: Optional[Any] = None
    module_name: str = 'dipole_vec'

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """Compute molecular dipole vectors from partial charges and atomic positions.

        Calculates partial charges via the ``partial_charges`` sub-module, then
        computes per-atom dipole contributions as ``positions * charge`` and sums
        them per structure.

        Args:
            inputs (Dict):
                batch_segments (Array): Batch segments, shape: (num_nodes)
                graph_mask (Array): Graph mask, shape: (num_graphs)
                positions (Array): Atomic positions, shape: (num_nodes, 3)
                (plus all keys required by ``PartialChargesSparse``)

        Returns:
            Dict: ``{'dipole_vec': Array}`` with dipole vectors, shape: (num_graphs, 3).
        """

        batch_segments = inputs['batch_segments']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        positions = inputs.get('positions')  # (num_nodes, 3)

        num_graphs = len(graph_mask)

        # Calculate partial charges
        partial_charges = self.partial_charges(inputs)['partial_charges']

        if positions is None:
            # TODO: do not calculate DipoleVecSparse if there is no positions
            mu_i = 1 * partial_charges[:, None]
        else:
            mu_i = positions * partial_charges[:, None]

        dipole = segment_sum(
            mu_i,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs, 3)

        dipole_vec = safe_scale(dipole, graph_mask[:, None])

        return dict(dipole_vec=dipole_vec)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention


@jax.jit
def sigma(x):
    return safe_mask(x > 0, fn=lambda u: jnp.exp(-1. / u), operand=x, placeholder=0)


@jax.jit
def switching_fn(x, x_on, x_off):
    c = (x - x_on) / (x_off - x_on)
    return sigma(1 - c) / (sigma(1 - c) + sigma(c))


@partial(jax.jit, static_argnames=('neighborlist_format',))
def vdw_QDO_disp_damp(
        R,
        gamma,
        C6,
        alpha_ij,
        gamma_scale,
        neighborlist_format: str = 'sparse'
):
    """Compute damped QDO (Quantum Drude Oscillator) dispersion energy.

    Evaluates C6, C8, and C10 dispersion terms with Tang-Toennies-style damping
    using a power-law regularization of the 1/R^n singularity.

    Args:
        R: Pairwise distances in Bohr.
        gamma: Damping parameter from ``gamma_cubic_fit``.
        C6: Isotropic C6 dispersion coefficients (a.u.).
        alpha_ij: Mean polarizabilities for each pair (a.u.).
        gamma_scale: Scaling factor for the damping radius.
        neighborlist_format: ``'sparse'`` (factor 0.5) or ``'ordered_sparse'`` (factor 1.0).

    Returns:
        Pairwise dispersion energies in eV.
    """
    # Determine the input dtype
    input_dtype = R.dtype

    #  Compute the vdW-QDO dispersion energy (in eV)
    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    C8 = 5 / gamma * C6
    C10 = 245 / 8 / gamma ** 2 * C6
    p = gamma_scale * 2 * 2.54 * alpha_ij ** (1 / 7)

    C8 = jnp.asarray(C8, dtype=input_dtype)
    C10 = jnp.asarray(C10, dtype=input_dtype)
    p = jnp.asarray(p, dtype=input_dtype)

    V3 = -C6 / (jnp.power(R, 6) + jnp.power(p, 6)) - C8 / (jnp.power(R, 8) + jnp.power(p, 8)) - C10 / (
                jnp.power(R, 10) + jnp.power(p, 10))

    return c * V3 * jnp.asarray(Hartree, dtype=input_dtype)

@partial(jax.jit, static_argnames=('neighborlist_format',))
def vdw_QDO_disp_damp_nosigma(
        R,
        gamma,
        C6,
        alpha_ij,
        gamma_scale,
        neighborlist_format: str = 'sparse'
):
    """Compute undamped QDO dispersion energy (no short-range regularization).

    Same as ``vdw_QDO_disp_damp`` but without the power-law damping radius ``p``,
    so the C6/R^6, C8/R^8, and C10/R^10 terms are evaluated directly. Used to
    obtain the undamped reference for computing damping corrections.

    Args:
        R: Pairwise distances in Bohr.
        gamma: Damping parameter from ``gamma_cubic_fit``.
        C6: Isotropic C6 dispersion coefficients (a.u.).
        alpha_ij: Mean polarizabilities for each pair (a.u.).
        gamma_scale: Scaling factor (unused in undamped form, kept for API consistency).
        neighborlist_format: ``'sparse'`` (factor 0.5) or ``'ordered_sparse'`` (factor 1.0).

    Returns:
        Pairwise undamped dispersion energies in eV.
    """
    # Determine the input dtype
    input_dtype = R.dtype

    #  Compute the vdW-QDO dispersion energy (in eV)
    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    C8 = 5 / gamma * C6
    C10 = 245 / 8 / gamma ** 2 * C6

    C8 = jnp.asarray(C8, dtype=input_dtype)
    C10 = jnp.asarray(C10, dtype=input_dtype)

    V3 = -C6 / (jnp.power(R, 6)) - C8 / (jnp.power(R, 8)) - C10 / (
                jnp.power(R, 10))

    return c * V3 * jnp.asarray(Hartree, dtype=input_dtype)


def compute_avg_dispersion_coefficients(
        atomic_numbers: jnp.ndarray,
        hirshfeld_ratios: jnp.ndarray,
        c6_ratios: jnp.ndarray,
):
    """Compute species-weighted average C6, C8, C10 coefficients in atomic units.

    Uses all-pairs mixing rules and gamma fitting to derive average dispersion
    coefficients for use in the analytical tail correction.

    Args:
        atomic_numbers: Atomic numbers, shape ``(N,)``.
        hirshfeld_ratios: Per-atom Hirshfeld volume ratios, shape ``(N,)``.
        c6_ratios: Per-atom C6 ratios, shape ``(N,)``.

    Returns:
        Tuple ``(C6_avg, C8_avg, C10_avg)`` in atomic units (float).
    """
    N = len(atomic_numbers)
    idx_i, idx_j = jnp.triu_indices(N, k=1)
    alpha_ij, C6_ij = mixing_rules(atomic_numbers, idx_i, idx_j, hirshfeld_ratios, c6_ratios)
    gamma_ij = gamma_cubic_fit(alpha_ij)
    C8_ij = 5 / gamma_ij * C6_ij
    C10_ij = 245 / 8 / gamma_ij ** 2 * C6_ij

    # Also include self-interaction (i==i) pairs
    idx_self = jnp.arange(N)
    alpha_self, C6_self = mixing_rules(atomic_numbers, idx_self, idx_self, hirshfeld_ratios, c6_ratios)
    gamma_self = gamma_cubic_fit(alpha_self)
    C8_self = 5 / gamma_self * C6_self
    C10_self = 245 / 8 / gamma_self ** 2 * C6_self

    # Total: sum of all unique pairs (i<j) counted twice + self pairs
    # For the tail correction we need <C_n> = (1/N^2) * sum_{i,j} C_n_ij
    C6_avg = float((2 * jnp.sum(C6_ij) + jnp.sum(C6_self)) / N ** 2)
    C8_avg = float((2 * jnp.sum(C8_ij) + jnp.sum(C8_self)) / N ** 2)
    C10_avg = float((2 * jnp.sum(C10_ij) + jnp.sum(C10_self)) / N ** 2)

    return C6_avg, C8_avg, C10_avg


def dispersion_tail_correction_energy(
        box,
        N: int,
        C6_avg: float,
        C8_avg: float,
        C10_avg: float,
        r_on: float,
        r_off: float,
        n_quad: int = 20,
):
    """Analytical dispersion tail correction energy for NPT simulations.

    Computes the mean-field tail correction for dispersion energy beyond the
    switching region, assuming g(r) = 1 beyond ``r_on``. The correction includes:

    1. A numerical integral over the switching region ``[r_on, r_off]`` where
       dispersion is partially suppressed by ``switching_fn``.
    2. An analytical integral beyond ``r_off`` where dispersion is fully cut off.

    The result is volume-dependent through ``N/V``, so JAX autodiff produces the
    correct pressure correction when used with NPT barostats.

    Args:
        box: Box matrix or diagonal, in Angstrom. Must be differentiable.
        N: Number of atoms.
        C6_avg: Species-averaged C6 coefficient (a.u.).
        C8_avg: Species-averaged C8 coefficient (a.u.).
        C10_avg: Species-averaged C10 coefficient (a.u.).
        r_on: Start of switching region in Angstrom.
        r_off: End of switching region (cutoff) in Angstrom.
        n_quad: Number of Gauss-Legendre quadrature points. Defaults to 20.

    Returns:
        Tail correction energy in eV (scalar, differentiable w.r.t. box).
    """
    _Bohr = jnp.asarray(Bohr, dtype=box.dtype)
    _Hartree = jnp.asarray(Hartree, dtype=box.dtype)
    _pi = jnp.asarray(jnp.pi, dtype=box.dtype)

    # Volume in Bohr^3
    if box.ndim == 1:
        V_ang = box[0] * box[1] * box[2]
    else:
        V_ang = jnp.abs(jnp.linalg.det(box))
    V_bohr = V_ang / (_Bohr ** 3)

    # Number density in Bohr^-3
    rho = jnp.asarray(N, dtype=box.dtype) / V_bohr

    # Convert cutoffs to Bohr
    r_on_bohr = jnp.asarray(r_on, dtype=box.dtype) / _Bohr
    r_off_bohr = jnp.asarray(r_off, dtype=box.dtype) / _Bohr

    _C6 = jnp.asarray(C6_avg, dtype=box.dtype)
    _C8 = jnp.asarray(C8_avg, dtype=box.dtype)
    _C10 = jnp.asarray(C10_avg, dtype=box.dtype)

    # --- Analytical integral beyond r_off (where w=1 so contribution is fully missing) ---
    # int_{r_off}^{inf} 4*pi*r^2 * (-C6/r^6 - C8/r^8 - C10/r^10) dr
    # = -4*pi * (C6/(3*r_off^3) + C8/(5*r_off^5) + C10/(7*r_off^7))
    I_beyond = -4 * _pi * (
        _C6 / (3 * r_off_bohr ** 3) +
        _C8 / (5 * r_off_bohr ** 5) +
        _C10 / (7 * r_off_bohr ** 7)
    )

    # --- Numerical integral over switching region [r_on, r_off] ---
    # In this region the switching function w(r) suppresses the dispersion.
    # The missing energy is: int 4*pi*r^2 * [1-w(r)] * (-C6/r^6 - C8/r^8 - C10/r^10) dr
    # Use Gauss-Legendre quadrature.
    nodes, weights = _gauss_legendre_nodes_weights(n_quad, box.dtype)
    # Map from [-1, 1] to [r_on_bohr, r_off_bohr]
    half_width = (r_off_bohr - r_on_bohr) / 2
    mid = (r_off_bohr + r_on_bohr) / 2
    r_quad = mid + half_width * nodes  # quadrature points in Bohr

    # switching_fn expects Angstrom inputs
    r_quad_ang = r_quad * _Bohr
    w = switching_fn(r_quad_ang, r_on, r_off)

    integrand = (1 - w) * 4 * _pi * r_quad ** 2 * (
        -_C6 / r_quad ** 6 - _C8 / r_quad ** 8 - _C10 / r_quad ** 10
    )
    I_switch = jnp.sum(weights * integrand) * half_width

    # E_tail = 0.5 * N * rho * (I_switch + I_beyond) in Hartree, convert to eV
    E_tail = 0.5 * jnp.asarray(N, dtype=box.dtype) * rho * (I_switch + I_beyond) * _Hartree

    return E_tail


def c6_raw_potential():
    """Return a jaxpme RawPotential for the C6/r^6 dispersion interaction.

    Uses numerically stable formulas for the Ewald splitting of 1/r^6:
    - Short-range: exp(-x)*(1 + x + x^2/2) / r^6  where x = r^2/(2*sigma^2)
    - Long-range real-space: [1 - exp(-x)*(1 + x + x^2/2)] / r^6
    - Long-range k-space: via upper incomplete gamma Gamma(-3/2, x) computed
      through recursion from Gamma(1/2, x) = sqrt(pi)*erfc(sqrt(x))
    - Self correction: 1 / (6*(2*sigma^2)^3)
    - Background: pi^(3/2) * (2*sigma^2)^(-3/2) / (-6)
    """
    from jaxpme.potentials import RawPotential

    def sr_r(smearing, distances):
        sigma = smearing
        two_sigma2 = 2.0 * sigma ** 2
        x = distances ** 2 / two_sigma2
        exp_neg_x = jnp.exp(-x)
        return exp_neg_x * (1.0 + x + 0.5 * x ** 2) / distances ** 6

    def lr_r(smearing, distances):
        sigma = smearing
        two_sigma2 = 2.0 * sigma ** 2
        x = distances ** 2 / two_sigma2
        exp_neg_x = jnp.exp(-x)
        return (1.0 - exp_neg_x * (1.0 + x + 0.5 * x ** 2)) / distances ** 6

    def lr_k2(smearing, k2):
        sigma = smearing
        two_sigma2 = 2.0 * sigma ** 2
        x = sigma ** 2 * k2 / 2.0  # = two_sigma2 * k2 / 4

        sqrt_x = jnp.sqrt(jnp.maximum(x, 1e-30))
        # Gamma(1/2, x) = sqrt(pi) * erfc(sqrt(x))
        G12 = jnp.sqrt(jnp.pi) * jax.lax.erfc(sqrt_x)
        # Recursion: Gamma(a-1, x) = (Gamma(a, x) - x^(a-1)*exp(-x)) / (a - 1)
        # Gamma(-1/2, x) = (Gamma(1/2, x) - x^(-1/2)*exp(-x)) / (-1/2)
        #                 = -2 * (G12 - exp(-x) / sqrt(x))
        exp_neg_x = jnp.exp(-x)
        G_neg12 = -2.0 * (G12 - exp_neg_x / sqrt_x)
        # Gamma(-3/2, x) = (Gamma(-1/2, x) - x^(-3/2)*exp(-x)) / (-3/2)
        #                 = -2/3 * (G_neg12 - exp(-x) / x^(3/2))
        G_neg32 = -2.0 / 3.0 * (G_neg12 - exp_neg_x / (x * sqrt_x))

        # lr_k2 = (pi^(3/2) / 2) * (2*sigma^2)^(-3/2) * G_neg32 * x^(3/2)
        factor = jnp.pi ** 1.5 / 2.0 * two_sigma2 ** (-1.5)
        return factor * G_neg32 * x ** 1.5

    def correction_self(smearing):
        sigma = smearing
        two_sigma2 = 2.0 * sigma ** 2
        return 1.0 / (6.0 * two_sigma2 ** 3)

    def correction_background(smearing):
        sigma = smearing
        two_sigma2 = 2.0 * sigma ** 2
        return jnp.pi ** 1.5 * two_sigma2 ** (-1.5) / (-6.0)

    def real(distances):
        return 1.0 / distances ** 6

    def correction_pbc(positions, cell, charges, pbc):
        return jnp.zeros(positions.shape[0], dtype=positions.dtype)

    return RawPotential(
        sr_r, lr_r, lr_k2, real,
        correction_background, correction_self, correction_pbc,
    )


def _gauss_legendre_nodes_weights(n: int, dtype=jnp.float64):
    """Return Gauss-Legendre quadrature nodes and weights on [-1, 1]."""
    import numpy as _np
    nodes, weights = _np.polynomial.legendre.leggauss(n)
    return jnp.asarray(nodes, dtype=dtype), jnp.asarray(weights, dtype=dtype)


@partial(jax.jit, static_argnames=('legacy',))
def mixing_rules(
        atomic_numbers: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        hirshfeld_ratios: jnp.ndarray,
        c6_ratios: jnp.ndarray,
        legacy: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Casimir-Polder mixing rules to obtain pairwise dispersion parameters.

    Scales free-atom reference polarizabilities and C6 coefficients by Hirshfeld
    volume ratios, then combines them into pairwise quantities using the
    Casimir-Polder combination rule for C6 coefficients.

    Args:
        atomic_numbers: Atomic numbers, shape: (num_nodes).
        idx_i: Sender indices for long-range edges.
        idx_j: Receiver indices for long-range edges.
        hirshfeld_ratios: Per-atom Hirshfeld volume ratios, shape: (num_nodes).
        c6_ratios: Per-atom C6 volume ratios, shape: (num_nodes).
        legacy: If True, use the legacy SO3LR (v1) free-atom reference
            coefficients (``alphas_legacy``/``C6_coef_legacy``) instead of the
            refitted ones used by the so3lr-s/m/l models.

    Returns:
        Tuple of (alpha_ij, C6_ij): mean polarizabilities and combined C6
        coefficients for each pair.
    """
    dtype = hirshfeld_ratios.dtype

    # Free-atom reference coefficients: legacy v1 vs refitted (so3lr-s/m/l).
    alphas_ref = alphas_legacy if legacy else alphas
    atomic_number_i = atomic_numbers[idx_i] - 1
    atomic_number_j = atomic_numbers[idx_j] - 1
    hirshfeld_ratio_i = hirshfeld_ratios[idx_i]
    hirshfeld_ratio_j = hirshfeld_ratios[idx_j]

    # Cast the reference tables to the working dtype BEFORE indexing them.
    # They are numpy constants captured at trace time, so their dtype used to
    # follow whatever jax_enable_x64 was when this function was first traced;
    # a later trace under the other setting then re-used the cached `_take`
    # with a mismatched operand ("expected tensor<86xf64>, provided
    # tensor<86xf32>"). Indexing an already-cast table makes the operand dtype
    # part of the trace rather than a side effect of global state.
    alphas_ref = jnp.asarray(alphas_ref, dtype=dtype)
    alpha_i = jnp.take(alphas_ref, atomic_number_i, axis=0) * hirshfeld_ratio_i
    C6_i = _effective_atomic_c6(
        atomic_numbers[idx_i], c6_ratios[idx_i], legacy=legacy, dtype=dtype
    )
    alpha_j = jnp.take(alphas_ref, atomic_number_j, axis=0) * hirshfeld_ratio_j
    C6_j = _effective_atomic_c6(
        atomic_numbers[idx_j], c6_ratios[idx_j], legacy=legacy, dtype=dtype
    )
    
    alpha_ij = (alpha_i + alpha_j) / 2
    C6_ij = 2 * C6_i * C6_j * alpha_j * alpha_i / (alpha_i ** 2 * C6_j + alpha_j ** 2 * C6_i)

    return alpha_ij, C6_ij


@jax.jit
def gamma_cubic_fit(alpha):
    """Compute the QDO damping parameter gamma from polarizabilities via cubic fit.

    Converts polarizabilities to van der Waals radii, then applies a cubic
    polynomial fit to obtain sigma, from which gamma = 1/(2*sigma^2).

    Args:
        alpha: Pairwise mean polarizabilities (a.u.).

    Returns:
        Damping parameter gamma for each pair.
    """
    input_dtype = alpha.dtype

    vdW_radius = fine_structure ** (jnp.asarray(-4. / 21, input_dtype)) * alpha ** jnp.asarray(1. / 7, input_dtype)
    b0 = jnp.asarray(-0.00433008, dtype=input_dtype)
    b1 = jnp.asarray(0.24428889, dtype=input_dtype)
    b2 = jnp.asarray(0.04125273, dtype=input_dtype)
    b3 = jnp.asarray(-0.00078893, dtype=input_dtype)

    sigma = b3 * jnp.power(vdW_radius, 3) + b2 * jnp.square(vdW_radius) + b1 * vdW_radius + b0
    gamma = jnp.asarray(1. / 2, dtype=input_dtype) / jnp.square(sigma)
    return gamma


@partial(jax.jit, static_argnames=('neighborlist_format',))
def coulomb_bare(
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        ke: float,
        neighborlist_format: str = 'sparse'
) -> jnp.ndarray:
    """ Pairwise bare Coulomb interaction (undamped 1/r) """
    input_dtype = rij.dtype

    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    _ke = jnp.asarray(ke, dtype=input_dtype)
    return c * _ke * q[idx_i] * q[idx_j] / rij


@partial(jax.jit, static_argnames=('neighborlist_format',))
def coulomb_erf(
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        ke: float,
        sigma: float,
        cutoff: float = None,
        neighborlist_format: str = 'sparse'
) -> jnp.ndarray:
    """ Pairwise Coulomb interaction with erf damping """
    input_dtype = rij.dtype

    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    # Cast constants to input dtype
    _ke = jnp.asarray(ke, dtype=input_dtype)
    _sigma = jnp.asarray(sigma, dtype=input_dtype)

    pairwise = c * _ke * q[idx_i] * q[idx_j] / rij
    if cutoff is None:
        return pairwise * jax.lax.erf(rij / _sigma)
    else:
        _cutoff = jnp.asarray(cutoff, dtype=input_dtype)
        return pairwise * (jax.lax.erf(rij / _sigma) - jax.lax.erf(rij / (_cutoff * jnp.sqrt(2.0))))

@partial(jax.jit, static_argnames=('neighborlist_format',))
def coulomb_erf_shifted_force_smooth_pme(
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        ke: float,
        sigma: float,
        cutoff: float = None,
        cuton: float = None,
        smearing: float = None,
        neighborlist_format: str = 'sparse'
) -> jnp.ndarray:
    """Pairwise Coulomb interaction with erf damping for PME (real-space part).

    Computes the short-range real-space contribution to the Coulomb energy for use
    with Particle Mesh Ewald. The potential is ``erf(r/sigma)/r - erf(r/smearing)/r``,
    smoothed to zero at the cutoff using a switching function and shifted-force correction.

    Args:
        q: Partial charges, shape: (num_nodes).
        rij: Pairwise distances, shape: (num_edges).
        idx_i: Sender atom indices.
        idx_j: Receiver atom indices.
        ke: Coulomb constant in eV*Angstrom/e^2.
        sigma: Short-range damping width (Angstrom).
        cutoff: Real-space cutoff distance (Angstrom).
        cuton: Distance at which the switching function begins.
        smearing: Ewald smearing parameter (Angstrom).
        neighborlist_format: ``'sparse'`` or ``'ordered_sparse'``.

    Returns:
        Pairwise electrostatic energies, shape: (num_edges).
    """

    input_dtype = rij.dtype

    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    _ke = jnp.asarray(ke, dtype=input_dtype)
    _sigma = jnp.asarray(sigma, dtype=input_dtype)
    _smearing = jnp.asarray(smearing, dtype=input_dtype)* jnp.sqrt(2.0)
    _cuton = jnp.asarray(cuton, dtype=input_dtype)

    def potential(r):
        return jax.lax.erf(r / _sigma) / r - jax.lax.erf(r / _smearing ) / r

    def force1(r,cut):
        return (2 * r * jnp.exp(-(r / cut) ** 2) / (jnp.sqrt(jnp.pi) * cut) - jax.lax.erf(r / cut)) / r ** 2

    def force(r):
        return force1(r, _sigma) - force1(r, _smearing)

    _cutoff = jnp.asarray(cutoff, dtype=input_dtype)
    f = switching_fn(rij, _cuton, _cutoff)
    pairwise = potential(rij)
    shift = potential(_cutoff)
    force_shift = force(_cutoff)

    shifted_potential = pairwise - shift - force_shift * (rij - _cutoff)

    return jnp.where(
        rij < _cutoff,
        c * _ke * q[idx_i] * q[idx_j] * (f * (pairwise - shift) + (1 - f) * shifted_potential),
        0.0
    )

@partial(jax.jit, static_argnames=('neighborlist_format',))
def coulomb_erf_shifted_force_smooth_pme_nosigma(
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        ke: float,
        sigma: float,
        cutoff: float = None,
        cuton: float = None,
        smearing: float = None,
        neighborlist_format: str = 'sparse'
) -> jnp.ndarray:
    """Pairwise Coulomb for PME without short-range sigma damping.

    Same as ``coulomb_erf_shifted_force_smooth_pme`` but with undamped bare Coulomb
    ``1/r`` instead of ``erf(r/sigma)/r``. The potential is ``1/r - erf(r/smearing)/r``,
    used to obtain the undamped reference for computing damping corrections.

    Args:
        q: Partial charges, shape: (num_nodes).
        rij: Pairwise distances, shape: (num_edges).
        idx_i: Sender atom indices.
        idx_j: Receiver atom indices.
        ke: Coulomb constant in eV*Angstrom/e^2.
        sigma: Unused (kept for API consistency).
        cutoff: Real-space cutoff distance (Angstrom).
        cuton: Distance at which the switching function begins.
        smearing: Ewald smearing parameter (Angstrom).
        neighborlist_format: ``'sparse'`` or ``'ordered_sparse'``.

    Returns:
        Pairwise electrostatic energies, shape: (num_edges).
    """

    input_dtype = rij.dtype

    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    _ke = jnp.asarray(ke, dtype=input_dtype)
    _sigma = jnp.asarray(sigma, dtype=input_dtype)
    _smearing = jnp.asarray(smearing, dtype=input_dtype)* jnp.sqrt(2.0)
    _cuton = jnp.asarray(cuton, dtype=input_dtype)

    def potential(r):
        return 1.0 / r - jax.lax.erf(r / _smearing ) / r

    def force1(r,cut):
        return (2 * r * jnp.exp(-(r / cut) ** 2) / (jnp.sqrt(jnp.pi) * cut) - jax.lax.erf(r / cut)) / r ** 2

    def force(r):
        return - 1.0 / r ** 2 - force1(r, _smearing)

    _cutoff = jnp.asarray(cutoff, dtype=input_dtype)
    f = switching_fn(rij, _cuton, _cutoff)
    pairwise = potential(rij)
    shift = potential(_cutoff)
    force_shift = force(_cutoff)

    shifted_potential = pairwise - shift - force_shift * (rij - _cutoff)

    return jnp.where(
        rij < _cutoff,
        c * _ke * q[idx_i] * q[idx_j] * (f * (pairwise - shift) + (1 - f) * shifted_potential),
        0.0
    )

@partial(jax.jit, static_argnames=('neighborlist_format',))
def coulomb_erf_shifted_force_smooth(
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        ke: float,
        sigma: float,
        cutoff: float,
        cuton: float,
        neighborlist_format: str = 'sparse'
) -> jnp.ndarray:
    """Pairwise Coulomb interaction with erf damping, shifted-force, and smooth switching.

    Computes the erf-damped Coulomb potential ``erf(r/sigma)/r`` with a shifted-force
    correction to ensure continuity at the cutoff, blended via a smooth switching
    function between ``cuton`` and ``cutoff``.

    Args:
        q: Partial charges, shape: (num_nodes).
        rij: Pairwise distances, shape: (num_edges).
        idx_i: Sender atom indices.
        idx_j: Receiver atom indices.
        ke: Coulomb constant in eV*Angstrom/e^2.
        sigma: Damping width for the erf function (Angstrom).
        cutoff: Cutoff distance (Angstrom).
        cuton: Distance at which the switching function begins (Angstrom).
        neighborlist_format: ``'sparse'`` or ``'ordered_sparse'``.

    Returns:
        Pairwise electrostatic energies, shape: (num_edges).
    """

    input_dtype = rij.dtype

    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    # Cast the constants to input dtype
    _sigma = jnp.asarray(sigma, dtype=input_dtype)
    _ke = jnp.asarray(ke, dtype=input_dtype)
    _cutoff = jnp.asarray(cutoff, dtype=input_dtype)
    _cuton = jnp.asarray(cuton, dtype=input_dtype)

    def potential(r):
        return jax.lax.erf(r / _sigma) / r

    def force(r):
        return (2 * r * jnp.exp(-(r / _sigma) ** 2) / (jnp.sqrt(jnp.pi) * _sigma) - jax.lax.erf(r / _sigma)) / r ** 2

    f = switching_fn(rij, _cuton, _cutoff)
    pairwise = potential(rij)
    shift = potential(_cutoff)
    force_shift = force(_cutoff)

    shifted_potential = pairwise - shift - force_shift * (rij - _cutoff)

    return jnp.where(
        rij < _cutoff,
        c * _ke * q[idx_i] * q[idx_j] * (f * (pairwise - shift) + (1 - f) * shifted_potential),
        0.0
    )

@partial(jax.jit, static_argnames=('neighborlist_format',))
def coulomb_erf_shifted_force_smooth_nosigma(
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        ke: float,
        sigma: float,
        cutoff: float,
        cuton: float,
        neighborlist_format: str = 'sparse'
) -> jnp.ndarray:
    """Pairwise bare Coulomb interaction with shifted-force and smooth switching.

    Same as ``coulomb_erf_shifted_force_smooth`` but without erf damping: the
    potential is simply ``1/r``. Used to obtain the undamped reference for
    computing damping corrections.

    Args:
        q: Partial charges, shape: (num_nodes).
        rij: Pairwise distances, shape: (num_edges).
        idx_i: Sender atom indices.
        idx_j: Receiver atom indices.
        ke: Coulomb constant in eV*Angstrom/e^2.
        sigma: Unused (kept for API consistency).
        cutoff: Cutoff distance (Angstrom).
        cuton: Distance at which the switching function begins (Angstrom).
        neighborlist_format: ``'sparse'`` or ``'ordered_sparse'``.

    Returns:
        Pairwise electrostatic energies, shape: (num_edges).
    """

    input_dtype = rij.dtype

    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    # Cast the constants to input dtype
    _ke = jnp.asarray(ke, dtype=input_dtype)
    _cutoff = jnp.asarray(cutoff, dtype=input_dtype)
    _cuton = jnp.asarray(cuton, dtype=input_dtype)

    def potential(r):
        return 1.0 / r

    def force(r):
        return - 1.0 / r ** 2

    f = switching_fn(rij, _cuton, _cutoff)
    pairwise = potential(rij)
    shift = potential(_cutoff)
    force_shift = force(_cutoff)

    shifted_potential = pairwise - shift - force_shift * (rij - _cutoff)

    return jnp.where(
        rij < _cutoff,
        c * _ke * q[idx_i] * q[idx_j] * (f * (pairwise - shift) + (1 - f) * shifted_potential),
        0.0
    )

class ZBLRepulsionSparse(BaseSubModule):
    """
    Ziegler-Biersack-Littmark repulsion (original vanilla-SO3LR, learnable).

    Used by the legacy model. The newer models use NLHRepulsionSparse instead;
    both share module_name='zbl_repulsion' so the checkpoint key matches whichever
    one the model config selects.
    """
    prop_keys: Dict
    # input_convention: str = 'positions'
    module_name: str = 'zbl_repulsion'
    a0: float = 0.5291772105638411
    ke: float = 14.399645351950548

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        a1 = softplus(self.param('a1', constant(softplus_inverse(3.20000)), (1,)))  # shape: (1)
        a2 = softplus(self.param('a2', constant(softplus_inverse(0.94230)), (1,)))  # shape: (1)
        a3 = softplus(self.param('a3', constant(softplus_inverse(0.40280)), (1,)))  # shape: (1)
        a4 = softplus(self.param('a4', constant(softplus_inverse(0.20160)), (1,)))  # shape: (1)
        c1 = softplus(self.param('c1', constant(softplus_inverse(0.18180)), (1,)))  # shape: (1)
        c2 = softplus(self.param('c2', constant(softplus_inverse(0.50990)), (1,)))  # shape: (1)
        c3 = softplus(self.param('c3', constant(softplus_inverse(0.28020)), (1,)))  # shape: (1)
        c4 = softplus(self.param('c4', constant(softplus_inverse(0.02817)), (1,)))  # shape: (1)
        p = softplus(self.param('p', constant(softplus_inverse(0.23)), (1,)))  # shape: (1)
        d = softplus(self.param('d', constant(softplus_inverse(1 / (0.8854 * self.a0))), (1,)))  # shape: (1)

        c_sum = c1 + c2 + c3 + c4
        c1 = c1 / c_sum
        c2 = c2 / c_sum
        c3 = c3 / c_sum
        c4 = c4 / c_sum

        atomic_numbers = inputs['atomic_numbers']
        node_mask = inputs['node_mask']
        phi_r_cut_ij = inputs['cut']
        idx_i = inputs['idx_i']
        idx_j = inputs['idx_j']
        d_ij = inputs['d_ij']

        num_nodes = len(node_mask)

        z_i = atomic_numbers[idx_i]
        z_j = atomic_numbers[idx_j]

        z_d_ij = safe_mask(mask=d_ij != 0,
                           operand=d_ij,
                           fn=lambda u: z_i * z_j / u,
                           placeholder=0.
                           )

        x = self.ke * phi_r_cut_ij * z_d_ij

        rzd = d_ij * (jnp.power(z_i, p) + jnp.power(z_j, p)) * d
        y = c1 * jnp.exp(-a1 * rzd) + c2 * jnp.exp(-a2 * rzd) + c3 * jnp.exp(-a3 * rzd) + c4 * jnp.exp(-a4 * rzd)

        w = switching_fn(d_ij, x_on=0, x_off=1.5)

        e_rep_edge = w * x * y / jnp.asarray(2, dtype=d_ij.dtype)
        e_rep_edge = segment_sum(e_rep_edge, segment_ids=idx_i, num_segments=num_nodes)
        e_rep_edge = safe_scale(e_rep_edge, node_mask)

        return dict(zbl_repulsion=e_rep_edge)

    def reset_output_convention(self, output_convention):
        pass


class NLHRepulsionSparse(BaseSubModule):
    """
    NLH repulsion (used by the newer so3lr-s/m/l models).
    """
    prop_keys: Dict
    module_name: str = 'zbl_repulsion'
    a0: float = 0.5291772105638411
    ke: float = 14.399645351950548

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        atomic_numbers = inputs['atomic_numbers']
        node_mask = inputs['node_mask']
        phi_r_cut_ij = inputs['cut']
        idx_i = inputs['idx_i']
        idx_j = inputs['idx_j']
        d_ij = inputs['d_ij']
        dtype = d_ij.dtype

        num_nodes = len(node_mask)

        # Gather atomic numbers per edge, start from 0 so -1
        z_i = atomic_numbers[idx_i] - 1
        z_j = atomic_numbers[idx_j] - 1

        # Symmetrize for table lookup so we always hit the filled triangle
        zi = jnp.minimum(z_i, z_j)                       # (E,)
        zj = jnp.maximum(z_i, z_j)                       # (E,)

        # Get NLH coefficients
        # Two-index lookup straight into the (92, 92, 3) tables. Gathering along
        # axis 0 first materialised an (E, 92, 3) intermediate per table only to
        # keep one row of it -- ~550 MB per table for a 5e5-edge system, twice
        # over, plus the autodiff residuals. Result is numerically identical.
        # NLH_AA/NLH_BB are numpy arrays, so they must become jnp arrays before
        # being indexed with traced zi/zj (numpy would call __array__ on a tracer).
        a = jnp.asarray(NLH_AA, dtype=dtype)[zi, zj]               # (E, 3)
        b = jnp.asarray(NLH_BB, dtype=dtype)[zi, zj]               # (E, 3)

        # Coulomb factor ke * Z_i * Z_j / r (dtype-safe) 
        z_over_r = safe_mask(
            mask=(d_ij != 0),
            operand=d_ij,
            fn=lambda u: (z_i + 1) * (z_j + 1) / u,
            placeholder=0.0
        )                                    
        x = self.ke * phi_r_cut_ij * z_over_r

        # NLH screening phi(r)
        y = jnp.sum(a * jnp.exp(-b * d_ij[:, None]), axis=-1)

        # Switching, edge energy, and reduction
        w = switching_fn(d_ij, x_on=0.0, x_off=1.5)

        e_rep_edge = w * x * y / jnp.asarray(2, dtype=d_ij.dtype)
        e_rep_edge = segment_sum(e_rep_edge, segment_ids=idx_i, num_segments=num_nodes)
        e_rep_edge = safe_scale(e_rep_edge, node_mask)

        # Expose under both keys: 'zbl_repulsion' keeps the energy-summation /
        # checkpoint path working (shared module_name), 'nlh_repulsion' is the
        # clear name for this model's repulsion, requestable as an observable.
        return dict(zbl_repulsion=e_rep_edge, nlh_repulsion=e_rep_edge)

    def reset_output_convention(self, output_convention):
        pass


class ElectrostaticEnergySparse(BaseSubModule):
    prop_keys: Dict
    partial_charges: Any
    cutoff_lr: float
    ke: float = 14.399645351950548
    electrostatic_energy_scale: float = 1.0
    neighborlist_format: str = 'sparse'  # or 'ordered_sparse'
    module_name: str = 'electrostatic_energy'

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        """Compute per-atom electrostatic energy from partial charges and long-range neighbor list.

        Selects the appropriate Coulomb kernel depending on whether Ewald (PME)
        splitting is active (``k_smearing`` present) and whether sigma damping
        is applied (``no_sigma`` flag). Sums pairwise contributions per atom.

        Args:
            inputs (Dict): Must contain ``node_mask``, ``idx_i_lr``, ``idx_j_lr``,
                ``d_ij_lr``, ``partial_charges`` (or keys for computing them), and
                optionally ``k_smearing`` and ``no_sigma``.

        Returns:
            Dict: ``{'electrostatic_energy': Array}`` with per-atom electrostatic
                energies, shape: (num_nodes).
        """
        node_mask = inputs['node_mask']  # (num_nodes)
        num_nodes = len(node_mask)
        idx_i_lr = inputs['idx_i_lr']
        idx_j_lr = inputs['idx_j_lr']
        d_ij_lr = inputs['d_ij_lr']
        k_smearing = inputs.get('k_smearing',None)
        no_sigma = inputs.get('no_sigma',None)

        # Calculate partial charges
        partial_charges = inputs.get('partial_charges')
        if partial_charges is None:
            partial_charges = self.partial_charges(inputs)['partial_charges']

        # If cutoff is set, we apply damping with error function with smoothing to zero at cutoff_lr.
        # We also apply force shifting to reduce discontinuity artifacts.
        if self.cutoff_lr is not None:
            if k_smearing is None:
                if no_sigma is not None:
                    # Calculate electrostatic energies per long-range edge
                    atomic_electrostatic_energy_ij = coulomb_erf_shifted_force_smooth_nosigma(
                    partial_charges,
                    d_ij_lr,
                    idx_i_lr,
                    idx_j_lr,
                    ke=self.ke,
                    sigma=self.electrostatic_energy_scale,
                    cutoff=self.cutoff_lr,
                    cuton=self.cutoff_lr * 0.45,
                    neighborlist_format=self.neighborlist_format
                    )
                else:
                    # Calculate electrostatic energies per long-range edge
                    atomic_electrostatic_energy_ij = coulomb_erf_shifted_force_smooth(
                    partial_charges,
                    d_ij_lr,
                    idx_i_lr,
                    idx_j_lr,
                    ke=self.ke,
                    sigma=self.electrostatic_energy_scale,
                    cutoff=self.cutoff_lr,
                    cuton=self.cutoff_lr * 0.45,
                    neighborlist_format=self.neighborlist_format
                    )
            else:
                if no_sigma is not None:
                    atomic_electrostatic_energy_ij = coulomb_erf_shifted_force_smooth_pme_nosigma(
                    partial_charges,
                    d_ij_lr,
                    idx_i_lr,
                    idx_j_lr,
                    ke=self.ke,
                    sigma=self.electrostatic_energy_scale,
                    cutoff=self.cutoff_lr,
                    cuton=self.cutoff_lr * 0.45,
                    smearing=k_smearing,
                    neighborlist_format=self.neighborlist_format
                    )
                else:
                    atomic_electrostatic_energy_ij = coulomb_erf_shifted_force_smooth_pme(
                    partial_charges,
                    d_ij_lr,
                    idx_i_lr,
                    idx_j_lr,
                    ke=self.ke,
                    sigma=self.electrostatic_energy_scale,
                    cutoff=self.cutoff_lr,
                    cuton=self.cutoff_lr * 0.45,
                    smearing=k_smearing,
                    neighborlist_format=self.neighborlist_format
                    )

        # If no cutoff is set, we just apply damping with error function and no explicit smoothing to zero.
        else:
            if no_sigma is not None:
                # Undamped bare Coulomb (no erf, no cutoff)
                atomic_electrostatic_energy_ij = coulomb_bare(
                    partial_charges,
                    d_ij_lr,
                    idx_i_lr,
                    idx_j_lr,
                    ke=self.ke,
                    neighborlist_format=self.neighborlist_format
                )
            else:
                # Calculate electrostatic energies per long-range edge
                atomic_electrostatic_energy_ij = coulomb_erf(
                    partial_charges,
                    d_ij_lr,
                    idx_i_lr,
                    idx_j_lr,
                    ke=self.ke,
                    sigma=self.electrostatic_energy_scale,
                    cutoff=None,
                    neighborlist_format=self.neighborlist_format
                )            

        # Calculate electrostatic atomic energies via summing over long-range neighbors
        atomic_electrostatic_energy = segment_sum(
            atomic_electrostatic_energy_ij,
            segment_ids=idx_i_lr,
            num_segments=num_nodes
        )  # (num_nodes)

        # Mask padded nodes
        atomic_electrostatic_energy = safe_scale(atomic_electrostatic_energy, node_mask)

        return dict(electrostatic_energy=atomic_electrostatic_energy)

    def reset_output_convention(self, output_convention):
        pass

class ElectrostaticEnergyKspace(BaseSubModule):
    prop_keys: Dict
    partial_charges: Any
    kspace_method: str = 'pme'  # 'ewald', 'pme', or 'p3m'
    interpolation_nodes: int = 4
    ke: float = 14.399645351950548
    electrostatic_energy_scale: float = 1.0
    module_name: str = "electrostatic_energy_kspace"

    def setup(self):

        if self.kspace_method == 'ewald':
            self.solver =  jaxpme.solvers.ewald(jaxpme.potentials.potential())
        elif self.kspace_method == 'pme':
            self.solver =  jaxpme.solvers.pme(jaxpme.potentials.potential(), interpolation_nodes=self.interpolation_nodes)
        elif self.kspace_method == 'p3m':
            self.solver = jaxpme.solvers.p3m(jaxpme.potentials.potential(), interpolation_nodes=self.interpolation_nodes)
        else:
            raise ValueError(
                f"Invalid kspace_method: {self.kspace_method}. Must be 'ewald', 'pme', or 'p3m'."
            )
    
    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        """Compute per-atom k-space electrostatic energy via Ewald or PME.

        Evaluates the reciprocal-space contribution to the electrostatic energy
        using either standard Ewald summation or Particle Mesh Ewald (PME),
        depending on the ``kspace_method`` attribute.

        Args:
            inputs (Dict): Must contain ``positions``, ``k_grid``, ``k_smearing``,
                ``cell``, ``node_mask``, and ``partial_charges`` (or keys for computing them).

        Returns:
            Dict: ``{'electrostatic_energy_kspace': Array}`` with per-atom k-space
                electrostatic energies, shape: (num_nodes).
        """

        positions = inputs['positions']
        k_grid = inputs['k_grid']
        k_smearing = inputs['k_smearing']
        cell = inputs['cell']
        node_mask = inputs["node_mask"]
        # Calculate partial charges
        partial_charges = inputs.get('partial_charges')
        if partial_charges is None:
            partial_charges = self.partial_charges(inputs)['partial_charges']

        assert positions is not None, "Positions must be provided for k-space calculation."
        assert k_grid is not None, "k_grid must be provided for k-space calculation."
        assert cell is not None, "Cell must be provided for k-space calculation."
        assert k_smearing is not None, "k_smearing must be provided for k-space calculation."
        assert cell.shape == (3, 3), f"Invalid cell shape {cell.shape}. Expected (3, 3)."

        volume = jnp.abs(jnp.linalg.det(cell))
        reciprocal_cell = jaxpme.kspace.get_reciprocal(cell)
        kvectors = jaxpme.kspace.generate_kvectors(
            reciprocal_cell, k_grid.shape, dtype=positions.dtype, for_ewald=(self.kspace_method == 'ewald')
        )
        pbc = jnp.array([True, True, True])

        #if node_mask is not None:
        #    partial_charges *= node_mask

        if self.kspace_method == 'ewald':
            potentials = self.solver.kspace(k_smearing, partial_charges, kvectors, positions, volume, cell, pbc)
        else:  # 'pme' or 'p3m'
            potentials = self.solver.kspace(k_smearing, partial_charges, reciprocal_cell, k_grid, kvectors, positions, volume, cell, pbc)

        #if node_mask is not None:
        #    potentials *= node_mask

        energies = partial_charges * potentials
        energies *= self.ke

        # Mask padded nodes
        energies = safe_scale(energies, node_mask)

        return dict(electrostatic_energy_kspace=energies)

    def reset_output_convention(self, output_convention):
        pass


class DispersionEnergyKspace(BaseSubModule):
    """K-space dispersion energy via PME or Ewald for the C6/r^6 interaction.

    Uses geometric-mean charges q_i = sqrt(C6_free[Z_i] * c6_ratio_i) for PME.
    ``C6_free`` comes from the legacy or refitted table to match the model
    generation, with the Casimir-Polder residual handled in real space by
    :class:`DispersionEnergySparse`.
    """
    prop_keys: Dict
    hirshfeld_ratios: Optional[Any] = None
    c6_ratios: Optional[Any] = None
    do_ewald: bool = False
    interpolation_nodes: int = 4
    legacy: bool = False  # Legacy SO3LR: derive C6 from Hirshfeld ratios (no c6 head).
    module_name: str = "dispersion_energy_kspace"

    def setup(self):
        from jaxpme.solvers import ewald, pme
        from jaxpme.potentials import potential as get_potential

        custom_pot = c6_raw_potential()
        if self.do_ewald:
            self.solver = ewald(get_potential(custom_potential=custom_pot))
        else:
            self.solver = pme(
                get_potential(custom_potential=custom_pot),
                interpolation_nodes=self.interpolation_nodes,
            )

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        """Compute per-atom k-space C6 dispersion energy via Ewald or PME.

        Args:
            inputs (Dict): Must contain ``positions``, ``k_grid_disp``,
                ``k_smearing_disp``, ``cell``, ``node_mask``, ``atomic_numbers``,
                and ``c6_ratios`` (or keys for computing them).

        Returns:
            Dict: ``{'dispersion_energy_kspace': Array}`` with per-atom k-space
                dispersion energies in eV, shape: (num_nodes).
        """
        from jaxpme.kspace import generate_kvectors, get_reciprocal

        positions = inputs['positions']
        k_grid_disp = inputs['k_grid_disp']
        k_smearing_disp = inputs['k_smearing_disp']
        cell = inputs['cell']
        pbc = jnp.array([True, True, True])
        node_mask = inputs['node_mask']
        atomic_numbers = inputs['atomic_numbers']

        # Get c6_ratios
        if self.legacy:
            hirshfeld_ratios = inputs.get('hirshfeld_ratios')
            if hirshfeld_ratios is None:
                hirshfeld_ratios = self.hirshfeld_ratios(inputs)['hirshfeld_ratios']
            c6_ratios = hirshfeld_ratios ** 2
        else:
            # Always the head's prediction: inputs['c6_ratios'] is a dataset label.
            c6_ratios = self.c6_ratios(inputs)['c6_ratios']

        # Dispersion charges use the same generation-specific free-atom C6
        # table as the real-space mixing rule and PME subtraction.
        C6_i = _effective_atomic_c6(
            atomic_numbers, c6_ratios, legacy=self.legacy,
            dtype=positions.dtype
        )
        q_disp = _dispersion_charges(C6_i, node_mask)

        # Convert positions and cell from Angstrom to Bohr (C6 is in a.u.)
        _Bohr = jnp.asarray(Bohr, dtype=positions.dtype)
        _Hartree = jnp.asarray(Hartree, dtype=positions.dtype)
        positions_bohr = positions / _Bohr
        cell_bohr = cell / _Bohr
        smearing_bohr = k_smearing_disp / _Bohr

        volume_bohr = jnp.abs(jnp.linalg.det(cell_bohr))
        reciprocal_cell_bohr = get_reciprocal(cell_bohr)
        kvectors = generate_kvectors(
            reciprocal_cell_bohr, k_grid_disp.shape,
            dtype=positions.dtype, for_ewald=self.do_ewald,
        )

        if self.do_ewald:
            potentials = self.solver.kspace(
                smearing_bohr, q_disp, kvectors,
                positions_bohr, volume_bohr, cell_bohr, pbc,
            )
        else:
            potentials = self.solver.kspace(
                smearing_bohr, q_disp, reciprocal_cell_bohr,
                k_grid_disp, kvectors,
                positions_bohr, volume_bohr, cell_bohr, pbc,
            )

        # Energy = -q_i * potential_i  (negative = attractive dispersion)
        # Convert from Hartree to eV
        energies = -q_disp * potentials * _Hartree

        energies = safe_scale(energies, node_mask)

        return dict(dispersion_energy_kspace=energies)

    def reset_output_convention(self, output_convention):
        pass


class DispersionEnergySparse(nn.Module):
    prop_keys: Dict
    cutoff_lr: float
    cutoff_lr_damping: float
    hirshfeld_ratios: Optional[Any]
    c6_ratios: Optional[Any] = None
    dispersion_energy_scale: float = 1.0
    do_dispersion_kspace: bool = False
    legacy: bool = False  # Legacy SO3LR: derive C6 from Hirshfeld ratios (no c6 head).

    neighborlist_format: str = 'sparse'  # or 'ordered_sparse'
    module_name = 'dispersion_energy'

    def _switch_off(self, energy_ij, d_ij_lr):
        """Smoothly switch pair energies to zero at cutoff_lr (no-op without cutoff_lr)."""
        if self.cutoff_lr is None:
            return energy_ij
        if self.cutoff_lr_damping is None:
            raise ValueError(
                f"cutoff_lr is set but cutoff_lr_damping is not. "
                f"Received {self.cutoff_lr=} and {self.cutoff_lr_damping=}."
            )
        w = safe_mask(
            d_ij_lr > 0,
            partial(switching_fn, x_on=self.cutoff_lr - self.cutoff_lr_damping, x_off=self.cutoff_lr),
            d_ij_lr,
            0.
        )
        return safe_scale(energy_ij, w, 0.)

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        """Compute per-atom dispersion energy using Hirshfeld-scaled QDO model.

        Calculates pairwise C6/C8/C10 dispersion interactions from Hirshfeld volume
        ratios and free-atom reference data, with optional damping and a smooth
        switching function to taper the interaction to zero at ``cutoff_lr``.

        When ``do_dispersion_kspace`` is True and ``k_smearing_disp`` is present in
        inputs, the C6 term is split: the real-space part becomes the short-range
        residual (Casimir-Polder C6 minus geometric-mean C6 long-range). The
        complete C6+C8+C10 real-space residual is smoothly switched off before
        the neighbor-list cutoff; reciprocal-space C6 remains untruncated.

        Args:
            inputs (Dict): Must contain ``node_mask``, ``idx_i_lr``, ``idx_j_lr``,
                ``d_ij_lr``, ``atomic_numbers``, ``hirshfeld_ratios`` (or keys for
                computing them), and optionally ``no_sigma``, ``k_smearing_disp``.

        Returns:
            Dict: ``{'dispersion_energy': Array}`` with per-atom dispersion energies,
                shape: (num_nodes).
        """
        node_mask = inputs['node_mask']  # (num_nodes)
        num_nodes = len(node_mask)
        idx_i_lr = inputs['idx_i_lr']
        idx_j_lr = inputs['idx_j_lr']
        d_ij_lr = inputs['d_ij_lr']
        no_sigma = inputs.get('no_sigma',None)
        k_smearing_disp = inputs.get('k_smearing_disp', None)

        # Determine input dtype
        input_dtype = d_ij_lr.dtype

        # Polarizability/volume ratios used for alpha scaling. The head emits
        # 'hirshfeld_ratios' for the legacy model and 'a0_ratios' for the new ones.
        ratio_key = 'hirshfeld_ratios' if self.legacy else 'a0_ratios'
        hirshfeld_ratios = inputs.get(ratio_key)
        if hirshfeld_ratios is None:
            hirshfeld_ratios = self.hirshfeld_ratios(inputs)[ratio_key]

        if self.legacy:
            # Legacy SO3LR has no dedicated C6 head: C6 is derived from Hirshfeld
            # ratios (C6  scales with h^2). Feeding c6_ratios = h^2 into the shared
            # mixing rule reproduces the original C6 = C6_free * h^2 exactly.
            c6_ratios = hirshfeld_ratios ** 2
        else:
            # Always the head's prediction: inputs['c6_ratios'] is a dataset label.
            c6_ratios = self.c6_ratios(inputs)['c6_ratios']

        # Get atomic numbers (needed to link to the free-atom reference values)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)

        # Calculate alpha_ij and C6_ij using mixing rules
        alpha_ij, C6_ij = mixing_rules(
            atomic_numbers,
            idx_i_lr,
            idx_j_lr,
            hirshfeld_ratios,
            c6_ratios,
            legacy=self.legacy
        )

        # Use cubic fit for gamma
        gamma_ij = gamma_cubic_fit(alpha_ij)

        # Check if dispersion PME is active
        use_disp_pme = self.do_dispersion_kspace and (k_smearing_disp is not None)

        if use_disp_pme:
            # --- PME-aware real-space splitting ---
            # C6 real-space residual: full damped Casimir-Polder C6 minus geometric-mean LR
            # The geometric-mean LR part is what PME adds back in k-space.
            _Bohr = jnp.asarray(Bohr, dtype=input_dtype)
            _Hartree = jnp.asarray(Hartree, dtype=input_dtype)
            R_bohr = d_ij_lr / _Bohr

            if self.neighborlist_format == 'sparse':
                c = jnp.asarray(0.5, dtype=input_dtype)
            else:
                c = jnp.asarray(1.0, dtype=input_dtype)

            C8_ij = 5 / gamma_ij * C6_ij
            C10_ij = 245 / 8 / gamma_ij ** 2 * C6_ij
            p = jnp.asarray(self.dispersion_energy_scale, dtype=input_dtype) * 2 * 2.54 * alpha_ij ** (1.0 / 7)
            p = jnp.asarray(p, dtype=input_dtype)

            # Full damped C6 (Casimir-Polder mixing rule)
            V_C6_full = -C6_ij / (jnp.power(R_bohr, 6) + jnp.power(p, 6))

            # Geometric-mean C6 for PME subtraction.  It must use the same
            # generation-specific table as both the Casimir-Polder term above
            # and the reciprocal-space dispersion charges.
            C6_i = _effective_atomic_c6(
                atomic_numbers[idx_i_lr], c6_ratios[idx_i_lr],
                legacy=self.legacy, dtype=input_dtype
            )
            C6_j = _effective_atomic_c6(
                atomic_numbers[idx_j_lr], c6_ratios[idx_j_lr],
                legacy=self.legacy, dtype=input_dtype
            )
            C6_geom_ij = jnp.sqrt(C6_i * C6_j)

            # lr_r(sigma, R) from the c6_raw_potential: the long-range part that PME handles
            smearing_bohr = k_smearing_disp / _Bohr
            two_sigma2 = 2.0 * smearing_bohr ** 2
            x = R_bohr ** 2 / two_sigma2
            exp_neg_x = jnp.exp(-x)
            R_bohr_safe = jnp.where(R_bohr > 0, R_bohr, 1.0)
            lr_r_val = (1.0 - exp_neg_x * (1.0 + x + 0.5 * x ** 2)) / jnp.power(R_bohr_safe, 6)

            # C6 residual = full damped C6 + geometric-mean * lr_r (subtract LR part that PME adds)
            V_C6_residual = V_C6_full + C6_geom_ij * lr_r_val

            # C8 + C10 are not represented by PME.
            V_C8C10 = -C8_ij / (jnp.power(R_bohr, 8) + jnp.power(p, 8)) - C10_ij / (jnp.power(R_bohr, 10) + jnp.power(p, 10))

            # The complete real-space residual disappears with the neighbor
            # pair, so switch C6 together with C8+C10.  switching_fn is flat at
            # both endpoints; multiplying the whole residual makes both its
            # value and radial derivative vanish at cutoff_lr.
            V_real_space_residual = self._switch_off(V_C6_residual + V_C8C10, d_ij_lr)
            dispersion_energy_ij = c * V_real_space_residual * _Hartree

        else:
            # --- Original code path (no PME) ---
            if no_sigma is not None:
                dispersion_energy_ij = vdw_QDO_disp_damp_nosigma(
                d_ij_lr / jnp.asarray(Bohr, dtype=input_dtype),
                gamma_ij,
                C6_ij,
                alpha_ij,
                jnp.asarray(self.dispersion_energy_scale, dtype=input_dtype),
                self.neighborlist_format
                )
            else:
                dispersion_energy_ij = vdw_QDO_disp_damp(
                d_ij_lr / jnp.asarray(Bohr, dtype=input_dtype),
                gamma_ij,
                C6_ij,
                alpha_ij,
                jnp.asarray(self.dispersion_energy_scale, dtype=input_dtype),
                self.neighborlist_format
                )

            dispersion_energy_ij = self._switch_off(dispersion_energy_ij, d_ij_lr)

        atomic_dispersion_energy = segment_sum(
            dispersion_energy_ij,
            segment_ids=idx_i_lr,
            num_segments=num_nodes
        )  # (num_nodes)

        atomic_dispersion_energy = safe_scale(atomic_dispersion_energy, node_mask)

        return dict(dispersion_energy=atomic_dispersion_energy)

    def reset_output_convention(self, output_convention):
        pass
