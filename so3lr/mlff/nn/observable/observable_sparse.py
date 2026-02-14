import flax.linen as nn
import jax
import jax.numpy as jnp

from ase.units import Bohr, Hartree
from ase.units import alpha as fine_structure
from functools import partial
from jax.ops import segment_sum
from jax.nn.initializers import constant
from typing import Any, Callable, Dict, Tuple, Optional, Sequence

from so3lr.mlff.nn.sub_module import BaseSubModule
from so3lr.mlff.masking.mask import safe_mask
from so3lr.mlff.nn.observable.dispersion_ref_data import alphas, C6_coef
from so3lr.mlff.masking.mask import safe_scale
from so3lr.mlff.nn.activation_function import softplus_inverse, softplus


class EnergySparse(BaseSubModule):
    prop_keys: Dict
    zmax: int = 118
    regression_dim: int = None
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

        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='energy_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
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
                inputs.update(**self.electrostatic_energy_kspace(inputs))
                atomic_energy += inputs['electrostatic_energy_kspace']

        if self.dispersion_energy_bool:
            inputs.update(**self.hirshfeld_ratios(inputs))
            inputs.update({'no_sigma':True})
            inputs.update({'dispersion_energy_undamped':self.dispersion_energy(inputs)['dispersion_energy']})
            del inputs['no_sigma']
            inputs.update(**self.dispersion_energy(inputs))
            atomic_energy += inputs['dispersion_energy']

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
            if imq_key in ['dipole_vec', 'energy', 'hirshfeld_ratios']:
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

        num_features = x.shape[-1]

        v_shift = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (num_nodes)
        q = nn.Embed(num_embeddings=100, features=int(num_features / 2))(atomic_numbers)  # shape: (n,F/2)

        if self.regression_dim is not None:
            y = nn.Dense(
                int(self.regression_dim / 2),
                kernel_init=nn.initializers.lecun_normal(),
                name='hirshfeld_ratios_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            k = nn.Dense(
                int(num_features / 2),
                kernel_init=self.kernel_init,
                name='hirshfeld_ratios_dense_final'
            )(y)  # (num_nodes)
        else:
            k = nn.Dense(
                int(num_features / 2),
                kernel_init=self.kernel_init,
                name='hirshfeld_ratios_dense_final'
            )(x)  # (num_nodes)

        qk = (q * k / jnp.sqrt(k.shape[-1])).sum(axis=-1)

        v_eff = v_shift + qk  # shape: (n)
        hirshfeld_ratios = safe_scale(jnp.abs(v_eff), node_mask)

        return dict(hirshfeld_ratios=hirshfeld_ratios)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention


class PartialChargesSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
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
        atom_counts = jnp.bincount(batch_segments, length=num_graphs)
        charge_conservation = jnp.reciprocal(atom_counts) * (total_charge - predicted_charge)
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
        positions = inputs['positions']  # (num_nodes, 3)

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

@jax.jit
def mixing_rules(
        atomic_numbers: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        hirshfeld_ratios: jnp.ndarray,
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

    Returns:
        Tuple of (alpha_ij, C6_ij): mean polarizabilities and combined C6
        coefficients for each pair.
    """
    dtype = hirshfeld_ratios.dtype

    atomic_number_i = atomic_numbers[idx_i] - 1
    atomic_number_j = atomic_numbers[idx_j] - 1
    hirshfeld_ratio_i = hirshfeld_ratios[idx_i]
    hirshfeld_ratio_j = hirshfeld_ratios[idx_j]

    alpha_i = jnp.asarray(jnp.take(alphas, atomic_number_i, axis=0), dtype=dtype) * hirshfeld_ratio_i
    C6_i = jnp.asarray(jnp.take(C6_coef, atomic_number_i, axis=0), dtype=dtype) * jnp.square(hirshfeld_ratio_i)
    alpha_j = jnp.asarray(jnp.take(alphas, atomic_number_j, axis=0), dtype=dtype) * hirshfeld_ratio_j
    C6_j = jnp.asarray(jnp.take(C6_coef, atomic_number_j, axis=0), dtype=dtype) * jnp.square(hirshfeld_ratio_j)

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
    _sigma = jnp.asarray(sigma, dtype=input_dtype)
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
    Ziegler-Biersack-Littmark repulsion.
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
                    cuton=4.5,
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
                    cuton=4.5,
                    smearing=k_smearing,
                    neighborlist_format=self.neighborlist_format
                    )

        # If no cutoff is set, we just apply damping with error function and no explicit smoothing to zero.
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
    do_ewald: bool = False
    interpolation_nodes: int = 4
    ke: float = 14.399645351950548
    electrostatic_energy_scale: float = 1.0
    module_name: str = "electrostatic_energy_kspace"

    def setup(self):
        from jaxpme.solvers import ewald, pme
        from jaxpme.potentials import potential as get_potential

        if self.do_ewald:
            self.solver = ewald(get_potential())
        else:
            self.solver = pme(get_potential(), interpolation_nodes=self.interpolation_nodes)
    
    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        """Compute per-atom k-space electrostatic energy via Ewald or PME.

        Evaluates the reciprocal-space contribution to the electrostatic energy
        using either standard Ewald summation or Particle Mesh Ewald (PME),
        depending on the ``do_ewald`` flag.

        Args:
            inputs (Dict): Must contain ``positions``, ``k_grid``, ``k_smearing``,
                ``cell``, ``node_mask``, and ``partial_charges`` (or keys for computing them).

        Returns:
            Dict: ``{'electrostatic_energy_kspace': Array}`` with per-atom k-space
                electrostatic energies, shape: (num_nodes).
        """
        from jaxpme.kspace import generate_kvectors, get_reciprocal

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
        reciprocal_cell = get_reciprocal(cell)
        kvectors = generate_kvectors(
            reciprocal_cell, k_grid.shape, dtype=positions.dtype, for_ewald=self.do_ewald
        )

        #if node_mask is not None:
        #    partial_charges *= node_mask

        if self.do_ewald:
            potentials = self.solver.kspace(k_smearing, partial_charges, kvectors, positions, volume)
        else:
            potentials = self.solver.kspace(k_smearing, partial_charges, reciprocal_cell, k_grid, kvectors, positions, volume)

        #if node_mask is not None:
        #    potentials *= node_mask

        energies = partial_charges * potentials
        energies *= self.ke

        # Mask padded nodes
        energies = safe_scale(energies, node_mask)

        return dict(electrostatic_energy_kspace=energies)

    def reset_output_convention(self, output_convention):
        pass

class DispersionEnergySparse(nn.Module):
    prop_keys: Dict
    cutoff_lr: float
    cutoff_lr_damping: float
    hirshfeld_ratios: Optional[Any]
    dispersion_energy_scale: float = 1.0

    neighborlist_format: str = 'sparse'  # or 'ordered_sparse'
    module_name = 'dispersion_energy'

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        """Compute per-atom dispersion energy using Hirshfeld-scaled QDO model.

        Calculates pairwise C6/C8/C10 dispersion interactions from Hirshfeld volume
        ratios and free-atom reference data, with optional damping and a smooth
        switching function to taper the interaction to zero at ``cutoff_lr``.

        Args:
            inputs (Dict): Must contain ``node_mask``, ``idx_i_lr``, ``idx_j_lr``,
                ``d_ij_lr``, ``atomic_numbers``, ``hirshfeld_ratios`` (or keys for
                computing them), and optionally ``no_sigma``.

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

        # Determine input dtype
        input_dtype = d_ij_lr.dtype

        # Calculate Hirshfeld ratios
        hirshfeld_ratios = inputs.get('hirshfeld_ratios')
        if hirshfeld_ratios is None:
            hirshfeld_ratios =  self.hirshfeld_ratios(inputs)['hirshfeld_ratios']

        # Get atomic numbers (needed to link to the free-atom reference values)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)

        # Calculate alpha_ij and C6_ij using mixing rules
        alpha_ij, C6_ij = mixing_rules(
            atomic_numbers,
            idx_i_lr,
            idx_j_lr,
            hirshfeld_ratios
        )

        # Use cubic fit for gamma
        gamma_ij = gamma_cubic_fit(alpha_ij)

        # Get dispersion energy, positions are converted to to a.u.
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

        # If long-range cutoff is given, one needs to damp dispersion smoothly to zero at cutoff_lr.
        if self.cutoff_lr is not None:
            if self.cutoff_lr_damping is None:
                raise ValueError(
                    f"cutoff_lr is but cutoff_lr_damping is not set. "
                    f"received {self.cutoff_lr=} and {self.cutoff_lr_damping=}."
                )

            w = safe_mask(
                d_ij_lr > 0,
                partial(switching_fn, x_on=self.cutoff_lr - self.cutoff_lr_damping, x_off=self.cutoff_lr),
                d_ij_lr,
                0.
            )

            dispersion_energy_ij = safe_scale(dispersion_energy_ij, w, 0.)

        atomic_dispersion_energy = segment_sum(
            dispersion_energy_ij,
            segment_ids=idx_i_lr,
            num_segments=num_nodes
        )  # (num_nodes)

        atomic_dispersion_energy = safe_scale(atomic_dispersion_energy, node_mask)

        return dict(dispersion_energy=atomic_dispersion_energy)

    def reset_output_convention(self, output_convention):
        pass
