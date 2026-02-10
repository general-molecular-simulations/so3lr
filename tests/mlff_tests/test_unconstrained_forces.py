"""
Tests for unconstrained force prediction (hybrid mode).

Tests that the model can predict forces directly (unconstrained) and switch to 
gradient-based forces (constrained) with the same parameters.
"""
import numpy.testing as npt
import pytest
import jax
import jax.numpy as jnp
import jraph

from so3lr.mlff.nn.stacknet import StackNetSparse
from so3lr.mlff.nn.embed import GeometryEmbedSparse, AtomTypeEmbedSparse
from so3lr.mlff.nn.layer import SO3kratesLayerSparse
from so3lr.mlff.nn.observable import EnergySparse
from so3lr.mlff.nn.stacknet import get_energy_and_force_fn_sparse, get_hybrid_energy_force_fn_sparse


def graph_to_input(graph: jraph.GraphsTuple, long_range: jraph.GraphsTuple):
    return dict(
        positions=graph.nodes.get('positions'),
        atomic_numbers=graph.nodes.get('atomic_numbers'),
        idx_i=graph.receivers,
        idx_j=graph.senders,
        cell=None,
        cell_offset=None,
        batch_segments=jnp.zeros((graph.n_node.item(),), dtype=jnp.int32),
        graph_mask=jnp.ones((1,)).astype(jnp.bool_),
        node_mask=jnp.ones((graph.n_node.item(),)).astype(jnp.bool_),
        theory_mask=jnp.ones((1, 1)).astype(jnp.bool_),
        idx_i_lr=long_range.receivers,
        idx_j_lr=long_range.senders,
    )


# Test graph
graph = jraph.GraphsTuple(
    nodes=dict(
        positions=jnp.array([
            [0., 1., 2.],
            [1., 1., 1.],
            [2., 2., 2.],
            [0., 0., 0.],
        ]),
        atomic_numbers=jnp.array([1, 2, 3, 4]),
        forces=jnp.ones((4, 3)) * 3,
    ),
    receivers=jnp.array([0, 0, 1, 1, 2, 3]),
    senders=jnp.array([1, 3, 0, 2, 1, 0]),
    globals=dict(energy=jnp.array([3])),
    edges=None,
    n_node=jnp.array([4]),
    n_edge=jnp.array([6]),
)

long_range = jraph.GraphsTuple(
    nodes=None, edges=None,
    senders=jnp.array([], dtype=jnp.int64),
    receivers=jnp.array([], dtype=jnp.int64),
    n_node=jnp.array([4]),
    n_edge=jnp.array([0]),
    globals=None,
)


num_features = 32
num_layers = 2


def create_model(predict_forces_directly: bool = False, force_regression_dim: int = None):
    """Create a StackNetSparse model with optional direct force prediction."""
    atom_type_embed = AtomTypeEmbedSparse(
        num_features=num_features,
        prop_keys=None
    )
    
    geometry_embed = GeometryEmbedSparse(
        degrees=[1, 2],
        radial_basis_fn='bernstein',
        num_radial_basis_fn=16,
        cutoff_fn='exponential',
        cutoff=2.5,
        input_convention='positions',
        prop_keys=None
    )
    
    layers = [SO3kratesLayerSparse(
        degrees=[1, 2],
        use_spherical_filter=i > 0,
        num_heads=2,
        num_features_head=8,
        qk_non_linearity=jax.nn.softplus,
        residual_mlp_1=True,
        residual_mlp_2=True,
        layer_normalization_1=False,
        layer_normalization_2=False,
        activation_fn=jax.nn.softplus,
        behave_like_identity_fn_at_init=False
    ) for i in range(num_layers)]
    
    energy = EnergySparse(
        prop_keys=None,
        output_is_zero_at_init=False,
        predict_forces_directly=predict_forces_directly,
        force_regression_dim=force_regression_dim
    )
    
    return StackNetSparse(
        geometry_embeddings=[geometry_embed],
        feature_embeddings=[atom_type_embed],
        layers=layers,
        observables=[energy],
        prop_keys=None
    )


def test_constrained_mode():
    """Test that constrained mode (gradient-based forces) still works."""
    model = create_model(predict_forces_directly=False)
    inputs = graph_to_input(graph, long_range)
    params = model.init(jax.random.PRNGKey(0), inputs)
    
    # Constrained mode
    force_fn = get_hybrid_energy_force_fn_sparse(model, force_mode='constrained')
    out = force_fn(params, **inputs)
    
    npt.assert_equal(out.get('energy').shape, (1,))
    npt.assert_equal(out.get('forces').shape, (4, 3))
    
    # Forces should not be trivially zero
    with npt.assert_raises(AssertionError):
        npt.assert_allclose(out.get('forces'), jnp.zeros((4, 3)))


def test_unconstrained_mode():
    """Test that unconstrained mode (direct force prediction) works."""
    model = create_model(predict_forces_directly=True, force_regression_dim=16)
    inputs = graph_to_input(graph, long_range)
    params = model.init(jax.random.PRNGKey(0), inputs)
    
    # Unconstrained mode
    force_fn = get_hybrid_energy_force_fn_sparse(model, force_mode='unconstrained')
    out = force_fn(params, **inputs)
    
    npt.assert_equal(out.get('energy').shape, (1,))
    npt.assert_equal(out.get('forces').shape, (4, 3))


def test_unconstrained_mode_requires_force_prediction():
    """Test that unconstrained mode fails if model doesn't predict forces."""
    model = create_model(predict_forces_directly=False)  # No force prediction
    inputs = graph_to_input(graph, long_range)
    params = model.init(jax.random.PRNGKey(0), inputs)
    
    force_fn = get_hybrid_energy_force_fn_sparse(model, force_mode='unconstrained')
    
    with pytest.raises(ValueError, match="does not output 'nn_forces'"):
        force_fn(params, **inputs)


def test_mode_switching_same_energy():
    """Test that switching modes gives the same energy (different forces by design)."""
    model_with_forces = create_model(predict_forces_directly=True, force_regression_dim=16)
    inputs = graph_to_input(graph, long_range)
    params = model_with_forces.init(jax.random.PRNGKey(0), inputs)
    
    # Get energy from unconstrained mode
    force_fn_unconstrained = get_hybrid_energy_force_fn_sparse(
        model_with_forces, force_mode='unconstrained'
    )
    out_unconstrained = force_fn_unconstrained(params, **inputs)
    
    # Get energy from constrained mode (same model, same params)
    force_fn_constrained = get_hybrid_energy_force_fn_sparse(
        model_with_forces, force_mode='constrained'
    )
    out_constrained = force_fn_constrained(params, **inputs)
    
    # Energy should be the same
    npt.assert_allclose(out_unconstrained['energy'], out_constrained['energy'], atol=1e-5)
    
    # Forces will be different (by design - one is direct, one is gradient)
    # This is expected behavior


def test_nn_forces_output():
    """Test that model outputs nn_forces when predict_forces_directly=True."""
    model = create_model(predict_forces_directly=True)
    inputs = graph_to_input(graph, long_range)
    params = model.init(jax.random.PRNGKey(0), inputs)
    
    # Direct model apply should include nn_forces
    outputs = model.apply(params, inputs)
    
    assert 'nn_forces' in outputs
    npt.assert_equal(outputs['nn_forces'].shape, (4, 3))


def test_dict_repr_includes_force_fields():
    """Test that __dict_repr__ includes the new force prediction fields."""
    model = create_model(predict_forces_directly=True, force_regression_dim=32)
    
    # Get the observable from model
    energy_obs = model.observables[0]
    repr_dict = energy_obs.__dict_repr__()
    
    assert 'predict_forces_directly' in repr_dict['energy_sparse']
    assert repr_dict['energy_sparse']['predict_forces_directly'] == True
    assert 'force_regression_dim' in repr_dict['energy_sparse']
    assert repr_dict['energy_sparse']['force_regression_dim'] == 32


def test_invalid_force_mode():
    """Test that invalid force_mode raises an error."""
    model = create_model()
    
    with pytest.raises(ValueError, match="force_mode must be"):
        get_hybrid_energy_force_fn_sparse(model, force_mode='invalid')
