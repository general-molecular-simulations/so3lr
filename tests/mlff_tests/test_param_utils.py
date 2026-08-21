"""Parameter selection/counting must agree with what the optimizer freezes.

`count_trainable_params` used to do a flat `key in params` test, so it reported
0 trainable parameters for every hierarchical ('a/b') or prefix ('a/b*')
strategy and 100% trainable for the blacklist strategies -- i.e. the printed
counts contradicted the parameters actually being updated.
"""
import jax.numpy as jnp
import pytest

from so3lr.mlff.utils.param_utils import (
    MAX_NUM_THEORY_LEVELS,
    count_params,
    count_trainable_params,
    iter_param_leaves,
    path_matches_key,
    unmatched_param_keys,
)


@pytest.fixture
def params():
    """A miniature stand-in for a SO3LR checkpoint."""
    return {'params': {
        'layers_0': {'dense': {'kernel': jnp.ones((2, 3))}},          # 6
        'layers_1': {'dense': {'kernel': jnp.ones((2, 5))}},          # 10
        'observables_0': {
            'atomic_scales': jnp.ones((4,)),                          # 4
            'energy_offset': jnp.ones((4,)),                          # 4
            'energy_dense_regression': {'kernel': jnp.ones((3, 3))},  # 9
            'energy_dense_final': {'kernel': jnp.ones((3, 1))},       # 3
            'electrostatic_energy': {'partial_charges': {'kernel': jnp.ones((2, 2))}},  # 4
        },
        'observables_2': {'embed': jnp.ones((7,))},                   # 7
    }}


def test_path_matches_key_forms():
    path = ('params', 'observables_0', 'energy_dense_final', 'kernel')
    # Plain component matches any single path component.
    assert path_matches_key(path, 'observables_0')
    assert not path_matches_key(path, 'observables_2')
    # Hierarchical keys match a contiguous run of components.
    assert path_matches_key(path, 'observables_0/energy_dense_final')
    assert not path_matches_key(path, 'observables_2/energy_dense_final')
    assert not path_matches_key(path, 'observables_0/kernel')  # not contiguous
    # Trailing '*' is a prefix match.
    assert path_matches_key(path, 'observables_0/energy_dense*')
    assert path_matches_key(path, 'layers_*') is False


def test_iter_param_leaves_yields_full_paths(params):
    paths = {p for p, _ in iter_param_leaves(params)}
    assert ('params', 'layers_0', 'dense', 'kernel') in paths
    assert ('params', 'observables_2', 'embed') in paths
    assert len(paths) == 8


def test_count_trainable_params_no_selection_counts_everything(params):
    assert count_trainable_params(params) == count_params(params) == 47


def test_count_trainable_params_hierarchical_and_prefix_keys(params):
    """The `final_mlp` strategy's keys used to count as 0 trainable parameters."""
    final_mlp = [
        'observables_0/atomic_scales',
        'observables_0/energy_dense*',
        'observables_0/energy_offset',
    ]
    # atomic_scales(4) + energy_offset(4) + regression(9) + final(3)
    assert count_trainable_params(params, trainable_keys=final_mlp) == 20


def test_count_trainable_params_plain_key(params):
    assert count_trainable_params(params, trainable_keys=['layers_1']) == 10


def test_count_trainable_params_blacklist(params):
    """`frozen_keys` used to be ignored entirely, reporting everything trainable."""
    frozen = ['observables_0/electrostatic_energy', 'observables_2']
    assert count_trainable_params(params, frozen_keys=frozen) == 47 - 4 - 7


def test_count_trainable_params_unmatched_whitelist_is_zero(params):
    """A whitelist that matches nothing freezes the model -- it must not silently
    report the full parameter count."""
    assert count_trainable_params(params, trainable_keys=['observables_1']) == 0
    assert count_trainable_params(params, trainable_keys=['layers_2']) == 0


def test_unmatched_param_keys(params):
    assert unmatched_param_keys(params, ['observables_0', 'observables_2']) == []
    assert unmatched_param_keys(
        params, ['observables_1', 'observables_0', 'layers_9']
    ) == ['observables_1', 'layers_9']
    assert unmatched_param_keys(params, ['observables_0/energy_dense*']) == []


def test_max_num_theory_levels_is_shared():
    """The dataloaders' one-hot width and the tiled energy head must agree."""
    from so3lr.mlff.data import dataloader_sparse_ase
    from so3lr.mlff.utils import training_utils

    assert MAX_NUM_THEORY_LEVELS == 16
    assert (training_utils.expand_observables_to_theory_levels.__defaults__[0]
            == MAX_NUM_THEORY_LEVELS)
