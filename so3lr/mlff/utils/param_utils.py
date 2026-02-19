import jax.numpy as jnp
import numpy as np


def print_param_shapes(params, prefix=''):
    if isinstance(params, dict):
        for key, value in params.items():
            if isinstance(value, (dict, jnp.ndarray, np.ndarray)):
                if isinstance(value, (jnp.ndarray, np.ndarray)):
                    print(f"{prefix}{key}: {value.shape}")
                else:
                    print(f"{prefix}{key}:")
                    print_param_shapes(value, prefix + '  ')


def count_params(params_dict):
    total = 0
    if isinstance(params_dict, dict):
        for key, value in params_dict.items():
            if isinstance(value, jnp.ndarray):
                total += value.size
            elif isinstance(value, dict):
                total += count_params(value)
            elif isinstance(value, np.ndarray):
                total += value.size
    return total


def count_trainable_params(params_dict, trainable_keys):
    """Count only parameters under specified trainable keys."""
    if trainable_keys is None:
        # All parameters are trainable
        return count_params(params_dict)

    total = 0
    params_subdict = params_dict.get('params', params_dict)

    for key in trainable_keys:
        if key in params_subdict:
            total += count_params(params_subdict[key])
    return total
