import optax

from optax import exponential_decay

from flax import traverse_util
from flax.core.frozen_dict import freeze, unfreeze

from dataclasses import dataclass


@dataclass
class Optimizer:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    transition_steps: int = None
    decay_rate: float = None
    weight_decay: float = None
    clip_by_global_norm: float = None

    def get(self,
            learning_rate: float,
            *args,
            **kwargs) -> optax.GradientTransformation:
        """
        Get the optax optimizer, for a specified learning rate.

        Args:
            learning_rate (float): The learning rate
            *args ():
            **kwargs ():

        Returns:

        """
        self.learning_rate = learning_rate
        weight_decay = 0. if self.weight_decay is None else self.weight_decay
        mask = None if self.weight_decay is None else flattened_traversal(lambda path, _: path[-1] != 'bias')

        if self.transition_steps is None or self.decay_rate is None:
            step_size_fn = None
        else:
            step_size_fn = exponential_decay(init_value=1.,
                                             transition_steps=self.transition_steps,
                                             decay_rate=self.decay_rate
                                             )

        return optimizer(learning_rate=self.learning_rate,
                                                   b1=self.b1,
                                                   b2=self.b2,
                                                   eps=self.eps,
                                                   eps_root=self.eps_root,
                                                   weight_decay=weight_decay,
                                                   mask=mask,
                                                   step_size_fn=step_size_fn,
                                                   clip_by_global_norm=self.clip_by_global_norm)

    def __dict_repr__(self):
        return {'optimizer': {'learning_rate': self.learning_rate,
                              'transition_steps': self.transition_steps,
                              'decay_rate': self.decay_rate,
                              'weight_decay': self.weight_decay,
                              'clip_by_global_norm': self.clip_by_global_norm}}


def optimizer(learning_rate,
              b1: float = 0.9,
              b2: float = 0.999,
              eps: float = 1e-8,
              eps_root: float = 0.0,
              weight_decay: float = 0.0,
              mask=None,
              step_size_fn=None,
              clip_by_global_norm=None):

    if clip_by_global_norm is None:
        clip_fn = optax.scale(1.)
    else:
        clip_fn = optax.clip_by_global_norm(max_norm=clip_by_global_norm)

    if step_size_fn is not None:
        msg = 'Scheduled LR decay has been moved to the train_state class. No LR decay is used.'
        raise DeprecationWarning(msg)

    return optax.chain(
        clip_fn,
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.add_decayed_weights(weight_decay, mask),
        optax.inject_hyperparams(optax.scale)(step_size=-learning_rate),
    )


def flattened_traversal(fn):
    def mask(data):
        flat = traverse_util.flatten_dict(unfreeze(data))
        return freeze(traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()}))
    return mask


def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(unfreeze(params))
    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
    return freeze(traverse_util.unflatten_dict(flat_mask))
