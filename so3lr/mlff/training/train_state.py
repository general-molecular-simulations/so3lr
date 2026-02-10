import jax
import jax.numpy as jnp
import optax

from optax import warmup_exponential_decay_schedule, exponential_decay, linear_schedule
from typing import Any, Callable, Dict, Tuple
from flax.linen import Module
from flax.core import FrozenDict
from flax import core
from flax import struct
from optax import GradientTransformation

from so3lr.mlff.training.lr_decay import get_lr_decay


class CustomTrainState(struct.PyTreeNode):

    """MLFF custom train state for the common case with a single Optax optimizer.

    Synopsis::

        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx)
        grad_fn = jax.grad(make_loss_fn(state.apply_fn))
        for batch in data:
            grads = grad_fn(state.params, batch)
            state = state.apply_gradients(grads=grads)

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Attributes:
        step: Counter starts at 0 and is incremented by every call to `.apply_gradients()`.
        apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
            convenience to have a shorter params list for the `train_step()` function in your training loop.
        params: The parameters to be updated by `tx` and used by `apply_fn`.
        tx: An Optax gradient transformation.
        opt_state: The state for `tx`.
        polyak_step_size: Step size of exponential moving average for parameter updates. If None, no moving average
        used.
    """
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)  # TODO: do we really need that
    params: core.FrozenDict[str, Any]
    valid_params: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    reduce_lr_on_plateau_fn: Callable = struct.field(pytree_node=False)
    lr_decay_fn: Callable = struct.field(pytree_node=False)
    polyak_step_size: float = None
    plateau_length: int = 0
    plateau_count: int = 0

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)

        _cd_plateau, _plateau_length, _plateau_count = self.reduce_lr_on_plateau_fn(plateau_count=self.plateau_count,
                                                                                    plateau_length=self.plateau_length)

        _cd_decay = self.lr_decay_fn(self.step)

        updates = jax.tree_util.tree_map(lambda x: jnp.asarray(x*_cd_plateau*_cd_decay).astype(jnp.asarray(x).dtype), updates)
        new_params = optax.apply_updates(self.params, updates)

        if self.polyak_step_size is not None:
            valid_params = optax.incremental_update(new_tensors=new_params,
                                                    old_tensors=self.params,
                                                    step_size=self.polyak_step_size)
        else:
            valid_params = new_params

        return self.replace(step=self.step + 1,
                            params=new_params,
                            valid_params=valid_params,
                            opt_state=new_opt_state,
                            plateau_length=_plateau_length,
                            plateau_count=self.plateau_count + _plateau_count,
                            **kwargs,
                            )

    def reset_params(self, params, valid_params):
        return self.replace(params=params,
                            valid_params=valid_params)

    def reset_opt_state(self, opt_state):
        return self.replace(opt_state=opt_state)

    def improved(self, x: bool):
        if x:
            _plateau_length = 0
        else:
            _plateau_length = self.plateau_length + 1

        return self.replace(plateau_length=_plateau_length)

    @classmethod
    def create(cls,
               *,
               apply_fn,
               params,
               tx,
               polyak_step_size=None,
               reduce_lr_on_plateau_fn: Callable,
               lr_decay_fn: Callable,
               **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            valid_params=params,
            tx=tx,
            opt_state=opt_state,
            polyak_step_size=polyak_step_size,
            reduce_lr_on_plateau_fn=reduce_lr_on_plateau_fn,
            lr_decay_fn=lr_decay_fn,
            **kwargs,
        )


def create_train_state(module: Module,
                       params: FrozenDict,
                       tx: GradientTransformation,
                       polyak_step_size: float = None,
                       plateau_lr_decay: Dict = None,
                       scheduled_lr_decay: Dict = None,
                       lr_warmup: Dict = None) -> Tuple[CustomTrainState, Dict]:
    """
    Creates an initial TrainState.

    Args:
        module (Module): A FLAX module.
        params (FrozenDict): A FrozenDict with the model parameters.
        tx (GradientTransformation): An optax GradientTransformation.
        polyak_step_size (float): Step size for exponential moving average.
        plateau_lr_decay (Dict): Parameters for learning rate decrease on plateau.
        scheduled_lr_decay (Dict): Parameters for scheduled learning rate decay.
        lr_warmup (Dict): Parameters for learning rate warmup.

    Returns: A FLAX TrainState.

    """

    def_lr_decay_fn = lambda *args, **kwargs: 1.
    def_plat_decay_fn = lambda *args, **kwargs: (1., 0, 0)

    if lr_warmup is None and scheduled_lr_decay is not None:
        lrd_fn = exponential_decay(init_value=1.,
                                   transition_steps=int(scheduled_lr_decay['exponential']['transition_steps']),
                                   decay_rate=float(scheduled_lr_decay['exponential']['decay_factor']))
    elif lr_warmup is not None and scheduled_lr_decay is not None:
        lrd_fn = warmup_exponential_decay_schedule(init_value=float(lr_warmup['init_value']),
                                                   peak_value=float(lr_warmup['peak_value']),
                                                   warmup_steps=int(lr_warmup['warmup_steps']),
                                                   transition_steps=int(scheduled_lr_decay['exponential']['transition_steps']),
                                                   decay_rate=float(scheduled_lr_decay['exponential']['decay_factor']))
    elif lr_warmup is not None and scheduled_lr_decay is None:
        lrd_fn = linear_schedule(init_value=float(lr_warmup['init_value']),
                                 end_value=float(lr_warmup['peak_value']),
                                 transition_steps=int(lr_warmup['warmup_steps']))
    else:
        lrd_fn = def_lr_decay_fn

    lrp_fn = get_lr_decay(name='on_plateau', h=plateau_lr_decay) if plateau_lr_decay is not None else def_plat_decay_fn

    return CustomTrainState.create(apply_fn=module.apply,
                                   params=params,
                                   tx=tx,
                                   polyak_step_size=polyak_step_size,
                                   reduce_lr_on_plateau_fn=lrp_fn,
                                   lr_decay_fn=lrd_fn), {'train_state': {'plateau_lr_decay': plateau_lr_decay,
                                                                         'scheduled_lr_decay': scheduled_lr_decay,
                                                                         'lr_warmup': lr_warmup}
                                                         }
