import flax.linen as nn
import jax.numpy as jnp
import json
import os

from pathlib import Path
from typing import (Any, Callable, Dict, Sequence)

from so3lr.mlff.nn.layer import get_layer
from so3lr.mlff.nn.embed import get_embedding_module
from so3lr.mlff.nn.observable import get_observable_module
from so3lr.mlff.io import read_json

Array = Any


class StackNetSparse(nn.Module):
    geometry_embeddings: Sequence[Callable]
    feature_embeddings: Sequence[Callable]
    layers: Sequence[Callable]
    observables: Sequence[Callable]
    prop_keys: Dict
    return_representations_bool: bool = False
    use_residual_scalars: bool = False  # Per-layer learnable residual scalars

    def setup(self):
        if len(self.feature_embeddings) == 0:
            msg = "At least one embedding module in `feature_embeddings` is required."
            raise ValueError(msg)
        if len(self.observables) == 0:
            msg = "At least one observable module in `observables` is required."
            raise ValueError(msg)

    @classmethod
    def create_from_ckpt_dir(cls, ckpt_dir: str):
        h_path = Path(ckpt_dir).absolute().resolve() / 'hyperparameters.json'
        stack_net = init_stack_net_sparse(read_json(h_path))
        return stack_net

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Energy function of the NN.

        Args:
            inputs (Dict):
            args (Tuple):
            kwargs (Dict):

        Returns: energy, shape: (1)

        """

        quantities = {}
        quantities.update(inputs)

        # Initialize the geometric quantities
        for geom_emb in self.geometry_embeddings:
            geom_quantities = geom_emb(quantities)
            quantities.update(geom_quantities)

        # Initialize the per atom embedding
        embeds = []
        for embed_fn in self.feature_embeddings:
            embeds += [embed_fn(quantities)]  # len: n_embeds, shape: (n,F)
        x = jnp.stack(embeds, axis=-1).sum(axis=-1) / jnp.sqrt(len(embeds))  # shape: (n,F)
        quantities.update({'x': x})

        # Per-layer residual scalars (inspired by nanochat/modded-nanogpt)
        # resid_lambda: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambda: blends initial embedding back in at each layer (init 0.1 = small contribution)
        if self.use_residual_scalars:
            n_layers = len(self.layers)
            resid_lambdas = self.param(
                'resid_lambdas',
                nn.initializers.ones,
                (n_layers,)
            )
            x0_lambdas = self.param(
                'x0_lambdas',
                lambda key, shape: jnp.full(shape, 0.1),
                (n_layers,)
            )
            x0 = x  # Save initial embedding for skip connection

        for (n, layer) in enumerate(self.layers):
            if self.use_residual_scalars:
                # Scale residual stream and blend in initial embedding
                x_current = quantities['x']
                quantities['x'] = resid_lambdas[n] * x_current + x0_lambdas[n] * x0
            updated_quantities = layer(**quantities)
            quantities.update(updated_quantities)

        if self.return_representations_bool:
            return {
                'atomic_representations': quantities['x']
            }

        observables = {}
        for o_fn in self.observables:
            o_dict = o_fn(quantities)
            observables.update(o_dict)

        # return jax.tree_util.tree_map(lambda y: y[..., None], observables)
        return observables

    def __dict_repr__(self):
        geometry_embeddings = [x.__dict_repr__() for x in self.geometry_embeddings]
        feature_embeddings = []
        layers = []
        observables = []
        for x in self.feature_embeddings:
            feature_embeddings += [x.__dict_repr__()]
        for (n, x) in enumerate(self.layers):
            layers += [x.__dict_repr__()]
        for x in self.observables:
            observables += [x.__dict_repr__()]

        return {'stack_net_sparse': {'geometry_embeddings': geometry_embeddings,
                                     'feature_embeddings': feature_embeddings,
                                     'layers': layers,
                                     'observables': observables,
                                     'prop_keys': self.prop_keys,
                                     'n_layers': len(layers),
                                     'use_residual_scalars': self.use_residual_scalars}}

    def to_json(self, ckpt_dir, name='hyperparameters.json'):
        j = self.__dict_repr__()
        with open(os.path.join(ckpt_dir, name), 'w', encoding='utf-8') as f:
            json.dump(j, f, ensure_ascii=False, indent=4)

    def reset_prop_keys(self, prop_keys, sub_modules=True) -> None:
        self.prop_keys.update(prop_keys)
        if sub_modules:
            all_modules = self.geometry_embeddings + self.feature_embeddings + self.observables
            for m in all_modules:
                m.reset_prop_keys(prop_keys=prop_keys)

    def reset_input_convention(self, input_convention):
        for g in self.geometry_embeddings:
            g.reset_input_convention(input_convention=input_convention)

    def reset_output_convention(self, output_convention):
        for o in self.observables:
            o.reset_output_convention(output_convention=output_convention)


def init_stack_net_sparse(h) -> StackNetSparse:
    _h = h['stack_net_sparse']
    geom_embs = [get_embedding_module(*tuple(x.items())[0]) for x in _h['geometry_embeddings']]
    feature_embs = [get_embedding_module(*tuple(x.items())[0]) for x in _h['feature_embeddings']]
    lays = [get_layer(*tuple(x.items())[0]) for x in _h['layers']]
    obs = [get_observable_module(*tuple(x.items())[0]) for x in _h['observables']]
    use_residual_scalars = _h.get('use_residual_scalars', False)
    return StackNetSparse(
        **{'geometry_embeddings': geom_embs,
           'feature_embeddings': feature_embs,
           'layers': lays,
           'observables': obs,
           'prop_keys': _h['prop_keys'],
           'use_residual_scalars': use_residual_scalars
           }
    )
