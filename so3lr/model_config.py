"""A model directory's hyperparameters, with the flags older configs leave out read from the checkpoint.

The original v1 config sets neither ``legacy_so3lr_bool`` nor ``use_final_bias_bool``, and
both default to False, so a v1 checkpoint would be v2 to every reader of its config. Every
reader goes through :func:`load_model_config`. An explicit value is kept, and the checkpoint
is read only when a flag is missing.
"""
import copy
import functools
import json
import logging
import pathlib
import pickle

logger = logging.getLogger("SO3LR")


def load_model_config(workdir):
    """``hyperparameters.json`` of a model directory as a dict (a copy; cached per directory)."""
    return copy.deepcopy(_load(pathlib.Path(workdir).expanduser().resolve()))


@functools.lru_cache(maxsize=None)
def _load(workdir):
    cfg = json.loads((workdir / 'hyperparameters.json').read_text())
    model = cfg['model']
    if 'legacy_so3lr_bool' in model and 'use_final_bias_bool' in model:
        return cfg
    if (workdir / 'params.pkl').exists():  # numpy arrays: no JAX or orbax import needed
        with open(workdir / 'params.pkl', 'rb') as fp:
            params = pickle.load(fp)
    else:
        from so3lr.mlff.utils.checkpoint_utils import load_params_from_workdir
        params = load_params_from_workdir(workdir)
    params = params.get('params', params)
    inferred = {
        # the v1 Hirshfeld head stores a query embedding ('Embed_1'); the v2 scalar head has none
        'legacy_so3lr_bool': 'Embed_1' in params.get('observables_2', {}),
        'use_final_bias_bool': 'bias' in params.get('observables_0', {}).get('energy_dense_final', {}),
    }
    for key, value in inferred.items():
        if key not in model:
            model[key] = value
            if value:
                logger.info('%s does not set %s; the checkpoint says True.', workdir, key)
    return cfg
