"""A model directory's hyperparameters, with the flags older configs leave out read from the checkpoint.

The original v1 config sets neither ``legacy_so3lr_bool`` nor ``use_final_bias_bool``, and
both default to False, so a v1 checkpoint would be v2 to every reader of its config. Every
reader goes through :func:`load_model_config`. An explicit value is kept, and the checkpoint
is read only when a flag is missing.

Reading the checkpoint starts no JAX backend (flash's ``reserve_gpu`` resolves the config
before it sets the GPU memory fraction): ``params.pkl`` holds numpy arrays, and an orbax
``checkpoints/`` directory is read from its JSON tree metadata.
"""
import copy
import functools
import json
import logging
import pathlib
import pickle

logger = logging.getLogger("SO3LR")


def load_model_config(workdir):
    """``hyperparameters.json`` of a model directory as a dict (a copy; cached per file version)."""
    path = pathlib.Path(workdir).expanduser().resolve() / 'hyperparameters.json'
    st = path.stat()
    # mtime alone misses a rewrite within the filesystem's timestamp resolution; a file
    # replaced by a rename has a new inode, and an edit usually changes the size
    return copy.deepcopy(_load(path, (st.st_mtime_ns, st.st_ino, st.st_size)))


@functools.lru_cache(maxsize=None)
def _load(path, version):
    cfg = json.loads(path.read_text())
    model = cfg['model']
    if 'legacy_so3lr_bool' in model and 'use_final_bias_bool' in model:
        return cfg
    keys = _parameter_key_paths(path.parent)
    inferred = {
        # the v1 Hirshfeld head stores a query embedding ('Embed_1'); the v2 scalar head has none
        'legacy_so3lr_bool': any(_contains(k, ('observables_2', 'Embed_1')) for k in keys),
        'use_final_bias_bool': any(_contains(k, ('observables_0', 'energy_dense_final', 'bias')) for k in keys),
    }
    for key, value in inferred.items():
        if key not in model:
            model[key] = value
            if value:
                logger.info('%s does not set %s; the checkpoint says True.', path.parent, key)
    return cfg


def _parameter_key_paths(workdir):
    """Key paths of the directory's parameter tree, without loading arrays where possible."""
    if (workdir / 'params.pkl').exists():
        with open(workdir / 'params.pkl', 'rb') as fp:
            return list(_flatten(pickle.load(fp)))
    steps = sorted(workdir.glob('checkpoints/ckpt_*/params/_METADATA'),
                   key=lambda p: int(p.parent.parent.name.rsplit('_', 1)[-1]))
    if steps:  # orbax's tree metadata (a CheckpointManager would start the JAX backend)
        tree = json.loads(steps[-1].read_text())['tree_metadata']
        return [tuple(k['key'] for k in v['key_metadata']) for v in tree.values()]
    from so3lr.mlff.utils.checkpoint_utils import load_params_from_workdir
    return list(_flatten(load_params_from_workdir(workdir)))


def _flatten(tree, prefix=()):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from _flatten(value, prefix + (key,))
        else:
            yield prefix + (key,)


def _contains(path, sub):
    return any(path[i:i + len(sub)] == sub for i in range(len(path) - len(sub) + 1))
