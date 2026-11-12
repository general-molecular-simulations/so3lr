"""Utilities for checkpointing."""
import pickle
import orbax.checkpoint as ocp

from pathlib import Path
from typing import Optional

from .logging_utils import suppress_absl_warnings


def _validation_loss(metrics):
    return metrics['loss']


# Training ranks its checkpoints by the saved validation loss, lowest first.
_RANK_BY_LOSS = dict(step_prefix='ckpt', best_fn=_validation_loss, best_mode='min')


def load_params_from_checkpoint(
        ckpt_dir: str,
        step: Optional[int] = None,
):
    """Load parameters from checkpoint.

    Args:
        ckpt_dir (str): Checkpoint directory.
        step (int, optional): Step to load. Defaults to the checkpoint with the
            lowest saved validation loss, or to the latest one when no
            checkpoint saved a loss.

    Returns:
        PyTree of parameters.

    Raises:
        ValueError: Workdir does not have a checkpoint directory.
        RuntimeError: Loaded parameters are None.

    """

    # Make the path absolute. First expanduser() then resolve().
    ckpt_dir_cleaned = Path(ckpt_dir).expanduser().resolve()

    if not ckpt_dir_cleaned.exists():
        raise ValueError(
            f'Trying to load parameters from {ckpt_dir_cleaned} but path does not exist.'
        )

    loaded_mngr = make_checkpoint_manager(
        ckpt_dir=ckpt_dir_cleaned,
        ckpt_mngr_options=ocp.CheckpointManagerOptions(**_RANK_BY_LOSS),
    )
    if step is None:
        # None when no checkpoint saved a loss; restore(None) then takes the latest.
        step = loaded_mngr.best_step()

    with suppress_absl_warnings():
        mngr_state = loaded_mngr.restore(
            step,
            # A training checkpoint may contain additional atomic items (for
            # example the optimizer/iterator state).  Asking explicitly for
            # ``params`` keeps inference compatible with both those composite
            # checkpoints and the historical params-only layout.
            args=ocp.args.Composite(params=ocp.args.StandardRestore()),
        )
    params = mngr_state.get("params")

    if params is None:
        raise RuntimeError(
            f'Parameters loaded from {ckpt_dir_cleaned} are None.'
        )

    loaded_mngr.close()

    return params


def load_params_pkl(path):
    """Load a ``params.pkl`` parameter tree."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_params_from_workdir(workdir):
    """Load parameters from workdir.

    Args:
        workdir (str): Path to `workdir`.

    Returns:
        PyTree of parameters.

    Raises:
        ValueError: Workdir does not have a checkpoint directory.
        RuntimeError: Loaded parameters are None.

    """
    workdir_path = Path(workdir).expanduser().resolve()

    # Try loading from params.pkl first (faster, more portable)
    pkl_path = workdir_path / "params.pkl"
    if pkl_path.exists():
        return load_params_pkl(pkl_path)

    # Fall back to orbax checkpoint
    ckpt_dir = workdir_path / "checkpoints"
    params = load_params_from_checkpoint(ckpt_dir=ckpt_dir)

    return params


def make_checkpoint_manager(
        ckpt_dir,
        ckpt_mngr_options: Optional[ocp.CheckpointManagerOptions] = None,
        item_names=('params',),
):
    # Make the path absolute. First expanduser() then resolve().
    ckpt_dir_cleaned = Path(ckpt_dir).expanduser().resolve()

    if ckpt_mngr_options is None:
        options = ocp.CheckpointManagerOptions(step_prefix='ckpt')
    else:
        options = ckpt_mngr_options

    return ocp.CheckpointManager(
        ckpt_dir_cleaned,
        item_names=item_names,
        item_handlers={
            name: ocp.StandardCheckpointHandler() for name in item_names
        },
        options=options,
    )


def make_training_checkpoint_options(ckpt_manager_options: Optional[dict] = None):
    """CheckpointManagerOptions for training: keep the ``max_to_keep`` (default 3)
    checkpoints with the lowest validation loss and always the newest one.

    orbax's own policy for a ``best_fn`` keeps only the best N, so it deletes the
    newest checkpoint as soon as its loss is not among them, and a restart (which
    resumes from ``latest_step()``) silently goes back to an older step.
    """
    options = {'max_to_keep': 3} if ckpt_manager_options is None else dict(ckpt_manager_options)
    # orbax refuses max_to_keep next to a preservation_policy.
    max_to_keep = options.pop('max_to_keep', None)
    policies = ocp.checkpoint_managers
    return ocp.CheckpointManagerOptions(
        **_RANK_BY_LOSS,
        preservation_policy=policies.AnyPreservationPolicy([
            # reverse=True sorts the lowest loss last, as orbax does for best_mode='min'.
            policies.BestN(get_metric_fn=_validation_loss, reverse=True, n=max_to_keep),
            policies.LatestN(n=1),
        ]),
        **options
    )


def make_training_checkpoint_manager(
        ckpt_dir,
        ckpt_mngr_options: Optional[ocp.CheckpointManagerOptions] = None
):
    """Create the atomic two-item manager used by iterator-based training.

    ``params`` remains the deployable/inference item.  ``training_state`` is a
    versioned PyTree with everything needed to continue training.
    CheckpointManager commits and prunes the two items as one checkpoint, so a
    retained params item can never point at a separately-pruned sidecar.
    """
    return make_checkpoint_manager(
        ckpt_dir, ckpt_mngr_options, item_names=('params', 'training_state')
    )
