"""Utilities for checkpointing."""
import pickle
import orbax.checkpoint as ocp

from pathlib import Path
from typing import Optional

from .logging_utils import suppress_absl_warnings


def load_params_from_checkpoint(
        ckpt_dir: str
):
    """Load parameters from checkpoint.

    Args:
        ckpt_dir (str): Checkpoint directory.

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

    loaded_mngr = make_checkpoint_manager(ckpt_dir=ckpt_dir_cleaned)

    with suppress_absl_warnings():
        mngr_state = loaded_mngr.restore(
            loaded_mngr.latest_step(),
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
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)
        return params

    # Fall back to orbax checkpoint
    ckpt_dir = workdir_path / "checkpoints"
    params = load_params_from_checkpoint(ckpt_dir=ckpt_dir)

    return params


def make_checkpoint_manager(
        ckpt_dir,
        ckpt_mngr_options: Optional[ocp.CheckpointManagerOptions] = None
):
    # Make the path absolute. First expanduser() then resolve().
    ckpt_dir_cleaned = Path(ckpt_dir).expanduser().resolve()

    if ckpt_mngr_options is None:
        options = ocp.CheckpointManagerOptions(step_prefix='ckpt')
    else:
        options = ckpt_mngr_options

    ckpt_mngr = ocp.CheckpointManager(
        ckpt_dir_cleaned,
        item_names=('params',),
        item_handlers={'params': ocp.StandardCheckpointHandler()},
        options=options
    )
    return ckpt_mngr


def make_training_checkpoint_manager(
        ckpt_dir,
        ckpt_mngr_options: Optional[ocp.CheckpointManagerOptions] = None
):
    """Create the atomic two-item manager used by iterator-based training.

    ``params`` remains the deployable/inference item.  ``training_state`` is a
    versioned PyTree containing everything needed to reproduce the next update.
    CheckpointManager commits and prunes the two items as one checkpoint, so a
    retained params item can never point at a separately-pruned sidecar.
    """
    ckpt_dir_cleaned = Path(ckpt_dir).expanduser().resolve()

    if ckpt_mngr_options is None:
        options = ocp.CheckpointManagerOptions(step_prefix='ckpt')
    else:
        options = ckpt_mngr_options

    return ocp.CheckpointManager(
        ckpt_dir_cleaned,
        item_names=('params', 'training_state'),
        item_handlers={
            'params': ocp.StandardCheckpointHandler(),
            'training_state': ocp.StandardCheckpointHandler(),
        },
        options=options,
    )
