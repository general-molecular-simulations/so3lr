"""Utilities for checkpointing."""
import orbax.checkpoint as ocp

from pathlib import Path
from typing import Optional


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

    mngr_state = loaded_mngr.restore(
        loaded_mngr.latest_step(),
    )
    params = mngr_state.get("params")

    if params is None:
        raise RuntimeError(
            f'Parameters loaded from {ckpt_dir_cleaned} are None.'
        )

    del loaded_mngr

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
    ckpt_dir = Path(workdir).expanduser().resolve() / "checkpoints"

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
