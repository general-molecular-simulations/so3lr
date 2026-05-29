"""Central registry resolving SO3LR model names to bundled checkpoint directories.

A "model" can be referred to either by a short name (``so3lr-s``, ``so3lr-m``,
``so3lr-l``) — which maps to a checkpoint bundled inside the package — or by an
arbitrary filesystem path to a custom/finetuned model directory.

This is the single source of truth used by the Python API (``make_so3lr``,
``make_ase_calculator``, ``make_potential_fn``), the ASE calculator, and the CLI.
"""

import pathlib

# .../so3lr/
_PKG = pathlib.Path(__file__).parent.resolve()
_MODELS = _PKG / "models"

# Model used when nothing is specified (bare calculator / CLI without --model /
# YAML without a model key).
DEFAULT_MODEL = "so3lr-m"

# Theory-level head index used at inference. All bundled models are trained to
# emit their production prediction on this head (one-hot ``theory_mask``).
DEFAULT_HEAD_INDEX = 5

_REGISTRY = {
    "so3lr-s": _MODELS / "so3lr-s",
    "so3lr-m": _MODELS / "so3lr-m",
    "so3lr-l": _MODELS / "so3lr-l",
    # Legacy vanilla SO3LR, kept reachable by name for reproducibility.
    "so3lr": _PKG / "params",
    "so3lr-legacy": _PKG / "params",
}


def available_models():
    """Return the tuple of known bundled model names."""
    return tuple(_REGISTRY)


def resolve_model(name_or_path=None):
    """Resolve a model name or filesystem path to load parameters.

    Args:
        name_or_path: A bundled model name (see :func:`available_models`), a path
            to a custom model directory, or ``None`` (uses :data:`DEFAULT_MODEL`).

    Returns:
        Tuple ``(workdir: pathlib.Path, from_file: bool, head_index: int)`` where
        ``from_file`` is ``True`` only for legacy directories that ship a
        ``params.pkl`` and no orbax ``checkpoints/`` folder.

    Raises:
        FileNotFoundError: If ``name_or_path`` is neither a known name nor an
            existing path.
    """
    key = str(name_or_path) if name_or_path is not None else DEFAULT_MODEL

    if key in _REGISTRY:
        workdir = _REGISTRY[key]
    else:
        workdir = pathlib.Path(key).expanduser().resolve()
        if not workdir.exists():
            raise FileNotFoundError(
                f"Model '{key}' is not a known name {available_models()} "
                f"and not an existing path."
            )

    # Prefer orbax checkpoints; fall back to a legacy params.pkl only when there
    # is no checkpoints directory.
    from_file = (
        not (workdir / "checkpoints").exists()
    ) and (workdir / "params.pkl").exists()

    return workdir, from_file, DEFAULT_HEAD_INDEX
