"""Warnings and helpers for legacy executable serialization formats."""

import pickle
from pathlib import Path
from typing import Any
import warnings


class UnsafePickleWarning(RuntimeWarning):
    """A pickle-compatible artifact is about to execute deserialization."""


def _is_bundled_model_path(path: Any) -> bool:
    """Return whether ``path`` is part of the installed, code-trusted package."""
    try:
        resolved = Path(path).expanduser().resolve()
        bundled_models = (Path(__file__).resolve().parent / "models").resolve()
        return resolved.is_relative_to(bundled_models)
    except (OSError, RuntimeError, ValueError):
        return False


def warn_before_pickle_load(path: Any, *, artifact: str) -> None:
    """Warn prominently immediately before invoking a pickle-based loader."""
    warnings.warn(
        "SECURITY WARNING: loading "
        f"{artifact} from {Path(path)!s}. Pickle-compatible files can execute "
        "arbitrary code during deserialization. Only load this file if it is "
        "from a trusted source and its checksum has been verified.",
        UnsafePickleWarning,
        stacklevel=2,
    )


def load_pickle_with_warning(path: Any, *, artifact: str = "params.pkl") -> Any:
    """Load a legacy pickle, warning for paths outside the installed package.

    Bundled weights share the package installation's trust boundary with the
    Python code itself. Arbitrary model/work directories do not.
    """
    if not _is_bundled_model_path(path):
        warn_before_pickle_load(path, artifact=artifact)
    with open(path, "rb") as stream:
        return pickle.load(stream)
