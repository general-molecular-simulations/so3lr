"""Lightweight validation shared by CLI, YAML, and Python MD entry points."""

from numbers import Integral
from typing import Any, Mapping


_POSITIVE_INTEGER_SETTINGS = (
    "md_cycles",
    "md_steps",
    "save_buffer",
    "kspace_npt_cycles",
)


def validate_positive_md_settings(settings: Mapping[str, Any]) -> None:
    """Require count/frequency settings to be positive integers when present.

    ``kspace_npt_cycles`` is optional, so ``None`` continues to mean that the
    caller did not request an override.  Boolean values are rejected even
    though ``bool`` subclasses ``int`` in Python; accepting ``true`` from YAML
    as a one-cycle frequency would hide a configuration typo.
    """
    for key in _POSITIVE_INTEGER_SETTINGS:
        value = settings.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(
                f"{key} must be a positive integer (received {value!r})."
            )
