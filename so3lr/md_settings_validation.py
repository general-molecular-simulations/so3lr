"""Import-light MD settings checks, shared by the CLI and ``so3lr_md``."""

from numbers import Integral


def validate_md_counts(settings) -> None:
    """Require positive integer MD counts (Click checks CLI flags; YAML/Python land here)."""
    for name in ('md_cycles', 'md_steps', 'save_buffer', 'kspace_npt_cycles'):
        value = settings.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
            raise ValueError(f"{name} must be a positive integer (received {value!r}).")
