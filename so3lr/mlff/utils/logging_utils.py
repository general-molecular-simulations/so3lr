"""Utilities for suppressing noisy log messages."""
import logging
from contextlib import contextmanager


@contextmanager
def suppress_absl_warnings():
    """Temporarily suppress absl Python logger warnings (e.g. from orbax)."""
    absl_logger = logging.getLogger('absl')
    original_level = absl_logger.level
    absl_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        absl_logger.setLevel(original_level)
