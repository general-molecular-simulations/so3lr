"""Progress lines from training and data loading that suit both the CLI and scripts."""
import logging

_LOGGER = logging.getLogger('SO3LR')


def say(message, level=logging.INFO):
    """Log under the CLI, whose live display owns the terminal; print for a script that set
    up no SO3LR logging, so its stdout keeps the lines it always had. Under the CLI, a
    DEBUG line (such as the parameter tree) stays out of the log file."""
    if _LOGGER.isEnabledFor(logging.INFO) and _LOGGER.hasHandlers():
        _LOGGER.log(level, message)
    else:
        print(message)
