"""Import-light settings helpers shared by the CLI and ``so3lr_md``: loading a
settings YAML, resolving the ensemble and the MD state's precision, checking the MD counts."""

import difflib
import logging
import re
from numbers import Integral

import yaml

ENSEMBLES = ('nve', 'nvt', 'nvt-langevin', 'npt')
PRECISIONS = ('float32', 'float64')
DEFAULT_PRESSURE = 1.0  # bar


def canonical_ensemble(name) -> str:
    """The ensemble name perform_md expects. Case and '_' vs '-' do not matter
    (nvt_langevin is nvt-langevin); anything else raises with a did-you-mean."""
    canonical = str(name).strip().lower().replace('_', '-')
    if canonical not in ENSEMBLES:
        hint = difflib.get_close_matches(canonical, ENSEMBLES, n=1)
        raise ValueError(
            f"Unknown ensemble {name!r}; expected one of {', '.join(ENSEMBLES)}"
            + (f" (did you mean {hint[0]!r}?)" if hint else "") + ".")
    return canonical


def canonical_precision(precision) -> str:
    """The model's precision (float32 if unset); anything but PRECISIONS raises."""
    p = 'float32' if precision is None else str(precision).strip().lower()
    if p not in PRECISIONS:
        raise ValueError(f"precision must be float32 or float64 (received {precision!r})")
    return p


def state_precision(precision, settings) -> str:
    """The MD state's precision (positions, momenta, box, energy sums) on both engines, around a
    model in ``precision``: ``md_precision`` (float64 if unset; float32 coordinates of a 500 A
    box round to 6e-5 A), or the older ``flash_mixed`` (false: float32). A float64 model always
    has a float64 state."""
    if precision == 'float64':
        return 'float64'
    if settings.get('md_precision') is not None:
        return canonical_precision(settings['md_precision'])
    return 'float64' if bool(settings.get('flash_mixed', True)) else 'float32'


def resolve_ensemble(settings: dict) -> dict:
    """A copy of the settings with a canonical 'ensemble' and a pressure only for NPT.

    Without 'ensemble' a pressure (md_P) selects NPT, else NVT. NPT without md_P
    gets DEFAULT_PRESSURE; another ensemble drops md_P (perform_md reads a
    pressure as NPT), with a warning when the ensemble was set explicitly.
    """
    settings = dict(settings)
    explicit = settings.get('ensemble')
    ensemble = canonical_ensemble(explicit if explicit is not None
                                  else 'npt' if settings.get('md_P') is not None else 'nvt')
    settings['ensemble'] = ensemble
    if ensemble == 'npt':
        if settings.get('md_P') is None:
            settings['md_P'] = DEFAULT_PRESSURE
    else:
        if settings.get('md_P') is not None:
            logging.getLogger("SO3LR").warning(
                f"md_P={settings['md_P']} is ignored: the ensemble is {ensemble}, not npt.")
        settings['md_P'] = None
    return settings


def validate_md_counts(settings) -> None:
    """Require positive integer MD counts (Click checks CLI flags; YAML/Python land here)."""
    for name in ('md_cycles', 'md_steps', 'save_buffer', 'kspace_npt_cycles'):
        value = settings.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
            raise ValueError(f"{name} must be a positive integer (received {value!r}).")


class _SettingsLoader(yaml.SafeLoader):
    """safe_load that reads 1e-3 as a float. PyYAML follows YAML 1.1, where a
    float needs a dot ('1.0e-3'), so '1e-3' arrived as the string '1e-3'."""


_SettingsLoader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile(r'^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$'),
    list('-+0123456789'))


def load_settings_yaml(path) -> dict:
    """A settings YAML as a dict (empty for an empty file)."""
    with open(path) as f:
        return yaml.load(f, Loader=_SettingsLoader) or {}
