"""Import-light settings helpers shared by the CLI and ``so3lr_md``: loading a
settings YAML, resolving the ensemble and the run's precision, checking the MD counts."""

import difflib
import logging
import re
from numbers import Integral

import yaml

ENSEMBLES = ('nve', 'nvt', 'nvt-langevin', 'npt')
PRECISIONS = ('float32', 'float64', 'mixed')
DEFAULT_PRECISION = 'mixed'
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
    """A run's precision (mixed if unset): float32 or float64 throughout, or mixed, a float32
    model around a float64 MD state (float32 coordinates of a 500 A box round to 6e-5 A).
    Anything but PRECISIONS raises."""
    p = DEFAULT_PRECISION if precision is None else str(precision).strip().lower()
    if p not in PRECISIONS:
        raise ValueError(f"precision must be float32, float64 or mixed (received {precision!r})")
    return p


def model_precision(precision) -> str:
    """The model's precision in a run of ``precision``."""
    return 'float64' if canonical_precision(precision) == 'float64' else 'float32'


def state_precision(precision) -> str:
    """The MD state's precision (positions, momenta, box, energy sums; displacements are formed
    in it before the model) in a run of ``precision``, on both engines."""
    return 'float32' if canonical_precision(precision) == 'float32' else 'float64'


def resolve_precision(settings) -> str:
    """The run precision of ``settings``, with the retired forms: ``flash_mixed`` (deprecated)
    makes a float32 model's run mixed (true) or float32 (false), as it did before precision
    took mixed; md_precision raises."""
    if settings.get('md_precision') is not None:
        raise ValueError("md_precision was replaced by precision: mixed (a float64 MD state around "
                         "a float32 model, the default) or float32 (float32 throughout)")
    precision, flash_mixed = canonical_precision(settings.get('precision')), settings.get('flash_mixed')
    if flash_mixed is None:
        return precision
    if not isinstance(flash_mixed, bool):
        raise ValueError(f"flash_mixed must be true or false (received {flash_mixed!r})")
    if precision != 'float64':
        precision = 'mixed' if flash_mixed else 'float32'
    logging.getLogger("SO3LR").warning(
        f"flash_mixed is deprecated: this run is precision {precision}; set that instead.")
    return precision


def available_precision(settings) -> str:
    """resolve_precision for a library caller, which may run without x64 (the CLI turns it on):
    there an unset precision is float32, and a mixed or float64 one raises."""
    import jax
    from so3lr.mlff.calculators.potential import require_x64
    if (not jax.config.jax_enable_x64 and settings.get('precision') is None
            and settings.get('flash_mixed') is None):
        return 'float32'
    precision = resolve_precision(settings)
    require_x64(state_precision(precision))
    return precision


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
