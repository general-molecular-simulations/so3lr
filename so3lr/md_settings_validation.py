"""Import-light MD settings checks, shared by the CLI and ``so3lr_md``."""

import difflib
import re
from numbers import Integral

import yaml

ENSEMBLES = ('nve', 'nvt', 'nvt-langevin', 'npt')


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
