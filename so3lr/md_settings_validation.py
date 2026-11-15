"""Import-light settings helpers shared by the CLI and ``so3lr_md``: loading a
settings YAML, resolving the ensemble and the run's precision, checking the MD counts."""

import difflib
import logging
import math
import re
import stat
from numbers import Integral, Real
from pathlib import Path
from typing import Optional, Union

import yaml

ENSEMBLES = ('nve', 'nvt', 'nvt-langevin', 'npt')
PRECISIONS = ('float32', 'float64', 'mixed')
DEFAULT_PRECISION = 'mixed'
DEFAULT_PRESSURE = 1.0  # bar


def expand_indices(spec):
    """Expand a YAML index specification into a flat list of ints.

    Supported forms:
      - int:                       5             -> [5]
      - list / tuple (any mix):    [0, 1, 2]     -> [0, 1, 2]
                                   [0, "3-5"]    -> [0, 3, 4, 5]
      - inclusive range string:    "0-19"        -> [0, 1, ..., 19]
      - multi-token string:        "0-2, 5, 8-9" -> [0, 1, 2, 5, 8, 9]
        (tokens separated by commas and/or whitespace; each token is a single
        index or an inclusive "a-b" range with a <= b)
      - a NumPy array or integer of the above.

    Ranges "a-b" include BOTH endpoints. Negative indices are not supported.
    """
    def expand_token(tok):
        if isinstance(tok, bool):
            raise ValueError(f"Invalid index: {tok!r}")
        if isinstance(tok, Integral):
            return [int(tok)]
        if isinstance(tok, str):
            out = []
            for piece in tok.replace(',', ' ').split():
                if '-' in piece:
                    a, b = piece.split('-')
                    a, b = int(a), int(b)
                    if b < a:
                        raise ValueError(
                            f"Invalid index range '{piece}': end < start."
                        )
                    out.extend(range(a, b + 1))
                else:
                    out.append(int(piece))
            return out
        raise ValueError(f"Unsupported index token: {tok!r}")

    if hasattr(spec, 'tolist') and not isinstance(spec, str):   # NumPy array or scalar
        spec = spec.tolist()
    if isinstance(spec, (list, tuple)):
        result = []
        for item in spec:
            result.extend(expand_token(item))
        return result
    return expand_token(spec)


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


def normalize_npt_coupling(npt_coupling: str) -> str:
    """Return 'isotropic' or 'semi-isotropic' (spaces/underscores/hyphens accepted).

    Semi-isotropic coupling is experimental. Fully anisotropic coupling is
    rejected: the integrator would evolve shear, but the neighbour-list,
    reciprocal-space and dispersion-tail paths assume an orthorhombic cell.
    """
    coupling = '-'.join(str(npt_coupling).strip().lower().replace('_', ' ').replace('-', ' ').split())
    if coupling == 'anisotropic':
        raise ValueError(
            "npt_coupling='anisotropic' is experimental and not supported by SO3LR MD: "
            "the cell would shear, but the neighbour-list, reciprocal-space and "
            "dispersion-tail paths assume an orthorhombic cell. Use 'isotropic' or "
            "the experimental 'semi-isotropic' (coupled x/y, independent z)."
        )
    if coupling not in ('isotropic', 'semi-isotropic'):
        raise ValueError(
            f"Unknown npt_coupling={npt_coupling!r}. Supported values: isotropic, semi-isotropic."
        )
    return coupling


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
    precision = resolve_precision(settings)  # validates also what an unset precision ignores
    if (not jax.config.jax_enable_x64 and settings.get('precision') is None
            and settings.get('flash_mixed') is None):
        return 'float32'
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
    """Require positive integer MD counts and a positive timestep (Click checks CLI
    flags; YAML/Python land here)."""
    for name in ('md_cycles', 'md_steps', 'save_buffer', 'kspace_npt_cycles'):
        value = settings.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
            raise ValueError(f"{name} must be a positive integer (received {value!r}).")
    dt = settings.get('md_dt')
    if dt is not None and (isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0):
        raise ValueError(f"md_dt must be a positive, finite timestep (received {dt!r}).")


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


def _existing_regular_output_mode(path: Path) -> Optional[int]:
    """Return an existing output file's mode (symlinks are followed)."""
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f'Output must be a regular file: {path}')
    return stat.S_IMODE(metadata.st_mode)


def _validate_md_output_write_policy(
    *,
    output_file: Optional[Union[str, Path]],
    output_format: Optional[str],
    overwrite: bool,
    restart_load_path: Optional[Union[str, Path]],
) -> bool:
    """Reject an existing trajectory that the selected writer would truncate.

    EXTXYZ is intentionally appendable when continuing from a restart. HDF5 is
    opened in ``mode='w'`` and therefore needs a fresh output path unless the
    caller explicitly opts in to replacement with ``--overwrite``. Return
    whether the first EXTXYZ batch should append to an existing restart output.
    """
    append = (
        restart_load_path is not None and output_format == 'extxyz'
        and not overwrite
    )
    if output_file is None:
        return append

    output_path = Path(output_file)
    output_mode = _existing_regular_output_mode(output_path)
    if output_mode is None or overwrite:
        return append
    if append:
        return True

    if restart_load_path is not None and output_format == 'hdf5':
        raise RuntimeError(
            f"HDF5 output file already exists: {output_file}. Refusing to "
            "truncate it during restart. Choose a new --output path for the "
            "continued trajectory, or pass --overwrite to replace it."
        )

    raise RuntimeError(
        f"Output file already exists: {output_file}. Refusing to overwrite. "
        "Pass --overwrite to replace it, or choose a different --output."
    )


def output_refusal(settings, task='md'):
    """Why a run would refuse to write its outputs (one exists, no overwrite), or ''.

    The CLI asks before it opens the run's journal, so a refused start leaves no
    record behind; the writers keep their own checks."""
    output_file = settings.get('output_file')
    overwrite = bool(settings.get('overwrite', False))
    try:
        if task == 'opt':
            if output_file and _existing_regular_output_mode(Path(output_file)) is not None and not overwrite:
                raise RuntimeError(f"Output file already exists: {output_file}. Refusing to overwrite. "
                                   f"Pass --overwrite to overwrite, or choose a different --output.")
            return ''
        restart = settings.get('restart_load_path')
        _validate_md_output_write_policy(output_file=output_file, output_format=settings.get('output_format'),
                                         overwrite=overwrite, restart_load_path=restart)
        if output_file and settings.get('relax_before_run', True) and restart is None:
            relaxation = Path(output_file).with_name(f'{Path(output_file).stem}_opt.xyz')
            if _existing_regular_output_mode(relaxation) is not None and not overwrite:
                raise RuntimeError(f"Output file already exists: {relaxation}. Refusing to overwrite. "
                                   f"Pass --overwrite to replace it, or choose a different --output.")
    except RuntimeError as exc:
        return str(exc)
    return ''
