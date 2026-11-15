"""Long-range settings shared by the CLI commands and the guided setup.

Periodic structures use PME for electrostatics and dispersion by default. A cell
too small for the usual 12 Å real-space cutoff gets a cutoff that fits it and a
finer mesh. These are starting settings, displayed for review, not an accuracy
tuning procedure; the numerical backend's minimum-image bounds stay
authoritative. Import-light: no JAX.
"""
import math

from so3lr.kspace_defaults import (
    DEFAULT_KSPACE_DISPERSION_SMEARING, DEFAULT_KSPACE_DISPERSION_SPACING,
    DEFAULT_KSPACE_SMEARING, DEFAULT_KSPACE_SPACING, MIN_DIRECT_LR_CUTOFF)

KSPACE_METHOD_KEYS = ('kspace_electrostatics', 'kspace_dispersion')
# The cutoff and mesh a small-cell proposal may replace when the user left them unset.
_CELL_FITTED_KEYS = ('lr_cutoff', 'kspace_smearing', 'kspace_spacing')


def long_range_settings(periodic, minimum_cell_height, model, cutoff=None, *, ensemble=None):
    """Propose the long-range settings for a structure: PME for periodic cells.

    The water example's 12.4 Å cell cannot use the usual 12 Å real-space
    cutoff with the neighbour-list engine. Both long-range terms stay in
    reciprocal space, with a cutoff that fits the cell, rather than presenting
    a truncated short cutoff as equivalent physics.
    """
    from .so3lr_cli import DEFAULT_LR_CUTOFF
    if not periodic:
        value = 100.0 if cutoff is None else float(cutoff)
        if not math.isfinite(value) or value < MIN_DIRECT_LR_CUTOFF:
            raise ValueError('Gas-phase MD requires a finite long-range cutoff of at least '
                             f'{MIN_DIRECT_LR_CUTOFF:g} Å.')
        return {'lr_cutoff': value}

    from so3lr.flash.check import NPT_BOX_MARGIN, model_cutoff
    from so3lr.pbc import minimum_image_problem
    height = float(minimum_cell_height or 0)
    if not math.isfinite(height) or height <= 0:
        raise ValueError('A periodic structure needs a finite, nonzero cell.')
    short_cutoff = model_cutoff(model)
    margin = NPT_BOX_MARGIN if ensemble == 'npt' else 0.0
    available_height = height * (1 - margin)
    cell = [[available_height, 0, 0], [0, available_height, 0], [0, 0, available_height]]
    margin_detail = f' with a {margin:.0%} NPT contraction margin' if margin else ''
    if minimum_image_problem(cell, short_cutoff):
        raise ValueError(f'The {height:g} Å cell height is too small for this model’s '
                         f'{short_cutoff:g} Å short-range cutoff{margin_detail}. Choose a larger cell.')
    small_cell = minimum_image_problem(cell, DEFAULT_LR_CUTOFF) is not None
    if cutoff is None:
        # NPT must also reserve the backend's initial contraction margin when
        # deciding whether the usual cutoff fits. Existing small-cell proposals
        # already leave 10%; the runtime still checks later cell changes.
        value = max(short_cutoff, math.floor(height * 0.45 * 10) / 10) if small_cell else DEFAULT_LR_CUTOFF
    else:
        value = float(cutoff)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Long-range cutoff must be finite and positive.')
    if value < short_cutoff:
        raise ValueError(f'Guided setup requires a long-range cutoff of at least '
                         f'the model’s {short_cutoff:g} Å short-range cutoff.')
    if minimum_image_problem(cell, value):
        raise ValueError(f'Cutoff must be at most {available_height / 2:g} Å for the '
                         f'{height:g} Å minimum cell height{margin_detail}.')
    smearing, spacing = DEFAULT_KSPACE_SMEARING, DEFAULT_KSPACE_SPACING
    if small_cell or value < MIN_DIRECT_LR_CUTOFF:
        # Keep the narrower real-space split for short cutoffs, while retaining
        # the normal spacing/smearing ratio rather than inheriting a coarse mesh
        # chosen for a 4 Å Gaussian. This is a starting point, not a convergence guarantee.
        smearing = 2.0
        spacing = smearing * DEFAULT_KSPACE_SPACING / DEFAULT_KSPACE_SMEARING
    return {'lr_cutoff': value,
            'kspace_electrostatics': 'pme', 'kspace_smearing': smearing, 'kspace_spacing': spacing,
            'kspace_dispersion': 'pme',
            'kspace_dispersion_smearing': DEFAULT_KSPACE_DISPERSION_SMEARING,
            'kspace_dispersion_spacing': DEFAULT_KSPACE_DISPERSION_SPACING}


def kspace_method(value, explicit):
    """'auto', None (off) or a method name, from a CLI/YAML k-space value.

    Unset means 'auto'. 'none', 'off' or false turn the term off; so does an
    explicit YAML null.
    """
    if value is None:
        return None if explicit else 'auto'
    if value is False or str(value).strip().lower() in ('none', 'off', 'false'):
        return None
    return str(value).strip().lower()


def structure_cell(path, index=-1):
    """(periodic, minimum cell height) of frame ``index`` of a structure file (by
    default the frame MD reads); None if unreadable."""
    try:
        import numpy as np
        from ase.io import read
        from so3lr.pbc import cell_heights
        atoms = read(str(path), index=index)
        cell = np.asarray(atoms.cell)
        periodic = bool(np.all(atoms.pbc)) and np.linalg.matrix_rank(cell) == 3
        return periodic, float(np.min(cell_heights(cell))) if periodic else None
    except Exception:
        return None


def resolve_long_range(settings, explicit, structure_path, *, fit_cell=True, index=-1):
    """Fill the long-range settings the user left unset, in place, for this structure.

    ``explicit`` holds the setting names the user gave. 'auto' (or unset)
    k-space means PME for a periodic structure and off otherwise. With PME
    electrostatics and no explicit cutoff, a cell too small for the usual
    cutoff gets the guided setup's cell-fitting cutoff and mesh (not with
    fit_cell=False: evaluation lists every periodic image). Frame ``index`` of
    the structure file is read only when one of these choices needs it.
    Returns a short description of what was chosen, for the log ('' when nothing was).
    """
    methods = {key: kspace_method(settings.get(key), key in explicit) for key in KSPACE_METHOD_KEYS}
    fit_cutoff = fit_cell and 'lr_cutoff' not in explicit and methods['kspace_electrostatics'] is not None
    needs_cell = structure_path is not None and ('auto' in methods.values() or fit_cutoff)
    cell = structure_cell(structure_path, index) if needs_cell else None
    periodic = bool(cell and cell[0])
    chosen = []
    for key, method in methods.items():
        if method == 'auto':
            method = 'pme' if periodic else None
            if method:
                chosen.append(key.removeprefix('kspace_'))
        settings[key] = method
    notes = [f"PME {' and PME '.join(chosen)} (the default for periodic structures)"] if chosen else []
    if periodic and settings['kspace_electrostatics'] and fit_cutoff:
        try:
            proposal = long_range_settings(True, cell[1], settings.get('model_path'),
                                           ensemble=settings.get('ensemble'))
        except (ValueError, OSError, KeyError):
            proposal = None  # the run itself reports a cell that no cutoff fits
        if proposal and proposal['lr_cutoff'] != settings.get('lr_cutoff'):
            for key in _CELL_FITTED_KEYS:
                if key not in explicit:
                    settings[key] = proposal[key]
            notes.append(f'{settings["lr_cutoff"]:g} Å cutoff and {settings["kspace_spacing"]:g} Å mesh '
                         f'spacing to fit the {cell[1]:g} Å cell')
    return '; '.join(notes)
