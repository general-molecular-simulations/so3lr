"""Long-range settings shared by the CLI commands and the guided setup.

Periodic structures use PME for electrostatics and dispersion by default, with a
10 Å real-space cutoff and a 2 Å electrostatic smearing. A cell too small for
10 Å gets a cutoff that fits it and a finer mesh. These are starting settings,
displayed for review, not an accuracy tuning procedure; the numerical backend's minimum-image bounds stay
authoritative. Import-light: no JAX.
"""
import math
from pathlib import Path

from so3lr.kspace_defaults import (
    DEFAULT_KSPACE_DISPERSION_SMEARING, DEFAULT_KSPACE_DISPERSION_SPACING, MIN_DIRECT_LR_CUTOFF)

KSPACE_METHOD_KEYS = ('kspace_electrostatics', 'kspace_dispersion')
# The cutoff and mesh the PME defaults set when the user left them unset.
_CELL_FITTED_KEYS = ('lr_cutoff', 'kspace_smearing', 'kspace_spacing')
# What resolve_long_range chooses unless the user set it.
LONG_RANGE_KEYS = KSPACE_METHOD_KEYS + _CELL_FITTED_KEYS
# The PME default for periodic structures, as in the SO3LR Tauri liquid and scaling runs.
# Against a converged Ewald split (512 waters) it errs by 0.16 meV/Å in forces and 5 bar in
# pressure, a third of 12 Å with the library's 4 Å smearing (0.54 meV/Å, 22 bar). The
# real-space term fades out from 0.45 × the cutoff, so a 4 Å smearing at 10 Å would be
# 4.5 times worse; a 1 Å mesh changes nothing.
PME_LR_CUTOFF = 10.0     # Å
PME_SMEARING = 2.0       # Å
PME_SPACING = 1.55       # Å
GAS_PHASE_LR_CUTOFF = 100.0  # Å: the guided setup's proposal for isolated structures


def long_range_settings(periodic, minimum_cell_height, model, cutoff=None, *, ensemble=None):
    """Propose the long-range settings for a structure: PME for periodic cells.

    The water example's 12.4 Å cell cannot use the usual 10 Å real-space
    cutoff with the neighbour-list engine. Both long-range terms stay in
    reciprocal space, with a cutoff that fits the cell, rather than presenting
    a truncated short cutoff as equivalent physics.
    """
    if not periodic:
        value = GAS_PHASE_LR_CUTOFF if cutoff is None else float(cutoff)
        if not math.isfinite(value) or value < MIN_DIRECT_LR_CUTOFF:
            raise ValueError('Gas-phase MD requires a finite long-range cutoff of at least '
                             f'{MIN_DIRECT_LR_CUTOFF:g} Å.')
        return {'lr_cutoff': value, 'kspace_electrostatics': 'none', 'kspace_dispersion': 'none'}

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
    small_cell = minimum_image_problem(cell, PME_LR_CUTOFF) is not None
    if cutoff is None:
        # NPT must also reserve the backend's initial contraction margin when
        # deciding whether the usual cutoff fits. Existing small-cell proposals
        # already leave 10%; the runtime still checks later cell changes.
        value = max(short_cutoff, math.floor(height * 0.45 * 10) / 10) if small_cell else PME_LR_CUTOFF
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
    smearing, spacing = pme_mesh(value)
    return {'lr_cutoff': value,
            'kspace_electrostatics': 'pme', 'kspace_smearing': smearing, 'kspace_spacing': spacing,
            'kspace_dispersion': 'pme',
            'kspace_dispersion_smearing': DEFAULT_KSPACE_DISPERSION_SMEARING,
            'kspace_dispersion_spacing': DEFAULT_KSPACE_DISPERSION_SPACING}


def pme_mesh(cutoff):
    """Electrostatic smearing and mesh spacing (Å) for a real-space cutoff: 2 and 1.55 from
    10 Å up, and a 1 Å mesh for the shorter cutoffs of small cells. A starting point, not a
    convergence guarantee."""
    return (PME_SMEARING, PME_SPACING) if cutoff >= MIN_DIRECT_LR_CUTOFF else (PME_SMEARING, 1.0)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _g(value):
    return f'{value:g}' if _is_number(value) else str(value)


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


def _skip_lines(stream, count):
    """Move a binary stream past count lines without parsing them."""
    while count > 0:
        start, chunk = stream.tell(), stream.read(1 << 20)
        if not chunk:
            return
        found = chunk.count(b'\n')
        if found < count:
            count -= found
            continue
        end = -1
        for _ in range(count):
            end = chunk.index(b'\n', end + 1)
        stream.seek(start + end + 1)
        return


def _xyz_headers(path):
    """Each frame's comment line of an (ext)xyz file, read without parsing atoms."""
    with open(path, 'rb') as stream:
        while (count := stream.readline().strip()):
            yield stream.readline().decode()
            _skip_lines(stream, int(count))


def _cells(path, index):
    """[(cell, pbc)] of frame index (0 or -1), or of every frame (None); for (ext)xyz
    from the comment lines alone."""
    import numpy as np
    if Path(path).suffix.lower() in ('.xyz', '.extxyz') and index in (0, -1, None):
        try:
            from collections import deque
            from itertools import islice
            from ase.io.extxyz import key_val_str_to_dict
            headers = _xyz_headers(path)
            headers = (list(islice(headers, 1)) if index == 0 else list(deque(headers, maxlen=1)) if index == -1
                       else list(headers))
            cells = []
            for header in headers:
                info = key_val_str_to_dict(header)
                cells.append((np.reshape(info['Lattice'], (3, 3)), np.broadcast_to(info.get('pbc', True), (3,)))
                             if 'Lattice' in info else (np.zeros((3, 3)), np.zeros(3, bool)))
            if cells:
                return cells
        except (OSError, ValueError, TypeError, KeyError):
            pass  # an unusual header: ASE reads the frames below
    from ase.io import read
    frames = read(str(path), index=':' if index is None else index)
    return [(np.asarray(atoms.cell), atoms.pbc) for atoms in (frames if index is None else [frames])]


def structure_cell(path, index=-1):
    """(periodic, minimum cell height) of frame ``index`` of a structure file (by
    default the frame MD reads), or of every frame with index None (periodic only
    if all are); None if unreadable. An (ext)xyz file is read only up to its
    comment lines: a multi-million-atom input is not parsed twice. Mixed periodicity
    (a slab) raises ValueError."""
    from so3lr.pbc import is_fully_periodic, minimum_cell_height
    try:
        cells = _cells(path, index)
    except Exception:
        return None  # resolve_long_range says the cell could not be read; the run reports why
    # A slab or wire (mixed pbc) raises here with the reason, as the run itself would.
    heights = [minimum_cell_height(cell) if is_fully_periodic(pbc, context=str(path)) else 0.0
               for cell, pbc in cells]
    periodic = bool(heights) and min(heights) > 0
    return periodic, min(heights) if periodic else None


def resolve_long_range(settings, explicit, structure_path, *, fit_cell=True, index=-1):
    """Fill the long-range settings the user left unset, in place, for this structure.

    ``explicit`` holds the setting names the user gave. 'auto' (or unset)
    k-space means PME for a periodic structure and off otherwise. With both
    terms in k-space and no explicit cutoff, the cutoff is the PME default (10 Å)
    or, for a cell too small for it, the guided setup's cell-fitting one (not with
    fit_cell=False: evaluation lists every periodic image). With real-space
    dispersion the cutoff stays, so dispersion is not cut short. With k-space
    electrostatics an unset smearing and mesh come from pme_mesh. Frame
    ``index`` of the structure file (None: every frame) is read only when one of
    these choices needs it.
    Returns a short description of what was chosen, for the log ('' when nothing was).
    """
    methods = {key: kspace_method(settings.get(key), key in explicit) for key in KSPACE_METHOD_KEYS}
    pme_defaults = 'lr_cutoff' not in explicit and None not in methods.values()
    fit_cutoff = fit_cell and pme_defaults
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
    if needs_cell and cell is None and 'auto' in methods.values():
        notes.append(f'the cell of {structure_path} could not be read, so k-space stays off')
    if settings['kspace_electrostatics']:  # periodic, or k-space the user asked for
        cutoff = settings.get('lr_cutoff')
        if periodic and all(settings[key] for key in KSPACE_METHOD_KEYS) and pme_defaults:
            cutoff = PME_LR_CUTOFF
            if fit_cutoff:
                try:
                    cutoff = long_range_settings(True, cell[1], settings.get('model_path'),
                                                 ensemble=settings.get('ensemble'))['lr_cutoff']
                except (ValueError, OSError, KeyError):
                    cutoff = settings.get('lr_cutoff')  # the run itself reports a cell that no cutoff fits
        unset = [key for key in _CELL_FITTED_KEYS if key not in explicit]
        proposal = dict(zip(_CELL_FITTED_KEYS, (cutoff, *pme_mesh(cutoff)))) if _is_number(cutoff) else {}
        if proposal and 'kspace_smearing' in explicit and _is_number(settings.get('kspace_smearing')):
            proposal['kspace_spacing'] = settings['kspace_smearing'] * PME_SPACING / PME_SMEARING
        if proposal and any(settings.get(key) != proposal[key] for key in unset):
            settings.update((key, proposal[key]) for key in unset)
            fits = (f' to fit the {cell[1]:g} Å cell'
                    if 'lr_cutoff' in unset and cell and settings['lr_cutoff'] < PME_LR_CUTOFF else '')
            notes.append(f'{_g(settings["lr_cutoff"])} Å cutoff, {_g(settings.get("kspace_smearing"))} Å smearing '
                         f'and {_g(settings.get("kspace_spacing"))} Å mesh spacing{fits}')
    return '; '.join(notes)
