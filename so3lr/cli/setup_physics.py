"""Import-light long-range proposals for guided MD setup.

These are starting settings, displayed for review, not an accuracy tuning
procedure. Keep the numerical backend's minimum-image bounds authoritative.
"""
import math

from so3lr.kspace_defaults import (
    DEFAULT_KSPACE_DISPERSION_SMEARING, DEFAULT_KSPACE_DISPERSION_SPACING,
    DEFAULT_KSPACE_SMEARING, DEFAULT_KSPACE_SPACING, MIN_DIRECT_LR_CUTOFF)


def long_range_settings(periodic, minimum_cell_height, model, cutoff=None, *, ensemble=None):
    """Propose a cutoff that fits the input cell, with PME for small cells.

    The water example's 12.4 Å cell cannot use the usual 12 Å real-space
    cutoff. Carry both long-range terms in reciprocal space rather than
    presenting a truncated short cutoff as equivalent physics.
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
    result = {'lr_cutoff': value}
    if small_cell or value < MIN_DIRECT_LR_CUTOFF:
        # Keep the narrower real-space split for short cutoffs, while retaining
        # the normal spacing/smearing ratio rather than inheriting a coarse mesh
        # chosen for a 4 Å Gaussian. This is a starting point, not a convergence guarantee.
        smearing = 2.0
        result.update(kspace_electrostatics='pme', kspace_smearing=smearing,
                      kspace_spacing=smearing * DEFAULT_KSPACE_SPACING / DEFAULT_KSPACE_SMEARING,
                      kspace_dispersion='pme',
                      kspace_dispersion_smearing=DEFAULT_KSPACE_DISPERSION_SMEARING,
                      kspace_dispersion_spacing=DEFAULT_KSPACE_DISPERSION_SPACING)
    return result
