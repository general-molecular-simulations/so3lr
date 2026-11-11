"""Import-light long-range proposals for guided MD setup.

These are starting settings, displayed for review, not an accuracy tuning
procedure. Keep the numerical backend's minimum-image bounds authoritative.
"""
import math


def long_range_settings(periodic, minimum_cell_height, model, cutoff=None):
    """Propose a cutoff that fits the input cell, with PME for small cells.

    The water example's 12.4 Å cell cannot use the usual 12 Å real-space
    cutoff. Carry both long-range terms in reciprocal space rather than
    presenting a truncated short cutoff as equivalent physics.
    """
    from .so3lr_cli import DEFAULT_LR_CUTOFF
    if not periodic:
        value = 100.0 if cutoff is None else float(cutoff)
        if not math.isfinite(value) or value < 10:
            raise ValueError('Gas-phase MD requires a finite long-range cutoff of at least 10 Å.')
        return {'lr_cutoff': value}

    from so3lr.flash.check import model_cutoff
    height = float(minimum_cell_height or 0)
    if not math.isfinite(height) or height <= 0:
        raise ValueError('A periodic structure needs a finite, nonzero cell.')
    short_cutoff = model_cutoff(model)
    if height < 2 * short_cutoff:
        raise ValueError(f'The {height:g} Å cell height is too small for this model’s '
                         f'{short_cutoff:g} Å short-range cutoff. Choose a larger cell.')
    small_cell = height < 2 * DEFAULT_LR_CUTOFF
    if cutoff is None:
        # Leave room for NPT cell changes; do not put the cutoff exactly on the
        # initial half-cell boundary. The normal runtime still checks each cell.
        value = max(short_cutoff, math.floor(height * 0.45 * 10) / 10) if small_cell else DEFAULT_LR_CUTOFF
    else:
        value = float(cutoff)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Long-range cutoff must be finite and positive.')
    if value < short_cutoff:
        raise ValueError(f'Guided setup requires a long-range cutoff of at least '
                         f'the model’s {short_cutoff:g} Å short-range cutoff.')
    if value > height / 2:
        raise ValueError(f'Cutoff must be at most {height / 2:g} Å for the '
                         f'{height:g} Å minimum cell height; leave room for NPT contraction.')
    result = {'lr_cutoff': value}
    if small_cell or value < 10:
        result.update(kspace_electrostatics='pme', kspace_smearing=2.0,
                      kspace_dispersion='pme', kspace_dispersion_smearing=2.0)
    return result
