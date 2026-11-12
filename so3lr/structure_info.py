"""Charge and spin recorded in structure metadata, read one way everywhere.

Used by `so3lr eval`, Flash evaluation, guided setup and the MD/optimization
input checks. Spin multiplicity (2S+1) is stored under ``multiplicity=`` in
most datasets and under ``spin=`` in OMol25/OPoly26 extxyz files; the
canonical key wins. No numerical backend is imported here.
"""
import math


def charge_and_multiplicity(mol):
    """The total charge and spin multiplicity in ``mol.info`` (or a dict), None where absent."""
    info = getattr(mol, 'info', mol)
    multiplicity = info.get('multiplicity')
    if multiplicity is None:
        multiplicity = info.get('spin')
    return info.get('charge'), multiplicity


def _integer(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def recorded_charge_and_unpaired(mol):
    """Integer total charge and unpaired electrons (multiplicity - 1) recorded in metadata."""
    charge, multiplicity = charge_and_multiplicity(mol)
    multiplicity = _integer(multiplicity)
    unpaired = multiplicity - 1 if multiplicity is not None and multiplicity >= 1 else None
    return _integer(charge), unpaired


def electron_count_problem(electrons, charge, unpaired):
    """Explain an impossible charge/multiplicity combination, or return ''."""
    if electrons <= 0:
        return ''
    count = electrons - charge
    if count < 0:
        return f'A total charge of {charge:+d} removes more than the {electrons} electrons of the neutral structure.'
    if unpaired > count:
        return f'{unpaired} unpaired electrons exceed the {count} electrons at charge {charge:+d}.'
    if (count - unpaired) % 2:
        parity = 'an odd' if count % 2 else 'an even'
        return (f'Charge {charge:+d} leaves {count} electrons ({parity} number), which cannot have '
                f'{unpaired} unpaired electrons. Check the total charge or the multiplicity.')
    return ''
