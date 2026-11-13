"""Binding energies from evaluated dimer/separated frame pairs.

``python -m so3lr.prepare_dimer_xyz`` writes each dimer twice: as bound
(``structure_type=dimer``) and with one monomer translated far away
(``structure_type=dimer_translated``). The binding energy is the difference of
the two predicted energies; the reference is the frames' ``benchmark_Eint``,
which SAPT10k and NCIAtlas give in kcal/mol. No numerical backend is imported.
"""
import math

KCAL_PER_EV = 23.060547830619026  # kcal/mol per eV
BOUND = ('dimer', 'dimer_original')  # prepare_dimer_xyz keeps an input's own label


def dimer_references(infos):
    """Reference interaction energies per pair, or None unless every frame pairs up."""
    infos = list(infos)
    if len(infos) < 2 or len(infos) % 2:
        return None
    references = []
    for bound, separated in zip(infos[0::2], infos[1::2]):
        if bound.get('structure_type') not in BOUND or separated.get('structure_type') != 'dimer_translated':
            return None
        try:
            reference = float(bound['benchmark_Eint'])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(reference):
            return None
        references.append(reference)
    return references


def binding_energy_summary(energies_eV, references_kcal):
    """Binding energies (kcal/mol) from alternating bound/separated energies, with errors."""
    if len(energies_eV) != 2 * len(references_kcal):
        raise ValueError('Expected one bound and one separated energy per reference.')
    binding = [(energies_eV[2 * index] - energies_eV[2 * index + 1]) * KCAL_PER_EV
               for index in range(len(references_kcal))]
    errors = [predicted - reference for predicted, reference in zip(binding, references_kcal)]
    count = len(errors)
    return {
        'binding_kcal': binding,
        'n': count,
        'mae': sum(abs(error) for error in errors) / count,
        'rmse': math.sqrt(sum(error * error for error in errors) / count),
        'max_abs': max(abs(error) for error in errors),
    }
