"""Subprocess driver for tests/test_eval_kspace.py.

Runs every k-space agreement case in one interpreter and prints one JSON dict.
It is a separate process because the test needs x64 on BEFORE the first so3lr
import (the eval path turns it on itself via setup_jax_globals, which is too late
once another test in the same process has already traced at float32), and
setting it at module scope in a test would leak into the rest of the group.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

# Small on purpose: an agreement test, not a production setting. The cell must
# still satisfy minimum image for the ASE calculator (heights > 2 * LR_CUTOFF).
LR_CUTOFF, SMEARING, SPACING = 5.0, 2.0, 1.0
A, REPS = 6.4, 2                       # 2x2x2 of a 6.4 A box: 12.8 A heights, 24 atoms


def box(a=A):
    base = Atoms('OH2', positions=[(0.0, 0.0, 0.0), (0.9572, 0.0, 0.0),
                                   (-0.2400, 0.9266, 0.0)],
                 cell=[a, a, a], pbc=True)
    cell = base.repeat((REPS, REPS, REPS))
    cell.positions += np.random.default_rng(0).normal(0, 0.01, cell.positions.shape)
    return cell


def ase_energy(atoms, kspace):
    from so3lr import So3lrCalculator
    kw = dict(lr_cutoff=LR_CUTOFF, calculate_stress=False, dtype=np.float64)
    if kspace:
        kw.update(kspace_electrostatics=kspace, kspace_smearing=SMEARING,
                  kspace_spacing=SPACING)
    a = atoms.copy()
    a.calc = So3lrCalculator(**kw)
    return float(a.get_potential_energy())


def eval_energies(atoms_list, kspace, tmp, tag):
    from so3lr.cli.so3lr_eval import evaluate_so3lr_on
    src = Path(tmp) / f'{tag}.extxyz'
    out = Path(tmp) / f'{tag}_pred.extxyz'
    write(str(src), atoms_list)
    evaluate_so3lr_on(datafile=str(src), batch_size=1, lr_cutoff=LR_CUTOFF,
                      jit_compile=True, save_to=str(out), precision='float64',
                      targets='energy', overwrite=True,
                      kspace_electrostatics=kspace, kspace_smearing=SMEARING,
                      kspace_spacing=SPACING)
    # eval writes its prediction to info['energy_so3lr']; info['energy'] is the
    # reference from the input, which these synthetic frames do not have.
    return [float(x.info['energy_so3lr']) for x in read(str(out), index=':')]


def main():
    tmp = tempfile.mkdtemp()
    b = box()
    b2 = box(a=A + 0.4)
    # One PME evaluation of the base box serves three checks below.
    pme_b = eval_energies([b], 'pme', tmp, 'pme')[0]
    out = {
        'n': len(b),
        'pme':   {'ase': ase_energy(b, 'pme'),   'eval': pme_b},
        'ewald': {'ase': ase_energy(b, 'ewald'), 'eval': eval_energies([b], 'ewald', tmp, 'ewald')[0]},
        'onoff': {'on': pme_b, 'off': eval_energies([b], None, tmp, 'off')[0]},
        'mixed': {'grouped': eval_energies([b, b2], 'pme', tmp, 'mixed'),
                  'singles': [pme_b, eval_energies([b2], 'pme', tmp, 'single1')[0]]},
    }
    mol = Atoms('OH2', positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])
    try:
        eval_energies([mol], 'pme', tmp, 'mol')
        out['nocell'] = None
    except ValueError as exc:
        out['nocell'] = str(exc)
    slab = b.copy()
    slab.pbc = [True, True, False]
    try:
        eval_energies([slab], 'pme', tmp, 'slab')
        out['mixed_pbc_eval'] = None
    except ValueError as exc:
        out['mixed_pbc_eval'] = str(exc)
    try:
        ase_energy(slab, 'pme')
        out['mixed_pbc_ase'] = None
    except ValueError as exc:
        out['mixed_pbc_ase'] = str(exc)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
