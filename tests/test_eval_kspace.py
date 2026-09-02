"""`so3lr eval` with k-space must agree with the ASE calculator on a periodic cell.

The two paths build neighbours differently -- eval uses ASE's all-image
`neighbor_list('ijS', ...)`, the calculator uses glp's minimum-image
displacement -- so they only have to agree on a cell large enough for the
minimum-image convention to hold. Both then need the same reciprocal-space term,
which is what these tests cover.

Everything runs in a subprocess. x64 has to be enabled before the first so3lr
import, because so3lr builds its free-atom reference tables as module-level jnp
constants; flipping it afterwards (as the eval path does via setup_jax_globals)
leaves a cached float32 `_take` to collide with a freshly traced float64 one:

    'func.call' op operand type mismatch:
      expected 'tensor<86xf64>', but provided 'tensor<86xf32>'

Setting it at module scope here would leak into every other test sharing the
process, so it is set through JAX_ENABLE_X64 in the child environment instead --
the same isolation the CLI tests use.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from subprocess_env import so3lr_subprocess_env

# Small on purpose: this is an agreement test, not a production setting. The
# cell must still satisfy minimum image for the calculator, i.e. every height
# above 2 * LR_CUTOFF.
LR_CUTOFF, SMEARING, SPACING = 5.0, 2.0, 1.0
A, REPS = 6.4, 2                     # 2x2x2 of a 6.4 A box -> 12.8 A heights, 24 atoms

_DRIVER = textwrap.dedent('''
    import json, sys, tempfile
    from pathlib import Path
    import numpy as np
    from ase import Atoms
    from ase.io import read, write

    LR_CUTOFF, SMEARING, SPACING = {lr}, {sm}, {sp}

    def box(a={a}, reps={reps}, jitter=True):
        base = Atoms('OH2',
                     positions=[(0.0, 0.0, 0.0), (0.9572, 0.0, 0.0),
                                (-0.2400, 0.9266, 0.0)],
                     cell=[a, a, a], pbc=True)
        cell = base.repeat((reps, reps, reps))
        if jitter:
            cell.positions += np.random.default_rng(0).normal(
                0, 0.01, cell.positions.shape)
        return cell

    def ase_energy(atoms, kspace):
        from so3lr import So3lrCalculator
        kw = dict(lr_cutoff=LR_CUTOFF, calculate_stress=False, dtype=np.float64)
        if kspace:
            kw.update(kspace_electrostatics=kspace, kspace_smearing=SMEARING,
                      kspace_spacing=SPACING)
        a = atoms.copy(); a.calc = So3lrCalculator(**kw)
        return float(a.get_potential_energy())

    def eval_energies(atoms_list, kspace, tmp, tag):
        from so3lr.cli.so3lr_eval import evaluate_so3lr_on
        src = Path(tmp) / (tag + '.extxyz')
        write(str(src), atoms_list)
        out = Path(tmp) / (tag + '_pred.extxyz')
        evaluate_so3lr_on(
            datafile=str(src), batch_size=1, lr_cutoff=LR_CUTOFF,
            jit_compile=True, save_to=str(out), precision='float64',
            targets='energy', overwrite=True,
            kspace_electrostatics=kspace, kspace_smearing=SMEARING,
            kspace_spacing=SPACING)
        # eval writes predictions to info['energy_so3lr']; info['energy'] is the
        # reference from the input file, which these synthetic frames lack.
        return [float(x.info['energy_so3lr'])
                for x in read(str(out), index=':')]

    def main():
        case = sys.argv[1]
        tmp = tempfile.mkdtemp()
        if case in ('pme', 'ewald'):
            b = box()
            print(json.dumps({{'ase': ase_energy(b, case),
                              'eval': eval_energies([b], case, tmp, case)[0],
                              'n': len(b)}}))
        elif case == 'onoff':
            b = box()
            print(json.dumps({{'on': eval_energies([b], 'pme', tmp, 'on')[0],
                              'off': eval_energies([b], None, tmp, 'off')[0]}}))
        elif case == 'nocell':
            mol = Atoms('OH2', positions=[(0, 0, 0), (0.96, 0, 0),
                                          (-0.24, 0.93, 0)])
            try:
                eval_energies([mol], 'pme', tmp, 'mol')
            except ValueError as exc:
                print(json.dumps({{'raised': str(exc)}}))
                return
            print(json.dumps({{'raised': None}}))
        elif case == 'mixed':
            bs = [box(a={a}), box(a={a} + 0.4)]
            grouped = eval_energies(bs, 'pme', tmp, 'mixed')
            singles = [eval_energies([b], 'pme', tmp, 'single%d' % i)[0]
                       for i, b in enumerate(bs)]
            print(json.dumps({{'grouped': grouped, 'singles': singles,
                              'n': [len(b) for b in bs]}}))

    main()
''')


def _run(case):
    script = _DRIVER.format(lr=LR_CUTOFF, sm=SMEARING, sp=SPACING, a=A, reps=REPS)
    proc = subprocess.run(
        [sys.executable, '-c', script, case],
        capture_output=True, text=True,
        # x64 before the first so3lr import; CPU so CI is deterministic.
        env=so3lr_subprocess_env({'JAX_ENABLE_X64': '1', 'JAX_PLATFORMS': 'cpu'}))
    assert proc.returncode == 0, f'{case} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith('{')][-1]
    return json.loads(line)


@pytest.mark.parametrize('kspace', ['pme', 'ewald'])
def test_eval_kspace_matches_ase_calculator(kspace):
    """The whole point of the feature: eval == ASE calculator, periodic + k-space."""
    r = _run(kspace)
    per_atom = abs(r['eval'] - r['ase']) / r['n']
    assert per_atom < 1e-4, (f"{kspace}: eval {r['eval']:.6f} vs ASE {r['ase']:.6f} eV "
                             f'({per_atom:.2e} eV/atom)')


def test_kspace_changes_the_energy():
    """Guard against the flag silently doing nothing."""
    r = _run('onoff')
    assert abs(r['on'] - r['off']) > 1e-3


def test_kspace_requires_a_cell():
    """Non-periodic input with k-space must fail loudly, not silently skip it."""
    r = _run('nocell')
    assert r['raised'] is not None, 'expected a ValueError for non-periodic input'
    assert 'periodic' in r['raised']


def test_mixed_cells_are_grouped_not_silently_merged():
    """Different cells get their own mesh, and results stay in dataset order.

    Reusing one cell's mesh for every structure would be silently wrong, so the
    per-structure energies must match evaluating each structure on its own.
    """
    r = _run('mixed')
    for got, want, n in zip(r['grouped'], r['singles'], r['n']):
        assert abs(got - want) / n < 1e-4, f'grouped {got} vs single {want}'
