"""`so3lr eval` with k-space must agree with the ASE calculator on a periodic cell.

The two paths build neighbours differently -- eval uses ASE's all-image
`neighbor_list('ijS', ...)`, the calculator uses glp's minimum-image
displacement -- so they only have to agree on a cell large enough for the
minimum-image convention to be valid. Both then need the same reciprocal-space
term, which is what these tests cover.
"""
import jax
# so3lr builds its free-atom reference tables as module-level jnp constants at
# import time, so x64 has to be on BEFORE the first so3lr import. The eval path
# turns x64 on itself (setup_jax_globals), which is too late if something has
# already imported so3lr at float32: the cached f32 `_take` then collides with a
# freshly traced f64 one ("expected tensor<86xf64>, provided tensor<86xf32>").
jax.config.update('jax_enable_x64', True)

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

SMEARING, SPACING, LR_CUTOFF = 4.0, 2.0, 12.0


def _water_box(reps=(4, 4, 4), a=6.4):
    """A cheap periodic system, replicated so minimum image is valid."""
    rng = np.random.default_rng(0)
    base = Atoms('OH2',
                 positions=[(0.0, 0.0, 0.0), (0.9572, 0.0, 0.0),
                            (-0.2400, 0.9266, 0.0)],
                 cell=[a, a, a], pbc=True)
    cell = base.repeat(reps)
    cell.positions += rng.normal(0, 0.01, cell.positions.shape)
    return cell


def _ase_energy(atoms, kspace):
    from so3lr import So3lrCalculator
    kw = dict(lr_cutoff=LR_CUTOFF, calculate_stress=False, dtype=np.float64)
    if kspace:
        kw.update(kspace_electrostatics=kspace, kspace_smearing=SMEARING,
                  kspace_spacing=SPACING)
    a = atoms.copy()
    a.calc = So3lrCalculator(**kw)
    return float(a.get_potential_energy())


def _eval_energy(tmp_path, atoms, kspace, name='sys'):
    from so3lr.cli.so3lr_eval import evaluate_so3lr_on
    src = tmp_path / f'{name}.extxyz'
    write(str(src), atoms)
    out = evaluate_so3lr_on(
        datafile=str(src), batch_size=1, lr_cutoff=LR_CUTOFF,
        jit_compile=True, save_to=str(tmp_path / f'{name}_pred.extxyz'),
        precision='float64', targets='energy', overwrite=True,
        kspace_electrostatics=kspace, kspace_smearing=SMEARING,
        kspace_spacing=SPACING)
    from ase.io import read
    # eval writes its prediction to info['energy_so3lr']; info['energy'] holds the
    # reference from the input file, which a synthetic structure does not have.
    # The value is rounded to 6 significant figures on the way out, which is ~1e-3 eV
    # here -- well inside the per-atom tolerance below.
    return float(read(str(tmp_path / f'{name}_pred.extxyz')).info['energy_so3lr'])


@pytest.mark.parametrize('kspace', ['pme', 'ewald'])
def test_eval_kspace_matches_ase_calculator(tmp_path, kspace):
    """The whole point of the feature: eval == ASE calculator, periodic + k-space."""
    atoms = _water_box()
    e_ase = _ase_energy(atoms, kspace)
    e_eval = _eval_energy(tmp_path, atoms, kspace, name=f'box_{kspace}')
    # Per atom, so the tolerance does not scale with system size.
    assert abs(e_eval - e_ase) / len(atoms) < 1e-4, (
        f'{kspace}: eval {e_eval:.6f} vs ASE {e_ase:.6f} eV '
        f'({(e_eval - e_ase) / len(atoms):.2e} eV/atom)')


def test_kspace_changes_the_energy(tmp_path):
    """Guard against the flag silently doing nothing."""
    atoms = _water_box()
    assert abs(_eval_energy(tmp_path, atoms, 'pme', name='on')
               - _eval_energy(tmp_path, atoms, None, name='off')) > 1e-3


def test_kspace_requires_a_cell(tmp_path):
    """Non-periodic input with k-space must fail loudly, not silently skip it."""
    mol = Atoms('OH2', positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])
    with pytest.raises(ValueError, match='periodic'):
        _eval_energy(tmp_path, mol, 'pme', name='mol')


def test_mixed_cells_are_grouped_not_silently_merged(tmp_path):
    """Different cells get their own mesh, and results stay in dataset order.

    Reusing one cell's mesh for every structure would be silently wrong, so the
    per-structure energies must match evaluating each structure on its own.
    """
    from ase.io import read
    from so3lr.cli.so3lr_eval import evaluate_so3lr_on
    boxes = [_water_box(a=6.4), _water_box(a=6.8)]
    src = tmp_path / 'mixed.extxyz'
    write(str(src), boxes)
    evaluate_so3lr_on(
        datafile=str(src), batch_size=1, lr_cutoff=LR_CUTOFF, jit_compile=True,
        save_to=str(tmp_path / 'mixed_pred.extxyz'), precision='float64',
        targets='energy', overwrite=True, kspace_electrostatics='pme',
        kspace_smearing=SMEARING, kspace_spacing=SPACING)
    got = [float(a.info['energy_so3lr'])
           for a in read(str(tmp_path / 'mixed_pred.extxyz'), index=':')]
    want = [_eval_energy(tmp_path, b, 'pme', name=f'single{i}')
            for i, b in enumerate(boxes)]
    for g, w, b in zip(got, want, boxes):
        assert abs(g - w) / len(b) < 1e-4, f'grouped {g} vs single {w}'
