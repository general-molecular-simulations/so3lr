"""`so3lr eval` with k-space must agree with the ASE calculator on a periodic cell.

The two paths build neighbours differently -- eval uses ASE's all-image
`neighbor_list('ijS', ...)`, the calculator uses glp's minimum-image
displacement -- so they only have to agree on a cell large enough for minimum
image to hold. Both then need the same reciprocal-space term; that is what these
tests check. All cases run in ONE subprocess (tests/_eval_kspace_driver.py) with
JAX_ENABLE_X64 set in the child environment; see that module for why.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from subprocess_env import so3lr_subprocess_env

DRIVER = Path(__file__).with_name('_eval_kspace_driver.py')
TOL_EV_PER_ATOM = 1e-4


@pytest.fixture(scope='module')
def results():
    proc = subprocess.run(
        [sys.executable, str(DRIVER)], capture_output=True, text=True,
        env=so3lr_subprocess_env({'JAX_ENABLE_X64': '1', 'JAX_PLATFORMS': 'cpu'}))
    assert proc.returncode == 0, f'driver failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
    return json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith('{')][-1])


@pytest.mark.parametrize('kspace', ['pme', 'ewald'])
def test_eval_kspace_matches_ase_calculator(results, kspace):
    """The point of the feature: eval == ASE calculator, periodic + k-space."""
    r = results[kspace]
    per_atom = abs(r['eval'] - r['ase']) / results['n']
    assert per_atom < TOL_EV_PER_ATOM, (
        f"{kspace}: eval {r['eval']:.6f} vs ASE {r['ase']:.6f} eV ({per_atom:.2e} eV/atom)")


def test_kspace_changes_the_energy(results):
    """Guard against the flag silently doing nothing."""
    assert abs(results['onoff']['on'] - results['onoff']['off']) > 1e-3


def test_kspace_requires_a_cell(results):
    """Non-periodic input with k-space must fail loudly, not silently skip it."""
    assert results['nocell'] is not None and 'periodic' in results['nocell']


def test_mixed_cells_match_single_evaluations(results):
    """Different cells (here also different mesh shapes) evaluate like singles, in order."""
    for got, want in zip(results['mixed']['grouped'], results['mixed']['singles']):
        assert abs(got - want) / results['n'] < TOL_EV_PER_ATOM, f'grouped {got} vs single {want}'
