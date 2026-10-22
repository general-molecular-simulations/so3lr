"""Regenerate the saved predictions the model regression tests compare against.

Usage::

    python tools/make_test_references.py so3lr-tauri-m-2606 [--molecule dha]

Writes ``tests/test_data/<molecule>_<model>_ase.npz`` with the energy, forces and
c6_ratios that ``tests/test_ase.py::test_v2_ase_calculator`` checks, computed in
float64 through So3lrCalculator exactly as that test evaluates them. A reference
pins a model's current outputs: regenerate it only when those outputs are meant
to change, and say why in the commit.
"""
import argparse
import pathlib

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from ase.io import read  # noqa: E402

from so3lr import So3lrCalculator  # noqa: E402

TEST_DATA = pathlib.Path(__file__).resolve().parent.parent / 'tests' / 'test_data'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('model')
    parser.add_argument('--molecule', default='dha')
    args = parser.parse_args()

    atoms = read(TEST_DATA / f'{args.molecule}.xyz')
    atoms.cell = None
    atoms.pbc = False
    # Two calculators, as the test uses: a plain one for energy and forces and
    # one with the c6_ratios observable.
    atoms.calc = So3lrCalculator(model=args.model, dtype=np.float64)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms_obs = atoms.copy()
    atoms_obs.calc = So3lrCalculator(model=args.model, dtype=np.float64, has_aux=True,
                                     observables=['c6_ratios'])
    atoms_obs.get_potential_energy()
    c6_ratios = np.asarray(atoms_obs.calc.results['aux']['c6_ratios'])

    out = TEST_DATA / f'{args.molecule}_{args.model}_ase.npz'
    np.savez(out, energy=energy, forces=forces, c6_ratios=c6_ratios)
    print(f'wrote {out}: energy {energy:.8f} eV, max |F| {np.abs(forces).max():.6f} eV/A')


if __name__ == '__main__':
    main()
