"""Regenerate the saved predictions the model regression tests compare against.

Usage::

    python tools/make_test_references.py so3lr-tauri-m-2606

Writes ``tests/test_data/dha_<model>_ase.npz`` with the energy, forces and
c6_ratios that ``tests/test_ase.py::test_v2_ase_calculator`` checks, computed in
float64 through So3lrCalculator exactly as that test evaluates them. A reference
pins a model's current outputs: regenerate it only when those outputs are meant
to change, and say why in the commit. A new model also needs an entry in
V2_MODELS and V2_TOLERANCES in tests/test_ase.py.
"""
import argparse
import pathlib

import jax
import numpy as np
from ase.io import read

from so3lr import So3lrCalculator

TEST_DATA = pathlib.Path(__file__).resolve().parent.parent / 'tests' / 'test_data'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('model')
    args = parser.parse_args()
    # Without x64 a float64 So3lrCalculator silently computes in float32.
    jax.config.update('jax_enable_x64', True)

    atoms = read(TEST_DATA / 'dha.xyz')
    atoms.cell = None
    atoms.pbc = False
    atoms.calc = So3lrCalculator(model=args.model, dtype=np.float64, has_aux=True,
                                 observables=['c6_ratios'])
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    c6_ratios = np.asarray(atoms.calc.results['aux']['c6_ratios'])

    out = TEST_DATA / f'dha_{args.model}_ase.npz'
    np.savez(out, energy=energy, forces=forces, c6_ratios=c6_ratios)
    print(f'wrote {out}: energy {energy:.8f} eV, max |F| {np.abs(forces).max():.6f} eV/A')


if __name__ == '__main__':
    main()
