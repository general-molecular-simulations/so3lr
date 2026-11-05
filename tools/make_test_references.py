"""Regenerate the saved predictions the model regression tests compare against.

Usage::

    python tools/make_test_references.py so3lr-tauri-m-2606

Writes ``tests/test_data/dha_<model>_ase.npz`` with the energy, forces and
c6_ratios that ``tests/test_ase.py::test_v2_ase_calculator`` checks, computed in
float64 through So3lrCalculator exactly as that test evaluates them. A reference
pins one release's current outputs, so the model must be a canonical (dated)
name, not a floating alias. Regenerate a reference only when those outputs are
meant to change, and say why in the commit. A new model also needs an entry in
V2_MODELS in tests/model_references.py.
"""
import argparse
import pathlib

import jax
import numpy as np
from ase.io import read

from so3lr import So3lrCalculator
from so3lr.model_registry import canonical_models

TEST_DATA = pathlib.Path(__file__).resolve().parent.parent / 'tests' / 'test_data'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('model')
    args = parser.parse_args()
    if args.model not in canonical_models():
        parser.error(f'{args.model!r} is not a canonical model name: one of {sorted(canonical_models())}')
    # Without x64 a float64 So3lrCalculator silently computes in float32.
    jax.config.update('jax_enable_x64', True)

    atoms = read(TEST_DATA / 'dha.xyz')
    atoms.cell = None
    atoms.pbc = False
    atoms.calc = So3lrCalculator(model=args.model, dtype=np.float64)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms_obs = atoms.copy()
    atoms_obs.calc = So3lrCalculator(model=args.model, dtype=np.float64, has_aux=True,
                                     observables=['c6_ratios'])
    atoms_obs.get_potential_energy()
    c6_ratios = np.asarray(atoms_obs.calc.results['aux']['c6_ratios'])

    out = TEST_DATA / f'dha_{args.model}_ase.npz'
    np.savez(out, energy=energy, forces=forces, c6_ratios=c6_ratios)
    print(f'wrote {out}: energy {energy:.8f} eV, max |F| {np.abs(forces).max():.6f} eV/A')


if __name__ == '__main__':
    main()
