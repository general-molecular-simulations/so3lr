"""Saved model predictions the regression tests compare against.

Each reference is ``tests/test_data/dha_<model>_ase.npz`` (energy, forces,
c6_ratios), written by ``tests/make_test_references.py``. Exact releases only,
never floating aliases: a reference pins one checkpoint's outputs. A plain module
(like subprocess_env.py) so test files share it without importing each other.
"""
import pathlib

V2_MODELS = ['so3lr-tauri-s-2602', 'so3lr-tauri-m-2606']

# Absolute bounds, eV, eV/A and dimensionless, for any float64 path against a
# saved reference. Measured on MeluXina, every model and path (So3lrCalculator,
# jax-md) matches its reference to <= 6e-14; these leave >= 6 orders of margin
# for other hardware and are still far below any real change in a model output.
TOLERANCES = dict(energy=1e-7, forces=1e-6, c6_ratios=1e-8)


def reference_path(model):
    return pathlib.Path(__file__).resolve().parent / 'test_data' / f'dha_{model}_ase.npz'
