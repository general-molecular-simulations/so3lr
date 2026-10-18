"""MD observables: the single source of truth for ``--observable``.

Three hand-maintained lists had drifted apart, so most of the documented
``--observable`` choices could not work end to end:

* ``so3lr_cli.py`` listed nine choices in its ``click.Choice``;
* ``so3lr_md.py`` only forwarded ``output_intermediate_quantities`` to the model
  when ``partial_charges`` happened to be among them, so everything else raised
  ``KeyError`` after the first MD cycle;
* ``init_hdf5_store`` created datasets for a *different* eight names, four of
  which the CLI cannot even select, and omitted five that it can.

Everything now derives from the tables below. This module must stay
import-cheap -- ``so3lr/cli/so3lr_cli.py`` imports it at module scope and
``so3lr --help`` must not pull in JAX.
"""

from typing import Tuple

__all__ = [
    'MD_OBSERVABLES',
    'PER_ATOM_OBSERVABLES',
    'is_per_atom_observable',
]

# Every value accepted by `--observable`, in help order. All are emitted by the
# model as intermediate quantities (see ObservableSparse.get_intermediate_quantities).
# Note `zbl_repulsion` (legacy `so3lr`) and `nlh_repulsion` (so3lr-tauri-s/m/l) are the
# same physical quantity under the two model generations' different output keys;
# requesting the wrong one for a model raises a clear ValueError from the model.
MD_OBSERVABLES: Tuple[str, ...] = (
    'nn_energy',
    'partial_charges',
    'electrostatic_energy',
    'dispersion_energy',
    'zbl_repulsion',
    'nlh_repulsion',
    'electrostatic_energy_undamped',
    'dispersion_energy_undamped',
    'x',
)

# Observables shaped (num_atoms,) -- one scalar per atom. These are the ones the
# HDF5 writer can size a dataset for up front. Everything else (currently only
# 'x', the (num_atoms, num_features) feature vectors) has a model-dependent
# width and is supported for extxyz output only.
PER_ATOM_OBSERVABLES: Tuple[str, ...] = tuple(
    o for o in MD_OBSERVABLES if o != 'x'
)


def is_per_atom_observable(name: str) -> bool:
    """True if ``name`` is a one-scalar-per-atom observable (HDF5-storable)."""
    return name in PER_ATOM_OBSERVABLES
