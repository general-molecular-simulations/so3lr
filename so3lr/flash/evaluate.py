"""SO3LR-Flash in ``so3lr eval``: the energies and forces of a dataset's structures.

``so3lr_eval.evaluate_so3lr_on`` runs it when every structure is periodic and orthorhombic
(and wide enough for the cell grid), the targets are energy and/or forces, and flash takes
the other settings (``check.flash_problem`` with task "eval"); else the neighbour-list model.
The structures run one at a time: an engine serves consecutive structures with the same
atoms, charge and spin (a new box compiles nothing within the cell grid's margin, unless
its PME mesh differs), and the next composition builds a new one.
"""
import gc
import time

import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError

from so3lr.mlff.nn.observable.dispersion_ref_data import (
    check_atomic_numbers_supported, supported_atomic_numbers_for_model)

from .check import NPT_BOX_MARGIN
from .md import FlashUnavailable, build_engine


def charge_and_spin(atoms, total_charge=None, num_unpaired_electrons=None):
    """A structure's total charge and unpaired electrons as so3lr eval reads them: the
    overrides, else info's charge (0) and multiplicity, or spin, (1) less one."""
    charge = atoms.info.get("charge", 0) if total_charge is None else total_charge
    if num_unpaired_electrons is not None:
        return float(charge), float(num_unpaired_electrons)
    multiplicity = atoms.info.get("multiplicity")
    if multiplicity is None:
        multiplicity = atoms.info.get("spin", 1)
    return float(charge), float(multiplicity) - 1.0


def label(atoms, target):
    """The structure's energy or forces label, None if it has none."""
    if atoms.calc is None:
        return None
    try:
        return atoms.get_potential_energy() if target == "energy" else atoms.get_forces()
    except PropertyNotImplementedError:
        return None


def evaluate_flash(frames, settings, targets, keep_predictions=False):
    """Energy and forces of each structure of frames: (sums, predictions, timing).

    sums: per target, |error| and error^2 summed, and the number of values, over the
    structures with that label (as so3lr eval sums them); predictions: copies of the
    structures with energy_so3lr and forces_so3lr (and their labels) if keep_predictions;
    timing: (seconds to build each engine and run its first call, which compiles;
    seconds of the other calls; their number). Raises ``FlashUnavailable`` if the first
    structure's engine fails, before any result."""
    s = settings
    dtype = jnp.float64 if s.get("precision", "float32") == "float64" else jnp.float32
    # positions in float64 around a float32 model, as in MD
    sdt = jnp.float64 if dtype == jnp.float32 and bool(s.get("flash_mixed", True)) else dtype
    supported = supported_atomic_numbers_for_model(s.get("model_path"))
    sums = {t: [0.0, 0.0, 0] for t in targets}
    predictions = []
    t_first = t_rest = 0.0
    n_rest = 0
    eng = ef = key = None
    for i, atoms in enumerate(frames):
        charge, spin = charge_and_spin(atoms, s.get("total_charge"), s.get("num_unpaired_electrons"))
        composition = (atoms.numbers.tobytes(), charge, spin)
        if composition != key:
            check_atomic_numbers_supported(atoms.numbers, supported, context=f"structure {i}")
        t0 = time.time()
        try:
            if composition != key:
                eng = ef = None
                gc.collect()                 # the last engine's arrays, before the next one's
                eng, staged = build_engine(atoms, dict(s, total_charge=charge, num_unpaired_electrons=spin),
                                           dtype, box_margin=NPT_BOX_MARGIN)
                ef = eng.energy_forces_staged if staged else eng.energy_forces
                key, first = composition, True
            else:
                program = eng._jit
                eng.set_box(np.diag(np.asarray(atoms.cell[:])), exact_mesh=True)
                first = eng._jit is not program  # laid out again: the call compiles
            e, F = ef(jnp.asarray(atoms.get_positions(wrap=True), sdt))
            e, F = float(e), np.asarray(F, np.float64)
        except Exception as exc:
            if i == 0:
                raise FlashUnavailable(f"{type(exc).__name__}: {exc}") from exc
            raise
        if first:
            t_first += time.time() - t0
        else:
            t_rest, n_rest = t_rest + time.time() - t0, n_rest + 1
        out = atoms.copy() if keep_predictions else None
        for target, value in (("energy", e), ("forces", F)):
            y = label(atoms, target)
            if target in sums and y is not None:
                d = np.asarray(value, np.float64) - np.asarray(y, np.float64)
                sums[target][0] += float(np.abs(d).sum())
                sums[target][1] += float((d * d).sum())
                sums[target][2] += int(d.size)
            if out is not None:
                where = out.info if target == "energy" else out.arrays
                where[f"{target}_so3lr"] = value
                if y is not None:
                    where[target] = float(y) if target == "energy" else np.asarray(y, np.float64)
        if out is not None:
            out.info["charge"] = int(round(charge))
            predictions.append(out)
    return sums, predictions, (t_first, t_rest, n_rest)
