"""SO3LR-Flash in ``so3lr eval``: the energies and forces of a dataset's structures.

``so3lr_eval.evaluate_so3lr_on`` runs it when every structure is periodic and orthorhombic
(and wide enough for the cell grid), the targets are energy and/or forces, and flash takes
the other settings (``check.flash_problem`` with task "eval"); else the neighbour-list model.
The structures run one at a time (``md.SinglePoints``), grouped by atoms, charge and spin,
then by PME mesh and box, so each engine is built once and the boxes it is laid out for
come in order.
"""
import time
from collections import defaultdict

import numpy as np

from so3lr.mlff.data.dataloader_sparse_ase import charge_and_multiplicity
from so3lr.mlff.nn.observable.dispersion_ref_data import (
    check_atomic_numbers_supported, supported_atomic_numbers_for_model)

from .check import NPT_BOX_MARGIN
from .engine import pme_mesh_shape
from .md import FlashUnavailable, SinglePoints, _pme


def charge_and_spin(atoms, total_charge=None, num_unpaired_electrons=None):
    """A structure's total charge and unpaired electrons as so3lr eval reads them: the
    overrides, else info's (charge 0 and multiplicity 1 where absent), as integers."""
    charge, multiplicity = charge_and_multiplicity(atoms)
    charge = (0 if charge is None else charge) if total_charge is None else total_charge
    spin = (1 if multiplicity is None else multiplicity) - 1 if num_unpaired_electrons is None else num_unpaired_electrons
    return float(np.int16(charge)), float(np.int16(spin))


def _order(frames, systems, settings):
    """The structures by atoms, charge and spin (in order of first appearance), then PME mesh
    and box; and whether a group has more than one box."""
    pme = [p for p in _pme(settings) if p]
    groups, meshes, boxes, keys = {}, {}, {}, []
    for atoms, system in zip(frames, systems):
        group = groups.setdefault((atoms.numbers.tobytes(),) + system, len(groups))
        L = tuple(float(v) for v in np.diag(atoms.cell[:]))
        if L not in meshes:
            meshes[L] = tuple(pme_mesh_shape(L, p[1]) for p in pme)
        boxes.setdefault(group, set()).add(L)
        keys.append((group, meshes[L], L))
    return sorted(range(len(frames)), key=keys.__getitem__), any(len(b) > 1 for b in boxes.values())


def evaluate_flash(frames, settings, targets, keep_predictions=False):
    """Energy and forces of each structure of frames: (sums, timing).

    sums: (tot_abs, tot_sq, tot_n), |error| and error^2 summed and the number of values per
    target, over the structures with that label (as so3lr eval sums them). timing: (seconds
    of the calls that built or laid out an engine, which compile; seconds of the others;
    their number). With keep_predictions each structure gets energy_so3lr and forces_so3lr
    in place, its labels as energy and forces instead of a calculator. Raises
    ``FlashUnavailable`` if the first structure fails, before any result."""
    s = settings
    # every structure's elements first, as the neighbour-list path checks them
    supported = supported_atomic_numbers_for_model(s.get("model_path"))
    for i, atoms in enumerate(frames):
        check_atomic_numbers_supported(atoms.numbers, supported, context=f"structure {i}")
    systems = [charge_and_spin(a, s.get("total_charge"), s.get("num_unpaired_electrons")) for a in frames]
    order, boxes_move = _order(frames, systems, s)
    tot_abs, tot_sq, tot_n = defaultdict(float), defaultdict(float), defaultdict(int)
    for target in targets:
        tot_n[target] += 0           # a target without labels reports NaN, as on neighbour lists
    t_first = t_rest = 0.0
    n_rest = 0
    points = None
    for k, i in enumerate(order):
        atoms, (charge, spin) = frames[i], systems[i]
        t0 = time.time()
        try:
            if points is None:
                # the margin only where a group's box moves: it costs a cell per side
                points = SinglePoints(s, task="eval", box_margin=NPT_BOX_MARGIN if boxes_move else 0.0)
            e, F, compiled = points(atoms, charge, spin)
        except Exception as exc:
            if k == 0:
                raise FlashUnavailable(f"{type(exc).__name__}: {exc}") from exc
            raise
        if compiled:
            t_first += time.time() - t0
        else:
            t_rest, n_rest = t_rest + time.time() - t0, n_rest + 1
        labels = {} if atoms.calc is None else atoms.calc.results
        for target, value in (("energy", e), ("forces", F)):
            y = labels.get(target)
            if target in targets and y is not None:
                d = np.asarray(value, np.float64) - np.asarray(y, np.float64)
                tot_abs[target] += float(np.abs(d).sum())
                tot_sq[target] += float((d * d).sum())
                tot_n[target] += int(d.size)
        if keep_predictions:
            atoms.calc = None
            atoms.info["energy_so3lr"], atoms.arrays["forces_so3lr"] = e, F
            if labels.get("energy") is not None:
                atoms.info["energy"] = float(labels["energy"])
            if labels.get("forces") is not None:
                atoms.arrays["forces"] = np.asarray(labels["forces"], np.float64)
            atoms.info["charge"] = int(charge)
    return (tot_abs, tot_sq, tot_n), (t_first, t_rest, n_rest)
