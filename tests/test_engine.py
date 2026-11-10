"""So3lrEngine: energy and forces call after call, against So3lrCalculator on the same
structures (float64, so the ways agree to round-off), the way it takes for each box, its
neighbour lists allocated again on overflow, pickling, and the arguments it refuses."""
import pathlib
import pickle

import numpy as np
import numpy.testing as npt
import pytest
from ase.io import read

from so3lr import So3lrCalculator, So3lrEngine
from so3lr.flash.check import device_problem

pytestmark = pytest.mark.usefixtures("x64")

DATA = pathlib.Path(__file__).resolve().parent / "test_data"
MODEL = "so3lr-tauri-s-2602"
LR = 6.0                # a 12.4 A box of 64 waters holds 2 x this: minimum image
GPU = pytest.mark.skipif(device_problem() is not None, reason="flash needs an NVIDIA or AMD Instinct GPU")


def _water():
    atoms = read(DATA / "water_64.xyz")
    atoms.wrap()
    return atoms


def _molecule():
    atoms = read(DATA / "dha.xyz")
    atoms.pbc = False
    atoms.set_cell(np.zeros(3))
    return atoms


def _reference(atoms, charge=0, multiplicity=1, **kwargs):
    atoms = atoms.copy()
    atoms.info.update(charge=charge, multiplicity=multiplicity)
    atoms.calc = So3lrCalculator(model=MODEL, lr_cutoff=LR, precision="float64", **kwargs)
    return atoms.get_potential_energy(), atoms.get_forces()


def _engine(atoms, **kwargs):
    kwargs = dict(dict(model=MODEL, lr_cutoff=LR, precision="float64", engine="standard"), **kwargs)
    return So3lrEngine(atoms.numbers, **kwargs)


def _box(atoms):
    return np.asarray(atoms.cell[:]) if atoms.pbc.any() else None


def _assert_close(result, reference, energy=1e-8, forces=1e-7):
    npt.assert_allclose(result[0], reference[0], rtol=0, atol=energy)
    npt.assert_allclose(result[1], reference[1], rtol=0, atol=forces)


@pytest.mark.parametrize("charge,multiplicity", [(0, 1), (1, 2)])
def test_free_space_matches_calculator(charge, multiplicity):
    atoms = _molecule()
    engine = _engine(atoms, charge=charge, multiplicity=multiplicity)
    _assert_close(engine(atoms.positions), _reference(atoms, charge, multiplicity))
    assert engine.backend == "neighbour-list"


def test_periodic_matches_calculator_as_positions_and_box_change():
    atoms = _water()
    engine = _engine(atoms)
    rng = np.random.default_rng(0)
    for scale in (1.0, 1.0, 1.01, 0.995):
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)
        atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
        _assert_close(engine(atoms.positions, _box(atoms)), _reference(atoms))
        assert engine.backend == "neighbour-list"


def test_pme_matches_calculator():
    atoms = _water()
    kspace = dict(kspace_electrostatics="pme", kspace_smearing=2.0, kspace_spacing=1.0)
    engine = _engine(atoms, **kspace)
    _assert_close(engine(atoms.positions, _box(atoms)), _reference(atoms, **kspace), energy=1e-6, forces=1e-5)


def test_thin_and_triclinic_cells_take_exact_images():
    atoms = _water()
    exact = dict(neighbor_backend="exact")
    thin = _engine(atoms, skin=1.0)                           # 12.4 A < 2 x (6 + 1) A
    _assert_close(thin(atoms.positions, _box(atoms)), _reference(atoms, **exact))
    assert thin.backend == "exact"
    skewed = atoms.copy()
    cell = np.asarray(skewed.cell[:]).copy()
    cell[1, 0] = 2.0
    skewed.set_cell(cell, scale_atoms=False)
    skewed.wrap()
    engine = _engine(skewed)
    _assert_close(engine(skewed.positions, _box(skewed)), _reference(skewed, **exact))
    assert engine.backend == "exact"


def test_overflowing_lists_are_allocated_again():
    atoms = _water()
    dilute = atoms.copy()
    dilute.set_cell(dilute.cell * 1.15, scale_atoms=True)
    engine = _engine(atoms)
    engine(dilute.positions, _box(dilute))
    capacity = engine._nbrs.idx.shape
    # 1.5 x the density of the box the lists were sized in: they overflow
    _assert_close(engine(atoms.positions, _box(atoms)), _reference(atoms))
    assert engine._nbrs.idx.shape[1] > capacity[1]


def test_pickles_as_its_arguments():
    atoms = _molecule()
    engine = _engine(atoms, charge=1, multiplicity=2)
    first = engine(atoms.positions)
    copy = pickle.loads(pickle.dumps(engine))
    assert copy._step is None
    _assert_close(copy(atoms.positions), first, energy=1e-10, forces=1e-10)


@pytest.mark.parametrize("kwargs,match", [
    (dict(engine="fast"), "engine must be one of"),
    (dict(kspace_electrostatics="p3m"), "kspace_electrostatics must be"),
    (dict(kspace_smearing=2.0), "need kspace_electrostatics"),
])
def test_refuses_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _engine(_molecule(), **kwargs)


def test_refuses_unsupported_elements():
    with pytest.raises(ValueError, match="SO3LR model so3lr"):
        So3lrEngine([1, 8, 92], model="so3lr", precision="float64")


@pytest.mark.skipif(device_problem() is None, reason="on a GPU flash can run")
def test_engine_flash_raises_where_flash_cannot_run():
    atoms = _molecule()
    engine = _engine(atoms, engine="flash")
    with pytest.raises(ValueError, match="flash: "):
        engine(atoms.positions)


@GPU
@pytest.mark.parametrize("periodic", [False, True])
def test_flash_matches_neighbour_lists(periodic):
    atoms = _water() if periodic else _molecule()
    if periodic:
        atoms = atoms.repeat(2)
    flash = _engine(atoms, engine="auto")
    lists = _engine(atoms)
    result = flash(atoms.positions, _box(atoms))
    assert flash.backend == "flash"
    _assert_close(result, lists(atoms.positions, _box(atoms)), energy=1e-7, forces=1e-6)
