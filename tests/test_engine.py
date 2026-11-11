"""So3lrEngine: energy and forces call after call, against So3lrCalculator on the same
structures (float64, so the ways agree to round-off), the way it takes for each box, its
neighbour lists allocated again on overflow, pickling, and the arguments it refuses."""
import pathlib
import pickle
import warnings

import numpy as np
import numpy.testing as npt
import pytest
from ase.io import read

from so3lr import So3lrCalculator, So3lrEngine
from so3lr.flash.check import NPT_BOX_MARGIN, device_problem

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


def _calculator(**kwargs):
    return So3lrCalculator(model=MODEL, lr_cutoff=LR, precision="float64", **kwargs)


def _reference(atoms, charge=0, multiplicity=1, calc=None, **kwargs):
    atoms = atoms.copy()
    atoms.info.update(charge=charge, multiplicity=multiplicity)
    atoms.calc = calc or _calculator(**kwargs)
    return atoms.get_potential_energy(), atoms.get_forces()


def _engine(atoms, **kwargs):
    return So3lrEngine(atoms.numbers, **{"model": MODEL, "lr_cutoff": LR, "precision": "float64",
                                         "engine": "standard", **kwargs})


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
    engine, calc = _engine(atoms), _calculator()
    rng = np.random.default_rng(0)
    for scale in (1.0, 1.0, 1.01, 0.995):
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)
        atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
        _assert_close(engine(atoms.positions, _box(atoms)), _reference(atoms, calc=calc))
        assert engine.backend == "neighbour-list"


@pytest.mark.parametrize("method", ["pme", "p3m"])
def test_mesh_kspace_matches_calculator(method):
    atoms = _water()
    kspace = dict(kspace_electrostatics=method, kspace_smearing=2.0, kspace_spacing=1.0)
    engine = _engine(atoms, **kspace)
    _assert_close(engine(atoms.positions, _box(atoms)), _reference(atoms, **kspace), energy=1e-6, forces=1e-5)


def _skewed(atoms):
    cell = np.asarray(atoms.cell[:]).copy()
    cell[1, 0] = 2.0
    atoms.set_cell(cell, scale_atoms=False)
    atoms.wrap()
    return atoms


# thin: 12.4 A < 2 x (6 + 1) A; the triclinic one charged, its charge and spin through atoms.info
@pytest.mark.parametrize("skin,skew,charge,multiplicity", [(1.0, False, 0, 1), (0.0, True, 1, 2)],
                         ids=["thin", "triclinic"])
def test_thin_and_triclinic_cells_take_exact_images(skin, skew, charge, multiplicity):
    atoms = _skewed(_water()) if skew else _water()
    engine = _engine(atoms, skin=skin, charge=charge, multiplicity=multiplicity)
    _assert_close(engine(atoms.positions, _box(atoms)),
                  _reference(atoms, charge, multiplicity, neighbor_backend="exact"))
    assert engine.backend == "exact"


def test_one_engine_goes_between_free_space_and_a_box():
    atoms = _water()
    cluster = atoms.copy()
    cluster.pbc = False
    cluster.set_cell(np.zeros(3))
    # one calculator for each: a So3lrCalculator keeps the neighbour list of its first system
    references = {False: _reference(cluster), True: _reference(atoms)}
    engine = _engine(atoms)
    for periodic in (False, True, False):
        _assert_close(engine(atoms.positions, _box(atoms) if periodic else None), references[periodic])
        assert engine.backend == "neighbour-list"


def test_overflowing_lists_are_allocated_again():
    atoms = _water()
    dilute = atoms.copy()
    dilute.set_cell(dilute.cell * 1.15, scale_atoms=True)
    engine = _engine(atoms)
    engine(dilute.positions, _box(dilute))
    capacity = engine._nbrs.idx.shape[1]
    # 1.5 x the density of the box the lists were sized in: they overflow
    _assert_close(engine(atoms.positions, _box(atoms)), _reference(atoms))
    assert engine._nbrs.idx.shape[1] > capacity


def test_a_zero_box_is_free_space_and_a_flat_one_is_refused():
    atoms = _molecule()
    engine = _engine(atoms)
    _assert_close(engine(atoms.positions, np.zeros((3, 3))), engine(atoms.positions), energy=1e-10, forces=1e-10)
    with pytest.raises(ValueError, match="spans no volume"):
        engine(atoms.positions, np.diag([10.0, 10.0, 0.0]))


def test_pickles_as_its_arguments():
    atoms = _molecule()
    engine = _engine(atoms, charge=1, multiplicity=2)
    first = engine(atoms.positions)
    copy = pickle.loads(pickle.dumps(engine))
    assert copy._step is None
    _assert_close(copy(atoms.positions), first, energy=1e-10, forces=1e-10)


@pytest.mark.parametrize("kwargs,match", [
    (dict(engine="fast"), "engine must be one of"),
    (dict(kspace_electrostatics="spme"), "kspace_electrostatics must be"),
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


def test_auto_falls_back_where_flash_fails_to_start(monkeypatch):
    import so3lr.flash.check as flash_check
    flash_md = pytest.importorskip("so3lr.flash.md")

    def fails(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(flash_md, "SinglePoints", fails)
    monkeypatch.setattr(flash_check, "structure_problem", lambda *args: None)
    atoms = _molecule()
    engine = _engine(atoms, engine="auto")
    engine._flash_refusal = None            # as on a GPU where the settings, model and device pass
    with pytest.warns(UserWarning, match="running without flash, as it failed"):
        result = engine(atoms.positions)
    assert engine.backend == "neighbour-list"
    _assert_close(result, _reference(atoms))


class _FlashWithoutRoomForTheMargin:
    """Flash that runs a fixed box, but fails to start again with NPT's margin."""
    def __init__(self, settings, task, matmul_precision, box_margin):
        if box_margin:
            raise RuntimeError("out of memory")
        self.box_margin = box_margin

    def __call__(self, atoms, charge, spin):
        return 0.0, np.zeros((len(atoms), 3)), True


def test_auto_falls_back_where_flash_fails_to_start_again_for_a_moving_box(monkeypatch):
    import so3lr.flash.check as flash_check
    flash_md = pytest.importorskip("so3lr.flash.md")
    monkeypatch.setattr(flash_md, "SinglePoints", _FlashWithoutRoomForTheMargin)
    monkeypatch.setattr(flash_check, "structure_problem", lambda *args: None)
    atoms = _water()
    engine = _engine(atoms, engine="auto")
    engine._flash_refusal = None
    engine(atoms.positions, _box(atoms))
    assert engine.backend == "flash"
    atoms.set_cell(atoms.cell * 0.995, scale_atoms=True)
    with pytest.warns(UserWarning, match="running without flash, as it failed"):
        result = engine(atoms.positions, _box(atoms))
    assert engine.backend == "neighbour-list"
    _assert_close(result, _reference(atoms))
    # flash stays off: the next box is not tried on it again
    atoms.set_cell(atoms.cell * 0.995, scale_atoms=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine(atoms.positions, _box(atoms))
    assert not [w for w in caught if "flash" in str(w.message)]
    assert engine.backend == "neighbour-list"
    # engine="flash" raises instead
    strict = _engine(atoms, engine="flash")
    strict._flash_refusal = None
    strict(atoms.positions, _box(atoms))
    atoms.set_cell(atoms.cell * 0.995, scale_atoms=True)
    with pytest.raises(RuntimeError, match="out of memory"):
        strict(atoms.positions, _box(atoms))


@GPU
@pytest.mark.parametrize("periodic", [False, True])
def test_flash_matches_neighbour_lists(periodic):
    atoms = _water().repeat(2) if periodic else _molecule()
    flash = _engine(atoms, engine="auto")
    lists = _engine(atoms)
    for scale in (1.0, 0.99, 0.995):         # periodic: the box moves, flash built again once
        if periodic:
            atoms.set_cell(atoms.cell * scale, scale_atoms=True)
        result = flash(atoms.positions, _box(atoms))
        assert flash.backend == "flash"
        _assert_close(result, lists(atoms.positions, _box(atoms)), energy=1e-7, forces=1e-6)
    assert flash._flash.box_margin == (NPT_BOX_MARGIN if periodic else 0.0)
