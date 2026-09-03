"""
Test PME/Ewald electrostatics against direct jax-pme calculations.

This test file validates that the electrostatic_energy_kspace extracted from
the SO3LR model matches direct calculations using jax-pme with the same
partial charges.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pathlib
import pytest
import logging
import sys
import types
from functools import partial
from collections import namedtuple

from ase import Atoms
from ase.io import read
from ase.calculators.calculator import Calculator

from so3lr.mlff.calculators.ase_calculator import AseCalculatorSparse
from so3lr import So3lrCalculator, So3lrPotential, to_jax_md
from so3lr.model_registry import resolve_model

# Original/legacy SO3LR model, resolved via the registry so the test is
# decoupled from the on-disk layout (the model now lives under models/so3lr).
MODEL_DIR = resolve_model('so3lr').dir
import jax_md

# Enable 64-bit precision
jax.config.update('jax_enable_x64', True)
DTYPE = jnp.float64

# Constants
KE = 14.399645351950548  # Coulomb constant in eV*Angstrom

DummySystem = namedtuple('DummySystem', ['R', 'k_grid', 'k_smearing'])


def make_stub_ase_calculator():
    calc = object.__new__(AseCalculatorSparse)
    Calculator.__init__(calc)
    calc.calculate_fn = lambda system, neighbors: {
        'energy': np.array(0.0),
        'forces': np.zeros((len(system.R), 3)),
    }
    calc.neighbors = None
    calc.spatial_partitioning = None
    calc.capacity_multiplier = 1.25
    calc.buffer_size_multiplier = 1.25
    calc.skin = 0.0
    calc.cutoff = 5.0
    calc.hessian_slice = False
    calc.lr_cutoff = -1.0
    calc.dtype = np.float64
    calc.kspace_electrostatics = 'pme'
    calc.kspace_smearing = 4.0
    calc.kspace_spacing = 0.5
    calc._kspace_reported_source = None
    calc.supported_atomic_numbers = None
    return calc


def get_package_dir():
    """Return the package directory path."""
    return pathlib.Path(__file__).parent.parent.resolve()


@pytest.fixture
def package_dir():
    """Fixture providing the package directory path."""
    return get_package_dir()


def _pme_test_box(package_dir, full_test):
    """The periodic box the k-space tests run on.

    The full version is water_64 replicated 2x2x2 (1536 atoms) at a 12 Ang
    cutoff in float64 -- the paper's inference settings. On a 2-core GitHub
    runner that took the so3lr-other job past its 60-minute budget every time
    (three parametrised methods x six tests, each compiling a 1536-atom f64
    model), which is what has been cancelling that job on main. By default use
    the unreplicated 192-atom box at KSPACE_LR_CUTOFF (the same constants the
    high-level k-space tests further down already use); minimum image still
    holds (12.4 Ang cell against a 6 Ang cutoff) and every assertion here is
    about shapes, finiteness, or agreement with a direct jax-pme evaluation on
    the same charges, none of which depends on system size. --full restores
    the large box.
    """
    box = read(package_dir / 'tests/test_data/water_64.xyz')
    return box * [2, 2, 2] if full_test else box


class TestPMEDirectComparison:
    """
    Test that electrostatic_energy_kspace and partial_charges from SO3LR
    can be used with jax-pme to reproduce the k-space energy.
    
    These tests extract the partial_charges from SO3LR and use them directly
    with jax-pme solvers to verify the k-space calculation is correct.
    """

    @pytest.mark.parametrize('kspace_method', ['ewald', 'pme', 'p3m'])
    def test_kspace_energy_matches_direct_calculation(self, kspace_method, package_dir, full_test):
        """
        Test that electrostatic_energy_kspace from SO3LR matches direct jax-pme calculation.
        
        This test:
        1. Runs SO3LR to get partial_charges and electrostatic_energy_kspace
        2. Uses the same partial charges with jax-pme directly
        3. Compares the k-space energies
        """
        from jaxpme.solvers import ewald, pme, p3m
        from jaxpme.potentials import potential as get_potential
        from jaxpme.kspace import generate_kvectors, get_reciprocal, get_kgrid_mesh, get_kgrid_ewald

        # Create a periodic water box - repeat to ensure enough for 12 Å cutoff
        atoms = _pme_test_box(package_dir, full_test)
        
        # Set up k-space parameters
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        positions = jnp.array(atoms.get_positions(), dtype=DTYPE)
        
        # Choose smearing based on cell size
        smearing = 4.0 if full_test else KSPACE_SMEARING
        
        # Set up k-grid based on method
        if kspace_method == 'ewald':
            k_spacing = jnp.array([smearing / 2])
            k_grid = get_kgrid_ewald(cell, k_spacing)
        else:  # pme or p3m
            k_spacing = jnp.array([smearing / 8])
            k_grid = get_kgrid_mesh(cell, k_spacing)
        
        k_smearing = jnp.array([smearing])
        
        # Store k-space info in atoms
        atoms.info['k_grid'] = np.array(k_grid)
        atoms.info['k_smearing'] = np.array(k_smearing)

        # Create SO3LR calculator with observables
        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12. if full_test else KSPACE_LR_CUTOFF,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['partial_charges', 'electrostatic_energy_kspace'],
            coulomb_kspace_method=kspace_method,
        )
        
        atoms.calc = calc
        
        # Get energy to trigger calculation
        _ = atoms.get_potential_energy()
        
        # Extract observables
        results = calc.results.get('aux', {})
        
        assert 'partial_charges' in results, "partial_charges not found in observables"
        assert 'electrostatic_energy_kspace' in results, "electrostatic_energy_kspace not found in observables"
        
        partial_charges = jnp.array(results['partial_charges'], dtype=DTYPE)
        so3lr_kspace_energy = jnp.sum(results['electrostatic_energy_kspace'])
        
        # Now compute k-space energy directly with jax-pme using the same charges
        volume = jnp.abs(jnp.linalg.det(cell))
        reciprocal_cell = get_reciprocal(cell)
        pbc = jnp.array([True, True, True])
        
        # Set up solver and compute potentials
        if kspace_method == 'ewald':
            solver = ewald(get_potential())
            kvectors = generate_kvectors(reciprocal_cell, k_grid.shape, dtype=DTYPE, for_ewald=True)
            potentials = solver.kspace(k_smearing, partial_charges, kvectors, positions, volume, cell, pbc)
        elif kspace_method == 'pme':
            solver = pme(get_potential(), interpolation_nodes=4)
            kvectors = generate_kvectors(reciprocal_cell, k_grid.shape, dtype=DTYPE, for_ewald=False)
            potentials = solver.kspace(k_smearing, partial_charges, reciprocal_cell, k_grid, kvectors, positions, volume, cell, pbc)
        else:  # p3m
            solver = p3m(get_potential(), interpolation_nodes=4)
            kvectors = generate_kvectors(reciprocal_cell, k_grid.shape, dtype=DTYPE, for_ewald=False)
            potentials = solver.kspace(k_smearing, partial_charges, reciprocal_cell, k_grid, kvectors, positions, volume, cell, pbc)
        
        # Calculate energy: E = sum(q * phi) with KE factor
        # Note: jax-pme already returns phi/2 to avoid double counting, so we just do sum(q * phi)
        direct_kspace_energy = jnp.sum(partial_charges * potentials) * KE
        
        print(f"\n{kspace_method.upper()} test:")
        print(f"  SO3LR k-space energy: {so3lr_kspace_energy:.6f} eV")
        print(f"  Direct k-space energy: {direct_kspace_energy:.6f} eV")
        print(f"  Difference: {so3lr_kspace_energy - direct_kspace_energy:.6e} eV")
        print(f"  Total charge: {jnp.sum(partial_charges):.6f}")
        print(f"  Number of atoms: {len(atoms)}")
        print(f"  k_grid shape: {k_grid.shape}")
        
        # Compare energies
        npt.assert_allclose(
            so3lr_kspace_energy,
            direct_kspace_energy,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"K-space energies don't match for {kspace_method}"
        )

    @pytest.mark.parametrize('kspace_method', ['ewald', 'pme', 'p3m'])
    def test_partial_charges_are_reasonable(self, kspace_method, package_dir, full_test):
        """
        Test that extracted partial charges have reasonable properties.
        
        For neutral water, the total charge should be approximately zero.
        """
        from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

        # Create a periodic water box
        atoms = _pme_test_box(package_dir, full_test)
        
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        smearing = 4.0 if full_test else KSPACE_SMEARING
        
        if kspace_method == 'ewald':
            k_spacing = jnp.array([smearing / 2])
            k_grid = get_kgrid_ewald(cell, k_spacing)
        else:
            k_spacing = jnp.array([smearing / 8])
            k_grid = get_kgrid_mesh(cell, k_spacing)
        
        atoms.info['k_grid'] = np.array(k_grid)
        atoms.info['k_smearing'] = np.array([smearing])

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12. if full_test else KSPACE_LR_CUTOFF,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['partial_charges'],
            coulomb_kspace_method=kspace_method,
        )
        
        atoms.calc = calc
        _ = atoms.get_potential_energy()
        
        results = calc.results.get('aux', {})
        partial_charges = results['partial_charges']
        
        # Check shape
        assert partial_charges.shape[0] == len(atoms), \
            f"Expected {len(atoms)} charges, got {partial_charges.shape[0]}"
        
        # For neutral water, total charge should be ~0
        total_charge = np.sum(partial_charges)
        
        print(f"\n{kspace_method.upper()} partial charges test:")
        print(f"  Number of atoms: {len(atoms)}")
        print(f"  Total charge: {total_charge:.6f}")
        print(f"  Min charge: {np.min(partial_charges):.6f}")
        print(f"  Max charge: {np.max(partial_charges):.6f}")
        
        npt.assert_allclose(
            total_charge,
            0.0,
            atol=1e-5,
            err_msg=f"Total charge should be ~0 for neutral water with {kspace_method}"
        )


class TestASECalculatorKSpaceReporting:
    def test_logs_atoms_info_override_for_kspace_kwargs(self, monkeypatch, caplog):
        calc = make_stub_ase_calculator()
        atoms = Atoms('H2', positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
        atoms.cell = np.eye(3) * 10.0
        atoms.pbc = True
        atoms.info['k_grid'] = np.ones((2, 2, 2))
        atoms.info['k_smearing'] = np.array([4.0])

        dummy_system = DummySystem(
            R=np.asarray(atoms.get_positions()),
            k_grid=atoms.info['k_grid'],
            k_smearing=atoms.info['k_smearing'],
        )
        monkeypatch.setattr(
            'so3lr.mlff.calculators.ase_calculator.atoms_to_system',
            lambda atoms, dtype: dummy_system,
        )

        dummy_neighbors = types.SimpleNamespace(overflow=False, cell_list=None)
        dummy_partitioning = types.SimpleNamespace(
            update_fn=lambda positions, neighbors, new_cell=None: dummy_neighbors,
        )
        monkeypatch.setattr(
            'so3lr.mlff.calculators.ase_calculator.neighbor_list',
            lambda **kwargs: (dummy_neighbors, dummy_partitioning),
        )

        # setup_logger() (so3lr_md.py) sets propagate=False on "SO3LR", and
        # caplog listens on the root logger, so after any test in the same
        # process has run the CLI/eval path these records never reach it.
        # Re-enable propagation for this test only.
        monkeypatch.setattr(logging.getLogger("SO3LR"), "propagate", True)
        with caplog.at_level(logging.INFO, logger="SO3LR"):
            calc.calculate(atoms)

        assert 'Using k-space grid from atoms.info; it overrides ASE calculator k-space kwargs.' in caplog.text

    def test_logs_generated_kspace_grid_from_kwargs(self, monkeypatch, caplog):
        calc = make_stub_ase_calculator()
        atoms = Atoms('H2', positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
        atoms.cell = np.eye(3) * 10.0
        atoms.pbc = True

        dummy_system = DummySystem(
            R=np.asarray(atoms.get_positions()),
            k_grid=None,
            k_smearing=None,
        )
        monkeypatch.setattr(
            'so3lr.mlff.calculators.ase_calculator.atoms_to_system',
            lambda atoms, dtype: dummy_system,
        )

        dummy_neighbors = types.SimpleNamespace(overflow=False, cell_list=None)
        dummy_partitioning = types.SimpleNamespace(
            update_fn=lambda positions, neighbors, new_cell=None: dummy_neighbors,
        )
        monkeypatch.setattr(
            'so3lr.mlff.calculators.ase_calculator.neighbor_list',
            lambda **kwargs: (dummy_neighbors, dummy_partitioning),
        )

        fake_jaxpme = types.SimpleNamespace(
            kspace=types.SimpleNamespace(
                get_kgrid_ewald=lambda cell, lr_wavelength: np.ones((2, 2, 2)),
                get_kgrid_mesh=lambda cell, mesh_spacing: np.ones((2, 2, 2)),
            )
        )
        monkeypatch.setitem(sys.modules, 'jaxpme', fake_jaxpme)

        # setup_logger() (so3lr_md.py) sets propagate=False on "SO3LR", and
        # caplog listens on the root logger, so after any test in the same
        # process has run the CLI/eval path these records never reach it.
        # Re-enable propagation for this test only.
        monkeypatch.setattr(logging.getLogger("SO3LR"), "propagate", True)
        with caplog.at_level(logging.INFO, logger="SO3LR"):
            calc.calculate(atoms)

        assert 'Generating PME k-space grid from ASE calculator kwargs.' in caplog.text


class TestPMEEnergyDecomposition:
    """Test that k-space energy properly contributes to total energy."""

    @pytest.mark.parametrize('kspace_method', ['pme'])  # Start with PME only for speed
    def test_kspace_energy_contribution(self, kspace_method, package_dir, full_test):
        """
        Test that electrostatic_energy_kspace adds a sensible contribution 
        to the total electrostatic energy.
        """
        from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

        # Create a periodic water box
        atoms = _pme_test_box(package_dir, full_test)
        
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        smearing = 4.0 if full_test else KSPACE_SMEARING
        
        if kspace_method == 'ewald':
            k_spacing = jnp.array([smearing / 2])
            k_grid = get_kgrid_ewald(cell, k_spacing)
        else:
            k_spacing = jnp.array([smearing / 8])
            k_grid = get_kgrid_mesh(cell, k_spacing)
        
        atoms.info['k_grid'] = np.array(k_grid)
        atoms.info['k_smearing'] = np.array([smearing])

        # Get calculation with both short-range and k-space electrostatics
        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12. if full_test else KSPACE_LR_CUTOFF,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['electrostatic_energy', 'electrostatic_energy_kspace'],
            coulomb_kspace_method=kspace_method,
        )
        
        atoms.calc = calc
        total_energy = atoms.get_potential_energy()
        
        results = calc.results.get('aux', {})
        
        sr_energy = np.sum(results.get('electrostatic_energy', 0))
        kspace_energy = np.sum(results.get('electrostatic_energy_kspace', 0))
        
        print(f"\n{kspace_method.upper()} energy decomposition:")
        print(f"  Total energy: {total_energy:.6f} eV")
        print(f"  Short-range electrostatic: {sr_energy:.6f} eV")
        print(f"  K-space electrostatic: {kspace_energy:.6f} eV")
        
        # k-space energy should be non-zero for a charged system
        # (though it may be small for neutral water)
        assert np.isfinite(kspace_energy), "K-space energy should be finite"
        assert np.isfinite(sr_energy), "Short-range energy should be finite"


class TestPMEForces:
    """Test that forces can be computed with PME."""

    @pytest.mark.parametrize('kspace_method', ['ewald', 'pme', 'p3m'])
    def test_forces_are_finite(self, kspace_method, package_dir, full_test):
        """
        Test that forces are computed correctly with PME.
        """
        from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

        # Use a periodic water box
        atoms = _pme_test_box(package_dir, full_test)
        
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        smearing = 4.0 if full_test else KSPACE_SMEARING
        
        if kspace_method == 'ewald':
            k_spacing = jnp.array([smearing / 2])
            k_grid = get_kgrid_ewald(cell, k_spacing)
        else:
            k_spacing = jnp.array([smearing / 8])
            k_grid = get_kgrid_mesh(cell, k_spacing)
        
        atoms.info['k_grid'] = np.array(k_grid)
        atoms.info['k_smearing'] = np.array([smearing])

        calc = So3lrCalculator(
            lr_cutoff=12. if full_test else KSPACE_LR_CUTOFF,
            dtype=np.float64,
            coulomb_kspace_method=kspace_method,
        )
        
        atoms.calc = calc
        
        # Get analytical forces
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        print(f"\n{kspace_method.upper()} forces test:")
        print(f"  Energy: {energy:.6f} eV")
        print(f"  Max force magnitude: {np.max(np.linalg.norm(forces, axis=1)):.6f} eV/Å")
        print(f"  Mean force magnitude: {np.mean(np.linalg.norm(forces, axis=1)):.6f} eV/Å")
        
        # Forces should be finite
        assert np.all(np.isfinite(forces)), "Forces contain non-finite values"
        
        # Check shape
        assert forces.shape == (len(atoms), 3), \
            f"Expected forces shape ({len(atoms)}, 3), got {forces.shape}"


class TestPMEStress:
    """Test PME stress calculations."""

    @pytest.mark.parametrize('kspace_method', ['ewald', 'pme', 'p3m'])
    def test_stress_calculation(self, kspace_method, package_dir, full_test):
        """
        Test that stress can be calculated with PME.
        """
        from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

        atoms = _pme_test_box(package_dir, full_test)
        
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        smearing = 4.0 if full_test else KSPACE_SMEARING
        
        if kspace_method == 'ewald':
            k_spacing = jnp.array([smearing / 2])
            k_grid = get_kgrid_ewald(cell, k_spacing)
        else:
            k_spacing = jnp.array([smearing / 8])
            k_grid = get_kgrid_mesh(cell, k_spacing)
        
        atoms.info['k_grid'] = np.array(k_grid)
        atoms.info['k_smearing'] = np.array([smearing])

        calc = So3lrCalculator(
            lr_cutoff=12. if full_test else KSPACE_LR_CUTOFF,
            dtype=np.float64,
            calculate_stress=True,
            coulomb_kspace_method=kspace_method,
        )
        
        atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        stress = atoms.get_stress()
        
        print(f"\n{kspace_method.upper()} stress test:")
        print(f"  Energy: {energy:.6f} eV")
        print(f"  Stress (Voigt): {stress}")
        
        # Stress should be in Voigt notation (6 components)
        assert stress.shape == (6,), f"Expected stress shape (6,), got {stress.shape}"
        
        # Stress should be finite
        assert np.all(np.isfinite(stress)), "Stress contains non-finite values"


class TestPMEJaxMD:
    """Test PME with JAX-MD backend."""

    @pytest.mark.parametrize('kspace_method', ['ewald', 'pme', 'p3m'])
    def test_jaxmd_pme_energy(self, kspace_method, package_dir, full_test):
        """
        Test PME energy calculation through JAX-MD interface.
        """
        from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

        atoms = _pme_test_box(package_dir, full_test)
        
        cell_ase = np.array(atoms.get_cell().array, dtype=np.float64)
        smearing = 4.0 if full_test else KSPACE_SMEARING
        
        if kspace_method == 'ewald':
            k_spacing = jnp.array([smearing / 2])
            k_grid = get_kgrid_ewald(jnp.array(cell_ase), k_spacing)
        else:
            k_spacing = jnp.array([smearing / 8])
            k_grid = get_kgrid_mesh(jnp.array(cell_ase), k_spacing)
        
        k_smearing = jnp.array([smearing])

        species = jnp.array(atoms.get_atomic_numbers())
        masses = jnp.array(atoms.get_masses())
        box = jnp.diag(jnp.array(cell_ase))
        
        displacement, shift = jax_md.space.periodic_general(
            box=box, fractional_coordinates=True
        )
        
        positions = jnp.array(atoms.get_positions(), dtype=DTYPE) / box
        
        neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
            potential=So3lrPotential(
                dtype=DTYPE,
                # Same cutoff switch as the ASE cases: at the default 12 A the
                # 192-atom box (12.4 A) would violate minimum image.
                lr_cutoff=12. if full_test else KSPACE_LR_CUTOFF,
                coulomb_kspace_method=kspace_method,
            ),
            displacement_or_metric=displacement,
            shift_fn=shift,
            box_size=box,
            species=species,
            capacity_multiplier=1.25,
            buffer_size_multiplier_sr=1.25,
            buffer_size_multiplier_lr=1.25,
            minimum_cell_size_multiplier_sr=1.0,
            disable_cell_list=False,
            fractional_coordinates=True,
        )
        
        energy_fn = jax.jit(partial(energy_fn, has_aux=False))
        
        nbrs = neighbor_fn.allocate(positions, box=box)
        nbrs_lr = neighbor_fn_lr.allocate(positions, box=box)
        
        energy = energy_fn(
            positions,
            nbrs.idx,
            nbrs_lr.idx,
            box=box,
            species=species,
            masses=masses,
            k_grid=k_grid,
            k_smearing=k_smearing,
        )
        
        print(f"\n{kspace_method.upper()} JAX-MD energy test:")
        print(f"  Energy: {float(energy):.6f} eV")
        
        assert np.isfinite(energy), "Energy is not finite"


class TestKspaceMethodPropagation:
    """`kspace_electrostatics` must configure the solver, not just the grid.

    Regression test. ``AseCalculatorSparse.create_from_workdir`` used to accept
    ``kspace_electrostatics`` and use it only when building the k-grid in
    ``calculate``; the solver inside the model is configured from
    ``long_range_kwargs['coulomb_kspace_method']``, which defaults to 'pme'.
    Requesting 'ewald' therefore produced an Ewald k-point set evaluated by the
    PME solver -- wrong energies, no error. The CLI was unaffected because it
    populates ``coulomb_kspace_method`` itself.
    """

    @staticmethod
    def _captured_long_range_kwargs(monkeypatch, **calc_kwargs):
        """Build a calculator, capturing long_range_kwargs without loading a model."""
        captured = {}

        class _Stub:
            @staticmethod
            def create_from_workdir(*args, **kwargs):
                captured.update(kwargs.get('long_range_kwargs', {}))
                return object()

        monkeypatch.setattr(
            'so3lr.mlff.calculators.ase_calculator.PotentialSparse', _Stub
        )
        # __init__ inspects the potential; bypass it, we only want the kwargs.
        monkeypatch.setattr(
            AseCalculatorSparse, '__init__', lambda self, **kw: None
        )
        AseCalculatorSparse.create_from_workdir(workdir=MODEL_DIR, **calc_kwargs)
        return captured

    @pytest.mark.parametrize('method', ['ewald', 'pme', 'p3m'])
    def test_method_reaches_the_solver(self, monkeypatch, method):
        lrk = self._captured_long_range_kwargs(
            monkeypatch,
            kspace_electrostatics=method,
            kspace_smearing=4.0,
            kspace_spacing=0.5,
        )
        assert lrk['coulomb_kspace_method'] == method, (
            f"kspace_electrostatics='{method}' must select the '{method}' solver; "
            f"got '{lrk.get('coulomb_kspace_method')}'. The grid and the solver "
            f"would disagree."
        )

    def test_explicit_low_level_spelling_still_wins(self, monkeypatch):
        """An explicit coulomb_kspace_method overrides the high-level name."""
        lrk = self._captured_long_range_kwargs(
            monkeypatch,
            kspace_electrostatics='ewald',
            kspace_smearing=4.0,
            kspace_spacing=0.5,
            coulomb_kspace_method='p3m',
        )
        assert lrk['coulomb_kspace_method'] == 'p3m'

    def test_absent_when_kspace_disabled(self, monkeypatch):
        """No k-space requested -> the key is not injected, preserving the default."""
        lrk = self._captured_long_range_kwargs(monkeypatch)
        assert 'coulomb_kspace_method' not in lrk



# --- End-to-end guard on the high-level ASE k-space entry point -------------
#
# Absolute tolerance, eV, for "the two ASE entry points must agree". They build
# the same grid from the same spacing and hand it to the same solver, so the
# energies come out bitwise identical in practice; 1e-8 leaves room for XLA
# reassociation across backends while staying ~100x tighter than the smallest
# divergence the PR #22 bug produced (1.2e-6 eV for p3m, 8e-3 eV for ewald).
KSPACE_PATH_AGREEMENT_EV = 1e-8

# Absolute tolerance, eV, for "ewald, pme and p3m describe the same physics".
# Measured spread at the spacings below is ~1.2e-6 eV; the bug put ewald 8e-3 eV
# away from the other two, so 1e-4 catches it with margin on both sides.
KSPACE_METHOD_AGREEMENT_EV = 1e-4

# 12.4 A cubic box, so the long-range cutoff must stay under L/2 = 6.2 A or glp
# warns that the cutoff does not fit and the energies are meaningless.
KSPACE_LR_CUTOFF = 6.0
KSPACE_SMEARING = 2.0

# Converged spacing per method: a real-space wavelength for ewald, a mesh
# spacing for the two mesh methods, following the convention used above.
KSPACE_SPACING = {
    'ewald': KSPACE_SMEARING / 2,
    'pme': KSPACE_SMEARING / 8,
    'p3m': KSPACE_SMEARING / 8,
}

# Each energy costs a calculator build plus a JIT trace, and the two tests below
# ask for overlapping ones. Compute each (path, method) pair at most once.
_KSPACE_ENERGY_CACHE = {}


def _kspace_water_box(package_dir):
    """Periodic water box sized for KSPACE_LR_CUTOFF."""
    return read(package_dir / 'tests/test_data/water_64.xyz')


def _kspace_energy(package_dir, method, path):
    """Energy of the water box with `method`, requested via `path`.

    path='high' goes through the ASE calculator's own kspace_electrostatics /
    kspace_smearing / kspace_spacing kwargs, letting `calculate` build the grid.
    path='low' pre-builds the grid into atoms.info and names the solver with the
    lower-level coulomb_kspace_method. Both must describe the same calculation.
    """
    key = (method, path)
    if key in _KSPACE_ENERGY_CACHE:
        return _KSPACE_ENERGY_CACHE[key]

    atoms = _kspace_water_box(package_dir)
    calc_kwargs = dict(
        workdir=MODEL_DIR,
        lr_cutoff=KSPACE_LR_CUTOFF,
        dispersion_energy_cutoff_lr_damping=2.,
        from_file=True,
        dtype=np.float64,
    )

    if path == 'high':
        calc_kwargs.update(
            kspace_electrostatics=method,
            kspace_smearing=KSPACE_SMEARING,
            kspace_spacing=KSPACE_SPACING[method],
        )
    else:
        import jaxpme

        # Match calculate()'s cell convention (column vectors) so the grid is
        # the same object the high-level path would have built.
        cell = jnp.array(np.array(atoms.get_cell()), dtype=DTYPE).T
        spacing = jnp.array([KSPACE_SPACING[method]])
        if method == 'ewald':
            k_grid = jaxpme.kspace.get_kgrid_ewald(cell, lr_wavelength=spacing)
        else:
            k_grid = jaxpme.kspace.get_kgrid_mesh(cell, mesh_spacing=spacing)
        atoms.info['k_grid'] = np.array(k_grid)
        atoms.info['k_smearing'] = np.array([KSPACE_SMEARING], dtype=DTYPE)
        calc_kwargs['coulomb_kspace_method'] = method

    atoms.calc = AseCalculatorSparse.create_from_workdir(**calc_kwargs)
    energy = float(atoms.get_potential_energy())
    assert np.isfinite(energy), f"{method} via '{path}' gave a non-finite energy"

    _KSPACE_ENERGY_CACHE[key] = energy
    return energy


class TestKspaceHighLevelPathEnergies:
    """The high-level ASE k-space kwargs must produce correct energies.

    Regression test for PR #22, at the level the bug actually mattered. Every
    other k-space test in this file names the solver with the lower-level
    ``coulomb_kspace_method`` and pre-builds the grid into ``atoms.info``, so
    all of them bypassed the broken path: ``kspace_electrostatics`` chose the
    grid but never reached the solver, which silently stayed on its 'pme'
    default. TestKspaceMethodPropagation pins the kwarg plumbing against a stub;
    these two tests pin the numbers with a real model, and so also catch a grid
    and solver that disagree for some future reason other than that one kwarg.
    """

    @pytest.mark.parametrize('method', ['ewald', 'pme', 'p3m'])
    def test_high_level_matches_low_level(self, method, package_dir):
        """Both ASE entry points must agree for the same requested method."""
        high = _kspace_energy(package_dir, method, 'high')
        low = _kspace_energy(package_dir, method, 'low')

        assert abs(high - low) < KSPACE_PATH_AGREEMENT_EV, (
            f"kspace_electrostatics='{method}' gave {high!r} eV but "
            f"coulomb_kspace_method='{method}' with the same grid gave {low!r} eV "
            f"(difference {abs(high - low):.3e} eV). The high-level kwarg is not "
            f"reaching the solver, so the grid and the solver disagree."
        )

    def test_methods_agree_through_high_level_path(self, package_dir):
        """ewald, pme and p3m are three algorithms for one physical energy.

        Requested through the high-level kwargs they must land on the same
        number. This is the assertion the PR #22 bug failed outright: 'ewald'
        was 8e-3 eV away from the other two, and 'p3m' was *bitwise identical*
        to 'pme' because both were running the PME solver.
        """
        energies = {
            method: _kspace_energy(package_dir, method, 'high')
            for method in ('ewald', 'pme', 'p3m')
        }

        for a, b in (('ewald', 'pme'), ('ewald', 'p3m'), ('pme', 'p3m')):
            assert abs(energies[a] - energies[b]) < KSPACE_METHOD_AGREEMENT_EV, (
                f"{a} gave {energies[a]!r} eV and {b} gave {energies[b]!r} eV "
                f"(difference {abs(energies[a] - energies[b]):.3e} eV); these are "
                f"different algorithms for the same quantity and must agree."
            )

        assert energies['pme'] != energies['p3m'], (
            "pme and p3m returned bitwise identical energies, which means both "
            "ran the same solver -- exactly the PR #22 failure mode."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
