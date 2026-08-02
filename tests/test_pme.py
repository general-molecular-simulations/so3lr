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
    return calc


def get_package_dir():
    """Return the package directory path."""
    return pathlib.Path(__file__).parent.parent.resolve()


@pytest.fixture
def package_dir():
    """Fixture providing the package directory path."""
    return get_package_dir()


class TestPMEDirectComparison:
    """
    Test that electrostatic_energy_kspace and partial_charges from SO3LR
    can be used with jax-pme to reproduce the k-space energy.
    
    These tests extract the partial_charges from SO3LR and use them directly
    with jax-pme solvers to verify the k-space calculation is correct.
    """

    @pytest.mark.parametrize('kspace_method', ['ewald', 'pme', 'p3m'])
    def test_kspace_energy_matches_direct_calculation(self, kspace_method, package_dir):
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
        atoms = read(package_dir / 'tests/test_data/water_64.xyz') * [2, 2, 2]
        
        # Set up k-space parameters
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        positions = jnp.array(atoms.get_positions(), dtype=DTYPE)
        
        # Choose smearing based on cell size
        smearing = 4.0  # Angstrom
        
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
            lr_cutoff=12.,
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
    def test_partial_charges_are_reasonable(self, kspace_method, package_dir):
        """
        Test that extracted partial charges have reasonable properties.
        
        For neutral water, the total charge should be approximately zero.
        """
        from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

        # Create a periodic water box
        atoms = read(package_dir / 'tests/test_data/water_64.xyz') * [2, 2, 2]
        
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        smearing = 4.0
        
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
            lr_cutoff=12.,
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

        with caplog.at_level(logging.INFO, logger="SO3LR"):
            calc.calculate(atoms)

        assert 'Generating PME k-space grid from ASE calculator kwargs.' in caplog.text


class TestPMEEnergyDecomposition:
    """Test that k-space energy properly contributes to total energy."""

    @pytest.mark.parametrize('kspace_method', ['pme'])  # Start with PME only for speed
    def test_kspace_energy_contribution(self, kspace_method, package_dir):
        """
        Test that electrostatic_energy_kspace adds a sensible contribution 
        to the total electrostatic energy.
        """
        from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

        # Create a periodic water box
        atoms = read(package_dir / 'tests/test_data/water_64.xyz') * [2, 2, 2]
        
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        smearing = 4.0
        
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
            lr_cutoff=12.,
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
    def test_forces_are_finite(self, kspace_method, package_dir):
        """
        Test that forces are computed correctly with PME.
        """
        from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

        # Use a periodic water box
        atoms = read(package_dir / 'tests/test_data/water_64.xyz') * [2, 2, 2]
        
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        smearing = 4.0
        
        if kspace_method == 'ewald':
            k_spacing = jnp.array([smearing / 2])
            k_grid = get_kgrid_ewald(cell, k_spacing)
        else:
            k_spacing = jnp.array([smearing / 8])
            k_grid = get_kgrid_mesh(cell, k_spacing)
        
        atoms.info['k_grid'] = np.array(k_grid)
        atoms.info['k_smearing'] = np.array([smearing])

        calc = So3lrCalculator(
            lr_cutoff=12.,
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
    def test_stress_calculation(self, kspace_method, package_dir):
        """
        Test that stress can be calculated with PME.
        """
        from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

        atoms = read(package_dir / 'tests/test_data/water_64.xyz') * [2, 2, 2]
        
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        smearing = 4.0
        
        if kspace_method == 'ewald':
            k_spacing = jnp.array([smearing / 2])
            k_grid = get_kgrid_ewald(cell, k_spacing)
        else:
            k_spacing = jnp.array([smearing / 8])
            k_grid = get_kgrid_mesh(cell, k_spacing)
        
        atoms.info['k_grid'] = np.array(k_grid)
        atoms.info['k_smearing'] = np.array([smearing])

        calc = So3lrCalculator(
            lr_cutoff=12.,
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
    def test_jaxmd_pme_energy(self, kspace_method, package_dir):
        """
        Test PME energy calculation through JAX-MD interface.
        """
        from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

        atoms = read(package_dir / 'tests/test_data/water_64.xyz') * [2, 2, 2]
        
        cell_ase = np.array(atoms.get_cell().array, dtype=np.float64)
        smearing = 4.0
        
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
