"""
Test PME/Ewald electrostatics against direct jax-pme calculations.

This test file validates that the electrostatic_energy_kspace extracted from
the SO3LR model matches direct calculations using jax-pme with the same
partial charges.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pathlib
import pytest
import logging
import sys
import types
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

# The ASE calculator against a jax-md path (the Python API, or the CLI MD setup)
# on the same model and k-space grid, in float64, eV and eV/A. Measured on
# MeluXina: <= 9e-13 eV and <= 2.5e-13 eV/A for the 192- and 1536-atom boxes, i.e.
# summation-order roundoff; these bounds leave ~4 orders of margin and still
# catch any configuration difference (a float32 k-space term, another damping).
ASE_JAXMD_ENERGY_ATOL = 1e-8
ASE_JAXMD_FORCES_ATOL = 1e-8


DummySystem = namedtuple('DummySystem', ['R', 'k_grid', 'k_smearing'])


def make_stub_ase_calculator():
    calc = object.__new__(AseCalculatorSparse)
    Calculator.__init__(calc)
    calc.calculate_fn = lambda system, neighbors: {
        'energy': np.array(0.0),
        'forces': np.zeros((len(system.R), 3)),
    }
    calc.neighbors = None
    calc.neighbor_backend = 'glp'
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


@pytest.fixture
def pme_case(package_dir, full_test):
    """(atoms, smearing, lr_cutoff) for the k-space tests below.

    Default: the 192-atom water_64 box at KSPACE_LR_CUTOFF/KSPACE_SMEARING (minimum
    image holds: 12.4 A cell, 6 A cutoff). --full: 2x2x2 (1536 atoms) at the
    paper's 12 A / 4 A, which on a 2-core CI runner blew the group's 60-minute
    budget. Every assertion here is about agreement with a direct jax-pme
    evaluation on the same charges, between entry points or with finite
    differences, none of which depends on size.
    """
    box = read(package_dir / 'tests/test_data/water_64.xyz')
    if full_test:
        return box * [2, 2, 2], 4.0, 12.
    return box, KSPACE_SMEARING, KSPACE_LR_CUTOFF


@functools.cache
def _jaxmd_potential(kspace_method, lr_cutoff):
    """So3lrPotential for the default model, built once per (method, cutoff): the
    ASE-agreement test (pme) and the shear test use the same one."""
    return So3lrPotential(dtype=DTYPE, lr_cutoff=lr_cutoff, coulomb_kspace_method=kspace_method)


@pytest.fixture(scope='module')
def ase_calculator():
    """``get(kspace_method, lr_cutoff, smearing, atoms)``: a So3lrCalculator for the
    default model with those user-level k-space settings.

    Shared within this module: TestPMEJaxMD hands it a pre-built grid in
    atoms.info (which overrides the kwargs), TestCLIMDPotential lets it build its
    own. Keyed by the structure's atom count and cell too, because a calculator
    keeps its neighbour-list allocation; reset on every use, since ASE would
    otherwise return the previous test's results for the same positions. Released
    with the module.
    """
    cache = {}

    def get(kspace_method, lr_cutoff, smearing, atoms):
        key = (kspace_method, lr_cutoff, smearing, len(atoms), tuple(atoms.cell.array.ravel()))
        if key not in cache:
            cache[key] = So3lrCalculator(
                lr_cutoff=lr_cutoff,
                dtype=np.float64,
                kspace_electrostatics=kspace_method,
                kspace_smearing=smearing,
                kspace_spacing=_converged_spacing(kspace_method, smearing),
            )
        cache[key].reset()
        return cache[key]

    return get


def _converged_spacing(method, smearing):
    """The converged spacing used throughout this file: smearing/2 for ewald
    (a real-space wavelength), /8 for the meshes."""
    return smearing / 2 if method == 'ewald' else smearing / 8


def _kgrid(cell, method, smearing):
    """(k_grid, k_smearing) from the production builder, at the converged spacing."""
    from so3lr.mlff.utils.kspace_utils import setup_kspace_grid

    return setup_kspace_grid(method, smearing, _converged_spacing(method, smearing), cell)


def _cli_md_settings(kspace_method, smearing, lr_cutoff):
    """The settings dict `so3lr --settings` hands perform_md for a float64 k-space
    run: YAML_DEFAULTS plus the keys a user would set."""
    from so3lr.cli.so3lr_cli import YAML_DEFAULTS

    settings = dict(YAML_DEFAULTS)
    settings.update(
        precision='float64',
        lr_cutoff=lr_cutoff,
        kspace_electrostatics=kspace_method,
        kspace_smearing=smearing,
        kspace_spacing=_converged_spacing(kspace_method, smearing),
    )
    return settings


def _cli_md_energy_and_forces(atoms, settings):
    """(energy, Cartesian forces) from the potential `so3lr nvt/npt/nve`
    integrates.

    Repeats perform_md's own setup calls, in its order and with its .get defaults,
    as so3lr/flash/verify.py does for its neighbour-list reference: cell, wrap,
    fractional box, load_model, process_model, then the k-grids from the jax-md
    box. Running perform_md itself would integrate and write a trajectory, so a
    change to those calls there must be repeated here. Left out are
    setup_jax_globals (process-wide: this module enables x64 and conftest sets the
    matmul precision) and everything after the neighbour lists.
    """
    from so3lr import kspace_defaults as KS
    from so3lr.cli import so3lr_md
    from so3lr.jaxmd_utils import nbrs_cutoff
    from so3lr.model_registry import DEFAULT_MODEL

    s = settings
    precision = s.get('precision')
    dtype = jnp.float64 if precision == 'float64' else jnp.float32
    lr_cutoff = s.get('lr_cutoff')
    kspace_electrostatics = s.get('kspace_electrostatics', None)
    kspace_smearing = s.get('kspace_smearing', KS.DEFAULT_KSPACE_SMEARING)
    kspace_spacing = s.get('kspace_spacing', KS.DEFAULT_KSPACE_SPACING)
    kspace_dispersion = s.get('kspace_dispersion', None)
    observables = s.get('observables', [])

    atoms = atoms.copy()
    cell = so3lr_md.cell_from_atoms_boundary_conditions(
        atoms, lr_cutoff, kspace_electrostatics, kspace_smearing)
    atoms.wrap()
    geometry = so3lr_md.atoms_to_jnp(atoms, precision=precision)
    # 'periodic' is what perform_md picks for a cell when PySAGES is off.
    position, box, displacement, _, fractional_coordinates = so3lr_md.handle_box(
        'periodic', geometry['positions'], cell, precision=dtype)

    model_path = s.get('model_path')
    potential = so3lr_md.load_model(
        model_path if model_path is not None else DEFAULT_MODEL,
        precision=dtype,
        lr_cutoff=lr_cutoff,
        dispersion_damping=s.get('dispersion_damping'),
        kspace_electrostatics=kspace_electrostatics,
        kspace_interp_nodes=s.get('kspace_interp_nodes', KS.DEFAULT_KSPACE_INTERP_NODES),
        kspace_dispersion=kspace_dispersion,
        kspace_dispersion_interp_nodes=s.get(
            'kspace_dispersion_interp_nodes', KS.DEFAULT_KSPACE_DISPERSION_INTERP_NODES),
        charge_embed_total_charge=s.get('charge_embed_total_charge', None),
        charge_equilibration_total_charge=s.get('charge_equilibration_total_charge', None),
        **so3lr_md.intermediate_quantity_kwargs(observables),
    )
    neighbor_fn, neighbor_fn_lr, energy_or_obs_fn = so3lr_md.process_model(
        potential=potential,
        species=geometry['species'],
        displacement=displacement,
        box=box,
        total_charge=s.get('total_charge'),
        num_unpaired_electrons=s.get('num_unpaired_electrons', 0),
        precision=precision,  # the string, as perform_md passes it
        fractional_coordinates=fractional_coordinates,
        buffer_size_multiplier_lr=s.get('buffer_size_multiplier_lr'),
        buffer_size_multiplier_sr=s.get('buffer_size_multiplier_sr'),
        minimum_cell_size_multiplier=s.get('minimum_cell_size_multiplier', 1.0),
        has_aux=len(observables) > 0,
    )
    so3lr_md.check_box_for_neighbour_lists(box, nbrs_cutoff(potential), 'MD box')

    k_grid, k_smearing = so3lr_md.setup_kspace_grid(
        kspace_electrostatics=kspace_electrostatics,
        kspace_spacing=kspace_spacing,
        kspace_smearing=kspace_smearing,
        box=box,
    )
    k_grid_disp, k_smearing_disp = so3lr_md.setup_kspace_grid(
        kspace_electrostatics=kspace_dispersion,
        kspace_spacing=s.get('kspace_dispersion_spacing', KS.DEFAULT_KSPACE_DISPERSION_SPACING),
        kspace_smearing=s.get('kspace_dispersion_smearing', KS.DEFAULT_KSPACE_DISPERSION_SMEARING),
        box=box,
    )
    energy_or_obs_fn = functools.partial(
        energy_or_obs_fn, k_smearing=k_smearing, k_smearing_disp=k_smearing_disp)
    energy_fn = functools.partial(energy_or_obs_fn, has_aux=False)

    nbrs = neighbor_fn.allocate(position, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(position, box=box)

    # The integrators take the force as -grad in these fractional positions
    # (jax_md.quantity.force). jax_md.space.transform's custom JVP passes the
    # gradient through unscaled, so it is already the Cartesian force; chaining
    # through R / box would divide it by the box length again.
    energy, grad = jax.jit(jax.value_and_grad(energy_fn))(
        position, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box,
        k_grid=k_grid, k_grid_disp=k_grid_disp)
    return float(energy), -np.asarray(grad)


class TestPMEDirectComparison:
    """
    Test that electrostatic_energy_kspace and partial_charges from SO3LR
    can be used with jax-pme solvers to reproduce the k-space energy.
    """

    @pytest.mark.parametrize('kspace_method', ['ewald', 'pme', 'p3m'])
    def test_kspace_energy_matches_direct_calculation(self, kspace_method, package_dir, pme_case):
        """
        Test that electrostatic_energy_kspace from SO3LR matches direct jax-pme calculation.

        This test:
        1. Runs SO3LR to get partial_charges and electrostatic_energy_kspace
        2. Uses the same partial charges with jax-pme directly
        3. Compares the k-space energies
        """
        from jaxpme.solvers import ewald, pme, p3m
        from jaxpme.potentials import potential as get_potential
        from jaxpme.kspace import generate_kvectors, get_reciprocal

        # Create a periodic water box - repeat to ensure enough for 12 Å cutoff
        atoms, smearing, lr_cutoff = pme_case

        # Set up k-space parameters
        cell = jnp.array(atoms.get_cell().array, dtype=DTYPE)
        positions = jnp.array(atoms.get_positions(), dtype=DTYPE)

        k_grid, k_smearing = _kgrid(cell, kspace_method, smearing)

        # Store k-space info in atoms
        atoms.info['k_grid'] = np.array(k_grid)
        atoms.info['k_smearing'] = np.array(k_smearing)

        # Create SO3LR calculator with observables
        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=lr_cutoff,
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

        # Neutral water: one charge per atom, summing to zero. The charges come
        # from the model before any solver runs.
        assert partial_charges.shape[0] == len(atoms), \
            f"Expected {len(atoms)} charges, got {partial_charges.shape[0]}"
        npt.assert_allclose(
            float(jnp.sum(partial_charges)),
            0.0,
            atol=1e-5,
            err_msg="Total charge should be ~0 for neutral water"
        )

        # Now compute k-space energy directly with jax-pme using the same charges
        volume = jnp.abs(jnp.linalg.det(cell))
        reciprocal_cell = get_reciprocal(cell)
        pbc = jnp.array([True, True, True])

        if kspace_method == 'ewald':
            solver = ewald(get_potential())
            kvectors = generate_kvectors(reciprocal_cell, k_grid.shape, dtype=DTYPE, for_ewald=True)
            potentials = solver.kspace(k_smearing, partial_charges, kvectors, positions, volume, cell, pbc)
        else:
            solver = {'pme': pme, 'p3m': p3m}[kspace_method](get_potential(), interpolation_nodes=4)
            kvectors = generate_kvectors(reciprocal_cell, k_grid.shape, dtype=DTYPE, for_ewald=False)
            potentials = solver.kspace(k_smearing, partial_charges, reciprocal_cell, k_grid, kvectors,
                                       positions, volume, cell, pbc)

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


class TestASECalculatorKSpaceReporting:
    @pytest.fixture(autouse=True)
    def _so3lr_logs_reach_caplog(self, monkeypatch):
        # setup_logger() (so3lr_md.py) sets propagate=False on "SO3LR" while
        # caplog listens on the root logger: after any test in this process has
        # run the CLI/eval path, these records would otherwise never arrive.
        monkeypatch.setattr(logging.getLogger("SO3LR"), "propagate", True)

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


class TestPMEJaxMD:
    """Test PME with JAX-MD backend."""

    @pytest.mark.parametrize('kspace_method', ['ewald', 'pme', 'p3m'])
    def test_jaxmd_matches_ase_calculator(self, kspace_method, package_dir, pme_case,
                                          ase_calculator):
        """The Python-API jax-md potential (So3lrPotential + to_jax_md) and the ASE
        calculator agree.

        Same default model, k-space method and grid through both entry points:
        energy and Cartesian forces must match, though each builds its own
        neighbour lists and long-range inputs. The CLI MD path builds its
        long-range settings separately again (so3lr_md.load_model);
        TestCLIMDPotential compares that one.
        """
        atoms, smearing, lr_cutoff = pme_case

        cell_ase = np.array(atoms.get_cell().array, dtype=np.float64)
        k_grid, k_smearing = _kgrid(jnp.array(cell_ase), kspace_method, smearing)

        # ASE calculator, grid handed over in atoms.info
        atoms_ase = atoms.copy()
        atoms_ase.info['k_grid'] = np.array(k_grid)
        atoms_ase.info['k_smearing'] = np.array(k_smearing)
        atoms_ase.calc = ase_calculator(kspace_method, lr_cutoff, smearing, atoms_ase)
        ase_energy = atoms_ase.get_potential_energy()
        ase_forces = atoms_ase.get_forces()

        # jax-md
        species = jnp.array(atoms.get_atomic_numbers())
        masses = jnp.array(atoms.get_masses())
        box = jnp.diag(jnp.array(cell_ase))

        displacement, shift = jax_md.space.periodic_general(
            box=box, fractional_coordinates=True
        )

        positions = jnp.array(atoms.get_positions(), dtype=DTYPE)

        neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
            # Same cutoff switch as the ASE cases: at the default 12 A the
            # 192-atom box (12.4 A) would violate minimum image.
            potential=_jaxmd_potential(kspace_method, lr_cutoff),
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

        positions_frac = positions / box
        nbrs = neighbor_fn.allocate(positions_frac, box=box)
        nbrs_lr = neighbor_fn_lr.allocate(positions_frac, box=box)

        # Differentiate in the fractional positions, as the integrators do:
        # jax_md.space.transform's custom JVP passes the gradient through
        # unscaled, so this is already the Cartesian force. Differentiating
        # through `R / box` instead would divide it by the box length again.
        def energy_of(R_frac):
            return energy_fn(
                R_frac,
                nbrs.idx,
                nbrs_lr.idx,
                box=box,
                species=species,
                masses=masses,
                k_grid=k_grid,
                k_smearing=k_smearing,
                has_aux=False,
            )

        energy, grad = jax.jit(jax.value_and_grad(energy_of))(positions_frac)
        forces = -np.asarray(grad)

        print(f"\n{kspace_method.upper()} ASE vs JAX-MD:")
        print(f"  ASE energy: {ase_energy:.6f} eV, JAX-MD energy: {float(energy):.6f} eV")
        print(f"  Max force difference: {np.max(np.abs(forces - ase_forces)):.2e} eV/A")

        assert np.isfinite(float(energy)), "Energy is not finite"
        npt.assert_allclose(float(energy), ase_energy, rtol=0, atol=ASE_JAXMD_ENERGY_ATOL)
        npt.assert_allclose(forces, ase_forces, rtol=0, atol=ASE_JAXMD_FORCES_ATOL)

    def test_jaxmd_pme_shear_derivative_matches_finite_difference(self, pme_case):
        """The strain perturbation used for the NPT-flex stress deforms the lattice
        the model's PME sees in the right direction.

        jax_md.quantity.stress differentiates the energy through
        ``perturbation = I + eps``. The box holds the lattice as columns, so it must
        deform as ``pert @ box``; applying the perturbation to its rows gives the
        transposed deformation once the potential hands PME the lattice as rows,
        and the off-diagonal k-space stress comes out scaled by the wrong box length.
        """
        atoms, smearing, lr_cutoff = pme_case
        # Stretch one axis so the box is not cubic: a cubic box hides the error.
        atoms = atoms.copy()
        atoms.set_cell(np.diag(np.diag(atoms.get_cell().array) * [1.0, 1.25, 1.0]),
                       scale_atoms=True)
        cell = np.array(atoms.get_cell().array, dtype=np.float64)
        box = jnp.array(cell.T)
        species = jnp.array(atoms.get_atomic_numbers())
        masses = jnp.array(atoms.get_masses())
        k_grid, k_smearing = _kgrid(jnp.array(cell), 'pme', smearing)

        displacement, shift = jax_md.space.periodic_general(box=box, fractional_coordinates=True)
        frac = jnp.array(atoms.get_scaled_positions(), dtype=DTYPE)
        neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
            potential=_jaxmd_potential('pme', lr_cutoff),
            displacement_or_metric=displacement, shift_fn=shift, box_size=box,
            species=species, capacity_multiplier=1.5, buffer_size_multiplier_sr=1.5,
            buffer_size_multiplier_lr=1.5, minimum_cell_size_multiplier_sr=1.0,
            disable_cell_list=True, fractional_coordinates=True)
        nbrs = neighbor_fn.allocate(frac, box=box)
        nbrs_lr = neighbor_fn_lr.allocate(frac, box=box)

        def energy(bx, **kw):
            return energy_fn(frac, nbrs.idx, nbrs_lr.idx, box=bx, species=species,
                             masses=masses, k_grid=k_grid, k_smearing=k_smearing,
                             has_aux=False, **kw)

        # d E / d eps_xy through the perturbation path (what quantity.stress uses)
        e01 = jnp.zeros((3, 3), dtype=DTYPE).at[0, 1].set(1.0)
        analytic = jax.grad(lambda t: energy(box, perturbation=jnp.eye(3, dtype=DTYPE) + t * e01))(0.0)
        # ... and by finite differences of the deformed box itself
        h = 1e-5
        numeric = (energy((jnp.eye(3) + h * e01) @ box) - energy((jnp.eye(3) - h * e01) @ box)) / (2 * h)
        npt.assert_allclose(float(analytic), float(numeric), rtol=1e-4, atol=1e-6)


class TestCLIMDPotential:
    """The potential the CLI MD path integrates agrees with So3lrCalculator.

    so3lr/cli builds its own long_range_kwargs and the ASE calculator another, and
    the two have drifted before (AGENTS.md; PR #22: kspace_electrostatics chose the
    k-space grid but not the solver, on the ASE side only). TestPMEJaxMD covers the
    Python-API jax-md path and hands both sides one pre-built grid. Here each side
    gets only the user-level settings (method, smearing, spacing, cutoffs) and
    builds its own long-range kwargs, solver and k-grid from them: the CLI side
    through so3lr_md.load_model as perform_md does, the ASE side through the
    calculator's kspace_electrostatics/kspace_smearing/kspace_spacing kwargs.
    """

    @pytest.mark.parametrize('kspace_method', ['ewald', 'pme'])
    def test_cli_md_potential_matches_ase_calculator(self, kspace_method, pme_case, monkeypatch,
                                                     ase_calculator):
        import jaxpme.kspace

        atoms, smearing, lr_cutoff = pme_case
        settings = _cli_md_settings(kspace_method, smearing, lr_cutoff)

        # Record every grid the method's builder makes: first the CLI side's, then
        # the ASE calculator's (built in calculate(), not stored anywhere). Two
        # converged but different grids could agree to the energy tolerance.
        shapes = []
        name = 'get_kgrid_ewald' if kspace_method == 'ewald' else 'get_kgrid_mesh'
        build = getattr(jaxpme.kspace, name)

        def spy(*args, **kwargs):
            grid = build(*args, **kwargs)
            shapes.append(tuple(grid.shape))
            return grid
        monkeypatch.setattr(jaxpme.kspace, name, spy)

        cli_energy, cli_forces = _cli_md_energy_and_forces(atoms, settings)

        # Both sides run the default model (the CLI resolves an unset model_path to
        # it) with the dispersion damping the calculator also defaults to.
        assert settings['dispersion_damping'] == 2.0
        atoms_ase = atoms.copy()
        atoms_ase.calc = ase_calculator(kspace_method, lr_cutoff, smearing, atoms_ase)
        ase_energy = atoms_ase.get_potential_energy()
        ase_forces = atoms_ase.get_forces()

        grids = f"k-grids (CLI, ASE): {shapes}"
        print(f"\n{kspace_method.upper()} CLI MD potential vs ASE calculator:")
        print(f"  ASE energy: {ase_energy:.6f} eV, CLI energy: {cli_energy:.6f} eV")
        print(f"  Max force difference: {np.max(np.abs(cli_forces - ase_forces)):.2e} eV/A")
        print(f"  {grids}")

        npt.assert_allclose(cli_energy, ase_energy, rtol=0, atol=ASE_JAXMD_ENERGY_ATOL, err_msg=grids)
        npt.assert_allclose(cli_forces, ase_forces, rtol=0, atol=ASE_JAXMD_FORCES_ATOL, err_msg=grids)
        assert len(shapes) == 2 and shapes[0] == shapes[1], (
            f"{grids}: from the same kspace_electrostatics/smearing/spacing the two "
            f"paths did not build one grid each, of one shape")


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
KSPACE_SPACING = {method: _converged_spacing(method, KSPACE_SMEARING)
                  for method in ('ewald', 'pme', 'p3m')}

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
