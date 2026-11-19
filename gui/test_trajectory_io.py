"""Regression checks for recorded kinetic observables and Tauri L charges."""
import unittest

from trajectory_io import analyze_file


def two_frames(first, second, model="so3lr-tauri-l"):
    xyz = ""
    for step, fields in enumerate((first, second)):
        xyz += (f"2\nProperties=species:S:1:pos:R:3:partial_charges:R:1 time_fs={step} "
                f"model={model} charge_source=so3lr {fields}\n"
                "H 0 0 0 -0.1\nF 0 0 0.917 0.1\n")
    return analyze_file({"xyz": xyz})["result"]


class TrajectoryObservablesTest(unittest.TestCase):
    def test_import_recorded_kinetics_and_l_charges(self):
        result = two_frames("kinetic_energy_eV=0.13 temperature_K=196 target_temperature_K=300",
                            "kinetic_energy_eV=0.17 temperature_K=302 target_temperature_K=300")
        self.assertEqual(result["kinetic_energies"], [0.13, 0.17])
        self.assertEqual(result["temperatures"], [196, 302])
        self.assertEqual(result["md"]["temperature"], 300)
        self.assertEqual(result["model"], "so3lr-tauri-l")
        self.assertEqual(result["partial_charges"], [[-0.1, 0.1], [-0.1, 0.1]])

    def test_versioned_model_keeps_its_identifier_and_charges(self):
        result = two_frames("", "", model="so3lr-tauri-l-2606")
        self.assertEqual(result["model"], "so3lr-tauri-l-2606")
        self.assertEqual(result["partial_charges"], [[-0.1, 0.1]] * 2)

    def test_force_backend_round_trip_keeps_one_consistent_recorded_name(self):
        custom = 'force_backend="SO3LR JAX-MD"'
        self.assertEqual(two_frames(custom, custom)["md"]["force_backend"], "SO3LR JAX-MD")
        mixed = two_frames(custom, 'force_backend="ASE"')
        self.assertNotIn("force_backend", mixed.get("md", {}))

    def test_temperature_without_known_degrees_of_freedom(self):
        result = two_frames("temperature_K=100", "temperature_K=120")
        self.assertIsNone(result["kinetic_energies"])
        self.assertEqual(result["temperatures"], [100, 120])

    def test_incomplete_or_unphysical_kinetic_data(self):
        for fields in ("", "kinetic_energy_eV=nan", "kinetic_energy_eV=-1"):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                two_frames("kinetic_energy_eV=0.17", fields)


class PeriodicTrajectoryTest(unittest.TestCase):
    def test_optimizer_import_preserves_row_lattices_steps_and_energies(self):
        xyz = ""
        for index, a in enumerate((4, 5)):
            xyz += (f'2\nLattice="{a} 0 0 1 5 0 0.3 0.4 6" pbc="T T T" '
                    f'Properties=species:S:1:pos:R:3:partial_charges:R:1 step={2 * index} '
                    f'task=relax optimizer=BFGS energy={-8 - index} model=so3lr-tauri-l '
                    'charge_source=so3lr dipole_debye="9 9 9"\nNa 0 0 0 0.7\nCl 2 1 1 -0.7\n')
        data = analyze_file({"xyz": xyz, "spacing_fs": 10})
        structure, result = data["structure"], data["result"]
        self.assertEqual(structure["pbc"], [True, True, True])
        self.assertEqual(structure["cell"], [[4, 0, 0], [1, 5, 0], [.3, .4, 6]])
        self.assertEqual(result["cells"][0], structure["cell"])
        self.assertEqual(result["cells"][1][0][0], 5)
        self.assertAlmostEqual(result["analysis"]["cell_volumes"][0], 120)
        self.assertAlmostEqual(result["analysis"]["cell_volumes"][1], 150)
        self.assertEqual(result["frame_steps"], [0, 2])
        self.assertEqual(result["energies"], [-8, -9])
        self.assertEqual(result["task"], "relax")
        self.assertEqual(result["optimizer"], "BFGS")
        self.assertIsNone(result["times_fs"])
        self.assertIsNone(result["analysis"]["rmsd"])
        self.assertIsNone(result["analysis"]["peptide"])
        self.assertIsNone(result["dipoles_debye"])
        self.assertEqual(result["partial_charges"], [[.7, -.7], [.7, -.7]])

    def test_reject_incomplete_or_changing_periodic_boundaries(self):
        for field in ['pbc="T F T"', 'pbc="F F F"', 'pbc="T T T" Lattice="4 0 0 0 0 0 0 0 4"']:
            with self.subTest(field=field), self.assertRaises(ValueError):
                two_frames('pbc="T T T" Lattice="4 0 0 0 4 0 0 0 4"', field)

    def test_archive_thermostat_and_clock_roundtrip(self):
        common = ('md_engine=jax integrator="JAX-MD Nosé–Hoover chain" ensemble=NVT '
                  'thermostat="Nosé–Hoover chain" timestep_fs=0.5 seed=43 '
                  'target_temperature_K=300 temperature_dof=486 chain_length=3')
        xyz = ''.join(f'2\nProperties=species:S:1:pos:R:3 time_fs={time} '
                      f'step={2 * time} {common}\nH 0 0 0\nH 0.7 0 0\n'
                      for time in (500, 50000))
        result = analyze_file({"xyz": xyz, "spacing_fs": 1})["result"]
        self.assertEqual(result["md"], dict(engine="jax", integrator="JAX-MD Nosé–Hoover chain",
                                            ensemble="NVT", thermostat="Nosé–Hoover chain",
                                            timestep=0.5, seed=43, temperature=300,
                                            temperature_dof=486, chain_length=3))
        self.assertEqual(result["times_fs"], [500, 50000])
        self.assertEqual(result["frame_steps"], [1000, 100000])

    def test_bulk_box_atom_limit_is_independent_of_isolated_engine(self):
        rows = "H 0 0 0\n" * 1000
        xyz = ('1000\nLattice="30 0 0 0 30 0 0 0 30" pbc="T T T"\n' + rows) * 2
        data = analyze_file({"xyz": xyz})
        self.assertEqual(len(data["structure"]["elements"]), 1000)
        self.assertIsNone(data["result"]["times_fs"])
        with self.assertRaisesRegex(ValueError, "2–256"):
            analyze_file({"xyz": xyz.replace("T T T", "F F F")})
