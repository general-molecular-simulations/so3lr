"""Regression checks for center-of-mass removal and neighbor-list rollback."""
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms, units

from web import native_md


class NativeDynamicsTest(unittest.TestCase):
    def test_constrained_maxwell_distribution(self):
        numbers = (1, 1, 1, 1)
        cpu = jax.local_devices(backend="cpu")[0]
        with jax.default_device(cpu), jax.enable_x64(True):
            velocities, temperature, _, _ = native_md._thermostat(numbers, "mixed", cpu)
            sample = jax.vmap(velocities, in_axes=(None, 0, None))
            v, _ = sample(jnp.zeros((4, 3)), jax.random.split(jax.random.PRNGKey(12), 16384), 300 * units.kB)
            self.assertAlmostEqual(float(jax.vmap(temperature)(v).mean()), 300, delta=3)
            np.testing.assert_allclose(v.mean(axis=1), 0., atol=1e-14)

    def test_compacting_fragments_replay_overflow_without_losing_forces(self):
        # Initially, the two H₂ molecules are outside the short-range list.
        # Seed 17 points them at each other, exceeding the initial capacity.
        cpu = jax.local_devices(backend="cpu")[0]
        numbers = (1, 1, 1, 1)
        positions = [[0, 0, 0], [0, .75, 0], [5.1, 0, 0], [5.1, .75, 0]]
        with jax.default_device(cpu), jax.enable_x64(True):
            from so3lr import So3lrCalculator
            near, far, init, advance = native_md._dynamics_kernel(numbers, "so3lr-tauri-s", 0, "mixed", cpu)
            overflows = []

            def check_advance(*args, **kwargs):
                result = advance(*args, **kwargs)
                if jax.device_get(result[2]).any():
                    overflows.append(True)
                return result

            with patch.object(native_md, "_dynamics_kernel", return_value=(near, far, init, check_advance)):
                frames = list(native_md.run_dynamics(
                    numbers, positions, model="so3lr-tauri-s", charge=0, precision="mixed",
                    temperature=2000, timestep=.5, seed=17, steps=65, sample_stride=8))
            self.assertTrue(overflows)
            self.assertEqual([frame["step"] for frame in frames], [*range(0, 65, 8), 65])
            atoms = Atoms(numbers=numbers)
            atoms.calc = So3lrCalculator(model="so3lr-tauri-s", lr_cutoff=100., precision="mixed")
            for frame in frames:
                atoms.positions = frame["positions"]
                np.testing.assert_allclose(frame["forces"], atoms.get_forces(), atol=4e-5, rtol=2e-5)
                self.assertAlmostEqual(frame["energy"], atoms.get_potential_energy(), delta=3e-5)
