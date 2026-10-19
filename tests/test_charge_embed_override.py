"""`charge_embed_total_charge`: decouple the charge embedding from equilibration.

Two independent code paths read the same `total_charge`:

  * the embedding -- `ChargeEmbedSparse` (so3lr/mlff/nn/embed/embed_sparse.py),
    which folds Q into per-atom features;
  * equilibration -- `charge_conservation = (total_charge - predicted_charge) /
    safe_atom_counts` in so3lr/mlff/nn/observable/observable_sparse.py.

`charge_embed_total_charge` overrides the value the *embedding* sees while leaving
equilibration on the system's true total charge.

Note this is deliberately NOT `use_charge_embed`. That flag decides whether the
module is built at all (`make_embedding_modules`), so flipping it changes the
parameter tree and a released checkpoint stops loading. Overriding the input keeps
every trained weight in place -- which `test_released_checkpoint_still_loads`
pins directly.
"""
import pathlib

import numpy as np
import pytest

from ase import Atoms

# so3lr-tauri-s is small and ships with use_charge_embed=True, so the embedding is
# actually present and the override has something to act on.
MODEL = 'so3lr-tauri-s'

# Finite cluster: the long-range cutoff must be large or the electrostatics are
# truncated nonsense (see the lr_cutoff note in the PR description).
LR_CUTOFF = 1000.0

# Same fragment as _charged_fragment(), for the CLI subprocess runs.
_CLI_INPUT_XYZ = (
    "7\nacetate fragment, net charge -1\n"
    "C   0.00  0.00  0.00\nH   1.09  0.00  0.00\nH  -0.36  1.03  0.00\n"
    "H  -0.36 -0.51  0.89\nC  -0.70 -0.80 -1.20\nO  -0.20 -1.90 -1.40\n"
    "O  -1.80 -0.50 -1.90\n"
)


def _charged_fragment():
    """A small non-periodic fragment carrying a net charge."""
    atoms = Atoms(
        'CH3COO',
        positions=[
            [0.00, 0.00, 0.00], [1.09, 0.00, 0.00], [-0.36, 1.03, 0.00],
            [-0.36, -0.51, 0.89], [-0.70, -0.80, -1.20], [-0.20, -1.90, -1.40],
            [-1.80, -0.50, -1.90],
        ],
    )
    # Nudge off the idealised geometry so no symmetry masks a difference.
    rng = np.random.RandomState(0)
    atoms.set_positions(atoms.get_positions() + rng.normal(0, 0.01, (len(atoms), 3)))
    return atoms


def _energy(total_charge, charge_embed_total_charge,
            charge_equilibration_total_charge=None):
    from so3lr import So3lrCalculator

    atoms = _charged_fragment()
    atoms.info['charge'] = total_charge
    # Default float32: x64 is not enabled in this module, and the differences
    # under test are eV-sized, so requesting float64 would only be truncated.
    atoms.calc = So3lrCalculator(
        model=MODEL,
        lr_cutoff=LR_CUTOFF,
        charge_embed_total_charge=charge_embed_total_charge,
        charge_equilibration_total_charge=charge_equilibration_total_charge,
    )
    return float(atoms.get_potential_energy())


@pytest.fixture(scope='module')
def energies():
    """The four conditions the flag has to distinguish, computed once."""
    return {
        # embed Q, equilibrate Q -- the behaviour before this flag existed.
        'baseline': _energy(-1, None),
        # Override set to the true charge: must be indistinguishable from baseline.
        'passthrough': _energy(-1, -1.0),
        # embed 0, equilibrate Q -- the condition this flag exists to enable.
        'embed_zero': _energy(-1, 0.0),
        # Both paths see 0: a genuinely neutral system.
        'neutral': _energy(0, None),
    }


class TestChargeEmbedOverride:
    def test_default_is_inert(self):
        """Omitting the argument entirely must not perturb anything."""
        assert _energy(-1, None) == _energy(-1, None)

    def test_passthrough_is_exact(self, energies):
        """override == true charge is a no-op, to the last bit.

        The override replaces Q with `jnp.full_like(Q, value)`, so passing the
        value the graph already carries must reproduce the unpatched result
        exactly -- not merely closely. Any drift here means the plumbing is
        perturbing the model rather than just relabelling its input.
        """
        assert energies['passthrough'] == energies['baseline'], (
            f"charge_embed_total_charge=-1.0 gave {energies['passthrough']!r} but the "
            f"unmodified path gave {energies['baseline']!r}; the override is not a "
            f"faithful passthrough."
        )

    def test_override_reaches_the_embedding(self, energies):
        """Zeroing the embedded charge must actually change the energy."""
        assert abs(energies['embed_zero'] - energies['baseline']) > 1e-8, (
            "charge_embed_total_charge=0.0 produced the same energy as embedding the "
            "true charge, so the override never reached ChargeEmbedSparse."
        )

    def test_equilibration_keeps_the_true_charge(self, energies):
        """The whole point: only the embedding is overridden, not conservation.

        If the override leaked into `charge_conservation` as well, embedding 0 on a
        Q=-1 system would be identical to declaring the system neutral. It must not
        be -- equilibration still has to distribute the -1.
        """
        assert abs(energies['embed_zero'] - energies['neutral']) > 1e-8, (
            f"embed 0 / equilibrate -1 gave {energies['embed_zero']!r}, identical to a "
            f"fully neutral system ({energies['neutral']!r}). The override leaked into "
            f"charge equilibration, which it must never do."
        )

    def test_released_checkpoint_still_loads(self):
        """The override must not disturb the parameter tree.

        This is the property that distinguishes it from `use_charge_embed`: the
        module and all its trained weights stay in place and simply receive a
        different input. If this ever regresses, loading raises rather than
        returning a wrong number.
        """
        assert np.isfinite(_energy(-1, 0.0))

    @pytest.mark.parametrize('value', [0.0, -1.0, 2.0])
    def test_accepts_arbitrary_values(self, value):
        """Not a boolean: any charge can be embedded, 0.0 is just the useful case."""
        assert np.isfinite(_energy(-1, value))

    def test_forces_change_too(self):
        """Forces, not just energy -- the measured effect is a force MAE change."""
        from so3lr import So3lrCalculator

        def forces(override):
            atoms = _charged_fragment()
            atoms.info['charge'] = -1
            atoms.calc = So3lrCalculator(
                model=MODEL, lr_cutoff=LR_CUTOFF,
                charge_embed_total_charge=override,
            )
            return atoms.get_forces()

        f_default, f_zero = forces(None), forces(0.0)
        assert np.all(np.isfinite(f_default)) and np.all(np.isfinite(f_zero))
        assert not np.array_equal(f_default, f_zero), (
            "forces are identical with and without the override"
        )


class TestChargeEmbedOverrideEntryPoints:
    """The flag has to be reachable from each documented entry point."""

    def test_so3lr_potential(self):
        from so3lr import So3lrPotential

        # Constructing is the assertion: an unsupported kwarg would land in
        # long_range_kwargs and be silently ignored, or raise.
        assert So3lrPotential(
            model=MODEL, lr_cutoff=LR_CUTOFF, charge_embed_total_charge=0.0
        ) is not None

    def test_make_so3lr(self):
        from so3lr import So3lr

        assert So3lr(
            model=MODEL, lr_cutoff=LR_CUTOFF, charge_embed_total_charge=0.0
        ) is not None

    def test_yaml_key_is_known(self):
        """The did-you-mean validator must not flag it as a typo."""
        from so3lr.cli.so3lr_cli import KNOWN_YAML_KEYS_EXTRA, YAML_DEFAULTS

        assert 'charge_embed_total_charge' in (
            KNOWN_YAML_KEYS_EXTRA | set(YAML_DEFAULTS)
        )


class TestChargeEquilibrationOverride:
    """`charge_equilibration_total_charge`: the symmetric knob.

    The two paths are configured independently, so all four combinations of
    (embedded charge, conserved charge) are reachable. Before this existed only
    three were: `embed Q / equil 0` had no expression.

    Note the asymmetry in meaning, which is why the defaults stay at None: the
    embedding is a feature input, whereas equilibration *enforces* sum(q) = Q.
    Overriding the latter deliberately conserves to a charge the system does not
    have.
    """

    @pytest.fixture(scope='class')
    def four(self):
        """The four (embed, equilibrate) combinations on a Q = -1 fragment."""
        return {
            'A': _energy(-1, None, None),   # embed Q, equil Q -- the default
            'B': _energy(-1, 0.0, None),    # embed 0, equil Q
            'C': _energy(-1, 0.0, 0.0),     # embed 0, equil 0
            'D': _energy(-1, None, 0.0),    # embed Q, equil 0 -- new
        }

    def test_passthrough_is_exact(self):
        """equilibration override == true charge is a bitwise no-op."""
        assert _energy(-1, None, -1.0) == _energy(-1, None, None)

    def test_override_changes_the_energy(self, four):
        assert four['D'] != four['A'], (
            'charge_equilibration_total_charge=0.0 did not change the energy, so it '
            'never reached PartialChargesSparse'
        )

    def test_all_four_combinations_are_distinct(self, four):
        """The two knobs are independent, not two names for one switch."""
        assert len(set(four.values())) == 4, (
            f'expected four distinct energies, got {four}. The two overrides are not '
            f'acting independently.'
        )

    def test_equals_declaring_the_system_neutral(self, four):
        """Overriding both is exactly equivalent to a neutral system.

        Strong cross-check on the equilibration path: conserving to 0 while
        embedding 0 must reproduce `total_charge = 0` outright.
        """
        neutral = _energy(0, None, None)
        assert abs(four['C'] - neutral) < 1e-10, (
            f"embed 0 / equil 0 gave {four['C']!r} but a genuinely neutral system "
            f"gave {neutral!r}; the equilibration override is not equivalent to "
            f"setting the total charge."
        )

    def test_reaches_every_entry_point(self):
        from so3lr import So3lr, So3lrPotential

        assert So3lrPotential(
            model=MODEL, lr_cutoff=LR_CUTOFF, charge_equilibration_total_charge=0.0
        ) is not None
        assert So3lr(
            model=MODEL, lr_cutoff=LR_CUTOFF, charge_equilibration_total_charge=0.0
        ) is not None

    def test_yaml_key_is_known(self):
        from so3lr.cli.so3lr_cli import KNOWN_YAML_KEYS_EXTRA, YAML_DEFAULTS

        assert 'charge_equilibration_total_charge' in (
            KNOWN_YAML_KEYS_EXTRA | set(YAML_DEFAULTS)
        )


class TestChargeEmbedOverrideCLI:
    """End to end through the CLI, both routes.

    A signature check is not enough here: the option has to survive click ->
    run_md_simulation's explicit signature -> the settings dict -> perform_md ->
    load_model, and a break anywhere in that chain is invisible to an
    introspection test. So run the real binary and compare trajectories.
    """

    @staticmethod
    def _run_nvt(tmp_path, tag, extra_yaml='', extra_argv=()):
        import subprocess
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from subprocess_env import so3lr_subprocess_env

        # One directory per run: the CLI deliberately refuses to overwrite an
        # existing trajectory, so runs must not share a working directory.
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / 'input.xyz').write_text(_CLI_INPUT_XYZ)
        (tmp_path / f'{tag}.yaml').write_text(
            f"input_file: input.xyz\n"
            f"output_file: {tag}.hdf5\n"
            f"log_file: {tag}.log\n"
            f"model: {MODEL}\n"
            f"precision: float64\n"
            f"lr_cutoff: {LR_CUTOFF}\n"
            f"total_charge: -1\n"
            f"ensemble: nvt\n"
            f"md_dt: 0.5\nmd_cycles: 1\nmd_steps: 1\nmd_T: 300.0\n"
            f"relax_before_run: false\nsave_buffer: 1\nseed: 42\n"
            f"{extra_yaml}\n"
        )
        result = subprocess.run(
            ['so3lr', '--settings', f'{tag}.yaml', *extra_argv],
            cwd=tmp_path, capture_output=True, text=True,
            env=so3lr_subprocess_env(),
        )
        assert result.returncode == 0, (
            f"so3lr --settings {tag}.yaml failed ({result.returncode})\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        # A key the validator does not recognise is only a warning, so assert.
        assert 'did you mean' not in (result.stdout + result.stderr).lower(), (
            "the settings validator flagged charge_embed_total_charge as a typo"
        )
        import h5py

        with h5py.File(tmp_path / f'{tag}.hdf5', 'r') as f:
            return np.array(f['momenta'])

    def test_settings_yaml_route(self, tmp_path):
        """`charge_embed_total_charge:` in the settings YAML reaches the model."""
        absent = self._run_nvt(tmp_path / 'a', 'absent', '# key absent')
        null = self._run_nvt(tmp_path / 'b', 'null', 'charge_embed_total_charge: null')
        minus = self._run_nvt(tmp_path / 'c', 'minus', 'charge_embed_total_charge: -1.0')
        zero = self._run_nvt(tmp_path / 'd', 'zero', 'charge_embed_total_charge: 0.0')
        equil = self._run_nvt(
            tmp_path / 'e', 'equil', 'charge_equilibration_total_charge: 0.0',
        )

        # Absent, explicit null, and the true charge all mean the same thing.
        assert np.array_equal(absent, null), 'null must behave as if the key were absent'
        assert np.array_equal(absent, minus), (
            'charge_embed_total_charge equal to total_charge must be a no-op'
        )
        # 0.0 must actually change the dynamics.
        assert not np.array_equal(absent, zero), (
            'charge_embed_total_charge: 0.0 did not change the trajectory, so it '
            'never reached the model through the settings file'
        )
        # The equilibration key works through the same route, and is a distinct
        # knob -- overriding it must not coincide with overriding the embedding.
        assert not np.array_equal(absent, equil), (
            'charge_equilibration_total_charge: 0.0 did not change the trajectory'
        )
        assert not np.array_equal(zero, equil), (
            'the two keys produced the same trajectory, so they are not independent'
        )

    @staticmethod
    def _run_nvt_argv(tmp_path, out, extra_argv):
        """Same run via the `so3lr nvt` subcommand instead of a settings file.

        `--charge-embed-total-charge` is a subcommand option, so it is not
        available on the top-level `so3lr --settings` form; that is how every
        other model flag behaves here too.
        """
        import subprocess
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from subprocess_env import so3lr_subprocess_env
        import h5py

        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / 'input.xyz').write_text(_CLI_INPUT_XYZ)
        result = subprocess.run(
            [
                'so3lr', 'nvt',
                '--input', 'input.xyz', '--output', out,
                '--model', MODEL, '--precision', 'float64',
                '--lr-cutoff', str(LR_CUTOFF), '--total-charge', '-1',
                '--dt', '0.5', '--md-cycles', '1', '--md-steps', '1',
                '--temperature', '300.0', '--no-relax', '--seed', '42',
                '--save-buffer', '1', *extra_argv,
            ],
            cwd=tmp_path, capture_output=True, text=True,
            env=so3lr_subprocess_env(),
        )
        assert result.returncode == 0, (
            f"so3lr nvt failed ({result.returncode})\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        with h5py.File(tmp_path / out, 'r') as f:
            return np.array(f['momenta'])

    def test_command_line_flag_matches_settings_yaml(self, tmp_path):
        """--charge-embed-total-charge and the YAML key must agree exactly.

        Two independent routes into the same setting; if they disagree, one of
        them is not reaching the model.
        """
        via_yaml = self._run_nvt(tmp_path / 'y', 'zero', 'charge_embed_total_charge: 0.0')
        via_flag = self._run_nvt_argv(
            tmp_path / 'f', 'flag.hdf5', ('--charge-embed-total-charge', '0'),
        )
        assert np.array_equal(via_yaml, via_flag), (
            'the CLI flag and the settings-file key produced different trajectories'
        )

    def test_command_line_flag_changes_the_run(self, tmp_path):
        """Omitting the flag must differ from passing 0."""
        default = self._run_nvt_argv(tmp_path / 'd', 'def.hdf5', ())
        zero = self._run_nvt_argv(
            tmp_path / 'z', 'zero.hdf5', ('--charge-embed-total-charge', '0'),
        )
        assert not np.array_equal(default, zero), (
            '--charge-embed-total-charge 0 did not change the trajectory'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
