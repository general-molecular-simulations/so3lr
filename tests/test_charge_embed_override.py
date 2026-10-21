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
every trained weight in place: the 'embed_zero' case of the `energies` fixture
loads the released checkpoint with the override set, and would raise if the
parameter tree changed.
"""
import functools
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


@functools.lru_cache(maxsize=None)
def _evaluate(total_charge, charge_embed_total_charge, charge_equilibration_total_charge):
    """(energy, forces) of the charged fragment under one (charge, embed, equilibrate)
    condition.

    The fixtures and tests below ask for overlapping conditions (the default one
    alone four times), and every evaluation builds a model, so each distinct
    condition is evaluated once per session. Comparisons between two conditions
    still compare two separately built calculators.
    """
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
    return float(atoms.get_potential_energy()), atoms.get_forces()


def _energy(total_charge, charge_embed_total_charge, charge_equilibration_total_charge=None):
    # Fills in the default, so (Q, embed) and (Q, embed, None) share one cache entry.
    return _evaluate(total_charge, charge_embed_total_charge,
                     charge_equilibration_total_charge)[0]


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

    def test_accepts_arbitrary_values(self):
        """Not a boolean: any charge can be embedded, 0.0 is just the useful case.

        0.0 and -1.0 (the true charge) are already evaluated by the `energies`
        fixture; a charge of the opposite sign covers the rest.
        """
        assert np.isfinite(_energy(-1, 2.0))

    def test_forces_change_too(self):
        """Forces, not just energy -- the measured effect is a force MAE change."""
        f_default, f_zero = _evaluate(-1, None, None)[1], _evaluate(-1, 0.0, None)[1]
        assert np.all(np.isfinite(f_default)) and np.all(np.isfinite(f_zero))
        assert not np.array_equal(f_default, f_zero), (
            "forces are identical with and without the override"
        )


class TestChargeEmbedOverrideEntryPoints:
    """The flags have to be reachable from each documented entry point."""

    def test_python_entry_points_accept_both_overrides(self):
        """So3lrPotential and So3lr take both overrides (So3lrCalculator is above)."""
        from so3lr import So3lr, So3lrPotential

        # Constructing is the assertion: an unsupported kwarg would land in
        # long_range_kwargs and be silently ignored, or raise.
        overrides = dict(charge_embed_total_charge=0.0, charge_equilibration_total_charge=0.0)
        assert So3lrPotential(model=MODEL, lr_cutoff=LR_CUTOFF, **overrides) is not None
        assert So3lr(model=MODEL, lr_cutoff=LR_CUTOFF, **overrides) is not None

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

    def test_settings_yaml_route(self, momenta):
        """`charge_embed_total_charge:` in the settings YAML reaches the model.

        That an override equal to the true charge is a no-op is pinned in-process
        by TestChargeEmbedOverride.test_passthrough_is_exact, and that a YAML null
        means "absent" by test_yaml_null_is_the_key_absent.
        """
        absent, zero, equil = momenta['absent'], momenta['zero'], momenta['equil']

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

    def test_yaml_null_is_the_key_absent(self, tmp_path, monkeypatch):
        """`charge_embed_total_charge: null` must reach perform_md as the key absent
        (None), not 0.0 or the string 'None'. In-process: perform_md is stubbed, so
        this compares the settings dicts, not trajectories."""
        from click.testing import CliRunner
        from so3lr.cli import so3lr_cli, so3lr_md

        captured = {}
        monkeypatch.setattr(so3lr_md, 'setup_jax_globals', lambda **kwargs: None)
        monkeypatch.setattr(so3lr_cli, 'reserve_gpu', lambda *args, **kwargs: None)
        (tmp_path / 'input.xyz').write_text(_CLI_INPUT_XYZ)
        for tag, line in (('absent', '# key absent'), ('null', 'charge_embed_total_charge: null')):
            monkeypatch.setattr(so3lr_md, 'perform_md',
                                lambda settings, *args, tag=tag, **kwargs: captured.setdefault(tag, settings))
            (tmp_path / f'{tag}.yaml').write_text(
                f"input_file: {tmp_path / 'input.xyz'}\noutput_file: {tmp_path / tag}.hdf5\n"
                f"log_file: {tmp_path / tag}.log\nmodel: {MODEL}\nlr_cutoff: {LR_CUTOFF}\n"
                f"total_charge: -1\nensemble: nvt\nrelax_before_run: false\n{line}\n"
            )
            result = CliRunner().invoke(so3lr_cli.cli, ['--settings', str(tmp_path / f'{tag}.yaml')])
            assert result.exit_code == 0, result.output
        assert captured['null']['charge_embed_total_charge'] is None
        assert captured['null']['charge_embed_total_charge'] == captured['absent']['charge_embed_total_charge']

    @pytest.fixture(scope='class')
    def momenta(self, tmp_path_factory):
        """The four runs the tests in this class compare, each run once.

        Three through `so3lr --settings` (key absent, embedding override 0,
        equilibration override 0) and the embedding override 0 again through the
        `so3lr nvt` flag. Every run is a full CLI start-up, model build and JIT.
        """
        root = tmp_path_factory.mktemp('charge_embed_cli')
        return {
            'absent': self._run_nvt(root / 'a', 'absent', '# key absent'),
            'zero': self._run_nvt(root / 'z', 'zero', 'charge_embed_total_charge: 0.0'),
            'equil': self._run_nvt(
                root / 'e', 'equil', 'charge_equilibration_total_charge: 0.0',
            ),
            'flag_zero': self._run_nvt_argv(
                root / 'f', 'flag.hdf5', ('--charge-embed-total-charge', '0'),
            ),
        }

    def test_command_line_flag_matches_settings_yaml(self, momenta):
        """--charge-embed-total-charge and the YAML key must agree exactly.

        Two independent routes into the same setting; if they disagree, one of
        them is not reaching the model.
        """
        assert np.array_equal(momenta['zero'], momenta['flag_zero']), (
            'the CLI flag and the settings-file key produced different trajectories'
        )

    def test_command_line_flag_changes_the_run(self, momenta):
        """Omitting the override must differ from passing 0 on the command line.

        The run without an override goes through the settings file; the test above
        shows both routes give the same trajectory for the same setting.
        """
        assert not np.array_equal(momenta['absent'], momenta['flag_zero']), (
            '--charge-embed-total-charge 0 did not change the trajectory'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
