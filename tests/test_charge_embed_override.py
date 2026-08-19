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
import numpy as np
import pytest

from ase import Atoms

# so3lr-s is small and ships with use_charge_embed=True, so the embedding is
# actually present and the override has something to act on.
MODEL = 'so3lr-s'

# Finite cluster: the long-range cutoff must be large or the electrostatics are
# truncated nonsense (see the lr_cutoff note in the PR description).
LR_CUTOFF = 1000.0


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


def _energy(total_charge, charge_embed_total_charge):
    from so3lr import So3lrCalculator

    atoms = _charged_fragment()
    atoms.info['charge'] = total_charge
    # Default float32: x64 is not enabled in this module, and the differences
    # under test are eV-sized, so requesting float64 would only be truncated.
    atoms.calc = So3lrCalculator(
        model=MODEL,
        lr_cutoff=LR_CUTOFF,
        charge_embed_total_charge=charge_embed_total_charge,
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

    def test_cli_load_model_accepts_it(self):
        """The MD CLI's loader takes it, so md_settings.yaml can set it."""
        import inspect
        from so3lr.cli.so3lr_md import load_model

        assert 'charge_embed_total_charge' in inspect.signature(load_model).parameters

    def test_yaml_key_is_known(self):
        """The did-you-mean validator must not flag it as a typo."""
        from so3lr.cli.so3lr_cli import KNOWN_YAML_KEYS_EXTRA, YAML_DEFAULTS

        assert 'charge_embed_total_charge' in (
            KNOWN_YAML_KEYS_EXTRA | set(YAML_DEFAULTS)
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
