# Notes for coding agents

Guidance for an AI assistant asked to **modify this repository**. If you are
instead helping someone *use* SO3LR — writing settings files, choosing cutoffs,
calling the API — read
<https://so3lr.github.io/advanced/llm-agents/> instead; the traps are different.

Human contributors: see `DEVELOPERS.md` and
<https://so3lr.github.io/contributing/>.

## Setup

Python **3.12 or 3.13** is supported. Create the exact locked development
environment with uv:

```sh
uv sync --locked --group dev
uv run python -c "import so3lr, jax"
```

Use `--extra cuda12`, `--extra tfds`, `--extra pysages`, or `--extra wandb` when
the task needs those optional stacks. `DEVELOPERS.md` documents the
supported commands.
`uv sync --locked` pulls Git dependencies at immutable commits, so first-time
setup still needs network access and is not fast. A plain pip install remains
supported, but GLP's own metadata names its `comms` dependency by branch; only
the uv lock and override pin that transitive dependency exactly. Do not assume
a working environment before concluding that a failure is a code bug.

## Tests

```sh
uv run pytest tests/                                # short suite (the default)
uv run pytest tests/ --full                         # everything, incl. slow tests
uv run pytest tests/test_potential.py -v            # one file
uv run pytest tests/test_integration.py --shard 1/3 # how CI splits the slow ones
```

By default, tests marked `@pytest.mark.slow` are skipped and the parametrised
molecule tests (`atat`, `dha`, `bb`) run only `dha`. Full runs take tens of
minutes; prefer targeting the file that covers your change.

| Area | Tests |
|---|---|
| Model name resolution | `tests/test_model_registry.py` |
| ASE calculator, observables | `tests/test_ase.py`, `tests/test_observables.py` |
| k-space electrostatics | `tests/test_pme.py` |
| CLI surface | `tests/test_cli*.py` |
| End-to-end MD | `tests/test_integration.py` |
| Docs examples | `tests/test_readme.py` |

## Invariants worth knowing before you edit

**`so3lr/model_registry.py` is the single source of truth for model names.** The
CLI, the ASE calculator and the Python API all resolve through it. `_CANONICAL`
maps one name to one directory; `_FLOATING_ALIASES` are pointers that move
between releases. `DEFAULT_MODEL` is deliberately pinned to a *canonical* name
so the no-argument default cannot drift when an alias is repointed. Repointing
an alias silently changes everyone's results — say so in the changelog.

**`so3lr/__init__.py` must stay import-cheap.** It uses PEP 562 lazy attributes
so that `import so3lr` does not pull in JAX, Flax or JAX-MD; `so3lr --help`
depends on this. Adding a module-level import of a heavy dependency is a
regression even though nothing will fail.

**The CLI and ASE paths are separate and have drifted before.** `so3lr/cli/`
builds its own `long_range_kwargs`, and `so3lr/mlff/calculators/ase_calculator.py`
builds another. A fix applied to one is not automatically in the other — this
exact split produced a bug where `kspace_electrostatics` selected the k-space
grid but not the solver on the ASE side only (PR #22). When you change how a
model is configured, check both.

**`so3lr/finetuning_strategies.py` is the single source of truth for
`--strategy`.** The CLI builds its `click.Choice` from that table at import time
and `run_fine_tuning` resolves the same table into parameter keys. Keeping the
two hand-written lists in sync had already failed: `freeze_all_but_final_mlp`
was implemented but rejected by click. Selection keys use the matching grammar
of `param_utils.path_matches_key` (`a`, `a/b`, `a/b*`), and a key that matches
nothing is rejected rather than silently freezing the model. That module must
stay import-cheap — the CLI imports it at module scope.

**Training and fine-tuning are separate loops.** `run_training` and
`run_fine_tuning` in `so3lr/mlff/config/from_config.py` do not share a code
path, and they read different `training:` keys — fine-tuning ignores `use_ema`,
`early_stopping_patience` and `stage_two_*` entirely. A feature added to one
does not appear in the other, and nothing warns about it.

**Two model generations produce different output keys.** `so3lr-s/m/l` emit
`a0_ratios`, `c6_ratios` and NLH repulsion; the legacy `so3lr` emits
`hirshfeld_ratios` and ZBL. Code touching observables must handle both — see
the `legacy` branches in `so3lr/mlff/nn/observable/observable_sparse.py`.

**`head_index` is not fully plumbed.** `ModelSpec.head_index` exists, but the
`make_so3lr` / `make_potential_fn` / jax-md paths still assume head 5 (there is
a hardcoded `jnp.eye(16)[5]` in `so3lr/jaxmd_utils.py`). A model family needing
a different head requires threading it through those entry points first.

**Failures should exit non-zero.** The CLI is used in shell chains and SLURM
dependencies; keep `sys.exit(1)` on error paths.

**Unknown settings-YAML keys warn rather than raise**, on purpose — one typo
should not abort a queued multi-hour job. Preserve that, and keep new keys
listed in `YAML_DEFAULTS` or `KNOWN_YAML_KEYS_EXTRA` in `so3lr/cli/so3lr_cli.py`
so the did-you-mean validator does not flag them as typos.

## Conventions

- Match the surrounding style; the codebase is not uniformly formatted, so
  follow the file you are in rather than reformatting it.
- Explain *why* in comments when behaviour is surprising. Several non-obvious
  decisions here are load-bearing and already carry such comments — do not strip
  them.
- Add or update tests with any behaviour change.
- User-facing changes need a docs update at <https://so3lr.github.io/>. The
  reference tables that mirror code are listed in that site's README; a stale
  row there is worse than a missing one.
- Do not commit model checkpoints or datasets beyond what `so3lr/models/`
  already ships.

## Things that are not bugs

- The CLI refuses to overwrite an existing trajectory. This is deliberate;
  `--overwrite` opts in.
- Geometry relaxation runs before MD by default. Its frames go to a separate
  `<output-stem>_opt.xyz` file; the MD trajectory contains accepted MD frames.
- The active Langevin PRNG key is saved with the integrator state and restored
  on restart. The setup key is reseeded, but does not replace that saved key.
  Reproducibility still depends on numerical precision and backend.
- SO3LR is single-device. There is no `pmap`/sharding, and adding one is a
  substantial change rather than a fix.
