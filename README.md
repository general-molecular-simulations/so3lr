![workflow-test-ci](https://github.com/general-molecular-simulations/so3lr/actions/workflows/CI.yml/badge.svg)
[![examples-link](https://img.shields.io/badge/example-notebooks-F37726)](./examples)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-molecular-simulations/so3lr/blob/main/examples/so3lr_colab_example.ipynb)
[![cite-link](https://img.shields.io/badge/how_to-cite-000000)](https://github.com/general-molecular-simulations/so3lr?tab=readme-ov-file#Citation)
[![preprint-link](https://img.shields.io/badge/paper-JACS-FFCE34)](https://doi.org/10.1021/jacs.5c09558)
[![data](https://zenodo.org/badge/DOI/10.5281/zenodo.14779793.svg)](https://doi.org/10.5281/zenodo.14779793)
![Logo](./logo.png)

## About
SO3LR - pronounced *Solar* - is a pretrained machine-learned force field for (bio)molecular simulations. It integrates the fast and stable SO3krates neural network for semi-local interactions with universal pairwise force fields designed for short-range repulsion, long-range electrostatics, and dispersion interactions.

## Quick Start
Try SO3LR without any local installation using our [Colab notebook](https://colab.research.google.com/github/general-molecular-simulations/so3lr/blob/main/examples/so3lr_colab_example.ipynb)!

## Installation
SO3LR supports Python 3.12 and 3.13. For a locked installation from a
repository checkout, use [uv](https://docs.astral.sh/uv/) with the committed
lockfile:

```shell script
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
uv sync --locked
uv run so3lr --help
```

For paper reproduction, check out a published release tag or an immutable
40-character commit before `uv sync --locked`; cloning a moving default branch
does not identify an exact source version.

This creates a CPU environment. For CUDA 12 on Linux x86-64, select the locked
CUDA extra with `uv sync --locked --extra cuda12`. See
[`DEVELOPERS.md`](./DEVELOPERS.md) for development environments and the
lockfile policy. SO3LR runs significantly faster on a compatible GPU for
large-scale simulations.

### Optional extras

The base install uses JAX and the scientific Python stack, but does not install
telemetry clients. Optional extras are:

- `uv sync --locked --extra pysages` — PySAGES enhanced-sampling MD.
- `uv sync --locked --extra tfds` — the TFDS training and fine-tuning loader;
  this selects the platform-appropriate TensorFlow package.
- `uv sync --locked --extra wandb` — the Weights & Biases client, used only
  when `training.use_wandb: true` is explicitly enabled.
- `uv sync --locked --extra cuda12` — CUDA 12 JAX packages on Linux x86-64.

### Installing into an existing environment (pip)

On clusters with a working JAX install, install SO3LR into that environment
with pip. The dependencies only cap JAX (`jax<0.11`, `flax<0.12.9`), so a
compatible existing JAX/CUDA or ROCm stack is kept:

```shell script
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
pip install .            # or: pip install ".[tfds,wandb]"
```

Without an existing JAX, install `jax` (CPU) or `"jax[cuda12]<0.11"` first.

## Models

SO3LR ships in three sizes. All are bundled inside the package, so you select one **by name** — no download or path needed. They trade off speed against accuracy:

| Model | Layers | Features | Heads | Cutoff (Å) |
|-------|--------|----------|-------|-----------|
| `so3lr-s-2606` (`so3lr-s`) | 2 | 128 | 4 | 4.5 |
| `so3lr-m-2606` (`so3lr-m`) *(default)* | 3 | 128 | 4 | 4.5 |
| `so3lr-l-2606` (`so3lr-l`) | 3 | 256 | 8 | 5.0 |
| `so3lr` (`so3lr-o`) | 3 | 128 | 4 | 4.5 |

If you do not specify a model, the pinned canonical model
**`so3lr-m-2606`** is used. `so3lr-o` (also reachable as `so3lr`) is the
original SO3LR v1, kept for backward compatibility and for reproducing the
[JACS paper](https://doi.org/10.1021/jacs.5c09558).

### Model names and versions

Names follow `so3lr-<size>[-<release>]` and come in two tiers:

| Name | Means | Stability |
|------|-------|-----------|
| `so3lr-m` | current recommended M | moves when a new release lands |
| `so3lr-m-2606` | one exact release | immutable |

`<size>` is `s`, `m` or `l`; `<release>` is the `YYMM` of the release.

The short names (`so3lr-s`/`-m`/`-l`) are convenience aliases for the current
release. **Cite the dated name in papers and benchmark tables** — the aliases are
repointed when a new release lands, so a script saying `so3lr-m` keeps running
but returns different numbers. Resolving an alias logs which target it picked.
Currently bundled:

| Release | Canonical names |
|---------|-----------------|
| 2606 *(current, default)* | `so3lr-{s,m,l}-2606` |
| 2602 | `so3lr-{s,m,l}-2602` |
| Original v1 | `so3lr` (alias `so3lr-o`) |

`so3lr.model_registry.canonical_models()` lists the canonical names and
`model_aliases()` the alias → target mapping.

### Verifying bundled model artifacts

Every bundled checkpoint and its adjacent metadata file is covered by
`so3lr/models/SHA256SUMS`:

```shell script
cd so3lr/models && sha256sum -c SHA256SUMS   # macOS: shasum -a 256 -c SHA256SUMS
```

The checksum proves byte identity with this release; it does not make an
untrusted pickle safe to load. Only load checkpoints obtained from a trusted
SO3LR release.

### Selecting a model

The same names work across every interface:

```python
# Python / ASE
from so3lr import So3lrCalculator
calc = So3lrCalculator(model="so3lr-m-2606")
```

```shell script
# CLI (works with opt / nvt / npt / nve / eval)
so3lr nvt --input geometry.xyz --model so3lr-m-2606
```

```yaml
# In an MD settings YAML (model: is an alias of model_path:)
model: so3lr-m-2606
```

Custom and fine-tuned models still work — just pass a path instead of a name:

```python
So3lrCalculator(model="/path/to/finetuned")      # or  --model /path/to/finetuned  on the CLI
```

> **Backward compatibility:** The no-arg default is `so3lr-m-2606`. To reproduce the **original SO3LR v1** model from the [JACS paper](https://doi.org/10.1021/jacs.5c09558), pass `model="so3lr"` (alias `"so3lr-o"`).

## Command Line Interface (CLI)

SO3LR provides a unified command-line interface that leverages the performance of JAX-MD under the hood (detailed in the JAX-MD section below). This allows you to perform geometry optimizations and MD simulations with simple commands. The CLI supports several key functionalities:

- `so3lr opt`: Geometry optimization
- `so3lr nvt`: NVT molecular dynamics (Nose-Hoover chain thermostat)
- `so3lr nvt-langevin`: NVT molecular dynamics with a Langevin thermostat
- `so3lr npt`: NPT molecular dynamics
- `so3lr nve`: NVE molecular dynamics
- `so3lr eval`: Model evaluation on a dataset
- `so3lr finetune`: Fine-tune the model on custom datasets
- `so3lr train`: Train a SO3LR/SO3krates model from scratch
- `so3lr tune-ewald`: Auto-tune Ewald/PME/P3M k-space parameters for a structure

Each subcommand has its own set of options and can be run with `--help` to see all available parameters.

```shell script
so3lr opt --help
```

All MD subcommands also accept a single `--settings` YAML file that bundles every CLI flag (plus features that have no CLI flag, like atom constraints and enhanced sampling). This is the recommended way to drive longer or more elaborate runs:

```shell script
so3lr --settings settings.yaml
```

See the *Freezing atoms during MD*, *Long-Range Electrostatics*, and *Enhanced Sampling with PySAGES* sections below for example settings files.

Optimize a structure using the FIRE algorithm:

```shell script
so3lr opt --input geometry.xyz --force-conv 0.05 --lr-cutoff 12.0
```

Run NVT (constant volume and temperature) simulation:

```shell script
so3lr nvt --input geometry.xyz --temperature 300 --dt 0.5 --md-cycles 100 --md-steps 100
```

Run NPT (constant pressure and temperature) simulation:

```shell script
so3lr npt --input geometry.xyz --temperature 300 --pressure 1.0 --dt 0.5 --md-cycles 100 --md-steps 100
```

Run NVE (constant volume and energy) simulation:

```shell script
so3lr nve --input geometry.xyz --temperature 300 --dt 0.5 --md-cycles 100 --md-steps 100
```

Both NVT and NPT ensembles are supported using the Nosé–Hoover chain thermostat and barostat. The trajectories can be saved in `.hdf5` or `.xyz` format. In addition, checkpoints can be saved as `.npz` files throughout the simulation to restart if needed.

NPT settings support `npt_coupling: isotropic` and
`npt_coupling: semi-isotropic` (the x/y dimensions scale together while z
changes independently). Fully flexible `anisotropic` coupling is currently
rejected: its shear components are not yet represented consistently by every
neighbor-list, reciprocal-space, and dispersion-tail path, so accepting it
could silently produce incorrect dynamics.

HDF5 trajectories use a frame-major layout: `positions` and `momenta` have
shape `(n_frames, n_atoms, 3)`, while `box` has shape `(n_frames, 3)` for an
isotropic orthorhombic cell or `(n_frames, 3, 3)` for a semi-isotropic cell. The
`fractional_coordinates`, `trajectory_layout`, and `box_convention` root
attributes describe how to interpret these arrays. The write buffer controls
I/O chunking only; a final partial buffer is always written.

When `--relax` is enabled, relaxation frames are written to a separate
`<output-stem>_opt.xyz` file. The requested MD trajectory contains accepted MD
frames only. Standalone `so3lr opt` continues to use its requested EXTXYZ path.
With `--overwrite`, EXTXYZ replaces an existing trajectory after the first
batch is serialized, whereas HDF5 initializes its replacement container before
integration. Use a new output path if a failed run must leave an old HDF5 file
untouched. Later EXTXYZ batches use normal streaming appends.

Example with additional parameters for NVT:

```shell script
so3lr nvt --input geometry.xyz --output trajectory.hdf5 --temperature 300 \
    --dt 0.5 --md-cycles 1000 --md-steps 1000 \
    --nhc-chain 3 --nhc-steps 2 --nhc-thermo 100.0 \
    --relax --force-conv 0.1 --seed 42 \
    --restart-save checkpoint.npz
```

MD simulations can be restarted from a previously saved checkpoint. This is useful for extending simulations or recovering from interruptions. To enable restart:

```shell script
#1. First save a checkpoint (updated every `save_buffer` cycles and at the final cycle) during your simulation:
so3lr nvt --input geometry.xyz --output trajectory.xyz --temperature 300 --md-cycles 100 --restart-save checkpoint.npz

#2. Then restart from this checkpoint to continue the simulation:
so3lr nvt --input geometry.xyz --output trajectory.xyz --temperature 300 --md-cycles 100 --restart-load checkpoint.npz --restart-save checkpoint_new.npz
```

The restart will continue the simulation from the exact state where it was saved, preserving atom positions, velocities, thermostat/barostat state, and simulation timestep.
SO3LR's NPZ checkpoint is written to a same-directory temporary file and replaces
the previous checkpoint only after serialization succeeds. PySAGES saves its
restart separately; the two files are not a crash-atomic pair. Keep matching
checkpoint generations when restarting enhanced sampling.

SO3LR's own `.npz` restart format contains only numeric arrays and strings and
is loaded with NumPy object unpickling disabled. This does **not** make other
serialization formats safe: legacy/custom model `params.pkl` files and
PySAGES `.pkl` restart files are pickle-compatible and can execute code while
being loaded. SO3LR emits a runtime security warning before loading either
format. Only use them from a trusted source after verifying a published or
out-of-band checksum; do not load an arbitrary downloaded pickle.

#### What restart_save preserves (and what it doesn't)

The `.npz` written by `--restart-save` is produced by `save_state()` in `so3lr/cli/so3lr_md.py` and reloaded by `load_state()` in the same file. The exact set of fields is:

**Preserved across a restart:**
- atomic positions, momenta, forces, masses
- the simulation box and the `fractional_coordinates` flag (so a cartesian/fractional mismatch between save and restart runs is detected and converted automatically)
- the cycle / step counter (the run resumes at the saved cycle, not from 0)
- the ensemble tag (NVT/NPT/NVE), and for NVT/NPT the full Nose-Hoover chain state of both the thermostat and the barostat (position, momentum, mass, tau, kinetic energy, degrees of freedom)
- for NPT: the reference box, box position/momentum/mass, and the saved stress (or stress_tensor under `npt_flex`)
- enhanced-sampling (PySAGES) state, when `restart_pysages_save_path` is also used — saved/loaded as a separate `.pkl` (see the PySAGES section below)

**NOT preserved (regenerated on each launch):**
- the initial JAX/NumPy setup key; the active Langevin-thermostat key is part of the saved integrator state
- elapsed wall-clock time (the wall-time log resets to zero for the restarted process)
- neighbor-list allocations and JIT-compiled functions (rebuilt on launch from current geometry and buffer multipliers)
- output/log file buffers (EXTXYZ appends on restart unless `--overwrite` is
  requested; use a new output path to keep runs separate. HDF5 requires a new
  trajectory path unless replacement is explicitly requested with `--overwrite`.)

The deterministic NVT/NPT Nose-Hoover and NVE integrators, and the saved
Langevin integrator state including its active RNG key, are reproducible across
a restart up to numerical and backend differences.

### Freezing atoms during MD

A subset of atoms can be held fixed during any MD run (`nve` / `nvt` / `npt`). Frozen atoms keep their reference positions while their momenta and forces are zeroed every step, so the rest of the system evolves around them (useful e.g. for holding a surface, a binding pocket, or a substrate rigid). Constraints are configured through the MD **settings YAML** (run with `so3lr --settings settings.yaml`):

```yaml
# settings.yaml
ensemble: nvt
input_file: geometry.xyz
model: so3lr-m-2606
md_T: 300
md_cycles: 100
md_steps: 100

# --- atom constraints (all optional; omit to run unconstrained) ---
fixed_atom_indices: [0, 1, 2]            # 0-based indices of atoms to freeze
# mobile_atom_indices: [3, 4, 5]         # alternative: list the atoms allowed to MOVE (all others are frozen)
# atom_mask_path: masks.npz              # alternative: load 'mobile_atom_indices'/'fixed_atom_indices' arrays from a .npz
fixed_atom_constraint: cartesian_anchor  # default; pins frozen atoms in real (Cartesian) space, robust to box changes under NPT
output_cartesian_positions: false        # for periodic runs, write saved positions in Cartesian coordinates
```

```shell script
so3lr --settings settings.yaml
```

Notes:
- Provide **either** `fixed_atom_indices` **or** `mobile_atom_indices` — the complement is derived automatically. If both are given, they are used as-is.
- `atom_mask_path` points to a NumPy `.npz` holding `mobile_atom_indices` and/or `fixed_atom_indices` arrays; explicit `*_atom_indices` keys in the YAML override it.
- With no constraint keys set, MD runs normally (the feature is a no-op). On start-up a constrained run logs e.g. `Fixed atoms: 3; mobile atoms: 97; constraint=cartesian_anchor`.

### Restricting trajectory output to a subset of atoms

The `output_atom_indices` YAML key (also surfaced as `--output-atom-indices` on the CLI) restricts the saved trajectory to a chosen subset of atoms — the MD itself still propagates the full system, but only the listed atoms are written to the `.hdf5` / `.xyz` trajectory at each save step. This is useful for keeping trajectory files small in large solvent boxes when you only care about a solute, a binding pocket, an interface slab, or a tagged ligand:

```yaml
# settings.yaml — save only the first 50 atoms (e.g. the solute) of a solvated box
output_atom_indices: [0, 1, 2, ..., 49]
```

The full box and the full per-step energies/forces are still computed; only the trajectory writer is filtered. Leave it as `null` (default) to dump every atom.

Evaluate the SO3LR model on a dataset:

```shell script
so3lr eval --datafile dataset.extxyz --batch-size 1 --lr-cutoff 100 --save-to predictions.extxyz
```

Add `--model so3lr-s-2606` (or `so3lr-l-2606`) to evaluate a different
bundled model; without it, the pinned `so3lr-m-2606` default is used. Floating
aliases such as `so3lr-s` are convenient for exploration but should not appear
in publication scripts.

The input can be any file that is digestible by [`ase.io.iread`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.iread).

> [!IMPORTANT]
> The legacy `so3lr` checkpoint was not trained with an energy loss, so only
> relative energies are meaningful for that checkpoint. The dated v2
> `so3lr-{s,m,l}-2602` and `-2606` checkpoints include energy supervision.
> Always report the exact canonical checkpoint used. Labels are assumed to be
> in `eV` and `Ångström`. For gas-phase simulations, we suggest using
> `--lr-cutoff 100`.

The command will collect and print metrics on the dataset and save the predictions to the specified output file. The predicted properties are `energy`, `forces`, `dipole_vec` and the per-atom ratio head. The ratio head is named `a0_ratios` for the new `so3lr-s/m/l` models (polarizability volume ratios) and `hirshfeld_ratios` for the legacy `so3lr` model -- you can pass either name to `--targets` and the CLI will normalize it to whatever the chosen model actually outputs (and accept dataset labels stored under either key). The new `so3lr-s/m/l` models additionally predict `c6_ratios` (per-atom C6-coefficient ratios used by the dispersion term); pass `c6_ratios` to `--targets` to score it against dataset labels stored under the same key. Energy and forces are assumed to be present in the datafile, while dipole vectors, `a0_ratios`/`hirshfeld_ratios`, and `c6_ratios` labels are optional. If they are not present in the data, the metrics will simply be `NaN`.

**Observable targets** — intermediate model quantities (no ground truth required) can be added to `--targets`. They are saved to the output extxyz file alongside the standard predictions:

```shell script
so3lr eval --datafile dataset.extxyz --lr-cutoff 100 --save-to predictions.extxyz \
    --targets forces,dipole_vec,nn_energy,partial_charges,electrostatic_energy,dispersion_energy
```

**Atom embeddings** — the final-layer per-atom feature vectors (`x`) are written to a separate compressed NumPy file `<output>_x.npz` because of their size:

```shell script
so3lr eval --datafile dataset.extxyz --lr-cutoff 100 --save-to predictions.extxyz \
    --targets forces,x
# creates: predictions.extxyz  (standard predictions)
#          predictions_x.npz   (atom embeddings)
```

```python
import numpy as np

data = np.load('predictions_x.npz')
embeddings = data['embeddings']  # shape: (total_atoms, features)
n_atoms    = data['n_atoms']     # shape: (n_structures,)

# Split back into per-structure arrays
per_structure = np.split(embeddings, np.cumsum(n_atoms[:-1]))
```

Evaluation outputs are replaced only when `--overwrite` is requested. Prediction
and embedding files are saved independently, not as a multi-file transaction.

The predictions can be analyzed in Python:

```python
import numpy as np
from ase.io import iread

property = 'forces'

true = []
so3lr = []
for a in iread('predictions.extxyz'):
    true.append(a.get_forces())
    so3lr.append(a.arrays[f'{property}_so3lr'])

rmse = np.sqrt(np.mean(np.square(np.stack(true) - np.stack(so3lr))))
print(rmse)
```

For more in-depth evaluation using Python, check out the [example notebook](https://github.com/general-molecular-simulations/so3lr/blob/main/examples/evaluate_so3lr_on_dataset.ipynb).

## Fine-tuning SO3LR (experimental)

SO3LR can be fine-tuned on custom datasets to improve performance on specific systems or molecular environments. The input datafile should be readable by ASE (e.g., `.extxyz`, `.xyz`) and include atomic positions and forces (ideally also dipole moments and Hirshfeld ratios). Please refer to the Zenodo repository for an example data file.

Basic fine-tuning command:

```shell script
so3lr finetune --datafile dataset.xyz --workdir so3lr_finetuned --num-train 1000 --num-valid 100
```

You can customize the training process with a config file (`--config custom_finetune.yaml`), choose different fine-tuning strategies (`--strategy full`), or fine-tune from a previously trained model (`--model-path ./previous_finetune_workdir`). Possible strategies include `full`, `final_mlp`, `last_layer`, `last_layer_and_final_mlp`, `first_layer`, and `first_layer_and_last_layer`. We recommend using the `full` or `last_layer_and_final_mlp` strategies. The default configuration in `so3lr/config/finetune.yaml` includes settings for the optimizer, learning rate schedule, batch size, loss weights, and data filtering. To use a fine-tuned model, use the `--model path_to_finetuned_model` flag.

### Fine-tuning on forces only (recommended: `freeze_long_range`)

SO3LR's long-range energy terms are driven by dedicated prediction heads: electrostatics from the partial-charge/dipole head, and dispersion from the polarizability (`a0_ratios`) and `c6_ratios` heads. These heads are trained against their own labels (dipoles/charges, Hirshfeld/polarizability, C6). If your fine-tuning dataset contains **only forces** (no dipole/charge/Hirshfeld/C6 labels) and you train with `--strategy full`, the force gradients will silently drag these heads away from their physically-meaningful values — degrading the electrostatics and dispersion even though the forces look fine.

For forces-only datasets, use the dedicated strategy that **freezes the long-range heads** and trains only the short-range representation and energy head:

```shell script
so3lr finetune --datafile forces_only.xyz --strategy freeze_long_range --num-train 1000 --num-valid 100
```

`freeze_long_range` (alias `forces_only`) freezes the dipole/partial-charge, `a0_ratios`, and `c6_ratios` heads (`observables_1/2/3`) while keeping the message-passing layers and the energy head (`observables_0`) trainable, so the pretrained electrostatics and dispersion are preserved. If your dataset *does* include dipole/Hirshfeld/C6 labels, prefer `full` (or add those terms to `loss_weights`) so the long-range heads adapt with supervision.

## Training from Scratch

You can train a SO3krates model from scratch using a YAML configuration file. The default configuration is in `so3lr/config/config.yaml` and includes settings for the model architecture, optimizer, learning rate schedule, batch size, loss weights, and data filtering.

Basic training command:

```shell script
so3lr train --config config.yaml
```

### Optional W&B telemetry and configuration privacy

Weights & Biases logging is disabled by default for both training and
fine-tuning. SO3LR does not initialize W&B or send metrics when
`training.use_wandb` is absent or `false`, and the base installation does not
include the W&B client. Install it with `uv sync --locked --extra wandb`, then opt in
explicitly:

```yaml
training:
  use_wandb: true
  wandb_init_args:
    project: my-project
```

Enabling it sends training metrics and a copy of the effective SO3LR
configuration to Weights & Biases. Before `wandb.init`, SO3LR replaces obvious
credential-like configuration values and absolute local data/work paths with
redaction markers. Keep credentials in the W&B login/environment rather than
the YAML file, and review the remaining configuration before enabling
telemetry. The W&B client may collect additional runtime, system, repository,
or environment metadata under its own settings and privacy policy; SO3LR's
redaction applies to the configuration payload it supplies.

### Training on multiple datasets

Instead of a single `data.filepath`, you can list several datasets under `data.datasets` and the trainer will draw from them according to per-dataset `weight` values:

```yaml
data:
  filepath: null
  datasets:
    - path: /data/dataset_a.extxyz
      weight: 1.0
      # num_train: 5000   # optional per-dataset cap (TFDS only)
    - path: /data/dataset_b.extxyz
      weight: 0.5
```

Multi-dataset sampling currently requires the **TFDS** dataloader; install it
via `uv sync --locked --extra tfds` (see *Optional extras*). `num_train` caps each source's
training split before its weight is applied.

For TFDS mixtures, one training epoch makes a finite number of weighted draws
equal to the total selected record count across datasets with non-zero weights.
Sampling is with replacement and is reproducible from `training_seed` for a
fixed `batch_n_proc`. When a dataset's weight equals its selected size, it is
visited once per epoch in expectation. Validation does not repeat records and
visits every selected record exactly once in a fixed order, so unchanged model
parameters are evaluated on identical batches. Each new TFDS checkpoint
atomically stores deployable `params` and a versioned training-state item. The
latter contains raw parameters, optimizer/EMA state, logical epoch and batch
position, counters, early-stop/plateau/two-stage state, and the JAX RNG key.
Restart rebuilds the seeded epoch and skips the recorded prefix. A SHA-256
fingerprint rejects changes to TFDS builder identity metadata, shard names and
sizes, dataset weights/caps, batch budgets, neighbor/filter/unit settings,
seeds, process count, model/optimizer/objective configuration, relevant runtime
versions, and trajectory-control settings. Metadata fingerprinting does not
hash every TFRecord payload and cannot prove that Python model/loss/optimizer
code is semantically unchanged despite identical configuration;
keep data directories immutable and use source/environment provenance for that.
Fine-tuning fingerprints also include the resolved trainable/frozen parameter
masks. Older TFDS fine-tuning checkpoints without that fingerprint are rejected
for exact training continuation; inference from their parameters is unchanged.
Legacy params-only checkpoints remain loadable for inference but are rejected
for TFDS training restart because their exact trajectory cannot be recovered.
ASE/NPZ restarts remain weights-only continuations: they load the target
checkpoint's parameters and step, but reinitialize the optimizer, EMA, and
data-order state. They do not provide TFDS's exact full-state continuation.
Training and fine-tuning publish updated split/model metadata only after the
target checkpoint has passed restart validation, so a rejected restart leaves
the retained checkpoint's metadata intact.
TFDS records are converted to eV/Angstrom before filters and neighbor lists are
evaluated. `filter.max_force`
uses the largest absolute force component and rejects non-finite forces, while
`filter.min_distance` uses the minimum-image distance along exactly the axes
enabled by each record's `pbc` value.

### Optional training features

The `training:` block of the config supports a few optional features, all
disabled by default (see the commented entries in `so3lr/config/config.yaml`):

- **EMA of weights** — `use_ema: true` (with `ema_decay`, default `0.999`) keeps an
  exponential moving average of the parameters. The raw weights keep training
  while the smoothed EMA copy is used for validation and saved in the checkpoint,
  which is usually more stable and accurate at deployment.
- **Early stopping** — `early_stopping_patience: <N>` halts training once the
  validation loss has not improved (by more than `early_stopping_min_delta`) for
  `N` consecutive evaluations.
- **Reduce LR on plateau** — `reduce_lr_on_plateau: true` multiplies the learning
  rate by `reduce_lr_factor` (default `0.75`) whenever the validation loss stalls
  for `reduce_lr_patience` evaluations, down to a floor of
  `learning_rate * reduce_lr_min_scale`. This composes with the configured
  `learning_rate_schedule`.
- **Two-stage loss reweighting** — set `stage_two_start_epoch: <epoch>` and
  `stage_two_loss_weights:` to switch to a second set of loss weights partway
  through training (e.g. start energy-heavy, then emphasise forces). The
  best-loss, early-stopping and plateau trackers reset at the switch.

## Dimer Binding Energy Calculations

You can calculate binding energies with SO3LR as reported in the paper (Fig. 2B). The binding energy is computed as the difference between the energies of the bound dimer and the non-interacting monomers (separated by a distance larger than the long-range cutoff), with charges assigned to each monomer separately to prevent spurious charge transfer.

For XYZ files with dimer metadata (`charge_a`, `charge_b`, `selection_a`, `selection_b`, like [NCIAtlas](https://github.com/Honza-R/NCIAtlas/blob/main/geometries/NCIA_D1200/1.01.01_100.xyz) format), generate original and translated structures:

```shell script
python so3lr/prepare_dimer_xyz.py --datafile sapt10k.xyz
```

Then evaluate energies (see [SAPT10k](https://doi.org/10.1063/5.0204064) dataset in examples/data):

```shell script
so3lr eval --datafile sapt10k_prepared.xyz --targets energy --save-to sapt10k_eval.xyz --lr-cutoff 100
```

Finally, compute the binding energy:
```python
from ase.io import read

mols = read('sapt10k_eval.xyz', index=':')
energy = [a.info['energy_so3lr'] for a in mols]
bind_energy = [energy[i]-energy[i+1] for i in range(0, len(energy), 2)]
```

## Atomic Simulation Environment
To obtain an Atomic Simulation Environment (ASE) calculator that predicts energies and forces using SO3LR, simply run: 
```python
from jax import config

# JAX disables 64-bit arrays by default. This must run before SO3LR is imported.
config.update("jax_enable_x64", True)

import numpy as np

from so3lr import So3lrCalculator
from ase import Atoms

atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)])
atoms.info['charge'] = 0.0

calc = So3lrCalculator(
    model="so3lr-m-2606",    # canonical dated name for reproducible results
    calculate_stress=False,
    lr_cutoff=100, # for gas-phase systems
    dtype=np.float64
)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print('Energy and forces in ASE')
print('Energy = ', energy)
print('Forces = ', forces)

```

## Advanced Calculator Features

The SO3LR calculator provides access to individual energy components, their gradients, and Hessian matrices for advanced analysis and method development.

### Extracting Observables

You can extract individual energy components and other observables from the model by enabling `has_aux=True` and specifying which observables to output:

```python
from jax import config

# Enable this before importing SO3LR whenever dtype=np.float64 is requested.
config.update("jax_enable_x64", True)

import numpy as np
from so3lr import So3lrCalculator
from ase import Atoms

atoms = Atoms('OH2', positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])
atoms.info['charge'] = 0.0

calc = So3lrCalculator(
    model="so3lr-m-2606",
    lr_cutoff=12.0,
    dtype=np.float64,
    has_aux=True,
    observables=[
        'nn_energy',              # Neural network energy
        'electrostatic_energy',   # Electrostatic energy
        'dispersion_energy',      # Dispersion energy
        'zbl_repulsion',          # ZBL repulsion energy
        'partial_charges',        # Atomic partial charges
        'a0_ratios'               # Polarizability volume ratios (v2 models)
    ]
)

atoms.calc = calc
energy = atoms.get_potential_energy()

# Access observables from auxiliary outputs
aux = calc.results['aux']

# Energy components (per-atom arrays, sum for total)
nn_energy = np.sum(aux['nn_energy'])
electrostatic_energy = np.sum(aux['electrostatic_energy'])
dispersion_energy = np.sum(aux['dispersion_energy'])
zbl_repulsion = np.sum(aux['zbl_repulsion'])

# Atomic properties
partial_charges = aux['partial_charges']      # shape: (n_atoms,)
a0_ratios = aux['a0_ratios']                  # shape: (n_atoms,)

print(f"Total energy: {energy:.6f} eV")
print(f"  NN energy: {nn_energy:.6f} eV")
print(f"  Electrostatic: {electrostatic_energy:.6f} eV")
print(f"  Dispersion: {dispersion_energy:.6f} eV")
print(f"  ZBL repulsion: {zbl_repulsion:.6f} eV")
print(f"Partial charges: {partial_charges}")
print(f"Polarizability volume ratios: {a0_ratios}")
```

> **Note:** All energy-related observables (`nn_energy`, `electrostatic_energy`, `dispersion_energy`, `zbl_repulsion`, `electrostatic_energy_undamped`, `dispersion_energy_undamped`) are returned as **per-atom arrays**. Sum over atoms to obtain the total contribution: `np.sum(aux['nn_energy'])`.

**Available observables:**
| Observable | Description | Shape |
|------------|-------------|-------|
| `nn_energy` | Neural network energy contribution | `(n_atoms,)` |
| `electrostatic_energy` | Damped electrostatic energy | `(n_atoms,)` |
| `dispersion_energy` | Damped dispersion energy | `(n_atoms,)` |
| `zbl_repulsion` | Short-range repulsion (NLH for so3lr-s/m/l, ZBL for legacy `so3lr`) | `(n_atoms,)` |
| `nlh_repulsion` | NLH short-range repulsion (so3lr-s/m/l; alias of `zbl_repulsion`) | `(n_atoms,)` |
| `partial_charges` | Atomic partial charges | `(n_atoms,)` |
| `a0_ratios` | Polarizability volume ratios (`so3lr-s/m/l`) | `(n_atoms,)` |
| `hirshfeld_ratios` | Hirshfeld volume ratios (legacy `so3lr`) | `(n_atoms,)` |
| `electrostatic_energy_undamped` | Undamped electrostatic energy | `(n_atoms,)` |
| `dispersion_energy_undamped` | Undamped dispersion energy | `(n_atoms,)` |
| `x` | Final-layer per-atom embeddings | `(n_atoms, features)` |

> **Note:** Every observable in the table above is also reachable from the CLI via `so3lr eval --targets <name>` — the names are identical. The only special case is `x`, which is written to a separate `<output>_x.npz` (see the *Atom embeddings* example above) rather than the extxyz file.

> **Note:** The `zbl_repulsion` key returns the model's short-range repulsion energy. For `so3lr-s/m/l` this is a parameter-free NLH repulsion (also available under the clearer alias `nlh_repulsion`); for the legacy `so3lr` model it is the original ZBL repulsion. The `zbl_repulsion` key is kept for backward compatibility.

### Observable Gradients

To compute gradients of observables with respect to atomic positions, append `_grad` to the observable name and specify `output_atom_indices`:

```python
calc = So3lrCalculator(
    model="so3lr-m-2606",
    lr_cutoff=12.0,
    dtype=np.float64,
    has_aux=True,
    observables=['nn_energy_grad', 'partial_charges_jac'],
    output_atom_indices=[0, 1, 2]  # Compute gradients for these atoms
)

atoms.calc = calc
_ = atoms.get_potential_energy()

aux = calc.results['aux']

# Gradient of scalar observable (energy)
# Returns dict: {atom_idx: (value, gradient)}
nn_grad_dict = aux['nn_energy_grad']
for idx in [0, 1, 2]:
    value, grad = nn_grad_dict[idx]
    # value: scalar (atomic energy)
    # grad: shape (3,) - gradient w.r.t. atom position [x, y, z]
    print(f"Atom {idx}: energy={value:.6f}, grad={grad}")

# Jacobian of vector observable (partial charges)
# Returns dict: {atom_idx: (value, jacobian)}
charges_jac_dict = aux['partial_charges_jac']
for idx in [0, 1, 2]:
    value, jac = charges_jac_dict[idx]
    # value: scalar - partial charge for atom idx
    # jac: shape (n_atoms, 3) - d(all charges)/d(atom_idx position)
    print(f"Atom {idx}: charge={value:.4f}, Jacobian shape={jac.shape}")
```

**Gradient/Jacobian suffixes:**
- `_grad`: For scalar observables (energies). Returns gradient of shape `(3,)`.
- `_jac`: For vector observables (charges and per-atom ratio heads). Returns Jacobian of shape `(n_atoms, 3)`.

> **Note:** The `_grad` and `_jac` suffixes are only supported through the ASE calculator (`So3lrCalculator`) and are **not currently supported** via the `so3lr eval` CLI or `make_so3lr`.

### Hessian Calculation

The Hessian matrix (second derivatives of energy with respect to positions) can be computed for vibrational analysis, transition state searches, or other applications:

```python
calc = So3lrCalculator(
    model="so3lr-m-2606",
    lr_cutoff=12.0,
    dtype=np.float64,
    calculate_hessian=True
)

atoms.calc = calc
_ = atoms.get_potential_energy()

hessian = calc.results['hessian']
# Shape: (n_atoms, 3, n_atoms, 3)
# hessian[i, a, j, b] = d²E / (dR_i^a dR_j^b)

print(f"Hessian shape: {hessian.shape}")

# Reshape to 2D for eigenvalue analysis
n_atoms = len(atoms)
hessian_2d = hessian.reshape(n_atoms * 3, n_atoms * 3)
eigenvalues = np.linalg.eigvalsh(hessian_2d)
print(f"Hessian eigenvalues: {eigenvalues}")
```

**Subsystem Hessians:** For large systems, compute the Hessian only for a subset of atoms:

```python
# Compute Hessian block only for atoms 0-9 (must be contiguous)
calc = So3lrCalculator(
    model="so3lr-m-2606",
    lr_cutoff=12.0,
    dtype=np.float64,
    calculate_hessian=True,
    output_atom_indices=list(range(10))
)

atoms.calc = calc
_ = atoms.get_potential_energy()

hessian = calc.results['hessian']
# Shape: (10, 3, 10, 3) - subset x subset block of the full Hessian
```

> **Note:** `calculate_hessian` cannot be used simultaneously with `calculate_stress` or observable gradients.

## JAX MD
Large-scale simulations can be performed via jax-md which is a molecular dynamics library optimized for GPUs. Here we 
give a small example for a structure in vacuum. For realistic simulations, use CLI or take a look at the `./examples/` folder.
```python
import jax
import jax.numpy as jnp
import numpy as np

from ase import Atoms
from jax_md import space
from jax_md import quantity

from so3lr import to_jax_md
from so3lr import So3lrPotential

# Create a simple molecule
atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)])
assert np.asarray(
        atoms.get_pbc()
    ).all().item() is False, "Readme example assumes no box. See `examples/` folder for simulations in box."

positions = jnp.array(atoms.get_positions())
atomic_numbers = jnp.array(atoms.get_atomic_numbers()) 

# Assume there is no box
box = None
displacement, shift = space.free()

# Initialize SO3LR with JAX-MD
neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
    potential=So3lrPotential(model="so3lr-m-2606"),
    displacement_or_metric=displacement,
    box_size=box,
    species=atomic_numbers,
    capacity_multiplier=1.25,
    buffer_size_multiplier_sr=1.25,
    buffer_size_multiplier_lr=1.25,
    minimum_cell_size_multiplier_sr=1.0,
    disable_cell_list=True,
    fractional_coordinates=False
)

# JIT-compile energy and force functions
energy_fn = jax.jit(energy_fn)
force_fn = jax.jit(quantity.force(energy_fn))

# Initialize the short and long-range neighbor lists
nbrs = neighbor_fn.allocate(
    positions, 
    box=box
)
nbrs_lr = neighbor_fn_lr.allocate(
    positions, 
    box=box
)

# Calculate energy and forces
energy = energy_fn(
    positions, 
    neighbor=nbrs.idx,
    neighbor_lr=nbrs_lr.idx,
    box=box
)
forces = force_fn(
    positions, 
    neighbor=nbrs.idx,
    neighbor_lr=nbrs_lr.idx,
    box=box
)

print('Energy and forces in JAX-MD')
print('Energy = ', np.array(energy))
print('Forces = ', np.array(forces))
```

## Potential energy function
To obtain a potential energy function which is not specifically tailored for `jax-md` we provide a convenience 
interface. You can do  
```python
from so3lr import So3lrPotential
from so3lr import Graph

graph = Graph(...)

so3lr_potential = So3lrPotential(model="so3lr-m-2606")

energy = so3lr_potential(graph)
```
The `Graph` object is a `collections.namedtuple` which abstracts the molecule as a graph, common practice in the 
context of message passing neural networks. The `So3lrPotential` is a pure `python` function which takes a graph as 
an input and returns a potential energy. It is compatible with common `jax` transformations as `jax.jit`, `jax.vmap`, 
`jax.grad`, `...`. Its use targets developers, interested in integrating SO3LR into their own MD code base. From a 
high-level perspective, all that needs to be done is to define some function `system_to_graph` which transforms 
whatever input structure one has to a `Graph` object. When passed to `so3lr_potential` one gets the potential energy of 
the system.

## Long-Range Electrostatics (Ewald/PME/P3M)

For periodic systems, SO3LR supports long-range electrostatic calculations using three different k-space methods powered by [jax-pme](https://github.com/lab-cosmo/jax-pme/) (a JAX port of [torch-pme](https://github.com/lab-cosmo/torch-pme/)):

- **Ewald summation** (`ewald`): Classical Ewald method, exact but O(N²) scaling
- **Particle Mesh Ewald** (`pme`): FFT-accelerated method with O(N log N) scaling
- **Particle-Particle Particle-Mesh** (`p3m`): Similar to PME with improved accuracy for the same grid size

### When to Use K-Space Electrostatics

K-space electrostatics should be used for:
- **Periodic systems** (bulk liquids, crystals, interfaces)
- **Charged systems** where long-range Coulomb interactions are important
- Systems where electrostatic cutoffs would introduce artifacts

For **gas-phase/molecular systems**, use `lr_cutoff=100` instead (no k-space needed). This matches the gas-phase recommendation used throughout the eval examples above.

### ASE Calculator with K-Space Electrostatics

```python
from jax import config

# Enable this before importing SO3LR whenever dtype=np.float64 is requested.
config.update("jax_enable_x64", True)

import numpy as np
from ase import Atoms
from ase.io import read
from so3lr import So3lrCalculator
from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald

# Load a periodic water box
atoms = read('water_box.xyz')
cell = np.array(atoms.get_cell())

# Set up k-space parameters
smearing = 4.0  # Ewald smearing parameter in Angstrom

# For PME/P3M: use mesh grid
k_spacing = np.array([smearing / 8])  # Grid spacing
k_grid = get_kgrid_mesh(cell, k_spacing)

# For Ewald: use ewald grid instead
# k_spacing = np.array([smearing / 2])
# k_grid = get_kgrid_ewald(cell, k_spacing)

# Store k-space parameters in atoms.info
atoms.info['k_grid'] = k_grid
atoms.info['k_smearing'] = np.array([smearing])

# Create calculator with PME
calc = So3lrCalculator(
    model="so3lr-m-2606",
    lr_cutoff=12.0,
    dtype=np.float64,
    coulomb_kspace_method='pme',  # Options: 'ewald', 'pme', 'p3m'
)

atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"Energy: {energy:.6f} eV")
```

### Extracting K-Space Energy Components

```python
calc = So3lrCalculator(
    model="so3lr-m-2606",
    lr_cutoff=12.0,
    dtype=np.float64,
    coulomb_kspace_method='pme',
    has_aux=True,
    observables=[
        'electrostatic_energy',        # Short-range (real-space) contribution
        'electrostatic_energy_kspace', # Long-range (k-space) contribution
        'partial_charges'
    ]
)

atoms.calc = calc
energy = atoms.get_potential_energy()

aux = calc.results['aux']
sr_energy = np.sum(aux['electrostatic_energy'])
kspace_energy = np.sum(aux['electrostatic_energy_kspace'])
charges = aux['partial_charges']

print(f"Short-range electrostatic: {sr_energy:.6f} eV")
print(f"K-space electrostatic: {kspace_energy:.6f} eV")
print(f"Total charge: {np.sum(charges):.6f}")
```

### CLI with K-Space Electrostatics

For MD simulations via the CLI:

```shell script
# NVT with PME electrostatics
so3lr nvt --input water_box.xyz --temperature 300 --dt 0.5 \
    --md-cycles 100 --md-steps 100 \
    --kspace-electrostatics pme \
    --kspace-smearing 4.0 \
    --kspace-spacing 0.5

# Available options:
# --kspace-electrostatics: 'ewald', 'pme', or 'p3m'
# --kspace-smearing: Ewald smearing parameter (Angstrom)
# --kspace-spacing: K-space grid spacing (Angstrom)
```

### Method Comparison

| Method | Scaling | Accuracy | Best For |
|--------|---------|----------|----------|
| `ewald` | O(N²) | Exact | Small systems (<1000 atoms) |
| `pme` | O(N log N) | High | Large systems, production MD |
| `p3m` | O(N log N) | Higher | When accuracy is critical |

### Tuning K-Space Parameters

The `k_smearing` parameter controls the split between real-space and k-space:
- **Larger smearing** → More work in k-space, less in real-space
- **Smaller smearing** → More work in real-space, less in k-space
- Typical values: 3.0 - 5.0 Å

The `k_spacing` controls the k-space grid density:
- **Smaller spacing** → Finer grid, higher accuracy, more computation
- For PME/P3M: `k_spacing ≈ smearing / 8` is a good starting point
- For Ewald: `k_spacing ≈ smearing / 2` is typical

#### Automatic tuning with `so3lr tune-ewald`

Rather than choosing `smearing`/`spacing` by hand, the `tune-ewald` command grid-searches
real-space cutoff, smearing, and grid resolution for a given structure and reports the
parameter set that reaches a target force accuracy at the lowest cost:

```shell script
# Tune PME parameters for a periodic box (use float64 for the accuracy estimate)
so3lr tune-ewald --input water_box.xyz \
    --kspace-electrostatics pme \
    --lr-cutoff 12.0 \
    --precision float64
```

- `--kspace-electrostatics`: method to tune — `ewald`, `pme`, or `p3m`.
- `--lr-cutoff`: long-range cutoff (Å); used as the upper bound of the cutoff search.
- `--precision`: run the tuning in `float64` so the error bounds are meaningful.
- Other model flags (`--model`, `--total-charge`, `--dispersion-damping`, buffer sizes)
  are shared with the MD commands.

Results (including per-candidate error/timing bounds and the recommended
`smearing`/`spacing`/`interp-nodes`) are written to the log file, and the command
prints a ready-to-run MD invocation plus the equivalent settings-YAML keys.

Plug the recommended values into your MD run either on the CLI
(`--kspace-smearing` / `--kspace-spacing` / `--kspace-interp-nodes`) or, for a
`so3lr --settings settings.yaml` run, by adding the matching keys to the YAML:

```yaml
# settings.yaml
lr_cutoff: 12.0
kspace_electrostatics: pme      # ewald / pme / p3m
kspace_smearing: 4.0            # recommended value from tune-ewald
kspace_spacing: 0.5            # recommended value from tune-ewald
kspace_interp_nodes: 4         # for pme / p3m
```

## Enhanced Sampling with PySAGES

SO3LR integrates with [PySAGES](https://github.com/SSAGESLabs/PySAGES), a JAX-based enhanced sampling library, to enable biased molecular dynamics simulations with collective variables (CVs). This allows free energy calculations and enhanced exploration of configuration space directly within SO3LR's JAX-MD engine.

### Installation

PySAGES is an optional dependency. To install SO3LR with PySAGES support:

```shell script
cd so3lr
uv sync --locked --extra pysages
```

SO3LR runs normally without PySAGES installed. The enhanced sampling features are only activated when `do_enhanced_sampling: true` is set in the settings file.

### Supported Ensembles

Enhanced sampling works in any ensemble that SO3LR's MD engine supports — **NVT (Nose-Hoover), NPT, and NVE** — *for the non-adaptive methods*. The adaptive methods that learn a bias on-the-fly (`abf`, `spectralabf`, `metad`) currently require **NVT only**, because they assume a canonical sampling distribution when accumulating the free-energy estimator. `harmonic` (umbrella sampling / string method) and `unbiased` (CV tracking only) impose no such constraint and run in NVT / NPT / NVE alike. `nvt-langevin` is not yet wired to PySAGES.

### Supported Methods

| Method | Description |
|--------|-------------|
| `unbiased` | Unbiased MD with CV tracking (useful for baseline comparisons) |
| `abf` | Adaptive Biasing Force |
| `spectralabf` | Spectral ABF (with spectral decomposition) |
| `metad` | Well-tempered Metadynamics |
| `harmonic` | Harmonic bias potential (for umbrella sampling, string method, etc.) |

### Supported Collective Variables

| CV Type | Description |
|---------|-------------|
| `distance` | Distance between two atom groups |
| `differenceofdistances` | Difference of two pairwise distances |
| `angle` | Bond angle (3 atoms) |
| `dihedral` | Dihedral angle (4 atoms) |
| `gyrrad` | Radius of gyration |
| `princmom` | Principal moment of inertia |
| `asphericity` | Asphericity |
| `acylindricity` | Acylindricity |
| `shapeanisotropy` | Shape anisotropy |
| `coordinationnumber` | Coordination number with element-specific filtering |

### Usage

Enhanced sampling is configured via two files: the main MD settings file and a separate PySAGES input file that defines the CVs and sampling method.

**MD settings file** (`md_settings.yaml`):

```yaml
ensemble: "nvt"
input_file: "geometry.xyz"
output_file: "trajectory.xyz"
md_dt: 0.5
md_T: 300
md_cycles: 100
md_steps: 100

# Enhanced sampling
do_enhanced_sampling: true
pysages_input: "./pysages_input.yaml"
restart_pysages_save_path: "./restart_pysages.pkl"
```

**PySAGES input file** (`pysages_input.yaml`) — example with ABF:

```yaml
method: "abf"

cv:
  - type: "distance"
    grp1: [0]
    grp2: [1]
    grid_min: 1.0
    grid_max: 5.0
    grid_stride: 50
    periodic: false

# Optional: restraints to keep CVs within bounds
restraints:
  lower: [1.5]
  upper: [4.5]
  kl: [50.0]
  ku: [50.0]
```

**PySAGES input file** — example with harmonic bias:

```yaml
method: "harmonic"

cv:
  - type: "distance"
    grp1: [0]
    grp2: [1]
    grid_min: 1.0
    grid_max: 5.0
    grid_stride: 50
    periodic: false

method_args:
  kspring: [100.0]
  center: [3.0]
```

Then run:

```shell script
so3lr --settings md_settings.yaml
```

### Restarting Enhanced Sampling Runs

PySAGES state (bias potential, histograms, etc.) can be saved and restored independently of the SO3LR MD state:

> **Security warning:** PySAGES restart files are pickle-compatible and may
> execute arbitrary code when loaded. Only load a restart you trust and whose
> checksum you have verified. SO3LR warns at runtime but cannot make this
> third-party serialization format safe.

```yaml
# First run — save both states
restart_save_path: "./restart_so3lr.npz"
restart_pysages_save_path: "./restart_pysages.pkl"

# Restart — load both states and continue
restart_load_path: "./restart_so3lr.npz"
restart_pysages_load_path: "./restart_pysages.pkl"
restart_pysages_save_path: "./restart_pysages_2.pkl"
```

### Notes

- For periodic systems, Cartesian coordinates are automatically enforced when enhanced sampling is enabled (PySAGES requires Cartesian, not fractional, coordinates).
- When using multiple CVs with mixed periodicity (e.g., a dihedral and a distance), the grid is treated as non-periodic. This is a limitation of the PySAGES grid implementation.

## Coming soon!

- General robust loss
- Multi-head fine-tuning

(The previously listed *Fast TFDS dataloader* now ships — install via `uv sync --locked --extra tfds`; see the *Optional extras* note in the Installation section.)

If you would like to see new features, add your own functionality, or share ideas, please reach out or open an issue. We are always happy to collaborate and hear your thoughts!

The CLI, repository, and model are still evolving, and your feedback is invaluable. We would appreciate if you report any errors or inconsistencies.

## Datasets
The quantum mechanical datasets used for training and testing SO3LR are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14779793.svg)](https://doi.org/10.5281/zenodo.14779793)

## Citation
Machine-readable software and preferred-paper metadata are available in
[`CITATION.cff`](./CITATION.cff). If you use parts of the code, cite the exact
software version and canonical dated checkpoint together with the relevant
papers below. The SO3LR-2 citation will be added only after its bibliographic
record is public; no DOI is claimed by this repository yet.

```text
SO3LR software, version 0.2.0, model so3lr-m-2606
https://github.com/general-molecular-simulations/so3lr
```

```
@article{kabylda2025molecular,
  title={Molecular Simulations with a Pretrained Neural Network and Universal Pairwise Force Fields},
  author={Kabylda, Adil and Frank, J Thorben and Su{\'a}rez-Dou, Sergio and Khabibrakhmanov, Almaz and Medrano Sandonas, Leonardo and Unke, Oliver T and Chmiela, Stefan and M{\"u}ller, Klaus-Robert and Tkatchenko, Alexandre},
  journal={Journal of the American Chemical Society},
  volume={147},
  number={37},
  pages={33723},
  year={2025},
  doi={10.1021/jacs.5c09558}
}

@article{frank2024euclidean,
  title={A Euclidean transformer for fast and stable machine learned force fields},
  author={Frank, Thorben and Unke, Oliver and M{\"u}ller, Klaus-Robert and Chmiela, Stefan},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={6539},
  year={2024},
  doi={10.1038/s41467-024-50620-6}
}
```

## License and third-party code

Original SO3LR code is distributed under the MIT License. Portions derived
from Google Research and JAX MD remain under Apache-2.0, and the Ewald tuning
code derived from torch-pme remains under BSD-3-Clause. See
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) and the accompanying
license files for the exact scope and notices. Checkpoint and dataset licenses
must be confirmed separately before the SO3LR-2 release. An archive-wide license
expression is deliberately left unspecified until that review; the source
licenses do not relicense checkpoints, datasets, or scientific reference data.
