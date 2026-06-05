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
SO3LR can be either used with CPU or with GPU. If you want to use SO3LR on GPU, you have to install the 
corresponding JAX installation via 
```shell script
# SO3LR on GPU
pip install --upgrade pip
pip install "jax[cuda12]"
```
> **Note**: SO3LR runs significantly faster on GPU, making it the preferred choice for large-scale simulations. More details about JAX installation can be found [here](https://jax.readthedocs.io/en/latest/installation.html). Also, we recommend installing with [uv](https://docs.astral.sh/uv/), it's magical.

If you want to use SO3LR on CPU, e.g. for testing on your local machine which does not have a GPU, you can do
```shell script
# SO3LR on CPU
pip install --upgrade pip
pip install jax
```
Next clone the repository and install by doing 
```shell script
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
pip install .
```

## Models

SO3LR ships in three sizes. All are bundled inside the package, so you select one **by short name** — no download or path needed. They trade off speed against accuracy:

| Model | Layers | Features | Heads | Cutoff (Å) | Best for |
|-------|--------|----------|-------|-----------|----------|
| `so3lr-s` | 2 | 128 | 4 | 4.5 | Fastest — quick tests and very large systems |
| `so3lr-m` *(default)* | 3 | 128 | 4 | 4.5 | Balanced — general-purpose use |
| `so3lr-l` | 3 | 256 | 8 | 5.0 | Most accurate — highest fidelity, largest model |
| `so3lr-o` | 3 | 128 | 4 | 4.5 | Original SO3LR v1 — for reproducing the [JACS paper](https://doi.org/10.1021/jacs.5c09558) |

If you do not specify a model, **`so3lr-m`** is used. `so3lr-o` (also reachable as `so3lr`) is the original v1 model kept for backward compatibility.

### Selecting a model

The same names work across every interface:

```python
# Python / ASE
from so3lr import So3lrCalculator
calc = So3lrCalculator(model="so3lr-l")          # also "so3lr-s" / "so3lr-m"
```

```shell script
# CLI (works with opt / nvt / npt / nve / eval)
so3lr nvt --input geometry.xyz --model so3lr-l
```

```yaml
# In an MD settings YAML (model: is an alias of model_path:)
model: so3lr-l
```

Custom and fine-tuned models still work — just pass a path instead of a name:

```python
So3lrCalculator(model="/path/to/finetuned")      # or  --model /path/to/finetuned  on the CLI
```

> **Backward compatibility:** The no-arg default is now `so3lr-m`. To reproduce the **original SO3LR v1** model from the [JACS paper](https://doi.org/10.1021/jacs.5c09558), pass `model="so3lr"` (alias `"so3lr-o"`).

## Command Line Interface (CLI)

SO3LR provides a unified command-line interface that leverages the performance of JAX-MD under the hood (detailed in the JAX-MD section below). This allows you to perform geometry optimizations and MD simulations with simple commands. The CLI supports several key functionalities:

- `so3lr opt`: Geometry optimization
- `so3lr nvt`: NVT molecular dynamics
- `so3lr npt`: NPT molecular dynamics
- `so3lr nve`: NVE molecular dynamics
- `so3lr eval`: Model evaluation on a dataset
- `so3lr finetune`: Fine-tune the model on custom datasets
- `so3lr train`: Train a SO3LR/SO3krates model from scratch

Each subcommand has its own set of options and can be run with `--help` to see all available parameters.

```shell script
so3lr opt --help
```

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
#1. First save a checkpoint (updated every `save_buffer` cycles) during your simulation:
so3lr nvt --input geometry.xyz --output trajectory.xyz --temperature 300 --md-cycles 100 --restart-save checkpoint.npz

#2. Then restart from this checkpoint to continue the simulation:
so3lr nvt --input geometry.xyz --output trajectory.xyz --temperature 300 --md-cycles 100 --restart-load checkpoint.npz --restart-save checkpoint_new.npz
```

The restart will continue the simulation from the exact state where it was saved, preserving atom positions, velocities, thermostat/barostat state, and simulation timestep.

### Freezing atoms during MD

A subset of atoms can be held fixed during any MD run (`nve` / `nvt` / `npt`). Frozen atoms keep their reference positions while their momenta and forces are zeroed every step, so the rest of the system evolves around them (useful e.g. for holding a surface, a binding pocket, or a substrate rigid). Constraints are configured through the MD **settings YAML** (run with `so3lr --settings settings.yaml`):

```yaml
# settings.yaml
ensemble: nvt
input_file: geometry.xyz
model: so3lr-m
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

Evaluate the SO3LR model on a dataset:

```shell script
so3lr eval --datafile dataset.extxyz --batch-size 1 --lr-cutoff 100 --save-to predictions.extxyz
```

Add `--model so3lr-s` (or `so3lr-l`) to evaluate a different bundled model; without it, `so3lr-m` is used.

The input can be any file that is digestible by [`ase.io.iread`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.iread).

> [!IMPORTANT]
> SO3LR was not trained on energies, so only relative energies are meaningful. Labels are assumed to be in `eV` and `Ångström`. For gas-phase simulations, we suggest using `--lr-cutoff 100`.

The command will collect and print metrics on the dataset and save the predictions to the specified output file. The predicted properties are `energy`, `forces`, `dipole_vec` and `hirshfeld_ratios`. Energy and forces are assumed to be present in the datafile, while dipole vectors and Hirshfeld ratios are optional. If they are not present in the data, the metrics will simply be `NaN`.

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
  rate by `reduce_lr_factor` (default `0.5`) whenever the validation loss stalls
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
```shell script
mols = read('sapt10k_eval.xyz', index=':')
energy = [a.info['energy_so3lr'] for a in mols]
bind_energy = [energy[i]-energy[i+1] for i in range(0, len(energy), 2)]
```

## Atomic Simulation Environment
To obtain an Atomic Simulation Environment (ASE) calculator that predicts energies and forces using SO3LR, simply run: 
```python
import numpy as np

from so3lr import So3lrCalculator
from ase import Atoms

atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)])
atoms.info['charge'] = 0.0

calc = So3lrCalculator(
    model="so3lr-m",         # bundled model; also "so3lr-s" / "so3lr-l" or a path. Default: so3lr-m
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
import numpy as np
from so3lr import So3lrCalculator
from ase import Atoms

atoms = Atoms('H2O', positions=[(0, 0, 0), (0.96, 0, 0), (0.24, 0.93, 0)])
atoms.info['charge'] = 0.0

calc = So3lrCalculator(
    lr_cutoff=12.0,
    dtype=np.float64,
    has_aux=True,
    observables=[
        'nn_energy',              # Neural network energy
        'electrostatic_energy',   # Electrostatic energy
        'dispersion_energy',      # Dispersion energy
        'zbl_repulsion',          # ZBL repulsion energy
        'partial_charges',        # Atomic partial charges
        'hirshfeld_ratios'        # Hirshfeld volume ratios
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
hirshfeld_ratios = aux['hirshfeld_ratios']    # shape: (n_atoms,)

print(f"Total energy: {energy:.6f} eV")
print(f"  NN energy: {nn_energy:.6f} eV")
print(f"  Electrostatic: {electrostatic_energy:.6f} eV")
print(f"  Dispersion: {dispersion_energy:.6f} eV")
print(f"  ZBL repulsion: {zbl_repulsion:.6f} eV")
print(f"Partial charges: {partial_charges}")
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
| `hirshfeld_ratios` | Hirshfeld volume ratios | `(n_atoms,)` |
| `electrostatic_energy_undamped` | Undamped electrostatic energy | `(n_atoms,)` |
| `dispersion_energy_undamped` | Undamped dispersion energy | `(n_atoms,)` |
| `x` | Final-layer per-atom embeddings | `(n_atoms, features)` |

> **Note:** The `zbl_repulsion` key returns the model's short-range repulsion energy. For `so3lr-s/m/l` this is a parameter-free NLH repulsion (also available under the clearer alias `nlh_repulsion`); for the legacy `so3lr` model it is the original ZBL repulsion. The `zbl_repulsion` key is kept for backward compatibility.

### Observable Gradients

To compute gradients of observables with respect to atomic positions, append `_grad` to the observable name and specify `output_atom_indices`:

```python
calc = So3lrCalculator(
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
- `_jac`: For vector observables (charges, Hirshfeld ratios). Returns Jacobian of shape `(n_atoms, 3)`.

> **Note:** The `_grad` and `_jac` suffixes are only supported through the ASE calculator (`So3lrCalculator`) and are **not currently supported** via the `so3lr eval` CLI or `make_so3lr`.

### Hessian Calculation

The Hessian matrix (second derivatives of energy with respect to positions) can be computed for vibrational analysis, transition state searches, or other applications:

```python
calc = So3lrCalculator(
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
    potential=So3lrPotential(),
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

so3lr_potential = So3lrPotential()

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

For **gas-phase/molecular systems**, use `lr_cutoff=1000` instead (no k-space needed).

### ASE Calculator with K-Space Electrostatics

```python
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
pip install .[pysages]
```

SO3LR runs normally without PySAGES installed. The enhanced sampling features are only activated when `do_enhanced_sampling: true` is set in the settings file.

### Supported Ensembles

Enhanced sampling currently supports **NVT (Nose-Hoover thermostat)** only. NPT, NVE, and NVT-Langevin are not yet supported.

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

- Fast TFDS dataloader
- General robust loss
- Multi-head fine-tuning

If you would like to see new features, add your own functionality, or share ideas, please reach out or open an issue. We are always happy to collaborate and hear your thoughts!

The CLI, repository, and model are still evolving, and your feedback is invaluable. We would appreciate if you report any errors or inconsistencies.

## Datasets
The quantum mechanical datasets used for training and testing SO3LR are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14779793.svg)](https://doi.org/10.5281/zenodo.14779793)

## Citation
If you use parts of the code please cite
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
