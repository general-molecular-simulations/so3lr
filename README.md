![workflow-test-ci](https://github.com/general-molecular-simulation/so3lr/actions/workflows/CI.yml/badge.svg)
[![examples-link](https://img.shields.io/badge/example-notebooks-F37726)](./examples)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-molecular-simulations/so3lr/blob/main/examples/so3lr_colab_example.ipynb)
[![cite-link](https://img.shields.io/badge/how_to-cite-000000)](https://github.com/general-molecular-simulation/so3lr?tab=readme-ov-file#Citation)
[![preprint-link](https://img.shields.io/badge/paper-chemRxiv.org-A9A8AD)](https://doi.org/10.26434/chemrxiv-2024-bdfr0-v2)
[![data](https://zenodo.org/badge/DOI/10.5281/zenodo.14779793.svg)](https://doi.org/10.5281/zenodo.14779793)
![Logo](./logo.png)

## About
SO3LR - pronounced *Solar* - is a pretrained machine-learned force field for (bio)molecular simulations. It integrates the fast and stable SO3krates neural network for semi-local interactions with universal pairwise force fields designed for short-range repulsion, long-range electrostatics, and dispersion interactions.

## Quick Start
Try SO3LR without any local installation using our [Colab notebook](https://colab.research.google.com/github/general-molecular-simulations/so3lr/blob/main/examples/so3lr_colab_example.ipynb)!

## Installation
SO3RL can be either used with CPU or with GPU. If you want to use SO3LR on GPU, you have to install the 
corresponding JAX installation via 
```shell script
# SO3LR on GPU
pip install --upgrade pip
pip install "jax[cuda12]==0.5.3"
```
> **Note**: SO3LR runs significantly faster on GPU, making it the preferred choice for large-scale simulations. More 
details about JAX installation can be found [here](https://jax.readthedocs.io/en/latest/installation.html).

If you want to use SO3LR on CPU, e.g. for testing on your local machine which does not have a GPU, you can do
```shell script
# SO3LR on CPU
pip install --upgrade pip
pip install jax==0.5.3
```
Next clone the repository and install by doing 
```shell script
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
pip install .
```

## Command Line Interface (CLI)

SO3LR provides a unified command-line interface that leverages the performance of JAX-MD under the hood (detailed in the JAX-MD section below). This allows you to perform geometry optimizations and MD simulations with simple commands. The CLI supports several key functionalities:

- `so3lr opt`: Geometry optimization
- `so3lr nvt`: NVT molecular dynamics
- `so3lr npt`: NPT molecular dynamics
- `so3lr nve`: NVE molecular dynamics
- `so3lr eval`: Model evaluation on a dataset

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

Evaluate the SO3LR model on a dataset:

```shell script
so3lr eval --datafile dataset.extxyz --batch-size 1 --lr-cutoff 12.0 --save-to predictions.extxyz
```

The input can be any file that is digestible by [`ase.io.iread`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.iread).

The command will collect and print metrics on the dataset and save the predictions to the specified output file. The predicted properties are `energy`, `forces`, `dipole_vec` and `hirshfeld_ratios`. Energy and forces are assumed to be present in the datafile, while dipole vectors and Hirshfeld ratios are optional. If they are not present in the data, the metrics will simply be `NaN`.

> **Important**: SO3LR was not trained specifically on energies, so only relative energies are meaningful. Labels are assumed to be in `eV` and `Ångström`. 


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

The CLI, repository, and model are still developing. We would appreciate if you report any errors or incosistencies.

## Atomic Simulation Environment
To get an Atomic Simulation Environment (ASE) calculator with energies and forces predicted
from SO3LR just do 
```python
import numpy as np

from so3lr import So3lrCalculator
from ase import Atoms

atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)])
atoms.info['charge'] = 0.0

calc = So3lrCalculator(
    calculate_stress=False,
    dtype=np.float32
)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print('Energy and forces in ASE')
print('Energy = ', energy)
print('Forces = ', forces)

```
## JAX MD
Large scale simulations can be performed via jax-md which is a molecular dynamics library optimized for GPUs. Here we 
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
To obtain a potential energy function which is not specifally tailored for `jax-md` we provide a convenience 
interface. You can do  
```python
from so3lr import So3lrPotential
from so3lr import Graph

graph = Graph(...)

so3lr_potential = So3lrPotential()

energy = so3lr_potential(graph)
```
The `Graph` object is a `collections.namedtuple` which abstracts the molecule as a graph common practice in the 
context of message passing neural networks. The `So3lrPotential` is a pure `python` function which takes a graph as 
an input and returns a potential energy. It is compatible with common `jax` transformations as `jax.jit`, `jax.vmap`, 
`jax.grad`, `...`. Its use targets developers, interested in integrating SO3LR into their own MD code base. From a 
high-level perspective, all that needs to be done is to define some function `system_to_graph` which transforms 
whatever input structure one has to a `Graph` object. Passed to `so3lr_potential` one gets the potential energy of 
the system.
## Datasets
The quantum mechanical datasets used for training and testing SO3LR are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14779793.svg)](https://doi.org/10.5281/zenodo.14779793)

## Citation
If you use parts of the code please cite
```
@article{kabylda2024molecular,
  title={Molecular Simulations with a Pretrained Neural Network and Universal Pairwise Force Fields},
  author={Kabylda, A. and Frank, J. T. and Dou, S. S. and Khabibrakhmanov, A. and Sandonas, L. M.
          and Unke, O. T. and Chmiela, S. and M{\"u}ller, K.R. and Tkatchenko, A.},
  journal={ChemRxiv},
  year={2024},
  doi={10.26434/chemrxiv-2024-bdfr0-v2}
}

@article{frank2024euclidean,
  title={A Euclidean transformer for fast and stable machine learned force fields},
  author={Frank, Thorben and Unke, Oliver and M{\"u}ller, Klaus-Robert and Chmiela, Stefan},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={6539},
  year={2024}
}
```
