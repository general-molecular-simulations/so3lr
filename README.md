![workflow-test-ci](https://github.com/thorben-frank/solar/actions/workflows/CI.yml/badge.svg)
[![examples-link](https://img.shields.io/badge/example-notebooks-F37726)](./examples)
[![preprint-link](https://img.shields.io/badge/paper-arxiv.org-B31B1B)](https://arxiv.org/)
![Logo](./logo.png)
## Installation
First clone the repository and install by doing 
```shell script
git clone https://github.com/thorben-frank/solar.git
cd solar
pip install .
```
## Atomic Simulation Environment
To get an Atomic Simulation Environment (ASE) calculator with energies and forces predicted
from SO3LR (pronounced "Solar") just do 
```python
from solar import SolarCalculator
from ase import Atoms

atoms = Atoms(...)
calc = SolarCalculator()
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print('Energy and forces in ASE')
print('Energy = ', energy)
print('Forces = ', forces)

```
## JAX MD
Large scale simulations can be performed via Jax-MD which is a molecular dynamics library optimized for GPUs. Here we 
give a small example for a structure in vacuum. For realistic simulations with periodic water boxes take a look at the 
`./examples/` folder.
```python
import jax
import jax.numpy as jnp
import numpy as np

from ase import Atoms
from jax_md import space
from jax_md import quantity

from solar import to_jax_md
from solar import SolarPotential

atoms = Atoms(...)
assert np.asarray(
        atoms.get_pbc()
    ).all().item() is False, "Readme example assumes no box. See `examples/` folder for simulations in box."

positions = jnp.array(atoms.get_positions())
atomic_numbers = jnp.array(atoms.get_atomic_numbers()) 

# We assume there is no box.
box = None
displacement, shift = space.free()

neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
    potential=SolarPotential(),
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

# Energy and force functions.
energy_fn = jax.jit(energy_fn)
force_fn = jax.jit(quantity.force(energy_fn))

# Initialize the short and long-range neighbor lists.
nbrs = neighbor_fn.allocate(
    positions, 
    box=box
)
nbrs_lr = neighbor_fn_lr.allocate(
    positions, 
    box=box
)
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
from solar import SolarPotential
from solar import Graph

graph = Graph(...)

solar_potential = SolarPotential()

energy = solar_potential(graph)
```
The `Graph` object is a `collections.namedtuple` which abstracts the molecule as a graph common practice in the 
context of message passing neural networks. The `SolarPotential` is a pure `python` function which takes a graph as 
an input and returns a potential energy. It is compatible with common `jax` transformations as `jax.jit`, `jax.vmap`, 
`jax.grad`, `...`. Its use targets developers, interested in integrating SO3LR into their own MD code base. From a 
high-level perspective, all that needs to be done is to define some function `system_to_graph` which transforms 
whatever input structure one has to a `Graph` object. Passed to `solar_potential` one gets the potential energy of 
the system.