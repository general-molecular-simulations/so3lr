![Logo](./sup-gems-logo.png)
### Installation
First clone the repository and install by doing 
```shell script
git clone https://github.com/thorben-frank/sup_gems.git
cd sup_gems
pip install .
```
### Atomic Simulation Environment
To get an Atomic Simulation Environment (ASE) calculator with energies and forces predicted
from SUP-GEMS just do 
```python
from sup_gems import SupGemsCalculator
from ase import Atoms

atoms = Atoms(...)
calc = SupGemsCalculator()
atoms.calc = calc

atoms.get_forces()
```
### JAX MD
```python
from sup_gems import to_jax_md

neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
    potential=mlff_potential,
    displacement_or_metric=displacement,
    box_size=box,
    species=species,
    capacity_multiplier=capacity_multiplier,
    buffer_size_multiplier_sr=buffer_size_multiplier_sr,
    buffer_size_multiplier_lr=buffer_size_multiplier_lr,
    minimum_cell_size_multiplier_sr=minimum_cell_size_multiplier_sr
)
```
