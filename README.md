![Logo](./sup-gems-logo.png)
### Installation
First clone the repository and install by doing 
```shell script
git clone https://github.com/thorben-frank/sup_gems.git
cd sup_gems
pip install .
```
### ASE Calculator
To load an Atomistic Simulation Environment (ASE) calculator powered by
SUP-GEMS you can do 
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
TBD
```
