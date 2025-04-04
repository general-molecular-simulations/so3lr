# SO3LR-tools

## Settings

### General
- **`initial_geometry`** (`str`):  
  Path to the initial geometry file.
- **`model_path`** (`str`):  
  Path to the model directory.
- **`precision`** (`str`, default: `'float32'`):  
  Numerical precision.

### Cutoffs and Buffers
- **`lr_cutoff`** (`float`, default: `12.0`):  
  Long-range cutoff.
- **`dispersion_energy_cutoff_lr_damping`** (`float`, default: `2.0`):  
  Damping for long-range dispersion.
- **`buffer_size_multiplier_sr`** (`float`, default: `1.25`):  
  Buffer multiplier for short-range.
- **`buffer_size_multiplier_lr`** (`float`, default: `1.25`):  
  Buffer multiplier for long-range.
- **`hdf5_buffer_size`** (`int`, default: `5`):  
  Buffer size for HDF5 writing.

### File Paths
- **`trajectory_hdf5_file`** (`str`, default: `'trajectory.hdf5'`):  
  File name for trajectory output.
- **`restart_save_path`** (`str`, default: `None`):  
  Path to save restart file.
- **`restart_load_path`** (`str`, default: `None`):
  Path to load restart file.
- **`save_exist_ok`** (`bool`, default: `False`):
  Whether to overwrite existing files.

### Molecular Dynamics (MD)
- **`md_dt`** (`float`, default: 0.0005):  
  MD timestep in picoseconds.
- **`md_T`** (`float`, default: 300.0):  
  Temperature in Kelvin.
- **`md_cycles`** (`int`, default: 100):  
  Number of MD cycles.
- **`md_steps`** (`int`, default: 100):  
  MD steps per cycle.

### Nose-Hoover Chain (NVT)
- **`nhc_chain_length`** (`int`, default: `3`):  
  Length of Nose-Hoover chain.
- **`nhc_steps`** (`int`, default: `2`):  
  Integration steps per MD step.
- **`nhc_thermo`** (`float`, default: `100`):  
  Thermostat timescale.
- **`nhc_tau`** (`float`, default: `md_dt * nhc_thermo`):  
  Thermostat coupling constant.

### Nose-Hoover Chain (NPT)
- **`md_P`** (`float`):  
  Pressure in atm.
- **`nhc_baro`** (`float`, default: `1000`):  
  Barostat timescale.
- **`nhc_sy_steps`** (`int`, default: `3`):  
  Number of Suzuki-Yoshida steps.
- **`nhc_npt_tau`** (`float`, default: `md_dt * nhc_baro`):  
  Barostat coupling constant.

### Miscellaneous
- **`seed`** (`int`, default: `0`):  
  Random seed.
- **`total_charge`** (`int`, default: `0`):  
  Total system charge.

### Minimization
- **`min_cycles`** (`int`, default: `10`):  
  Number of minimization cycles.
- **`min_steps`** (`int`, default: `10`):  
  Steps per minimization cycle.
- **`min_start_dt`** (`float`, default: `0.05`):  
  Initial minimization timestep.
- **`min_max_dt`** (`float`, default: `0.1`):  
  Maximum minimization timestep.
- **`min_n_min`** (`int`, default: `2`):  
  Number of minimizers to average.
