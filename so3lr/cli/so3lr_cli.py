"""Command line interfaces for SO3LR."""
import os
import sys
import time
import yaml
import click
import difflib
import logging
import warnings
import platform
import numpy as np

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from .. import __version__

# NOTE: heavy imports (jax, and the so3lr_eval/finetune/train/md/tune_ewald
# submodules, which pull in jax/flax/jax-md/glp/orbax) are deferred so that
# `so3lr --help` / `so3lr <cmd> --help` stay fast and do not trigger JAX
# initialization. The thin wrappers below import the real implementation lazily
# on first call, keeping every call site unchanged.
def setup_logger(*args, **kwargs):
    from .so3lr_md import setup_logger as _f
    return _f(*args, **kwargs)


def run(*args, **kwargs):
    from .so3lr_md import run as _f
    return _f(*args, **kwargs)


def perform_min(*args, **kwargs):
    from .so3lr_md import perform_min as _f
    return _f(*args, **kwargs)


def evaluate_so3lr_on(*args, **kwargs):
    from .so3lr_eval import evaluate_so3lr_on as _f
    return _f(*args, **kwargs)


def finetune_so3lr(*args, **kwargs):
    from .so3lr_finetune import finetune_so3lr as _f
    return _f(*args, **kwargs)


def train_so3lr(*args, **kwargs):
    from .so3lr_train import train_so3lr as _f
    return _f(*args, **kwargs)


def tune(*args, **kwargs):
    from .tune_ewald import tune as _f
    return _f(*args, **kwargs)

# Get logger
logger = logging.getLogger("SO3LR")

# ASCII art banner
SO3LR_ASCII = """
  ███████╗ ██████╗ ██████╗ ██╗     ██████╗ 
  ██╔════╝██╔═══██╗╚════██╗██║     ██╔══██╗
  ███████╗██║   ██║ █████╔╝██║     ██████╔╝
  ╚════██║██║   ██║ ╚═══██╗██║     ██╔══██╗
  ███████║╚██████╔╝██████╔╝███████╗██║  ██║
  ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
"""

# Ignore FutureWarnings or warnings that do not affect the model's performance
warnings.filterwarnings("ignore", message="scatter inputs have incompatible types")
warnings.filterwarnings("ignore", message="Explicitly requested dtype.*truncated")
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_hardware_info():
    """Detect and print information about the computing hardware being used.

    Checks if JAX is using GPU or CPU and prints relevant device properties.
    """
    import jax
    try:
        backend = jax.default_backend()
        devices = jax.local_devices(backend=backend)
    except RuntimeError as exc:
        logger.warning(f"Unable to query JAX hardware details: {exc}")
        logger.info("JAX default backend:       unavailable")
        logger.info("Number of devices:         unavailable")
        logger.info("Proceeding without device diagnostics.")
        logger.info("=" * 60)
        return

    # logger.info("Hardware information")
    logger.info(f"JAX default backend:       {backend}")
    logger.info(f"Number of devices:         {len(devices)}")

    if backend == "cpu":
        # Get CPU info
        logger.info(f"Processor:                 {platform.processor()}")
        logger.info(f"CPU cores:                 {os.cpu_count()}")

        # XLA CPU devices
        for i, device in enumerate(devices):
            logger.info(f"Device {i}:                    {device}")

    elif backend in ["gpu", "cuda", "rocm"]:
        # Running on GPU
        # Check all devices
        for i, device in enumerate(devices):
            # Get device memory
            try:
                mem_info = device.memory_stats()
                mem_available = mem_info.get('bytes_available', 'unknown')
                mem_total = mem_info.get('bytes_total', 'unknown')

                if mem_total != 'unknown' and mem_available != 'unknown':
                    mem_used = mem_total - mem_available
                    mem_used_gb = mem_used / (1024**3)
                    mem_total_gb = mem_total / (1024**3)
                    mem_str = f"{mem_used_gb:.2f}GB / {mem_total_gb:.2f}GB"
                else:
                    mem_str = "unknown"

                logger.info(f"Device {i}:                  {device}")
                logger.info(f"Memory usage:              {mem_str}")
            except:
                logger.info(f"Device {i}:                {device}")
                logger.info(
                    "Memory usage:               Unable to retrieve memory info")

    else:
        # Other backends like TPU
        logger.info(f"Running on {backend} devices:")
        for i, device in enumerate(devices):
            logger.info(f"Device {i}:                {device}")
    logger.info("=" * 60)


# Default settings
DEFAULT_PRECISION = 'float32'
DEFAULT_LR_CUTOFF = 12.0  # Angstrom
DEFAULT_SEED = 52
DEFAULT_TOTAL_CHARGE = 0
DEFAULT_SPIN = 0  # number of unpaired electrons (multiplicity - 1)
DEFAULT_DISPERSION_DAMPING = 2.0  # Angstrom
DEFAULT_BUFFER_MULTIPLIER = 1.25
DEFAULT_SAVE_BUFFER = 50
DEFAULT_OUTPUT_FORMAT = 'extxyz'
DEFAULT_TIMESTEP = 0.5  # femtoseconds
DEFAULT_TEMPERATURE = 300  # Kelvin
DEFAULT_PRESSURE = 1.0  # bar

# Nose-Hoover chain parameters
DEFAULT_MD_CYCLES = 100
DEFAULT_MD_STEPS_PER_CYCLE = 100
DEFAULT_NHC_CHAIN_LENGTH = 3
DEFAULT_NHC_INTEGRATION_STEPS = 2
DEFAULT_NHC_THERMO = 100.0  # thermostat damping parameter multiplier, i.e. Tdamp = dt * DEFAULT_NHC_THERMO
DEFAULT_NHC_BARO = 1000.0  # barostat damping parameter multiplier, i.e. Pdamp = dt * DEFAULT_NHC_BARO

# Langevin thermostat parameters
DEFAULT_LANGEVIN_REMOVECMMOTION = True
DEFAULT_LANGEVIN_THERMO = 1.0

# Geometry optimization parameters with FIRE
# More details here: https://jax-md.readthedocs.io/en/main/jax_md.minimize.html#jax_md.minimize.fire_descent
DEFAULT_MIN_CYCLES = 10
DEFAULT_MIN_STEPS = 10
DEFAULT_MIN_START_DT = 0.05
DEFAULT_MIN_MAX_DT = 0.1
DEFAULT_MIN_N_MIN = 2
DEFAULT_SUZUKI_YOSHIDA_STEPS = 3

# K-space electrostatics parameters
DEFAULT_KSPACE_SMEARING = 4.0       # Å
DEFAULT_KSPACE_SPACING = 2.0        # Å
DEFAULT_KSPACE_INTERP_NODES = 4

# Additional parameters
DEFAULT_OBSERVABLES = ()

# Help strings for CLI
FULL_HELP_STRING = """
Run simulations using SO3LR Machine Learned Force Field.

## Commands

so3lr opt [options]           Run geometry optimization
so3lr nvt [options]           Run NVT (constant volume and temperature) MD simulation
so3lr nvt-langevin [options]  Run NVT MD simulation with a Langevin thermostat
so3lr npt [options]           Run NPT (constant pressure and temperature) MD simulation
so3lr nve [options]           Run NVE (constant volume and energy) MD simulation
so3lr eval [options]          Run evaluation (energies/forces/observables) on a dataset
so3lr train [options]         Train a SO3LR/SO3krates model from a config file
so3lr finetune [options]      Finetune an existing model on a new dataset
so3lr tune-ewald [options]    Auto-tune Ewald/PME/P3M k-space parameters for a structure

## Usage Examples

Optimize a structure with all options:
  so3lr opt --input geometry.xyz --output so3lr_opt.xyz 
      --total-charge 0 --min-cycles 10 --min-steps 10
      --min-start-dt 0.05 --min-max-dt 0.1 --min-n-min 2
      --force-conv 0.05 --precision float32 --lr-cutoff 12.0 
      --dispersion-damping 2.0 --buffer-sr 1.25 --buffer-lr 1.25 
      --log-file so3lr_opt.log --model /path/to/model

Run NVT simulation with all options:
  so3lr nvt --input geometry.xyz --output so3lr_nvt.hdf5 --temperature 300
      --model /path/to/model --dt 0.5 --md-cycles 100 --md-steps 100
      --nhc-chain 3 --nhc-steps 2 --nhc-thermo 100.0 --relax --force-conv 0.05
      --seed 42 --log-file so3lr_nvt.log --precision float32
      --lr-cutoff 12.0 --dispersion-damping 2.0 --buffer-sr 1.25 --buffer-lr 1.25
      --total-charge 0 --save-buffer 50
      --restart-save so3lr_nvt.npz --restart-load so3lr_nvt_previous.npz

Run Langevin NVT simulation with all options:
  so3lr nvt-langevin --input geometry.xyz --output so3lr_nvt.hdf5 --temperature 300
      --model /path/to/model --dt 0.5 --md-cycles 100 --md-steps 100
      --langevin-removecmmotion True --langevin-thermo 1.0 --relax --force-conv 0.05
      --seed 42 --log-file so3lr_nvt.log --precision float32
      --lr-cutoff 12.0 --dispersion-damping 2.0 --buffer-sr 1.25 --buffer-lr 1.25
      --total-charge 0 --save-buffer 50
      --restart-save so3lr_nvt.npz --restart-load so3lr_nvt_previous.npz
      
Run NPT simulation with all options:
  so3lr npt --input geometry.xyz --output npt_traj.hdf5 --temperature 300
      --pressure 1.0 --model /path/to/model --dt 0.5
      --md-cycles 100 --md-steps 100 --nhc-chain 3 --nhc-steps 2 --nhc-thermo 100.0
      --nhc-baro 1000.0 --relax --seed 42
      --log-file npt_md.log --precision float32 --lr-cutoff 12.0
      --dispersion-damping 2.0 --buffer-sr 1.25 --buffer-lr 1.25 --total-charge 0
      --save-buffer 50 --force-conv 0.05
      --restart-save so3lr_npt.npz --restart-load so3lr_npt_previous.npz

Run NVE simulation with all options:
  so3lr nve --input geometry.xyz --output so3lr_nve.hdf5 --temperature 300
      --model /path/to/model --dt 0.5 --md-cycles 100 --md-steps 100
      --seed 42 --log-file so3lr_nve.log --precision float32
      --lr-cutoff 12.0 --dispersion-damping 2.0 --buffer-sr 1.25 --buffer-lr 1.25
      --total-charge 0 --save-buffer 50
      --restart-save so3lr_nve.npz --restart-load so3lr_nve_previous.npz

Evaluate SO3LR on a dataset with all options:
  so3lr eval --datafile data.extxyz --batch-size 10 --lr-cutoff 12.0
      --dispersion-damping 2.0 --jit-compile --save-to predictions.extxyz
      --targets forces,dipole_vec,hirshfeld_ratios --precision float32
      --model /path/to/model --log-file eval.log

Finetune SO3LR on a new dataset:
  so3lr finetune --datafile dataset.extxyz --workdir finetune_so3lr
      --num-train 50 --num-valid 10 --strategy full
      --config /path/to/finetune.yaml --model /path/to/model

Train a SO3krates/SO3LR model from a config file:
  so3lr train --config config.yaml

Auto-tune k-space electrostatics parameters (Ewald/PME/P3M):
  so3lr tune-ewald --input geometry.xyz --kspace-electrostatics ewald
      --kspace-smearing 4.0 --kspace-spacing 2.0 --kspace-interp-nodes 4
      --lr-cutoff 12.0 --model /path/to/model

"""

# Run with settings file:
#   so3lr --settings md_settings.yaml
# so3lr [options]         Run MD simulation with options specified via command line or settings file
# 
# ### Settings File Examples
# Example settings file for NVT simulation (nvt_settings.yaml):
# ```yaml
# # Input/Output
# input_file: "path/to/structure.xyz"    # Path to the initial geometry file
# output_file: "nvt_trajectory.hdf5"           # Output trajectory file in 'hdf5' or 'extxyz' format
# log_file: "nvt_md.log"                       # File to write logs to

# # Model settings
# model_path: "/path/to/model"                 # Optional path to custom MLFF model
# precision: "float32"                         # Numerical precision: 'float32' or 'float64'
# lr_cutoff: 12.0                              # Long-range cutoff distance in Å
# dispersion_damping: 2.0                      # Dispersion interactions start to switch off at (lr_cutoff-dispersion_damping) Å
# buffer_size_multiplier_sr: 1.25              # Buffer size multiplier for short-range interactions
# buffer_size_multiplier_lr: 1.25              # Buffer size multiplier for long-range interactions
# save_buffer: 50                              # Number of frames to buffer before writing

# # MD settings
# md_dt: 0.5                                   # MD timestep in femtoseconds
# md_T: 300.0                                  # Simulation temperature in Kelvin
# md_cycles: 100                               # Number of MD cycles to run
# md_steps: 100                                # Number of steps per MD cycle
# relax_before_run: true                       # Whether to perform geometry relaxation before MD
# force_convergence: 0.05                      # Force convergence criterion in eV/Å for initial relaxation

# # Nose-Hoover thermostat settings
# nhc_chain_length: 3                          # Length of the Nose-Hoover thermostat chain
# nhc_steps: 2                                 # Number of integration steps per MD step
# nhc_thermo: 100.0                            # Thermostat timescale in femtoseconds

# # Langevin thermostat settings
# langevin_removecmmotion: True                # Removal of the Center of Mass Motion
# langevin_thermo: 1.0                         # Friction (Gamma) coefficient for the Langevin thermostat

# # Restart options
# restart_save_path: "restart.npz"             # Path to save restart data
# restart_load_path: "previous.npz"            # Path to load restart data from previous run

# # Additional settings
# total_charge: 0                              # Total charge of the system
# seed: 42                                     # Random seed for MD
# observables: []                              # List of observable quantities to compute and save
# output_atom_indices: null                    # Optional list of atom indices to include in output

# # K-space electrostatics (optional, for periodic systems)
# kspace_electrostatics: null                  # K-space method: null/"ewald"/"pme"/"p3m"
# kspace_smearing: 4.0                         # Smearing width for k-space summation (Å)
# kspace_spacing: 2.0                          # Grid spacing for k-space summation (Å)
# kspace_interp_nodes: 4                       # Number of interpolation nodes for PME/P3M

# # Restart settings
# restart_save_path: "restart.npz"             # Path to save restart data
# restart_load_path: "previous.npz"            # Path to load restart data from previous run
# restart_reset: false                         # Reset accumulated thermostat/barostat state on restart
# ```

# Example settings file for NVE simulation (nve_settings.yaml):
# ```yaml
# # Input/Output
# input_file: "path/to/structure.xyz"          # Path to the initial geometry file
# output_file: "nve_trajectory.hdf5"           # Output trajectory file in 'hdf5' or 'extxyz' format
# log_file: "nve_md.log"                       # File to write logs to

# # Model settings
# model_path: "/path/to/model"                 # Optional path to custom MLFF model
# precision: "float32"                         # Numerical precision: 'float32' or 'float64'
# lr_cutoff: 12.0                              # Long-range cutoff distance in Å
# dispersion_damping: 2.0                      # Dispersion interactions start to switch off at (lr_cutoff-dispersion_damping) Å
# buffer_size_multiplier_sr: 1.25              # Buffer size multiplier for short-range interactions
# buffer_size_multiplier_lr: 1.25              # Buffer size multiplier for long-range interactions
# save_buffer: 50                              # Number of frames to buffer before writing

# # MD settings
# md_dt: 0.5                                   # MD timestep in femtoseconds
# init_T: 300.0                                # Initial temperature for velocity initialization (NVE only)
# md_cycles: 100                               # Number of MD cycles to run
# md_steps: 100                                # Number of steps per MD cycle
# relax_before_run: true                       # Whether to perform geometry relaxation before MD
# force_convergence: 0.05                      # Force convergence criterion in eV/Å for initial relaxation

# # Additional settings
# total_charge: 0                              # Total charge of the system
# seed: 42                                     # Random seed for MD
# ```

# Example settings file for NPT simulation (npt_settings.yaml):
# ```yaml
# # Input/Output
# input_file: "path/to/structure.xyz"    # Path to the initial geometry file
# output_file: "npt_trajectory.hdf5"           # Output trajectory file in 'hdf5' or 'xyz' format
# log_file: "npt_md.log"                       # File to write logs to

# # Model settings
# model_path: "/path/to/model"                 # Optional path to custom MLFF model
# precision: "float32"                         # Numerical precision: 'float32' or 'float64'
# lr_cutoff: 12.0                              # Long-range cutoff distance in Å
# dispersion_damping: 2.0                      # Dispersion interactions start to switch off at (lr_cutoff-dispersion_damping) Å
# buffer_size_multiplier_sr: 1.25              # Buffer size multiplier for short-range interactions
# buffer_size_multiplier_lr: 1.25              # Buffer size multiplier for long-range interactions
# save_buffer: 50                              # Number of frames to buffer before writing

# # MD settings
# md_dt: 0.5                                   # MD timestep in femtoseconds
# md_T: 300.0                                  # Simulation temperature in Kelvin
# md_P: 1.0                                    # Pressure in bar (enables NPT simulation)
# md_cycles: 100                               # Number of MD cycles to run
# md_steps: 100                                # Number of steps per MD cycle
# relax_before_run: true                       # Whether to perform geometry relaxation before MD
# force_convergence: 0.05                      # Force convergence criterion in eV/Å for initial relaxation

# # Thermostat and barostat settings
# nhc_chain_length: 3                          # Length of the Nose-Hoover thermostat chain
# nhc_steps: 2                                 # Number of integration steps per MD step
# nhc_thermo: 100.0                            # Thermostat damping factor in units of timestep
# nhc_baro: 1000.0                             # Barostat damping factor in units of timestep
# nhc_sy_steps: 3                              # Number of Suzuki-Yoshida integration steps

# # Restart options
# restart_save_path: "restart.npz"             # Path to save restart data
# restart_load_path: "previous.npz"            # Path to load restart data from previous run

# # Additional settings
# total_charge: 0                              # Total charge of the system
# seed: 42                                     # Random seed for MD
# observables: []                              # List of observable quantities to compute and save
# output_atom_indices: null                    # Optional list of atom indices to include in output

# # K-space electrostatics (optional, for periodic systems)
# kspace_electrostatics: null                  # K-space method: null/"ewald"/"pme"/"p3m"
# kspace_smearing: 4.0                         # Smearing width for k-space summation (Å)
# kspace_spacing: 2.0                          # Grid spacing for k-space summation (Å)
# kspace_interp_nodes: 4                       # Number of interpolation nodes for PME/P3M
# kspace_npt_cycles: 10                        # Frequency (in MD cycles) to re-tune k-space grid for NPT

# # Restart settings
# restart_save_path: "restart.npz"             # Path to save restart data
# restart_load_path: "previous.npz"            # Path to load restart data from previous run
# restart_reset: false                         # Reset accumulated thermostat/barostat state on restart
# ```

# Example settings file for geometry optimization (opt_settings.yaml):
# ```yaml
# # Input/Output
# input_file: "path/to/structure.xyz"          # Path to the initial geometry file
# output_file: "optimized.xyz"                 # Output file to save the optimized geometry
# log_file: "opt.log"                          # File to write logs to

# # Model settings
# model_path: "/path/to/model"                 # Optional path to custom MLFF model
# precision: "float32"                         # Numerical precision: 'float32' or 'float64'
# lr_cutoff: 12.0                              # Long-range cutoff distance in Å
# dispersion_damping: 2.0                      # Dispersion interactions start to switch off at (lr_cutoff-dispersion_damping) Å
# buffer_size_multiplier_sr: 1.25              # Buffer size multiplier for short-range interactions
# buffer_size_multiplier_lr: 1.25              # Buffer size multiplier for long-range interactions

# # Optimization settings
# min_cycles: 10                               # Number of minimization cycles
# min_steps: 10                                # Number of steps per minimization cycle
# min_start_dt: 0.05                           # Initial timestep for minimization
# min_max_dt: 0.1                              # Maximum timestep for minimization
# min_n_min: 2                                 # Number of steps moving in correct direction before dt update
# force_convergence: 0.05                      # Force convergence criterion in eV/Å

# # Additional settings
# total_charge: 0                              # Total charge of the system
# ```

# Example settings file for model evaluation (eval_settings.yaml):
# ```yaml
# # Input/Output
# datafile: "path/to/dataset.extxyz"           # Path to data file for evaluation
# save_to: "so3lr_eval.extxyz"                 # Output file to save predictions
# log_file: "eval.log"                         # File to write logs to

# # Model settings
# model_path: "/path/to/model"                 # Optional path to custom MLFF model
# precision: "float32"                         # Numerical precision: 'float32' or 'float64'
# lr_cutoff: 12.0                              # Long-range cutoff distance in Å
# dispersion_damping: 2.0                      # Dispersion interactions start to switch off at (lr_cutoff-dispersion_damping) Å

# # Evaluation settings
# batch_size: 10                               # Number of molecules per batch
# targets: "forces,dipole_vec,hirshfeld_ratios" # Targets to evaluate
# jit_compile: true                            # Use JIT compilation for speed

# # Experimental settings
#
# Ewald electrostatics settings
# kspace_electrostatics: null                  # K-space method for long-range electrostatics (null/"ewald"/"pme"/"p3m")
# kspace_smearing: 4.0                         # Smearing width for k-space summation
# kspace_spacing: 2.0                          # Grid spacing for k-space summation
# kspace_interp_nodes: 4                       # Number of interpolation nodes for PME/P3M
#
# observables: ["partial_charges","dipole_vec","hirshfeld_ratios"]
#
# output_atom_indices: null                    # Optional list of atom indices to output
# ```


BASIC_HELP_STRING = """
Run simulations using SO3LR Machine Learned Force Field.

## Commands

so3lr opt [options]            Run geometry optimization
so3lr nvt [options]            Run NVT (constant volume and temperature) MD simulation
so3lr nvt-langevin [options]   Run NVT MD with a Langevin thermostat
so3lr npt [options]            Run NPT (constant pressure and temperature) MD simulation
so3lr nve [options]            Run NVE (constant volume and energy) MD simulation
so3lr eval [options]           Evaluate SO3LR model on a dataset
so3lr finetune [options]       Finetune SO3LR model on a dataset
so3lr train [options]          Train a SO3LR/SO3krates model from a config file
so3lr tune-ewald [options]     Auto-tune Ewald/PME/P3M k-space parameters

## Usage Examples

Optimize a structure:
  so3lr opt --input geometry.xyz --force-conv 0.05 --lr-cutoff 12.0

Run NVT simulation:
  so3lr nvt --input geometry.xyz --temperature 300 --dt 0.5 --md-cycles 100 --md-steps 100

Run NPT simulation:
  so3lr npt --input geometry.xyz --temperature 300 --dt 0.5 --md-cycles 100 --md-steps 100 --pressure 1.0

Run NVE simulation:
  so3lr nve --input geometry.xyz --temperature 300 --dt 0.5 --md-cycles 100 --md-steps 100

Evaluate on a dataset:
  so3lr eval --datafile dataset.xyz --save-to predictions.extxyz --targets forces,dipole_vec,hirshfeld_ratios

Finetune on a dataset:
  so3lr finetune --datafile dataset.xyz --workdir so3lr_finetuned --num-train 100 --num-valid 10

Train from a config file:
  so3lr train --config config.yaml

Use --help-full to see all available options.

"""

# so3lr [options]         Run with options specified via command line or settings file
# Run with settings file:
#   so3lr --settings md_settings.yaml

def infer_output_format(output_file):
    """Determine output format based on file extension. Returns 'extxyz' for
    None/empty input (caller is expected to supply an actual filename later)."""
    if not output_file:
        return 'extxyz'
    if output_file.endswith('.hdf5') or output_file.endswith('.h5'):
        return 'hdf5'
    elif output_file.endswith('.xyz') or output_file.endswith('.extxyz'):
        return 'extxyz'
    else:
        logger.warning(f"Unknown output extension for {output_file}, defaulting to extxyz format")
        return 'extxyz'


def _format_runtime(seconds: float) -> str:
    """Adaptive runtime formatter for the end-of-run 'Total runtime' line.

    The legacy 'X.XX seconds (Y.YY hours)' format made the hours figure useless
    for typical minute-scale MD runs (and the seconds figure useless for
    multi-hour training jobs). Pick a unit pair appropriate to the magnitude:
    s+ms below 1 min, m+s below 1 h, h+m at or above 1 h.
    """
    if seconds < 60.0:
        total_ms = int(round(seconds * 1000))
        s, ms = divmod(total_ms, 1000)
        return f"{s} s {ms} ms"
    if seconds < 3600.0:
        total_s = int(round(seconds))
        m, s = divmod(total_s, 60)
        return f"{m} m {s} s"
    total_s = int(round(seconds))
    h, rem = divmod(total_s, 3600)
    m = rem // 60
    return f"{h} h {m} m"


# ============================================================================
# Decorator factories for common CLI options
# ============================================================================

def _validate_model_choice(ctx, param, value):
    """Click callback: resolve ``--model`` at parse time so typos fail fast.

    A typo like ``--model so3lr-x`` used to fail 20-40 s into JAX / orbax
    initialisation. By calling :func:`so3lr.model_registry.resolve_model` here we
    surface unknown-name / missing-path errors before any heavy import, and add
    a ``difflib`` did-you-mean over the bundled-model list.

    Returns ``value`` unchanged on success (including ``None``, which downstream
    code interprets as the default model). Re-raises any resolution failure as
    :class:`click.BadParameter` so Click prints a clean, traceback-free error
    alongside the offending option.
    """
    if value is None:
        return value
    # model_registry is intentionally JAX-free (stdlib imports only), so this
    # import is safe to do during Click parsing without forcing a slow init.
    from ..model_registry import available_models, resolve_model
    try:
        resolve_model(value)
    except FileNotFoundError:
        names = available_models()
        suggestions = difflib.get_close_matches(str(value), names, n=3, cutoff=0.5)
        msg = (
            f"Model '{value}' is not a bundled name and no such directory exists. "
            f"Bundled names: {', '.join(names)}."
        )
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        raise click.BadParameter(msg, ctx=ctx, param=param)
    return value


def common_io_options(f: Callable) -> Callable:
    """Common input/output options for all MD commands."""
    f = click.option('--input', '--input_file', 'input_file', type=click.Path(exists=True, dir_okay=False),
                     help='Input geometry file (any ASE-readable format). [default: None]')(f)
    f = click.option('--output', '--output_file', 'output_file', type=click.Path(),
                     help='Output trajectory file (.hdf5 or .xyz). [default: <input>_<ensemble>.xyz]')(f)
    f = click.option('--log-file', default=None, type=click.Path(),
                     help='File to write logs to. [default: <output>.log]')(f)
    f = click.option('--overwrite/--no-overwrite', 'overwrite', default=False,
                     help='If the trajectory output already exists, overwrite it. '
                          'Default is to refuse (rename your previous run, or pass --overwrite).')(f)
    return f


def common_model_options(f: Callable) -> Callable:
    """Common model options for all commands."""
    f = click.option('--model', '--model_path', '--model-path', 'model_path', type=str, default=None,
                     callback=_validate_model_choice,
                     help='Model to use: a bundled name (so3lr-s, so3lr-m, so3lr-l, so3lr) '
                          'or a path to a custom MLFF model directory. '
                          'If not provided, so3lr-m is used. [default: so3lr-m]')(f)
    f = click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
                     help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')(f)
    f = click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
                     help=f'Long-range cutoff distance in Å. The default ({DEFAULT_LR_CUTOFF}) '
                          f'is intended for condensed-phase systems with periodic boundaries; '
                          f'for gas-phase / non-periodic systems use --lr-cutoff 100 so the '
                          f'long-range electrostatics and dispersion are not artificially truncated. '
                          f'[default: {DEFAULT_LR_CUTOFF}]')(f)
    f = click.option('--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
                     help=f'Dispersion interactions start to switch off at (lr_cutoff - dispersion_damping) Å. [default: {DEFAULT_DISPERSION_DAMPING}]')(f)
    f = click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
                     help=f'Buffer size multiplier for short-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')(f)
    f = click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
                     help=f'Buffer size multiplier for long-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')(f)
    f = click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
                     help=f'Total charge of the system. [default: {DEFAULT_TOTAL_CHARGE}]')(f)
    f = click.option('--charge-embed-total-charge', 'charge_embed_total_charge',
                     default=None, type=float,
                     help='Total charge seen by the charge EMBEDDING. Unset '
                          '(default) follows --total-charge; 0 embeds a neutral '
                          'system. [default: follow --total-charge]')(f)
    f = click.option('--charge-equilibration-total-charge',
                     'charge_equilibration_total_charge',
                     default=None, type=float,
                     help='Total charge that charge EQUILIBRATION conserves to. '
                          'Unset (default) follows --total-charge; 0 conserves to '
                          'neutral. Independent of --charge-embed-total-charge. '
                          '[default: follow --total-charge]')(f)
    f = click.option('--spin', 'num_unpaired_electrons', default=DEFAULT_SPIN, type=int,
                     help=f'Spin: number of unpaired electrons (multiplicity - 1; equivalently, multiplicity = spin + 1). [default: {DEFAULT_SPIN}]')(f)
    return f


def common_md_options(f: Callable) -> Callable:
    """Common MD options shared by all ensemble commands."""
    f = click.option('--save-buffer', default=DEFAULT_SAVE_BUFFER, type=int,
                     help=f'Number of frames to buffer before writing. [default: {DEFAULT_SAVE_BUFFER}]')(f)
    f = click.option('--temperature', default=DEFAULT_TEMPERATURE, type=float,
                     help=f'Simulation temperature in Kelvin. [default: {DEFAULT_TEMPERATURE}]')(f)
    f = click.option('--dt', default=DEFAULT_TIMESTEP, type=float,
                     help=f'MD timestep in femtoseconds. [default: {DEFAULT_TIMESTEP}]')(f)
    f = click.option('--md-cycles', default=DEFAULT_MD_CYCLES, type=int,
                     help=f'Number of MD cycles to run. [default: {DEFAULT_MD_CYCLES}]')(f)
    f = click.option('--md-steps', default=DEFAULT_MD_STEPS_PER_CYCLE, type=int,
                     help=f'Number of steps per MD cycle. [default: {DEFAULT_MD_STEPS_PER_CYCLE}]')(f)
    f = click.option('--restart-save', type=click.Path(), default=None,
                     help='Path to save restart data.')(f)
    f = click.option('--restart-load', type=click.Path(exists=True, dir_okay=False), default=None,
                     help='Path to load restart data from a previous run.')(f)
    f = click.option('--relax/--no-relax', default=True,
                     help='Perform geometry relaxation before MD. [default: enabled]')(f)
    f = click.option('--force-conv', default=None, type=float,
                     help='Force convergence criterion in eV/Å for initial relaxation. [default: None]')(f)
    f = click.option('--seed', default=DEFAULT_SEED, type=int,
                     help=f'Random seed for MD. [default: {DEFAULT_SEED}]')(f)
    f = click.option('--observable', default=DEFAULT_OBSERVABLES, multiple=True,
                     type=click.Choice([
                         'nn_energy', 'partial_charges', 'electrostatic_energy',
                         'dispersion_energy', 'zbl_repulsion',
                         'nlh_repulsion',
                         'electrostatic_energy_undamped', 'dispersion_energy_undamped',
                         'x',
                     ]),
                     help='Observable to compute and save during MD. Repeatable. '
                          'Allowed: nn_energy, partial_charges, electrostatic_energy, '
                          'dispersion_energy, zbl_repulsion (legacy name) / nlh_repulsion '
                          '(canonical for so3lr-s/m/l; same quantity, different model output key), '
                          'electrostatic_energy_undamped, dispersion_energy_undamped, x. '
                          '[default: none]')(f)
    f = click.option('--help', '-h', is_flag=True, help='Show brief command overview.')(f)
    return f


def thermostat_options(f: Callable) -> Callable:
    """Nose-Hoover thermostat options for NVT/NPT ensembles."""
    f = click.option('--nhc-chain', default=DEFAULT_NHC_CHAIN_LENGTH, type=int,
                     help=f'Length of the Nose-Hoover thermostat chain. [default: {DEFAULT_NHC_CHAIN_LENGTH}]')(f)
    f = click.option('--nhc-steps', default=DEFAULT_NHC_INTEGRATION_STEPS, type=int,
                     help=f'Number of integration steps per MD step. [default: {DEFAULT_NHC_INTEGRATION_STEPS}]')(f)
    f = click.option('--nhc-thermo', default=DEFAULT_NHC_THERMO, type=float,
                     help=f'Thermostat damping factor (tau = dt * nhc_thermo). [default: {DEFAULT_NHC_THERMO}]')(f)
    f = click.option('--nhc-sy-steps', default=DEFAULT_SUZUKI_YOSHIDA_STEPS, type=int,
                     help=f'Number of Suzuki-Yoshida integration steps. [default: {DEFAULT_SUZUKI_YOSHIDA_STEPS}]')(f)
    return f


def barostat_options(f: Callable) -> Callable:
    """Nose-Hoover barostat options for NPT ensemble."""
    f = click.option('--pressure', default=DEFAULT_PRESSURE, type=float,
                     help=f'Simulation pressure in bar. [default: {DEFAULT_PRESSURE}]')(f)
    f = click.option('--nhc-baro', default=DEFAULT_NHC_BARO, type=float,
                     help=f'Barostat damping factor (tau = dt * nhc_baro). [default: {DEFAULT_NHC_BARO}]')(f)
    return f


def langevin_options(f: Callable) -> Callable:
    """Langevin thermostat options for NVT-Langevin ensemble.
    Note: The friction coefficient gamma is computed as gamma = 1.0 / (md_dt * langevin_thermo).
    langevin_thermo is a dimensionless timescale parameter, not gamma itself."""
    f = click.option('--langevin-removecmmotion', default=DEFAULT_LANGEVIN_REMOVECMMOTION, type=bool,
                     help=f'Remove center of mass motion. [default: {DEFAULT_LANGEVIN_REMOVECMMOTION}]')(f)
    f = click.option('--langevin-thermo', default=DEFAULT_LANGEVIN_THERMO, type=float,
                     help=f'Langevin timescale parameter (gamma = 1.0 / (md_dt * langevin_thermo)). [default: {DEFAULT_LANGEVIN_THERMO}]')(f)
    return f


def kspace_options(f: Callable) -> Callable:
    """K-space electrostatics options for periodic systems."""
    f = click.option('--kspace-electrostatics', 'kspace_electrostatics', default=None,
                     type=click.Choice(['ewald', 'pme', 'p3m'], case_sensitive=False),
                     help='K-space method for long-range electrostatics: "ewald", "pme", or "p3m". [default: None]')(f)
    f = click.option('--kspace-smearing', 'kspace_smearing', default=DEFAULT_KSPACE_SMEARING, type=float,
                     help=f'Smearing width for k-space summation in Å. [default: {DEFAULT_KSPACE_SMEARING}]')(f)
    f = click.option('--kspace-spacing', 'kspace_spacing', default=DEFAULT_KSPACE_SPACING, type=float,
                     help=f'Grid spacing for k-space summation in Å. [default: {DEFAULT_KSPACE_SPACING}]')(f)
    f = click.option('--kspace-interp-nodes', 'kspace_interp_nodes', default=DEFAULT_KSPACE_INTERP_NODES, type=int,
                     help=f'Number of interpolation nodes for PME/P3M. [default: {DEFAULT_KSPACE_INTERP_NODES}]')(f)
    f = click.option('--kspace-npt-cycles', 'kspace_npt_cycles', default=None, type=int,
                     help='Frequency (in MD cycles) to re-tune k-space grid in NPT. [default: None]')(f)
    return f


class CustomCommandClass(click.Group):
    """Custom Group class that displays the correct help based on flags."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Format the help display with ASCII art and appropriate help string."""
        # Display ascii art at the top
        formatter.write(SO3LR_ASCII)

        # For basic help, show the basic help string
        if not ctx.params.get('help_full', False):
            formatter.write(BASIC_HELP_STRING)
        else:
            formatter.write(FULL_HELP_STRING)


# Default values applied to the settings dict when using the YAML pathway.
# These mirror the CLI option defaults so that omitting a key from the YAML
# produces the same result as not passing the corresponding CLI flag.
YAML_DEFAULTS: Dict[str, Any] = {
    'precision':                  DEFAULT_PRECISION,
    'lr_cutoff':                  DEFAULT_LR_CUTOFF,
    'dispersion_damping':         DEFAULT_DISPERSION_DAMPING,
    'buffer_size_multiplier_sr':  DEFAULT_BUFFER_MULTIPLIER,
    'buffer_size_multiplier_lr':  DEFAULT_BUFFER_MULTIPLIER,
    'total_charge':               DEFAULT_TOTAL_CHARGE,
    'num_unpaired_electrons':     DEFAULT_SPIN,
    'seed':                       DEFAULT_SEED,
    'save_buffer':                DEFAULT_SAVE_BUFFER,
    'md_dt':                      DEFAULT_TIMESTEP / 1000,  # stored in ps
    'md_T':                       DEFAULT_TEMPERATURE,
    'md_cycles':                  DEFAULT_MD_CYCLES,
    'md_steps':                   DEFAULT_MD_STEPS_PER_CYCLE,
    'nhc_chain_length':           DEFAULT_NHC_CHAIN_LENGTH,
    'nhc_steps':                  DEFAULT_NHC_INTEGRATION_STEPS,
    'nhc_thermo':                 DEFAULT_NHC_THERMO,
    'nhc_baro':                   DEFAULT_NHC_BARO,
    'nhc_sy_steps':               DEFAULT_SUZUKI_YOSHIDA_STEPS,
    'npt_coupling':               'isotropic',
    'langevin_removecmmotion':    DEFAULT_LANGEVIN_REMOVECMMOTION,
    'langevin_thermo':            DEFAULT_LANGEVIN_THERMO,
    'relax_before_run':           True,
    'min_cycles':                 DEFAULT_MIN_CYCLES,
    'min_steps':                  DEFAULT_MIN_STEPS,
    'min_start_dt':               DEFAULT_MIN_START_DT,
    'min_max_dt':                 DEFAULT_MIN_MAX_DT,
    'min_n_min':                  DEFAULT_MIN_N_MIN,
    'observables':                list(DEFAULT_OBSERVABLES),
}


# Keys that are NOT in YAML_DEFAULTS but are legitimately read by perform_md /
# the YAML loader / advanced subsystems. Listing them here lets the unknown-key
# validator distinguish a real configurable knob from a user typo.
KNOWN_YAML_KEYS_EXTRA: set = {
    # Required / standard I/O (CLI defaults are None, not present in YAML_DEFAULTS)
    'input_file', 'output_file', 'log_file', 'model_path', 'model',
    'output_atom_indices', 'ensemble',
    # Restart
    'restart_load_path', 'restart_save_path', 'restart_reset',
    # Atom constraints
    'atom_mask_path', 'fixed_atom_indices', 'mobile_atom_indices',
    'fixed_atom_constraint', 'output_cartesian_positions',
    # K-space electrostatics
    'kspace_electrostatics', 'kspace_smearing', 'kspace_spacing',
    'kspace_interp_nodes', 'kspace_npt_cycles',
    # Charge overrides (default None -> follow total_charge)
    'charge_embed_total_charge', 'charge_equilibration_total_charge',
    # K-space dispersion
    'kspace_dispersion', 'kspace_dispersion_smearing',
    'kspace_dispersion_spacing', 'kspace_dispersion_interp_nodes',
    # Numerical / safety knobs
    'matmul_precision', 'minimum_cell_size_multiplier',
    'energy_drift_threshold', 'drift_warmup_ps', 'dispersion_tail_correction',
    # Ensemble-specific
    'md_T', 'md_P', 'init_T', 'npt_coupling', 'num_unpaired_electrons',
    # Enhanced sampling (PySAGES)
    'do_enhanced_sampling', 'enforce_cartesian_coordinates', 'pysages_input',
    'restart_pysages_save_path', 'restart_pysages_load_path',
    # Minimization (overrides cycles/steps if set)
    'force_convergence',
    # Eval keys (allowed in the same YAML even though `so3lr eval` does not use
    # --settings yet, since users often share a single YAML across runs)
    'datafile', 'save_to', 'targets', 'batch_size', 'jit_compile',
    # Derived (re-appears if a user copies the merged-settings log dump back in)
    'output_format',
    # Trajectory-overwrite policy: refuse pre-existing output by default;
    # set true (CLI flag --overwrite) to permit truncation.
    'overwrite',
}


# YAML keys that mirror the CLI flag spelling for the same setting. The YAML
# loader canonicalises these to the internal name before any other processing,
# so a user can write `temperature: 300` in the YAML and have it land as `md_T`.
# Keep in sync with the CLI option names in run_md_simulation / nvt/npt/nve.
YAML_KEY_ALIASES: Dict[str, str] = {
    # Identity / model
    'model':                'model_path',
    # MD ensemble shorthand
    'temperature':          'md_T',
    'pressure':             'md_P',
    'dt':                   'md_dt',
    # Neighbor-list buffers (the CLI flags are --buffer-sr / --buffer-lr)
    'buffer_sr':            'buffer_size_multiplier_sr',
    'buffer_lr':            'buffer_size_multiplier_lr',
    # NHC / thermostat shorthand
    'nhc_chain':            'nhc_chain_length',
    # Other CLI ↔ YAML mismatches users hit
    'spin':                 'num_unpaired_electrons',
    'observable':           'observables',
    'force_conv':           'force_convergence',
}


def apply_yaml_key_aliases(settings_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Translate CLI-flag-style YAML keys to their internal names.

    Modifies and returns the dict in place. If both the alias and the canonical
    key are present, the canonical wins and a warning is logged.
    """
    for alias, canonical in YAML_KEY_ALIASES.items():
        if alias in settings_dict:
            if canonical in settings_dict:
                logger.warning(
                    f"YAML has both '{alias}' (alias) and '{canonical}'; "
                    f"using '{canonical}' and ignoring '{alias}'."
                )
                del settings_dict[alias]
            else:
                settings_dict[canonical] = settings_dict.pop(alias)
    return settings_dict


def known_yaml_keys() -> set:
    """Union of every key the MD/YAML pipeline knows how to consume."""
    return set(YAML_DEFAULTS.keys()) | KNOWN_YAML_KEYS_EXTRA | set(YAML_KEY_ALIASES.keys())


def validate_yaml_keys(settings_dict: Dict[str, Any], settings_path: Optional[Path] = None) -> None:
    """Warn on unknown keys in a settings YAML, with did-you-mean suggestions.

    A typo in a settings file used to be silent — the loader simply dropped the
    unknown key on the floor and ran with defaults. This catches typos like
    `md_t: 350` (real: md_T), `tempreature: 300`, or `hdf5_buffer_size: 50`
    (real: save_buffer) and surfaces them before the simulation starts. Warns
    rather than errors so that a YAML with a single typo does not silently
    abort a multi-hour job; the warning is loud and lists candidate spellings.
    """
    known = known_yaml_keys()
    unknown = sorted(set(settings_dict.keys()) - known)
    if not unknown:
        return
    src = f" in {settings_path.name}" if settings_path else ""
    for k in unknown:
        suggestions = difflib.get_close_matches(k, sorted(known), n=3, cutoff=0.6)
        msg = f"Unknown setting '{k}'{src}; ignoring."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        logger.warning(msg)


def canonicalize_ensemble(settings_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Determine ensemble from settings dict, defaulting to NVT.

    Ensemble can be set explicitly via ``settings_dict['ensemble']`` (e.g. from a
    YAML settings file), or is inferred: presence of ``md_P`` implies NPT, otherwise
    NVT is used as the default.
    """
    if settings_dict.get('ensemble') is None:
        if settings_dict.get('md_P') is not None:
            settings_dict['ensemble'] = 'npt'
        else:
            settings_dict['ensemble'] = 'nvt'
    return settings_dict

@click.group(cls=CustomCommandClass, invoke_without_command=True,
             help="Run molecular dynamics using SO3LR Machine Learned Force Field.",
             context_settings={"help_option_names": []})
@click.option('--settings', type=click.Path(exists=True, dir_okay=False), default=None,
              help='Path to YAML settings file. All simulation parameters are read from this file.')
@click.option('--help-full', is_flag=True,
              help='Show detailed information about MD settings.')
@click.option('--help', '-h', is_flag=True,
              help='Show brief command overview.')
@click.pass_context
def cli(ctx: click.Context, settings: Optional[str], help_full: bool, help: bool) -> None:
    """
    Run molecular dynamics simulations with SO3LR or MLFF models.

    \b
    Subcommands:
      opt           Run geometry optimization
      nvt           Run NVT MD (Nose-Hoover thermostat)
      nvt-langevin  Run NVT MD (Langevin thermostat)
      npt           Run NPT MD
      nve           Run NVE MD

    When called without a subcommand, all settings are read from a YAML file:

      so3lr --settings settings.yaml
    """
    if help or help_full:
        ctx.info_name = ctx.command.name
        click.echo(ctx.get_help())
        return

    if ctx.invoked_subcommand is not None:
        return

    if not settings:
        click.echo(ctx.get_help())
        return

    try:
        settings_path = Path(settings).expanduser().resolve()
        with open(settings_path, 'r') as f:
            settings_dict = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Error loading settings file: {str(e)}")
        sys.exit(1)

    # Translate CLI-flag-style aliases (e.g. temperature -> md_T,
    # pressure -> md_P, dt -> md_dt, model -> model_path) before any other
    # processing. See YAML_KEY_ALIASES for the full mapping.
    apply_yaml_key_aliases(settings_dict)

    # Surface typos early. Unknown keys (e.g. `md_t` for `md_T`,
    # `tempreature` for `md_T`) would otherwise be silently ignored and the
    # default value used, wasting GPU time on the wrong physics.
    validate_yaml_keys(settings_dict, settings_path)

    # Unit boundary: md_dt is in fs in the YAML / CLI (documented in
    # md_settings.yaml and the CLI --dt help), but every internal consumer
    # past this point — settings dict, perform_md, handle_units — expects
    # picoseconds. Convert here once so the rest of the code only sees ps.
    # NB: the 'md_dt' key stays the same; only its unit changes.
    if 'md_dt' in settings_dict:
        settings_dict['md_dt'] = settings_dict['md_dt'] / 1000

    # Apply defaults for any keys not present in the YAML
    for key, default in YAML_DEFAULTS.items():
        settings_dict.setdefault(key, default)

    settings_dict = canonicalize_ensemble(settings_dict)

    # Fast-fail on missing required keys before setup_logger / banner /
    # get_hardware_info (which triggers a slow JAX init). `input_file` is the
    # only truly-required YAML key with no YAML_DEFAULTS entry; without it,
    # downstream perform_md / run would crash with an opaque TypeError on
    # Path(None) / ase.read(None).
    if not settings_dict.get('input_file'):
        raise ValueError(
            f"Missing required key 'input_file' in {settings_path.name}: "
            f"path to the initial geometry file (any ASE-readable format, "
            f"e.g. .xyz / .extxyz / .pdb). Add a line like\n"
            f"  input_file: path/to/structure.xyz\n"
            f"to the YAML, or pass it on the CLI via a subcommand "
            f"(e.g. `so3lr nvt --input ...`)."
        )

    # Default output filename uses ensemble-specific suffix, mirroring the CLI
    # subcommand pathway (see e.g. the nvt/npt/nve subcommands). Previously the
    # YAML branch left output_file=None and immediately crashed in
    # infer_output_format. We compute the same `<input>_<ensemble>.xyz` here.
    if not settings_dict.get('output_file') and settings_dict.get('input_file'):
        _suffix = {'nvt': 'nvt', 'nvt-langevin': 'nvt_langevin', 'npt': 'npt', 'nve': 'nve'}
        _ens = settings_dict.get('ensemble', 'nvt')
        settings_dict['output_file'] = f"{Path(settings_dict['input_file']).stem}_{_suffix.get(_ens, _ens)}.xyz"

    settings_dict['output_format'] = infer_output_format(settings_dict.get('output_file', ''))

    # Default the log file to the settings YAML name with a .log extension
    # (e.g. `npt.yaml` -> `npt.log`), unless the YAML specifies `log_file`.
    if not settings_dict.get('log_file'):
        settings_dict['log_file'] = str(settings_path.with_suffix('.log'))

    setup_logger(settings_dict.get('log_file'))
    logger.info(SO3LR_ASCII)
    logger.info(f"Loading settings from {settings}")

    log_simulation_settings(settings_dict)
    get_hardware_info()

    time_start = time.time()
    run(settings_dict)
    time_end = time.time()
    logger.info('Simulation completed successfully!')
    logger.info(f'Total runtime: {_format_runtime(time_end - time_start)}')




# ============================================================================
# Settings logging
# ============================================================================

def log_simulation_settings(s: Dict[str, Any]) -> None:
    """Log all simulation settings from the internal settings dict."""
    ensemble = s.get('ensemble', 'nvt')
    _label = {'nvt': 'NVT', 'nvt-langevin': 'NVT (Langevin)', 'npt': 'NPT', 'nve': 'NVE'}
    dt_ps = s.get('md_dt', DEFAULT_TIMESTEP / 1000)  # md_dt is stored internally in ps
    dt_fs = dt_ps * 1000  # for display in fs
    total_steps = s.get('md_cycles', DEFAULT_MD_CYCLES) * s.get('md_steps', DEFAULT_MD_STEPS_PER_CYCLE)
    simulation_time = total_steps * s.get('md_dt', DEFAULT_TIMESTEP / 1000)  # ps

    logger.info("=" * 60)
    logger.info(f"SO3LR {_label.get(ensemble, ensemble)} Molecular Dynamics Simulation (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Initial geometry:          {s.get('input_file')}")
    logger.info(f"Output file:               {s.get('output_file')}")
    logger.info(f"Log file:                  {s.get('log_file')}")
    from so3lr.model_registry import DEFAULT_MODEL, available_models
    _model = s.get('model_path') or DEFAULT_MODEL
    _kind = 'bundled' if _model in available_models() else 'custom path'
    logger.info(f"Force field:               {_model} ({_kind})")
    logger.info(f"Precision:                 {s.get('precision', DEFAULT_PRECISION)}")
    logger.info(f"Long-range cutoff:         {s.get('lr_cutoff', DEFAULT_LR_CUTOFF)} Å")
    logger.info(f"Dispersion damping:        {s.get('dispersion_damping', DEFAULT_DISPERSION_DAMPING)} Å")
    logger.info(f"Short-range buffer:        {s.get('buffer_size_multiplier_sr', DEFAULT_BUFFER_MULTIPLIER)}")
    logger.info(f"Long-range buffer:         {s.get('buffer_size_multiplier_lr', DEFAULT_BUFFER_MULTIPLIER)}")
    logger.info(f"Total charge:              {s.get('total_charge', DEFAULT_TOTAL_CHARGE)}")
    logger.info(f"Spin (unpaired e-):        {s.get('num_unpaired_electrons', DEFAULT_SPIN)}")
    logger.info(f"Temperature:               {s.get('md_T', s.get('init_T', DEFAULT_TEMPERATURE))} K")
    if ensemble == 'npt':
        logger.info(f"Pressure:                  {s.get('md_P')} bar")
    logger.info(f"Ensemble:                  {_label.get(ensemble, ensemble)}")
    logger.info(f"Simulation length:         {total_steps} steps ({simulation_time:.2f} ps)")
    logger.info(f"MD cycles:                 {s.get('md_cycles', DEFAULT_MD_CYCLES)}")
    logger.info(f"Steps per cycle:           {s.get('md_steps', DEFAULT_MD_STEPS_PER_CYCLE)}")
    logger.info(f"Timestep:                  {dt_fs:.4g} fs")
    logger.info(f"Saving buffer size:        {s.get('save_buffer', DEFAULT_SAVE_BUFFER)}")
    if ensemble in ('nvt', 'npt'):
        logger.info(f"NHC chain length:          {s.get('nhc_chain_length', DEFAULT_NHC_CHAIN_LENGTH)}")
        logger.info(f"Nose-Hoover steps:         {s.get('nhc_steps', DEFAULT_NHC_INTEGRATION_STEPS)}")
        logger.info(f"Nose-Hoover Tdamp:         {s.get('nhc_thermo', DEFAULT_NHC_THERMO) * dt_fs:.4g} fs")
    if ensemble == 'npt':
        logger.info(f"Nose-Hoover Pdamp:         {s.get('nhc_baro', DEFAULT_NHC_BARO) * dt_fs:.4g} fs")
    if ensemble == 'nvt-langevin':
        logger.info(f"Center of Mass Motion Removal: {s.get('langevin_removecmmotion', DEFAULT_LANGEVIN_REMOVECMMOTION)}")
        _gamma = 1.0 / (dt_ps * s.get('langevin_thermo', DEFAULT_LANGEVIN_THERMO))
        logger.info(f"Langevin friction (gamma): {_gamma:.4g} 1/ps")
    logger.info(f"Random seed:               {s.get('seed', DEFAULT_SEED)}")
    logger.info(f"Observables:               {s.get('observables', DEFAULT_OBSERVABLES)}")
    if s.get('restart_load_path'):
        logger.info(f"Restart from:              {s['restart_load_path']}")
    if s.get('restart_save_path'):
        logger.info(f"Save restart to:           {s['restart_save_path']}")
    if s.get('relax_before_run'):
        logger.info(f"Geometry relaxation:       Enabled")
        if s.get('force_convergence'):
            logger.info(f"Force convergence:         {s['force_convergence']} eV/Å")
    else:
        logger.info(f"Geometry relaxation:       Disabled")
    if s.get('kspace_electrostatics'):
        logger.info(f"K-space method:            {s['kspace_electrostatics']}")
        if s.get('kspace_smearing'):
            logger.info(f"K-space smearing:          {s['kspace_smearing']}")
        if s.get('kspace_spacing'):
            logger.info(f"K-space spacing:           {s['kspace_spacing']}")
        if s.get('kspace_interp_nodes'):
            logger.info(f"K-space interp nodes:      {s['kspace_interp_nodes']}")
        if s.get('kspace_npt_cycles'):
            logger.info(f"K-space NPT cycles:        {s['kspace_npt_cycles']}")
    if s.get('output_atom_indices') is not None:
        logger.info(f"Output atom indices:       {s['output_atom_indices']}")
    if s.get('restart_reset'):
        logger.info(f"Restart reset:             {s['restart_reset']}")
    logger.info("=" * 60)

    # Dump the full merged settings dict (after YAML_DEFAULTS applied, CLI->settings
    # converted) so the log file alone is enough to reproduce the run. md_dt is
    # internally stored in ps; convert back to fs here so the dump matches the
    # YAML/CLI convention and the file is copy-pasteable into a settings YAML.
    logger.info("Full merged settings (md_dt in fs, matches YAML/CLI input):")
    for _k in sorted(s.keys()):
        # Skip derived keys that are recomputed at load time; printing them
        # would mislead a user copying the dump back into a YAML.
        if _k == 'output_format':
            continue
        _v = s[_k]
        if _k == 'md_dt':
            _v = _v * 1000  # ps -> fs for display, matches YAML/CLI convention
        elif _k == 'model_path' and _v is None:
            # CLI default is None; the real model resolved is _model (see above).
            _v = _model
        elif _k == 'observables' and isinstance(_v, tuple):
            # CLI's --observable is multiple=True (tuple); coerce so the dump
            # is a YAML list, not a Python tuple repr.
            _v = list(_v)
        logger.info(f"    {_k}: {_v}")
    logger.info("=" * 60)

# ============================================================================
# Shared MD simulation runner
# ============================================================================

def run_md_simulation(
    ensemble: str,
    input_file: Optional[str],
    output_file: Optional[str],
    log_file: Optional[str],
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    total_charge: int,
    num_unpaired_electrons: int,
    save_buffer: int,
    temperature: float,
    dt: float,
    md_cycles: int,
    md_steps: int,
    restart_save: Optional[str],
    restart_load: Optional[str],
    relax: bool,
    force_conv: Optional[float],
    seed: int,
    observable: Optional[tuple],
    help: bool,
    overwrite: bool = False,
    # Nose-Hoover thermostat (NVT/NPT)
    nhc_chain: Optional[int] = None,
    nhc_steps: Optional[int] = None,
    nhc_thermo: Optional[float] = None,
    nhc_sy_steps: Optional[int] = None,
    # Barostat (NPT)
    pressure: Optional[float] = None,
    nhc_baro: Optional[float] = None,
    # Langevin thermostat (NVT-Langevin)
    langevin_removecmmotion: Optional[bool] = None,
    langevin_thermo: Optional[float] = None,
    # K-space electrostatics
    kspace_electrostatics: Optional[str] = None,
    kspace_smearing: float = DEFAULT_KSPACE_SMEARING,
    kspace_spacing: float = DEFAULT_KSPACE_SPACING,
    kspace_interp_nodes: int = DEFAULT_KSPACE_INTERP_NODES,
    kspace_npt_cycles: Optional[int] = None,
    # Charge embedding: overrides only what the embedding sees, never
    # equilibration. Must be declared here -- nvt_md(**kwargs) forwards every
    # common_model_options flag straight into this explicit signature.
    charge_embed_total_charge: Optional[float] = None,
    charge_equilibration_total_charge: Optional[float] = None,
) -> None:
    """Shared runner for NVT, NVT-Langevin, NPT, and NVE MD simulations."""
    if not input_file or help:
        ctx = click.get_current_context()
        click.echo(SO3LR_ASCII)
        click.echo(ctx.get_help())
        return

    # Default output filename uses ensemble-specific suffix.
    # Note: NVT-Langevin uses the underscore form 'nvt_langevin' in filenames
    # (and YAML `ensemble:` values) but the hyphen form 'nvt-langevin' as the
    # CLI subcommand name (`so3lr nvt-langevin ...`); the mapping below keeps
    # the two conventions in sync.
    _suffix = {'nvt': 'nvt', 'nvt-langevin': 'nvt_langevin', 'npt': 'npt', 'nve': 'nve'}
    if output_file is None:
        output_file = f"{Path(input_file).stem}_{_suffix[ensemble]}.xyz"

    if log_file is None:
        # Use with_suffix so the log file lands next to --output, not in CWD.
        log_file = str(Path(output_file).with_suffix('.log'))

    setup_logger(log_file)
    logger.info(SO3LR_ASCII)

    settings = {
        'input_file': input_file,
        'output_file': output_file,
        'log_file': log_file,
        'model_path': model_path,
        'precision': precision,
        'md_dt': dt / 1000,
        'md_cycles': md_cycles,
        'md_steps': md_steps,
        'lr_cutoff': lr_cutoff,
        'dispersion_damping': dispersion_damping,
        'buffer_size_multiplier_sr': buffer_sr,
        'buffer_size_multiplier_lr': buffer_lr,
        'save_buffer': save_buffer,
        'total_charge': total_charge,
        'num_unpaired_electrons': num_unpaired_electrons,
        'seed': seed,
        'restart_load_path': restart_load,
        'restart_save_path': restart_save,
        'relax_before_run': relax,
        'force_convergence': force_conv,
        'min_n_min': DEFAULT_MIN_N_MIN,
        'min_start_dt': DEFAULT_MIN_START_DT,
        'min_max_dt': DEFAULT_MIN_MAX_DT,
        'min_cycles': DEFAULT_MIN_CYCLES,
        'min_steps': DEFAULT_MIN_STEPS,
        'ensemble': ensemble,
        'observables': observable,
        'output_format': infer_output_format(output_file),
        'overwrite': overwrite,
        'kspace_electrostatics': kspace_electrostatics,
        'kspace_smearing': kspace_smearing,
        'kspace_spacing': kspace_spacing,
        'kspace_interp_nodes': kspace_interp_nodes,
        'charge_embed_total_charge': charge_embed_total_charge,
        'charge_equilibration_total_charge': charge_equilibration_total_charge,
    }

    # Only insert kspace_npt_cycles when the user explicitly passed
    # --kspace-npt-cycles on the command line. The Click default is None,
    # which would otherwise override the documented 100-cycle fallback that
    # perform_md applies via all_settings.get('kspace_npt_cycles', 100) and
    # silently disable NPT k-space re-tuning for any CLI user who omits the
    # flag. The YAML path already gets the 100 default by omission; this
    # mirrors that behaviour for the CLI path. Uses the public Click API
    # (same pattern as the tune-ewald subcommand below).
    _ctx = click.get_current_context(silent=True)
    if _ctx is not None and _ctx.get_parameter_source('kspace_npt_cycles') == click.core.ParameterSource.COMMANDLINE:
        settings['kspace_npt_cycles'] = kspace_npt_cycles

    settings['md_T'] = temperature
    if ensemble == 'nve':
        settings['init_T'] = temperature

    if ensemble in ('nvt', 'npt'):
        settings.update({
            'nhc_chain_length': nhc_chain,
            'nhc_steps': nhc_steps,
            'nhc_thermo': nhc_thermo,
            'nhc_sy_steps': nhc_sy_steps,
        })

    if ensemble == 'npt':
        settings['md_P'] = pressure
        settings['nhc_baro'] = nhc_baro

    if ensemble == 'nvt-langevin':
        settings['langevin_removecmmotion'] = langevin_removecmmotion
        settings['langevin_thermo'] = langevin_thermo

    log_simulation_settings(settings)
    get_hardware_info()

    time_start = time.time()
    run(settings)
    time_end = time.time()
    logger.info("=" * 60)
    logger.info('Simulation completed successfully!')
    logger.info(f'Total runtime: {_format_runtime(time_end - time_start)}')


# Define the 'opt' subcommand
@cli.command(name='opt', help="Run geometry optimization with `so3lr opt --input geometry.xyz`.")
@common_io_options
@common_model_options
@kspace_options
# Optimization settings
@click.option('--min-cycles', '--min_cycles', 'min_cycles', default=DEFAULT_MIN_CYCLES, type=int,
              help=f'Number of minimization cycles to perform. [default: {DEFAULT_MIN_CYCLES}]')
@click.option('--min-steps', '--min_steps', 'min_steps', default=DEFAULT_MIN_STEPS, type=int,
              help=f'Number of steps per minimization cycle. [default: {DEFAULT_MIN_STEPS}]')
@click.option('--min-start-dt', '--dt-start', 'min_start_dt', default=DEFAULT_MIN_START_DT, type=float,
              help='The minimum step size during minimization. [default: 0.05]')
@click.option('--min-max-dt', '--dt-max', 'min_max_dt', default=DEFAULT_MIN_MAX_DT, type=float,
              help='The maximum step size during minimization. [default: 0.1]')
@click.option('--min-n-min', 'min_n_min', default=DEFAULT_MIN_N_MIN, type=int,
              help='Minimum number of steps moving in the correct direction before dt is updated. [default: 2]')
@click.option('--force-conv', default=None, type=float,
              help='Force convergence criterion in eV/Å, overrides --min-cycles and --min-steps. [default: None]')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def fire_optimization(
    input_file: Optional[str],
    output_file: Optional[str],
    log_file: Optional[str],
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    total_charge: int,
    num_unpaired_electrons: int,
    min_cycles: int,
    min_steps: int,
    min_start_dt: float,
    min_max_dt: float,
    min_n_min: int,
    force_conv: Optional[float],
    kspace_electrostatics: Optional[str],
    kspace_smearing: float,
    kspace_spacing: float,
    kspace_interp_nodes: int,
    kspace_npt_cycles: Optional[int],
    charge_embed_total_charge: Optional[float],
    charge_equilibration_total_charge: Optional[float],
    help: bool,
    overwrite: bool = False,
) -> None:
    """
    Run geometry optimization using the FIRE algorithm.

    This command performs geometry optimization of a molecular structure
    using the FIRE (Fast Inertial Relaxation Engine) algorithm with either
    the SO3LR potential or a custom MLFF model.

    Example:
        so3lr opt --input geometry.xyz
    """
    # Print help
    if not input_file or help:
        click.echo(SO3LR_ASCII)
        click.echo(fire_optimization.get_help(click.get_current_context()))
        return

    # HDF5 output is not supported for `opt` (the writer falls back to xyz
    # silently otherwise, leaving the file in a different location than the
    # banner advertises). Error out early so the user picks an .xyz/.extxyz path.
    if output_file is not None and Path(output_file).suffix.lower() in ('.hdf5', '.h5'):
        raise click.UsageError(
            f"--output '{output_file}': HDF5 output is not supported for `so3lr opt`. "
            f"Use an .xyz or .extxyz extension instead."
        )

    # Generate default output file name if not provided
    if input_file is not None and output_file is None:
        input_path = Path(input_file)
        output_file = f"{input_path.stem}_opt.xyz"

    # Generate default log file name based on output file if not explicitly provided
    if log_file is None:
        # Use with_suffix so the log file lands next to --output, not in CWD.
        log_file = str(Path(output_file).with_suffix('.log'))

    # Setup logging with default levels
    setup_logger(log_file)

    # Log the ASCII art
    logger.info(SO3LR_ASCII)

    logger.info("=" * 60)
    logger.info(f"SO3LR Geometry Optimization (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Initial geometry:          {input_file}")
    logger.info(f"Output geometry:           {output_file}")
    logger.info(f"Log file:                  {log_file}")
    logger.info(f"Total charge:              {total_charge}")
    logger.info(f"Spin (unpaired e-):        {num_unpaired_electrons}")
    # Mirror the MD / eval banner: log the resolved short name even when the
    # user did not pass --model (otherwise the user has no record of which
    # bundled checkpoint was loaded by default, e.g. so3lr-m vs. legacy v1).
    from so3lr.model_registry import DEFAULT_MODEL, available_models
    _model = model_path or DEFAULT_MODEL
    _kind = 'bundled' if _model in available_models() else 'custom path'
    logger.info(f"Force field:               {_model} ({_kind})")
    logger.info(f"Long-range cutoff:         {lr_cutoff} Å")
    
    if force_conv is not None:
        logger.info(f"Force convergence:         {force_conv} eV/Å")
        
    else:
        logger.info(f"Minimization cycles:       {min_cycles} cycles, each {min_steps} steps")
        
    logger.info(f"Initial step size:         {min_start_dt} Å")
    logger.info(f"Maximum step size:         {min_max_dt} Å")
    logger.info(f"Min steps between updates: {min_n_min}")
    logger.info(f"Precision:                 {precision}")
    logger.info(f"Dispersion damping:        {dispersion_damping} Å")
    logger.info(f"Short-range buffer:        {buffer_sr}")
    logger.info(f"Long-range buffer:         {buffer_lr}")
    logger.info("=" * 60)

    # Log hardware info
    get_hardware_info()

    # Create settings dictionary
    settings = {
        'input_file': input_file,
        'output_file': output_file,
        'model_path': model_path,
        'min_cycles': min_cycles,
        'min_steps': min_steps,
        'min_start_dt': min_start_dt,
        'min_max_dt': min_max_dt,
        'min_n_min': min_n_min,
        'force_convergence': force_conv,
        'precision': precision,
        'lr_cutoff': lr_cutoff,
        'dispersion_damping': dispersion_damping,
        'buffer_size_multiplier_sr': buffer_sr,
        'buffer_size_multiplier_lr': buffer_lr,
        'total_charge': total_charge,
        'num_unpaired_electrons': num_unpaired_electrons,
        'kspace_electrostatics': kspace_electrostatics,
        'kspace_smearing': kspace_smearing,
        'kspace_spacing': kspace_spacing,
        'kspace_interp_nodes': kspace_interp_nodes,
        'charge_embed_total_charge': charge_embed_total_charge,
        'charge_equilibration_total_charge': charge_equilibration_total_charge,
        'overwrite': overwrite,
    }

    # Add log settings to the settings dictionary
    settings['log_file'] = log_file

    # Set format to extxyz for optimization, since they are usually short
    settings['output_format'] = infer_output_format(output_file)

    # Run optimization
    time_start = time.time()
    perform_min(settings)
    time_end = time.time()
    logger.info("=" * 60)
    logger.info('Optimization completed successfully!')
    logger.info(f'Total runtime: {_format_runtime(time_end - time_start)}')

# Define the 'nvt' subcommand
@cli.command(name='nvt', help="Run NVT molecular dynamics simulation with `so3lr nvt --input geometry.xyz`.")
@common_io_options
@common_model_options
@kspace_options
@common_md_options
@thermostat_options
def nvt_md(**kwargs) -> None:
    """
    Run NVT (constant volume and temperature) MD simulation using the Nose-Hoover chain thermostat.

    Example:
        so3lr nvt --input geometry.xyz --temperature 300
    """
    run_md_simulation(ensemble='nvt', **kwargs)


# Define the 'nvt-langevin' subcommand
@cli.command(name='nvt-langevin', help="Run NVT (Langevin thermostat) molecular dynamics simulation with `so3lr nvt-langevin --input geometry.xyz`.")
@common_io_options
@common_model_options
@kspace_options
@common_md_options
@langevin_options
def nvt_langevin_md(**kwargs) -> None:
    """
    Run NVT (constant volume and temperature) MD simulation using the Langevin thermostat.

    Example:
        so3lr nvt-langevin --input geometry.xyz --temperature 300
    """
    run_md_simulation(ensemble='nvt-langevin', **kwargs)


# Define the 'npt' subcommand
@cli.command(name='npt', help="Run NPT molecular dynamics simulation with `so3lr npt --input geometry.xyz`.")
@common_io_options
@common_model_options
@kspace_options
@common_md_options
@thermostat_options
@barostat_options
def npt_md(**kwargs) -> None:
    """
    Run NPT (constant pressure and temperature) MD simulation using the Nose-Hoover chain thermostat and barostat.

    Note: NPT simulations require a periodic cell in the input structure.

    Example:
        so3lr npt --input geometry.xyz --temperature 300 --pressure 1
    """
    run_md_simulation(ensemble='npt', **kwargs)


# Define the 'nve' subcommand
@cli.command(name='nve', help="Run NVE molecular dynamics simulation with `so3lr nve --input geometry.xyz`.")
@common_io_options
@common_model_options
@kspace_options
@common_md_options
def nve_md(**kwargs) -> None:
    """
    Run NVE (constant volume and energy) MD simulation with no thermostat.

    Note: --temperature sets the initial-velocity-sampling temperature in
    Kelvin only; NVE has no thermostat, so the kinetic temperature drifts
    freely during the trajectory. [default: 300.0]

    Example:
        so3lr nve --input geometry.xyz --temperature 300
    """
    run_md_simulation(ensemble='nve', **kwargs)


# Define the 'eval' subcommand
@cli.command(name='eval', help="Evaluate SO3LR model on a dataset with `so3lr eval --datafile dataset.extxyz`.")
# Input/Output group
@click.option('--datafile', '--data-file', '--datafile_path', 'datafile',
              type=click.Path(exists=True, dir_okay=False), required=False,
              help='Path to dataset file containing structures to evaluate '
                   '(extxyz or any ASE-readable format, e.g. xyz, pdb, cif).')
@click.option('--save-to', type=click.Path(), default=None,
              help='Output file to save predictions (.extxyz format). If not provided, predictions are not saved.')
@click.option('--log-file', default=None, type=click.Path(),
              help='File to write logs to [default: None].')
@click.option('--overwrite/--no-overwrite', 'overwrite', default=False,
              help='If the --save-to output already exists, overwrite it. '
                   'Default is to refuse (rename your previous run, or pass --overwrite).')
# Model settings group
@click.option('--model', '--model_path', 'model_path', type=str, default=None,
              callback=_validate_model_choice,
              help='Model to use: a bundled name (so3lr-s, so3lr-m, so3lr-l, so3lr) '
                   'or a path to a custom MLFF model directory. '
                   'If not provided, so3lr-m is used. [default: so3lr-m]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Dispersion interactions start to switch off at (lr_cutoff - dispersion_damping) Å. [default: {DEFAULT_DISPERSION_DAMPING}].')
@click.option('--total-charge', 'total_charge', default=None, type=int,
              help='Override total charge for every structure in the dataset. '
                   'If omitted, the per-structure `charge=` field in extxyz info{} is used '
                   '(defaulting to 0 when absent).')
@click.option('--spin', 'num_unpaired_electrons', default=None, type=int,
              help='Override spin (number of unpaired electrons = multiplicity - 1) for every '
                   'structure. If omitted, the per-structure `multiplicity=` field in extxyz '
                   'info{} is used (defaulting to 1 when absent, i.e. singlet).')
# Evaluation settings
@click.option('--batch-size', default=1, type=int,
              help='Number of molecules per batch. '
                   '[default: 1; consider 16-64 on GPU for throughput on large datasets]')
@click.option('--targets', default='forces,dipole_vec,hirshfeld_ratios', type=str,
              help='Comma-separated list of targets to evaluate. '
                   'Standard targets: forces, energy, dipole_vec, hirshfeld_ratios '
                   '(canonical; aliases hirsh_ratios and a0_ratios are accepted, '
                   'and the new so3lr-s/m/l models report ratios as a0_ratios). '
                   'Observables (saved to output, metrics only if ground truth exists): '
                   'nn_energy, partial_charges (alias: charges), electrostatic_energy, dispersion_energy, '
                   'nlh_repulsion (canonical for so3lr-s/m/l; legacy models use zbl_repulsion), '
                   'electrostatic_energy_undamped, dispersion_energy_undamped. '
                   'Embeddings: x (final-layer per-atom features, saved to <output>_x.npz). '
                   '[default: forces,dipole_vec,hirshfeld_ratios]')
@click.option('--jit-compile/--no-jit-compile', 'jit_compile', default=True,
              help='Use JIT compilation for speed. [default: enabled]')
@click.option('--charge-embed-total-charge', 'charge_embed_total_charge', default=None, type=float,
              help='Total charge seen by the charge EMBEDDING only. Unset (default) '
                   'follows --total-charge; 0 embeds a neutral system. Charge '
                   'equilibration always uses the true total charge.')
@click.option('--charge-equilibration-total-charge', 'charge_equilibration_total_charge', default=None, type=float,
              help='Total charge that charge EQUILIBRATION conserves to. Unset (default) '
                   'follows --total-charge; 0 conserves to neutral. Independent of '
                   '--charge-embed-total-charge.')
# Help option
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def eval_model(
    # Input/Output group
    datafile: str,
    save_to: Optional[str],
    log_file: Optional[str],
    # Model settings group
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    total_charge: Optional[int],
    num_unpaired_electrons: Optional[int],
    charge_embed_total_charge: Optional[float],
    charge_equilibration_total_charge: Optional[float],
    # Evaluation settings
    batch_size: int,
    targets: str,
    jit_compile: bool,
    # Output policy
    overwrite: bool,
    # Help option
    help: bool
) -> None:
    """
    Evaluate SO3LR model on a dataset.

    This command evaluates the SO3LR model on a dataset of molecular structures.
    It calculates metrics for the specified targets and can optionally save the
    predictions to a file in the extxyz format.

    Example:
        so3lr eval --datafile dataset.extxyz --save-to predictions.extxyz
    """
    # Print help if needed
    if not datafile or help:
        click.echo(SO3LR_ASCII)
        click.echo(eval_model.get_help(click.get_current_context()))
        return

    # Set default output file name if needed
    output_file = save_to if save_to else f"{Path(datafile).stem}_eval.extxyz"

    # Generate default log file name based on output file if not explicitly provided.
    # Only auto-derive when predictions are actually being saved: without --save-to
    # we have no real output basename to anchor the log to, and creating a
    # `<datafile>_eval.log` next to an extxyz that is never written is misleading.
    # The documented default for --log-file is None (console-only) — honor that.
    if log_file is None and save_to is not None:
        # Use with_suffix so the log file lands next to --output, not in CWD.
        log_file = str(Path(output_file).with_suffix('.log'))

    # Setup logging with default levels
    setup_logger(log_file)

    # Log ASCII art
    logger.info(SO3LR_ASCII)

    # Log all settings
    logger.info("=" * 60)
    logger.info(f"SO3LR Model Evaluation (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Dataset file:              {datafile}")
    logger.info(f"Output file:               {output_file if save_to else 'Not saving'}")
    logger.info(f"Log file:                  {log_file}")
    logger.info(f"Batch size:                {batch_size}")
    # Mirror the MD-side banner: show the resolved short name (so the user knows
    # which bundled checkpoint actually loaded — so3lr-s/m/l/so3lr — instead of
    # the generic 'SO3LR' label) and tag it as bundled vs. a custom-path model.
    from so3lr.model_registry import DEFAULT_MODEL, available_models
    _model = model_path or DEFAULT_MODEL
    _kind = 'bundled' if _model in available_models() else 'custom path'
    logger.info(f"Force field:               {_model} ({_kind})")
    logger.info(f"Precision:                 {precision}")
    logger.info(f"Long-range cutoff:         {lr_cutoff} Å")
    logger.info(f"Dispersion damping:        {dispersion_damping} Å")
    logger.info(f"Total charge:              {'(from info{} or 0)' if total_charge is None else total_charge}")
    logger.info(f"Spin (unpaired e-):        {'(from info{} or 0)' if num_unpaired_electrons is None else num_unpaired_electrons}")
    logger.info(f"JIT compilation:           {'Enabled' if jit_compile else 'Disabled'}")
    if save_to:
        logger.info(f"Saving to:                 {save_to}")
        logger.info(f"Overwrite if exists:       {'Yes' if overwrite else 'No (default)'}")
    logger.info(f"Targets:                   {targets}")
    logger.info("=" * 60)

    # Log hardware info
    get_hardware_info()

    # Validate file existence
    if not Path(datafile).exists():
        logger.error(f"Error: Dataset file not found: {datafile}")
        sys.exit(1)

    # Validate output path. Honors --overwrite so eval, opt, MD and finetune
    # all use the same opt-in policy (refuse by default; pass --overwrite to
    # truncate). The downstream evaluate_so3lr_on() repeats this check for
    # callers that bypass the CLI.
    if save_to:
        save_path = Path(save_to)
        if save_path.exists() and not overwrite:
            logger.error(
                f"Output file already exists: {save_to}. Refusing to overwrite. "
                f"Pass --overwrite to overwrite, or choose a different --save-to."
            )
            sys.exit(1)
        # Create parent directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Call the evaluate_so3lr_on function from so3lr_eval.py
    evaluate_so3lr_on(
        datafile=datafile,
        batch_size=batch_size,
        lr_cutoff=lr_cutoff,
        dispersion_damping=dispersion_damping,
        jit_compile=jit_compile,
        save_to=save_to,
        model_path=model_path,
        precision=precision,
        targets=targets,
        log_file=log_file,
        total_charge=total_charge,
        num_unpaired_electrons=num_unpaired_electrons,
        overwrite=overwrite,
        charge_embed_total_charge=charge_embed_total_charge,
        charge_equilibration_total_charge=charge_equilibration_total_charge,
    )
    logger.info("=" * 60)
    logger.info("Evaluation completed successfully!")

    if save_to:
        logger.info(f"Predictions saved to: {save_to}")


# Define the 'finetune' subcommand
@cli.command(name='finetune', help="Finetune SO3LR model on a dataset with `so3lr finetune --datafile dataset.extxyz --workdir finetune_so3lr --num-train 50 --num-valid 10`.")
# Input/Output group
@click.option('--workdir', '--work-dir', '--workdir_path', 'workdir',
              type=click.Path(), help='Working directory for fine-tuned model checkpoints and logs.')
@click.option('--datafile', '--data-file', '--datafile_path', 'datafile',
              type=click.Path(), help='Path to dataset file (extxyz or tfds format) containing data used for finetuning.')
@click.option('--log-file', default=None, type=click.Path(),
              help='File to write logs to [default: None].')
# Finetuning settings.
@click.option('--strategy', type=click.Choice(['full','final_mlp','final_mlp_and_hirshfeld','hirshfeld','last_layer','last_layer_and_final_mlp','first_layer','first_layer_and_last_layer','obs_0','obs_023','obs_23','obs_3','ll_fmlp','ll_obs_023','freeze_long_range','forces_only']),
              default='full', help='Strategy for finetuning. [default: full]')
@click.option('--config', '--config_path', 'config_path', type=click.Path(exists=False), default=None,
              help='Path to the finetuning config (YAML only; unlike `so3lr train --config`, JSON is not supported here). '
                   'If not provided, defaults to so3lr/config/finetune.yaml shipped with the package. [default: None]')
@click.option('--model', '--model_path', 'model_path', type=click.Path(exists=False), default=None,
              callback=_validate_model_choice,
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
# Training settings
@click.option('--num-train', '--num_train', default=None, type=int, help='Number of data points used for gradient calculations.')
@click.option('--num-valid', '--num_valid', default=None, type=int, help='Number of data points used for validation.')
# Help option
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def finetune_model(
    # Input/Output group
    workdir: str,
    datafile: str,
    log_file: Optional[str],
    # Finetuning settings.
    strategy: str,
    model_path: Optional[str],
    config_path: Optional[str],
    # Training settings
    num_train: Optional[int],
    num_valid: Optional[int],
    # Help option
    help: bool
) -> None:
    """
    Finetune SO3LR or MLFF model on a dataset.

    This command finetunes the SO3LR model or another MLFF model on a dataset of molecular structures.

    Example:
        so3lr finetune --datafile dataset.extxyz --workdir finetune_so3lr --num-train 50 --num-valid 10
    """
    # Print help if needed (and reject the trivial `--workdir foo` invocation
    # that previously crashed with TypeError because `datafile` was None).
    if not datafile or not workdir or help:
        click.echo(SO3LR_ASCII)
        click.echo(finetune_model.get_help(click.get_current_context()))
        return

    # Generate default log file name based on output file if not explicitly provided
    if log_file is None:
        log_file = Path(workdir).resolve() / "finetuning.log"

    # Setup logging with default levels. The log file creates the workdir.
    setup_logger(log_file)

    # Log ASCII art
    logger.info(SO3LR_ASCII)

    # Log all settings
    logger.info("=" * 60)
    logger.info(f"SO3LR Model Finetuning (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Dataset file:              {datafile}")
    logger.info(f"Workdir:                   {workdir}")
    logger.info(f"Finetuning strategy:       {strategy}")
    logger.info(f"Force field:               {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        logger.info(f"Model path:            {model_path}")
    logger.info(f"Config:                    {'Custom' if config_path else 'Default'}")
    if config_path is not None:
        logger.info(f"Config path:           {config_path}")
    logger.info(f"Number training points:    {num_train}")
    logger.info(f"Number validation points:  {num_valid}")
    logger.info("=" * 60)

    # Log hardware info
    get_hardware_info()

    # Validate file existence
    if not Path(datafile).exists():
        logger.error(f"Error: Dataset file not found: {datafile}")
        sys.exit(1)

    finetune_so3lr(
        datafile=datafile,
        workdir=workdir,
        strategy=strategy,
        num_train=num_train,
        num_valid=num_valid,
        model_path=model_path,
        config_path=config_path,
        log_file=log_file
    )
    logger.info("=" * 60)
    logger.info("finetuning completed successfully!")


@cli.command(name='train', help="Train a SO3krates model from a config file with `so3lr train --config config.yaml`.")
@click.option('--config', 'config_path', type=click.Path(exists=True, dir_okay=False), default=None,
              help='Path to the training config file (YAML or JSON).')
@click.option('--log-file', default=None, type=click.Path(),
              help='File to write logs to [default: None].')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def train_model(
    config_path: Optional[str],
    log_file: Optional[str],
    help: bool
) -> None:
    """
    Train a SO3krates model from a config file.

    Example:
        so3lr train --config config.yaml
    """
    if help or not config_path:
        click.echo(SO3LR_ASCII)
        click.echo(train_model.get_help(click.get_current_context()))
        return

    # Validate file existence
    if not Path(config_path).exists():
        logger.error(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Setup logging: default the log file to the config name with a .log
    # extension (e.g. `config.yaml` -> `config.log`).
    if log_file is None:
        log_file = str(Path(config_path).with_suffix('.log'))

    setup_logger(log_file)

    # Log ASCII art
    logger.info(SO3LR_ASCII)

    # Log all settings
    logger.info("=" * 60)
    logger.info(f"SO3LR Model Training (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Config file:               {config_path}")
    logger.info(f"Log file:                  {log_file}")
    logger.info("=" * 60)

    # Log hardware info
    get_hardware_info()

    # Run training
    time_start = time.time()
    train_so3lr(config_path=config_path, log_file=log_file)
    time_end = time.time()
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Total runtime: {_format_runtime(time_end - time_start)}")


# Define the 'tune-ewald' subcommand
@cli.command(name='tune-ewald', help="Tune k-space electrostatics parameters (Ewald/PME/P3M): `so3lr tune-ewald --input geometry.xyz`.")
@click.option('--settings', type=click.Path(exists=True, dir_okay=False), default=None,
              help='Path to YAML settings file.')
@common_io_options
@common_model_options
@kspace_options
@click.option('--jit-compile/--no-jit-compile', 'jit_compile', default=True,
              help='Use JIT compilation for speed. [default: enabled]')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
@click.pass_context
def tune_ewald(
    ctx: click.Context,
    settings: Optional[str],
    # Input/Output group
    input_file: str,
    output_file: Optional[str],
    log_file: Optional[str],
    overwrite: bool,
    # Model settings group
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    total_charge: int,
    num_unpaired_electrons: int,
    jit_compile: bool,
    kspace_electrostatics: str,
    kspace_smearing: float,
    kspace_spacing: float,
    kspace_interp_nodes: int,
    kspace_npt_cycles: Optional[int],
    # Accepted because common_model_options supplies it; k-space tuning does
    # not apply it, since what is being tuned is grid convergence.
    charge_embed_total_charge: Optional[float],
    charge_equilibration_total_charge: Optional[float],
    # Help option
    help: bool
) -> None:
    """
    Tune k-space electrostatics parameters (Ewald/PME/P3M) for running SO3LR model on a given structure.

    Supports three methods:
    - ewald: Traditional Ewald summation
    - pme: Particle Mesh Ewald (faster, uses Lagrange interpolation)
    - p3m: Particle-Particle Particle-Mesh (uses B-spline interpolation)

    Example:
        so3lr tune-ewald --input geometry.xyz --kspace-electrostatics pme
    """
     # Print help if needed (and handle the no-args invocation that
     # previously fell through and crashed when input_file was None).
    if (not input_file and not settings) or help:
        click.echo(SO3LR_ASCII)
        click.echo(tune_ewald.get_help(click.get_current_context()))
        return

    # Load settings from file if provided
    settings_dict: Dict[str, Any] = {}
    if settings is not None:
        try:
            settings_path = Path(settings).expanduser().resolve()
            if not settings_path.exists():
                raise FileNotFoundError(f'Settings file not found: {settings}')

            with open(settings_path, 'r') as f:
                settings_dict = yaml.safe_load(f) or {}

            logger.info(f"Loading settings from {settings}")
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Error loading settings file: {str(e)}")
            sys.exit(1)

        # Same alias + unknown-key validation as the top-level --settings path.
        apply_yaml_key_aliases(settings_dict)
        validate_yaml_keys(settings_dict, settings_path)

    # CLI params with internal names; buffer_sr/buffer_lr map to longer internal names
    _cli_params = {
        'input_file':                input_file,
        'output_file':               output_file,
        'log_file':                  log_file,
        'model_path':                model_path,
        'precision':                 precision,
        'lr_cutoff':                 lr_cutoff,
        'dispersion_damping':        dispersion_damping,
        'buffer_size_multiplier_sr': buffer_sr,
        'buffer_size_multiplier_lr': buffer_lr,
        'total_charge':              total_charge,
        'jit_compile':               jit_compile,
        'kspace_electrostatics':     kspace_electrostatics,
    }
    # CLI param names that differ from their internal key
    _cli_key_for = {'buffer_size_multiplier_sr': 'buffer_sr', 'buffer_size_multiplier_lr': 'buffer_lr'}
    for internal_key, val in _cli_params.items():
        cli_key = _cli_key_for.get(internal_key, internal_key)
        # Use the public Click API rather than the private `_parameter_source`
        # mapping; both behave identically on Click 8.x but the public method
        # is the documented forward-compatible accessor.
        if ctx.get_parameter_source(cli_key) == click.core.ParameterSource.COMMANDLINE:
            settings_dict[internal_key] = val
        else:
            settings_dict.setdefault(internal_key, val)

    # Default the log file to the settings YAML name with a .log extension when
    # a settings file is given (e.g. `tune.yaml` -> `tune.log`); otherwise fall
    # back to the input-geometry name.
    if settings_dict.get('log_file') is None:
        if settings is not None:
            settings_dict['log_file'] = str(Path(settings).with_suffix('.log'))
        else:
            settings_dict['log_file'] = f"{Path(input_file).stem}_tune-kspace.log"

    # Setup logging with default levels
    setup_logger(settings_dict.get('log_file'))

    # Log the ASCII art
    logger.info(SO3LR_ASCII)
    model_path = settings_dict.get('model_path')
    logger.info("=" * 60)
    logger.info(f"Tuner for k-space electrostatics for SO3LR (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Initial geometry:          {settings_dict.get('input_file')}")
    logger.info(f"Log file:                  {settings_dict.get('log_file')}")
    logger.info(f"Force field:               {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        logger.info(f"Model path:                {model_path}")
    logger.info(f"Precision:                 {settings_dict.get('precision')}")
    logger.info(f"Long-range cutoff:         {settings_dict.get('lr_cutoff')} Å")
    logger.info(f"Total charge:              {settings_dict.get('total_charge')}")
    logger.info(f"Spin (unpaired e-):        {settings_dict.get('num_unpaired_electrons')}")
    logger.info(f"JIT compilation:           {'Enabled' if settings_dict.get('jit_compile') else 'Disabled'}")


    # Log hardware info
    get_hardware_info()

    # Run tuning
    time_start = time.time()
    tune(settings_dict)
    time_end = time.time()
    logger.info("=" * 60)
    logger.info('Tuning completed successfully!')
    logger.info(f'Total runtime: {_format_runtime(time_end - time_start)}')


def main() -> None:
    """Entry point for the command line interface."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2') # Suppress TF logging

    cli(standalone_mode=False)


if __name__ == '__main__':
    main()
