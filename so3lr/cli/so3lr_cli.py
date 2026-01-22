"""Command line interfaces for SO3LR."""
import os
import sys
import jax
import ase
import time
import yaml
import click
import logging
import warnings
import platform
import numpy as np

from pathlib import Path
from ase.io import read, write
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from .. import __version__
from .so3lr_eval import evaluate_so3lr_on
from .so3lr_finetune import finetune_so3lr
from .so3lr_md import perform_min, run, setup_logger
from .tune_ewald import tune

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
    devices = jax.devices()
    backend = jax.default_backend()

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
                mem_info = jax.devices()[i].memory_stats()
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
DEFAULT_DISPERSION_DAMPING = 2.0  # Angstrom
DEFAULT_BUFFER_MULTIPLIER = 1.25
DEFAULT_SAVE_BUFFER = 50
DEFAULT_OUTPUT_FORMAT = 'extxyz'
DEFAULT_TIMESTEP = 0.5  # femtoseconds
DEFAULT_TEMPERATURE = 300  # Kelvin
DEFAULT_PRESSURE = 1.0  # atmospheres

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

# Help strings for CLI
FULL_HELP_STRING = """
Run simulations using SO3LR Machine Learned Force Field.

## Commands

so3lr opt [options]     Run geometry optimization
so3lr nvt [options]     Run NVT (constant volume and temperature) MD simulation
so3lr npt [options]     Run NPT (constant pressure and temperature) MD simulation
so3lr nve [options]     Run NVE (constant volume and energy) MD simulation
so3lr eval [options]    Run evaluation on a dataset

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
# md_P: 1.0                                    # Pressure in atmospheres (enables NPT simulation)
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
# kspace_electrostatics: null                  # Whether to use Ewald summation for long-range electrostatics (null/"ewald"/"pme")
# kspace_smearing: 4.0                         # Smearing width for Ewald summation
# kspace_spacing: 2.0                          # Grid spacing for Ewald summation
# kspace_interp_nodes: 4                       # Number of interpolation nodes for PME
#
# observables: ["partial_charges","dipole_vec","hirshfeld_ratios"]
#
# output_atom_indices: null                    # Optional list of atom indices to output
# ```

PARAM_MAP = {
    'input_file': 'input_file',
    'output_file': 'output_file',
    'log_file': 'log_file',
    'model_path': 'model_path',
    'precision': 'precision',
    'dt':'md_dt',
    'temperature': 'md_T',
    'pressure':'md_P',
    'md_cycles': 'md_cycles',
    'md_steps': 'md_steps',
    'lr_cutoff': 'lr_cutoff',
    'dispersion_damping': 'dispersion_damping',
    'buffer_sr':'buffer_size_multiplier_sr',
    'buffer_lr':'buffer_size_multiplier_lr',
    'save_buffer': 'save_buffer',
    'total_charge': 'total_charge',
    'seed': 'seed',
    'restart_load': 'restart_load_path',
    'restart_save': 'restart_save_path',
    'relax': 'relax_before_run',
    'force_conv': 'force_convergence',
    # Additional MD settings
    'nhc_chain': 'nhc_chain_length',
    'nhc_steps': 'nhc_steps',
    'nhc_thermo': 'nhc_thermo',
    'nhc_baro': 'nhc_baro',
    'nhc_sy_steps': 'nhc_sy_steps',
    'langevin_removecmmotion': 'langevin_removecmmotion',
    'langevin_thermo': 'langevin_thermo',
    # Use default optimization settings
    'min_n_min': 'min_n_min',
    'min_start_dt': 'min_start_dt',
    'min_max_dt': 'min_max_dt',
    'min_cycles': 'min_cycles',
    'min_steps': 'min_steps',
    'kspace_type': 'kspace_electrostatics',
    'jit_compile': 'jit_compile'
}


BASIC_HELP_STRING = """
Run simulations using SO3LR Machine Learned Force Field.

## Commands

so3lr opt [options]          Run geometry optimization
so3lr nvt [options]          Run NVT (constant volume and temperature) MD simulation
so3lr npt [options]          Run NPT (constant pressure and temperature) MD simulation
so3lr nve [options]          Run NVT (constant volume and energy) MD simulation
so3lr eval [options]         Evaluate SO3LR model on a dataset
so3lr finetune [options]     Finetune SO3LR model on a dataset

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

Use --help-full to see all available options.

"""

# so3lr [options]         Run with options specified via command line or settings file
# Run with settings file:
#   so3lr --settings md_settings.yaml

def infer_output_format(output_file):
    """Determine output format based on file extension."""
    if output_file.endswith('.hdf5'):
        return 'hdf5'
    elif output_file.endswith('.xyz') or output_file.endswith('.extxyz'):
        return 'extxyz'
    else:
        logger.warning(f"Unknown output extension for {output_file}, defaulting to extxyz format")
        return 'extxyz'

def update_settings_from_defaults(ctx: click.Context, settings_dict: Dict[str, Any], key= str) -> None:
    """Update settings dictionary with default values."""
    param = ctx.params[key]
    if key in PARAM_MAP:
        if PARAM_MAP[key] in settings_dict:
            if ctx._parameter_source[key] == click.core.ParameterSource.COMMANDLINE:
                if param != settings_dict[PARAM_MAP[key]]:
                    logger.warning(f"settings.yaml: {PARAM_MAP[key]} is overridden by explicit command line argument.")
                    settings_dict[PARAM_MAP[key]] = param
                # Override settings with command line arguments if provided
                settings_dict[PARAM_MAP[key]] = param
        else:
            # If not provided, set default values
            if param is not None:
                settings_dict[PARAM_MAP[key]] = param
    return settings_dict

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


class NVTNPTGroup(CustomCommandClass):
    """Custom group to handle --nve, --nvt and --npt flags that set appropriate defaults."""

    def parse_args(self, ctx: click.Context, args: List[str]) -> List[str]:
        """Parse arguments with special handling for --nvt and --npt flags."""
        # Check for simulation mode flags and modify args accordingly
        if '--nve' in args:
            args.remove('--nve')
            if '--pressure' in args:
                # Remove any pressure argument to ensure NVT mode
                pressure_index = args.index('--pressure')
                if pressure_index < len(args) - 1 and not args[pressure_index + 1].startswith('--'):
                    # Remove the value too
                    args.pop(pressure_index + 1)
                args.remove('--pressure')
                ctx.command.params_map['pressure'].default = None
        if '--nvt' or '--nvt-langevin' in args:
            args.remove('--nvt')
            if '--pressure' in args:
                # Remove any pressure argument to ensure NVT mode
                pressure_index = args.index('--pressure')
                if pressure_index < len(args) - 1 and not args[pressure_index + 1].startswith('--'):
                    # Remove the value too
                    args.pop(pressure_index + 1)
                args.remove('--pressure')
                ctx.command.params_map['pressure'].default = None

        if '--npt' in args:
            args.remove('--npt')
            # If no pressure specified, use default of 1.0 atm
            if '--pressure' not in args:
                args.extend(['--pressure', str(DEFAULT_PRESSURE)])

        # Call the super parse_args
        return super().parse_args(ctx, args)

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        """Get a command from this group and set appropriate help options."""
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            cmd.context_settings["help_option_names"] = ["--help"]
        return cmd



    def canonicalize_ensemble(self, settings_dict: Dict[str, Any]):
        """Canonicalize the ensemble based on the provided settings."""
        if int(settings_dict.get('nve', False)) + int(settings_dict.get('nvt', False)) + int(settings_dict.get('nvt-langevin', False)) + int(settings_dict.get('npt', False)) > 1:
            raise ValueError("Only one of --nve, --nvt, --nvt-langevin or --npt can be specified.")
        if settings_dict.get('ensemble') is None:
            if settings_dict.get('npt') is True or settings_dict.get('md_P') is not None:
                if settings_dict.get('md_P') is not None:
                    settings_dict['ensemble'] = 'npt'
                else:
                    raise ValueError("Pressure must be specified for NPT ensemble.")
            elif settings_dict.get('nve') is True:
                settings_dict['ensemble'] = 'nve'
            elif settings_dict.get('nvt-langevin') is True:
                settings_dict['ensemble'] = 'nvt-langevin'
            else: # NVT is default
                settings_dict['ensemble'] = 'nvt'
        return settings_dict

@click.group(cls=NVTNPTGroup, invoke_without_command=True,
             help="Run molecular dynamics using SO3LR Machine Learned Force Field.",
             context_settings={"help_option_names": []})
@click.option('--settings', type=click.Path(exists=False), default=None,
              help='Path to YAML settings file.')
# Input/Output
@click.option('--input', 'input_file', type=click.Path(exists=False), default=None,
              help='Input geometry file (any ASE-readable format).')
@click.option('--output', 'output_file', type=click.Path(), default=None,
              help='Output trajectory file (.hdf5 or .xyz).')
@click.option('--log-file', default=None, type=click.Path(),
              help=f'File to write logs to [default: None].')
# Model settings
@click.option('--model', 'model_path', type=click.Path(exists=False), default=None,
              help='Path to MLFF model directory. If not provided, pretrained SO3LR model is used.')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations [default: {DEFAULT_PRECISION}].')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å [default: {DEFAULT_LR_CUTOFF} Å].')
@click.option('--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Dispersion interactions start to switch off at (lr_cutoff - dispersion_damping) Å. [default: {DEFAULT_DISPERSION_DAMPING}].')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for short-range interactions [default: {DEFAULT_BUFFER_MULTIPLIER}].')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for long-range interactions [default: {DEFAULT_BUFFER_MULTIPLIER}].')
@click.option('--save-buffer', default=DEFAULT_SAVE_BUFFER, type=int,
              help=f'Number of frames to buffer before writing [default: {DEFAULT_SAVE_BUFFER}].')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
              help=f'Total charge of the system [default: {DEFAULT_TOTAL_CHARGE}].')
# MD settings
@click.option('--dt', default=DEFAULT_TIMESTEP, type=float,
              help=f'MD timestep in femtoseconds [default: {DEFAULT_TIMESTEP} fs].')
@click.option('--temperature', default=DEFAULT_TEMPERATURE, type=float,
              help=f'Simulation temperature in Kelvin [default: {DEFAULT_TEMPERATURE} K].')
@click.option('--pressure', default=None, type=float,
              help='Simulation pressure in atmospheres (enables NPT) [default: None].')
@click.option('--md-cycles', default=DEFAULT_MD_CYCLES, type=int,
              help=f'Number of MD cycles to run [default: {DEFAULT_MD_CYCLES}].')
@click.option('--md-steps', default=DEFAULT_MD_STEPS_PER_CYCLE, type=int,
              help=f'Number of steps per MD cycle [default: {DEFAULT_MD_STEPS_PER_CYCLE}].')
# Optimization settings
@click.option('--min-cycles', '--min_cycles', 'min_cycles', default=DEFAULT_MIN_CYCLES, type=int,
              help=f'Number of minimization cycles to perform. [default: {DEFAULT_MIN_CYCLES}]')
@click.option('--min-steps', '--min_steps', 'min_steps', default=DEFAULT_MIN_STEPS, type=int,
              help=f'Number of steps per minimization cycle. [default: {DEFAULT_MIN_STEPS}]')
@click.option('--min-start-dt', default=DEFAULT_MIN_START_DT, type=float,
              help=f'Minimum initial timestep for minimization [default: {DEFAULT_MIN_START_DT}].')
@click.option('--min-max-dt', default=DEFAULT_MIN_MAX_DT, type=float,
              help=f'Maximum timestep for minimization [default: {DEFAULT_MIN_MAX_DT}].')
@click.option('--min-n-min', default=DEFAULT_MIN_N_MIN, type=int,
              help=f'Minimum number of minimization steps [default: {DEFAULT_MIN_N_MIN}].')
@click.option('--force-conv', default=None, type=float,
              help='Force convergence criterion in eV/Å for initial relaxation [default: None].')
# Nose-Hoover Thermostat/Barostat settings
@click.option('--nhc-chain', default=DEFAULT_NHC_CHAIN_LENGTH, type=int,
              help=f'Length of the Nose-Hoover thermostat chain [default: {DEFAULT_NHC_CHAIN_LENGTH}].')
@click.option('--nhc-steps', default=DEFAULT_NHC_INTEGRATION_STEPS, type=int,
              help=f'Number of integration steps per MD step [default: {DEFAULT_NHC_INTEGRATION_STEPS}].')
@click.option('--nhc-thermo', default=DEFAULT_NHC_THERMO, type=float,
              help=f'Thermostat damping factor in units of timestep, i.e. dt*nhc_thermo [default: {DEFAULT_NHC_THERMO}].')
@click.option('--nhc-baro', default=DEFAULT_NHC_BARO, type=float,
              help=f'Barostat damping factor in units of timestep,  i.e. dt*nhc_baro [default: {DEFAULT_NHC_BARO}].')
@click.option('--nhc-sy-steps', default=DEFAULT_SUZUKI_YOSHIDA_STEPS, type=int,
              help=f'Number of Suzuki-Yoshida integration steps [default: {DEFAULT_SUZUKI_YOSHIDA_STEPS}].')
# Langevin Thermostat settings
@click.option('--langevin-removecmmotion', default=DEFAULT_LANGEVIN_REMOVECMMOTION, type=bool,
              help=f'Removal of the Center of Mass Motion [default: {DEFAULT_LANGEVIN_REMOVECMMOTION}].')
@click.option('--langevin-thermo', default=DEFAULT_LANGEVIN_THERMO, type=float,
              help=f'Friction (Gamma) coefficient for the Langevin thermostat [default: {DEFAULT_LANGEVIN_THERMO}].')
# Restart options
@click.option('--restart-save', type=click.Path(), default=None,
              help='Path to save restart data.')
@click.option('--restart-load', type=click.Path(exists=False), default=None,
              help='Path to load restart data from a previous run.')
# Additional options
@click.option('--relax/--no-relax', default=True,
              help='Perform geometry relaxation before MD [default: enabled].')
@click.option('--seed', default=DEFAULT_SEED, type=int,
              help=f'Random seed for MD [default: {DEFAULT_SEED}].')
# Help options
@click.option('--help-full', is_flag=True,
              help='Show detailed information about MD settings.')
@click.option('--help', '-h', is_flag=True,
              help='Show brief command overview.')
@click.option('--nve', is_flag=True, hidden=True,
              help='Run NVE simulation.')
@click.option('--nvt', is_flag=True, hidden=True,
              help='Run NVT simulation (default).')
@click.option('--npt', is_flag=True, hidden=True,
              help='Run NPT simulation with default pressure of 1.0 atm.')
@click.pass_context
def cli(ctx: click.Context,
        settings: Optional[str],
        # Input/Output
        input_file: Optional[str],
        output_file: Optional[str],
        log_file: Optional[str],
        # Model settings
        model_path: Optional[str],
        precision: str,
        lr_cutoff: float,
        dispersion_damping: float,
        buffer_sr: float,
        buffer_lr: float,
        save_buffer: int,
        total_charge: int,
        # MD settings
        dt: float,
        temperature: float,
        pressure: Optional[float],
        md_cycles: int,
        md_steps: int,
        # Optimization settings
        min_cycles: int,
        min_steps: int,
        min_start_dt: float,
        min_max_dt: float,
        min_n_min: int,
        force_conv: Optional[float],
        # Nose-Hoover Thermostat/Barostat settings
        nhc_chain: int,
        nhc_steps: int,
        nhc_thermo: float,
        nhc_baro: float,
        nhc_sy_steps: int,
        # Langevin Thermostat settings
        langevin_removecmmotion: bool,
        langevin_thermo: float,
        # Restart options
        restart_save: Optional[str],
        restart_load: Optional[str],
        # Additional options
        relax: bool,
        seed: int,
        # Help options
        help_full: bool,
        help: bool,
        nve: bool,
        nvt: bool,
        npt: bool,
        # Optional arguments:
        ) -> None:
    """
    Run molecular dynamics simulations with SO3LR or MLFF models.

    This command provides three subcommands:

    \b
    opt: Run geometry optimization
    nvt: Run NVT (constant volume and temperature) MD simulation with Nose-Hoover integrator
    nvt-langevin: Run NVT (constant volume and temperature) MD simulation with Langevin integrator
    npt: Run NPT (constant pressure and temperature) MD simulation
    nve: Run NVE (constant volume and energy) MD simulation

    If run without a subcommand, it performs a full MD simulation
    with options specified via command line or settings file.

    Use --help-full to see detailed information about all settings.
    """
    # Handle help flags
    if help or help_full:
        # The format_help function in CustomCommandClass handles this
        ctx.info_name = ctx.command.name
        click.echo(ctx.get_help())
        return

    # No arguments provided shows help
    if ctx.invoked_subcommand is None and not any([settings, input_file]):
        click.echo(ctx.get_help())
        return

    # Skip execution if used in a command group context
    if ctx.invoked_subcommand is not None:
        return

    logger.info("=" * 60)
    logger.info(f"SO3LR Molecular Dynamics Simulation (v{__version__})")
    logger.info("=" * 60)

    # Load settings from file if provided
    settings_dict: Dict[str, Any] = {}
    if settings is not None:
        try:
            settings_path = Path(settings).expanduser().resolve()
            if not settings_path.exists():
                raise FileNotFoundError(f'Settings file not found: {settings}')

            with open(settings_path, 'r') as f:
                settings_dict = yaml.safe_load(f)

            # Use standard logging initially for any errors
            logger.info(f"Loading settings from {settings}")
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Error loading settings file: {str(e)}")
            sys.exit(1)

    if dt is not None:
        ctx.params['dt'] = dt/1000
    else:
        ctx.params['dt'] = ctx.params['dt']/1000
          
    for key in ctx.params:
        settings_dict = update_settings_from_defaults(ctx, settings_dict, key)

    # # Override settings with command line arguments if provided
    # if input_file is not None:
    #     settings_dict['input_file'] = input_file
    # if output_file is not None:
    #     settings_dict['output_file'] = output_file
    # if model_path is not None:
    #     settings_dict['model_path'] = model_path
    # if precision is not None:
    #     settings_dict['precision'] = precision
    # if lr_cutoff is not None:
    #     settings_dict['lr_cutoff'] = lr_cutoff
    # if dispersion_damping is not None:
    #     settings_dict['dispersion_damping'] = dispersion_damping
    # if buffer_sr is not None:
    #     settings_dict['buffer_size_multiplier_sr'] = buffer_sr
    # if buffer_lr is not None:
    #     settings_dict['buffer_size_multiplier_lr'] = buffer_lr
    # if save_buffer is not None:
    #     settings_dict['save_buffer'] = save_buffer
    # if restart_save is not None:
    #     settings_dict['restart_save_path'] = restart_save
    # if restart_load is not None:
    #     settings_dict['restart_load_path'] = restart_load
    # if dt is not None:
    #     settings_dict['md_dt'] = dt/1000
    # if temperature is not None:
    #     settings_dict['md_T'] = temperature
    #     if settings_dict.get('nve') == True:
    #         settings_dict['md_T'] = None
    #         settings_dict['init_T'] = None
    # if pressure is not None:
    #     settings_dict['md_P'] = pressure
    # if md_cycles is not None:
    #     settings_dict['md_cycles'] = md_cycles
    # if md_steps is not None:
    #     settings_dict['md_steps'] = md_steps
    # if min_cycles is not None:
    #     settings_dict['min_cycles'] = min_cycles
    # if min_steps is not None:
    #     settings_dict['min_steps'] = min_steps
    # if nhc_chain is not None:
    #     settings_dict['nhc_chain_length'] = nhc_chain
    # if nhc_steps is not None:
    #     settings_dict['nhc_steps'] = nhc_steps
    # if nhc_thermo is not None:
    #     settings_dict['nhc_thermo'] = nhc_thermo
    # if nhc_baro is not None:
    #     settings_dict['nhc_baro'] = nhc_baro
    # if nhc_sy_steps is not None:
    #     settings_dict['nhc_sy_steps'] = nhc_sy_steps
    # if total_charge is not None:
    #     settings_dict['total_charge'] = total_charge
    # if seed is not None:
    #     settings_dict['seed'] = seed
    # if relax is not None:
    #     settings_dict['relax_before_run'] = relax
    # if force_conv is not None:
    #     settings_dict['force_convergence'] = force_conv
    # if min_start_dt is not None:
    #     settings_dict['min_start_dt'] = min_start_dt
    # if min_max_dt is not None:
    #     settings_dict['min_max_dt'] = min_max_dt
    # if min_n_min is not None:
    #     settings_dict['min_n_min'] = min_n_min

    # Validate required settings
    if 'input_file' not in settings_dict:
        logger.error("Error: Initial geometry file must be specified either in settings file or with --input")
        sys.exit(1)

    # Set default output file if not specified
    if 'output_file' not in settings_dict:
        input_name = Path(settings_dict['input_file']).stem
        settings_dict['output_file'] = f'{input_name}_trajectory.xyz'
        logger.info(f"No output file specified, using default: {settings_dict['output_file']}")

    settings_dict['output_format'] = infer_output_format(settings_dict['output_file'])

    # Set default log file based on output file if not explicitly provided
    if 'log_file' not in settings_dict:
        output_path = Path(settings_dict['output_file'])
        log_file_output = f"{output_path.stem}.log"
        if log_file is not None:
            settings_dict['log_file'] = log_file
        else:
            log_file = log_file_output
            settings_dict['log_file'] = log_file
            logger.info(f"No log file specified, using default: {log_file_output}")
    else:
        if settings_dict['log_file'] is None:
            log_file = log_file_output
            settings_dict['log_file'] = log_file
            logger.info(f"No log file specified, using default: {log_file_output}")
        else:
            log_file = settings_dict['log_file']

    # # Add log settings to the settings dictionary
    # settings_dict['log_file'] = log_file

    # Setup logging with default levels
    setup_logger(log_file)

    # Log the ASCII art
    logger.info(SO3LR_ASCII)

    # Log that we loaded settings from file
    if settings is not None:
        logger.info(f"Loading settings from {settings}")

    settings_dict = ctx.command.canonicalize_ensemble(settings_dict)

    # Print simulation details
    logger.info(f"Initial geometry:          {settings_dict['input_file']}")
    logger.info(f"Output file:               {settings_dict['output_file']}")
    logger.info(f"Log file:                  {log_file}")
    logger.info(f"Force field:               {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        logger.info(f"Model path:                {settings_dict['model_path']}")
    logger.info(f"Long-range cutoff:         {settings_dict.get('lr_cutoff', DEFAULT_LR_CUTOFF)} Å")
    logger.info(f"Dispersion damping:        {settings_dict.get('dispersion_damping', DEFAULT_DISPERSION_DAMPING)} Å")
    logger.info(f"Short-range buffer:        {settings_dict.get('buffer_size_multiplier_sr', DEFAULT_BUFFER_MULTIPLIER)}")
    logger.info(f"Long-range buffer:         {settings_dict.get('buffer_size_multiplier_lr', DEFAULT_BUFFER_MULTIPLIER)}")
    logger.info(f"Total charge:              {settings_dict.get('total_charge', DEFAULT_TOTAL_CHARGE)}")
    logger.info(f"Precision:                 {settings_dict.get('precision', DEFAULT_PRECISION)}")

    logger.info(f"Temperature:               {settings_dict.get('md_T', DEFAULT_TEMPERATURE)} K")

    if settings_dict.get('ensemble') == 'npt':
        logger.info(f"Pressure:                  {settings_dict.get('md_P')} atm")
        logger.info(f"Ensemble:                  NPT")
    elif settings_dict.get('ensemble') == 'nvt':
        logger.info(f"Ensemble:                  NVT (NH)")
    elif settings_dict.get('ensemble') == 'nvt-langevin':
        logger.info(f"Ensemble:                  NVT (L)")
    else:
        logger.info(f"Ensemble:                  NVE")

    total_steps = settings_dict.get('md_cycles', DEFAULT_MD_CYCLES) * settings_dict.get('md_steps', DEFAULT_MD_STEPS_PER_CYCLE)
    simulation_time = total_steps * settings_dict.get('md_dt', DEFAULT_TIMESTEP)  # in ps

    logger.info(f"Simulation length:         {total_steps} steps ({simulation_time:.2f} ps)")
    logger.info(f"MD cycles:                 {settings_dict.get('md_cycles', DEFAULT_MD_CYCLES)}")
    logger.info(f"Steps per cycle:           {settings_dict.get('md_steps', DEFAULT_MD_STEPS_PER_CYCLE)}")
    logger.info(f"Timestep:                  {settings_dict.get('md_dt', DEFAULT_TIMESTEP)*1000} fs")
    logger.info(f"NHC length:                {settings_dict.get('nhc_chain_length', DEFAULT_NHC_CHAIN_LENGTH)}")
    logger.info(f"NHC steps:                 {settings_dict.get('nhc_steps', DEFAULT_NHC_INTEGRATION_STEPS)}")
    logger.info(f"NHC thermo:                {settings_dict.get('nhc_thermo', DEFAULT_NHC_THERMO)}")
    logger.info(f"Langevin CMMotion Removal: {settings_dict.get('langevin_removecmmotion', DEFAULT_LANGEVIN_REMOVECMMOTION)}")
    logger.info(f"Langevin Gamma:            {settings_dict.get('langevin_thermo', DEFAULT_LANGEVIN_THERMO)}")
    logger.info(f"Seed:                      {settings_dict.get('seed', DEFAULT_SEED)}")

    if settings_dict.get('relax_before_run', False):
        logger.info(f"Geometry relaxation:       Enabled")
        if force_conv is not None:
            logger.info(f"Force convergence:         {force_conv} eV/Å")
        else:
            logger.info(f"Optimization cycles:       {settings_dict.get('min_cycles', DEFAULT_MIN_CYCLES)} cycles, each {settings_dict.get('min_steps', DEFAULT_MIN_STEPS)} steps")
    else:
        logger.info(f"Geometry relaxation:       Disabled")

    logger.info("=" * 60)

    # Log hardware info
    get_hardware_info()

    # Run the simulation
    time_start = time.time()
    run(settings_dict)
    time_end = time.time()
    logger.info("=" * 60)
    logger.info('Simulation completed successfully!')
    logger.info(f'Total runtime: {(time_end - time_start):.2f} seconds ({(time_end - time_start)/3600:.2f} hours)')



class SubcommandHelpGroup(click.Group):
    """Custom Group class for subcommands to handle help displays appropriately."""

    def get_help(self, ctx: click.Context) -> str:
        """Format and return help text for the subcommand."""
        # Set short help mode for top-level options
        formatted_help = click.formatting.HelpFormatter()
        formatted_help.write(SO3LR_ASCII)
        self.format_usage(ctx, formatted_help)
        self.format_help_text(ctx, formatted_help)
        self.format_options(ctx, formatted_help)
        self.format_epilog(ctx, formatted_help)
        return formatted_help.getvalue()

# Define the 'opt' subcommand
@cli.command(name='opt', help="Run geometry optimization with `so3lr opt --input geometry.xyz`.")
# Input/Output
@click.option('--input', '--input_file', 'input_file', type=click.Path(exists=False),
              help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', type=click.Path(),
              help='Output file to save the optimized geometry. If not provided, defaults to <input_name_without_extension>_opt.xyz.')
@click.option('--log-file', default=None, type=click.Path(),
              help=f'File to write logs to [default: None].')
# Model settings
@click.option('--model', '--model_path', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Dispersion interactions starts to switch off at (lr_cutoff - dispersion_damping) Å. [default: {DEFAULT_DISPERSION_DAMPING}]')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for short-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for long-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
              help=f'Total charge of the system. [default: {DEFAULT_TOTAL_CHARGE}]')
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
              help='Force convergence criterion in eV/Å, overrides --cycles and --steps. [default: None]')
# Help option
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def fire_optimization(
    # Input/Output
    input_file: Optional[str],
    output_file: Optional[str],
    log_file: Optional[str],
    # Model settings
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    total_charge: int,
    # Optimization settings
    min_cycles: int,
    min_steps: int,
    min_start_dt: float,
    min_max_dt: float,
    min_n_min: int,
    force_conv: Optional[float],
    # Help option
    help: bool
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

    # Generate default output file name if not provided
    if input_file is not None and output_file is None:
        input_path = Path(input_file)
        output_file = f"{input_path.stem}_opt.xyz"

    # Generate default log file name based on output file if not explicitly provided
    if log_file is None:
        log_file = f"{Path(output_file).stem}.log"

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
    logger.info(f"Force field:               {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        
        logger.info(f"Model path:                {model_path}")
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
    logger.info(f'Total runtime: {(time_end - time_start):.2f} seconds ({(time_end - time_start)/3600:.2f} hours)')

# Define the 'nvt' subcommand
@cli.command(name='nvt', help="Run NVT molecular dynamics simulation with `so3lr nvt --input geometry.xyz`.")
# Input/Output
@click.option('--input', '--input_file', 'input_file', type=click.Path(exists=False),
              help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', type=click.Path(),
              help='Output file to save the trajectory in hdf5 or extxyz format. If not provided, defaults to <input_name_without_extension>_nvt.xyz.')
@click.option('--log-file', default=None, type=click.Path(),
              help=f'File to write logs to [default: None].')
# Model settings
@click.option('--model', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Dispersion interactions start to switch off at (lr_cutoff - dispersion_damping) Å. [default: {DEFAULT_DISPERSION_DAMPING}].')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for short-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for long-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--save-buffer', default=DEFAULT_SAVE_BUFFER, type=int,
              help=f'Number of frames to buffer before writing to HDF5 file. [default: {DEFAULT_SAVE_BUFFER}]')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
              help=f'Total charge of the system. [default: {DEFAULT_TOTAL_CHARGE}]')
# MD settings
@click.option('--temperature', default=DEFAULT_TEMPERATURE, type=float,
              help=f'Simulation temperature in Kelvin. [default: {DEFAULT_TEMPERATURE}]')
@click.option('--dt', default=DEFAULT_TIMESTEP, type=float,
              help=f'MD timestep in femtoseconds. [default: {DEFAULT_TIMESTEP}]')
@click.option('--md-cycles', default=DEFAULT_MD_CYCLES, type=int,
              help=f'Number of MD cycles to run. [default: {DEFAULT_MD_CYCLES}]')
@click.option('--md-steps', default=DEFAULT_MD_STEPS_PER_CYCLE, type=int,
              help=f'Number of steps per MD cycle. [default: {DEFAULT_MD_STEPS_PER_CYCLE}]')
# Thermostat settings
@click.option('--nhc-chain', default=DEFAULT_NHC_CHAIN_LENGTH, type=int,
              help=f'Length of the Nose-Hoover thermostat chain. [default: {DEFAULT_NHC_CHAIN_LENGTH}]')
@click.option('--nhc-steps', default=DEFAULT_NHC_INTEGRATION_STEPS, type=int,
              help=f'Number of integration steps per MD step. [default: {DEFAULT_NHC_INTEGRATION_STEPS}]')
@click.option('--nhc-thermo', 'nhc_thermo', default=DEFAULT_NHC_THERMO, type=float,
              help=f'Thermostat damping factor in units of timestep, i.e. dt*nhc_thermo. [default: {DEFAULT_NHC_THERMO}]')
@click.option('--nhc-sy-steps', default=DEFAULT_SUZUKI_YOSHIDA_STEPS, type=int,
              help=f'Number of Suzuki-Yoshida integration steps [default: {DEFAULT_SUZUKI_YOSHIDA_STEPS}].')
# Restart options
@click.option('--restart-save', type=click.Path(), default=None,
              help='Path to save restart data.')
@click.option('--restart-load', type=click.Path(exists=False), default=None,
              help='Path to load restart data from a previous run.')
# Additional options
@click.option('--relax/--no-relax', default=True,
              help='Perform geometry relaxation before MD. [default: enabled]')
@click.option('--force-conv', default=None, type=float,
              help='Force convergence criterion in eV/Å for initial relaxation. [default: None]')
@click.option('--seed', default=DEFAULT_SEED, type=int,
              help=f'Random seed for MD. [default: {DEFAULT_SEED}]')
# Help option
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def nvt_md(
    # Input/Output
    input_file: Optional[str],
    output_file: Optional[str],
    log_file: Optional[str],
    # Model settings
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    save_buffer: int,
    total_charge: int,
    # MD settings
    temperature: float,
    dt: float,
    md_cycles: int,
    md_steps: int,
    # Thermostat settings
    nhc_chain: int,
    nhc_steps: int,
    nhc_thermo: float,
    nhc_sy_steps: int,
    # Restart options
    restart_save: Optional[str],
    restart_load: Optional[str],
    # Additional options
    relax: bool,
    force_conv: Optional[float],
    seed: int,
    # Help option
    help: bool,
    # Optional arguments
) -> None:
    """
    Run NVT (constant volume and temperature) molecular dynamics simulation.

    This command runs a molecular dynamics simulation in the NVT ensemble
    (constant number of particles, volume, and temperature) using the
    Nose-Hoover chain thermostat.

    Example:
        so3lr nvt --input geometry.xyz --temperature 300
    """
    # Print help if needed
    if not input_file or help:
        click.echo(SO3LR_ASCII)
        click.echo(nvt_md.get_help(click.get_current_context()))
        return

    # Generate default output file name if not provided
    if input_file is not None and output_file is None:
        input_path = Path(input_file)
        output_file = f"{input_path.stem}_nvt.xyz"

    # Generate default log file name based on output file if not explicitly provided
    if log_file is None:
        log_file = f"{Path(output_file).stem}.log"

    # Setup logging with default levels
    setup_logger(log_file)

    # Log the ASCII art
    logger.info(SO3LR_ASCII)

    # Log all settings
    total_steps = md_cycles * md_steps
    simulation_time = total_steps * dt/1000  # in ps

    logger.info("=" * 60)
    logger.info(f"SO3LR NVT Molecular Dynamics Simulation (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Initial geometry:          {input_file}")
    logger.info(f"Output file:               {output_file}")
    logger.info(f"Log file:                  {log_file}")
    logger.info(f"Force field:               {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        logger.info(f"Model path:                {model_path}")
    logger.info(f"Precision:                 {precision}")
    logger.info(f"Long-range cutoff:         {lr_cutoff} Å")
    logger.info(f"Dispersion damping:        {dispersion_damping} Å")
    logger.info(f"Short-range buffer:        {buffer_sr}")
    logger.info(f"Long-range buffer:         {buffer_lr}")
    logger.info(f"Total charge:              {total_charge}")
    logger.info(f"Temperature:               {temperature} K")
    logger.info(f"Ensemble:                  NVT")
    logger.info(f"Simulation length:         {total_steps} steps ({simulation_time:.2f} ps)")
    logger.info(f"MD cycles:                 {md_cycles}")
    logger.info(f"Steps per cycle:           {md_steps}")
    logger.info(f"Timestep:                  {dt} fs")
    logger.info(f"Saving buffer size:        {save_buffer}")
    logger.info(f"NHC chain length:          {nhc_chain}")
    logger.info(f"Nose-Hoover steps:         {nhc_steps}")
    logger.info(f"Nose-Hoover Tdamp:         {nhc_thermo*dt} fs")
    logger.info(f"Random seed:               {seed}")

    if restart_load:
        logger.info(f"Restart from:              {restart_load}")
    if restart_save:
        logger.info(f"Save restart to:            {restart_save}")

    if relax:
        logger.info(f"Geometry relaxation:       Enabled")
        if force_conv is not None:
            logger.info(f"Force convergence:         {force_conv} eV/Å")
    else:
        logger.info(f"Geometry relaxation:       Disabled")

    logger.info("=" * 60)

    # Log hardware info
    get_hardware_info()

    # Override settings with command line arguments
    settings = {
        'input_file': input_file,
        'output_file': output_file,
        'model_path': model_path,
        'precision': precision,
        'md_dt': dt/1000,
        'md_T': temperature,
        'md_cycles': md_cycles,
        'md_steps': md_steps,
        'lr_cutoff': lr_cutoff,
        'dispersion_damping': dispersion_damping,
        'buffer_size_multiplier_sr': buffer_sr,
        'buffer_size_multiplier_lr': buffer_lr,
        'save_buffer': save_buffer,
        'total_charge': total_charge,
        'seed': seed,
        'restart_load_path': restart_load,
        'restart_save_path': restart_save,
        'relax_before_run': relax,
        'force_convergence': force_conv,
        # Additional MD settings
        'nhc_chain_length': nhc_chain,
        'nhc_steps': nhc_steps,
        'nhc_thermo': nhc_thermo,
        'nhc_sy_steps': nhc_sy_steps,
        # Use default optimization settings
        'min_n_min': DEFAULT_MIN_N_MIN, 
        'min_start_dt': DEFAULT_MIN_START_DT,
        'min_max_dt': DEFAULT_MIN_MAX_DT,
        'min_cycles': DEFAULT_MIN_CYCLES,
        'min_steps': DEFAULT_MIN_STEPS,
        'ensemble': 'nvt',
    }

    # Add log settings to the settings dictionary
    settings['log_file'] = log_file

    # Determine output format based on file extension
    settings['output_format'] = infer_output_format(output_file)

    # Run NVT simulation
    time_start = time.time()
    run(settings)
    time_end = time.time()
    logger.info("=" * 60)
    logger.info('Simulation completed successfully!')
    logger.info(f'Total runtime: {(time_end - time_start):.2f} seconds ({(time_end - time_start)/3600:.2f} hours)')


# Define the 'nvt-langevin' subcommand
@cli.command(name='nvt-langevin', help="Run NVT (Langevin thermostat) molecular dynamics simulation with `so3lr nvt-langevin --input geometry.xyz`.")
# Input/Output
@click.option('--input', '--input_file', 'input_file', type=click.Path(exists=False),
              help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', type=click.Path(),
              help='Output file to save the trajectory in hdf5 or extxyz format. If not provided, defaults to <input_name_without_extension>_nvt.xyz.')
@click.option('--log-file', default=None, type=click.Path(),
              help=f'File to write logs to [default: None].')
# Model settings
@click.option('--model', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Dispersion interactions start to switch off at (lr_cutoff - dispersion_damping) Å. [default: {DEFAULT_DISPERSION_DAMPING}].')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for short-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for long-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--save-buffer', default=DEFAULT_SAVE_BUFFER, type=int,
              help=f'Number of frames to buffer before writing to HDF5 file. [default: {DEFAULT_SAVE_BUFFER}]')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
              help=f'Total charge of the system. [default: {DEFAULT_TOTAL_CHARGE}]')
# MD settings
@click.option('--temperature', default=DEFAULT_TEMPERATURE, type=float,
              help=f'Simulation temperature in Kelvin. [default: {DEFAULT_TEMPERATURE}]')
@click.option('--dt', default=DEFAULT_TIMESTEP, type=float,
              help=f'MD timestep in femtoseconds. [default: {DEFAULT_TIMESTEP}]')
@click.option('--md-cycles', default=DEFAULT_MD_CYCLES, type=int,
              help=f'Number of MD cycles to run. [default: {DEFAULT_MD_CYCLES}]')
@click.option('--md-steps', default=DEFAULT_MD_STEPS_PER_CYCLE, type=int,
              help=f'Number of steps per MD cycle. [default: {DEFAULT_MD_STEPS_PER_CYCLE}]')
# Thermostat settings
@click.option('--langevin-removecmmotion', default=DEFAULT_LANGEVIN_REMOVECMMOTION, type=bool,
              help=f'Removal of the Center of Mass Motion [default: {DEFAULT_LANGEVIN_REMOVECMMOTION}].')
@click.option('--langevin-thermo', default=DEFAULT_LANGEVIN_THERMO, type=float,
              help=f'Friction (Gamma) coefficient for the Langevin thermostat [default: {DEFAULT_LANGEVIN_THERMO}].')
# Restart options
@click.option('--restart-save', type=click.Path(), default=None,
              help='Path to save restart data.')
@click.option('--restart-load', type=click.Path(exists=False), default=None,
              help='Path to load restart data from a previous run.')
# Additional options
@click.option('--relax/--no-relax', default=True,
              help='Perform geometry relaxation before MD. [default: enabled]')
@click.option('--force-conv', default=None, type=float,
              help='Force convergence criterion in eV/Å for initial relaxation. [default: None]')
@click.option('--seed', default=DEFAULT_SEED, type=int,
              help=f'Random seed for MD. [default: {DEFAULT_SEED}]')
# Help option
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def nvt_langevin_md(
    # Input/Output
    input_file: Optional[str],
    output_file: Optional[str],
    log_file: Optional[str],
    # Model settings
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    save_buffer: int,
    total_charge: int,
    # MD settings
    temperature: float,
    dt: float,
    md_cycles: int,
    md_steps: int,
    # Thermostat settings
    langevin_removecmmotion: bool,
    langevin_thermo: float,
    # Restart options
    restart_save: Optional[str],
    restart_load: Optional[str],
    # Additional options
    relax: bool,
    force_conv: Optional[float],
    seed: int,
    # Help option
    help: bool,
    # Optional arguments
) -> None:
    """
    Run NVT (constant volume and temperature) molecular dynamics simulation.

    This command runs a molecular dynamics simulation in the NVT ensemble
    (constant number of particles, volume, and temperature) using the
    Langevin thermostat.

    Example:
        so3lr nvt-langevin --input geometry.xyz --temperature 300
    """
    # Print help if needed
    if not input_file or help:
        click.echo(SO3LR_ASCII)
        click.echo(nvt_langevin_md.get_help(click.get_current_context()))
        return

    # Generate default output file name if not provided
    if input_file is not None and output_file is None:
        input_path = Path(input_file)
        output_file = f"{input_path.stem}_nvt_langevin.xyz"

    # Generate default log file name based on output file if not explicitly provided
    if log_file is None:
        log_file = f"{Path(output_file).stem}.log"

    # Setup logging with default levels
    setup_logger(log_file)

    # Log the ASCII art
    logger.info(SO3LR_ASCII)

    # Log all settings
    total_steps = md_cycles * md_steps
    simulation_time = total_steps * dt/1000  # in ps

    logger.info("=" * 60)
    logger.info(f"SO3LR NVT Molecular Dynamics Simulation (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Initial geometry:          {input_file}")
    logger.info(f"Output file:               {output_file}")
    logger.info(f"Log file:                  {log_file}")
    logger.info(f"Force field:               {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        logger.info(f"Model path:                {model_path}")
    logger.info(f"Precision:                 {precision}")
    logger.info(f"Long-range cutoff:         {lr_cutoff} Å")
    logger.info(f"Dispersion damping:        {dispersion_damping} Å")
    logger.info(f"Short-range buffer:        {buffer_sr}")
    logger.info(f"Long-range buffer:         {buffer_lr}")
    logger.info(f"Total charge:              {total_charge}")
    logger.info(f"Temperature:               {temperature} K")
    logger.info(f"Ensemble:                  NVT (L)")
    logger.info(f"Simulation length:         {total_steps} steps ({simulation_time:.2f} ps)")
    logger.info(f"MD cycles:                 {md_cycles}")
    logger.info(f"Steps per cycle:           {md_steps}")
    logger.info(f"Timestep:                  {dt} fs")
    logger.info(f"Saving buffer size:        {save_buffer}")
    logger.info(f"Center of Mass Motion Removal: {langevin_removecmmotion}")
    logger.info(f"Langevin friction value:   {dt * langevin_thermo}")
    logger.info(f"Random seed:               {seed}")

    if restart_load:
        logger.info(f"Restart from:              {restart_load}")
    if restart_save:
        logger.info(f"Save restart to:            {restart_save}")

    if relax:
        logger.info(f"Geometry relaxation:       Enabled")
        if force_conv is not None:
            logger.info(f"Force convergence:         {force_conv} eV/Å")
    else:
        logger.info(f"Geometry relaxation:       Disabled")

    logger.info("=" * 60)

    # Log hardware info
    get_hardware_info()

    # Override settings with command line arguments
    settings = {
        'input_file': input_file,
        'output_file': output_file,
        'model_path': model_path,
        'precision': precision,
        'md_dt': dt/1000,
        'md_T': temperature,
        'md_cycles': md_cycles,
        'md_steps': md_steps,
        'lr_cutoff': lr_cutoff,
        'dispersion_damping': dispersion_damping,
        'buffer_size_multiplier_sr': buffer_sr,
        'buffer_size_multiplier_lr': buffer_lr,
        'save_buffer': save_buffer,
        'total_charge': total_charge,
        'seed': seed,
        'restart_load_path': restart_load,
        'restart_save_path': restart_save,
        'relax_before_run': relax,
        'force_convergence': force_conv,
        # Additional MD settings
        'langevin_removecmmotion': langevin_removecmmotion,
        'langevin_thermo': langevin_thermo,
        # Use default optimization settings
        'min_n_min': DEFAULT_MIN_N_MIN, 
        'min_start_dt': DEFAULT_MIN_START_DT,
        'min_max_dt': DEFAULT_MIN_MAX_DT,
        'min_cycles': DEFAULT_MIN_CYCLES,
        'min_steps': DEFAULT_MIN_STEPS,
        'ensemble': 'nvt-langevin',
    }

    # Add log settings to the settings dictionary
    settings['log_file'] = log_file

    # Determine output format based on file extension
    settings['output_format'] = infer_output_format(output_file)

    # Run NVT simulation
    time_start = time.time()
    run(settings)
    time_end = time.time()
    logger.info("=" * 60)
    logger.info('Simulation completed successfully!')
    logger.info(f'Total runtime: {(time_end - time_start):.2f} seconds ({(time_end - time_start)/3600:.2f} hours)')

# Define the 'npt' subcommand with clear help
@cli.command(name='npt', help="Run NPT molecular dynamics simulation with `so3lr npt --input geometry.xyz`.")
# Input/Output
@click.option('--input', '--input_file', 'input_file', type=click.Path(exists=False),
              help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', type=click.Path(),
              help='Output trajectory file (.hdf5 or .xyz). [default: <input_name_without_extension>_npt.xyz]')
@click.option('--log-file', default=None, type=click.Path(),
              help=f'File to write logs to [default: None].')
# Model settings
@click.option('--model', '--model_path', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Dispersion interactions start to switch off at (lr_cutoff - dispersion_damping) Å. [default: {DEFAULT_DISPERSION_DAMPING}]')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for short-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for long-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--save-buffer', default=DEFAULT_SAVE_BUFFER, type=int,
              help=f'Number of frames to buffer before writing to HDF5 file. [default: {DEFAULT_SAVE_BUFFER}]')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
              help=f'Total charge of the system. [default: {DEFAULT_TOTAL_CHARGE}]')
# MD settings
@click.option('--temperature', default=DEFAULT_TEMPERATURE, type=float,
              help=f'Simulation temperature in Kelvin. [default: {DEFAULT_TEMPERATURE}]')
@click.option('--pressure', default=DEFAULT_PRESSURE, type=float,
              help=f'Simulation pressure in atmospheres. [default: {DEFAULT_PRESSURE}]')
@click.option('--dt', default=DEFAULT_TIMESTEP, type=float,
              help=f'MD timestep in femtoseconds. [default: {DEFAULT_TIMESTEP}]')
@click.option('--md-cycles', default=DEFAULT_MD_CYCLES, type=int,
              help=f'Number of MD cycles to run. [default: {DEFAULT_MD_CYCLES}]')
@click.option('--md-steps', default=DEFAULT_MD_STEPS_PER_CYCLE, type=int,
              help=f'Number of steps per MD cycle. [default: {DEFAULT_MD_STEPS_PER_CYCLE}]')
# Thermostat/Barostat settings
@click.option('--nhc-chain', default=DEFAULT_NHC_CHAIN_LENGTH, type=int,
              help=f'Length of the Nose-Hoover thermostat chain. [default: {DEFAULT_NHC_CHAIN_LENGTH}]')
@click.option('--nhc-steps', default=DEFAULT_NHC_INTEGRATION_STEPS, type=int,
              help=f'Number of integration steps per MD step. [default: {DEFAULT_NHC_INTEGRATION_STEPS}]')
@click.option('--nhc-thermo', default=DEFAULT_NHC_THERMO, type=float,
              help=f'Thermostat damping factor in units of timestep, i.e. dt*nhc_thermo [default: {DEFAULT_NHC_THERMO}].')
@click.option('--nhc-baro', default=DEFAULT_NHC_BARO, type=float,
              help=f'Barostat damping factor in units of timestep,  i.e. dt*nhc_baro [default: {DEFAULT_NHC_BARO}].')
@click.option('--nhc-sy-steps', default=DEFAULT_SUZUKI_YOSHIDA_STEPS, type=int,
              help=f'Number of Suzuki-Yoshida integration steps [default: {DEFAULT_SUZUKI_YOSHIDA_STEPS}].')
# Restart options
@click.option('--restart-save', type=click.Path(), default=None,
              help='Path to save restart data.')
@click.option('--restart-load', type=click.Path(exists=False), default=None,
              help='Path to load restart data from a previous run.')
# Additional options
@click.option('--relax/--no-relax', default=True,
              help='Perform geometry relaxation before MD. [default: enabled]')
@click.option('--force-conv', default=None, type=float,
              help='Force convergence criterion in eV/Å for initial relaxation. [default: None]')
@click.option('--seed', default=DEFAULT_SEED, type=int,
              help=f'Random seed for MD. [default: {DEFAULT_SEED}]')
# Help option
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def npt_md(
    # Input/Output
    input_file: Optional[str],
    output_file: Optional[str],
    log_file: Optional[str],
    # Model settings
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    save_buffer: int,
    total_charge: int,
    # MD settings
    temperature: float,
    pressure: float,
    dt: float,
    md_cycles: int,
    md_steps: int,
    # Thermostat/Barostat settings
    nhc_chain: int,
    nhc_steps: int,
    nhc_thermo: float,
    nhc_baro: float,
    nhc_sy_steps: int,
    # Restart options
    restart_save: Optional[str],
    restart_load: Optional[str],
    # Additional options
    relax: bool,
    force_conv: Optional[float],
    seed: int,
    # Help option
    help: bool
) -> None:
    """
    Run NPT (constant pressure and temperature) molecular dynamics simulation.

    This command runs a molecular dynamics simulation in the NPT ensemble
    (constant number of particles, pressure, and temperature) using the
    Nose-Hoover chain thermostat and barostat.

    Note: NPT simulations require a periodic cell in the input structure.

    Example:
        so3lr npt --input geometry.xyz --temperature 300 --pressure 1
    """
    # Print help if needed
    if not input_file or help:
        click.echo(SO3LR_ASCII)
        click.echo(npt_md.get_help(click.get_current_context()))
        return

    # Generate default output file name if not provided
    if input_file is not None and output_file is None:
        input_path = Path(input_file)
        output_file = f"{input_path.stem}_npt.xyz"

    # Generate default log file name based on output file if not explicitly provided
    if log_file is None:
        log_file = f"{Path(output_file).stem}.log"

    # Setup logging with default levels
    setup_logger(log_file)

    # Log the ASCII art
    logger.info(SO3LR_ASCII)

    # Log all settings
    total_steps = md_cycles * md_steps
    simulation_time = total_steps * dt/1000  # in ps

    logger.info("=" * 60)
    logger.info(f"SO3LR NPT Molecular Dynamics Simulation (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Initial geometry:          {input_file}")
    logger.info(f"Output file:               {output_file}")
    logger.info(f"Log file:                  {log_file}")
    logger.info(f"Force field:               {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        logger.info(f"Model path:                {model_path}")
    logger.info(f"Precision:                 {precision}")
    logger.info(f"Long-range cutoff:         {lr_cutoff} Å")
    logger.info(f"Dispersion damping:        {dispersion_damping} Å")
    logger.info(f"Short-range buffer:        {buffer_sr}")
    logger.info(f"Long-range buffer:         {buffer_lr}")
    logger.info(f"Total charge:              {total_charge}")
    logger.info(f"Temperature:               {temperature} K")
    logger.info(f"Pressure:                  {pressure} atm")
    logger.info(f"Ensemble:                  NPT")
    logger.info(f"Simulation length:         {total_steps} steps ({simulation_time:.2f} ps)")
    logger.info(f"MD cycles:                 {md_cycles}")
    logger.info(f"Steps per cycle:           {md_steps}")
    logger.info(f"Timestep:                  {dt} fs")
    logger.info(f"Saving buffer size:        {save_buffer}")
    logger.info(f"NHC chain length:          {nhc_chain}")
    logger.info(f"Nose-Hoover steps:         {nhc_steps}")
    logger.info(f"Nose-Hoover Tdamp:         {nhc_thermo*dt} fs")
    logger.info(f"Nose-Hoover Pdamp:         {nhc_baro*dt} fs")
    logger.info(f"Random seed:               {seed}")

    if restart_load:
        logger.info(f"Restart from:              {restart_load}")
    if restart_save:
        logger.info(f"Save restart to:           {restart_save}")

    if relax:
        logger.info(f"Geometry relaxation:       Enabled")
        if force_conv is not None:
            logger.info(f"Force convergence:         {force_conv} eV/Å")
    else:
        logger.info(f"Geometry relaxation:       Disabled")

    logger.info("=" * 60)

    # Log hardware info
    get_hardware_info()

    # Override settings with command line arguments
    settings = {
        'input_file': input_file,
        'output_file': output_file,
        'model_path': model_path,
        'precision': precision,
        'md_dt': dt/1000,
        'md_T': temperature,
        'md_P': pressure,
        'md_cycles': md_cycles,
        'md_steps': md_steps,
        'lr_cutoff': lr_cutoff,
        'dispersion_damping': dispersion_damping,
        'buffer_size_multiplier_sr': buffer_sr,
        'buffer_size_multiplier_lr': buffer_lr,
        'save_buffer': save_buffer,
        'total_charge': total_charge,
        'seed': seed,
        'restart_load_path': restart_load,
        'restart_save_path': restart_save,
        'relax_before_run': relax,
        'force_convergence': force_conv,
        # Additional MD settings
        'nhc_chain_length': nhc_chain,
        'nhc_steps': nhc_steps,
        'nhc_thermo': nhc_thermo,
        'nhc_baro': nhc_baro,
        'nhc_sy_steps': nhc_sy_steps,
        # Use default optimization settings
        'min_n_min': DEFAULT_MIN_N_MIN, 
        'min_start_dt': DEFAULT_MIN_START_DT,
        'min_max_dt': DEFAULT_MIN_MAX_DT,
        'min_cycles': DEFAULT_MIN_CYCLES,
        'min_steps': DEFAULT_MIN_STEPS,
        'ensemble': 'npt',
    }

    # Add log settings to the settings dictionary
    settings['log_file'] = log_file

    # Determine output format based on file extension
    settings['output_format'] = infer_output_format(output_file)

    # Run NPT simulation
    time_start = time.time()
    run(settings)
    time_end = time.time()
    logger.info("=" * 60)
    logger.info('Simulation completed successfully!')
    logger.info(f'Total runtime: {(time_end - time_start):.2f} seconds ({(time_end - time_start)/3600:.2f} hours)')


# Define the 'nve' subcommand
@cli.command(name='nve', help="Run NVE molecular dynamics simulation with `so3lr nve --input geometry.xyz`.")
# Input/Output
@click.option('--input', '--input_file', 'input_file', type=click.Path(exists=False),
              help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', type=click.Path(),
              help='Output file to save the trajectory in hdf5 or extxyz format. If not provided, defaults to <input_name_without_extension>_nvt.xyz.')
@click.option('--log-file', default=None, type=click.Path(),
              help=f'File to write logs to [default: None].')
# Model settings
@click.option('--model', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Dispersion interactions start to switch off at (lr_cutoff - dispersion_damping) Å. [default: {DEFAULT_DISPERSION_DAMPING}].')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for short-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for long-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--save-buffer', default=DEFAULT_SAVE_BUFFER, type=int,
              help=f'Number of frames to buffer before writing to HDF5 file. [default: {DEFAULT_SAVE_BUFFER}]')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
              help=f'Total charge of the system. [default: {DEFAULT_TOTAL_CHARGE}]')
# MD settings
@click.option('--temperature', default=DEFAULT_TEMPERATURE, type=float,
              help=f'Simulation temperature in Kelvin. [default: {DEFAULT_TEMPERATURE}]')
@click.option('--dt', default=DEFAULT_TIMESTEP, type=float,
              help=f'MD timestep in femtoseconds. [default: {DEFAULT_TIMESTEP}]')
@click.option('--md-cycles', default=DEFAULT_MD_CYCLES, type=int,
              help=f'Number of MD cycles to run. [default: {DEFAULT_MD_CYCLES}]')
@click.option('--md-steps', default=DEFAULT_MD_STEPS_PER_CYCLE, type=int,
              help=f'Number of steps per MD cycle. [default: {DEFAULT_MD_STEPS_PER_CYCLE}]')
# Restart options
@click.option('--restart-save', type=click.Path(), default=None,
              help='Path to save restart data.')
@click.option('--restart-load', type=click.Path(exists=False), default=None,
              help='Path to load restart data from a previous run.')
# Additional options
@click.option('--relax/--no-relax', default=True,
              help='Perform geometry relaxation before MD. [default: enabled]')
@click.option('--force-conv', default=None, type=float,
              help='Force convergence criterion in eV/Å for initial relaxation. [default: None]')
@click.option('--seed', default=DEFAULT_SEED, type=int,
              help=f'Random seed for MD. [default: {DEFAULT_SEED}]')
# Help option
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def nve_md(
    # Input/Output
    input_file: Optional[str],
    output_file: Optional[str],
    log_file: Optional[str],
    # Model settings
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    save_buffer: int,
    total_charge: int,
    # MD settings
    temperature: float,
    dt: float,
    md_cycles: int,
    md_steps: int,
    # Restart options
    restart_save: Optional[str],
    restart_load: Optional[str],
    # Additional options
    relax: bool,
    force_conv: Optional[float],
    seed: int,
    # Help option
    help: bool
) -> None:
    """
    Run NVE (constant volume and energy) molecular dynamics simulation.

    This command runs a molecular dynamics simulation in the NVE ensemble
    (constant number of particles, volume, and energy) using the
    no thermostat
    Example:
        so3lr nve --input geometry.xyz --temperature 300
    """
    # Print help if needed
    if not input_file or help:
        click.echo(SO3LR_ASCII)
        click.echo(nve_md.get_help(click.get_current_context()))
        return

    # Generate default output file name if not provided
    if input_file is not None and output_file is None:
        input_path = Path(input_file)
        output_file = f"{input_path.stem}_nve.xyz"

    # Generate default log file name based on output file if not explicitly provided
    if log_file is None:
        log_file = f"{Path(output_file).stem}.log"

    # Setup logging with default levels
    setup_logger(log_file)

    # Log the ASCII art
    logger.info(SO3LR_ASCII)

    # Log all settings
    total_steps = md_cycles * md_steps
    simulation_time = total_steps * dt/1000  # in ps

    logger.info("=" * 60)
    logger.info(f"SO3LR NVE Molecular Dynamics Simulation (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Initial geometry:          {input_file}")
    logger.info(f"Output file:               {output_file}")
    logger.info(f"Log file:                  {log_file}")
    logger.info(f"Force field:               {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        logger.info(f"Model path:                {model_path}")
    logger.info(f"Precision:                 {precision}")
    logger.info(f"Long-range cutoff:         {lr_cutoff} Å")
    logger.info(f"Dispersion damping:        {dispersion_damping} Å")
    logger.info(f"Short-range buffer:        {buffer_sr}")
    logger.info(f"Long-range buffer:         {buffer_lr}")
    logger.info(f"Total charge:              {total_charge}")
    logger.info(f"Temperature:               {temperature} K")
    logger.info(f"Ensemble:                  NVE")
    logger.info(f"Simulation length:         {total_steps} steps ({simulation_time:.2f} ps)")
    logger.info(f"MD cycles:                 {md_cycles}")
    logger.info(f"Steps per cycle:           {md_steps}")
    logger.info(f"Timestep:                  {dt} fs")
    logger.info(f"Saving buffer size:        {save_buffer}")
    logger.info(f"Random seed:               {seed}")

    if restart_load:
        logger.info(f"Restart from:              {restart_load}")
    if restart_save:
        logger.info(f"Save restart to:            {restart_save}")

    if relax:
        logger.info(f"Geometry relaxation:       Enabled")
        if force_conv is not None:
            logger.info(f"Force convergence:         {force_conv} eV/Å")
    else:
        logger.info(f"Geometry relaxation:       Disabled")

    logger.info("=" * 60)

    # Log hardware info
    get_hardware_info()

    # Override settings with command line arguments
    settings = {
        'input_file': input_file,
        'output_file': output_file,
        'model_path': model_path,
        'precision': precision,
        'md_dt': dt/1000,
        'init_T': temperature,
        'md_cycles': md_cycles,
        'md_steps': md_steps,
        'lr_cutoff': lr_cutoff,
        'dispersion_damping': dispersion_damping,
        'buffer_size_multiplier_sr': buffer_sr,
        'buffer_size_multiplier_lr': buffer_lr,
        'save_buffer': save_buffer,
        'total_charge': total_charge,
        'seed': seed,
        'restart_load_path': restart_load,
        'restart_save_path': restart_save,
        'relax_before_run': relax,
        'force_convergence': force_conv,
        # Use default optimization settings
        'min_n_min': DEFAULT_MIN_N_MIN,
        'min_start_dt': DEFAULT_MIN_START_DT,
        'min_max_dt': DEFAULT_MIN_MAX_DT,
        'min_cycles': DEFAULT_MIN_CYCLES,
        'min_steps': DEFAULT_MIN_STEPS,
        'ensemble': 'nve',
    }

    # Add log settings to the settings dictionary
    settings['log_file'] = log_file

    # Determine output format based on file extension
    settings['output_format'] = infer_output_format(output_file)

    # Run NVT simulation
    time_start = time.time()
    run(settings)
    time_end = time.time()
    logger.info("=" * 60)
    logger.info('Simulation completed successfully!')
    logger.info(f'Total runtime: {(time_end - time_start):.2f} seconds ({(time_end - time_start)/3600:.2f} hours)')


# Define the 'eval' subcommand
@cli.command(name='eval', help="Evaluate SO3LR model on a dataset with `so3lr eval --datafile dataset.extxyz`.")
# Input/Output group
@click.option('--datafile', type=click.Path(exists=False), required=False,
              help='Path to dataset file (extxyz format) containing structures to evaluate.')
@click.option('--save-to', type=click.Path(), default=None,
              help='Output file to save predictions (.extxyz format). If not provided, predictions are not saved.')
@click.option('--log-file', default=None, type=click.Path(),
              help='File to write logs to [default: None].')
# Model settings group
@click.option('--model', '--model_path', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Dispersion interactions start to switch off at (lr_cutoff - dispersion_damping) Å. [default: {DEFAULT_DISPERSION_DAMPING}].')
# Evaluation settings
@click.option('--batch-size', default=1, type=int,
              help='Number of molecules per batch. [default: 1]')
@click.option('--targets', default='forces,dipole_vec,hirshfeld_ratios', type=str,
              help='Comma-separated list of targets to evaluate (forces, energy, dipole_vec, hirshfeld_ratios). [default: forces,dipole_vec,hirshfeld_ratios]')
@click.option('--jit-compile/--no-jit-compile', 'jit_compile', default=True,
              help='Use JIT compilation for speed. [default: enabled]')
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
    # Evaluation settings
    batch_size: int,
    targets: str,
    jit_compile: bool,
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

    # Generate default log file name based on output file if not explicitly provided
    if log_file is None:
        log_file = f"{Path(output_file).stem}.log"

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
    logger.info(f"Force field:               {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        logger.info(f"Model path:                {model_path}")
    logger.info(f"Precision:                 {precision}")
    logger.info(f"Long-range cutoff:         {lr_cutoff} Å")
    logger.info(f"Dispersion damping:        {dispersion_damping} Å")
    logger.info(f"JIT compilation:           {'Enabled' if jit_compile else 'Disabled'}")
    if save_to:
        logger.info(f"Saving to:                 {save_to}")
    logger.info(f"Targets:                   {targets}")
    logger.info("=" * 60)

    # Log hardware info
    get_hardware_info()

    # Validate file existence
    if not Path(datafile).exists():
        logger.error(f"Error: Dataset file not found: {datafile}")
        sys.exit(1)

    # Validate output path
    if save_to:
        save_path = Path(save_to)
        if save_path.exists():
            logger.error(f"Error: Output file already exists: {save_to}")
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
        log_file=log_file
    )
    logger.info("=" * 60)
    logger.info("Evaluation completed successfully!")

    if save_to:
        logger.info(f"Predictions saved to: {save_to}")


# Define the 'finetune' subcommand
@cli.command(name='finetune', help="Finetune SO3LR model on a dataset with `so3lr finetune --datafile dataset.extxyz --workdir finetune_so3lr --num-train 50 --num-valid 10`.")
# Input/Output group
@click.option('--workdir', type=click.Path(), help='Output file to save predictions (.extxyz format). If not provided, predictions are not saved.')
@click.option('--datafile', type=click.Path(), help='Path to dataset file (extxyz or tfds format) containing data used for finetuning.')
@click.option('--log-file', default=None, type=click.Path(),
              help='File to write logs to [default: None].')
# Finetuning settings.
@click.option('--strategy', type=click.Choice(['full','final_mlp','final_mlp_and_hirshfeld','hirshfeld','last_layer','last_layer_and_final_mlp','first_layer','first_layer_and_last_layer']),
              default='full', help='Strategy for finetuning. [default: full]')
@click.option('--config', '--config_path', 'config_path', type=click.Path(exists=False), default=None,
              help='Path to the finetuning config. If not provided is defaults to the one pre-defined in the SO3LR repo. [default: None]')
@click.option('--model', '--model_path', 'model_path', type=click.Path(exists=False), default=None,
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
# Training settings
@click.option('--num-train', '--num_train', type=int, help='Number of data points used for gradient calculations.')
@click.option('--num-valid', '--num_valid', type=int, help='Number of data points used for validation.')
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
    num_train: int,
    num_valid: int,
    # Help option
    help: bool
) -> None:
    """
    Finetune SO3LR or MLFF model on a dataset.

    This command finetunes the SO3LR model or another MLFF model on a dataset of molecular structures.

    Example:
        so3lr finetune --datafile dataset.extxyz --workdir finetune_so3lr --num-train 50 --num-valid 10
    """
    # Print help if needed
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
    logger.info(f"SO3LR Model Evaluation (v{__version__})")
    logger.info("=" * 60)
    logger.info(f"Dataset file:              {datafile}")
    logger.info(f"Workdir:                   {workdir}")
    logger.info(f"Finetuning strategy:       {strategy}")
    logger.info(f"Force field:               {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        logger.info(f"Model path:            {model_path}")
    logger.info(f"Config:                    {'Default' if model_path else 'Custom'}")
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

    # Call the evaluate_so3lr_on function from so3lr_eval.py
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


# Define the 'tune-ewald' subcommand
@cli.command(name='tune-ewald', help="Tune Ewald parameters`so3lr tune-ewald --input geometry.xyz`.")
# Input/Output group
# Input/Output
@click.option('--settings', type=click.Path(exists=False), default=None,
              help='Path to YAML settings file.')
@click.option('--input', '--input_file', 'input_file', type=click.Path(exists=False),
              help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--log-file', default=None, type=click.Path(),
              help=f'File to write logs to [default: None].')
# Model settings
@click.option('--model', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Dispersion interactions start to switch off at (lr_cutoff - dispersion_damping) Å. [default: {DEFAULT_DISPERSION_DAMPING}]')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for short-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for long-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
              help=f'Total charge of the system. [default: {DEFAULT_TOTAL_CHARGE}]')
@click.option('--jit-compile/--no-jit-compile', 'jit_compile', default=True,
              help='Use JIT compilation for speed. [default: enabled]')
# Tuning settings
@click.option('--kspace_type', default=None, type=str,
              help=f'Type of kspace calculation. [default: {"pme"}]')
# Help option
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
@click.pass_context
def tune_ewald(
    ctx: click.Context,
    settings: Optional[str],
    # Input/Output group
    input_file: str,
    log_file: Optional[str],
    # Model settings group
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    total_charge: int,
    jit_compile: bool,
    kspace_type: str,
    # Help option
    help: bool
) -> None:
    """
    Tune Ewald parameters for running SO3LR model on a given structure.

    Example:
        so3lr tune-ewald --input geometry.xyz
    """
     # Print help if needed
    if help:
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
                settings_dict = yaml.safe_load(f)

            # Use standard logging initially for any errors
            logger.info(f"Loading settings from {settings}")
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Error loading settings file: {str(e)}")
            sys.exit(1)

    for key in ctx.params:
        settings_dict = update_settings_from_defaults(ctx, settings_dict, key)

    # Generate default log file name based on output file if not explicitly provided
    if settings_dict.get('log_file') is None:
        settings_dict.update({log_file: f"{Path(input_file).stem}_tune-pme.log"})

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
    logger.info(f"JIT compilation:           {'Enabled' if settings_dict.get('jit_compile') else 'Disabled'}")


    # Log hardware info
    get_hardware_info()

    # Run tuning
    time_start = time.time()
    tune(settings_dict)
    time_end = time.time()
    logger.info("=" * 60)
    logger.info('Tunings completed successfully!')
    logger.info(f'Total runtime: {(time_end - time_start):.2f} seconds ({(time_end - time_start)/3600:.2f} hours)')


def main() -> None:
    """Entry point for the command line interface."""

    cli(standalone_mode=False)


if __name__ == '__main__':
    main()



