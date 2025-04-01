"""Command line interfaces for SO3LR."""
import os
import sys
import click
import yaml
import numpy as np
import ase
import jax
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from ase.io import read, write

from .so3lr_eval import evaluate_so3lr_on
from .so3lr_md import (
    perform_min,
    perform_md,
    run
)

# ASCII art banner
SO3LR_ASCII = """
  ███████╗ ██████╗ ██████╗ ██╗     ██████╗ 
  ██╔════╝██╔═══██╗╚════██╗██║     ██╔══██╗
  ███████╗██║   ██║ █████╔╝██║     ██████╔╝
  ╚════██║██║   ██║ ╚═══██╗██║     ██╔══██╗
  ███████║╚██████╔╝██████╔╝███████╗██║  ██║
  ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
"""

# Default settings
DEFAULT_TEMPERATURE = 300.0  # Kelvin
DEFAULT_PRESSURE = 1.0  # atmospheres
DEFAULT_TIMESTEP = 0.0005  # picoseconds
DEFAULT_CYCLES = 100
DEFAULT_STEPS_PER_CYCLE = 100
DEFAULT_LR_CUTOFF = 12.0  # Angstrom
DEFAULT_DISPERSION_DAMPING = 2.0
DEFAULT_BUFFER_MULTIPLIER = 1.25
DEFAULT_PRECISION = "float32"
DEFAULT_NHC_CHAIN_LENGTH = 3
DEFAULT_NHC_STEPS = 2
DEFAULT_NHC_INTEGRATION_STEPS = 2
DEFAULT_NHC_TAU = 100.0
DEFAULT_NHC_THERMO = 100.0
DEFAULT_NHC_BARO = 1000.0
DEFAULT_SUZUKI_YOSHIDA_STEPS = 3
DEFAULT_NHC_NPT_TAU = 1000.0
DEFAULT_HDF5_BUFFER = 50
DEFAULT_SEED = 42
DEFAULT_LOG_FILE = "so3lr_md.log"
DEFAULT_TOTAL_CHARGE = 0
DEFAULT_MIN_CYCLES = 10
DEFAULT_MIN_STEPS = 10

# Help strings for CLI
FULL_HELP_STRING = """
Run simulations using SO3LR Machine Learned Force Field.

## Commands

so3lr [options]         Run MD simulation with options specified via command line or settings file
so3lr opt [options]     Run geometry optimization
so3lr nvt [options]     Run NVT (constant volume and temperature) MD simulation
so3lr npt [options]     Run NPT (constant pressure and temperature) MD simulation
so3lr eval [options]    Run evaluation on a dataset

## Usage Examples

Run with settings file:
  so3lr --settings_md.yaml

Optimize a structure with all options:
  so3lr opt --input geometry.xyz --output optimized.xyz --save-trajectory 
      --model /path/to/model --cycles 10 --steps 10 
      --n-min 2 --dt-start 0.05 --dt-max 0.1 --force-conv 0.05
      --precision float32 --lr-cutoff 12.0 --disp-damping 2.0 
      --buffer-sr 1.25 --buffer-lr 1.25 --total-charge 0

Run NVT simulation with all options:
  so3lr nvt --input geometry.xyz --trajectory nvt_traj.hdf5 --temperature 300
      --model /path/to/model --dt 0.0005 --cycles 100 --steps 100
      --nhc-chain 3 --nhc-steps 2 --nhc-tau 100.0 --relax
      --seed 42 --log-file nvt_md.log --precision float32
      --lr-cutoff 12.0 --disp-damping 2.0 --buffer-sr 1.25 --buffer-lr 1.25
      --total-charge 0

  # Using extxyz format instead of HDF5:
  so3lr nvt --input geometry.xyz --trajectory nvt_traj.xyz --temperature 300
      --model /path/to/model --dt 0.0005 --cycles 100 --steps 100

Run NPT simulation with all options:
  so3lr npt --input geometry.xyz --trajectory npt_traj.hdf5 --temperature 300
      --pressure 1.0 --model /path/to/model --dt 0.0005
      --cycles 100 --steps 100 --nhc-chain 3 --nhc-steps 2 --nhc-tau 100.0
      --nhc-baro 100.0 --nhc-npt-tau 1000.0 --relax --seed 42
      --log-file npt_md.log --precision float32 --lr-cutoff 12.0
      --disp-damping 2.0 --buffer-sr 1.25 --buffer-lr 1.25 --total-charge 0

Evaluate SO3LR on a dataset with all options:
  so3lr eval --datafile data.extxyz --batch-size 10 --lr-cutoff 12.0
      --disp-damping 2.0 --jit-compile --save-predictions-to predictions.extxyz

Run with all command line options:
  so3lr --input geometry.xyz --trajectory traj.hdf5
      --model /path/to/model --precision float32
      --lr-cutoff 12.0 --disp-damping 2.0 --buffer-sr 1.25 --buffer-lr 1.25
      --hdf5-buffer 100 --restart-save restart.npz --restart-load previous.npz
      --dt 0.0005 --temperature 300.0 --pressure 1.0 --cycles 100 --steps 100
      --nhc-chain 3 --nhc-steps 2 --nhc-thermo 100.0 --nhc-tau 100.0
      --nhc-baro 100.0 --nhc-sy-steps 3 --nhc-npt-tau 1000.0
      --total-charge 0 --seed 42 --log-file md.log --relax

### Settings File Examples
Example settings file for NVT simulation (nvt_settings.yaml):
```yaml
# Input/Output
initial_geometry: "path/to/structure.xyz"    # Path to the initial geometry file
output_file: "nvt_trajectory.hdf5"           # Output trajectory file
trajectory_format: "hdf5"                    # Format for trajectory: 'hdf5' or 'extxyz'

# Model settings
precision: "float32"                         # Numerical precision: 'float32' or 'float64'
lr_cutoff: 12.0                              # Long-range cutoff distance in Å
dispersion_energy_cutoff_lr_damping: 2.0     # Damping factor for long-range dispersion interactions
buffer_size_multiplier_sr: 1.25              # Buffer size multiplier for short-range interactions
buffer_size_multiplier_lr: 1.25              # Buffer size multiplier for long-range interactions

# MD settings
md_dt: 0.0005                                # MD timestep in picoseconds
md_T: 300.0                                  # Simulation temperature in Kelvin
md_cycles: 100                               # Number of MD cycles to run
md_steps: 100                                # Number of steps per MD cycle
relax_before_run: true                       # Whether to perform geometry relaxation before MD

# Thermostat settings
nhc_chain_length: 3                          # Length of the Nose-Hoover thermostat chain
nhc_steps: 2                                 # Number of integration steps per MD step
nhc_thermo: 100.0                            # Thermostat timescale in femtoseconds

# Additional settings
total_charge: 0                              # Total charge of the system
seed: 42                                     # Random seed for MD
```

Example settings file for NPT simulation (npt_settings.yaml):
```yaml
# Input/Output
initial_geometry: "path/to/structure.xyz"    # Path to the initial geometry file
output_file: "npt_trajectory.hdf5"           # Output trajectory file
trajectory_format: "hdf5"                    # Format for trajectory: 'hdf5' or 'extxyz'

# Model settings
precision: "float32"                         # Numerical precision: 'float32' or 'float64'
lr_cutoff: 12.0                              # Long-range cutoff distance in Å
dispersion_energy_cutoff_lr_damping: 2.0     # Damping factor for long-range dispersion interactions
buffer_size_multiplier_sr: 1.25              # Buffer size multiplier for short-range interactions
buffer_size_multiplier_lr: 1.25              # Buffer size multiplier for long-range interactions

# MD settings
md_dt: 0.0005                                # MD timestep in picoseconds
md_T: 300.0                                  # Simulation temperature in Kelvin
md_P: 1.0                                    # Pressure in atmospheres (enables NPT simulation)
md_cycles: 100                               # Number of MD cycles to run
md_steps: 100                                # Number of steps per MD cycle
relax_before_run: true                       # Whether to perform geometry relaxation before MD

# Thermostat and barostat settings
nhc_chain_length: 3                          # Length of the Nose-Hoover thermostat chain
nhc_steps: 2                                 # Number of integration steps per MD step
nhc_thermo: 100.0                            # Thermostat timescale in femtoseconds
nhc_baro: 1000.0                             # Barostat timescale
nhc_sy_steps: 3                              # Number of Suzuki-Yoshida integration steps
nhc_npt_tau: 1000.0                          # Barostat coupling constant

# Additional settings
total_charge: 0                              # Total charge of the system
seed: 42                                     # Random seed for MD
```

Example settings file for geometry optimization (opt_settings.yaml):
```yaml
# Input/Output
initial_geometry: "path/to/structure.xyz"    # Path to the initial geometry file
output_file: "optimized.xyz"                 # Output file to save the optimized geometry

# Model settings
precision: "float32"                         # Numerical precision: 'float32' or 'float64'
lr_cutoff: 12.0                              # Long-range cutoff distance in Å
dispersion_energy_cutoff_lr_damping: 2.0     # Damping factor for long-range dispersion interactions
buffer_size_multiplier_sr: 1.25              # Buffer size multiplier for short-range interactions
buffer_size_multiplier_lr: 1.25              # Buffer size multiplier for long-range interactions

# Optimization settings
min_cycles: 10                               # Number of minimization cycles
min_steps: 10                                # Number of steps per minimization cycle
min_start_dt: 0.05                           # Initial timestep for minimization
min_max_dt: 0.1                              # Maximum timestep for minimization
min_n_min: 2                                 # Number of minimizers to average
force_convergence: 0.05                      # Force convergence criterion in eV/Å
save_optimization_trajectory: true           # Save all optimization steps, not just final structure

# Additional settings
total_charge: 0                              # Total charge of the system
```
"""

BASIC_HELP_STRING = """
Run simulations using SO3LR Machine Learned Force Field.

## Commands

so3lr [options]         Run with options specified via command line or settings file
so3lr opt [options]     Run geometry optimization
so3lr nvt [options]     Run NVT (constant volume and temperature) MD simulation
so3lr npt [options]     Run NPT (constant pressure and temperature) MD simulation
so3lr eval [options]    Evaluate SO3LR model on a dataset

## Usage Examples

Run with settings file:
  so3lr --settings md_settings.yaml

Optimize a structure:
  so3lr opt --input geometry.xyz

Run NVT simulation:
  so3lr nvt --input geometry.xyz --trajectory so3lr_nvt.xyz

Run NPT simulation:
  so3lr npt --input geometry.xyz --trajectory so3lr_npt.xyz

Evaluate on a dataset:
  so3lr eval --datafile geometry.xyz --save-predictions-to so3lr_eval.extxyz

Use --help-full to see all available options.
"""

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
    """Custom group to handle --nvt and --npt flags that set appropriate defaults."""
    
    def parse_args(self, ctx: click.Context, args: List[str]) -> List[str]:
        """Parse arguments with special handling for --nvt and --npt flags."""
        # Check for simulation mode flags and modify args accordingly
        if '--nvt' in args:
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


@click.group(cls=NVTNPTGroup, invoke_without_command=True, 
             help="Run molecular dynamics using SO3LR Machine Learned Force Field.",
             context_settings={"help_option_names": []})
@click.option('--settings', type=click.Path(exists=False), default=None, 
              help='Path to YAML settings file.')
@click.option('--input', 'input_file', type=click.Path(exists=False), default=None, 
              help='Input geometry file (any ASE-readable format).')
@click.option('--output', 'output_file', type=click.Path(), default=None, 
              help='Output trajectory file (.hdf5 or .xyz).')
@click.option('--output-format', type=click.Choice(['hdf5', 'extxyz']), default=None, 
              help='Format for trajectory output (default: determined by file extension).')
@click.option('--model', 'model_path', type=click.Path(exists=False), default=None, 
              help='Path to MLFF model directory. If not provided, pretrained SO3LR model is used.')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']), 
              help=f'Numerical precision for calculations [default: {DEFAULT_PRECISION}].')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float, 
              help=f'Long-range cutoff distance in Å [default: {DEFAULT_LR_CUTOFF} Å].')
@click.option('--disp-damping', '--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float, 
              help=f'Damping factor for long-range dispersion interactions [default: {DEFAULT_DISPERSION_DAMPING}].')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float, 
              help=f'Buffer size multiplier for short-range interactions [default: {DEFAULT_BUFFER_MULTIPLIER}].')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float, 
              help=f'Buffer size multiplier for long-range interactions [default: {DEFAULT_BUFFER_MULTIPLIER}].')
@click.option('--hdf5-buffer', default=DEFAULT_HDF5_BUFFER, type=int, 
              help=f'Number of frames to buffer before writing [default: {DEFAULT_HDF5_BUFFER}].')
@click.option('--restart-save', type=click.Path(), default=None, 
              help='Path to save restart data.')
@click.option('--restart-load', type=click.Path(exists=False), default=None, 
              help='Path to load restart data from a previous run.')
@click.option('--dt', default=DEFAULT_TIMESTEP, type=float, 
              help=f'MD timestep in picoseconds [default: {DEFAULT_TIMESTEP} ps].')
@click.option('--temperature', default=DEFAULT_TEMPERATURE, type=float, 
              help=f'Simulation temperature in Kelvin [default: {DEFAULT_TEMPERATURE} K].')
@click.option('--pressure', default=None, type=float, 
              help='Simulation pressure in atmospheres (enables NPT) [default: None].')
@click.option('--cycles', default=DEFAULT_CYCLES, type=int, 
              help=f'Number of MD cycles to run [default: {DEFAULT_CYCLES}].')
@click.option('--steps', default=DEFAULT_STEPS_PER_CYCLE, type=int, 
              help=f'Number of steps per MD cycle [default: {DEFAULT_STEPS_PER_CYCLE}].')
@click.option('--opt-cycles', '--opt_cycles', 'opt_cycles', default=DEFAULT_MIN_CYCLES, type=int, 
              help=f'Number of minimization cycles to perform. [default: {DEFAULT_MIN_CYCLES}]')
@click.option('--opt-steps', '--opt_steps', 'opt_steps', default=DEFAULT_MIN_STEPS, type=int, 
              help=f'Number of steps per minimization cycle. [default: {DEFAULT_MIN_STEPS}]')
@click.option('--nhc-chain', default=DEFAULT_NHC_CHAIN_LENGTH, type=int, 
              help=f'Length of the Nose-Hoover thermostat chain [default: {DEFAULT_NHC_CHAIN_LENGTH}].')
@click.option('--nhc-steps', default=DEFAULT_NHC_INTEGRATION_STEPS, type=int, 
              help=f'Number of integration steps per MD step [default: {DEFAULT_NHC_INTEGRATION_STEPS}].')
@click.option('--nhc-thermo', default=DEFAULT_NHC_THERMO, type=float, 
              help=f'Thermostat timescale in femtoseconds [default: {DEFAULT_NHC_THERMO}].')
@click.option('--nhc-baro', default=DEFAULT_NHC_BARO, type=float, 
              help=f'Barostat timescale [default: {DEFAULT_NHC_BARO}].')
@click.option('--nhc-sy-steps', default=DEFAULT_SUZUKI_YOSHIDA_STEPS, type=int, 
              help=f'Number of Suzuki-Yoshida integration steps [default: {DEFAULT_SUZUKI_YOSHIDA_STEPS}].')
@click.option('--nhc-npt-tau', default=DEFAULT_NHC_NPT_TAU, type=float, 
              help=f'Barostat coupling constant [default: {DEFAULT_NHC_NPT_TAU}].')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int, 
              help=f'Total charge of the system [default: {DEFAULT_TOTAL_CHARGE}].')
@click.option('--seed', default=DEFAULT_SEED, type=int, 
              help=f'Random seed for MD [default: {DEFAULT_SEED}].')
@click.option('--log-file', default=DEFAULT_LOG_FILE, type=click.Path(), 
              help=f'File to write logs to [default: {DEFAULT_LOG_FILE}].')
@click.option('--relax/--no-relax', default=True, 
              help='Perform geometry relaxation before MD [default: enabled].')
@click.option('--force-conv', default=None, type=float, 
              help='Force convergence criterion in eV/Å for initial relaxation [default: None].')
@click.option('--help-full', is_flag=True, 
              help='Show detailed information about MD settings.')
@click.option('--help', '-h', is_flag=True, 
              help='Show brief command overview.')
@click.option('--nvt', is_flag=True, hidden=True, 
              help='Run NVT simulation (default).')
@click.option('--npt', is_flag=True, hidden=True, 
              help='Run NPT simulation with default pressure of 1.0 atm.')
@click.pass_context
def cli(ctx: click.Context, 
    settings: Optional[str], 
    input_file: Optional[str],
    output_file: Optional[str],
    output_format: Optional[str],
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    hdf5_buffer: int,
    restart_save: Optional[str],
    restart_load: Optional[str],
    dt: float,
    temperature: float,
    pressure: Optional[float],
    cycles: int,
    steps: int,
    opt_cycles: int,
    opt_steps: int,
    nhc_chain: int,
    nhc_steps: int,
    nhc_thermo: float,
    nhc_baro: float,
    nhc_sy_steps: int,
    nhc_npt_tau: float,
    total_charge: int,
    seed: int,
    log_file: str,
    relax: bool,
    force_conv: Optional[float],
    help_full: bool,
    help: bool,
    nvt: bool,
    npt: bool
) -> None:
    """
    Run molecular dynamics simulations with SO3LR or MLFF models.
    
    This command provides three subcommands:
    
    \b
    opt: Run geometry optimization
    nvt: Run NVT (constant volume and temperature) MD simulation
    npt: Run NPT (constant pressure and temperature) MD simulation
    
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
        click.echo(SO3LR_ASCII)
        click.echo(BASIC_HELP_STRING)
        return
    
    # Skip execution if used in a command group context
    if ctx.invoked_subcommand is not None:
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
                
            click.echo(f"Loading settings from {settings}")
        except (FileNotFoundError, yaml.YAMLError) as e:
            click.echo(f"Error loading settings file: {str(e)}")
            sys.exit(1)
    
    # Override settings with command line arguments if provided
    if input_file is not None:
        settings_dict['initial_geometry'] = input_file
    if output_file is not None:
        settings_dict['output_file'] = output_file
    if output_format is not None:
        settings_dict['output_format'] = output_format
    if model_path is not None:
        settings_dict['model_path'] = model_path
    if precision is not None:
        settings_dict['precision'] = precision
    if lr_cutoff is not None:
        settings_dict['lr_cutoff'] = lr_cutoff
    if dispersion_damping is not None:
        settings_dict['dispersion_energy_cutoff_lr_damping'] = dispersion_damping
    if buffer_sr is not None:
        settings_dict['buffer_size_multiplier_sr'] = buffer_sr
    if buffer_lr is not None:
        settings_dict['buffer_size_multiplier_lr'] = buffer_lr
    if hdf5_buffer is not None:
        settings_dict['hdf5_buffer_size'] = hdf5_buffer
    if restart_save is not None:
        settings_dict['restart_save_path'] = restart_save
    if restart_load is not None:
        settings_dict['restart_load_path'] = restart_load
    if dt is not None:
        settings_dict['md_dt'] = dt
    if temperature is not None:
        settings_dict['md_T'] = temperature
    if pressure is not None:
        settings_dict['md_P'] = pressure
    if cycles is not None:
        settings_dict['md_cycles'] = cycles
    if steps is not None:
        settings_dict['md_steps'] = steps
    if opt_cycles is not None:
        settings_dict['min_cycles'] = opt_cycles
    if opt_steps is not None:
        settings_dict['min_steps'] = opt_steps
    if nhc_chain is not None:
        settings_dict['nhc_chain_length'] = nhc_chain
    if nhc_steps is not None:
        settings_dict['nhc_steps'] = nhc_steps
    if nhc_thermo is not None:
        settings_dict['nhc_thermo'] = nhc_thermo
    if nhc_baro is not None:
        settings_dict['nhc_baro'] = nhc_baro
    if nhc_sy_steps is not None:
        settings_dict['nhc_sy_steps'] = nhc_sy_steps
    if nhc_npt_tau is not None:
        settings_dict['nhc_npt_tau'] = nhc_npt_tau
    if total_charge is not None:
        settings_dict['total_charge'] = total_charge
    if seed is not None:
        settings_dict['seed'] = seed
    if log_file is not None:
        settings_dict['log_file'] = log_file
    if relax is not None:
        settings_dict['relax_before_run'] = relax
    if force_conv is not None:
        settings_dict['force_convergence'] = force_conv
    
    # Validate required settings
    if 'initial_geometry' not in settings_dict:
        click.echo("Error: Initial geometry file must be specified either in settings file or with --input")
        sys.exit(1)
    
    # Set default output file if not specified
    if 'output_file' not in settings_dict:
        input_name = Path(settings_dict['initial_geometry']).stem
        settings_dict['output_file'] = f'{input_name}_trajectory.xyz'
        click.echo(f"No output file specified, using default: {settings_dict['output_file']}")
    
    # Print simulation details
    click.echo("=" * 60)
    click.echo(f"SO3LR Molecular Dynamics Simulation")
    click.echo("=" * 60)
    click.echo(f"Initial geometry:       {settings_dict['initial_geometry']}")
    click.echo(f"Output file:            {settings_dict['output_file']}")
    click.echo(f"Force field:            {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        click.echo(f"Model path:             {settings_dict['model_path']}")
    click.echo(f"Long-range cutoff:      {settings_dict.get('lr_cutoff', DEFAULT_LR_CUTOFF)} Å")
    click.echo(f"Dispersion damping:     {settings_dict.get('dispersion_energy_cutoff_lr_damping', DEFAULT_DISPERSION_DAMPING)} Å")
    click.echo(f"Short-range buffer:     {settings_dict.get('buffer_size_multiplier_sr', DEFAULT_BUFFER_MULTIPLIER)}")
    click.echo(f"Long-range buffer:      {settings_dict.get('buffer_size_multiplier_lr', DEFAULT_BUFFER_MULTIPLIER)}")
    click.echo(f"Total charge:           {settings_dict.get('total_charge', DEFAULT_TOTAL_CHARGE)}")
    click.echo(f"Precision:              {settings_dict.get('precision', DEFAULT_PRECISION)}")
    
    click.echo(f"Temperature:            {settings_dict.get('md_T', DEFAULT_TEMPERATURE)} K")
    
    if settings_dict.get('md_P') is not None:
        click.echo(f"Pressure:               {settings_dict.get('md_P')} atm")
        click.echo(f"Ensemble:               NPT")
    else:
        click.echo(f"Ensemble:               NVT")
    
    total_steps = settings_dict.get('md_cycles', DEFAULT_CYCLES) * settings_dict.get('md_steps', DEFAULT_STEPS_PER_CYCLE)
    simulation_time = total_steps * settings_dict.get('md_dt', DEFAULT_TIMESTEP)  # in ps
    
    click.echo(f"Simulation length:      {total_steps} steps ({simulation_time:.2f} ps)")
    click.echo(f"MD cycles:              {settings_dict.get('md_cycles', DEFAULT_CYCLES)}")
    click.echo(f"Steps per cycle:        {settings_dict.get('md_steps', DEFAULT_STEPS_PER_CYCLE)}")
    click.echo(f"Timestep:               {settings_dict.get('md_dt', DEFAULT_TIMESTEP)*1000} fs")
    click.echo(f"NHC length:             {settings_dict.get('nhc_chain_length', DEFAULT_NHC_CHAIN_LENGTH)}")
    click.echo(f"Nose-Hoover steps:      {settings_dict.get('nhc_steps', DEFAULT_NHC_INTEGRATION_STEPS)}")
    click.echo(f"Nose-Hoover thermo:     {settings_dict.get('nhc_thermo', DEFAULT_NHC_THERMO)}")
    click.echo(f"Seed:                   {settings_dict.get('seed', DEFAULT_SEED)}")
    
    if settings_dict.get('relax_before_run', False):
        click.echo(f"Geometry relaxation:    Enabled")
        if force_conv is not None:
            click.echo(f"Force convergence:      {force_conv} eV/Å")
        else:
            click.echo(f"Optimization cycles:   {settings_dict.get('min_cycles', DEFAULT_MIN_CYCLES)} cycles, each {settings_dict.get('min_steps', DEFAULT_MIN_STEPS)} steps")
    else:
        click.echo(f"Geometry relaxation:    Disabled")
        
    click.echo("=" * 60)
    
    # Run the simulation
    run(settings_dict)
    # try:
    #     run(settings_dict)
    #     # click.echo(f"Results saved to: {settings_dict['output_file']}")
    # except Exception as e:
    #     click.echo("=" * 60)
    #     click.echo(f"Error during simulation: {str(e)}")
    #     sys.exit(1)


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

# Define the 'opt' subcommand with clear help
@cli.command(name='opt', help="Run geometry optimization with `so3lr opt --input geometry.xyz`.")
@click.option('--input', '--input_file', 'input_file', type=click.Path(exists=False),
              help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', type=click.Path(),
              help='Output file to save the optimized geometry. If not provided, defaults to <input_name_without_extension>_opt.xyz.')
@click.option('--save-trajectory/--no-save-trajectory', 'save_optimization_trajectory', default=True,
              help='Save some optimization steps in the output file, not just the final structure. [default: True]')
@click.option('--model', '--model_path', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
              help=f'Total charge of the system. [default: {DEFAULT_TOTAL_CHARGE}]')
@click.option('--opt-cycles', '--opt_cycles', 'opt_cycles', default=DEFAULT_MIN_CYCLES, type=int,
              help=f'Number of minimization cycles to perform. [default: {DEFAULT_MIN_CYCLES}]')
@click.option('--opt-steps', '--opt_steps', 'opt_steps', default=DEFAULT_MIN_STEPS, type=int,
              help=f'Number of steps per minimization cycle. [default: {DEFAULT_MIN_STEPS}]')
@click.option('--min-start-dt', '--dt-start', 'min_start_dt', default=0.05, type=float,
              help='The initial step size during minimization. [default: 0.05]')
@click.option('--min-max-dt', '--dt-max', 'min_max_dt', default=0.1, type=float,
              help='The maximum step size during minimization. [default: 0.1]')
@click.option('--n-min', default=2, type=int,
              help='Minimum number of steps moving in the correct direction before dt is updated. [default: 2]')
@click.option('--force-conv', default=None, type=float,
              help='Force convergence criterion in eV/Å, overrides --cycles and --steps. [default: None]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--disp-damping', '--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Damping factor for long-range dispersion interactions. [default: {DEFAULT_DISPERSION_DAMPING}]')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for short-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for long-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def fire_optimization(
    input_file: Optional[str],
    output_file: Optional[str],
    save_optimization_trajectory: bool,
    model_path: Optional[str],
    total_charge: int,
    opt_cycles: int,
    opt_steps: int,
    min_start_dt: float,
    min_max_dt: float,
    n_min: int,
    force_conv: Optional[float],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
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
    click.echo(SO3LR_ASCII)
    
    if not input_file or help:
        click.echo(fire_optimization.get_help(click.get_current_context()))
        return

    # Generate default output file name if not provided
    if input_file is not None and output_file is None:
        input_path = Path(input_file)
        output_file = f"{input_path.stem}_opt{input_path.suffix}"
    
    click.echo("=" * 60)
    click.echo(f"SO3LR Geometry Optimization")
    click.echo("=" * 60)
    click.echo(f"Initial geometry:                 {input_file}")
    click.echo(f"Output geometry:                  {output_file}")
    click.echo(f"Save trajectory:                  {'Yes' if save_optimization_trajectory else 'No'}")
    click.echo(f"Total charge:                     {total_charge}")
    click.echo(f"Force field:                      {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        click.echo(f"Model path:                      {model_path}")
    click.echo(f"Long-range cutoff:                {lr_cutoff} Å")
    if force_conv is not None:
        click.echo(f"Force convergence:                {force_conv} eV/Å")
    else:
        click.echo(f"Optimization cycles:               {opt_cycles} cycles, each {opt_steps} steps")
    click.echo(f"Initial step size:                {min_start_dt} Å")
    click.echo(f"Maximum step size:                {min_max_dt} Å")
    click.echo(f"Steps between step size updates:  {n_min}")
    click.echo(f"Precision:                        {precision}")
    click.echo(f"Dispersion damping:               {dispersion_damping} Å")
    click.echo(f"Short-range buffer:               {buffer_sr}")
    click.echo(f"Long-range buffer:                {buffer_lr}")
    click.echo("=" * 60)
    
    # Create settings dictionary
    settings = {
        'initial_geometry': input_file,
        'output_file': output_file,
        'save_optimization_trajectory': save_optimization_trajectory,
        'model_path': model_path,
        'min_cycles': opt_cycles,
        'min_steps': opt_steps,
        'min_start_dt': min_start_dt,
        'min_max_dt': min_max_dt,
        'min_n_min': n_min,
        'force_convergence': force_conv,
        'precision': precision,
        'lr_cutoff': lr_cutoff,
        'dispersion_energy_cutoff_lr_damping': dispersion_damping,
        'buffer_size_multiplier_sr': buffer_sr,
        'buffer_size_multiplier_lr': buffer_lr,
        'total_charge': total_charge,
    }
    
    time_start = time.time()
    try:
        # Run optimization - result will be written directly to output_file
        perform_min(settings)
        time_end = time.time()
        click.echo(f"Optimization completed in {time_end - time_start:.1f} seconds")
        click.echo("=" * 60)
    except Exception as e:
        click.echo("=" * 60)
        click.echo(f"Error during optimization: {str(e)}")
        sys.exit(1)

# Define the 'nvt' subcommand with clear help
@cli.command(name='nvt', help="Run NVT molecular dynamics simulation with `so3lr nvt --input geometry.xyz`.")
@click.option('--input', '--input_file', 'input_file', type=click.Path(exists=False),
              help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', type=click.Path(),
              help='Output file to save the trajectory. If not provided, defaults to <input_name_without_extension>_nvt.xyz.')
@click.option('--output-format', type=click.Choice(['hdf5', 'extxyz']), default='extxyz',
              help='Format for trajectory output (default: extxyz).')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
              help=f'Total charge of the system. [default: {DEFAULT_TOTAL_CHARGE}]')
@click.option('--model', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--disp-damping', '--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Damping factor for long-range dispersion interactions. [default: {DEFAULT_DISPERSION_DAMPING}]')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for short-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for long-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--temperature', default=DEFAULT_TEMPERATURE, type=float,
              help=f'Simulation temperature in Kelvin. [default: {DEFAULT_TEMPERATURE}]')
@click.option('--dt', default=DEFAULT_TIMESTEP, type=float,
              help=f'MD timestep in picoseconds. [default: {DEFAULT_TIMESTEP}]')
@click.option('--cycles', default=DEFAULT_CYCLES, type=int,
              help=f'Number of MD cycles to run. [default: {DEFAULT_CYCLES}]')
@click.option('--steps', default=DEFAULT_STEPS_PER_CYCLE, type=int,
              help=f'Number of steps per MD cycle. [default: {DEFAULT_STEPS_PER_CYCLE}]')
@click.option('--nhc-chain', default=DEFAULT_NHC_CHAIN_LENGTH, type=int,
              help=f'Length of the Nose-Hoover thermostat chain. [default: {DEFAULT_NHC_CHAIN_LENGTH}]')
@click.option('--nhc-steps', default=DEFAULT_NHC_INTEGRATION_STEPS, type=int,
              help=f'Number of integration steps per MD step. [default: {DEFAULT_NHC_INTEGRATION_STEPS}]')
@click.option('--nhc-thermo', default=DEFAULT_NHC_THERMO, type=float,
              help=f'Thermostat timescale in femtoseconds. [default: {DEFAULT_NHC_THERMO}]')
@click.option('--relax/--no-relax', default=True,
              help='Perform geometry relaxation before MD. [default: enabled]')
@click.option('--force-conv', default=None, type=float,
              help='Force convergence criterion in eV/Å for initial relaxation. [default: None]')
@click.option('--seed', default=DEFAULT_SEED, type=int,
              help=f'Random seed for MD. [default: {DEFAULT_SEED}]')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def nvt_md(
    input_file: Optional[str], 
    output_file: Optional[str],
    output_format: str,
    total_charge: int,
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    temperature: float,
    dt: float,
    cycles: int,
    steps: int,
    nhc_chain: int,
    nhc_steps: int,
    nhc_thermo: float,
    relax: bool,
    force_conv: Optional[float],
    seed: int,
    help: bool
) -> None:
    """
    Run NVT (constant volume and temperature) molecular dynamics simulation.
    
    This command runs a molecular dynamics simulation in the NVT ensemble
    (constant number of particles, volume, and temperature) using the
    Nose-Hoover chain thermostat.
    
    Example:
        so3lr nvt --input geometry.xyz --temperature 300
    """
    # Print ASCII art at the beginning
    click.echo(SO3LR_ASCII)
    
    # Print possible arguments if no arguments are provided
    if not input_file or help:
        click.echo(nvt_md.get_help(click.get_current_context()))
        return

    # Generate default output file name if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = f"{input_path.stem}_nvt.xyz"

    # Forward to the main command with appropriate options
    ctx = click.get_current_context()
    ctx.forward(cli, 
                input_file=input_file,
                output_file=output_file,
                output_format=output_format,
                total_charge=total_charge,
                model_path=model_path,
                precision=precision,
                lr_cutoff=lr_cutoff,
                dispersion_damping=dispersion_damping,
                buffer_sr=buffer_sr,
                buffer_lr=buffer_lr,
                temperature=temperature,
                dt=dt,
                cycles=cycles,
                steps=steps,
                nhc_chain=nhc_chain,
                nhc_steps=nhc_steps,
                nhc_thermo=nhc_thermo,
                relax=relax,
                force_conv=force_conv,
                seed=seed)

# Define the 'npt' subcommand with clear help
@cli.command(name='npt', help="Run NPT molecular dynamics simulation.")
@click.option('--input', '--input_file', 'input_file', type=click.Path(exists=False),
              help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', type=click.Path(),
              help='Output trajectory file (.hdf5 or .xyz). [default: <input_name_without_extension>_npt.xyz]')
@click.option('--output-format', type=click.Choice(['hdf5', 'extxyz']), default='extxyz',
              help='Format for trajectory output (default: extxyz).')
@click.option('--total-charge', default=DEFAULT_TOTAL_CHARGE, type=int,
              help=f'Total charge of the system. [default: {DEFAULT_TOTAL_CHARGE}]')
@click.option('--model', '--model_path', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff distance in Å. [default: {DEFAULT_LR_CUTOFF}]')
@click.option('--disp-damping', '--dispersion-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Damping factor for long-range dispersion interactions. [default: {DEFAULT_DISPERSION_DAMPING}]')
@click.option('--buffer-sr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for short-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--buffer-lr', default=DEFAULT_BUFFER_MULTIPLIER, type=float,
              help=f'Buffer size multiplier for long-range interactions. [default: {DEFAULT_BUFFER_MULTIPLIER}]')
@click.option('--temperature', default=DEFAULT_TEMPERATURE, type=float,
              help=f'Simulation temperature in Kelvin. [default: {DEFAULT_TEMPERATURE}]')
@click.option('--pressure', default=DEFAULT_PRESSURE, type=float,
              help=f'Simulation pressure in atmospheres. [default: {DEFAULT_PRESSURE}]')
@click.option('--dt', default=DEFAULT_TIMESTEP, type=float,
              help=f'MD timestep in picoseconds. [default: {DEFAULT_TIMESTEP}]')
@click.option('--cycles', default=DEFAULT_CYCLES, type=int,
              help=f'Number of MD cycles to run. [default: {DEFAULT_CYCLES}]')
@click.option('--steps', default=DEFAULT_STEPS_PER_CYCLE, type=int,
              help=f'Number of steps per MD cycle. [default: {DEFAULT_STEPS_PER_CYCLE}]')
@click.option('--nhc-chain', default=DEFAULT_NHC_CHAIN_LENGTH, type=int,
              help=f'Length of the Nose-Hoover thermostat chain. [default: {DEFAULT_NHC_CHAIN_LENGTH}]')
@click.option('--nhc-steps', default=DEFAULT_NHC_INTEGRATION_STEPS, type=int,
              help=f'Number of integration steps per MD step. [default: {DEFAULT_NHC_INTEGRATION_STEPS}]')
@click.option('--nhc-thermo', 'nhc_thermo', default=DEFAULT_NHC_THERMO, type=float,
              help=f'Thermostat timescale in femtoseconds. [default: {DEFAULT_NHC_THERMO}]')
@click.option('--nhc-baro', default=DEFAULT_NHC_BARO, type=float,
              help=f'Barostat timescale. [default: {DEFAULT_NHC_BARO}]')
@click.option('--nhc-npt-tau', default=DEFAULT_NHC_NPT_TAU, type=float,
              help=f'Barostat coupling constant. [default: {DEFAULT_NHC_NPT_TAU}]')
@click.option('--relax/--no-relax', default=True,
              help='Perform geometry relaxation before MD. [default: enabled]')
@click.option('--force-conv', default=None, type=float,
              help='Force convergence criterion in eV/Å for initial relaxation. [default: None]')
@click.option('--seed', default=DEFAULT_SEED, type=int,
              help=f'Random seed for MD. [default: {DEFAULT_SEED}]')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def npt_md(
    input_file: Optional[str], 
    output_file: Optional[str], 
    output_format: str,
    total_charge: int,
    model_path: Optional[str],
    precision: str,
    lr_cutoff: float,
    dispersion_damping: float,
    buffer_sr: float,
    buffer_lr: float,
    temperature: float,
    pressure: float,
    dt: float,
    cycles: int,
    steps: int, 
    nhc_chain: int,
    nhc_steps: int,
    nhc_thermo: float,
    nhc_baro: float,
    nhc_npt_tau: float,
    relax: bool,
    force_conv: Optional[float],
    seed: int,
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
    # Print ASCII art at the beginning
    click.echo(SO3LR_ASCII)
    
    # Print possible arguments if no arguments are provided
    if not input_file or help:
        click.echo(npt_md.get_help(click.get_current_context()))
        return
    
    # Generate default output file name if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = f"{input_path.stem}_npt.xyz"

    # Forward to the main command with appropriate options
    ctx = click.get_current_context()
    ctx.forward(cli, 
                input_file=input_file,
                output_file=output_file,
                output_format=output_format,
                total_charge=total_charge,
                model_path=model_path,
                precision=precision,
                lr_cutoff=lr_cutoff,
                dispersion_damping=dispersion_damping,
                buffer_sr=buffer_sr,
                buffer_lr=buffer_lr,
                temperature=temperature,
                pressure=pressure,
                dt=dt,
                cycles=cycles,
                steps=steps,
                nhc_chain=nhc_chain,
                nhc_steps=nhc_steps,
                nhc_thermo=nhc_thermo,
                nhc_baro=nhc_baro,
                nhc_npt_tau=nhc_npt_tau,
                relax=relax,
                force_conv=force_conv,
                seed=seed)


@cli.command(name='eval', help="Evaluate SO3LR model on a dataset.")
@click.option('--datafile', type=click.Path(exists=False),
              help='Data file to evaluate the model on. Must be readable by ase.io.read. [default: None]')
@click.option('--batch-size', default=1, type=int,
              help='Number of molecules per batch. [default: 1]')
@click.option('--lr-cutoff', default=DEFAULT_LR_CUTOFF, type=float,
              help=f'Long-range cutoff for SO3LR in Å. [default: {DEFAULT_LR_CUTOFF} Å]')
@click.option('--precision', default=DEFAULT_PRECISION, type=click.Choice(['float32', 'float64']),
              help=f'Numerical precision for calculations. [default: {DEFAULT_PRECISION}]')
@click.option('--model', 'model_path', type=click.Path(exists=False),
              help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--disp-damping', 'dispersion_damping', default=DEFAULT_DISPERSION_DAMPING, type=float,
              help=f'Damping factor for long-range dispersion interactions. [default: {DEFAULT_DISPERSION_DAMPING}]')
@click.option('--jit-compile/--no-jit-compile', default=True,
              help='JIT compile the calculation. [default: enabled]')
@click.option('--save-predictions-to', type=click.Path(),
              help='File path where to save the predictions (.extxyz format). [default: None]')
@click.option('--targets', default='forces',
              help='Targets to evaluate, separated by commas (e.g. "forces,dipole_vec,hirshfeld_ratios"). [default: forces]')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def eval_model(
    datafile: Optional[str],
    batch_size: int,
    lr_cutoff: float,
    dispersion_damping: float,
    jit_compile: bool,
    save_predictions_to: Optional[str],
    targets: str,
    model_path: Optional[str],
    precision: str,
    help: bool
) -> None:
    """
    Evaluate SO3LR model on a dataset.
    
    This command evaluates the SO3LR model on a provided dataset,
    calculating metrics like MAE, MSE, and RMSE for forces, dipole
    vectors, and other targets. Predictions can optionally be saved
    to a file.
    
    Example:
        so3lr eval --datafile data.extxyz --save-predictions-to predictions.extxyz
    """
    # Print ASCII art at the beginning
    click.echo(SO3LR_ASCII)

    if not datafile or help:
        click.echo(eval_model.get_help(click.get_current_context()))
        return
    
    click.echo("=" * 60)
    click.echo(f"SO3LR Model Evaluation")
    click.echo("=" * 60)
    click.echo(f"Dataset file:           {datafile}")
    click.echo(f"Batch size:             {batch_size}")
    click.echo(f"Long-range cutoff:      {lr_cutoff} Å")
    click.echo(f"Dispersion damping:     {dispersion_damping}")
    click.echo(f"JIT compilation:        {'Enabled' if jit_compile else 'Disabled'}")
    if save_predictions_to:
        click.echo(f"Saving predictions to:  {save_predictions_to}")
    click.echo(f"Targets:                {targets}")
    if model_path:
        click.echo(f"Model path:             {model_path}")
    click.echo(f"Precision:              {precision}")
    
    click.echo("Starting evaluation...")
    click.echo("=" * 60)
    
    # Validate file existence
    if not Path(datafile).exists():
        click.echo(f"Error: Dataset file not found: {datafile}")
        sys.exit(1)
    
    # Validate output path
    if save_predictions_to:
        save_path = Path(save_predictions_to)
        if save_path.exists():
            click.echo(f"Error: Output file already exists: {save_predictions_to}")
            sys.exit(1)
        # Create parent directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Call the evaluate_so3lr_on function from so3lr_eval.py
    try:
        evaluate_so3lr_on(
            datafile=datafile,
            batch_size=batch_size,
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_damping,
            jit_compile=jit_compile,
            save_predictions_to=save_predictions_to,
            model_path=model_path,
            precision=precision,
            targets=targets
        )
        click.echo("=" * 60)
        click.echo("Evaluation completed successfully!")
        
        if save_predictions_to:
            click.echo(f"Predictions saved to: {save_predictions_to}")
            
    except Exception as e:
        click.echo("=" * 60)
        click.echo(f"Error during evaluation: {str(e)}")
        sys.exit(1)


def main() -> None:
    """Entry point for the command line interface."""
    cli(standalone_mode=False)
    # try:
    #     cli(standalone_mode=False)
    # except click.exceptions.Abort:
    #     # Handle --help flags gracefully
    #     pass
    # except click.exceptions.UsageError as e:
    #     # Handle usage errors
    #     click.echo(f"Usage error: {str(e)}")
    #     sys.exit(1)
    # except click.exceptions.FileError as e:
    #     # Handle file errors
    #     click.echo(f"File error: {str(e)}")
    #     sys.exit(1)
    # except Exception as e:
    #     # Handle unexpected errors
    #     click.echo(f"Error: {str(e)}")
    #     if os.environ.get('SO3LR_DEBUG', '0') == '1':
    #         import traceback
    #         click.echo(traceback.format_exc())
    #     sys.exit(1)


if __name__ == '__main__':
    main() 