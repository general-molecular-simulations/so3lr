"""Command line interfaces for SO3LR."""
import os
import click
import yaml
import numpy as np
import ase
import jax
import time
from ase.io import read, write

from ..ascii_string import so3lr_ascii_II
from .so3lr_eval import evaluate_so3lr_on
from .so3lr_md import (
    perform_min,
    perform_md,
    run,
    write_to_extxyz
)

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
trajectory_file: "nvt_trajectory.hdf5"       # Output trajectory file
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
nhc_tau: 100.0                               # Thermostat coupling constant

# Additional settings
total_charge: 0                              # Total charge of the system
seed: 42                                     # Random seed for MD
```

Example settings file for NPT simulation (npt_settings.yaml):
```yaml
# Input/Output
initial_geometry: "path/to/structure.xyz"    # Path to the initial geometry file
trajectory_file: "npt_trajectory.hdf5"       # Output trajectory file
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
nhc_tau: 100.0                               # Thermostat coupling constant
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

## Command Line Options

### Input/Output
--settings PATH        Path to YAML settings file
--input PATH           Input geometry file (any ASE-readable format) 
--trajectory PATH      Output trajectory file (.hdf5 or .xyz)
--trajectory-format    Format for trajectory output (hdf5 or extxyz)
--log-file PATH        File to write logs to [default: so3lr_md.log]

### Model Selection
--model PATH           Path to MLFF model directory to use custom MLFF model (default: None)

--precision CHOICE     Numerical precision [float32/float64, default: float32]

### Cutoffs and Buffers  
--lr-cutoff FLOAT     Long-range cutoff distance in Å [default: 12.0]
--disp-damping FLOAT  Damping factor for long-range dispersion [default: 2.0]
--buffer-sr FLOAT     Buffer size multiplier for short-range [default: 1.25]
--buffer-lr FLOAT     Buffer size multiplier for long-range [default: 1.25]
--hdf5-buffer INT     Number of frames to buffer before writing [default: 100]

### Restart Options
--restart-save PATH   Path to save restart data
--restart-load PATH   Path to load restart data from previous run

### MD Parameters
--dt FLOAT           MD timestep in picoseconds [default: 0.0005]
--temperature FLOAT  Simulation temperature in Kelvin [default: 300.0]
--pressure FLOAT     Simulation pressure in atmospheres (enables NPT)
--cycles INT         Number of MD cycles to run [default: 100]
--steps INT          Number of steps per MD cycle [default: 100]

### Nose-Hoover Chain Parameters
--nhc-chain INT     Length of thermostat chain [default: 3]
--nhc-steps INT     Integration steps per MD step [default: 2]
--nhc-thermo FLOAT  Thermostat timescale in femtoseconds
--nhc-tau FLOAT     Thermostat coupling constant [default: 100.0]
--nhc-baro FLOAT    Barostat timescale
--nhc-sy-steps INT  Number of Suzuki-Yoshida integration steps
--nhc-npt-tau FLOAT Barostat coupling constant

### Other Settings
--total-charge INT  Total charge of the system [default: 0]
--seed INT          Random seed for MD [default: 42]
--relax/--no-relax  Perform geometry relaxation before MD [default: relax]
--help-full         Show this detailed help message
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
    
    def format_help(self, ctx, formatter):
        # Display ascii art at the top
        formatter.write(so3lr_ascii_II)# + "\n")
        
        # For basic help, show the basic help string
        if not ctx.params.get('help_full', False):
            formatter.write(BASIC_HELP_STRING)
        else:
            formatter.write(FULL_HELP_STRING)

class NVTNPTGroup(CustomCommandClass):
    """Custom group to handle --nvt and --npt flags that set appropriate defaults."""
    
    def parse_args(self, ctx, args):
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
                args.extend(['--pressure', '1.0'])
                
        # Call the super parse_args
        return super().parse_args(ctx, args)
    
    def get_command(self, ctx, cmd_name):
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            cmd.context_settings["help_option_names"] = ["--help"]
        return cmd


@click.group(cls=NVTNPTGroup, invoke_without_command=True, 
             help="Run molecular dynamics using SO3LR Machine Learned Force Field.",
             context_settings={"help_option_names": []})
@click.option('--settings', default=None, help='Path to YAML settings file.')
@click.option('--input', 'input_file', default=None, help='Input geometry file (any ASE-readable format).')
@click.option('--output', 'output_file', default=None, help='Output trajectory file (.hdf5 or .xyz).')
@click.option('--output-format', type=click.Choice(['hdf5', 'extxyz']), default=None, help='Format for trajectory output (default: determined by file extension).')
@click.option('--model', 'model_path', default=None, help='Path to MLFF model directory. If not provided, pretrained SO3LR model is used.')
@click.option('--precision', default='float32', type=click.Choice(['float32', 'float64']), help='Numerical precision for calculations [default: float32].')
@click.option('--lr-cutoff', default=12.0, type=float, help='Long-range cutoff distance in Å [default: 12.0 Å].')
@click.option('--disp-damping', '--dispersion-damping', 'dispersion_damping', default=2.0, type=float, help='Damping factor for long-range dispersion interactions [default: 2.0].')
@click.option('--buffer-sr', default=1.25, type=float, help='Buffer size multiplier for short-range interactions [default: 1.25].')
@click.option('--buffer-lr', default=1.25, type=float, help='Buffer size multiplier for long-range interactions [default: 1.25].')
@click.option('--hdf5-buffer', default=100, type=int, help='Number of frames to buffer before writing [default: 100].')
@click.option('--restart-save', default=None, help='Path to save restart data.')
@click.option('--restart-load', default=None, help='Path to load restart data from a previous run.')
@click.option('--dt', default=0.0005, type=float, help='MD timestep in picoseconds [default: 0.0005 ps].')
@click.option('--temperature', default=300.0, type=float, help='Simulation temperature in Kelvin [default: 300.0 K].')
@click.option('--pressure', default=None, type=float, help='Simulation pressure in atmospheres (enables NPT) [default: None].')
@click.option('--cycles', default=100, type=int, help='Number of MD cycles to run [default: 100].')
@click.option('--steps', default=100, type=int, help='Number of steps per MD cycle [default: 100].')
@click.option('--opt_cycles', default=10, type=int, help='Number of minimization cycles to perform. [default: 10]')
@click.option('--opt_steps', default=10, type=int, help='Number of steps per minimization cycle. [default: 10]')
@click.option('--nhc-chain', default=3, type=int, help='Length of the Nose-Hoover thermostat chain [default: 3].')
@click.option('--nhc-steps', default=2, type=int, help='Number of integration steps per MD step [default: 2].')
@click.option('--nhc-thermo', default=None, type=float, help='Thermostat timescale in femtoseconds [default: None].')
@click.option('--nhc-tau', default=100.0, type=float, help='Thermostat coupling constant [default: 100.0].')
@click.option('--nhc-baro', default=None, type=float, help='Barostat timescale [default: None].')
@click.option('--nhc-sy-steps', default=None, type=int, help='Number of Suzuki-Yoshida integration steps [default: None].')
@click.option('--nhc-npt-tau', default=None, type=float, help='Barostat coupling constant [default: None].')
@click.option('--total-charge', default=0, type=int, help='Total charge of the system [default: 0].')
@click.option('--seed', default=42, type=int, help='Random seed for MD [default: 42].')
@click.option('--log-file', default='so3lr_md.log', help='File to write logs to [default: so3lr_md.log].')
@click.option('--relax/--no-relax', default=True, help='Perform geometry relaxation before MD [default: enabled].')
@click.option('--force-conv', default=None, type=float, help='Force convergence criterion in eV/Å for initial relaxation [default: None].')
@click.option('--help-full', is_flag=True, help='Show detailed information about MD settings.')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
@click.option('--nvt', is_flag=True, hidden=True, help='Run NVT simulation (default).')
@click.option('--npt', is_flag=True, hidden=True, help='Run NPT simulation with default pressure of 1.0 atm.')
@click.pass_context
def cli(ctx, 
    settings, 
    input_file,
    output_file,
    output_format,
    model_path,
    precision,
    lr_cutoff,
    dispersion_damping,
    buffer_sr,
    buffer_lr,
    hdf5_buffer,
    restart_save,
    restart_load,
    dt,
    temperature,
    pressure,
    cycles,
    steps,
    opt_cycles,
    opt_steps,
    nhc_chain,
    nhc_steps,
    nhc_thermo,
    nhc_tau,
    nhc_baro,
    nhc_sy_steps,
    nhc_npt_tau,
    total_charge,
    seed,
    log_file,
    relax,
    force_conv,
    help_full,
    help,
    nvt,
    npt
):
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
        click.echo(so3lr_ascii_II)
        click.echo(BASIC_HELP_STRING)
        return
    
    # Skip execution if used in a command group context
    if ctx.invoked_subcommand is not None:
        return
        
    # Load settings from file if provided
    if settings is not None:
        try:
            settings_dict = yaml.safe_load(open(settings, 'r'))
            click.echo(f"Loading settings from {settings}")
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find settings file at {settings}!')
    else:
        settings_dict = {}
    
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
        settings_dict['opt_cycles'] = opt_cycles
    if opt_steps is not None:
        settings_dict['opt_steps'] = opt_steps
    if nhc_chain is not None:
        settings_dict['nhc_chain_length'] = nhc_chain
    if nhc_steps is not None:
        settings_dict['nhc_steps'] = nhc_steps
    if nhc_thermo is not None:
        settings_dict['nhc_thermo'] = nhc_thermo
    if nhc_tau is not None:
        settings_dict['nhc_tau'] = nhc_tau
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
        settings_dict['force_conv'] = force_conv
    
    # Validate required settings
    if 'initial_geometry' not in settings_dict:
        click.echo("Error: Initial geometry file must be specified either in settings file or with --input")
        return
    
    if 'output_file' not in settings_dict:
        settings_dict['output_file'] = 'trajectory.xyz'
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
    click.echo(f"Long-range cutoff:      {settings_dict['lr_cutoff']} Å")
    click.echo(f"Dispersion damping:     {settings_dict['dispersion_energy_cutoff_lr_damping']} Å")
    click.echo(f"Short-range buffer:     {settings_dict['buffer_size_multiplier_sr']}")
    click.echo(f"Long-range buffer:      {settings_dict['buffer_size_multiplier_lr']}")
    click.echo(f"Total charge:           {settings_dict['total_charge']}")
    click.echo(f"Precision:              {settings_dict['precision']}")
    
    click.echo(f"Temperature:            {settings_dict.get('md_T', 300.0)} K")
    
    if settings_dict.get('md_P') is not None:
        click.echo(f"Pressure:               {settings_dict.get('md_P')} atm")
        click.echo(f"Ensemble:               NPT")
    else:
        click.echo(f"Ensemble:               NVT")
    
    click.echo(f"Simulation length:      {settings_dict.get('md_cycles', 100) * settings_dict.get('md_steps', 100)} steps")
    click.echo(f"MD steps:               {settings_dict.get('md_steps', 100)}")
    click.echo(f"MD cycles:              {settings_dict.get('md_cycles', 100)}")
    click.echo(f"Timestep:               {settings_dict.get('md_dt', 0.0005)*1000} fs")
    click.echo(f"NHC length:             {settings_dict.get('nhc_chain_length', 3)}")
    click.echo(f"Nose-Hoover steps:      {settings_dict.get('nhc_steps', 2)}")
    click.echo(f"Nose-Hoover tau:        {settings_dict.get('nhc_tau', 100.0)}")
    click.echo(f"Seed:                   {settings_dict.get('seed', 42)}")
    
    if settings_dict.get('relax_before_run', False):
        click.echo(f"Geometry relaxation:    True")
    else:
        click.echo(f"Geometry relaxation:    False")
    if force_conv is not None:
        click.echo(f"Force convergence:       {force_conv} eV/Å")
    else:
        click.echo(f"Force convergence:       {settings_dict.get('opt_cycles', 100)} cycles, each {settings_dict.get('opt_steps', 100)} steps")
        
    # click.echo("Starting simulation...")
    click.echo("=" * 60)
    
    # Run the simulation
    try:
        run(settings_dict)
        # click.echo("=" * 60)
        # click.echo("Simulation completed successfully!")
        click.echo(f"Results saved to: {settings_dict['output_file']}")
    except Exception as e:
        click.echo("=" * 60)
        click.echo(f"Error during simulation: {e}")
        raise


class SubcommandHelpGroup(click.Group):
    """Custom Group class for subcommands to handle help displays appropriately."""
    
    def get_help(self, ctx):
        # Set short help mode for top-level options
        formatted_help = click.formatting.HelpFormatter()
        formatted_help.write(so3lr_ascii_II)# + "\n\n")
        self.format_usage(ctx, formatted_help)
        self.format_help_text(ctx, formatted_help)
        self.format_options(ctx, formatted_help)
        self.format_epilog(ctx, formatted_help)
        return formatted_help.getvalue()

# Define the 'opt' subcommand with clear help
@cli.command(name='opt', help="Run geometry optimization with `so3lr opt --input geometry.xyz`.")
@click.option('--input', '--input_file', 'input_file', required=False, help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', default=None, help='Output file to save the optimized geometry. If not provided, defaults to <input_name_without_extension>_opt.xyz.')
@click.option('--save-trajectory/--no-save-trajectory', 'save_optimization_trajectory', default=True, help='Save some optimization steps in the output file, not just the final structure. [default: True]')
@click.option('--model', '--model_path', 'model_path', default=None, help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--total-charge', default=0, type=int, help='Total charge of the system. [default: 0]')
@click.option('--opt_cycles', default=10, type=int, help='Number of minimization cycles to perform. [default: 10]')
@click.option('--opt_steps', default=10, type=int, help='Number of steps per minimization cycle. [default: 10]')
@click.option('--dt-start', default=0.05, type=float, help='The initial step size during minimization as a float. [default: 0.05]')
@click.option('--dt-max', default=0.1, type=float, help='The maximum step size during minimization as a float. [default: 0.1]')
@click.option('--n-min', default=2, type=int, help='An integer specifying the minimum number of steps moving in the correct direction before dt and f_alpha should be updated. [default: 2]')
@click.option('--force-conv', default=None, type=float, help='Force convergence criterion in eV/Å, overrides --cycles and --steps. [default: None]')
@click.option('--precision', default='float64', type=click.Choice(['float32', 'float64']), help='Numerical precision for calculations. [default: float32]')
@click.option('--lr-cutoff', default=12.0, type=float, help='Long-range cutoff distance in Å. [default: 12.0]')
@click.option('--disp-damping', '--dispersion-damping', 'dispersion_damping', default=2.0, type=float, help='Damping factor for long-range dispersion interactions. [default: 2.0]')
@click.option('--buffer-sr', default=1.25, type=float, help='Buffer size multiplier for short-range interactions. [default: 1.25]')
@click.option('--buffer-lr', default=1.25, type=float, help='Buffer size multiplier for long-range interactions. [default: 1.25]')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def fire_optimization(
    input_file,
    output_file,
    save_optimization_trajectory,
    model_path,
    total_charge,
    opt_cycles,
    opt_steps,
    dt_start,
    dt_max,
    n_min,
    force_conv,
    precision,
    lr_cutoff,
    dispersion_damping,
    buffer_sr,
    buffer_lr,
    help
):
    """
    Run geometry optimization using the FIRE algorithm.
    
    This command performs geometry optimization of a molecular structure
    using the FIRE (Fast Inertial Relaxation Engine) algorithm with either
    the SO3LR potential or a custom MLFF model.
    
    Example:
        so3lr opt --input geometry.xyz
    """
    click.echo(so3lr_ascii_II)
    
    if not input_file or help:
        click.echo(fire_optimization.get_help(click.get_current_context()))
        return

    if input_file is not None and output_file is None:
        output_file = input_file.split('.')[0] + '_opt.xyz'
    
    click.echo("=" * 60)
    click.echo(f"SO3LR Geometry Optimization")
    click.echo(f"Initial geometry:                 {input_file}")
    click.echo(f"Output geometry:                  {output_file}")
    if save_optimization_trajectory:
        click.echo(f"Save trajectory:                  Yes")
    else:
        click.echo(f"Save trajectory:                  No")
    click.echo(f"Total charge:                     {total_charge}")
    click.echo(f"Force field:                      {'Custom MLFF' if model_path else 'SO3LR'}")
    if model_path is not None:
        click.echo(f"Model path:                      {model_path}")
    click.echo(f"Long-range cutoff:                {lr_cutoff} Å")
    if force_conv is not None:
        click.echo(f"Force convergence:                {force_conv} eV/Å")
    else:
        click.echo(f"Force convergence:                {opt_cycles} cycles, each {opt_steps} steps")
    click.echo(f"Initial step size:                {dt_start} Å")
    click.echo(f"Maximum step size:                {dt_max} Å")
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
        'dt_start': dt_start,
        'dt_max': dt_max,
        'n_min': n_min,
        'force_conv': force_conv,
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
        click.echo(f"Optimization completed in {time_end - time_start:.0f} seconds")
        click.echo("=" * 60)
    except Exception as e:
        click.echo("=" * 60)
        click.echo(f"Error during optimization: {e}")
        raise

# Define the 'nvt' subcommand with clear help
@cli.command(name='nvt', help="Run NVT molecular dynamics simulation with `so3lr nvt --input geometry.xyz`.")
@click.option('--input', '--input_file', 'input_file', required=False, help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', default=None, help='Output file to save the optimized geometry. If not provided, defaults to <input_name_without_extension>_nvt.xyz.')
@click.option('--output-format', type=click.Choice(['hdf5', 'extxyz']), default='extxyz', help='Format for trajectory output (default: extxyz).')
@click.option('--total-charge', default=0, type=int, help='Total charge of the system. [default: 0]')
@click.option('--model', 'model_path', default=None, help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default='float32', type=click.Choice(['float32', 'float64']), help='Numerical precision for calculations. [default: float32]')
@click.option('--lr-cutoff', default=12.0, type=float, help='Long-range cutoff distance in Å. [default: 12.0]')
@click.option('--disp-damping', '--dispersion-damping', 'dispersion_damping', default=2.0, type=float, help='Damping factor for long-range dispersion interactions. [default: 2.0]')
@click.option('--buffer-sr', default=1.25, type=float, help='Buffer size multiplier for short-range interactions. [default: 1.25]')
@click.option('--buffer-lr', default=1.25, type=float, help='Buffer size multiplier for long-range interactions. [default: 1.25]')
@click.option('--temperature', default=300.0, type=float, help='Simulation temperature in Kelvin. [default: 300.0]')
@click.option('--dt', default=0.0005, type=float, help='MD timestep in picoseconds. [default: 0.0005]')
@click.option('--cycles', default=100, type=int, help='Number of MD cycles to run. [default: 100]')
@click.option('--steps', default=100, type=int, help='Number of steps per MD cycle. [default: 100]')
@click.option('--nhc-chain', default=3, type=int, help='Length of the Nose-Hoover thermostat chain. [default: 3]')
@click.option('--nhc-steps', default=2, type=int, help='Number of integration steps per MD step. [default: 2]')
@click.option('--nhc-tau', default=100.0, type=float, help='Thermostat coupling constant. [default: 100.0]')
@click.option('--relax/--no-relax', default=True, help='Perform geometry relaxation before MD. [default: enabled]')
@click.option('--force-conv', default=None, type=float, help='Force convergence criterion in eV/Å for initial relaxation. [default: None]')
@click.option('--seed', default=42, type=int, help='Random seed for MD. [default: 42]')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def nvt_md(
    input_file, 
    output_file,
    output_format,
    total_charge,
    model_path,
    precision,
    lr_cutoff,
    dispersion_damping,
    buffer_sr,
    buffer_lr,
    temperature,
    dt,
    cycles,
    steps,
    nhc_chain,
    nhc_steps,
    nhc_tau,
    relax,
    force_conv,
    seed,
    help
):
    """
    Run NVT (constant volume and temperature) molecular dynamics simulation.
    
    This command runs a molecular dynamics simulation in the NVT ensemble
    (constant number of particles, volume, and temperature) using the
    Nose-Hoover chain thermostat.
    
    Example:
        so3lr nvt --input geometry.xyz --temperature 300
    """
    # Print ASCII art at the beginning
    click.echo(so3lr_ascii_II)
    
    # Print possible arguments if no arguments are provided
    if not input_file or help:
        click.echo(nvt_md.get_help(click.get_current_context()))
        return

    if output_file is None:
        output_file = input_file.split('.')[0] + '_nvt.xyz'

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
                nhc_tau=nhc_tau,
                relax=relax,
                force_conv=force_conv,
                seed=seed)

# Define the 'npt' subcommand with clear help
@cli.command(name='npt', help="Run NPT molecular dynamics simulation.")
@click.option('--input', '--input_file', 'input_file', required=False, help='Input geometry file (any ASE-readable format). [default: None]')
@click.option('--output', '--output_file', 'output_file', default=None, help='Output trajectory file (.hdf5 or .xyz). [default:  <input_name_without_extension>_npt.xyz.]')
@click.option('--output-format', type=click.Choice(['hdf5', 'extxyz']), default='extxyz', help='Format for trajectory output (default: extxyz).')
@click.option('--total-charge', default=0, type=int, help='Total charge of the system. [default: 0]')
@click.option('--model', '--model_path', 'model_path', default=None, help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--precision', default='float32', type=click.Choice(['float32', 'float64']), help='Numerical precision for calculations. [default: float32]')
@click.option('--lr-cutoff', default=12.0, type=float, help='Long-range cutoff distance in Å. [default: 12.0]')
@click.option('--disp-damping', '--dispersion-damping', 'dispersion_damping', default=2.0, type=float, help='Damping factor for long-range dispersion interactions. [default: 2.0]')
@click.option('--buffer-sr', default=1.25, type=float, help='Buffer size multiplier for short-range interactions. [default: 1.25]')
@click.option('--buffer-lr', default=1.25, type=float, help='Buffer size multiplier for long-range interactions. [default: 1.25]')
@click.option('--temperature', default=300.0, type=float, help='Simulation temperature in Kelvin. [default: 300.0]')
@click.option('--pressure', default=1.0, type=float, help='Simulation pressure in atmospheres. [default: 1.0]')
@click.option('--dt', default=0.0005, type=float, help='MD timestep in picoseconds. [default: 0.0005]')
@click.option('--cycles', default=10, type=int, help='Number of MD cycles to run. [default: 10]')
@click.option('--steps', default=100, type=int, help='Number of steps per MD cycle. [default: 100]')
@click.option('--nhc-chain', default=3, type=int, help='Length of the Nose-Hoover thermostat chain. [default: 3]')
@click.option('--nhc-steps', default=2, type=int, help='Number of integration steps per MD step. [default: 2]')
@click.option('--nhc-tau', default=100.0, type=float, help='Thermostat coupling constant. [default: 100.0]')
@click.option('--nhc-baro', default=None, type=float, help='Barostat timescale. [default: None]')
@click.option('--nhc-npt-tau', default=None, type=float, help='Barostat coupling constant. [default: None]')
@click.option('--relax/--no-relax', default=True, help='Perform geometry relaxation before MD. [default: enabled]')
@click.option('--force-conv', default=None, type=float, help='Force convergence criterion in eV/Å for initial relaxation. [default: None]')
@click.option('--seed', default=42, type=int, help='Random seed for MD. [default: 42]')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def npt_md(
    input_file, 
    output_file, 
    output_format,
    total_charge,
    model_path,
    precision,
    lr_cutoff,
    dispersion_damping,
    buffer_sr,
    buffer_lr,
    temperature,
    pressure,
    dt,
    cycles,
    steps, 
    nhc_chain,
    nhc_steps,
    nhc_tau,
    nhc_baro,
    nhc_npt_tau,
    relax,
    force_conv,
    seed,
    help
):
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
    click.echo(so3lr_ascii_II)
    
    # Print possible arguments if no arguments are provided
    if not input_file or help:
        click.echo(npt_md.get_help(click.get_current_context()))
        return
    
    if output_file is None:
        output_file = input_file.split('.')[0] + '_npt.xyz'

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
                nhc_tau=nhc_tau,
                nhc_baro=nhc_baro,
                nhc_npt_tau=nhc_npt_tau,
                relax=relax,
                force_conv=force_conv,
                seed=seed)


@cli.command(name='eval', help="Evaluate SO3LR model on a dataset.")
@click.option('--datafile', required=False, help='Data file to evaluate the model on. Must be readable by ase.io.read. [default: None]')
@click.option('--batch-size', default=1, type=int, help='Number of molecules per batch. [default: 1]')
@click.option('--lr-cutoff', default=12.0, type=float, help='Long-range cutoff for SO3LR in Å. [default: 12.0 Å]')
@click.option('--precision', default='float32', type=click.Choice(['float32', 'float64']), help='Numerical precision for calculations. [default: float32]')
@click.option('--model', 'model_path', default=None, help='Path to MLFF model directory. If not provided, SO3LR is used. [default: None]')
@click.option('--disp-damping', 'dispersion_damping', default=2.0, type=float, help='Damping factor for long-range dispersion interactions. [default: 2.0]')
@click.option('--jit-compile/--no-jit-compile', default=True, help='JIT compile the calculation. [default: enabled]')
@click.option('--save-predictions-to', default=None, help='File path where to save the predictions (.extxyz format). [default: None]')
@click.option('--targets', default='forces', help='Targets to evaluate, separated by commas (e.g. "forces,dipole_vec,hirshfeld_ratios"). [default: forces]')
@click.option('--help', '-h', is_flag=True, help='Show brief command overview.')
def eval_model(
    datafile,
    batch_size,
    lr_cutoff,
    dispersion_damping,
    jit_compile,
    save_predictions_to,
    targets,
    model_path,
    precision,
    help
):
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
    click.echo(so3lr_ascii_II)

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
        click.echo(f"Model path:              {model_path}")
    click.echo(f"Precision:              {precision}")
    
    click.echo("Starting evaluation...")
    
    # Call the evaluate_so3lr_on function from so3lr_eval.py
    try:
        evaluate_so3lr_on(
            datafile=datafile,
            batch_size=batch_size,
            lr_cutoff=lr_cutoff,
            dispersion_energy_lr_cutoff_damping=dispersion_damping,
            jit_compile=jit_compile,
            save_predictions_to=save_predictions_to,
            model_path=model_path,
            precision=precision,
            targets=targets
        )
        click.echo("=" * 60)
        click.echo("Evaluation completed successfully!")
    except Exception as e:
        click.echo("=" * 60)
        click.echo(f"Error during evaluation: {e}")
        raise

def main():
    """Entry point for the command line interface."""
    try:
        cli(standalone_mode=False)
    except click.exceptions.Abort:
        # Handle --help flags gracefully
        pass
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        import traceback
        click.echo(traceback.format_exc())
        exit(1)


if __name__ == '__main__':
    main() 