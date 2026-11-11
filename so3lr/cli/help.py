"""Import-cheap, generated command help with task-oriented option groups."""
import click


OPTION_GROUPS = (
    ('Input and output', {'input_file', 'output_file', 'datafile', 'save_to', 'log_file',
                          'config', 'workdir', 'settings', 'overwrite', 'targets'}),
    ('Model and precision', {'model_path', 'precision', 'total_charge', 'num_unpaired_electrons',
                             'charge_embed_total_charge', 'charge_equilibration_total_charge'}),
    ('Dynamics', {'temperature', 'pressure', 'dt', 'md_cycles', 'md_steps', 'seed',
                  'nhc_chain', 'nhc_steps', 'nhc_thermo', 'nhc_baro', 'nhc_sy_steps',
                  'langevin_thermo', 'langevin_removecmmotion'}),
    ('Relaxation and restart', {'relax', 'force_conv', 'min_cycles', 'min_steps',
                                'restart_load', 'restart_save', 'save_buffer'}),
    ('Engine and performance', {'engine', 'flash_mode', 'flash_kernels', 'flash_devices',
                                'flash_mixed', 'buffer_sr', 'buffer_lr', 'jit_compile', 'batch_size'}),
    ('Long-range interactions', {'lr_cutoff', 'dispersion_damping', 'kspace_electrostatics',
                                 'kspace_smearing', 'kspace_spacing', 'kspace_interp_nodes',
                                 'kspace_npt_cycles', 'kspace_dispersion', 'kspace_dispersion_smearing',
                                 'kspace_dispersion_spacing', 'kspace_dispersion_interp_nodes'}),
)


class GroupedCommand(click.Command):
    """Keep Click's option definitions authoritative, including future options."""

    def format_options(self, ctx, formatter):
        grouped = {label: [] for label, _ in OPTION_GROUPS}
        other, help_rows = [], []
        for parameter in self.get_params(ctx):
            row = parameter.get_help_record(ctx)
            if row is None:
                continue
            if parameter.name in {'help', 'help_full'}:
                help_rows.append(row)
                continue
            for label, names in OPTION_GROUPS:
                if parameter.name in names:
                    grouped[label].append(row)
                    break
            else:
                other.append(row)
        for label, rows in [*grouped.items(), ('Other options', other), ('Help', help_rows)]:
            if rows:
                with formatter.section(label):
                    formatter.write_dl(rows)


def format_overview(command, ctx, formatter, version):
    """Render commands/options from Click metadata and a few useful examples."""
    formatter.write_text(click.style(f'SO3LR TAURI  {version}', fg='yellow', bold=True))
    formatter.write_text('Molecular simulation with machine-learned force fields.')
    formatter.write_paragraph()
    formatter.write_usage(ctx.command_path, '[OPTIONS] [COMMAND] [ARGS]...')
    groups = (
        ('Prepare and monitor', ('init', 'show')),
        ('Simulate', ('nvt', 'nvt-langevin', 'npt', 'nve', 'opt')),
        ('Evaluate and train', ('eval', 'train', 'finetune', 'tune-ewald')),
    )
    descriptions = {
        'init': 'Choose a structure, review settings, then start or save',
        'show': 'Browse a run folder and inspect simulation plots',
        'nvt': 'NVT dynamics with a Nose–Hoover thermostat',
        'nvt-langevin': 'NVT dynamics with a Langevin thermostat',
        'npt': 'NPT dynamics at constant temperature and pressure',
        'nve': 'NVE dynamics at constant volume and energy',
        'opt': 'Relax a molecular structure',
        'eval': 'Evaluate a model on a dataset',
        'train': 'Train a model from a configuration file',
        'finetune': 'Fine-tune a model on custom data',
        'tune-ewald': 'Tune Ewald, PME or P3M parameters',
    }
    shown = set()
    for label, names in groups:
        rows = []
        for name in names:
            child = command.get_command(ctx, name)
            if child is not None and not child.hidden:
                rows.append((name, descriptions.get(name, child.get_short_help_str())))
                shown.add(name)
        if rows:
            with formatter.section(label):
                formatter.write_dl(rows)
    extras = []
    for name in command.list_commands(ctx):
        child = command.get_command(ctx, name)
        if name not in shown and child is not None and not child.hidden:
            extras.append((name, child.get_short_help_str()))
    if extras:
        with formatter.section('Other commands'):
            formatter.write_dl(extras)
    with formatter.section('Start here'):
        formatter.write_text('\b\nso3lr init\n'
                             'so3lr --settings run.yaml\n'
                             'so3lr show runs/\n'
                             'so3lr nvt --input molecule.xyz --lr-cutoff 100\n'
                             'so3lr nvt --help')
        formatter.write_paragraph()
        formatter.write_text('In an interactive terminal, bare so3lr also opens guided setup. '
                             'Use --settings for an existing YAML run, including batch jobs.')
    rows = [p.get_help_record(ctx) for p in command.get_params(ctx)]
    with formatter.section('Global options (before the command)'):
        formatter.write_dl([row for row in rows if row is not None])
    if ctx.params.get('help_full'):
        with formatter.section('More examples'):
            formatter.write_text('\b\nso3lr opt --input molecule.xyz --lr-cutoff 100\n'
                                 'so3lr eval --datafile dataset.extxyz --save-to predictions.extxyz\n'
                                 'so3lr --display plain --settings run.yaml')
        formatter.write_paragraph()
        formatter.write_text('Use COMMAND --help for every option and default. '
                             'YAML runs use --settings without a subcommand.')
    formatter.write_paragraph()
    formatter.write_text('Documentation: https://so3lr.github.io/usage/cli/')
