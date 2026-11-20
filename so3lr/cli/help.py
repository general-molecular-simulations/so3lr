"""Import-cheap, generated command help with task-oriented option groups."""
import click


OPTION_GROUPS = (
    ('Input and output', ('input_file', 'frame', 'datafile', 'config_path', 'settings',
                          'output_file', 'save_to', 'directory', 'workdir', 'print_only',
                          'log_file', 'overwrite', 'save_buffer')),
    ('Dynamics', ('ensemble', 'temperature', 'pressure', 'dt', 'duration', 'frame_interval',
                  'md_cycles', 'md_steps', 'seed', 'keep_velocities')),
    ('Relaxation', ('relax', 'force_conv', 'min_cycles', 'min_steps', 'min_start_dt',
                    'min_max_dt', 'min_n_min')),
    ('Checkpoints', ('restart_load', 'restart_save', 'or_start')),
    ('Values to compute', ('targets', 'observable')),
    ('Monitoring', ('once', 'interval', 'as_json')),
    ('Model and precision', ('model_path', 'model', 'precision', 'total_charge', 'charge',
                             'num_unpaired_electrons', 'spin', 'charge_embed_total_charge',
                             'charge_equilibration_total_charge')),
    ('Training', ('strategy', 'num_train', 'num_valid')),
    ('Thermostat and barostat', ('nhc_chain', 'nhc_steps', 'nhc_thermo', 'nhc_baro', 'nhc_sy_steps',
                                 'langevin_thermo', 'langevin_removecmmotion')),
    ('Long-range interactions', ('lr_cutoff', 'cutoff', 'dispersion_damping', 'kspace_electrostatics',
                                 'kspace_smearing', 'kspace_spacing', 'kspace_interp_nodes',
                                 'kspace_npt_cycles', 'kspace_dispersion', 'kspace_dispersion_smearing',
                                 'kspace_dispersion_spacing', 'kspace_dispersion_interp_nodes')),
    ('Engine and performance', ('engine', 'flash_mode', 'flash_kernels', 'flash_devices',
                                'flash_mixed', 'buffer_sr', 'buffer_lr', 'jit_compile', 'batch_size')),
)

NVT_EXAMPLE = ('so3lr nvt --input molecule.xyz --output trajectory.xyz \\\n'
               '    --temperature 300 --dt 0.5 \\\n'
               '    --md-cycles 100 --md-steps 20')


def _write_documentation(formatter):
    formatter.write_paragraph()
    formatter.write_text('Documentation: https://so3lr.github.io/')


class GroupedCommand(click.Command):
    """Keep Click's option definitions authoritative, including future options."""

    def format_epilog(self, ctx, formatter):
        super().format_epilog(ctx, formatter)
        _write_documentation(formatter)

    def format_options(self, ctx, formatter):
        records = {}
        help_rows = []
        auto_help = self.get_help_option(ctx)
        for parameter in self.get_params(ctx):
            row = parameter.get_help_record(ctx)
            if row is None:
                continue
            if parameter is auto_help or parameter.name in ('help', 'help_full'):
                help_rows.append(row)
            else:
                records[parameter.name] = row
        for label, names in OPTION_GROUPS:
            rows = [records.pop(name) for name in names if name in records]
            if rows:
                with formatter.section(label):
                    formatter.write_dl(rows)
        if records:
            with formatter.section('Other options'):
                formatter.write_dl(records.values())
        if help_rows:
            with formatter.section('Help'):
                formatter.write_dl(help_rows)


def format_overview(command, ctx, formatter, version):
    """Render commands/options from Click metadata and a few useful examples."""
    formatter.write_text(f'SO3LR v{version}')
    formatter.write_text(command.help)
    formatter.write_paragraph()
    formatter.write_usage(ctx.command_path, '[OPTIONS] [COMMAND] [ARGS]...')
    with formatter.section('Run a simulation'):
        formatter.write_text('Run 1 ps of NVT dynamics at 300 K.')
        formatter.write_paragraph()
        formatter.write_text(f'\b\n{NVT_EXAMPLE}')
        formatter.write_paragraph()
        formatter.write_text('\b\n'
                             f'Use {click.style("so3lr start", bold=True)} for guided setup.\n'
                             f'Use {click.style("so3lr --settings run.yaml", bold=True)} to run a saved setup.')

    groups = (
        ('Set up and run', ('start', 'opt', 'nvt', 'nvt-langevin', 'npt', 'nve')),
        ('Monitor and continue', ('show', 'resume')),
        ('Evaluate and train', ('eval', 'finetune', 'train')),
        ('Tune electrostatics', ('tune-ewald',)),
    )
    shown = set()
    for label, names in groups:
        rows = []
        for name in names:
            child = command.get_command(ctx, name)
            if child is not None and not child.hidden:
                rows.append((name, child.get_short_help_str()))
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
    rows = [p.get_help_record(ctx) for p in command.get_params(ctx)]
    with formatter.section('Global options (before the command)'):
        formatter.write_dl([row for row in rows if row is not None])
        formatter.write_text('--display, --verbose, --quiet and --no-color may also follow the command.')
    if ctx.params.get('help_full'):
        with formatter.section('More examples'):
            formatter.write_text('\b\nso3lr start --input box.xyz --ensemble npt \\\n'
                                 '    --duration 1 --print\n'
                                 'so3lr opt --input molecule.xyz --lr-cutoff 100\n'
                                 'so3lr eval --datafile dataset.extxyz \\\n'
                                 '    --save-to predictions.extxyz\n'
                                 'so3lr --display plain --settings run.yaml\n'
                                 'so3lr show so3lr-run\n'
                                 'so3lr resume so3lr-run')
    formatter.write_paragraph()
    formatter.write_text('Use so3lr COMMAND --help for examples and options.')
    _write_documentation(formatter)
