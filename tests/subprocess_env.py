import os


def so3lr_subprocess_env(extra=None):
    env = os.environ.copy()
    env.update({
        'JAX_TRACEBACK_FILTERING': 'off',
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'TF_NUM_INTRAOP_THREADS': '1',
        'TF_NUM_INTEROP_THREADS': '1',
        'WANDB_MODE': 'disabled',
    })
    if extra:
        env.update(extra)
    return env


def _parse_final_table_loss(result, loss_col='loss'):
    """Return the ``loss`` value from the last row of the metrics table printed by
    the trainer, or ``None`` if no table row is found.

    The trainer logs validation checkpoints as a pipe-delimited table
    (see ``mlff.utils.training_utils.print_metrics_header``/``print_metrics_row``)::

            epoch |          step |    forces_mae | ... |          loss | ...
        ----------+---------------+---------------+ ... +---------------+ ...
                0 |             5 |        0.0630 | ... |        0.0097 | ...

    We locate the header row (containing ``epoch``, ``step`` and ``loss``), then
    take the last all-numeric data row and read its ``loss`` column.
    """
    lines = (result.stdout + result.stderr).splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines)
         if {'epoch', 'step', loss_col} <= {c.strip() for c in ln.split('|')}),
        None,
    )
    if header_idx is None:
        return None
    cols = [c.strip() for c in lines[header_idx].split('|')]
    idx = cols.index(loss_col)
    final = None
    for ln in lines[header_idx + 1:]:
        if '|' not in ln:
            continue
        cells = [c.strip() for c in ln.split('|')]
        if len(cells) != len(cols):
            continue
        try:
            final = float(cells[idx])
        except ValueError:
            continue  # separator ('---+---') or non-data line
    return final
