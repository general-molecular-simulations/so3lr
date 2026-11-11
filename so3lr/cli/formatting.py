"""Shared terminal labels, without importing the numerical backend."""
import math


def precision_label(value):
    """Explain the resolved mixed precision used by both MD engines."""
    label = str(value)
    if label == 'mixed':
        return 'mixed (float32 model; float64 dynamics)'
    return label


def _positive_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def throughput_label(seconds_per_step, dt_fs=None, atoms=None):
    """Available cycle rates, normalized by all simulated atoms when known.

    ``seconds_per_step`` is the existing host timing, not startup or output
    time. Missing inputs suppress only the rate that depends on them.
    """
    seconds = _positive_number(seconds_per_step)
    if seconds is None:
        return ''
    labels = []
    dt = _positive_number(dt_fs)
    if dt is not None:
        ns_per_day = _positive_number(dt * 0.0864 / seconds)
        if ns_per_day is not None:
            labels.append(f'{ns_per_day:.3f} ns/day')
    atom_count = _positive_number(atoms)
    if atom_count is not None:
        seconds_per_atom = _positive_number(seconds / atom_count)
        if seconds_per_atom is not None:
            labels.append(f'{seconds_per_atom:.2e} s/atom/step')
    return ' · '.join(labels)


def optimization_status(info, status):
    """Separate finishing a job from satisfying its recorded force criterion."""
    if str(status).lower() in ('failed', 'stopped'):
        return 'Optimization stopped'
    # FIRE compares against every explicitly supplied finite threshold. Zero
    # is an exact-force target, not the absence of a stopping criterion.
    raw_target = info.get('opt_force_target_eVA')
    try:
        target = float(raw_target) if not isinstance(raw_target, bool) else float('nan')
    except (TypeError, ValueError, OverflowError):
        target = float('nan')
    target = target if math.isfinite(target) else None
    try:
        force = float(info.get('opt_force_max_eVA'))
    except (TypeError, ValueError, OverflowError):
        force = float('nan')
    if target is not None and math.isfinite(force) and 0 <= force <= target:
        return 'Force criterion met'
    if str(status).lower() in ('complete', 'completed'):
        return ('Step budget reached; force criterion not met' if target is not None
                else 'Step budget finished (no force target)')
    return 'Relaxing geometry'
