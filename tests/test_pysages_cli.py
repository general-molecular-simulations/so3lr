"""PySAGES and cross-coordinate restarts, end to end through the `so3lr` CLI.

Every test here compares the MD log rows of real `so3lr` runs: vanilla vs
unbiased PySAGES (must match), ABF vs vanilla (must differ), and three restarts
vs the continuous run they continue (must match). The runs are shared by the
module fixture below; each one happens at most once, on first use, and a
restart run first launches the run whose files it loads.

Every run: one benzoic acid molecule in a periodic 25 A box, float64, 100 steps
per cycle and no relaxation (for speed; a restart skips relaxation, so only the
first half of a restart would relax). The logs print E/KE/PE/H with 3 decimals
and T with 1, so the 1e-6 tolerances on log rows mean the printed rows are
identical; restarts are also compared on the final full-precision frame.
"""
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import yaml
from ase.io import read

pytest.importorskip("pysages", reason="PySAGES is an optional dependency")

from subprocess_env import so3lr_subprocess_env  # noqa: E402

GEOMETRY = pathlib.Path(__file__).resolve().parent / 'test_data' / 'benzoic_acid_aq_1.xyz'
SUBPROCESS_TIMEOUT = 1500

BASE_SETTINGS = {
    'precision': 'float64',
    'total_charge': 0.0,
    'lr_cutoff': 12.0,
    'relax_before_run': False,
    'md_dt': 0.5,  # fs (the CLI/YAML convention; converted fs->ps internally)
    'md_T': 300,
    'md_steps': 100,
    'input_file': GEOMETRY.name,
    'output_file': 'traj.xyz',
    'log_file': 'out.log',
}
CV = [{
    'type': 'coordinationnumber',
    'indices': [4, 7],
    'cn_exps': [6, 30],
    'species': 1,
    'grid_min': 0.0,
    'grid_max': 1.5,
    'grid_stride': 64,
    'periodic': False,
}]
PYSAGES = {'do_enhanced_sampling': True, 'enforce_cartesian_coordinates': True}

# name -> (settings overrides, PySAGES method or None, run whose restart files it loads)
RUNS = {
    'vanilla_4': ({'md_cycles': 4}, None, None),
    'unbiased_4': ({'md_cycles': 4, **PYSAGES}, 'unbiased', None),
    'abf_4': ({'md_cycles': 4, **PYSAGES}, 'abf', None),
    'vanilla_2_saved': ({'md_cycles': 2, 'restart_save_path': './restart.npz'}, None, None),
    'unbiased_2_saved': (
        {'md_cycles': 2, **PYSAGES, 'restart_save_path': './restart.npz',
         'restart_pysages_save_path': './restart_pysages.pkl'}, 'unbiased', None),
    # Fractional-coordinate (vanilla) restart loaded into a Cartesian PySAGES run.
    'restart_vanilla_to_unbiased': (
        {'md_cycles': 2, **PYSAGES, 'restart_load_path': './restart.npz',
         'restart_save_path': './restart2.npz'}, 'unbiased', 'vanilla_2_saved'),
    # Cartesian (PySAGES) restart loaded into a fractional-coordinate vanilla run.
    'restart_unbiased_to_vanilla': (
        {'md_cycles': 2, 'restart_load_path': './restart.npz',
         'restart_save_path': './restart2.npz'}, None, 'unbiased_2_saved'),
    # PySAGES restart that also resumes the saved PySAGES state.
    'restart_unbiased_with_state': (
        {'md_cycles': 2, **PYSAGES, 'restart_load_path': './restart.npz',
         'restart_save_path': './restart2.npz',
         'restart_pysages_load_path': './restart_pysages.pkl',
         'restart_pysages_save_path': './restart_pysages_2.pkl'},
        'unbiased', 'unbiased_2_saved'),
}


def _parse_md_rows(log_path):
    """{step, E, KE, PE, H, T} per MD log row: "<prefix> INFO - step\tE\tKE\tPE\tH\tT\ttime/step"."""
    rows = []
    for line in pathlib.Path(log_path).read_text().splitlines():
        parts = line.split("INFO - ", 1)[-1].strip().split('\t')
        if len(parts) != 7:
            continue
        try:
            rows.append({'step': int(parts[0]),
                         **dict(zip(('E', 'KE', 'PE', 'H', 'T'), map(float, parts[1:6])))})
        except ValueError:
            continue
    return rows


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """``run(name)`` -> the MD log rows of that `so3lr` run, launched once
    (``run.dirs[name]`` is its directory). A failed run is not relaunched."""
    run_dirs, failures = {}, {}

    def get(name):
        if name in failures:
            pytest.fail(failures[name])
        if name in run_dirs:
            return _parse_md_rows(run_dirs[name] / 'out.log')
        overrides, method, restart_from = RUNS[name]
        run_dir = tmp_path_factory.mktemp(f"pysages_{name}")
        shutil.copy2(GEOMETRY, run_dir / GEOMETRY.name)
        if restart_from is not None:
            get(restart_from)
            for key in ('restart_load_path', 'restart_pysages_load_path'):
                if key in overrides:
                    restart_file = pathlib.Path(overrides[key]).name
                    shutil.copy2(run_dirs[restart_from] / restart_file, run_dir / restart_file)
        with open(run_dir / 'md_settings.yaml', 'w') as f:
            yaml.dump({**BASE_SETTINGS, **overrides}, f)
        if method is not None:
            with open(run_dir / 'pysages_input.yaml', 'w') as f:
                yaml.dump({'method': method, 'cv': CV}, f)
        result = subprocess.run(
            ['so3lr', '--settings', 'md_settings.yaml'], cwd=run_dir, capture_output=True,
            text=True, env=so3lr_subprocess_env(), timeout=SUBPROCESS_TIMEOUT,
        )
        if result.returncode != 0:
            failures[name] = (f"SO3LR run '{name}' failed in {run_dir} with return code "
                              f"{result.returncode}\nSTDERR:\n{result.stderr}")
            pytest.fail(failures[name])
        run_dirs[name] = run_dir
        return _parse_md_rows(run_dir / 'out.log')

    get.dirs = run_dirs
    return get


ATOL = {'E': 1e-6, 'KE': 1e-6, 'PE': 1e-6, 'H': 1e-6, 'T': 0.1}


def _assert_rows_match(reference, other, label, keys=tuple(ATOL)):
    """Identical printed rows for `keys` (T to its 1 printed decimal), compared in order."""
    assert reference and other, f"{label}: no MD rows in a log"
    for ref, oth in zip(reference, other):
        for key in keys:
            np.testing.assert_allclose(oth[key], ref[key], atol=ATOL[key],
                                       err_msg=f"{label}: {key} mismatch at step {ref['step']}")


@pytest.mark.slow
def test_vanilla_and_unbiased_match(run):
    """An unbiased PySAGES run must reproduce plain MD row for row."""
    vanilla, unbiased = run('vanilla_4'), run('unbiased_4')
    assert [r['step'] for r in vanilla] == [r['step'] for r in unbiased]
    _assert_rows_match(vanilla, unbiased, "vanilla vs unbiased")


@pytest.mark.slow
def test_abf_diverges_from_vanilla(run):
    """ABF applies bias forces, so its log must differ from vanilla's.

    test_vanilla_and_unbiased_match shows that without a bias the printed rows
    are identical, so any difference above the print resolution (1e-3 eV, 0.1 K)
    is the bias. Requiring a fixed size would depend on how fast this system
    diverges in 400 steps.
    """
    vanilla, abf = run('vanilla_4'), run('abf_4')
    assert [r['step'] for r in vanilla] == [r['step'] for r in abf]
    differs = [
        abs(v[k] - a[k]) > (0.05 if k == 'T' else 5e-4)  # half a printed digit
        for v, a in zip(vanilla, abf) for k in ('E', 'KE', 'PE', 'H', 'T')
    ]
    assert any(differs), (
        "ABF and vanilla logs are identical to the printed digit; the bias force "
        "should change the trajectory."
    )


@pytest.mark.slow
@pytest.mark.parametrize('continuous,restarted,keys', [
    # Restart of both the SO3LR and the PySAGES state.
    ('unbiased_4', 'restart_unbiased_with_state', tuple(ATOL)),
    # Across coordinate systems the conserved H is not compared, as before.
    # Fractional (vanilla) restart continued as a Cartesian PySAGES run.
    ('unbiased_4', 'restart_vanilla_to_unbiased', ('E', 'KE', 'PE', 'T')),
    # Cartesian (PySAGES) restart continued as a fractional vanilla run. A
    # coordinate mismatch here once gave potential energies of ~45 million eV
    # (fixed in c5e3740).
    ('vanilla_4', 'restart_unbiased_to_vanilla', ('E', 'KE', 'PE', 'T')),
], ids=['pysages_state', 'frac_to_cart', 'cart_to_frac'])
def test_restart_matches_continuous(run, continuous, restarted, keys):
    """The 2 cycles after a 2-cycle restart must match cycles 3-4 of the
    continuous 4-cycle run."""
    continuous_rows, restarted_rows = run(continuous), run(restarted)
    assert len(continuous_rows) >= 4, f"Expected >= 4 rows in {continuous}, got {len(continuous_rows)}"
    assert len(restarted_rows) >= 2, f"Expected >= 2 rows in {restarted}, got {len(restarted_rows)}"
    _assert_rows_match(continuous_rows[-2:], restarted_rows[-2:], f"{restarted} vs {continuous}", keys)

    # The log rows only resolve 1e-3 eV; the last trajectory frame (step 400 of
    # both runs) holds the full-precision state, so a small restart error in
    # positions or momenta cannot hide below the print resolution.
    final = {n: read(run.dirs[n] / 'traj.xyz', index=-1) for n in (continuous, restarted)}
    a, b = final[continuous], final[restarted]
    box = np.diag(a.cell.array)
    displacement = b.positions - a.positions
    displacement -= box * np.round(displacement / box)  # the same atom across a wrap
    np.testing.assert_allclose(displacement, 0.0, atol=1e-6,
                               err_msg=f"{restarted}: final positions differ from {continuous}")
    np.testing.assert_allclose(b.get_momenta(), a.get_momenta(), atol=1e-6,
                               err_msg=f"{restarted}: final momenta differ from {continuous}")
