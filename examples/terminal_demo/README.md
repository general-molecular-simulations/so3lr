# Terminal demo

Run from the project root after `uv sync --locked`:

```sh
bash examples/terminal_demo/run.sh
```

The script uses the project's `.venv` when present, otherwise `so3lr` on your
PATH. It creates a fresh, ignored directory under `examples/terminal_demo/output`
for every invocation, copies the input and settings there, and runs
`so3lr --settings settings.yaml`. Additional CLI options are passed through.

The sample is a nine-atom ethanol molecule generated with
`ase.build.molecule('CH3CH2OH')`. It runs 10,000 NVT steps (5 ps) with the canonical
`so3lr-tauri-m-2606` model, mixed precision, and the standard engine, so it works
on a CPU. The 100 Å cutoff includes the full isolated molecule. Initial
relaxation is disabled to keep this interface demonstration short. This is not
an equilibrated production simulation. The first cycle includes JAX compilation
and can take substantially longer than subsequent cycles.

An interactive MD terminal starts with temperature and **total energy** graphs;
NPT adds volume. The plots share aligned time axes and grow at accepted cycle
boundaries. Temperature shows the thermostat target for NVT, NPT and Langevin;
NVE temperature and the energy/volume plots show a measured reference from the
latter half of the recorded time span. Speed includes both
`ns/day` and `s/atom/step` (seconds per step divided by the total atom count).
Mixed precision is labeled `mixed (float32 model; float64 dynamics)`.
Reference lines are solid and muted grey, behind the trajectory. Their exact
values appear in the captions; line positions use terminal-character rows.
Use `e` to change energy and up/down for plot pages in a short terminal.
The Tauri pulse and elapsed-time clock refresh twice per second, including during
initialization and compilation. The star expands and recedes in place over five
seconds, with a simple ASCII fallback. It redraws a frozen text snapshot and
wall time; it does not query numerical arrays or advance the graphs. Controls are still handled at
reporting boundaries. During MD, `q` toggles plots
and compact status; press it again to restore the graphs. The calculation
continues in the same terminal; Ctrl+C interrupts MD.

Each run produces:

- `ethanol_nvt.xyz`: the molecular trajectory.
- `ethanol_nvt.log`: the complete text log.
- `ethanol_restart.npz`: restart state.
- `ethanol_nvt.<run-id>.events.jsonl`: offline metrics and lifecycle events.
- `ethanol_nvt.<run-id>.settings.resolved.yaml`: settings in replayable YAML,
  with input units and resolved model/backend information.
- Copies of `ethanol.xyz` and `settings.yaml` for reproducibility.

The text log includes a run identity block and a final summary of the outcome,
recorded steps/time, warnings, checkpoint and saved files. Optimization summaries
distinguish meeting a force criterion from finishing a step budget. Run IDs link
each invocation's log summary, journal and resolved YAML. Model/backend updates
use `Model: ...; backend: ...`; full run IDs appear in the `BEGIN RUN`,
`RUN SUMMARY` and `END RUN` markers, journals and artifact paths.

To replay a resolved YAML, copy it and choose fresh output/log/checkpoint names,
then run `so3lr --settings repeat.yaml`. Optimization YAML carries
`operation: optimization` and uses the same command. Overwrite protection remains
enabled in saved settings.
Input and restart files are referenced, not archived; keep their original
contents when exact replay is required.

To rerun a saved setup manually, change into its output directory, give
`output_file`, `log_file`, and `restart_save_path` new names in `settings.yaml`,
then invoke `so3lr --settings settings.yaml`. Existing trajectories are protected
by `overwrite: false`.

## Watch an unattended run

Keep the simulation in its shell script. In another terminal, point the viewer
at the output directory printed by the script:

```sh
.venv/bin/so3lr show examples/terminal_demo/output
```

The viewer discovers up to three levels below the supplied directory. Temperature
and energy plots are stacked vertically; NPT adds volume in Å³. Use `j`/`k` or
left/right to choose a run, `e` to switch the energy plot, and `q` to detach.
Multiple runs open in an overview table; left/right cycles through the runs
and back to that table, and `o` returns directly to the overview. Mixed folders
label MD and OPT separately: MD progress counts the current segment, while
optimization progress counts FIRE steps. Up/down pages the overview or the
individual plots in a short terminal. Closing the
viewer does not stop the simulation. Completed runs
can be viewed in exactly the same way. `--once` prints one snapshot; redirected
output also uses snapshot mode. No W&B, server, account, or network is involved.
From inside an existing run directory, `so3lr show .` opens its journal.
Older runs with only a text `.log` and no `*.events.jsonl` are not supported.

The graphs show temperature and energy against **time within this invocation**,
including after a restart. New data arrives at accepted MD cycle boundaries;
the first cycle can spend time compiling. A quiet journal does not prove that
a job has failed. The full metrics remain in the journal even when the viewer
reduces long histories for display.
Time axes show intermediate ticks and line up across the stacked plots. Measured
references are labeled separately from thermostat targets; long-run aggregation
is marked approximate when its midpoint crosses a time bin.

`so3lr opt` also uses the live stage display and writes a metrics journal.
`so3lr show` displays its latest recorded step, potential energy and maximum
force, with an explicit distinction between meeting a force criterion and
finishing a step budget. It does not present optimization energy as MD total
energy or invent a temperature trace.

## Prepare a new run

```sh
.venv/bin/so3lr init
```

Use the arrow-key structure picker to browse to
`examples/terminal_demo/ethanol.xyz`, or choose Ala4 or DHA from
`tests/test_data`. Presets are found relative to the source checkout, independent
of the current directory. Select a frame for trajectories such as Ala4. Stored
velocities and DHA's padded periodic box have explicit keep/change choices;
the original files are never edited. Water64 remains available through manual
file selection, but its small periodic cell needs adjusted real-space/PME
settings and is no longer a featured preset.
All periodic CLI MD modes require a positive, axis-aligned orthorhombic cell;
NPT additionally requires periodic boundaries in all three axes. Setup checks
the cell before saving a run.

Setup is organized as **1/3 Structure**, **2/3 Dynamics**, and **3/3 Review and
start**. Select ensemble, model, timestep, duration and output directory.
The model menu shows M (default), S (faster), and L (recommended for reactions);
open Legacy models to choose one of the four older releases.
After review, **Save setup and start MD now** is the first and default choice:
Enter saves the structure, `settings.yaml` and `run.sh`, then immediately runs
MD in this terminal. Choose **Save setup for later** to create only those files.
Back and Cancel remain available; no files are written before this choice.
Run the saved script in an environment
where `so3lr` is on PATH. A bare `so3lr` opens setup only in an interactive
terminal outside a scheduler job. Simulation commands and `--settings` never
open guided setup.
