"""Build the 1 ns Ala15 replay from the first production segment of paper replica 1.

Run from the workbench with its Python environment:
  python benchmarks/prepare_ala15_recording.py /path/to/raw_trajectories/trajectories/ala15f

The archive README and paper Fig. S16 provenance map the internal run ID r2 to
replica 1. The original production run wrote one frame every 1,000 half-fs steps.
"""
import argparse
from collections import deque
from itertools import islice
import json
from pathlib import Path
import sys

import numpy as np
from ase import units
from ase.data import atomic_masses, atomic_numbers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peptide_analysis import analyze_trajectory


def build_recording(directory):
    reference = (ROOT / "web/structures/ala15_folded.xyz").read_text().splitlines()[2:164]
    elements = [line.split()[0] for line in reference]
    initial_positions = np.array([line.split()[1:4] for line in reference], dtype=float)
    masses = np.array([atomic_masses[atomic_numbers[symbol]] for symbol in elements])
    frames, steps, energies = [], [], []
    sources = [("ala15f_10ns_r2.xyz", 11, 2_000)]
    cycles = 0

    for filename, relaxation, production in sources:
        count = 0
        with (directory / filename).open("rb") as stream:
            for size in stream:
                if count == relaxation + production:
                    break
                if int(size) != len(elements):
                    raise ValueError(f"Unexpected atom count in {filename} at frame {count}")
                header = next(stream)
                rows = islice(stream, len(elements))
                is_production = count >= relaxation
                if (b"momenta:R:3" in header) != is_production or b'pbc="F F F"' not in header:
                    raise ValueError(f"Unexpected properties in {filename} at frame {count}")
                cycles += int(is_production)
                # Frame one has a real thermal state; frame zero is a relaxation
                # geometry with no momenta. Sample the excerpt every 2 ps.
                save = is_production and (cycles == 1 or cycles % 4 == 0)
                if save or count == cycles == 0:
                    rows = [line.split() for line in rows]
                    if [r[0].decode() for r in rows] != elements:
                        raise ValueError(f"Atom order changed in {filename} at frame {count}")
                    xyz = np.array([r[1:4] for r in rows], dtype=float)
                    if save:
                        p = np.array([r[4:7] for r in rows], dtype=float)
                        energies.append(float(np.sum(p * p / masses[:, None]) / 2))
                        steps.append(cycles * 1000)
                        frames.append(xyz)
                    else:
                        np.testing.assert_allclose(xyz, initial_positions, rtol=0, atol=1e-5)
                else:
                    deque(rows, maxlen=0)
                count += 1
        if count != relaxation + production:
            raise ValueError(f"Unexpected length in {filename}: found {count} frames")

    raw = np.stack(frames)
    display = raw.round(5)
    analysis = analyze_trajectory(elements, display)
    raw_fractions = analyze_trajectory(elements, raw)["peptide"]["fractions"]
    assert raw_fractions == analysis["peptide"]["fractions"]
    assert analysis["peptide"]["residue_count"] == 15 and len(display) == 501
    assert steps[0] == 1000 and steps[-1] == 2_000_000
    # Reproduce this campaign's jax_md.quantity.temperature(momentum=p, mass=m)
    # convention: 3N Cartesian degrees of freedom and ase.units.kB in eV/K.
    temperatures = (2 * np.array(energies) / (3 * len(elements) * units.kB)).round(6)
    md = {"ensemble": "NVT", "thermostat": "Nosé–Hoover chain", "engine": "jax",
          "integrator": "JAX-MD Nosé–Hoover chain", "steps": steps[-1], "requested_steps": steps[-1],
          "timestep": 0.5, "temperature": 300, "seed": 43, "chain_length": 3,
          "temperature_dof": 3 * len(elements), "lr_cutoff_A": 1000}
    recording = {"title": "Ala15 · 1 ns", "figure": "Tauri, Fig. S16",
                 "replica": 1, "run_id": "r2", "checkpoint": "so3lr-tauri-m-2606",
                 "production_duration_fs": 1_000_000, "source_interval_fs": 500,
                 "segments": [entry[0] for entry in sources],
                 "sampling": "The first thermal frame is at 0.5 ps, followed by 2 ps, 4 ps and every 2 ps through 1 ns.",
                 "source": "Gas-phase paper replica 1 (r2), the first 1 ns of the initial 10 ns segment. The first 11 relaxation geometries are excluded.",
                 "observables": "Kinetic energy and temperature use the archived momenta. Whole-peptide DSSP and heavy-atom RMSD use the saved positions. The trajectory files contain no potential energies."}
    result = {"frames": display.tolist(), "coordinates": display[-1].tolist(), "frame_steps": steps,
              "times_fs": [s // 2 for s in steps], "kinetic_energies": np.round(energies, 8).tolist(),
              "temperatures": temperatures.tolist(), "model": "so3lr-tauri-m", "precision": "float32",
              "device": "gpu", "device_name": "Cluster GPU", "analysis": analysis, "md": md,
              "units": {"coordinates": "Å", "time": "fs", "kinetic_energy": "eV", "temperature": "K"},
              "recording": recording}
    return {"id": "ala15_folded", "task": "md", "model": result["model"], "charge": 0,
            "steps": steps[-1], "timestep": md["timestep"], "seed": md["seed"],
            "temperature": md["temperature"], "device": result["device"],
            "precision": result["precision"], "result": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    data = build_recording(args.source)
    destination = ROOT / "web/ala15-paper.json"
    destination.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data['result']['frames'])} frames to {destination.name}")
