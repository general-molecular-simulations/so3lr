"""Run the local SO3LR web interface and calculation service."""

import argparse
from collections import OrderedDict, deque
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re
from pathlib import Path
from queue import Full
from secrets import randbelow
from threading import Condition, Thread
from urllib.parse import urlsplit
from uuid import uuid4

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
from jax import config

config.update("jax_enable_x64", True)

import numpy as np
from ase import Atoms, units
from ase.data import atomic_numbers
from ase.optimize import BFGS
with jax.default_device(jax.local_devices(backend="cpu")[0]):
    from so3lr import So3lrCalculator

from web.native_md import run_dynamics
from web.periodic_job import run_periodic, validate_periodic_geometry
from compute_runtime import choose_device, device_name, device_status, precision_context
from peptide_analysis import analyze_trajectory
from trajectory_io import MAX_UPLOAD, MODELS, analyze_file, charge_tolerance


DIST = Path(__file__).resolve().parent / "web"
LIMIT = 256


def real(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("Supply a JSON object with a molecule and calculation settings.")
    elements = payload.get("elements")
    if not isinstance(elements, list) or not 2 <= len(elements) <= LIMIT:
        raise ValueError(f"Use 2–{LIMIT} atoms for a molecular calculation.")
    if any(not isinstance(e, str) or not 1 <= atomic_numbers.get(e, 0) <= 85 for e in elements):
        raise ValueError("Use element symbols from hydrogen (H) through astatine (At).")
    coords = payload.get("coordinates")
    if not isinstance(coords, list) or len(coords) != len(elements) or any(
        not isinstance(row, list) or len(row) != 3 or not all(real(x) for x in row)
        for row in coords
    ):
        raise ValueError("Each atom needs three finite Cartesian coordinates in ångströms.")
    if payload.get("coordinate_unit", "Å") not in ("Å", "angstrom", "angstroms"):
        raise ValueError("Convert coordinates to ångströms before submitting.")
    pbc = payload.get("pbc", False)
    if pbc is None or type(pbc) is bool:
        periodic = bool(pbc)
    elif isinstance(pbc, list) and len(pbc) == 3 and all(type(flag) is bool for flag in pbc) and len(set(pbc)) == 1:
        periodic = pbc[0]
    else:
        raise ValueError("Choose isolated coordinates or a cell periodic along all three axes.")
    if periodic:
        box = validate_periodic_geometry(payload.get("cell"), coords)
        # Bound the work and memory of real-space image lists and PME on the local service.
        lengths = np.linalg.norm(box, axis=1)
        heights = 1 / np.linalg.norm(np.linalg.inv(box), axis=0)
        pair_estimate = len(coords)**2 * (4*np.pi/3)*12**3 / np.linalg.det(box)
        # jaxpme uses powers of two for a grid spanning positive and negative modes.
        mesh = np.exp2(np.ceil(np.log2(2 * np.ceil(lengths / 0.5) + 1)))
        if (min(heights) < 1 or max(lengths) > 200 or pair_estimate > 2_000_000
                or np.prod(mesh) > 8_388_608):
            raise ValueError("The periodic cell exceeds the local calculation budget for neighbors or the PME grid.")
    elif payload.get("cell") is not None and "pbc" not in payload:
        raise ValueError("Specify periodic boundary flags for coordinates with a supplied cell.")
    else:
        points = np.asarray(coords)
        separations = np.linalg.norm(points[:, None] - points, axis=2)
        if np.any(separations[np.triu_indices(len(points), 1)] < 1e-4):
            raise ValueError("Two atoms occupy the same position. Check the coordinates.")
    charge = payload.get("charge", 0)
    if type(charge) is not int or abs(charge) > 10:
        raise ValueError("Total charge must be an integer between −10 and +10, in units of e.")
    task, model = payload.get("task"), payload.get("model")
    if model not in MODELS or task not in ("energy", "relax", "md"):
        raise ValueError("Choose a supported SO3LR model and calculation type.")
    if periodic and (charge != 0 or task == "md"):
        raise ValueError("Periodic calculations support neutral cells and single points or geometry optimization.")
    precision = payload.get("precision", "mixed")
    if precision not in ("mixed", "float32"):
        raise ValueError("Choose mixed or float32 precision.")
    requested_device = payload.get("device", "auto")
    device, _ = choose_device(requested_device, precision)
    steps = payload.get("steps", 100)
    fmax = payload.get("fmax", 0.05)
    if task == "relax":
        if type(steps) is not int or not 1 <= steps <= 300:
            raise ValueError("Use 1–300 relaxation steps.")
        if not real(fmax) or not 0.001 <= fmax <= 1.0:
            raise ValueError("The convergence threshold must be 0.001–1.0 eV/Å.")
    md = {}
    if task == "md":
        temperature = payload.get("temperature", 300)
        timestep = payload.get("timestep", 1.0)
        seed = payload["seed"] if "seed" in payload else randbelow(101)
        if type(steps) is not int or not 1 <= steps <= 1_000_000:
            raise ValueError("Use 1–1,000,000 molecular dynamics steps.")
        if not real(temperature) or not 1 <= temperature <= 2000:
            raise ValueError("Choose a target temperature from 1 to 2000 K.")
        if not real(timestep) or not 0.05 <= timestep <= 2:
            raise ValueError("Choose a time step from 0.05 to 2 fs.")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Use an integer random seed from 0 to 4294967295.")
        md = dict(temperature=temperature, timestep=timestep, seed=seed)
    return dict(elements=elements, coordinates=coords, charge=charge, model=model,
                cell=box.tolist() if periodic else None,
                task=task, steps=steps, fmax=fmax, device=device, md=md, precision=precision)


@lru_cache(maxsize=4)
def calculator(model, atom_count, device, precision):
    # A compiled neighbor list depends on the atom count; parameters live on the selected device.
    return So3lrCalculator(model=model, lr_cutoff=100.0, precision=precision,
                           has_aux=True, observables=["partial_charges", "dipole_vec"])


def solve(elements, coordinates, charge, model, task, steps, fmax, device, md, progress, precision="mixed", cell=None, on_frame=None):
    platform, target = choose_device(device, precision)
    with precision_context(precision), jax.default_device(target):
        if cell is not None:
            result = run_periodic(elements, coordinates, cell, model=model, precision=precision,
                                  task=task, steps=steps, fmax=fmax, progress=progress, on_frame=on_frame)
        elif task == "md":
            result = dynamics_result(elements, coordinates, charge, model, precision, steps, md, progress, on_frame)
        else:
            atoms = Atoms(elements, positions=coordinates, pbc=False)
            atoms.info["charge"] = charge
            atoms.calc = calculator(model, len(atoms), target, precision)
            # ASE does not compare atoms.info when deciding whether a result is reusable.
            atoms.calc.reset()
            result = static_or_relax(atoms, task, steps, fmax, progress, on_frame)
    if cell is None:
        result["analysis"] = analyze_trajectory(elements, result["frames"])
    result.update(model=model, precision=precision, device=platform, device_name=device_name(target))
    return result


def electronic_observables(atoms):
    aux = atoms.calc.results["aux"]
    charges = np.asarray(aux["partial_charges"], dtype=float).reshape(-1)
    if len(charges) != len(atoms) or not np.isfinite(charges).all():
        raise ValueError("The model returned invalid atomic charges.")
    if abs(float(charges.sum()) - atoms.info["charge"]) > charge_tolerance(len(atoms)):
        raise ValueError("The predicted atomic charges do not sum to the input system charge.")
    # The neutral molecular dipole is origin independent; charged dipoles are omitted.
    dipole = None
    if atoms.info["charge"] == 0:
        vector = np.asarray(aux["dipole_vec"], dtype=float).reshape(3) / units.Debye
        if not np.isfinite(vector).all():
            raise ValueError("The model returned a non-finite dipole.")
        dipole = vector.tolist()
    return charges.tolist(), dipole


class StopAtStep(Exception):
    """End an ASE driver from its sampled step callback."""


def dynamics_result(elements, coords, charge, model, precision, steps, md, progress, on_frame):
    rows, frames = [], []
    for row in run_dynamics([atomic_numbers[el] for el in elements], coords, model=model, charge=charge,
                            precision=precision, steps=steps, progress=progress, **md):
        rows.append(row)
        frames.append(row["positions"].tolist())
        if on_frame:
            on_frame(row["step"], frames[-1], None)
    frame_steps = [frame["step"] for frame in rows]
    energies = [frame["energy"] for frame in rows]
    final = rows[-1]
    return dict(energy=final["energy"], forces=final["forces"].tolist(), coordinates=frames[-1],
                frames=frames, frame_steps=frame_steps, energies=energies, converged=None,
                temperatures=[frame["temperature"] for frame in rows],
                kinetic_energies=[frame["kinetic_energy"] for frame in rows],
                times_fs=[step * md["timestep"] for step in frame_steps],
                partial_charges=[frame["partial_charges"].tolist() for frame in rows],
                dipoles_debye=None if charge else [frame["dipole_debye"].tolist() for frame in rows],
                md={**md, "steps": final["step"], "requested_steps": steps, "friction_per_fs": 0.01,
                    "engine": "jax", "integrator": "JAX BAOAB Langevin", "center_of_mass_fixed": True,
                    "force_backend": final["force_backend"]},
                units=dict(coordinates="Å", energy="eV", forces="eV/Å", charge="e", dipole="D",
                           time="fs", temperature="K", kinetic_energy="eV"))


def static_or_relax(atoms, task, steps, fmax, progress, on_frame):
    frames, energies, frame_steps, atomic_charges, dipoles = [], [], [], [], []
    converged = None

    def sample(step):
        energy = float(atoms.get_potential_energy())
        if not math.isfinite(energy) or not np.isfinite(atoms.positions).all():
            raise ValueError("The model returned a non-finite result. Check the input geometry.")
        charges, dipole = electronic_observables(atoms)
        atomic_charges.append(charges)
        if dipole is not None:
            dipoles.append(dipole)
        energies.append(energy)
        frames.append(atoms.get_positions().tolist())
        frame_steps.append(step)

    if task == "energy":
        sample(0)
    else:
        driver = BFGS(atoms, logfile=None)

        def on_step():
            sample(driver.nsteps)
            if on_frame:
                on_frame(driver.nsteps, frames[-1], None)
            if progress(driver.nsteps, steps):
                raise StopAtStep

        driver.attach(on_step, interval=1)
        try:
            converged = bool(driver.run(fmax=fmax, steps=steps))
        except StopAtStep:
            pass
    forces = atoms.get_forces(apply_constraint=False)
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError("The model could not produce finite Cartesian forces.")
    result = dict(energy=energies[-1], forces=forces.tolist(), coordinates=frames[-1],
                  frames=frames, energies=energies, frame_steps=frame_steps, converged=converged,
                  partial_charges=atomic_charges, dipoles_debye=dipoles or None,
                  units={"coordinates": "Å", "energy": "eV", "forces": "eV/Å", "charge":"e", "dipole":"D"})
    return result


class JobQueue:
    def __init__(self):
        self.waiting = deque()
        self.records = OrderedDict()
        self.changed = Condition()
        Thread(target=self.work, daemon=True, name="so3lr-calculator").start()

    def submit(self, settings):
        job = uuid4().hex
        with self.changed:
            if len(self.waiting) >= 8:
                raise Full
            if len(self.records) >= 24:
                completed = next(key for key, record in self.records.items()
                                 if record["status"] in ("done", "error", "cancelled"))
                del self.records[completed]
            self.records[job] = {"status": "queued"}
            self.waiting.append((job, settings))
            self.changed.notify()
        return job

    def get(self, job):
        with self.changed:
            return self.records.get(job)

    def cancel(self, key):
        with self.changed:
            record = self.records.get(key)
            if record is None:
                return None
            if record["status"] == "queued":
                self.waiting = deque(entry for entry in self.waiting if entry[0] != key)
                self.records[key] = {"status": "cancelled"}
            elif record["status"] == "running":
                self.records[key] = {**record, "status": "cancelling"}
            return self.records[key]

    def publish_frame(self, key, step, positions, cell):
        with self.changed:
            self.records[key] = {**self.records[key], "preview": {
                "step": step, "positions": positions, "cell": cell}}

    def progress(self, key, step, total):
        with self.changed:
            status = self.records[key]["status"]
            self.records[key] = {**self.records[key], "progress": {"step": step, "total": total}}
            return status == "cancelling"

    def work(self):
        while True:
            with self.changed:
                self.changed.wait_for(lambda: self.waiting)
                key, settings = self.waiting.popleft()
                self.records[key] = {"status": "running"}
            try:
                result = solve(**settings, progress=partial(self.progress, key), on_frame=partial(self.publish_frame, key))
                record = {"status": "done", "result": result}
            except Exception as error:
                record = {"status": "error", "error": str(error) or type(error).__name__}
            with self.changed:
                if self.records[key]["status"] == "cancelling" and "result" in record:
                    record["status"] = "cancelled"
                self.records[key] = record


class Handler(SimpleHTTPRequestHandler):
    def parse_request(self):
        if not super().parse_request():
            return False
        # A browser must address the service through a loopback hostname.
        # SSH port forwarding may expose a different local port.
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or not re.fullmatch(
            r"(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?", hosts[0], re.IGNORECASE
        ):
            self.send_error(403, "Open this service through localhost.")
            return False
        return True

    def reply(self, status, payload):
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/status":
            self.reply(200, {"engine": "so3lr", "models": MODELS, "max_atoms": LIMIT, **device_status()})
        elif path.startswith("/api/jobs/"):
            record = self.server.jobs.get(path.rsplit("/", 1)[-1])
            self.reply(200 if record else 404, record or {"error": "Job no longer available."})
        elif path.startswith("/api/"):
            self.reply(404, {"error": "Unknown API endpoint."})
        else:
            super().do_GET()

    def do_POST(self):
        path = urlsplit(self.path).path
        parts = path.strip("/").split("/")
        cancel = len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel"
        if not cancel and path not in ("/api/calculate", "/api/smiles", "/api/analyze"):
            return self.reply(404, {"error": "Unknown API endpoint."})
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            return self.reply(403, {"error": "Submit calculations from this local workbench."})
        if self.headers.get_content_type() != "application/json" and not (cancel and self.headers.get("Content-Length", "0") == "0"):
            return self.reply(415, {"error": "Send the request as application/json."})
        if cancel:
            record = self.server.jobs.cancel(parts[2])
            return self.reply(200 if record else 404, record or {"error": "Job no longer available."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            maximum = MAX_UPLOAD if path == "/api/analyze" else 1_000_000
            if not 0 < length <= maximum:
                return self.reply(413, {"error": "The upload exceeds the limit for this request."})
            body = json.loads(self.rfile.read(length))
            if path == "/api/analyze":
                return self.reply(200, analyze_file(body))
            if path == "/api/smiles":
                from molecule_input import from_smiles
                if not isinstance(body, dict):
                    raise ValueError("Supply a SMILES string.")
                return self.reply(200, from_smiles(body.get("smiles")))
            settings = validate(body)
            job = self.server.jobs.submit(settings)
        except (ValueError, OverflowError, RecursionError, TypeError) as error:
            return self.reply(400, {"error": str(error)})
        except Full:
            return self.reply(503, {"error": "The calculation queue is full. Try again shortly."})
        self.reply(202, {"id": job, "status": "queued"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=DIST)) as server:
        server.jobs = JobQueue()
        print(f"SO3LR is ready at http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
