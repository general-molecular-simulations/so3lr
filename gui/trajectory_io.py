"""Read consistently ordered molecular and fully periodic extended XYZ trajectories."""

from io import StringIO
from math import isfinite

import numpy as np
from ase.io import iread
from so3lr.model_registry import available_models, model_aliases

from peptide_analysis import analyze_trajectory

MAX_UPLOAD = 56 * 1024 * 1024
MAX_FRAMES = 5000
MAX_PERIODIC_ATOMS = 50_000
MAX_ATOM_FRAMES = 500_000
MODELS = tuple(name for name in model_aliases() if name.startswith("so3lr-tauri-"))
TRAJECTORY_MODELS = {name for name in available_models() if name.startswith("so3lr-tauri-")}


def charge_tolerance(atom_count):
    # Both precisions use float32 model layers. Summation and charge
    # redistribution can accumulate rounding error across a large molecule.
    return max(1e-4, 128 * atom_count * np.finfo(np.float32).eps)


def check_size(text):
    stream = StringIO(text)
    count, total = 0, 0
    for header in stream:
        if not header.strip():
            continue
        try:
            natoms = int(header)
        except ValueError as error:
            raise ValueError("Start each XYZ frame with its atom count.") from error
        if not 1 <= natoms <= MAX_PERIODIC_ATOMS:
            raise ValueError("Each trajectory frame must contain 1–50,000 atoms.")
        count += 1
        total += natoms
        if count > MAX_FRAMES or total > MAX_ATOM_FRAMES:
            raise ValueError("Use at most 5,000 frames and 500,000 atom coordinates per trajectory.")
        if not stream.readline():
            raise ValueError(f"Frame {count} is missing its XYZ comment line.")
        for _ in range(natoms):
            if not stream.readline().strip():
                raise ValueError(f"Frame {count} is missing atom rows.")
    if count < 2:
        raise ValueError("Choose an XYZ trajectory containing at least two frames.")
    return count


def finite_scalar(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"Use a finite numeric value for {label}.")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"Use a finite numeric value for {label}.")
    return value


def optional_column(values, label):
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"The {label} field must be present in every saved frame or omitted throughout.")
    return values


def vector(value, shape, label):
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"The trajectory contains invalid {label}.")
    return arr


def analyze_file(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("xyz"), str):
        raise ValueError("Supply a multi-frame XYZ or extended XYZ trajectory.")
    text = payload["xyz"].strip()
    expected = check_size(text)
    elements, charge, formula = None, None, None
    frames, times, steps = [], [], []
    energies, temperatures, charges, dipoles, models, precisions = [], [], [], [], [], []
    kinetic_energies = []
    periodic = None
    cells, volumes = [], []
    tasks = []
    metadata = {}
    integrators = set()
    force_backends = set()
    temperature_targets = set()
    forces = None
    try:
        for index, atoms in enumerate(iread(StringIO(text), index=":", format="extxyz")):
            info = atoms.info
            force_backends.add(str(info.get("force_backend", "")))
            integrators.add((str(info.get("md_engine", "")), str(info.get("integrator", ""))))
            if atoms.pbc.any() and not atoms.pbc.all():
                raise ValueError("Use either three periodic axes or an isolated molecular trajectory.")
            is_periodic = bool(atoms.pbc.all())
            if periodic is None:
                periodic = is_periodic
            if periodic != is_periodic:
                raise ValueError("Use the same periodic boundary conditions in every frame.")
            if not periodic and not 2 <= len(atoms) <= 256:
                raise ValueError("Each isolated trajectory frame must contain 2–256 atoms.")
            if periodic:
                lattice = vector(atoms.cell.array, (3, 3), "periodic lattice vectors")
                volume = float(abs(np.linalg.det(lattice)))
                if not np.isfinite(volume) or volume <= 0 or np.linalg.matrix_rank(lattice) != 3:
                    raise ValueError("Each periodic frame requires three independent lattice vectors.")
                cells.append(lattice.tolist())
                volumes.append(volume)

            kind = str(info.get("task", "")).lower()
            tasks.append("relax" if kind in ("relax", "optimization") else kind)
            for field in ("ensemble", "thermostat", "optimizer", "seed", "timestep_fs",
                          "friction_per_fs", "temperature_dof", "chain_length"):
                value = info.get(field)
                if value is not None:
                    if field in ("ensemble", "thermostat", "optimizer"):
                        value = str(value)
                    else:
                        value = finite_scalar(value, field)
                        if field in ("seed", "temperature_dof", "chain_length"):
                            if not value.is_integer() or value < (0 if field == "seed" else 1):
                                raise ValueError(f"Invalid {field} in trajectory metadata.")
                            value = int(value)
                        elif value < 0 or (field == "timestep_fs" and value == 0):
                            raise ValueError(f"Invalid {field} in trajectory metadata.")
                metadata.setdefault(field, set()).add(value)
            length_unit = str(info.get("length_unit", info.get("coordinate_unit", info.get("units", "angstrom")))).lower()
            if length_unit not in ("angstrom", "angstroms", "å", "ångström", "ångströms"):
                raise ValueError("Trajectory coordinates must use ångströms.")
            symbols = atoms.get_chemical_symbols()
            qtotal = finite_scalar(info.get("charge", info.get("total_charge", 0)), "system charge")
            if not qtotal.is_integer() or abs(qtotal) > 10:
                raise ValueError("Total charge must be an integer between −10 and +10.")
            if index == 0:
                elements, charge, formula = symbols, int(qtotal), atoms.get_chemical_formula()
                if np.any((atoms.numbers < 1) | (atoms.numbers > 85)):
                    raise ValueError("Use elements from hydrogen (H) through astatine (At).")
            if elements != symbols or qtotal != charge:
                raise ValueError("Keep the same atoms, element order and total charge in every frame.")
            frames.append(vector(atoms.positions, (len(elements), 3), "Cartesian coordinates").tolist())
            times.append(finite_scalar(info["time_fs"], "time_fs") if "time_fs" in info else None)
            step = finite_scalar(info["step"], "step") if "step" in info else None
            if step is not None and (step < 0 or not step.is_integer()):
                raise ValueError("Trajectory step numbers must be nonnegative integers.")
            steps.append(int(step) if step is not None else None)

            results = atoms.calc.results if atoms.calc is not None else {}
            energy = results.get("energy")
            if energy is not None and str(info.get("energy_unit", "eV")).lower() != "ev":
                raise ValueError("Trajectory potential energies must use eV.")
            energies.append(finite_scalar(energy, "potential energy") if energy is not None else None)
            temperature = info.get("temperature_K")
            temperature = finite_scalar(temperature, "temperature_K") if temperature is not None else None
            if temperature is not None and temperature < 0:
                raise ValueError("Kinetic temperatures must be nonnegative.")
            temperatures.append(temperature)
            target = info.get("target_temperature_K")
            temperature_targets.add(finite_scalar(target, "target_temperature_K") if target is not None else None)
            kinetic = info.get("kinetic_energy_eV")
            if kinetic is None and atoms.has("momenta"):
                kinetic = atoms.get_kinetic_energy()
            kinetic = finite_scalar(kinetic, "kinetic_energy_eV") if kinetic is not None else None
            if kinetic is not None and kinetic < 0:
                raise ValueError("Kinetic energies must be nonnegative.")
            kinetic_energies.append(kinetic)
            forces = results.get("forces")
            if forces is not None:
                if str(info.get("force_unit", "eV/angstrom")).lower() not in ("ev/angstrom", "ev/å"):
                    raise ValueError("Trajectory forces must use eV/Å.")
                forces = vector(forces, (len(elements), 3), "Cartesian forces").tolist()

            declared_model = str(info.get("model", ""))
            trusted = declared_model in TRAJECTORY_MODELS and str(info.get("charge_source", "")).lower() == "so3lr"
            models.append(declared_model if declared_model in TRAJECTORY_MODELS else None)
            precision = str(info.get("precision", ""))
            precisions.append(precision if precision in ("mixed", "float32") else None)
            q = atoms.arrays.get("partial_charges") if trusted else None
            if q is not None:
                q = vector(q, (len(elements),), "atomic charges")
                if abs(float(q.sum()) - charge) > charge_tolerance(len(q)):
                    raise ValueError("The saved predicted charges do not sum to the system charge.")
            charges.append(None if q is None else q.tolist())
            dipole = info.get("dipole_debye") if q is not None and charge == 0 and not periodic else None
            dipoles.append(vector(dipole, (3,), "dipole vector").tolist() if dipole is not None else None)
    except (IndexError, KeyError, AssertionError, RuntimeError, StopIteration) as error:
        raise ValueError(f"Could not read the XYZ trajectory near frame {len(frames) + 1}.") from error
    if len(frames) != expected:
        raise ValueError("Could not read every XYZ frame. Check the file for blank or incomplete frames.")

    if len(set(tasks)) != 1:
        raise ValueError("Keep the same trajectory task in every frame.")
    task = tasks[0]
    steps = optional_column(steps, "step")
    if steps is not None and np.any(np.diff(steps) <= 0):
        raise ValueError("Frame steps must be strictly increasing.")
    times = optional_column(times, "time_fs")
    if task == "relax":
        times = None
    elif times is None:
        if payload.get("spacing_fs") is not None:
            spacing = finite_scalar(payload["spacing_fs"], "saved-frame spacing")
            if spacing <= 0:
                raise ValueError("The time between saved frames must be positive.")
            times = [i * spacing for i in range(len(frames))]
        elif not periodic:
            raise ValueError("Enter the time between saved frames in fs for a trajectory without time_fs.")
    if times is not None and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Frame times must be finite and strictly increasing.")
    energies = optional_column(energies, "energy")
    charges = optional_column(charges, "SO3LR partial charges")
    model = models[0] if len(set(models)) == 1 else None
    if charges is not None and model is None:
        raise ValueError("Keep one SO3LR model throughout a trajectory with predicted charges.")
    result = dict(frames=frames, coordinates=frames[-1], times_fs=times,
                  frame_steps=steps, energies=energies,
                  energy=energies[-1] if energies else None, forces=forces, converged=None,
                  temperatures=optional_column(temperatures, "temperature_K"),
                  kinetic_energies=optional_column(kinetic_energies, "kinetic_energy_eV"), partial_charges=charges,
                  dipoles_debye=optional_column(dipoles, "dipole_debye"), model=model,
                  precision=precisions[0] if len(set(precisions)) == 1 else None,
                  analysis=({"cell_volumes": volumes, "peptide": None, "rmsd": None} if periodic else
                            analyze_trajectory(elements, frames)),
                  units=dict(coordinates="Å", energy="eV", forces="eV/Å", time="fs",
                             temperature="K", kinetic_energy="eV", charge="e", dipole="D"))
    if task in ("md", "relax"):
        result["task"] = task
    if periodic:
        result.update(pbc=[True, True, True], cells=cells)
        result["units"]["cell_volume"] = "Å³"
    if len(integrators) == 1:
        engine, integrator = integrators.pop()
        if (engine, integrator) in {("ase", "ASE Langevin"), ("jax", "JAX BAOAB Langevin"),
                                    ("jax", "JAX-MD Nosé–Hoover chain")}:
            result["md"] = dict(engine=engine, integrator=integrator)
    if len(force_backends) == 1:
        backend = force_backends.pop()
        if backend in {"Custom Metal long-range", "SO3LR directed sparse (Metal)", "SO3LR JAX-MD"}:
            result.setdefault("md", {})["force_backend"] = backend
    if len(temperature_targets) == 1:
        target = temperature_targets.pop()
        if target is not None and target >= 0:
            result.setdefault("md", {})["temperature"] = target
    for field, values in metadata.items():
        if len(values) == 1 and None not in values:
            value = values.pop()
            if field == "optimizer":
                if task == "relax":
                    result["optimizer"] = value
            elif task != "relax":
                result.setdefault("md", {})["timestep" if field == "timestep_fs" else field] = value
    name = str(payload.get("name") or "Imported trajectory")[:150]
    structure = dict(id="trajectory", name=name, formula=formula, charge=charge, elements=elements,
                     coordinates=frames[0], source="Imported molecular trajectory in ångströms.")
    if periodic:
        structure.update(pbc=[True, True, True], cell=cells[0], source="Imported periodic structure in ångströms.")
    return {"structure": structure, "result": result}
