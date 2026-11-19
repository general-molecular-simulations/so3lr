"""Full three-dimensional periodic SO3LR single points and variable-cell relaxation."""
import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.geometry import find_mic
from ase.optimize import LBFGS
from scipy.spatial import cKDTree


def validate_periodic_geometry(cell, positions):
    """Accept a finite, right-handed unit cell and distinct periodic sites, in Å."""
    box = np.asarray(cell, dtype=float)
    if box.shape != (3, 3) or not np.isfinite(box).all():
        raise ValueError("Supply three finite lattice vectors in ångströms.")
    volume = float(np.linalg.det(box))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("The periodic cell must be three-dimensional and right-handed.")
    # First find candidates in a wrapped fractional cell, then measure in Å.
    # This also keeps standalone jobs on complete solvent boxes linear in memory.
    scaled = np.linalg.solve(box.T, np.asarray(positions, dtype=float).T).T % 1.
    scaled[scaled >= 1.] = 0.
    tolerance = 1e-4
    fractional_radius = tolerance / np.linalg.svd(box, compute_uv=False)[-1]
    pairs = cKDTree(scaled, boxsize=1.).query_pairs(fractional_radius, output_type="ndarray")
    if len(pairs):
        vectors = np.asarray(positions)[pairs[:, 0]] - np.asarray(positions)[pairs[:, 1]]
        _, distances = find_mic(vectors, box, pbc=True)
        if np.any(distances < tolerance):
            raise ValueError("Two sites coincide across periodic boundaries. Check the coordinates and cell.")
    return box


def run_periodic(elements, coordinates, cell, *, model, precision, task, steps=100, fmax=0.05, progress=None, on_frame=None):
    """Run on the caller's active JAX device and precision context."""
    from so3lr import So3lrCalculator

    if task not in ("energy", "relax"):
        raise ValueError("Choose a periodic single point or geometry optimization.")
    box = validate_periodic_geometry(cell, coordinates)
    atoms = Atoms(elements, positions=coordinates, cell=box, pbc=True)
    atoms.info["charge"] = 0
    relax = task == "relax"
    atoms.calc = So3lrCalculator(model=model, precision=precision, lr_cutoff=12.,
                                 calculate_stress=relax, neighbor_backend="exact",
                                 kspace_electrostatics="pme", kspace_smearing=4.,
                                 kspace_spacing=0.5)
    frames, cells, energies, indices = [], [], [], []
    converged = None

    def sample(step):
        energy = float(atoms.get_potential_energy())
        if not (np.isfinite(energy) and np.isfinite(atoms.positions).all() and np.isfinite(atoms.cell.array).all()):
            raise ValueError(f"The periodic calculation returned nonfinite values at step {step}.")
        frames.append(atoms.positions.tolist())
        cells.append(atoms.cell.array.tolist())
        energies.append(energy)
        indices.append(step)

    if relax:
        optimizer = LBFGS(FrechetCellFilter(atoms), logfile=None)
        for converged in optimizer.irun(fmax=fmax, steps=steps):
            sample(optimizer.nsteps)
            if on_frame:
                on_frame(optimizer.nsteps, frames[-1], cells[-1])
            if progress and progress(optimizer.nsteps, steps):
                converged = None
                break
        if converged is not None:
            converged = bool(converged)
    else:
        sample(0)
    forces = atoms.get_forces()
    if not np.isfinite(forces).all():
        raise ValueError("The periodic force field returned nonfinite forces.")
    return dict(task=task, energy=energies[-1], energies=energies, forces=forces.tolist(),
                coordinates=frames[-1], frames=frames, cells=cells, cell=cells[-1],
                pbc=[True]*3, frame_steps=indices, converged=converged,
                analysis={"cell_volumes": np.linalg.det(cells).tolist(), "peptide":None, "rmsd":None},
                units={"energy":"eV", "forces":"eV/Å", "coordinates":"Å", "cell":"Å", "cell_volume":"Å³"},
                periodic={"neighbor_backend":"exact", "electrostatics":"PME", "lr_cutoff_A":12.,
                          "smearing_A":4., "mesh_spacing_A":0.5, "variable_cell":relax,
                          "optimizer":"LBFGS / FrechetCellFilter" if relax else None})


def write_periodic_result(path, elements, result, model, precision):
    """Preserve a separate cell and energy for every saved optimization step."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    optimizer = result["periodic"]["optimizer"]
    task = result.get("task", "relax" if optimizer else "energy")
    with open(path, "w", encoding="utf-8") as output:
        for index, (xyz, cell, energy, step) in enumerate(zip(
                result["frames"], result["cells"], result["energies"], result["frame_steps"])):
            atoms = Atoms(elements, positions=xyz, cell=cell, pbc=True)
            atoms.info.update(task=task, step=step, model=model, precision=precision)
            if optimizer:
                atoms.info["optimizer"] = optimizer
            values = {"energy":energy}
            if index == len(result["frames"]) - 1:
                values["forces"] = np.asarray(result["forces"])
            atoms.calc = SinglePointCalculator(atoms, **values)
            write(output, atoms, format="extxyz")
