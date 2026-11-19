// Periodic-table order gives the elemental coverage of the public SO3LR Tauri models.
const supported = new Set(`
H He
Li Be B C N O F Ne
Na Mg Al Si P S Cl Ar
K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At
`.trim().split(/\s+/));

function validate({ elements, coordinates, charge }) {
  if (!Array.isArray(elements) || elements.length < 2 || elements.length > 256) {
    throw new Error('An isolated structure must contain 2 to 256 atoms.');
  }
  if (!Array.isArray(coordinates) || coordinates.length !== elements.length) {
    throw new Error('Supply one coordinate triplet for every atom.');
  }
  if (!Number.isInteger(charge) || Math.abs(charge) > 10) {
    throw new Error('Choose an integer charge from −10 to +10.');
  }
  const composition = new Map();
  elements.forEach((atom, i) => {
    if (!supported.has(atom)) throw new Error(`Atom ${i + 1} must use an element symbol from H through At.`);
    const point = coordinates[i];
    if (!Array.isArray(point) || point.length !== 3 || !point.every(Number.isFinite)) {
      throw new Error(`Atom ${i + 1} needs three finite coordinates in ångströms.`);
    }
    for (let k = 0; k < i; ++k) {
      const prior = coordinates[k];
      const distance = Math.hypot(prior[0] - point[0], prior[1] - point[1], prior[2] - point[2]);
      if (distance < 1e-4) throw new Error(`Atoms ${k + 1} and ${i + 1} overlap. Check their coordinates.`);
    }
    composition.set(atom, (composition.get(atom) ?? 0) + 1);
  });
  const alphabetic = [...composition.keys()].sort();
  const hill = composition.has('C')
    ? ['C', ...alphabetic.filter(x => x === 'H'), ...alphabetic.filter(x => x !== 'H' && x !== 'C')]
    : alphabetic;
  return hill.map(element => element + (composition.get(element) > 1 ? composition.get(element) : '')).join('');
}

// Accept the single-frame XYZ format: atom count, one comment, then element/x/y/z.
// Structured extxyz headers must be converted explicitly to keep their units, cell
// information or per-atom data from silently acquiring a different meaning.
export function parseXYZ(text, name = 'Imported molecule') {
  if (typeof text !== 'string' || !text.trim()) throw new Error('The XYZ file is empty.');
  if (text.length > 1000000) throw new Error('The XYZ file exceeds 1 MB.');
  const rows = text.trimEnd().split(/\r\n|[\r\n]/);
  const size = rows[0].trim();
  if (!/^\d+$/.test(size)) throw new Error('Start the XYZ file with its atom count.');
  const count = Number(size);
  if (count < 2 || count > 256) throw new Error('Import a structure containing 2 to 256 atoms.');
  if (rows.length > count + 2) throw new Error('Import one frame. There are extra rows after the last atom.');
  if (rows.length < count + 2) throw new Error(`Include a comment line and all ${count} atom rows.`);
  const comment = rows[1];
  if (/\b(?:Lattice|Properties|pbc|(?:length_|coordinate_)?units?|coordinates?)\s*=/i.test(comment)) {
    throw new Error('This header contains extended XYZ metadata. Export a single isolated structure as plain XYZ in ångströms.');
  }
  const assignedCharges = [...comment.matchAll(/(?:^|\s)(?:charge|total_charge)\s*=\s*("[^"]*"|'[^']*'|\S+|$)/gi)];
  if (assignedCharges.length > 1) throw new Error('The XYZ comment gives more than one total charge.');
  let charge = 0;
  if (assignedCharges.length) {
    const value = assignedCharges[0][1].replace(/^(?:"([^"]*)"|'([^']*)')$/, (_, double, single) => double ?? single);
    if (!/^[+-]?\d+(?:\.0+)?$/.test(value)) throw new Error('Use an integer for charge in the XYZ comment.');
    charge = Number(value);
  }
  const elements = [];
  const coordinates = [];
  const real = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i;
  for (let line = 2; line < rows.length; line++) {
    const [element, ...fields] = rows[line].trim().split(/\s+/);
    if (fields.length !== 3 || !fields.every(field => real.test(field))) {
      throw new Error(`Line ${line + 1} must contain an element symbol and three numeric coordinates.`);
    }
    elements.push(element);
    coordinates.push(fields.map(Number));
  }
  const formula = validate({elements, coordinates, charge});
  return {
    id: 'imported', name, formula, charge, elements, coordinates,
    source: 'Imported from a single-frame XYZ structure with coordinates in ångströms.',
  };
}

function parameter(raw, fallback, low, high, name, integer = false) {
  const value = Number(raw ?? fallback);
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value)) || value < low || value > high) {
    throw new Error(`Use ${integer ? 'a whole number' : 'a number'} from ${low} to ${high} for ${name}.`);
  }
  return value;
}

export function makePython(structure, model, task, params = {}, nativeSource = '') {
  validate(structure);
  if (!['so3lr-tauri-s', 'so3lr-tauri-m', 'so3lr-tauri-l'].includes(model)) throw new Error('Choose a supported SO3LR-Tauri model.');
  if (!['energy', 'relax', 'md'].includes(task)) throw new Error('Choose a supported calculation type.');
  const device = params.device ?? 'auto';
  if (!['auto', 'cpu', 'gpu'].includes(device)) throw new Error('Choose Auto, CPU, or GPU.');
  const precision = params.precision ?? 'mixed';
  if (!['mixed', 'float32'].includes(precision)) throw new Error('Choose mixed or float32 precision.');
  let imports = '';
  let mdSettings = '';
  let calculation = '';
  if (task === 'relax') {
    const fmax = parameter(params.fmax, 0.05, 0.001, 1, 'force tolerance (eV/Å)');
    const steps = parameter(params.steps, 100, 1, 300, 'optimization steps', true);
    imports = 'from ase.optimize import BFGS';
    calculation = `optimizer = BFGS(atoms, logfile=None)
converged = optimizer.run(fmax=${fmax}, steps=${steps})
print(f"Optimization converged: {converged} (steps: {optimizer.nsteps})")
`;
  }
  if (task === 'md') {
    const steps = parameter(params.steps, 100, 1, 1000000, 'MD steps', true);
    const temperature = parameter(params.temperature, 300, 1, 2000, 'temperature (K)');
    const timestep = parameter(params.timestep, 1.0, 0.05, 2, 'time step (fs)');
    if (params.seed == null) imports = 'from random import randint';
    const seed = params.seed == null ? 'randint(0, 100)' : parameter(params.seed, 0, 0, 2**32 - 1, 'seed', true);
    if (!nativeSource.includes('def run_dynamics(')) throw new Error('The JAX dynamics source could not be loaded.');
    mdSettings = `temperature_K, timestep_fs, steps, seed = ${temperature}, ${timestep}, ${steps}, ${seed}`;
    calculation = `samples = run_dynamics(atoms.numbers, atoms.positions, model=model, charge=atoms.info["charge"],
                       precision=precision, temperature=temperature_K, timestep=timestep_fs,
                       steps=steps, seed=seed)
for frame_number, sample in enumerate(samples):
    step = sample["step"]
    frame = atoms.copy()
    frame.positions = sample["positions"]
    frame.info.update(task="md", step=step, time_fs=step * timestep_fs, timestep_fs=timestep_fs,
                      temperature_K=sample["temperature"], target_temperature_K=temperature_K,
                      kinetic_energy_eV=sample["kinetic_energy"], force_backend=sample["force_backend"],
                      seed=seed, ensemble="NVT", thermostat="Langevin", integrator="JAX BAOAB Langevin", md_engine="jax",
                      friction_per_fs=0.01, center_of_mass_fixed=True)
    attach_observables(frame, sample["energy"], sample["forces"], sample["partial_charges"], sample["dipole_debye"])
    write("solar_md.extxyz", frame, format="extxyz", append=frame_number > 0)
print(f"Wrote {frame_number + 1} frames to solar_md.extxyz")
print(f"Final potential energy (eV): {sample['energy']:.8f}")
print(f"Final maximum force (eV/Å): {np.linalg.norm(sample['forces'], axis=1).max():.8f}")
`;
  }
  const symbols = structure.elements.map(s => JSON.stringify(s)).join(', ');
  const coords = structure.coordinates.map(xyz => '        [' + xyz.join(', ') + '],').join('\n');
  const periodic=structure.pbc?.some(Boolean);
  if (periodic) {
    if(!Array.isArray(structure.cell)||structure.cell.length!==3||structure.cell.some(row=>row.length!==3||!row.every(Number.isFinite)))throw Error('Supply a three-dimensional periodic cell.');
    if(task==='md'||structure.charge!==0||!nativeSource.includes('def run_periodic('))throw Error('Use a neutral periodic energy or optimization job.');
    const steps=task==='relax'?parameter(params.steps,100,1,300,'optimization steps',true):1;
    const fmax=task==='relax'?parameter(params.fmax,.05,.001,1,'force tolerance'):0.05;
    calculation=`result = run_periodic(atoms.get_chemical_symbols(), atoms.positions, atoms.cell.array, model=model,
                          precision=precision, task="${task}", steps=${steps}, fmax=${fmax})
write_periodic_result("so3lr_periodic.extxyz", atoms.get_chemical_symbols(), result, model, precision)
print(f"Saved {len(result['frames'])} periodic frames to so3lr_periodic.extxyz")`;
  }else if (task !== 'md') {
    calculation = `atoms.calc = So3lrCalculator(model=model, lr_cutoff=100., precision=precision,
                             has_aux=True, observables=["partial_charges", "dipole_vec"])
` + calculation + `energy = float(atoms.get_potential_energy())
forces = atoms.get_forces(apply_constraint=False)
aux = atoms.calc.results["aux"]
dipole = np.asarray(aux["dipole_vec"]).reshape(3) / units.Debye if atoms.info["charge"] == 0 else None
frame = atoms.copy()
attach_observables(frame, energy, forces, aux["partial_charges"], dipole)
write("solar_${task}.extxyz", frame, format="extxyz")
print(f"Potential energy (eV): {energy:.8f}")
print(f"Maximum force (eV/Å): {np.linalg.norm(forces, axis=1).max():.8f}")
print("Forces (eV/Å):")
print(forces)
print("Wrote solar_${task}.extxyz")`;
  }
  const cudaDependency = device !== 'cpu' ? '\n#   "jax[cuda12]==0.10.1; sys_platform == \'linux\' and platform_machine == \'x86_64\'",' : '';
  const bodyLines = calculation.trimEnd().split('\n');
  const body = bodyLines.map(line => line ? '    ' + line : '').join('\n');
  return `# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "so3lr @ git+https://github.com/general-molecular-simulations/so3lr.git@e13d51b094bd7e2e7b8a4758aacdbafea9e82819",
#   "jax==0.10.1",
#   "jaxlib==0.10.1",
#   "ase==3.29.0",${cudaDependency}
# ]
# ///
# Run with uv run <filename.py>.
${task === 'md' ? '# run_dynamics integrates with JAX BAOAB on the selected device. The ASE utilities\n# provide structure handling, the reference calculation and extended XYZ output.' : '# SO3LR evaluates this ASE calculation with the JAX force field.'}
import os
import sys
if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
with jax.default_device(jax.local_devices(backend='cpu')[0]):
    from so3lr import So3lrCalculator
${imports}

precision = "${precision}"
requested_device = "${device}"
atoms = Atoms(
    [${symbols}],
    positions=[
${coords}
    ],
    ${periodic?'cell='+JSON.stringify(structure.cell)+', pbc=True,':'pbc=False,'}
)
model = "${model}"
atoms.info["charge"] = ${structure.charge}
${mdSettings}

${task === 'md' || periodic ? nativeSource : ''}
def attach_observables(frame, energy, forces, charges, dipole):
    frame.info.update(model=model, precision=precision, charge_source="so3lr", compute_device=device.device_kind,
                      energy_unit="eV", length_unit="angstrom", force_unit="eV/angstrom")
    q = np.asarray(charges, dtype=float).reshape(len(frame))
    tolerance = max(1e-4, 128 * len(frame) * np.finfo(np.float32).eps)
    if not np.isfinite(q).all() or abs(q.sum() - frame.info["charge"]) > tolerance:
        raise ValueError("Invalid SO3LR partial charges.")
    frame.set_array("partial_charges", q)
    if dipole is not None:
        frame.info["dipole_debye"] = np.asarray(dipole, dtype=float).reshape(3)
    frame.calc = SinglePointCalculator(frame, energy=float(energy), forces=np.asarray(forces))

def water_reference(precision_name):
    water = Atoms("OH2", positions=[[0, 0, 0], [.9572, 0, 0], [-.24, .9266, 0]])
    water.calc = So3lrCalculator(model="so3lr-tauri-s", lr_cutoff=100., precision=precision_name,
                                has_aux=True, observables=["partial_charges", "dipole_vec"])
    return water.get_potential_energy(), water.get_forces(), water.calc.results["aux"]["partial_charges"]

# An explicit mixed-precision CPU job initializes the model at full precision on its own.
# Other paths initialize shared constants here.
if requested_device != "cpu" or precision != "mixed":
    with jax.enable_x64(True), jax.default_device(jax.local_devices(backend="cpu")[0]):
        water_reference("mixed")

def select_device():
    backends = ("gpu", "cpu") if requested_device == "auto" else (
        ("gpu",) if requested_device == "gpu" else ("cpu",))
    for backend in backends:
        try:
            candidates = jax.local_devices(backend=backend)
        except RuntimeError:
            continue
        if candidates:
            return candidates[0]
    raise RuntimeError(f"No {requested_device} device available.")

with jax.enable_x64(precision == "mixed"):
    device = select_device()
    print(f"SO3LR device: {device.device_kind} ({device.platform}), precision: {precision}")

with jax.enable_x64(precision == "mixed"), jax.default_device(device):
${body}
`;
}
