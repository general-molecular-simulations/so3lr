"""Check SO3LR-Flash against the neighbour-list potential of ``so3lr nve``: energy, forces
and dE/d(strain) (the NPT pressure's virial).

    python -m so3lr.flash.verify --input water.xyz [--model so3lr-m] [--precision float64]
        [--pme electrostatics,dispersion --kspace-smearing 2 --kspace-spacing 1.55]

Builds the neighbour-list potential exactly as ``perform_md`` does (fractional
coordinates, jax-md neighbour lists) and the flash engine as ``--engine flash`` does, on the
structure as given and on a copy with 40 hydrogens pushed into contact with oxygens
(the descriptor floor), and prints the differences. Run on an NVIDIA GPU; the box must
be at least 2 x lr_cutoff and 7 grid cells across (12,288 waters: 49.7 A).
"""
import argparse
import sys
from functools import partial

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--precision", default="float64", choices=["float32", "float64"])
    ap.add_argument("--lr-cutoff", type=float, default=12.0)
    ap.add_argument("--flash-mode", default="default")
    ap.add_argument("--pme", default="", help="comma list: electrostatics, dispersion (or both)")
    ap.add_argument("--kspace-smearing", type=float, default=2.0)
    ap.add_argument("--kspace-spacing", type=float, default=1.55)
    args = ap.parse_args(argv)
    pme = set(filter(None, args.pme.split(",")))
    kel = "pme" if "electrostatics" in pme else None
    kdi = "pme" if "dispersion" in pme else None

    from so3lr.cli.so3lr_md import (atoms_to_jnp, cell_from_atoms_boundary_conditions, handle_box,
                                    load_model, process_model, setup_jax_globals, setup_kspace_grid)
    setup_jax_globals(precision=args.precision)
    import jax
    import jax.numpy as jnp
    import jax_md
    from ase.io import read
    from so3lr.flash.md import build_engine

    dtype = jnp.float64 if args.precision == "float64" else jnp.float32
    atoms = read(args.input)
    atoms.wrap()
    rng = np.random.default_rng(0)
    clash = atoms.copy()
    pos, z = clash.get_positions(), clash.get_atomic_numbers()
    o_idx, h_idx = np.nonzero(z == 8)[0], np.nonzero(z == 1)[0]
    for k, hi in enumerate(rng.permutation(h_idx)[:40]):
        d = rng.normal(size=3)
        pos[hi] = pos[o_idx[(k * 97 + 13) % len(o_idx)]] + d / np.linalg.norm(d) * (0.55 + 0.2 * rng.random())
    clash.set_positions(pos)
    clash.wrap()

    potential = load_model(args.model, precision=dtype, lr_cutoff=args.lr_cutoff,
                           dispersion_damping=2.0, kspace_electrostatics=kel, kspace_dispersion=kdi)
    settings = dict(model_path=args.model, lr_cutoff=args.lr_cutoff, dispersion_damping=2.0,
                    flash_mode=args.flash_mode, total_charge=0, num_unpaired_electrons=0,
                    kspace_electrostatics=kel, kspace_smearing=args.kspace_smearing,
                    kspace_spacing=args.kspace_spacing, kspace_dispersion=kdi,
                    kspace_dispersion_smearing=args.kspace_smearing,
                    kspace_dispersion_spacing=args.kspace_spacing)
    eng, staged = build_engine(atoms, settings, dtype)
    strain = eng.energy_forces_strain_staged if staged else eng.energy_forces_strain
    ok = True
    for name, at in (("as given", atoms), ("with contacts", clash)):
        geo = atoms_to_jnp(at, precision=dtype)
        cell = cell_from_atoms_boundary_conditions(at, args.lr_cutoff, None, 4.0)
        position, box, displacement, shift, frac = handle_box("periodic", geo["positions"], cell, precision=dtype)
        neighbor_fn, neighbor_fn_lr, energy_or_obs_fn = process_model(
            potential=potential, species=geo["species"], displacement=displacement, box=box,
            total_charge=0, num_unpaired_electrons=0, precision=args.precision,
            fractional_coordinates=frac, buffer_size_multiplier_lr=1.25, buffer_size_multiplier_sr=1.25,
            minimum_cell_size_multiplier=1.0, has_aux=False)
        k_grid, k_smearing = setup_kspace_grid(kel, args.kspace_smearing, args.kspace_spacing, box)
        k_grid_disp, k_smearing_disp = setup_kspace_grid(kdi, args.kspace_smearing, args.kspace_spacing, box)
        if kel or kdi:
            print(f"  meshes: so3lr_dev {None if k_grid is None else k_grid.shape} "
                  f"{None if k_grid_disp is None else k_grid_disp.shape}, flash {eng.mesh_el} {eng.mesh_disp}")
        energy_fn = partial(energy_or_obs_fn, has_aux=False, k_smearing=k_smearing, k_smearing_disp=k_smearing_disp)
        nbrs = neighbor_fn.allocate(position, box=box)
        nbrs_lr = neighbor_fn_lr.allocate(position, box=box)
        kw = dict(neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box, k_grid=k_grid, k_grid_disp=k_grid_disp)
        e_ref = float(jax.jit(energy_fn)(position, **kw))
        f_ref = np.asarray(jax.jit(jax_md.quantity.force(energy_fn))(position, **kw), np.float64)
        # dE/d(strain): box and positions scaled together, as jax-md's NPT takes it
        w_ref = float(jax.jit(jax.grad(lambda eps: energy_fn(position, perturbation=1.0 + eps, **kw)))(
            jnp.asarray(0.0, dtype)))
        e, f, w = strain(jnp.asarray(at.get_positions(), dtype))
        de = abs(float(e) - e_ref) / len(at)
        df = float(np.abs(np.asarray(f, np.float64) - f_ref).max())
        dw = abs(float(w) - w_ref) / max(abs(w_ref), 1.0)
        tol = (1e-9, 1e-5, 1e-8) if dtype == jnp.float64 else (1e-6, 5e-3, 1e-3)
        good = de < tol[0] and df < tol[1] and dw < tol[2]
        ok &= good
        print(f"  {name:<14} {args.precision} pme={args.pme or '-'}  E/atom {e_ref / len(at):.6f}  |dE|/atom {de:.1e}  "
              f"max|dF| {df:.1e} eV/A  (max|F| {np.abs(f_ref).max():.1f})  dE/dstrain {w_ref:.4f} "
              f"rel diff {dw:.1e}  {'PASS' if good else 'FAIL'}",
              flush=True)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
