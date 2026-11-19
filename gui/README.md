# SO3LR

Explore a 1 ns Ala15 trajectory, inspect molecular and periodic structures from the Tauri paper, or run SO3LR-Tauri S, M and L on your own structures. The workbench ships its example coordinates and recordings and runs calculations on the computer hosting the Python service.

## Start the workbench

From the directory containing this README, run the following command. On first launch, uv downloads the pinned public dependencies into an isolated environment. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv run --locked serve.py --port 8773
```

Open the workbench in your browser. The first view plays the Ala15 recording. Playback starts paused when the browser requests reduced motion. During optimization and molecular dynamics, the viewer follows the latest saved geometry and shows its step. Periodic optimization updates the unit cell too. A refresh in a tab with an active calculation reconnects to that job. The static interface can browse the library and replay bundled results offline. SMILES generation, trajectory import and new calculations use the local Python service. [Open SO3LR](http://127.0.0.1:8773/)

## Explore the 1 ns Ala15 trajectory

The opening recording shows the 162-atom capped peptide Ac-(Ala)₁₅-NHMe in the gas phase at a target of 300 K. It comes from replica 1 (run `r2`) of the Tauri paper’s folded Ala15 study. The original SO3LR-Tauri M production run used JAX-MD with a Nosé–Hoover chain and a 0.5 fs integration step. The replay covers the first 1 ns in 501 snapshots, starting at 0.5 ps and then every 2 ps. The viewer aligns heavy atoms to the first frame to remove overall tumbling. Exported coordinates retain their original orientation. [Tauri, Figure S16](https://arxiv.org/html/2611.06237v1) · [Recording preparation](benchmarks/prepare_ala15_recording.py)

Pause the player or drag its slider to inspect a frame. Clicking or dragging in an analysis chart moves the structure and the other chart cursors to the same frame. Use the arrow keys when a chart has keyboard focus. Choose **Ala15 → Saved 1 ns trajectory** to reopen the recording.

- **Secondary structure** reports the fraction of all 15 alanine residues assigned to each DSSP class at every displayed time. Legend percentages average over all saved frames and all 15 residues. The chart follows the population at each frame. Caps are excluded. Toggle α-helix, 3₁₀-helix, turn, coil or another observed state in the legend. The GUI computes DSSP from the coordinates. The paper used STRIDE. Transitions faster than the display sampling interval can fall between snapshots. Peptide-specific analysis appears for Ala15, including recognized imported trajectories. [MDTraj DSSP](https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html)
- **Kinetic energy** and **instantaneous temperature** use the archived momenta and the paper run’s 3N temperature convention. The two raw trajectory segments omit potential energy, so the opening replay has no potential-energy curve.
- **Save trajectory** exports the displayed snapshots as extended XYZ. The replay’s timestamps preserve the original sampling times.

## Find an example

Open **Examples** to see all 10 runnable structures on one page. Each card shows the structure name, its dataset or coordinate source, and the atom count. All ten examples support optimization and molecular dynamics. Choose **Points** to see atoms as colored points.

## Enter your own structure

- **SMILES** under **Add XYZ or SMILES** accepts strings such as `CCO` and `C[NH3+]`. RDKit preserves specified stereochemistry and formal charge, adds hydrogens and generates a seeded ETKDGv3 conformer. Where parameters are available, it refines the geometry using MMFF94 or UFF. For disconnected inputs, the builder places fragments 3.5 Å apart along x.
- **XYZ coordinates** opens or pastes a single plain molecular XYZ in ångströms. An integer `charge=` in the comment sets the molecular charge.
- **Run history → Open trajectory** imports consistently ordered XYZ or extended XYZ frames. Fully periodic frames use three lattice vectors and three periodic axes. For molecular MD without `time_fs`, enter the time between saved frames. Imports support up to 5,000 frames and 500,000 atom positions, with up to 256 atoms in an isolated frame or 50,000 in a periodic frame. Charts use the available data. Coordinates suffice for molecular RMSD, lattice vectors for cell volume, and archived kinetic energies or momenta for kinetic energy.

## Run a calculation

In **New calculation**, use **Add XYZ or SMILES** or choose a molecule from the **Example library**. Choose a SO3LR-Tauri model and a calculation type. The local service queues one job at a time and accepts 2–256 atoms. Isolated systems support BFGS geometry optimization and Langevin MD with integer total charge from −10 to +10 e. Neutral three-dimensionally periodic cells support relaxation of both atom positions and cell vectors using LBFGS. Periodic calculations use exact real-space neighbors with a 12 Å cutoff and particle-mesh Ewald electrostatics. Isolated calculations use a 100 Å long-range cutoff. Energies are in eV and forces in eV/Å. [SO3LR ASE calculator](https://so3lr.github.io/usage/ase/)

For **Molecular dynamics**, the default is 100 steps of 1.0 fs at 300 K. Each new MD run draws an integer seed from 0 through 100. The saved MD trajectory includes the seed used. Live simulations and **Export Python** use the same JAX BAOAB Langevin integrator with friction 0.01 fs⁻¹. A fixed center of mass leaves 3N−3 thermal degrees of freedom for live dynamics. Jobs of up to 2,000 steps retain up to 101 frames and longer jobs retain up to 1,001. Cancellation takes effect at an integration block boundary and retains the most recent completed frame.

Live MD results include potential energy, kinetic energy, instantaneous temperature and heavy-atom RMSD. The interface colors atoms by element. **Save trajectory** writes available frame properties and the **Export Python** job can stream snapshots to disk. The Python export shares the workbench’s force and integration implementations.

## Optional CUDA support

Exported Python jobs with Automatic or GPU device selection include CUDA 12 dependencies on Linux x86-64. For the local GUI on Linux x86-64 with a compatible NVIDIA driver, use the CUDA 12 extra. Browser and server can run on different computers through SSH port forwarding. The server host performs the calculation. [JAX installation](https://docs.jax.dev/en/latest/installation.html)

```sh
uv run --locked --extra cuda12 serve.py --port 8773
```

## Sources and attribution

The core molecular inputs include aspirin imported from the SMILES `CC(=O)Oc1ccccc1C(=O)O` (PubChem CID 2244; the same seeded conformer is bundled for offline use), the stacked JSCH-71 guanine–cytosine pair, the first MD22 frame of docosahexaenoic acid, and two capped peptides. **Ala4** is the MD22 42-atom Ac-(Ala)₃-NHMe example with 100 conformers from the SO3LR test file. Use **Source XYZ** for the original peptide files. The Ala4 example includes a 50 fs replay. [MD22 data](https://www.sgdml.org/) · [MD22 paper](https://doi.org/10.1126/sciadv.adf0873) · [Ala4 source](https://github.com/general-molecular-simulations/so3lr/blob/e13d51b094bd7e2e7b8a4758aacdbafea9e82819/tests/test_data/ac.xyz) · [Ala15 source](https://github.com/general-molecular-simulations/so3lr/blob/e13d51b094bd7e2e7b8a4758aacdbafea9e82819/examples/data/ala15_folded.xyz)

Kabylda and coauthors released the Tauri paper data under CC BY 4.0. The library includes representative geometries for larger benchmark collections. The catalog identifies the selected archive members and frames. Additional upstream attributions are listed in the coverage notes. The header and favicon use the orbital mark from the SO3LR documentation. [Zenodo release](https://doi.org/10.5281/zenodo.22920714) · [Molecular benchmark catalog](web/paper-catalog.json) · [Coverage and upstream credits](benchmarks/paper-examples.md) · [SO3LR documentation](https://so3lr.github.io/)

The model is pinned to SO3LR 0.2.0 at commit `e13d51b094bd7e2e7b8a4758aacdbafea9e82819`. Upstream code is MIT-licensed. The Tauri model weights are licensed under CC BY-NC 4.0. [SO3LR source](https://github.com/general-molecular-simulations/so3lr/tree/e13d51b094bd7e2e7b8a4758aacdbafea9e82819) · [Upstream notices](https://github.com/general-molecular-simulations/so3lr/blob/e13d51b094bd7e2e7b8a4758aacdbafea9e82819/NOTICES.md) · [SO3LR Tauri paper and supporting information](https://arxiv.org/html/2611.06237v1)
