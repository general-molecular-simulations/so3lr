"""Analyse isolated Cartesian trajectories in Å using a fixed, inferred peptide topology."""

from collections import defaultdict

import mdtraj as md
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


class TopologyError(ValueError):
    """The first frame does not identify an unambiguous linear peptide backbone."""


def aligned_heavy_rmsd(elements, frames):
    """Equally weighted Kabsch RMSD to frame zero, allowing proper rotations only."""
    indices = [i for i, symbol in enumerate(elements) if symbol != "H"]
    if not indices:
        return [None] * len(frames), 0
    xyz = frames[:, indices, :]
    centered = xyz - xyz.mean(axis=1, keepdims=True)
    ref = centered[0]
    u, _, vt = np.linalg.svd(centered.transpose(0, 2, 1) @ ref)
    u[:, :, -1] *= np.linalg.det(u @ vt)[:, None]
    aligned = centered @ (u @ vt)
    rmsd = np.sqrt(np.square(aligned - ref).sum(axis=2).mean(axis=1))
    return rmsd.round(6).tolist(), len(indices)


def backbone(elements, first):
    """Find N–CA–C(=O) motifs joined by peptide bonds; termini may carry caps."""
    if elements.count("N") < 2 or elements.count("O") < 2:
        raise TopologyError("No connected peptide backbone was identified.")
    lines = [str(len(elements)), ""]
    lines.extend(f"{el} {x:.8f} {y:.8f} {z:.8f}" for el, (x, y, z) in zip(elements, first))
    molecule = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    if molecule is None:
        raise TopologyError("The first frame could not be read as an XYZ structure.")
    try:
        rdDetermineBonds.DetermineConnectivity(molecule)
    except (RuntimeError, ValueError) as exc:
        raise TopologyError("Could not infer covalent connectivity from the first frame.") from exc

    heavy = [{a.GetIdx() for a in site.GetNeighbors() if a.GetSymbol() != "H"}
             for site in molecule.GetAtoms()]
    oxygens = {}
    for i, sites in enumerate(heavy):
        if elements[i] != "C" or len(sites) != 3:
            continue
        ox = [j for j in sites if elements[j] == "O" and len(heavy[j]) == 1
              and np.linalg.norm(first[j] - first[i]) < 1.4]
        if ox and any(elements[j] in ("N", "O") for j in sites - {ox[0]}):
            # For a terminal carboxylate, use the shorter C–O as backbone O.
            oxygens[i] = min(ox, key=lambda j: (np.linalg.norm(first[j] - first[i]), j))
    motifs = []
    for ca, sites in enumerate(heavy):
        if elements[ca] != "C" or ca in oxygens or not 2 <= len(sites) <= 4:
            continue
        for n in sorted(i for i in sites if elements[i] == "N"):
            for c in sorted(sites.intersection(oxygens)):
                motifs.append((n, ca, c, oxygens[c]))
    by_n = defaultdict(list)
    for k, (n, _, _, _) in enumerate(motifs):
        by_n[n].append(k)
    successors = []
    predecessors = defaultdict(list)
    for k, (_, _, c, _) in enumerate(motifs):
        linked = [j for n in heavy[c] for j in by_n[n] if k != j]
        successors.append(linked)
        for j in linked:
            predecessors[j].append(k)
    connected = {k for k in range(len(motifs)) if successors[k] or predecessors[k]}
    if not connected:
        raise TopologyError("No connected peptide backbone was identified.")
    if any(len(successors[k]) > 1 or len(predecessors[k]) > 1 for k in connected):
        raise TopologyError("Branched or ambiguous peptide connectivity prevents a residue assignment.")
    assigned = set()
    chains = []
    for k in sorted(connected, key=lambda i: motifs[i][1]):
        if predecessors[k]:
            continue
        chain = []
        while True:
            motif = motifs[k]
            if k in assigned:
                raise TopologyError("Could not identify a unique order for the peptide residues.")
            chain.append(motif)
            assigned.add(k)
            if not successors[k]:
                break
            k = successors[k][0]
        chains.append(chain)
    flat = [atom for chain in chains for res in chain for atom in res]
    if assigned != connected or len(flat) != len(set(flat)):
        raise TopologyError("Cyclic or overlapping peptide backbones are not supported.")
    return chains, heavy


def assign_peptide(elements, frames):
    chains, heavy = backbone(elements, frames[0])
    if len(chains) != 1 or len(chains[0]) != 15:
        raise TopologyError("Peptide analysis is available for Ala15.")
    for n, ca, carbonyl, _ in chains[0]:
        side_chain = heavy[ca] - {n, carbonyl}
        if len(side_chain) != 1:
            raise TopologyError("Ala15 requires one methyl side chain on every residue.")
        beta = next(iter(side_chain))
        if elements[beta] != "C" or heavy[beta] != {ca}:
            raise TopologyError("Ala15 requires one methyl side chain on every residue.")
    top = md.Topology()
    mapped = []
    for chain in chains:
        md_chain = top.add_chain()
        for number, motif in enumerate(chain, 1):
            # MDTraj uses PRO to exclude a substituted amide N as a hydrogen donor.
            name = "PRO" if len(heavy[motif[0]]) >= 3 else "RES"
            residue = top.add_residue(name, md_chain, resSeq=number)
            for atom_name, idx in zip(("N", "CA", "C", "O"), motif):
                top.add_atom(atom_name, md.element.get_by_symbol(elements[idx]), residue)
                mapped.append(idx)
    # MDTraj expects nanometres. The complete peptide residues form the DSSP
    # topology; incomplete terminal caps are not counted as residues.
    trajectory = md.Trajectory(np.asarray(frames[:, mapped, :] * 0.1, dtype=np.float32), top)
    raw = md.compute_dssp(trajectory, simplified=False)
    codes = np.where(raw == " ", "C", raw)
    alphabet = "HGTCBEIS"
    if not np.isin(codes, list(alphabet)).all():
        raise TopologyError("DSSP could not assign all detected peptide residues.")
    fractions = {code: (codes == code).mean(axis=1).tolist() for code in alphabet}
    return dict(method="DSSP (MDTraj)", residue_count=top.n_residues, fractions=fractions)


def analyze_trajectory(elements, frames):
    """Return heavy-atom RMSD and, for Ala15, per-frame DSSP fractions.

    The residue denominator includes every detected complete backbone residue,
    including terminal residues. DSSP blank is represented by C; S remains bend.
    """
    elements = list(elements)
    coords = np.asarray(frames, dtype=float)
    if coords.ndim != 3 or coords.shape[1:] != (len(elements), 3) or not len(coords):
        raise ValueError("Expected at least one Cartesian frame with one position per atom.")
    if not np.isfinite(coords).all():
        raise ValueError("Trajectory coordinates must be finite.")
    rmsd, count = aligned_heavy_rmsd(elements, coords)
    result = dict(rmsd=rmsd, heavy_atom_count=count, peptide=None)
    if elements.count("N") < 15 or elements.count("O") < 15:
        return result
    try:
        result["peptide"] = assign_peptide(elements, coords)
    except TopologyError as error:
        result["reason"] = str(error)
    return result
