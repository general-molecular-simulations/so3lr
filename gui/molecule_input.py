"""Build a single, reproducible starting geometry from a user-supplied SMILES string."""

import hashlib

import numpy as np
from ase.formula import Formula
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

MAX_ATOMS = 256
SEED = 20261119


def from_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Enter a SMILES string, for example CCO for ethanol.")
    smiles = smiles.strip()
    if len(smiles) > 4096 or any(c.isspace() for c in smiles):
        raise ValueError("Enter one SMILES string of up to 4096 characters, without a name or spaces.")

    parser = Chem.SmilesParserParams()
    parser.parseName = False
    parser.allowCXSMILES = False
    molecule = Chem.MolFromSmiles(smiles, parser)
    if molecule is None:
        raise ValueError("RDKit could not read this SMILES. Check the rings, bonds and brackets.")
    if any(not 1 <= atom.GetAtomicNum() <= 85 for atom in molecule.GetAtoms()):
        raise ValueError("Use elements from hydrogen to astatine. Replace any wildcard atoms.")
    if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
        raise ValueError("Isotope-specific masses are unavailable in this workbench. Use an unlabeled structure.")

    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
    charge = Chem.GetFormalCharge(molecule)
    if abs(charge) > 10:
        raise ValueError("The formal charge must be between −10 and +10 e.")
    molecule = Chem.AddHs(molecule)
    if not 2 <= molecule.GetNumAtoms() <= MAX_ATOMS:
        raise ValueError(f"Use 2–{MAX_ATOMS} atoms including hydrogens. This SMILES has {molecule.GetNumAtoms()}.")

    embedding = AllChem.ETKDGv3()
    embedding.randomSeed = SEED
    embedding.enforceChirality = True
    embedding.timeout = 10
    if AllChem.EmbedMolecule(molecule, embedding) < 0:
        embedding.useRandomCoords = True
        if AllChem.EmbedMolecule(molecule, embedding) < 0:
            raise ValueError("RDKit could not generate a 3D conformer for this SMILES. Import XYZ coordinates instead.")

    method = "ETKDGv3"
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
        method += " and up to 500 MMFF94 optimization iterations"
    elif AllChem.UFFHasAllMoleculeParams(molecule):
        AllChem.UFFOptimizeMolecule(molecule, maxIters=500)
        method += " and up to 500 UFF optimization iterations"

    coords = molecule.GetConformer().GetPositions()
    source = f"3D conformer generated from {canonical} with RDKit {rdBase.rdkitVersion}, {method}, seed {SEED}."
    parts = Chem.GetMolFrags(molecule)
    if len(parts) > 1:
        # ETKDG embeds disconnected fragments separately around the same origin.
        # Separate their x extents, retaining internal geometry and atom ordering.
        offset = 0.0
        for part in parts:
            indices = list(part)
            coords[indices, 0] += offset - coords[indices, 0].min()
            offset = coords[indices, 0].max() + 3.5
        source += " Disconnected fragments start 3.5 Å apart along x."
    coords -= coords.mean(axis=0)
    if not np.isfinite(coords).all():
        raise ValueError("RDKit returned invalid coordinates. Import XYZ coordinates instead.")

    elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    formula = Formula.from_list(elements).format('hill')
    return {
        "id": "smiles-" + hashlib.sha256(canonical.encode()).hexdigest()[:16],
        "name": canonical if len(canonical) <= 32 else "SMILES molecule",
        "formula": formula,
        "charge": charge,
        "elements": elements,
        "coordinates": coords.round(8).tolist(),
        "smiles": canonical,
        "source": source,
    }
