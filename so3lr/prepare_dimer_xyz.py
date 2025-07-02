#!/usr/bin/env python3
"""
Prepare dimer XYZ files for SO3LR calculations.

This script takes an XYZ file containing a dimer structure that contains 
charge=0 charge_a=0 charge_b=0 selection_a=1-2 selection_b=3-4 benchmark_Eint=-0.090 
(like this one: https://github.com/Honza-R/NCIAtlas/blob/main/geometries/NCIA_D1200/1.01.01_100.xyz)
and creates an EXTXYZ file with both the original dimer and a version where one monomer is translated 10,000 Å away.
See methods section in SO3LR preprint for details: https://chemrxiv.org/engage/chemrxiv/article-details/68456a303ba0887c33e85a14
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase import Atoms
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Supported atomic numbers for the model (H, C, N, O, F, P, S, Cl)
SUPPORTED_ATOMIC_NUMBERS = {1, 6, 7, 8, 9, 15, 16, 17}

def check_supported_elements(atoms):
    """
    Check if all atoms in the structure have supported atomic numbers.
    Returns (is_supported, unsupported_elements).
    """
    atomic_numbers = set(atoms.get_atomic_numbers())
    unsupported = atomic_numbers - SUPPORTED_ATOMIC_NUMBERS
    return len(unsupported) == 0, unsupported

def parse_selection_range(selection_str):
    """Parse selection string like '1-4' into start and end indices (0-based)."""
    if not selection_str or selection_str == "":
        return None, None
    if '-' in selection_str:
        start, end = selection_str.split('-')
        return int(start) - 1, int(end) - 1  # Convert to 0-based indexing
    else:
        # Single atom selection
        idx = int(selection_str) - 1  # Convert to 0-based indexing
        return idx, idx

def get_monomer_info(atoms):
    """
    Extract monomer information from atoms object.
    Returns (selection_a, selection_b, charge_a, charge_b).
    """
    # Check if monomer info is already in atoms.info
    selection_a = atoms.info.get('selection_a')
    selection_b = atoms.info.get('selection_b')
    charge_a = atoms.info.get('charge_a')
    charge_b = atoms.info.get('charge_b')
    
    # Convert numeric values to strings if needed
    if selection_a is not None and not isinstance(selection_a, str):
        selection_a = str(selection_a)
    if selection_b is not None and not isinstance(selection_b, str):
        selection_b = str(selection_b)
    
    # Check if we have valid selections (both must be present and non-empty)
    if selection_a and selection_b and str(selection_a).strip() and str(selection_b).strip():
        # Parse existing selections
        start_a, end_a = parse_selection_range(str(selection_a))
        start_b, end_b = parse_selection_range(str(selection_b))
        
        if start_a is not None and end_a is not None:
            monomer_a_indices = list(range(start_a, end_a + 1))
        else:
            monomer_a_indices = []
            
        if start_b is not None and end_b is not None:
            monomer_b_indices = list(range(start_b, end_b + 1))
        else:
            monomer_b_indices = []
    
    return monomer_a_indices, monomer_b_indices, charge_a, charge_b


def create_translated_dimer(atoms, monomer_a_indices, monomer_b_indices, translation_distance=10000.0):
    """
    Create a copy of the dimer with one monomer translated by translation_distance Å.
    """
    atoms_copy = atoms.copy()
    positions = atoms_copy.get_positions()
    
    # Create translation vector (along x-axis)
    translation_vector = np.array([translation_distance, 0.0, 0.0])
    
    # Translate monomer B
    for idx in monomer_b_indices:
        positions[idx] += translation_vector
    
    atoms_copy.set_positions(positions)
    
    # Update info to indicate this is the translated version
    atoms_copy.info['structure_type'] = 'dimer_translated'
    atoms_copy.info['translation_distance'] = translation_distance
    
    return atoms_copy


def add_dimer_properties(atoms, monomer_a_indices, monomer_b_indices, charge_a, charge_b, structure_type='dimer'):
    """Add dimer-specific properties to atoms object."""
    # Add selection information
    if monomer_a_indices:
        atoms.info['selection_a'] = f"{min(monomer_a_indices)+1}-{max(monomer_a_indices)+1}"
    if monomer_b_indices:
        atoms.info['selection_b'] = f"{min(monomer_b_indices)+1}-{max(monomer_b_indices)+1}"
    
    # Add charges
    # print(atoms)
    atoms.info['charge_a'] = int(charge_a)
    atoms.info['charge_b'] = int(charge_b)
    
    # Add number of atoms in each monomer
    atoms.info['num_a'] = len(monomer_a_indices) if monomer_a_indices else 0
    atoms.info['num_b'] = len(monomer_b_indices) if monomer_b_indices else 0
    
    # Add overall charge if not present
    if 'charge' not in atoms.info:
        atoms.info['charge'] = int(charge_a + charge_b)

    # Copy benchmark energy to energy if present
    if 'benchmark_Eint' in atoms.info:
        atoms.info['energy'] = atoms.info['benchmark_Eint']
    else:
        atoms.info['energy'] = atoms.info.get('Energy')
    
    # Add structure type
    atoms.info['structure_type'] = structure_type
    
    return atoms


def process_dimer_file(input_file, output_file=None):
    """
    Process a dimer XYZ file and create EXTXYZ with original and translated dimers.
    Processes all molecules in the input file.
    """
    from ase.io import iread
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if output_file is None:
        # Create output filename with '_prepared.xyz' suffix
        output_file = input_path.with_name(input_path.stem + '_prepared.xyz')
    
    # Create unsupported elements output file
    unsupported_output_file = input_path.with_name(input_path.stem + '_unsupported.xyz')
    
    logging.info(f"Reading dimers from: {input_file}")
    
    # Count total molecules first
    try:
        molecule_count = sum(1 for _ in iread(input_file))
        logging.info(f"Found {molecule_count} molecules in input file")
    except Exception as e:
        logging.error(f"Failed to count molecules in input file: {e}")
        raise
    
    output_written = False
    unsupported_output_written = False
    processed_count = 0
    unsupported_count = 0
    
    # Process each molecule in the file
    try:
        for mol_idx, atoms in tqdm(enumerate(iread(input_file)), total=molecule_count):
            # Check if structure contains only supported elements
            is_supported, unsupported_elements = check_supported_elements(atoms)
            
            if not is_supported:
                # Write to unsupported file
                try:
                    if not unsupported_output_written:
                        write(unsupported_output_file, atoms, format='extxyz')
                        unsupported_output_written = True
                    else:
                        write(unsupported_output_file, atoms, format='extxyz', append=True)
                    unsupported_count += 1
                    
                    # Log which elements are unsupported
                    element_symbols = [atoms[i].symbol for i in range(len(atoms)) 
                                     if atoms[i].number in unsupported_elements]
                    unique_unsupported_symbols = list(set(element_symbols))
                    logging.debug(f"Molecule {mol_idx + 1} contains unsupported elements: {unique_unsupported_symbols}")
                    
                except Exception as e:
                    logging.error(f"Failed to write unsupported molecule {mol_idx + 1}: {e}")
                continue
            
            # Get monomer information
            try:
                monomer_a_indices, monomer_b_indices, charge_a, charge_b = get_monomer_info(atoms)
            except Exception as e:
                # Add debug information for problematic molecules
                logging.error(f"Failed to determine monomer information for molecule {mol_idx + 1}: {e}")
                continue
            
            # Add properties to original structure (with all dimer properties)
            atoms_original = add_dimer_properties(atoms.copy(), monomer_a_indices, monomer_b_indices, charge_a, charge_b, 'dimer_original')
            
            # Create translated dimer
            atoms_translated = create_translated_dimer(atoms, monomer_a_indices, monomer_b_indices)
            atoms_translated = add_dimer_properties(atoms_translated, monomer_a_indices, monomer_b_indices, charge_a, charge_b, 'dimer_translated')
            
            # Write structures to EXTXYZ file
            try:
                if not output_written:
                    # Write first original dimer (create new file)
                    write(output_file, atoms_original, format='extxyz')
                    output_written = True
                else:
                    # Append subsequent dimers
                    write(output_file, atoms_original, format='extxyz', append=True)
                
                # Append translated dimer
                write(output_file, atoms_translated, format='extxyz', append=True)
                processed_count += 1
                
            except Exception as e:
                logging.error(f"Failed to write molecule {mol_idx + 1} to output file: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        raise
    
    if processed_count == 0 and unsupported_count == 0:
        raise RuntimeError("No molecules were successfully processed")
    
    # Print summary with warnings
    logging.info(f"Processing complete:")
    if processed_count > 0:
        logging.info(f"✓ {processed_count}/{molecule_count} structures with supported elements written to: {output_file}")
        logging.info(f"  Output contains {processed_count * 2} structures (original + translated for each molecule)")
    
    if unsupported_count > 0:
        logging.warning(f"  {unsupported_count}/{molecule_count} structures with unsupported elements written to: {unsupported_output_file}")
        logging.warning(f"  Model currently supports elements: H, C, N, O, F, P, S, Cl")

    if processed_count == 0:
        logging.warning("No structures with supported elements were found!")

def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Prepare dimer XYZ files for SO3LR calculations",
        epilog="""
    This script takes an XYZ file containing a dimer structure that contains 
    charge=0 charge_a=0 charge_b=0 selection_a=1-2 selection_b=3-4 benchmark_Eint=-0.090 in atoms.info
    (like this one: https://github.com/Honza-R/NCIAtlas/blob/main/geometries/NCIA_D1200/1.01.01_100.xyz)
    and creates an EXTXYZ file with both the original dimer and a version where one monomer is translated 10,000 Å away.
    See methods section in SO3LR preprint for details: https://chemrxiv.org/engage/chemrxiv/article-details/68456a303ba0887c33e85a14

    Example:
    python3 prepare_dimer_xyz.py --datafile dimer.xyz
    python3 prepare_dimer_xyz.py --datafile dimer.xyz --output dimer_prepared.extxyz
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--datafile', 
        type=str, 
        required=True,
        help='Input XYZ file containing the dimer structure'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output EXTXYZ file (default: input filename with .extxyz extension)'
    )
    
    args = parser.parse_args()
    
    try:
        process_dimer_file(args.datafile, args.output)
        return 0
    except Exception as e:
        logging.error(f"Error processing dimer file: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
