import numpy as np

from ase import Atoms


# TODO: move this to mlff jraph_utils at some point
def jraph_to_ase_atoms(graph):
    """ Convert graph to ase.atoms object. """

    cell = graph.globals.get('cell')
    # A molecule batched with crystals carries a zero cell (see
    # mlff.utils.jraph_utils.with_uniform_periodic_fields): it stays non-periodic.
    if cell is not None and np.any(np.asarray(cell[0]) != 0):
        cell = cell[0]  # (1, 3, 3) -> (3, 3)
        pbc = True
    else:
        cell, pbc = None, False

    positions = graph.nodes['positions']
    numbers = graph.nodes['atomic_numbers']

    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=pbc
    )

    # Handle total_charge - could be 0-d or 1-d array after unbatching
    total_charge = graph.globals["total_charge"]
    if hasattr(total_charge, 'shape') and total_charge.shape:
        total_charge = total_charge.item() if total_charge.size == 1 else int(total_charge[0])
    atoms.info['charge'] = int(total_charge)
    atoms.info['energy'] = float(graph.globals["energy"][0])
    atoms.info['energy_so3lr'] = float(graph.globals["energy_so3lr"][0])
    atoms.arrays['forces'] = graph.nodes['forces']
    atoms.arrays['forces_so3lr'] = graph.nodes['forces_so3lr']
    for key in ['hirshfeld_ratios', 'hirshfeld_ratios_so3lr', 'c6_ratios', 'c6_ratios_so3lr']:
        if key in graph.nodes and graph.nodes[key] is not None:
            atoms.arrays[key] = graph.nodes[key]
    for key in ['dipole_vec', 'dipole_vec_so3lr']:
        if key in graph.globals and graph.globals[key] is not None:
            atoms.info[key] = np.asarray(graph.globals[key], np.float64).flatten()

    # Write any additional observable quantities that were added to nodes
    _handled_node_keys = {'positions', 'atomic_numbers', 'forces', 'forces_so3lr',
                          'hirshfeld_ratios', 'hirshfeld_ratios_so3lr'}
    for key in graph.nodes:
        if key.endswith('_so3lr') and key not in _handled_node_keys:
            atoms.arrays[key] = graph.nodes[key]

    return atoms

