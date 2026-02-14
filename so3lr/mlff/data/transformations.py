import jraph
import numpy as np
from typing import Dict, Sequence


def unit_conversion_graph(
        g,
        energy_unit: float,
        length_unit: float,
        dipole_vec_unit: float
):
    _energy_unit = np.asarray(energy_unit)
    _length_unit = np.asarray(length_unit)
    _dipole_vec_unit = np.asarray(dipole_vec_unit)

    g.globals['energy'] = g.globals.get('energy') * _energy_unit
    g.globals['stress'] = g.globals.get('stress') * _energy_unit / np.power(_length_unit, 3)

    g.nodes['forces'] = g.nodes.get('forces') * _energy_unit / _length_unit
    g.nodes['positions'] = g.nodes.get('positions') * _length_unit

    g.globals['dipole_vec'] = g.globals.get('dipole_vec') * _dipole_vec_unit

    return g


def _unpack_item(item):
    """Unpack an item that is either a (graph, long_range) tuple or a plain graph."""
    if isinstance(item, tuple) and len(item) == 2:
        return item[0], item[1]
    return item, None


def _repack_item(graph, lr):
    """Re-pack a graph with its long_range data."""
    if lr is not None:
        return graph, lr
    return graph


def unit_conversion(
        x: Sequence,
        energy_unit: float,
        length_unit: float,
        dipole_vec_unit: float
):
    _energy_unit = np.asarray(energy_unit)
    _length_unit = np.asarray(length_unit)
    _dipole_vec_unit = np.asarray(dipole_vec_unit)

    for item in x:
        g, lr = _unpack_item(item)
        g.globals['energy'] = g.globals.get('energy') * _energy_unit
        g.globals['stress'] = g.globals.get('stress') * _energy_unit / np.power(_length_unit, 3)

        g.nodes['forces'] = g.nodes.get('forces') * _energy_unit / _length_unit
        g.nodes['positions'] = g.nodes.get('positions') * _length_unit
        g.globals['dipole_vec'] = g.globals.get('dipole_vec') * _dipole_vec_unit

        yield _repack_item(g, lr)


def subtract_atomic_energy_shifts(x: Sequence, atomic_energy_shifts: Dict):
    # Create a NumPy array filled with zeros.
    result_array = np.zeros(118 + 1)

    # Fill the array using the values from the dictionary.
    for key, value in atomic_energy_shifts.items():
        result_array[key] = value

    # Convert to numpy array.
    atomic_energy_shifts_arr = np.array(result_array)

    for item in x:
        g, lr = _unpack_item(item)
        atomic_numbers = g.nodes.get('atomic_numbers')
        g.globals['energy'] = g.globals.get('energy') - np.take(atomic_energy_shifts_arr, atomic_numbers).sum()
        yield _repack_item(g, lr)


def calculate_energy_mean(x: Sequence):
    rolling_mean = np.asarray(0.)
    for n, item in enumerate(x):
        g, _ = _unpack_item(item)
        count = n + 1
        energy = g.globals.get('energy')
        if count == 1:
            rolling_mean = rolling_mean + energy / count
        else:
            rolling_mean = (rolling_mean + energy / (count - 1)) / count * (count - 1)

    return np.squeeze(rolling_mean)


def calculate_average_number_of_nodes(x: Sequence):
    rolling_mean = np.asarray(0.)
    for n, item in enumerate(x):
        g, _ = _unpack_item(item)
        count = n + 1
        num_nodes = len(g.nodes.get('atomic_numbers'))
        if count == 1:
            rolling_mean = (rolling_mean + num_nodes / count)
        else:
            rolling_mean = ((rolling_mean + num_nodes / (count - 1)) / count * (count - 1))

    return rolling_mean


def calculate_lse_energy_shifts(x: Sequence, energy_unit: float = 1.0):
    """Calculate per-element energy shifts via least-squares fit on training data.

    Args:
        x: Training data sequence of graphs (or (graph, long_range) tuples).
        energy_unit: Conversion factor applied to raw energies before fitting.

    Returns:
        Dict mapping atomic number strings ('0' .. '118') to shift values in eV.
    """
    from so3lr.mlff.data import preprocessing

    z_list = []
    e_list = []
    for item in x:
        g, _ = _unpack_item(item)
        z_list.append(np.array(g.nodes.get('atomic_numbers')))
        e_list.append(float(np.squeeze(g.globals.get('energy'))) * energy_unit)

    max_len = max(len(z) for z in z_list)
    z_padded = np.zeros((len(z_list), max_len), dtype=np.int32)
    for i, z in enumerate(z_list):
        z_padded[i, :len(z)] = z
    q = np.array(e_list)

    shifts, _ = preprocessing.get_per_atom_shift(z=z_padded, q=q, pad_value=0)
    return {str(a): float(shifts[a]) if a < len(shifts) else 0. for a in range(119)}


def calculate_average_number_of_neighbors(x: Sequence):
    rolling_mean = np.asarray(0.)
    for n, item in enumerate(x):
        g, _ = _unpack_item(item)
        count = n + 1
        num_edges = float(len(g.receivers))
        num_nodes = float(len(g.nodes.get('atomic_numbers')))
        if count == 1:
            rolling_mean = (rolling_mean + (num_edges / num_nodes) / count)
        else:
            rolling_mean = ((rolling_mean + (num_edges / num_nodes) / (count - 1)) / count * (count - 1))

    return rolling_mean
