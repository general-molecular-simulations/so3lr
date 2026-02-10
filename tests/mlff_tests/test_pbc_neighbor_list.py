import numpy as np

from .test_data import load_data


def compare_distances(x, y, cutoff):
    x_ = np.sort(x.reshape(-1))
    y_ = np.sort(y.reshape(-1))

    x_ = x_[x_ > 0]
    y_ = y_[y_ > 0]

    x_ = x_[x_ < cutoff]
    y_ = y_[y_ < cutoff]

    return np.isclose(x_, y_, atol=1e-5).all()


def compare_neighbors(x, y):
    x_ = x.reshape(-1)
    y_ = y.reshape(-1)

    x_ = x_[x_ > -1]
    y_ = y_[y_ > -1]

    x_ = np.sort(x_)
    y_ = np.sort(y_)

    assert len(x_) == len(y_)
    return (x_ == y_).all()


def test_pbc_neighbor_list_per_geometry():
    from so3lr.mlff.indexing.indices import pbc_neighbor_list
    from so3lr.mlff.geometric import add_cell_offsets

    trajectory = load_data('multi_sized_solids.traj')
    cutoff = 5.
    capacity_multiplier = 1.

    for atoms in trajectory:
        # for each geometry we initialize a new neighbor list and compare it with the ASE distances

        d_ij_ase = atoms.get_all_distances(mic=True)

        allocate_fn, update_fn = pbc_neighbor_list(cell=np.array(atoms.get_cell()).T,
                                                   cutoff=cutoff,
                                                   capacity_multiplier=capacity_multiplier)

        neighbors = allocate_fn(pos=atoms.get_positions(),
                                pbc=atoms.get_pbc(),
                                new_cell=np.array(atoms.get_cell()).T
                                )

        neighbors = update_fn(pos=atoms.get_positions(),
                              primitive_neighbors=neighbors,
                              pbc=atoms.get_pbc(),
                              new_cell=np.array(atoms.get_cell()).T
                              )

        R = atoms.get_positions()
        r_ij_mlff = np.array(list(map(lambda u: R[u[1]] - R[u[0]], zip(neighbors.idx_i, neighbors.idx_j))))

        r_ij_mlff = add_cell_offsets(r_ij=r_ij_mlff,
                                     cell=np.array(atoms.get_cell()),
                                     cell_offsets=neighbors.shifts
                                     )

        d_ij_mlff = np.linalg.norm(r_ij_mlff, axis=-1)

        assert compare_distances(d_ij_ase, d_ij_mlff, cutoff=cutoff)


def test_joint_pbc_neighbor_list_for_all_geometries():
    from so3lr.mlff.indexing.indices import pbc_neighbor_list
    from so3lr.mlff.geometric import add_cell_offsets

    trajectory = load_data('multi_sized_solids.traj')
    cutoff = 5.
    capacity_multiplier = 1.25

    neighbors = None
    idx_lengths = []

    for atoms in trajectory:
        # all geometries share the same neighborhood list

        if neighbors is None:
            allocate_fn, update_fn = pbc_neighbor_list(cell=np.array(atoms.get_cell()).T,
                                                       cutoff=cutoff,
                                                       capacity_multiplier=capacity_multiplier)
            neighbors = allocate_fn(pos=atoms.get_positions(),
                                    pbc=atoms.get_pbc(),
                                    new_cell=np.array(atoms.get_cell()).T
                                    )

        d_ij_ase = atoms.get_all_distances(mic=True)

        neighbors = update_fn(pos=atoms.get_positions(),
                              primitive_neighbors=neighbors,
                              pbc=atoms.get_pbc(),
                              new_cell=np.array(atoms.get_cell()).T
                              )

        if neighbors.overflow:
            neighbors = allocate_fn(pos=atoms.get_positions(),
                                    pbc=atoms.get_pbc(),
                                    new_cell=np.array(atoms.get_cell()).T
                                    )

        assert neighbors.idx_i[-1] == -1
        assert neighbors.idx_j[-1] == -1
        assert (neighbors.shifts[-1] == 0).all()

        idx_lengths += [len(neighbors.idx_i)]
        if len(idx_lengths) > 1:
            assert idx_lengths[-1] == idx_lengths[-2]

        R = atoms.get_positions()
        r_ij_mlff = np.array(list(map(lambda u: R[u[1]] - R[u[0]], zip(neighbors.idx_i, neighbors.idx_j))))

        r_ij_mlff = add_cell_offsets(r_ij=r_ij_mlff,
                                     cell=np.array(atoms.get_cell()),
                                     cell_offsets=neighbors.shifts
                                     )

        d_ij_mlff = np.linalg.norm(r_ij_mlff, axis=-1)

        assert compare_distances(d_ij_ase, d_ij_mlff, cutoff=cutoff)


def test_get_pbc_neighbors():
    from so3lr.mlff.indexing.indices import get_pbc_neighbors
    from ase import Atoms
    from ase.neighborlist import primitive_neighbor_list

    npz_data = dict(load_data('multi_sized_solids.npz'))

    cutoff = 4.
    n_test_data = 50

    nl = get_pbc_neighbors(pos=npz_data['R'][:n_test_data],
                           node_mask=npz_data['node_mask'][:n_test_data],
                           cell=npz_data['unit_cell'][:n_test_data],
                           pbc=npz_data['pbc'][:n_test_data],
                           cutoff=cutoff)

    for n, (R, msk, z, cell, pbc) in enumerate(zip(npz_data['R'][:n_test_data],
                                                   npz_data['node_mask'][:n_test_data],
                                                   npz_data['z'][:n_test_data],
                                                   npz_data['unit_cell'][:n_test_data],
                                                   npz_data['pbc'][:n_test_data])):

        atoms = Atoms(numbers=z[msk], positions=R[msk], cell=cell, pbc=pbc)
        idx_i, idx_j, s = primitive_neighbor_list('ijS',
                                                  positions=atoms.get_positions(),
                                                  pbc=atoms.get_pbc(),
                                                  cell=atoms.get_cell(),
                                                  cutoff=cutoff)

        assert compare_neighbors(nl['idx_i'][n], idx_i)
        assert compare_neighbors(nl['idx_j'][n], idx_j)

        edge_mask = nl['idx_i'][n] > -1

        assert (nl['shifts'][n][edge_mask] == s).all()
