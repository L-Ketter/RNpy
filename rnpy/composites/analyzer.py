
from matplotlib import axis
import numpy as np
from ._labels import get_labels
from ._neighbors import get_neighbor_ids

class VoxAnalyzer():
    def __init__(self, voxel_struc):
        self.voxel_struc = voxel_struc
        self.boundaries = ['x0', 'xN', 'y0', 'yN', 'z0', 'zN']
        self.faces  = [
            (0, slice(None), slice(None)), (-1, slice(None), slice(None)), # x0, xN
            (slice(None), 0, slice(None)), (slice(None), -1, slice(None)), # y0, yN
            (slice(None), slice(None), 0), (slice(None), slice(None), -1), # z0, zN
        ]

    def get_labels(self, phase_id, periodic=[True, True, True]):
        """
        Get the labels of the connected components of a given phase in the voxel structure.

        Parameters
        ----------
        phase_id : int
            The phase value to analyze.
        periodic : list of bool, optional
            A list of three boolean values indicating whether periodic boundaries
            are considered in the x, y, and z directions, respectively. Default is [True, True, True].

        Returns
        -------
        labels : np.ndarray
            An array of the same shape as the voxel structure. Labels are applied using
            a 6-connectivity scheme in 3D. The background phase is labeled as 0,
            and the connected components of the chosen phase are labeled with consecutive
            integers starting from 1.
        """
        return get_labels(self.voxel_struc, phase_id, periodic=periodic)

    def get_neighbor_ids(self, periodic=[True, True, True], constant_values=None):
        """
        Get the neighbor ids for each voxel in the voxel structure.

        Parameters
        ----------
        periodic : list of bool, optional
            A list of three boolean values indicating whether periodic boundaries
            are considered in the x, y, and z directions, respectively. Default is [True, True, True].
        constant_values : int or ((int,int),(int,int),(int,int)), optional
            The value to use for padding when no periodic boundaries are considered.
            Default is maximum id of the voxel structure + 1.

        Returns
        -------
        nbr_ids : np.ndarray
            An array of the same shape as the voxel structure with an additional dimension of size 6.
            The last dimension contains the neighbor ids in the order of +x, -x, +y, -y, +z, -z.
            When no periodic boundaries are considered, the neighbor ids at the boundaries will be the constant_values.
        """
        return get_neighbor_ids(self.voxel_struc, periodic, constant_values)

    def analyze_phase_by_env(
            self,
            phase_id,
            periodic=[True, True, True],
        ):
        """
        Analyze connected components of a given phase in the voxel structure with regard
        to size, surface area, and area fractions.

        Parameters
        ----------
        phase_id : int
            The phase value to analyze.
        periodic : list of bool, optional
            A list of three boolean values indicating whether periodic boundaries
            are considered in the x, y, and z directions, respectively.
            Default is [True, True, True].

        Returns
        -------
        labels : np.ndarray
            Array with the same shape as the voxel structure, with each connected component
            labeled with a unique integer.
        results : dict
            Dictionary containing analysis results (volume/voxels, surface/voxels, area fractions) for each label
        unique_surf_ids : np.ndarray
            Array of unique surface ids that are in contact with the given phase.
            If non-periodic boundaries are included, they get a unique id, which is one + max id of the voxel structure.
        """
        lbls = self.get_labels(phase_id, periodic=periodic)
        nbr_ids = self.get_neighbor_ids(periodic=periodic)
        lbls_flat = lbls.flatten()
        nbrs_flat = nbr_ids.reshape(-1, 6)
        mask = lbls_flat != 0
        mask_sum = mask.sum()
        unique_ids = np.unique(nbrs_flat)
        unique_surf_ids = unique_ids[unique_ids != phase_id]
        results = {}
        for i, (lbl, nbr_ids_lbl) in enumerate(zip(lbls_flat[mask], nbrs_flat[mask])):
            if lbl not in results:
                results[lbl] = {
                    "num_voxels": 0,
                    "num_surfaces": 0,
                    "surf_area_fracs": np.zeros_like(unique_surf_ids, dtype=float)
                }
            results[lbl]["num_voxels"] += 1
            results[lbl]["num_surfaces"] += np.sum(nbr_ids_lbl != phase_id)
            results[lbl]["surf_area_fracs"] += np.array(
                [np.sum(nbr_ids_lbl == surf_id) for surf_id in unique_surf_ids]
            )
            if i % 100000 == 0:
                print(f"\rProcessed {i}/{mask_sum} voxels", end='', flush=True)
        print()
        for lbl in results:
            results[lbl]["surf_area_fracs"] /= results[lbl]["num_surfaces"]
        return lbls, results, unique_surf_ids

    def _get_cluster_boundary_contacts(self, phase_id, periodic=[False, False, False]):
        """
        Get the labels of the connected components of a given phase in the voxel structure as well as the
        contacting ids with the boundaries.
        """
        # get labels
        print('Getting labels for phase', phase_id)
        lbls = self.get_labels(phase_id, periodic=periodic)
        unique_lbls = np.unique(lbls)
        # get neighbor ids
        print('Getting neighbor ids for phase', phase_id)
        max_id = self.voxel_struc.max()
        boundary_ids = (
            (max_id+1, max_id+2),
            (max_id+3, max_id+4),
            (max_id+5, max_id+6)
        )
        flat_boundary_ids = np.array(boundary_ids).flatten()
        nbr_ids = self.get_neighbor_ids(
            periodic=periodic,
            constant_values=boundary_ids
        )
        # get boundary contacts for each label
        print('Getting boundary contacts for phase', phase_id)
        boundary_contacts = {}
        outer_nbrs = np.concatenate([nbr_ids[face].reshape(-1,6) for face in self.faces])
        outer_lbls = np.concatenate([lbls[face].flatten() for face in self.faces])
        unique_outer_lbls = np.unique(outer_lbls)
        for i, lbl in enumerate(unique_outer_lbls):
            if lbl == 0:  # skip background (0)
                continue
            lbl_nbr_ids = set(outer_nbrs[outer_lbls == lbl].flatten())
            contacts = [b_id for b_id in flat_boundary_ids if b_id in lbl_nbr_ids]
            boundary_contacts[lbl] = contacts
            if i % 10 == 0:
                print(f"\rProcessed {i}/{unique_outer_lbls.shape[0]} labels", end='', flush=True)
        print()
        for lbl in unique_lbls:
            if lbl == 0 or lbl in boundary_contacts:
                continue
            boundary_contacts[lbl] = []  # empty contact lists for labels that do not touch any boundary
        return lbls, flat_boundary_ids, boundary_contacts

    def get_clusters_connected_to_boundary(self, phase_id, boundary='x0', periodic=[False, False, False]):
        """
        Get the clusters of a given phase that are connected to a specific boundary of the voxel structure.
        6-connectivity is used to determine connected components. A cluster is considered connected to a boundary
        if it has at least one voxel that is adjacent to the specified boundary.

        Parameters
        ----------
        phase_id : int
            The phase value to analyze.
        boundary : str, optional
            The boundary to check for connectivity. Default is 'x0'.
            boundary can be one of the following: 'x0', 'xN', 'y0', 'yN', 'z0', 'zN'.
        periodic : list of bool, optional
            A list of three boolean values indicating whether periodic boundaries
            are considered in the x, y, and z directions, respectively. Default is [False, False, False].
            The boundary in question cannot be periodic, so the corresponding value in the list must be False.

        Returns
        -------
        lbls_connected_to_boundary : np.ndarray
            An array of the same shape as the voxel structure,
            with the background labeled as 0, unconnected clusters labeled as 1, and connected clusters labeled as 2.
        """
        lbls, boundary_ids, boundary_contacts = self._get_cluster_boundary_contacts(phase_id, periodic=periodic)
        b_id = boundary_ids[self.boundaries.index(boundary)]

        connected_lbls = [lbl for lbl, contacts in boundary_contacts.items() if b_id in contacts]
        unconnected_lbls = [lbl for lbl in np.unique(lbls) if lbl != 0 and lbl not in connected_lbls]

        lbls_connected_to_boundary = np.zeros_like(lbls)
        lbls_connected_to_boundary[np.isin(lbls, connected_lbls)] = 2
        lbls_connected_to_boundary[np.isin(lbls, unconnected_lbls)] = 1
        return lbls_connected_to_boundary

    def get_percolating_clusters(self, phase_id, axis='x', periodic=[False, False, False]):
        """
        Get the percolating clusters of a given phase in the voxel structure.
        6-connectivity is used to determine connected components. A cluster is considered percolating
        if it connects the two opposite boundaries of the voxel structure along the specified axis.

        Parameters
        ----------
        phase_id : int
            The phase value to analyze.
        axis : str, optional
            The axis along which to check for percolation. Default is 'x' (x-axis).
            axis can be one of the following: 'x', 'y', 'z'.
        periodic : list of bool, optional
            A list of three boolean values indicating whether periodic boundaries.
            The axis in question cannot be periodic, so the corresponding value in the list must be False.

        Returns
        -------
        percolating_labels : np.ndarray
            An array of the same shape as the voxel structure,
            with the background labeled as 0, non-percolating clusters labeled as 1, and percolating clusters labeled as 2.
        """
        lbls, boundary_ids, boundary_contacts = self._get_cluster_boundary_contacts(phase_id, periodic=periodic )
        b_ids = [boundary_ids[i] for i, b in enumerate(self.boundaries) if b.startswith(axis)]

        percolating_labels = [lbl for lbl, contacts in boundary_contacts.items() if b_ids[0] in contacts and b_ids[1] in contacts]
        nonpercolating_labels = [lbl for lbl in np.unique(lbls) if lbl != 0 and lbl not in percolating_labels]

        lbls_percolating = np.zeros_like(lbls)
        lbls_percolating[np.isin(lbls, percolating_labels)] = 2
        lbls_percolating[np.isin(lbls, nonpercolating_labels)] = 1
        return lbls_percolating


