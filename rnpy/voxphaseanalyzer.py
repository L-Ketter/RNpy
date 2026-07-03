
import numpy as np
import porespy as ps
import networkx  as nx
import numpy as np
import os
from skimage.measure import label

class VoxPhaseAnalyzer():
    def __init__(self, voxel_struc):
        self.voxel_struc = voxel_struc

    def get_labels(self, phase_id, connectivity=1, periodic=True):
        """
        Get the labels of the connected components of a given phase in the voxel structure.

        Parameters
        ----------
        phase_id : int
            The phase value to analyze.
        connectivity : int, optional
            The connectivity for labeling. Default is 1 (6-connectivity in 3D).
        periodic : bool, optional
            If True, periodic boundaries are considered when labeling. Default is True.

        Returns
        -------
        labels : np.ndarray
            An array of the same shape as the voxel structure.
            The background phase is labeled as 0, and the connected components
            of the chosen phase are labeled with consecutive integers starting from 1.
        """
        labels = label(self.voxel_struc == phase_id, connectivity=connectivity)
        if periodic:
            return self._wrap_labels(labels)
        else:
            return labels

    def _wrap_labels(self, labels):
        """
        Wrap labels to account for periodic boundaries.
        """
        def _get_boundary_pairs(labels):
            """
            Find unique pairs of phase labels that are connected across
            the periodic boundaries of the voxel structure.
            """
            pairs = set()
            faces = [
                (labels[0, :, :], labels[-1, :, :]),  # x0, xN
                (labels[:, 0, :], labels[:, -1, :]),  # y0, yN
                (labels[:, :, 0], labels[:, :, -1]),  # z0, zN
            ]
            for slice_0, slice_n in faces:
                mask = (slice_0 > 0) & (slice_n > 0)
                pairs.update(zip(slice_0[mask], slice_n[mask]))
            return list(pairs)

        def _group_connected_labels(pairs):
            """
            Find connected labels in a list of pairs and group them together.
            Taken from here:
            https://stackoverflow.com/questions/53886120/combine-lists-with-common-elements
            Combines e.g. [[1,2],[2,4],[5,8]] to [[1,2,4], [5,8]]
            """
            graph = nx.Graph()
            graph.add_edges_from(pairs)
            groups = list(nx.connected_components(graph))
            return [sorted(list(group)) for group in groups]

        boundary_pairs = _get_boundary_pairs(labels)  # get boundary pairs from phase labels
        connected_label_groups = _group_connected_labels(boundary_pairs)  # group connected boundary labels together

        # Combine grouped labels into a single new label (can lead to nonsequential labels)
        periodic_labels = labels.copy()
        for new_label, group in enumerate(connected_label_groups, start=labels.max()+1):
            for old_label in group:
                periodic_labels[labels == old_label] = new_label
        # Reindex all labels to consecutive integers (0, 1, 2, ...)
        unique_labels = np.unique(periodic_labels)
        lookup = np.zeros(unique_labels.max() + 1, dtype=int)
        lookup[unique_labels] = np.arange(len(unique_labels))
        reindexed_labels = lookup[periodic_labels]
        return reindexed_labels

    def _get_neighbor_ids(self, periodic=True, constant_values=None):
        """
        Get the neighbor ids for each voxel of a given phase in the voxel structure.
        When periodic boundaries are considered, the neighbor ids will wrap around the
        edges of the voxel structure. The neighbor ids are returned in the order of
        +x, -x, +y, -y, +z, -z.
        When periodic boundaries are not considered, the neighbor ids will be padded with
        numbers increasing from the maximum id of the voxel structure.
        Boundary at x0 gets max_id+1, boundary at xN gets max_id+2, boundary at y0 gets max_id+3, ...
        """
        if periodic:
            struc = self.voxel_struc
        else:
            max_id = self.voxel_struc.max()
            struc = np.pad(
                self.voxel_struc,
                pad_width=((1,1),(1,1),(1,1)),
                mode='constant',
                constant_values= max_id+1 if not constant_values else constant_values
            )
        shifts = [
            (1, 0, 0), (-1, 0, 0), # +x, -x
            (0, 1, 0), (0, -1, 0), # +y, -y
            (0, 0, 1), (0, 0, -1)  # +z, -z
        ]
        neighbors = [np.roll(struc, shift, axis=(0, 1, 2)) for shift in shifts]
        neighbor_ids = np.stack(neighbors, axis=-1)
        return neighbor_ids if periodic else neighbor_ids[1:-1, 1:-1, 1:-1]

    def get_phase_sorted_by_neighbors(self, phase_id, connectivity=1,return_keylst=False, periodic=True):
        """
        Sort connected voxels of a phase by their neighbor environment.

        Parameters
        ----------
        phase_id : int
            The phase value to analyze.

        Returns
        -------
        sorted_phase : np.ndarray
            Array with voxels labeled by their neighbor environment.
        """
        labels = self.get_labels(phase_id, periodic=periodic, connectivity=connectivity)
        unique_labels = np.unique(labels)
        neighbor_ids = self._get_neighbor_ids(periodic=periodic, constant_values=None)

        contacts = {}
        sorted_labels = np.zeros_like(labels)
        for val in unique_labels[1:]:  # skip background (0)
            label_neighbor_ids = set(neighbor_ids[labels == val].ravel())
            key = frozenset(label_neighbor_ids)
            contacts[key] = contacts.get(key, []) + [val]

        keylst = sorted(contacts.keys(), key=lambda x: (len(x), x))
        for i, key in enumerate(keylst, start=1):
            for val in contacts[key]:
                sorted_labels[labels == val] = i

        if return_keylst:
            return sorted_labels, keylst
        return sorted_labels

    def get_percolating_clusters(self, phase_id, axis=0, connectivity=1):
        """
        Get the percolating clusters of a given phase in the voxel structure.

        Parameters
        ----------
        phase_id : int
            The phase value to analyze.
        axis : int, optional
            The axis along which to check for percolation. Default is 0 (x-axis).
        connectivity : int, optional
            The connectivity for labeling. Default is 1 (6-connectivity in 3D).

        Returns
        -------
        percolating_labels : np.ndarray
            An array of the same shape as the voxel structure,
            with non-percolating clusters labeled as 0 and percolating clusters labeled with their original labels.
        """
        labels = self.get_labels(phase_id, periodic=False, connectivity=connectivity)
        unique_labels = np.unique(labels)

        max_id = self.voxel_struc.max()
        boundary_ids = tuple(
                (max_id+i, max_id+i+1)
                for i in range(1, 7, 2)
            )
        neighbor_ids = self._get_neighbor_ids(periodic=False, constant_values=boundary_ids)
        ax_id0, ax_idN = boundary_ids[axis]
        percolating_labels = np.zeros_like(labels)
        for val in unique_labels[1:]:  # skip background (0)
            label_neighbor_ids = set(neighbor_ids[labels == val].ravel())
            if (
                ax_id0 in label_neighbor_ids and
                ax_idN in label_neighbor_ids
            ):
                percolating_labels[labels == val] = 2
            else:
                percolating_labels[labels == val] = 1
        return percolating_labels



