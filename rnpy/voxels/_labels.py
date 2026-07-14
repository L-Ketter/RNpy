import numpy as np
import networkx  as nx
from skimage.measure import label

def get_labels(
        voxel_struc,
        phase_id,
        periodic=[True, True, True],
        ):
    """
    Get the labels of the connected components of a given phase in the voxel structure.
    """
    labels = label(voxel_struc == phase_id, connectivity=1)
    if not any(periodic):
        return labels
    else:
        return _wrap_labels(labels, periodic)

def _wrap_labels(labels, periodic=[True, True, True]):
    """
    Wrap labels to account for periodic boundaries.
    """
    def _get_boundary_pairs(labels, periodic=[True, True, True]):
        """
        Find pairs of phase labels that are connected across
        the periodic boundaries of the voxel structure.
        """
        pairs = set()
        lbl_faces = [
            (labels[0,:,:], labels[-1,:,:]),  # x0, xN
            (labels[:,0,:], labels[:,-1,:]),  # y0, yN
            (labels[:,:,0], labels[:,:,-1])   # z0, zN
        ]
        for i, (slice_0, slice_n) in enumerate(lbl_faces):
            if not periodic[i]:
                continue
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

    boundary_pairs = _get_boundary_pairs(labels, periodic)  # get boundary pairs from phase labels
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