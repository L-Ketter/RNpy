from rnpy.voxels._label_snow import _get_snow_labels
from ._label_cc import _get_cc_labels
import numpy as np

def get_snow_labels(arr, phase_id, periodic=False, pad_width=50, sigma=0.35, r_max=5):
    """
    Get the labels of the snow phase in the voxel structure using the specified parameters.

    Parameters
    ----------
    arr : np.ndarray
        The voxel structure array.
    phase_id : int
        The phase value to analyze.
    periodic : bool, optional
        Whether to consider periodic boundaries. Default is False.
    pad_width : int, optional
        The width of the padding to apply if periodic is True. Default is 50.
    sigma : float, optional
        The standard deviation for the Gaussian filter. Default is 0.35.
    r_max : int, optional
        The maximum radius for the snow labeling algorithm. Default is 5.

    Returns
    -------
    labels : np.ndarray
        An array of the same shape as the voxel structure. Labels are applied to the snow phase.
    """
    return _get_snow_labels(arr, phase_id, periodic=periodic, pad_width=pad_width, sigma=sigma, r_max=r_max)

def get_cc_labels(arr, phase_id, periodic=[False, False, False]):
    """
    Partition a given phase into its connected components.

    Parameters
    ----------
    arr : np.ndarray
        The voxel structure array.
    phase_id : int
        The phase value to analyze.
    periodic : list of bool, optional
        A list of three boolean values indicating whether periodic boundaries
        are considered in the x, y, and z directions, respectively.
        Default is [False, False, False].

    Returns
    -------
    labels : np.ndarray
        An array of the same shape as the voxel structure. Labels are applied using
        a 6-connectivity scheme in 3D. The background phase is labeled as 0,
        and the connected components of the chosen phase are labeled with consecutive
        integers starting from 1.
    """
    return _get_cc_labels(arr, phase_id, periodic=periodic)

def get_boundary_contacting_labels(labels, boundaries=['x0','xN']):
    """
    Divide labels into those that contact the specified boundaries and those that do not.
    A label is marked as contacting only if it contacts every specified boundary.
    Note: it is important to label the connected components correctly before using this function
    e.g. to study percolating connected components do not chose periodic conditions along
    the direction of percolation during labeling.

    Parameters
    ----------
    labels : np.ndarray
        An array of labeled connected components. (Background is labeled as 0.)
    boundaries : list of str, optional
        A list of boundary names to check for contact.
        Possible boundary names are 'x0', 'xN', 'y0', 'yN', 'z0', 'zN'
        Default is ['x0','xN'] (investigating percolation along the x-axis).

    Returns
    -------
    labels_conn_to_bdry : np.ndarray
        An array of the same shape as `labels`.
        Labels are applied as follows:
        0: Background voxels
        1: Non-background voxels that do not contact all specified boundaries
        2: Non-background voxels that contact all specified boundaries
    """
    bdry_faces = {
        "x0": (0, slice(None), slice(None)),
        "xN": (-1, slice(None), slice(None)),
        "y0": (slice(None), 0, slice(None)),
        "yN": (slice(None), -1, slice(None)),
        "z0": (slice(None), slice(None), 0),
        "zN": (slice(None), slice(None), -1)
    }
    unique_bdry_labels = [
        np.unique(labels[bdry_faces[bdry]]) for bdry in boundaries
    ]
    common_bdries = set(unique_bdry_labels[0])
    for bdry_labels in unique_bdry_labels[1:]:
        common_bdries &= set(bdry_labels)
    common_bdries.discard(0)
    common_bdries = np.array(list(common_bdries))
    labels_conn_to_bdry = (labels != 0).astype(int)
    labels_conn_to_bdry[np.isin(labels, common_bdries)] = 2
    return labels_conn_to_bdry
