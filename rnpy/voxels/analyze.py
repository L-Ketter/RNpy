import numpy as np
from numba import njit

def get_neighbor_ids(arr, periodic=[True, True, True]):
    """
    Get the neighbor ids for each voxel in the voxel structure.
    Note: If non-periodic boundary conditions are used, the boundary voxels get assigned a unique id that is +1
    higher than the maximum id in the voxel structure.

    Parameters
    ----------
    periodic : list of bool, optional
        A list of three boolean values indicating whether periodic boundaries
        are considered in the x, y, and z directions, respectively. Default is [True, True, True].

    Returns
    -------
    nbr_ids : np.ndarray
        An array of the same shape as the voxel structure with an additional dimension of size 6.
        The last dimension contains the neighbor ids in the order of +x, -x, +y, -y, +z, -z.
        When no periodic boundaries are considered, the neighbor ids at the boundaries will be the constant_values.
    """
    padding = tuple((1, 1) if not p else (0, 0) for p in periodic)
    slicing = tuple(slice(1, -1) if not p else slice(None) for p in periodic)
    struc = np.pad(
        arr,
        pad_width=padding,
        mode='constant',
        constant_values = arr.max()+1
    )
    shifts = [
        (-1, 0, 0), (1, 0, 0), # +x, -x
        (0, -1, 0), (0, 1, 0), # +y, -y
        (0, 0, -1), (0, 0, 1)  # +z, -z
    ]
    nbrs = [np.roll(struc, shift, axis=(0, 1, 2)) for shift in shifts]
    nbr_ids = np.stack(nbrs, axis=-1)
    return nbr_ids[slicing]

@njit
def fill_analysis_fast(arr, nbrs, num_vox, surf_area_conts):
    nx, ny, nz = arr.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                val = arr[i,j,k]
                num_vox[val] += 1
                for nbr in nbrs[i,j,k]:
                    if nbr == val:
                        continue
                    surf_area_conts[val, nbr] += 1

def analyze_phase_by_env(arr, periodic=(False, False, False)):
    """
    Analyze the voxel structure to determine the number of voxels and surface area contacts for each phase.
    Note: If non-periodic boundary conditions are used, the boundary voxels get assigned a unique id that is +1
    higher than the maximum id in the voxel structure.

    Parameters
    ----------
    arr : np.ndarray
        The voxel structure array.
    periodic : tuple of bool, optional
        A tuple of three boolean values indicating whether periodic boundaries
        are considered in the x, y, and z directions, respectively. Default is (False, False, False).

    Returns
    -------
    results : dict
        Dictionary containing analysis results for each phase. Each entry contains:
            - "num_vox": Number of voxels of the phase.
            - "num_surf": Number of surface voxels in contact with other phases.
            - "surf_ids": Array of unique surface ids in contact with the phase.
            - "surf_fracs": Array of surface area fractions for each unique surface id.
    """
    nbrs = get_neighbor_ids(arr, periodic=periodic)
    ids = np.unique(nbrs)
    num_vox = np.zeros(ids.max()+1)
    surf_area_conts = np.zeros((ids.max()+1, ids.max()+1))
    #num_vox = np.zeros_like(ids)
    fill_analysis_fast(arr, nbrs, num_vox, surf_area_conts)
    results = {}
    for id_x in ids:
        num_surf = surf_area_conts[id_x,:].sum()
        if num_surf > 0:
            surf_fracs = np.array([
                surf_area_conts[id_x, id_y] / num_surf
                for id_y in ids
            ])
        else:
            surf_fracs = np.zeros(len(ids))
        results[id_x] = {}
        results[id_x]['num_vox'] = num_vox[id_x]
        results[id_x]['num_surf'] = num_surf
        results[id_x]['surf_ids'] = ids
        results[id_x]['surf_fracs'] = surf_fracs
    return results
