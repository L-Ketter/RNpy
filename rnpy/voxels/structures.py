import numpy as np
from numba import njit
import warnings
from .analyzer import VoxAnalyzer

@njit
def _blobs_gen_clusters(size, s_clust, num_clust, seed):
    if seed is not None:
        np.random.seed(seed)
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    clusters = set()
    for i in range(num_clust):
        start_pos = (
            np.random.randint(size),
            np.random.randint(size),
            np.random.randint(size)
        )
        clust = {start_pos}
        surf = []
        for dpos in directions:
            neigh = (
                (start_pos[0] + dpos[0]) % size,
                (start_pos[1] + dpos[1]) % size,
                (start_pos[2] + dpos[2]) % size
            )
            surf.append(neigh)
        while len(clust) < s_clust: # iteratively generate a clusters of connected positions
            new_pos = surf[np.random.randint(0, len(surf)-1)]
            surf.remove(new_pos)
            for d in directions:
                neigh = (
                    (new_pos[0] + d[0]) % size,
                    (new_pos[1] + d[1]) % size,
                    (new_pos[2] + d[2]) % size
                )
                if neigh not in clust:
                    surf.append(neigh)
            clust.add(new_pos)
        clusters.update(clust)
    return list(clusters)

def blobs(size, vf_disp, r_mean, fill_random=False, fill_attach=True, seed=None):
    """
    Generates a 3D cubic array representing a binary composite with dispersed phase and inclusions of size sclust.
    The dispersed phase is represented by 1s and the continuous phase by 0s.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    vf_disp : float
        Volume fraction of the dispersed phase in (0-1).
    r_mean : float
        Mean radius of the dispersed phase inclusions in normalized units (0-1, relative to the cube size).
    fill_random : bool, optional
        If True, remaining voxels to reach the desired volume fraction will be filled
        randomly without clustering.
        Default is False.
    fill_attach : bool, optional
        If True, remaining voxels to reach the desired volume fraction will be filled
        by attaching to existing clusters.
        Default is True.

    Returns
    -------
    np.ndarray
        A 3D cubic array with the dispersed phase inclusions represented by 1s and the continuous phase by 0s.
    """
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size, size), dtype=int)
    s_clust = int(4/3*np.pi*(r_mean*size)**3)  # approximate number of voxels per cluster based on a mean radius
    vox_num_disp = int(arr.size*vf_disp)
    num_clust = vox_num_disp//s_clust
    if num_clust < 1:
        warnings.warn("The number of clusters is less than 1. Switching to structures.sc_random()")
        return sc_random(size, [1-vf_disp, vf_disp])
    # generate and insert clusters
    clusters = _blobs_gen_clusters(size, s_clust, num_clust, seed)
    for x, y, z in list(clusters):
        arr[x, y, z] = 1
    # add voxels to reach desired volume fraction
    if fill_random or fill_attach:
        missing = vox_num_disp-(arr==1).sum()
        if fill_random:
            idx_zeros = np.argwhere(arr==0)
            idx_lst = rng.choice(
               np.arange(len(idx_zeros)),
                size=missing,
                replace=False
            )
            idx_chosen = idx_zeros
            for idx in idx_lst:
                x, y, z = idx_chosen[idx]
                arr[x, y, z] = 1
        elif fill_attach:
            inserted = 0
            while inserted < missing:
                vx = VoxAnalyzer(arr)
                nbr_ids = np.sum(vx.get_neighbor_ids(), axis=-1)
                idx_zeros = np.argwhere(arr==0)
                idx_chosen = [idx for idx in idx_zeros if nbr_ids[tuple(idx)] > 0]
                if len(idx_chosen) < (missing-inserted):
                    idx_lst = np.arange(len(idx_chosen))
                else:
                    idx_lst = rng.choice(
                        np.arange(len(idx_chosen)),
                        size=(missing-inserted),
                        replace=False
                    )
                for idx in idx_lst:
                    x, y, z = idx_chosen[idx]
                    arr[x, y, z] = 1
                inserted += len(idx_chosen)
    return arr

def parallel_connected(size, vfs):
    """
    Generates a 3D cubic array representing a parallel connected composite. (phases stacked along the z-axis))
    The number of voxels in each phase is determined by the volume fractions provided.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    vfs : list of float
        List of volume fractions for each phase in (0-1). The sum of the
        volume fractions should equal 1.

    Returns
    -------
    np.ndarray
        A 3D cubic array with phases stacked along the z-axis.
    """
    arr = series_connected(size, vfs)
    arr = np.rot90(arr, k=1, axes=(0,2))
    return arr

def sc_random(size, vfs, seed=None):
    """
    Generates a 3D cubic array representing a random composite with specified volume fractions.
    The number of voxels in each phase is determined by the volume fractions provided.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    vfs : list of float
        List of volume fractions for each phase in (0-1). The sum of the
        volume fractions should equal 1.
    seed : int, optional
        Random seed to use for reproducibility. Default is None.

    Returns
    -------
    np.ndarray
        A 3D cubic array with phases randomly distributed.
    """
    rng = np.random.default_rng(seed)
    ordered = series_connected(size, vfs).flatten()
    shuffled = rng.permutation(ordered)
    arr = shuffled.reshape((size, size, size))
    return arr

def series_connected(size, vfs):
    """
    Generates a 3D cubic array representing a series connected composite. (phases stacked along the x-axis))
    The number of voxels in each phase is determined by the volume fractions provided.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    vfs : list of float
        List of volume fractions for each phase in (0-1). The sum of the
        volume fractions should equal 1.

    Raises
    ------
    ValueError
        If the sum of the volume fractions does not equal 1.

    Returns
    -------
    np.ndarray
        A 3D cubic array with phases stacked along the x-axis.
    """
    if np.sum(vfs) != 1:
       raise ValueError("The sum of volume fractions must equal 1.")
    vox_tot = size**3
    vox_nums = np.round(vox_tot * np.array(vfs)).astype(int)
    off = vox_tot-np.sum(vox_nums)
    if off != 0:
        vox_nums[np.argmax(vox_nums)] += off
    id_lst = []
    for i, count in enumerate(vox_nums):
        id_lst += [i] * count
    id_lst = np.array(id_lst).flatten()
    arr = id_lst.reshape((size, size, size))
    return arr

@njit
def fill_spheres(arr, positions, radii):
    nx, ny, nz = arr.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = (i+0.5)/nx
                y = (j+0.5)/ny
                z = (k+0.5)/nz
                for s in range(len(radii)):
                    dx = abs(x-positions[s,0])
                    dy = abs(y-positions[s,1])
                    dz = abs(z-positions[s,2])
                    if dx>0.5:
                        dx=1.0-dx
                    if dy>0.5:
                        dy=1.0-dy
                    if dz>0.5:
                        dz=1.0-dz
                    if dx**2 + dy**2 + dz**2 <= radii[s]**2:
                        arr[i,j,k] = 1
                        break

def random_spheres(size, num_spheres, r_range, seed=None):
    """
    Generate a 3D cubic array with randomly placed overlapping spheres.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    num_spheres : int
        Number of spheres to place in the array.
    r_range : list
        Minimum and maximum radius of the spheres in normalized units (0-1).
        Radii are uniformly sampled within this range.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        A 3D cubic array with randomly placed spheres.
        Spheres are represented by 1s and the background by 0s.
    """
    rng = np.random.default_rng(seed)
    # generate sphere_positions and radii in a unit cube
    sphere_positions = rng.random((num_spheres, 3))
    sphere_radii = r_range[0] + (r_range[1]-r_range[0]) * (rng.random(num_spheres))
    arr = np.zeros((size, size, size)).astype(int)
    fill_spheres(arr, sphere_positions, sphere_radii)
    return arr

def ordered_rods(size, r_rod, d_space):
    """
    Generate a 3D cubic array with ordered rods along the z-axis.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    r_rod : float
        Radius of the rods in normalized units (0-1).
    d_space : float
        Distance between the surfaces of adjacent rods in normalized units (0-1).

    Returns
    -------
    np.ndarray
        A 3D cubic array with ordered rods.
        Rods are represented by 1s and the background by 0s.
    """
    arr = np.zeros((size, size), dtype=int)
    d_space_tot = 2*r_rod + d_space # distance between rod centers
    lin = np.linspace(0.5/size, 1.0-0.5/size, size)
    X, Y = np.meshgrid(lin, lin, indexing="ij")
    rod_pos = np.arange(r_rod, 1-r_rod, d_space_tot)
    for x_rod in rod_pos:
        for y_rod in rod_pos:
            mask = (X-x_rod)**2 + (Y-y_rod)**2 <= r_rod**2
            arr[mask]=1
    arr = np.stack([arr]*size, axis=0)
    arr = np.rot90(arr, k=-1, axes=(0, 2))
    return arr