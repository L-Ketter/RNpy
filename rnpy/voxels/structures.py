import numpy as np
from numba import njit
import warnings
from .analyzer import VoxAnalyzer

@njit
def _get_dist_sq(p0, p1, periodic=True):
    dpx = abs(p1[0]-p0[0])
    dpy = abs(p1[1]-p0[1])
    dpz = abs(p1[2]-p0[2])
    if periodic:
        if dpx>0.5:
            dpx=1.0-dpx
        if dpy>0.5:
            dpy=1.0-dpy
        if dpz>0.5:
            dpz=1.0-dpz
    return dpx**2 + dpy**2 + dpz**2

@njit
def _fill_blobs(arr, size, s_blob, p_blobs, seed=None):
    if seed is not None:
        np.random.seed(seed)
    def _step(p,dp,size):
        p_next = (
            (p[0] + dp[0]) % size,
            (p[1] + dp[1]) % size,
            (p[2] + dp[2]) % size
        )
        return p_next
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    blobs = set()
    for p_blob in p_blobs:
        p_blob = (p_blob[0], p_blob[1], p_blob[2])
        blob = {p_blob}
        surf = []
        for dp in directions:
            surf.append(_step(p_blob, dp, size))
        while len(blob) < s_blob:
            p_next = surf[np.random.randint(len(surf))]
            blob.add(p_next)
            surf.remove(p_next)
            for dp in directions:
                p_surf = _step(p_next, dp, size)
                if p_surf not in blob:
                    surf.append(p_surf)
        blobs.update(blob)
    for x, y, z in list(blobs):
        arr[x, y, z] = 1
    return arr

def _interpret_vf(vfs):
    if len(vfs) > 1 and (np.sum(vfs) != 1 and np.sum(vfs) != 100):
       raise ValueError("The sum of volume fractions must equal 1 or 100.")
    if any(vf > 1 for vf in vfs):
        warnings.warn("Volume fractions > 1 detected. Interpreting them as percentages. Please use 0-1 range in the future.")
        vfs = [vf/100 for vf in vfs]
    return vfs

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
    vf_disp = _interpret_vf([vf_disp])[0]
    rng = np.random.default_rng(seed)
    s_blob = int(4/3*np.pi*(r_mean*size)**3)  # approximate number of voxels per cluster based on a mean radius
    n_vox_disp = int(size**3*vf_disp)
    n_blobs = n_vox_disp//s_blob
    p_blobs = rng.integers(0, size, size=(n_blobs, 3))
    if n_blobs < 1:
        warnings.warn("The number of clusters is less than 1. Switching to structures.sc_random()")
        return sc_random(size, [1-vf_disp, vf_disp])
    arr = np.zeros((size, size, size), dtype=int)
    arr = _fill_blobs(arr, size, s_blob, p_blobs, seed=seed)
    # add voxels to reach desired volume fraction
    if fill_random or fill_attach:
        missing = n_vox_disp - (arr==1).sum()
        if fill_random:
            idx_zeros = np.argwhere(arr==0)
            idx_lst = rng.choice(
                np.arange(len(idx_zeros)),
                size=missing,
                replace=False
            )
            idx_chosen = idx_zeros[idx_lst]
            for x, y, z in idx_chosen:
                arr[x, y, z] = 1
        elif fill_attach:
            inserted = 0
            while inserted < missing:
                vx = VoxAnalyzer(arr)
                nbr_ids_sum = np.sum(vx.get_neighbor_ids(), axis=-1)
                idx_zeros = np.argwhere(arr==0)
                idx_chosen = [idx for idx in idx_zeros if nbr_ids_sum[tuple(idx)] > 0]
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

def stack_parallel(size, vfs):
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
    arr = stack_series(size, vfs)
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
    ordered = stack_series(size, vfs).flatten()
    shuffled = rng.permutation(ordered)
    arr = shuffled.reshape((size, size, size))
    return arr

def stack_series(size, vfs):
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
    vfs = _interpret_vf(vfs)
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
def _fill_spheres(arr, p_spheres, radii):
    nx, ny, nz = arr.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p_vox = ((i+0.5)/nx, (j+0.5)/ny, (k+0.5)/nz)
                for s, radius in enumerate(radii):
                    dist_sq = _get_dist_sq(p_vox, p_spheres[s], periodic=True)
                    if dist_sq <= radius**2:
                        arr[i,j,k] = 1
                        break

def random_spheres(size, n_spheres, r_range, seed=None):
    """
    Generate a 3D cubic array with randomly placed overlapping spheres.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    n_spheres : int
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
    p_spheres = rng.random((n_spheres, 3))
    r_spheres = r_range[0] + (r_range[1]-r_range[0]) * (rng.random(n_spheres))
    arr = np.zeros((size, size, size), dtype=int)
    _fill_spheres(arr, p_spheres, r_spheres)
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
    arr = np.zeros((size, size, size), dtype=int)
    d_space_tot = 2*r_rod + d_space # distance between rod centers
    lin = np.linspace(0.5/size, 1.0-0.5/size, size)
    X, Y = np.meshgrid(lin, lin, indexing="ij")
    p_rods = np.arange(r_rod, 1-r_rod, d_space_tot)
    for x_rod in p_rods:
        for y_rod in p_rods:
            mask = (X-x_rod)**2 + (Y-y_rod)**2 <= r_rod**2
            arr[mask]=1
    arr = np.stack([arr]*size, axis=0)
    arr = np.rot90(arr, k=-1, axes=(0, 2))
    return arr

@njit
def _fill_voronoi(arr, p_grains):
    nx, ny, nz = arr.shape
    d_bdries = np.empty((nx,ny,nz), dtype=float)
    inv_2dp = np.zeros((len(p_grains), len(p_grains)), dtype=float)
    for i, p0 in enumerate(p_grains):
        for j, p1 in enumerate(p_grains):
            if i != j:
                inv_2dp[i, j] = 1.0 / (2.0 * np.sqrt(_get_dist_sq(p0, p1, periodic=False)))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # get position of the voxel center
                p_vox = ((i+0.5)/nx, (j+0.5)/ny, (k+0.5)/nz)
                # get id of the nearest grain
                id_0 = -1
                d0_sq = float('inf')
                for id_x, p_grain in enumerate(p_grains):
                    d_sq = _get_dist_sq(p_vox, p_grain, periodic=False)
                    if d_sq < d0_sq:
                        id_0 = id_x
                        d0_sq = d_sq
                arr[i, j, k] = id_0
                # get dist between vox and nearest grain boundary
                d_bdry = float('inf')
                for id_1, p_grain in enumerate(p_grains):
                    if id_1 == id_0:
                        continue
                    d1_sq = _get_dist_sq(p_vox, p_grain, periodic=False)
                    d_tmp = (d1_sq-d0_sq)*inv_2dp[id_0, id_1]
                    d_bdry = min(d_bdry, d_tmp)
                d_bdries[i, j, k] = d_bdry
    return d_bdries, arr

def voronoi_tessellation(size, n_grains, l_bdry=0, labeled=False, seed=None):
    """
    Generate a 3D cubic array via Voronoi tessellation.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    n_grains : int
        Number of Voronoi grains.
    l_bdry : float, optional
        boundary thickness in normalized units (default is 0).
    labeled : bool, optional
        If True, each grain and the grain boundary is labeled with a unique integer.
        If False, boundary voxels are represented by 1s and the grains by 0s.
        default is False.
    seed : int, optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    np.ndarray
        A 3D cubic array representing the Voronoi tessellation..
    """
    rng = np.random.default_rng(seed)
    p_grains = rng.random((n_grains, 3))
    arr = np.zeros((size, size, size), dtype=int)
    d_bdries, arr = _fill_voronoi(arr, p_grains)
    bdries = d_bdries < l_bdry/2
    if labeled:
        arr[bdries] = n_grains
    else:
        arr = bdries.astype(int)
    return arr

