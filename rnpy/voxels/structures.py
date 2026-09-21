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
def _gen_blob_clusters(size, s_clust, p_blobs, seed=None):
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
    clusters = set()
    for p_blob in p_blobs:
        p_blob = (p_blob[0], p_blob[1], p_blob[2])
        clust = {p_blob}
        surf = []
        for dp in directions:
            surf.append(_step(p_blob, dp, size))
        while len(clust) < s_clust:
            p_next = surf[np.random.randint(len(surf))]
            clust.add(p_next)
            surf.remove(p_next)
            for dp in directions:
                p_surf = _step(p_next, dp, size)
                if p_surf not in clust:
                    surf.append(p_surf)
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
    p_blobs = rng.integers(0, size, size=(num_clust, 3))
    if num_clust < 1:
        warnings.warn("The number of clusters is less than 1. Switching to structures.sc_random()")
        return sc_random(size, [1-vf_disp, vf_disp])
    # generate and insert clusters
    clusters = _gen_blob_clusters(size, s_clust, p_blobs, seed=seed)
    for x, y, z in clusters:
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
    _fill_spheres(arr, sphere_positions, sphere_radii)
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
    rod_pos = np.arange(r_rod, 1-r_rod, d_space_tot)
    for x_rod in rod_pos:
        for y_rod in rod_pos:
            mask = (X-x_rod)**2 + (Y-y_rod)**2 <= r_rod**2
            arr[mask]=1
    arr = np.stack([arr]*size, axis=0)
    arr = np.rot90(arr, k=-1, axes=(0, 2))
    return arr

@njit
def _get_voronoi_dists(arr, point_lst):
    nx, ny, nz = arr.shape
    bdry_dists = np.empty((nx,ny,nz), dtype=float)
    inv_2dp = np.zeros((len(point_lst), len(point_lst)), dtype=float)
    for i, p0 in enumerate(point_lst):
        for j, p1 in enumerate(point_lst):
            if i != j:
                inv_2dp[i, j] = 1.0 / (2.0 * np.sqrt(_get_dist_sq(p0, p1, periodic=False)))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p_vox = ((i+0.5)/nx, (j+0.5)/ny, (k+0.5)/nz)
                idx0 = -1
                d0_sq = float('inf')
                for s, p_seed in enumerate(point_lst):
                    d_sq = _get_dist_sq(p_vox, p_seed, periodic=False)
                    if d_sq < d0_sq:
                        idx0 = s
                        d0_sq = d_sq
                bdry_dist = float('inf')
                for s, p_seed in enumerate(point_lst):
                    if s == idx0:
                        continue
                    d1_sq = _get_dist_sq(p_vox, p_seed, periodic=False)
                    bdry_dist_tmp = (d1_sq-d0_sq)*inv_2dp[idx0, s]
                    bdry_dist = min(bdry_dist, bdry_dist_tmp)
                arr[i, j, k] = idx0
                bdry_dists[i, j, k] = bdry_dist
    return bdry_dists, arr

def voronoi_tessellation(size, num_grains, d_bdry=0, labeled=False, seed=None):
    """
    Generate a 3D cubic array via Voronoi tessellation.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    num_grains : int
        Number of Voronoi grains.
    d_bdry : float, optional
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
    arr = np.zeros((size, size, size), dtype=int)
    rng = np.random.default_rng(seed)
    point_lst = rng.random((num_grains, 3))
    vor_dists, arr = _get_voronoi_dists(arr, point_lst)
    labels = vor_dists < d_bdry/2
    if labeled:
        arr[labels] = num_grains
    else:
        arr = labels.astype(int)
    return arr

