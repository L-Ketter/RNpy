import numpy as np

def blobs3D(size, disp_volfrac, sclust, fill_wo_clustering=True):
    """
    Generates a 3D cubic array representing a binary composite with dispersed phase and inclusions of size sclust.
    The dispersed phase is represented by 1s and the continuous phase by 0s.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    disp_volfrac : float
        Volume fraction of the dispersed phase in percent (0-100).
    sclust : int
        Number of voxels per cluster of the dispersed phase inclusions.
    fill_wo_clustering : bool, optional
        If True, remaining voxels to reach the desired volume fraction will be filled
        randomly without clustering.
        Default is True.

    Returns
    -------
    np.ndarray
        A 3D cubic array with the dispersed phase inclusions represented by 1s and the continuous phase by 0s.
    """
    if disp_volfrac == 100 or disp_volfrac == 0 or sclust == 1:
        array = sc_random(size, volfracs=[100-disp_volfrac, disp_volfrac], vf_exakt=True)
    else:
        array = np.zeros((size, size, size), dtype=int)
        disp_voxel_num = size**3 * disp_volfrac / 100
        clust_num = int(disp_voxel_num/sclust)
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        # generate and insert clusters
        for num in range(clust_num):
            cluster = []
            start_pos = np.random.choice(size, 3)
            cluster.append(tuple(start_pos))
            while len(cluster) < sclust: # iteratively generate a clusters of connected positions
                chosen_direc = directions[np.random.choice(len(directions))]
                chosen_pos = cluster[np.random.choice(len(cluster))]
                newpos = list(x + y for x, y in zip(chosen_pos, chosen_direc))
                newpos = tuple(coord % size for coord in newpos) # ensure periodic boundaries
                if newpos not in cluster:
                    cluster.append(newpos)
            for x,y,z in cluster: # insert a cluster into the array
                array[x,y,z] = 1
        # add voxels to reach desired volume fraction
        if fill_wo_clustering:
            missing = disp_voxel_num - (array==1).sum()
            count = 0
            while count < missing:
                x,y,z = np.random.choice(size, 3)
                if array[x,y,z] == 0:
                    array[x,y,z] = 1
                    count+=1
    return array

def parallel_connected(size, volfracs):
    """
    Generates a 3D cubic array representing a parallel connected composite. (phases stacked along the z-axis))
    The number of voxels in each phase is determined by the volume fractions provided.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    volfracs : list of float
        List of volume fractions for each phase in percent (0-100). The sum of the
        volume fractions should equal 100.

    Returns
    -------
    np.ndarray
        A 3D cubic array with phases stacked along the z-axis.
    """
    array = series_connected(size, volfracs)
    array = np.rot90(array, k=1, axes=(0,2))
    return array

def sc_random(size, volfracs, seed=None, vf_exakt=False):
    """
    Generates a 3D cubic array representing a random composite with specified volume fractions.
    The number of voxels in each phase is determined by the volume fractions provided.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    volfracs : list of float
        List of volume fractions for each phase in percent (0-100). The sum of the
        volume fractions should equal 100.
    seed : int, optional
        Random seed to use for reproducibility. Default is None.
    vf_exakt : bool, optional
        If True, the volume fractions will be exactly matched by shuffling a pre-ordered array.
        Else, the array will be constructed by thresholding random noise, which may lead to slight deviations from the specified volume fractions.
        Default is False.

    Returns
    -------
    np.ndarray
        A 3D cubic array with phases randomly distributed.
    """
    rng = np.random.default_rng(seed)
    if vf_exakt:
        ordered = series_connected(size, volfracs).flatten()
        shuffled = rng.permutation(ordered)
        array = shuffled.reshape((size, size, size))
    else:
        random_noise = rng.random((size, size, size))
        array = np.zeros((size, size, size), dtype=int)
        thres_lo = 0
        for i, vf in enumerate(volfracs):
            thres_up = thres_lo + vf/100
            array[(random_noise >= thres_lo) & (random_noise < thres_up)] = i
            thres_lo = thres_up
    return array

def series_connected(size, volfracs):
    """
    Generates a 3D cubic array representing a series connected composite. (phases stacked along the x-axis))
    The number of voxels in each phase is determined by the volume fractions provided.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    volfracs : list of float
        List of volume fractions for each phase in percent (0-100). The sum of the
        volume fractions should equal 100.

    Returns
    -------
    np.ndarray
        A 3D cubic array with phases stacked along the x-axis.

    Raises
    ------
    ValueError
        If the sum of the volume fractions does not equal 100.
    """
    if np.sum(volfracs) != 100:
       raise ValueError("The sum of volume fractions must equal 100.")
    vox_nums = (size**3 * np.array(volfracs)/100).astype(int)
    array_1D = []
    for i, count in enumerate(vox_nums):
        array_1D += [i] * count
    array_1D = np.array(array_1D).flatten()
    array = array_1D.reshape((size, size, size))
    return array











