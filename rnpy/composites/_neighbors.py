import numpy as np

def get_neighbor_ids(voxel_struc, periodic=[True, True, True], constant_values=None):
        """
        Get the neighbor ids for each voxel in the voxel structure.
        """
        padding = tuple((1, 1) if not p else (0, 0) for p in periodic)
        slicing = tuple(slice(1, -1) if not p else slice(None) for p in periodic)

        struc = np.pad(
            voxel_struc,
            pad_width=padding,
            mode='constant',
            constant_values= voxel_struc.max()+1 if constant_values is None else constant_values
        )

        shifts = [
            (-1, 0, 0), (1, 0, 0), # +x, -x
            (0, -1, 0), (0, 1, 0), # +y, -y
            (0, 0, -1), (0, 0, 1)  # +z, -z
        ]
        nbrs = [np.roll(struc, shift, axis=(0, 1, 2)) for shift in shifts]
        nbr_ids = np.stack(nbrs, axis=-1)
        return nbr_ids[slicing]