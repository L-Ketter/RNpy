import os
import numpy as np
from .network import Network

class BuildContext():
    def __init__(self,
                 voxel_struc,
                 phase_conds,
                 dtype=None,
                 use_gpu=False,
                 int_bounds=None,
                 mat_bounds=None,
                 u_start=None,
                 **kwargs):

        self.dtype = dtype
        self.use_gpu = use_gpu
        self.xp = Network._choose_array_backend(use_gpu)
        self.voxel_struc = Network._shuttle_to_cpu(voxel_struc)
        self.phase_conds = phase_conds
        self.int_bounds = int_bounds
        self.mat_bounds = mat_bounds

        self.u_start_center = u_start # absolute path to the initial temperature array file
        self.u_x0 = kwargs.get('u_x0', 1)
        self.u_xN = kwargs.get('u_xN', 0)
        self.L_x = kwargs.get('L_x', 1)
        self.dx = self.L_x/voxel_struc.shape[0]

        if (mat_bounds is not None) != (int_bounds is not None):
            raise ValueError("To consider interfaces, both 'mat_bounds' and 'int_bounds'"
                             " must be provided together. Only one was provided.")

        elif (mat_bounds is not None) and (int_bounds is not None):
            self.mat_bounds = np.array(mat_bounds, dtype=self.dtype)
            self.int_bounds = np.array(int_bounds, dtype=self.dtype)
            if len(int_bounds) != len(self.phase_conds):
                raise ValueError("'int_bounds' length must match the number "
                                 "of phases in 'phase_conds'.")
            if self.mat_bounds.shape != (len(self.phase_conds), len(self.phase_conds)):
                raise ValueError("'mat_bounds' must be a square matrix with dimensions "
                                 "equal to the number of phases in 'phase_conds'.")

        if len(np.unique(self.voxel_struc)) != len(self.phase_conds):
            raise ValueError("'phase_conds' length must match the number "
                             "of unique phases in 'voxel_struc'.")
        if not np.array_equal(np.unique(self.voxel_struc), np.arange(len(self.phase_conds))):
            raise ValueError(
                "'voxel_struc' must contain phase indices starting from 0 up to "
                "the number of phases minus one, without gaps."
            )

class NetworkBuilder:
    k_gen_slices = {
            'x_0': (slice(None, -1), slice(1, -1), slice(1, -1)),
            'x_1': (slice(1, None), slice(1, -1), slice(1, -1)),
            'y_0': (slice(1, -1), slice(None, -1), slice(1, -1)),
            'y_1': (slice(1, -1), slice(1, None), slice(1, -1)),
            'z_0': (slice(1, -1), slice(1, -1), slice(None, -1)),
            'z_1': (slice(1, -1), slice(1, -1), slice(1, None))
    }

    def _get_k_raw_arr(self):
        """
        Builds a 3D conductivity array with voxel conductivities
        of the pure phases and pads boundaries.

        Returns
        -------
        k_raw_arr : xp.ndarray
            3D array of conductivities with padded boundary values.
            - The corresponding conductivities from `phase_conds` are assigned
              for entries in `voxel_struc` matching the index in `phase_conds`
            - boundary planes are padded:
              * +x and -x faces set to `inf` (hotplates)
              * +y, -y, +z, -z faces set to 0 (adiabatic boundaries)
        """
        k_raw_arr = self.ctx.xp.zeros(self.ctx.voxel_struc.shape, dtype=self.ctx.dtype)
        for i, cond in enumerate(self.ctx.phase_conds):
            k_raw_arr[self.ctx.voxel_struc == i] = cond
        k_raw_arr = self.ctx.xp.pad(
            k_raw_arr,
            pad_width = ((1,1), (1,1), (1,1)),
            mode = 'constant',
            constant_values = ((float('inf'), float('inf')), (0,0), (0,0))
        )
        return k_raw_arr

    def _get_k_arrs(self, k_raw_arr, R_ints=None):
        """
        Builds the conductivity arrays for the simulation.
        If `mat_bounds` and `int_bounds` are provided, the interface resistances
        are calculated using the _get_R_int_arrs method and used in the bond
        conductivity calculation. Along each axis, the bond conductivities are
        calculated using the internal _calc_bond_cond method.
        Then the sum of all bond conductivities is calculated and stored in `k_sum_arr`.

        Parameters
        ----------
        k_raw_arr : xp.ndarray
            3D array of conductivities with padded boundary values.
        R_ints : dict, optional
            Dictionary with interface resistance arrays along each axis.

        Returns
        -------
        k_sum_arr : xp.ndarray
            3D array of the sum of bond conductivities for each voxel.
        k_dir_arrs : dict
            Dictionary with bond conductivity arrays along each axis.
        """

        def _calc_bond_cond(k_pos0, k_pos1, R_int=None):
            """
            Calculates the bond conductivities between two neighboring voxels.

            Parameters
            ----------
            k_pos0 : xp.ndarray
                Conductivities of voxels at position 0.
            k_pos1 : xp.ndarray
                Conductivities of voxels at position 1.
            R_int : xp.ndarray, optional
                Interface resistances between the voxels.

            Returns
            -------
            k_link : xp.ndarray
                Conductivities of the bonds between the voxels.
            """
            zero_mask = (k_pos0 == 0) | (k_pos1 == 0) | (R_int == float('inf'))
            inf_mask = (k_pos0 == float('inf')) & (k_pos1 == float('inf'))
            other_mask = ~(zero_mask) & ~(inf_mask)

            k_link = self.ctx.xp.zeros(k_pos0.shape, dtype=self.ctx.dtype)
            k_link[inf_mask] = float('inf')

            k_series_inv = 0.5*(1/k_pos0[other_mask] + 1/k_pos1[other_mask])
            if R_int is not None:
                k_link[other_mask] = (k_series_inv + R_int[other_mask]/self.ctx.dx)**-1
            else:
                k_link[other_mask] = (k_series_inv)**-1
            del zero_mask, inf_mask, other_mask
            return k_link

        k_dir_arrs = {}
        for ax in Network.axes:
            k_pos0 = k_raw_arr[self.k_gen_slices[f'{ax}_0']]
            k_pos1 = k_raw_arr[self.k_gen_slices[f'{ax}_1']]
            if self.ctx.mat_bounds is not None and self.ctx.int_bounds is not None:
                k_dir_arrs[ax] = _calc_bond_cond(k_pos0,k_pos1,R_ints[ax])
            else:
                k_dir_arrs[ax] = _calc_bond_cond(k_pos0,k_pos1)
            del k_pos0, k_pos1

        k_sum_arr = self.ctx.xp.zeros(self.ctx.voxel_struc.shape, dtype=self.ctx.dtype)
        for ax in Network.axes:
            k_sum_arr += (
                k_dir_arrs[ax][Network.k_slices[f'+{ax}']] +
                k_dir_arrs[ax][Network.k_slices[f'-{ax}']]
            )
        k_sum_nonzero_mask = (k_sum_arr!=0)
        k_sum_arr_inv = self.ctx.xp.zeros_like(k_sum_arr)
        k_sum_arr_inv[k_sum_nonzero_mask] = 1/k_sum_arr[k_sum_nonzero_mask]
        del k_sum_nonzero_mask

        return k_sum_arr, k_sum_arr_inv, k_dir_arrs

    def _get_R_int_arrs(self):
        """
        Builds the interface resistance arrays
        (on the cpu, and later shuttles it on the gpu if needed)
        First pad integers for the plates and
        adiabatic boundaries to the voxel_struc.
        Then combine `mat_bounds` and `int_bounds` into
        a boundary matrix as in the following example:
        int_bounds = [0,1]
        mat_bounds = [[2,3],
                      [4,5]]
        boundary_matrix = [[ 2.  3.  0. inf]
                           [ 4.  5.  1. inf]
                           [ 0.  1. inf inf]
                           [inf inf inf inf]]
        Then generate the interface resistance arrays for each direction,
        based on the padded voxel structure and the boundary matrix.

        Returns
        -------
        R_ints : dict
            Dictionary with interface resistance arrays along each axis.
            The keys are the different axes ('x', 'y', 'z')
            and the values are the corresponding resistance arrays.
        """
        plate_integ = self.ctx.voxel_struc.max()+ 1
        adiabatic_integ = self.ctx.voxel_struc.max() + 2
        voxel_struc_w_bound_integ = self.ctx.xp.pad(
            self.ctx.voxel_struc,
            pad_width = ((1,1), (1,1), (1,1)),
            mode = 'constant',
            constant_values = (
                (plate_integ,plate_integ),
                (adiabatic_integ,adiabatic_integ),
                (adiabatic_integ,adiabatic_integ)
            )
        )

        boundary_matrix = self.ctx.xp.full((plate_integ+2, plate_integ+2), self.ctx.xp.inf, dtype=self.ctx.dtype)
        boundary_matrix[:plate_integ, :plate_integ] = self.ctx.mat_bounds
        boundary_matrix[:plate_integ, plate_integ] = self.ctx.int_bounds
        boundary_matrix[plate_integ, :plate_integ] = self.ctx.int_bounds

        R_ints = {}
        for ax in Network.axes:
            integ_pos0 = voxel_struc_w_bound_integ[self.k_gen_slices[f'{ax}_0']]
            integ_pos1 = voxel_struc_w_bound_integ[self.k_gen_slices[f'{ax}_1']]
            interface_arr = boundary_matrix[integ_pos0, integ_pos1]
            R_ints[ax] = self.ctx.xp.asarray(interface_arr)
        return R_ints

    def _get_u_start(self, k_sum_arr):
        """
        Initializes the temperature array with a
        linear temperature distribution between the hotplates.
        If `T_start_center` is provided, it loads the initial temperature
        array from the specified file.
        It sets all temperatures for nodes with 0 conductivity in all directions to 0.

        Parameters
        ----------
        k_sum_arr_inv : xp.ndarray
            3D array of the sum of bond conductivities for each voxel.

        Returns
        -------
        T_start : xp.ndarray
            3D array of initial temperatures.
        """

        def _get_linguess():
            """
            Builds an initial temperature array with linearly de- or increasing
            temperatures between the hotplates

            Returns
            -------
            T_start : xp.ndarray
                3D array of temperatures.
                - the temperature entries are linearly de- or increasing
                from `T_l`, the temperature of the left hotplate
                to `T_r`, the temperature of the right hotplate.
            """
            nx, ny, nz = [dim+2 for dim in self.ctx.voxel_struc.shape] # size of voxelstruc + boundaries
            u_start = self.ctx.xp.zeros((nx, ny, nz), dtype=self.ctx.dtype)
            u_start[0,1:-1,1:-1] = self.ctx.u_x0
            u_start[-1,1:-1,1:-1] = self.ctx.u_xN
            positions = self.ctx.xp.arange(1,nx-1) - 1/2
            u_start[Network.u_slices['center']] = (self.ctx.u_x0+((self.ctx.u_xN-self.ctx.u_x0)/(nx-2))*positions)[:, None, None]
            u_start[Network.u_slices['center']][(k_sum_arr == 0)] = 0
            return u_start

        u_start = _get_linguess()
        if self.ctx.u_start_center:
            u_start[Network.u_slices['center']] = self.ctx.xp.load(self.ctx.u_start_center)
        return u_start

    def build(
            self,
            voxel_struc,
            phase_conds,
            u_start=None,
            int_bounds=None,
            mat_bounds=None,
            use_gpu=False,
            dtype=float,
            u_x0=1,
            u_xN=0,
            L_x=1,
            **kwargs
        ):

        self.ctx = BuildContext(
            voxel_struc=voxel_struc,
            phase_conds=phase_conds,
            dtype=dtype,
            use_gpu=use_gpu,
            int_bounds=int_bounds,
            mat_bounds=mat_bounds,
            u_start=u_start,
            u_x0=u_x0,
            u_xN=u_xN,
            L_x=L_x,
            **kwargs
        )

        k_raw_arr = self._get_k_raw_arr()
        R_ints = self._get_R_int_arrs() if self.ctx.mat_bounds is not None else None

        k_sum_arr, k_sum_arr_inv, k_dir_arrs = self._get_k_arrs(k_raw_arr, R_ints=R_ints)
        del R_ints, k_raw_arr
        u_start = self._get_u_start(k_sum_arr)

        nw = Network(
            k_x_arr=k_dir_arrs['x'],
            k_y_arr=k_dir_arrs['y'],
            k_z_arr=k_dir_arrs['z'],
            k_sum_arr=k_sum_arr,
            k_sum_arr_inv=k_sum_arr_inv,
            u_arr=u_start,
            xp = self.ctx.xp,
            dtype = self.ctx.dtype,
            u_x0 = self.ctx.u_x0,
            u_xN = self.ctx.u_xN,
            L_x = self.ctx.L_x,
            dx = self.ctx.dx
        )
        return nw
