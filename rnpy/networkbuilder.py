import os
import numpy as np
from .network import Network

class BuildContext():
    """
    The BuildContext class is a simple data container that holds all the parameters
    and intermediate variables needed during the building of the Network object.
    """
    def __init__(
            self,
            voxel_struc,
            phase_conds,
            dtype=None,
            use_gpu=False,
            int_bounds=None,
            mat_bounds=None,
            u_start=None,
            periodic=False,
            **kwargs
        ):

        self.dtype = dtype
        self.use_gpu = use_gpu
        self.xp = Network._choose_array_backend(use_gpu)
        self.voxel_struc = voxel_struc
        self.phase_conds = phase_conds
        self.int_bounds = int_bounds
        self.mat_bounds = mat_bounds
        self.periodic = periodic

        self.u_start_center = u_start
        self.u_x0 = kwargs.get('u_x0', 1)
        self.u_xN = kwargs.get('u_xN', 0)
        self.L_x = kwargs.get('L_x', 1)
        self.dx = self.L_x/voxel_struc.shape[0]

        if (mat_bounds is not None) != (int_bounds is not None):
            raise ValueError("To consider interfaces, both 'mat_bounds' and 'int_bounds'"
                             " must be provided together. Only one was provided.")

        elif (mat_bounds is not None) and (int_bounds is not None):
            self.mat_bounds = self.xp.array(mat_bounds, dtype=self.dtype)
            self.int_bounds = self.xp.array(int_bounds, dtype=self.dtype)
            if len(int_bounds) != len(self.phase_conds):
                raise ValueError("'int_bounds' length must match the number "
                                 "of phases in 'phase_conds'.")
            if self.mat_bounds.shape != (len(self.phase_conds), len(self.phase_conds)):
                raise ValueError("'mat_bounds' must be a square matrix with dimensions "
                                 "equal to the number of phases in 'phase_conds'.")
            if not self.xp.allclose(self.mat_bounds, self.xp.swapaxes(self.mat_bounds, 0, 1), equal_nan=True):
                raise ValueError("'mat_bounds' must be symmetric to represent directional interface conductances.")

class NetworkBuilder:
    """
    The NetworkBuilder class is responsible for constructing a Network object
    based on the provided voxel structure, phase conductivities, and other parameters.
    The build method takes in the necessary parameters, creates a BuildContext object,
    and then generates the conductivity arrays and initial potential array to create
    the Network object.
    """
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
        return Network.apply_periodic(k_raw_arr) if self.ctx.periodic else k_raw_arr

    def _reciprocal(self, arr):
            """
            Calculates the reciprocal of the values in the input array. Defines 1/0 = 0 and 1/inf = 0.
            """
            arr_inv = self.ctx.xp.zeros_like(arr)
            nonzero_noninf = (arr != 0) & (~self.ctx.xp.isinf(arr))
            arr_inv[nonzero_noninf] = 1/arr[nonzero_noninf]
            return arr_inv

    def _get_k_arrs(self, k_raw_arr, g_ints=None):
        """
        Builds the conductivity arrays for the simulation.
        If `mat_bounds` and `int_bounds` are provided, the interface conductances
        are calculated using the _get_g_int_arrs method and used in the bond
        conductivity calculation. Along each axis, the bond conductivities are
        calculated using the internal _calc_bond_cond method.
        Then the sum of all bond conductivities is calculated and stored in `k_sum_arr`.

        Parameters
        ----------
        k_raw_arr : xp.ndarray
            3D array of conductivities with padded boundary values.
        g_ints : dict, optional
            Dictionary with interface conductance arrays along each axis.

        Returns
        -------
        k_sum_arr : xp.ndarray
            3D array of the sum of bond conductivities for each voxel.
        k_sum_arr_inv : xp.ndarray
            3D array of the inverse of the sum of bond conductivities for each voxel.
        k_dir_arrs : dict
            Dictionary with bond conductivity arrays along each axis.
        """

        def _calc_bond_cond(k_pos0, k_pos1, g_int=None):
            """
            Calculates the bond conductivities between two neighboring voxels according
            to the series model.

            Parameters
            ----------
            k_pos0 : xp.ndarray
                Conductivities of voxels at position 0.
            k_pos1 : xp.ndarray
                Conductivities of voxels at position 1.
            g_int : xp.ndarray, optional
                Interface conductances between the voxels.

            Returns
            -------
            k_link : xp.ndarray
                Conductivities of the bonds between the voxels.
            """
            # build masks
            is_open = (self.ctx.xp.abs(k_pos0) == 0) | (self.ctx.xp.abs(k_pos1) == 0)
            is_short = (self.ctx.xp.isinf(k_pos0)) & (self.ctx.xp.isinf(k_pos1))
            if g_int is not None:
                is_open |= (self.ctx.xp.abs(g_int) == 0)
                is_short &= (self.ctx.xp.isinf(g_int))
            is_linker = ~(is_open) & ~(is_short)

            k_link = self.ctx.xp.zeros(k_pos0.shape, dtype=self.ctx.dtype)
            k_link[is_short] = self.ctx.xp.inf
            k_series_inv = 0.5*(self._reciprocal(k_pos0[is_linker]) + self._reciprocal(k_pos1[is_linker]))
            if g_int is not None:
                k_int_inv = self._reciprocal(g_int[is_linker] * self.ctx.dx)
                k_link[is_linker] = 1/(k_series_inv + k_int_inv)
            else:
                k_link[is_linker] = 1/(k_series_inv)
            del is_open, is_short, is_linker, k_series_inv
            return k_link

        k_dir_arrs = {}
        for ax in Network.axes:
            k_pos0 = k_raw_arr[Network.u_ax_slices[f'{ax}_0']]
            k_pos1 = k_raw_arr[Network.u_ax_slices[f'{ax}_1']]
            if self.ctx.mat_bounds is not None and self.ctx.int_bounds is not None:
                k_dir_arrs[ax] = _calc_bond_cond(k_pos0,k_pos1,g_ints[ax])
            else:
                k_dir_arrs[ax] = _calc_bond_cond(k_pos0,k_pos1)
            del k_pos0, k_pos1

        k_sum_arr = self.ctx.xp.zeros(self.ctx.voxel_struc.shape, dtype=self.ctx.dtype)
        for ax in Network.axes:
            k_sum_arr += (
                k_dir_arrs[ax][Network.k_slices[f'+{ax}']] +
                k_dir_arrs[ax][Network.k_slices[f'-{ax}']]
            )
        k_sum_arr_inv = self._reciprocal(k_sum_arr)
        return k_sum_arr, k_sum_arr_inv, k_dir_arrs

    def _get_g_int_arrs(self):
        """
        Builds the interface conductance arrays
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
        Then generate the interface conductance arrays for each direction,
        based on the padded voxel structure and the boundary matrix.

        Returns
        -------
        g_ints : dict
            Dictionary with interface conductance arrays along each axis.
            The keys are the different axes ('x', 'y', 'z')
            and the values are the corresponding conductance arrays.
        """
        plate_integ = len(self.ctx.phase_conds)
        adiabatic_integ = len(self.ctx.phase_conds) + 1
        voxel_struc_w_bound_integ = self.ctx.xp.pad(
            self.ctx.voxel_struc,
            pad_width = ((1,1), (1,1), (1,1)),
            mode = 'constant',
            constant_values = (
                (plate_integ, plate_integ),
                (adiabatic_integ, adiabatic_integ),
                (adiabatic_integ, adiabatic_integ)
            )
        )
        voxel_struc_w_bound_integ = Network.apply_periodic(voxel_struc_w_bound_integ) if self.ctx.periodic else voxel_struc_w_bound_integ
        boundary_matrix = self.ctx.xp.full((plate_integ+2, plate_integ+2), self.ctx.xp.inf, dtype=self.ctx.dtype)
        boundary_matrix[:plate_integ, :plate_integ] = self.ctx.mat_bounds
        boundary_matrix[:plate_integ, plate_integ] = self.ctx.int_bounds
        boundary_matrix[plate_integ, :plate_integ] = self.ctx.int_bounds

        g_ints = {}
        for ax in Network.axes:
            integ_pos0 = voxel_struc_w_bound_integ[Network.u_ax_slices[f'{ax}_0']]
            integ_pos1 = voxel_struc_w_bound_integ[Network.u_ax_slices[f'{ax}_1']]
            interface_arr = boundary_matrix[integ_pos0, integ_pos1]
            g_ints[ax] = self.ctx.xp.asarray(interface_arr)
        del voxel_struc_w_bound_integ, boundary_matrix
        return g_ints

    def _get_u_start(self, k_sum_arr):
        """
        Initializes the solution array with a
        linear distribution between the boundaries.
        If `u_start_center` is provided, it loads the initial solution array as the center values.
        It sets all solutions for nodes with 0 conductivity in all directions to 0.

        Parameters
        ----------
        k_sum_arr_inv : xp.ndarray
            3D array of the sum of bond conductivities for each voxel.

        Returns
        -------
        u_start : xp.ndarray
            3D array of initial solutions.
        """

        def _get_linguess():
            """
            Builds an initial solution array with linearly de- or increasing
            solutions between the boundaries

            Returns
            -------
            u_start : xp.ndarray
                3D array of solutions.
                - the solution entries are linearly de- or increasing
                from `u_x0`, the solution at the left boundary
                to `u_xN`, the solution at the right boundary.
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
        if self.ctx.u_start_center is not None:
            u_start[Network.u_slices['center']] = self.ctx.u_start_center

        return Network.apply_periodic(u_start) if self.ctx.periodic else u_start

    def build(
            self,
            voxel_struc,
            phase_conds,
            u_start=None,
            int_bounds=None,
            mat_bounds=None,
            periodic=False,
            use_gpu=False,
            dtype=float,
            u_x0=1,
            u_xN=0,
            L_x=1,
            **kwargs
        ):
        """
        Main mehthod of the NetworkBuilder class that builds the Network object based on the provided parameters.

        Parameters
        ----------
        voxel_struc : np.ndarray
            A 3D NumPy array representing the voxel structure of the material.
        phase_conds : list
            A list of conductivities corresponding to each phase in the voxel structure.
            The index of the conductivity in the list should match the phase index in the voxel_struc.
        dtype : data-type, optional
            The desired data-type for the arrays of the network object. Default is np.float64.
        use_gpu : bool, optional
            Determines whether a NumPy or CuPy backend will be used for array operations. Default is False (NumPy).
        int_bounds : list, optional
            A list of interface conductances between the phases and boundaries in the field direction.
            The length of the list should match the number of phases.
            Default is None (no boundaries between phases and boundaries).
        mat_bounds : list of lists, optional
            A square matrix of interface conductances between phases.
            The dimensions of the matrix should match the number of phases.
            Default is None (no boundaries between phases).
        periodic : bool, optional
            Determines whether the y- and z- direction are treated as periodic (True) or closed (False). Default is False (non-periodic).
        u_start : xp.ndarray, optional
            A 3D array containing the initial solution values for the simulation.
            If not provided, the initial solution array will be generated with a linear distribution between the boundaries. Default is None.
        u_x0 : float, optional
            The solution at the left boundary. Default is 1.
        u_xN : float, optional
            The solution at the right boundary. Default is 0.
        L_x : float, optional
            The length of the system in the x direction. Default is 1.

        Returns
        -------
        nw : Network
            A Network object initialized with the generated conductivity arrays and initial solution array.
        """
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
            periodic=periodic,
            **kwargs
        )

        k_raw_arr = self._get_k_raw_arr()
        g_ints = self._get_g_int_arrs() if self.ctx.mat_bounds is not None else None
        k_sum_arr, k_sum_arr_inv, k_dir_arrs = self._get_k_arrs(k_raw_arr, g_ints=g_ints)
        del g_ints, k_raw_arr
        u_start = self._get_u_start(k_sum_arr)

        nw = Network(
            k_x_arr=k_dir_arrs['x'],
            k_y_arr=k_dir_arrs['y'],
            k_z_arr=k_dir_arrs['z'],
            k_sum_arr=k_sum_arr,
            k_sum_arr_inv=k_sum_arr_inv,
            u_arr=u_start,
            periodic = self.ctx.periodic,
            xp = self.ctx.xp,
            dtype = self.ctx.dtype,
            u_x0 = self.ctx.u_x0,
            u_xN = self.ctx.u_xN,
            L_x = self.ctx.L_x,
            dx = self.ctx.dx
        )
        return nw
