import numpy as np
from . import networksolver as nws
from . import _networkanalyzer as nwa

class Network:
    """
    This class stores all the relevant information about the network,
    including the conductivity arrays, the solution array,
    and the parameters of the system.
    It also provides methods to calculate the local flux density (jloc)
    and the effective conductivity (kappa) of the network.

    Attributes
    ----------
    u : np.ndarray
        The solution array of the network, including boundaries.
    k_arrs : dict
        A dictionary pointing to the conductivity arrays for each direction and their sums.
    xp : module
        The array backend module (either NumPy or CuPy) used for calculations.
    u_x0 : float
        The solution at the left boundary.
    u_xN : float
        The solution at the right boundary.
    L_x : float
        The length of the system in the x direction.
    dx : float
        The distance between neighboring nodes in the network.
    dtype : data-type
        The data type used for the arrays in the network (e.g., np.float64, np.complex128, np.float32).
    """
    axes = ['x','y','z']
    directions = ['+x','-x','+y','-y','+z','-z']
    u_ax_slices = {
        'x_0': (slice(None, -1), slice(1, -1), slice(1, -1)),
        'x_1': (slice(1, None), slice(1, -1), slice(1, -1)),
        'y_0': (slice(1, -1), slice(None, -1), slice(1, -1)),
        'y_1': (slice(1, -1), slice(1, None), slice(1, -1)),
        'z_0': (slice(1, -1), slice(1, -1), slice(None, -1)),
        'z_1': (slice(1, -1), slice(1, -1), slice(1, None))
    }
    u_slices = {
        'center': (slice(1, -1), slice(1, -1), slice(1, -1)),
        '+x': (slice(2, None), slice(1, -1), slice(1, -1)),
        '-x': (slice(None, -2), slice(1, -1), slice(1, -1)),
        '+y': (slice(1, -1), slice(2, None), slice(1, -1)),
        '-y': (slice(1, -1), slice(None, -2), slice(1, -1)),
        '+z': (slice(1, -1), slice(1, -1), slice(2, None)),
        '-z': (slice(1, -1), slice(1, -1), slice(None, -2))
    }
    k_slices = {
        '+x': (slice(1, None), slice(None, None), slice(None, None)),
        '-x': (slice(None, -1), slice(None, None), slice(None, None)),
        '+y': (slice(None, None), slice(1, None), slice(None, None)),
        '-y': (slice(None, None), slice(None, -1), slice(None, None)),
        '+z': (slice(None, None), slice(None, None), slice(1, None)),
        '-z': (slice(None, None), slice(None, None), slice(None, -1))
    }
    def __init__(
            self,
            k_x_arr,
            k_y_arr,
            k_z_arr,
            k_sum_arr,
            k_sum_arr_inv,
            u_arr,
            periodic,
            xp,
            dtype,
            u_x0,
            u_xN,
            L_x,
            dx
        ):
        self.u = u_arr
        self.xp = xp
        self.dtype = dtype
        self.periodic = periodic
        self.u_x0 = u_x0
        self.u_xN = u_xN
        self.L_x = L_x
        self.dx = dx
        self.k_x_arr = k_x_arr
        self.k_y_arr = k_y_arr
        self.k_z_arr = k_z_arr
        self.k_sum_arr = k_sum_arr
        self.k_sum_arr_inv = k_sum_arr_inv
        self.k_arrs ={
            '+x': self.k_x_arr[Network.k_slices['+x']],
            '-x': self.k_x_arr[Network.k_slices['-x']],
            '+y': self.k_y_arr[Network.k_slices['+y']],
            '-y': self.k_y_arr[Network.k_slices['-y']],
            '+z': self.k_z_arr[Network.k_slices['+z']],
            '-z': self.k_z_arr[Network.k_slices['-z']],
            'sum_inv': self.k_sum_arr_inv,
            'sum': self.k_sum_arr
        }

    def get_j_act(self):
        """
        Computes a scalar flux activity measure at each node based on
        absolute face flux contributions in a steady-state transport field.
        1) Add the sums of all flux terms in each direction.
        2) Divide by the distance between neighboring nodes to get real flux densities.
        3) Due to steady state conditions, half of the calculated sum of absolute fluxes
        is going into the node and half is going out of the node. Hence, we divide by 2.

        Returns
        -------
        j_int: xp.ndarray
            The scalar flux intensity measure at each node based
            on absolute face flux contributions in a steady-state transport field.
        """
        return self._shuttle_to_cpu(nwa.get_j_act(self))

    def get_j(self, ax, centered=False):
        """
        Calculates the local flux density in the specified direction.

        Parameters
        ----------
        nw: Network
            The Network object containing the solution array and conductivity arrays.
        ax: str
            The direction along which to calculate the flux density ('x', 'y', or 'z').
        centered: bool, optional
            Whether to linearly interpolate the flux density
            to the cell centers (default is False).

        Returns
        -------
        j: xp.ndarray
            The local flux density in the specified direction.
        """
        return self._shuttle_to_cpu(nwa.get_j(self, ax=ax, centered=centered))

    def get_j_mag(self):
        """
        Calculates the magnitude of the flux density vector at each node.
        1) Calculates flux densities in each direction and interpolates to the node values
        2) Takes the magnitude of the resulting flux density vector.
        Corresponds to local current density in the case of electrical conductivity.
        Corresponds to local heat flux density in the case of thermal conductivity.

        Parameters
        ----------
        nw: Network
            The Network object containing the solution array and conductivity arrays.

        Returns
        -------
        j_mag: xp.ndarray
            The magnitude of the flux density vector at each node.
        """
        return self._shuttle_to_cpu(nwa.get_j_mag(self))

    def get_u_center(self):
        """
        Get the solution array without boundaries

        Returns
        -------
        u_center: np.ndarray
            An array of the same shape as the inner part of the solution array (i.e., excluding the boundaries).
        """
        return self._shuttle_to_cpu(nwa.get_u_center(self))

    def get_k_eff(self):
        """
        Calculates the effective conductivity.

        Returns
        -------
        kappa: float
            The effective conductivity.
        """
        return self._shuttle_to_cpu(nwa.get_k_eff(self))

    @staticmethod
    def apply_periodic(arr):
        """
        Applies periodic boundary conditions in the y- and z-directions to an input array that has boundary padding.
        """
        arr[:,0,:] = arr[:,-2,:]
        arr[:,-1,:] = arr[:,1,:]
        arr[:,:,0] = arr[:,:,-2]
        arr[:,:,-1] = arr[:,:,1]
        return arr

    @staticmethod
    def _choose_array_backend(use_gpu=False):
        """
        Selects NumPy (CPU) or CuPy (GPU) as the array backend.

        Parameters
        ----------
        use_gpu : bool, optional
            If True, CuPy is used as the array backend. If False, NumPy is used. Default is False.

        Raises
        ------
        ImportError
            If `use_gpu=True` but CuPy is not installed.
        RuntimeError
            If `use_gpu=True` but CUDA is unavailable.
        """
        if use_gpu:
            try:
                import cupy as cp
            except ImportError:
                raise ImportError("CuPy is not installed.")
            if not cp.cuda.is_available():
                raise RuntimeError("CuPy is installed, but CUDA is not available.")
            return cp
        else:
            return np

    @staticmethod
    def _shuttle_to_cpu(val):
        """
        Shuttles a value to CPU if it is a CuPy array, otherwise returns the value as is.

        Parameters
        ----------
        val : any
            The value to be shuttled to CPU. It can be any data type.

        Returns
        -------
        any
            The value shuttled to CPU if it was a CuPy array, otherwise the original value.
        """
        return val.get() if hasattr(val, "get") else val





