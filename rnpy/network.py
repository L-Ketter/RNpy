import numpy as np
from . import networksolver as nws
from . import _networkanalyzer as nwa

class Network:
    axes = ['x','y','z']
    directions = ['+x','-x','+y','-y','+z','-z']
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

    def get_jloc(self):
        return nwa.get_jloc(self)

    def get_u_center(self):
        return nwa.get_u_center(self)

    def get_kappa(self):
        return nwa.get_kappa(self)

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
        Converts a value to CPU if it is a CuPy array, otherwise returns the value as is.

        Parameters
        ----------
        val : any
            The value to be converted. It can be a CuPy array, NumPy array, or any other type.

        Returns
        -------
        any
            The value converted to CPU if it was a CuPy array, otherwise the original value.
        """
        return val.get() if hasattr(val, "get") else val





