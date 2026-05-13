from numba import njit
from .basesolver import BaseSolver

class Jacobi(BaseSolver):
    def __init__(self, **kwargs):
        self.omega = kwargs.get("omega", None)

    def update_arr(self, nw, arr, rhs=None):
        """
        Generates a temperature array with updated temperatures. The updated temperatures
        are calculated based on the conductivities and current temperatures using the Jacobi method.

        Parameters
        ----------
        network : Network
            The network object containing the temperature array and conductivities.

        Returns
        -------
        T_new : xp.ndarray
            3D array of temperatures.
            - the temperature entries are updated according to the Jacobi method
        """
        k = nw.k_arrs
        arr_new = nw.xp.zeros((arr[nw.u_slices['center']].shape), dtype=arr.dtype)
        for d in nw.directions:
            arr_new += arr[nw.u_slices[d]] * k[d]
        if rhs is not None:
            arr_new += rhs[nw.u_slices['center']]
        arr_new *= k['sum_inv']
        if self.omega:
            arr_new = arr[nw.u_slices['center']] + self.omega * (arr_new - arr[nw.u_slices['center']])
        arr[nw.u_slices['center']] = arr_new
        return arr


@njit
def _update_arr_SOR(arr, k_arrs, rhs, omega):
    """
    Generates a temperature array with updated temperatures. The updated temperatures
    are calculated based on the conductivities and current temperatures using the
    successive over relaxation (SOR) method.
    numbas njit is utilized and speeds up the calculation significantly.

    Parameters
    ----------
    T_arr : np.ndarray
        an array with temperatures of all voxel_struc entries and boundaries.
    omega: float
        over-relaxation parameter. Typical values lie between 1 < omega < 2.
        omega > 2 leads to unstable convergence. omega < 1 leads to slow convergence.
        rule of thumb: small values -> more stability,
                       large values -> faster convergence
    ks: list
        list of the conductivities in all directions. The conductivities
        entries need to be specified in the following order ['r','l','ba','f','t','b']

    Returns
    -------
    T_new : np.ndarray
        3D array of temperatures.
        - the temperature entries are updated according to the SOR method
    """
    k_px, k_mx, k_py, k_my, k_pz, k_mz, k_sum_inv, k_sum = k_arrs
    nx, ny, nz = arr.shape
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                rhs_val = 0.0
                if rhs is not None:
                    rhs_val = rhs[i, j, k]
                arr_new = (
                    arr[i+1, j, k] * k_px[i-1, j-1, k-1]
                    + arr[i-1, j, k] * k_mx[i-1, j-1, k-1]
                    + arr[i, j+1, k] * k_py[i-1, j-1, k-1]
                    + arr[i, j-1, k] * k_my[i-1, j-1, k-1]
                    + arr[i, j, k+1] * k_pz[i-1, j-1, k-1]
                    + arr[i, j, k-1] * k_mz[i-1, j-1, k-1]
                    + rhs_val
                ) * k_sum_inv[i-1, j-1, k-1]
                arr[i, j, k] += omega * (arr_new - arr[i, j, k])

class SOR(BaseSolver):
    def __init__(self, omega=1.979):
        self.omega = omega

    def update_arr(self, nw, arr, rhs=None):
        """
        Updates temperatures in the current temperature array using the SOR algorithm.

        Parameters
        ----------
        omega: float
            over-relaxation parameter. Typical values lie between 1 < omega < 2.
            omega > 2 leads to unstable convergence. omega < 1 leads to slow convergence.
            rule of thumb: small values -> more stability,
                           large values -> faster convergence
        """
        k_arrs = tuple(arr for arr in nw.k_arrs.values())
        _update_arr_SOR(arr, k_arrs, rhs=rhs, omega=self.omega)
        return arr

class SORRedBlack(BaseSolver):
    def __init__(self, omega=1.979):
        self.omega = omega
        self.red_black_slices = None

    def _get_red_black_slices(self, nw):
        nx, ny, nz = nw.k_arrs['sum_inv'].shape
        i_even, i_odd = slice(0,nx,2), slice(1,nx,2)
        j_even, j_odd = slice(0,ny,2), slice(1,ny,2)
        k_even, k_odd = slice(0,nz,2), slice(1,nz,2)
        red_slices = [
            (i_even, j_even, k_even),
            (i_even, j_odd,  k_odd),
            (i_odd,  j_even, k_odd),
            (i_odd,  j_odd,  k_even)
        ]
        black_slices = [
            (i_even, j_even, k_odd),
            (i_even, j_odd,  k_even),
            (i_odd,  j_even, k_even),
            (i_odd,  j_odd,  k_odd)
        ]
        return [red_slices, black_slices]

    def update_arr(self, nw, arr, rhs=None):
        """
        Updates temperatures in the current temperature array using the SOR redblack algorithm.
        The algorithm is utilizing a two-step procedure: In the first step, the temperatures
        in the red-mask, and in the second step, temperatures in the black mask are updated.
        In each step, new temperatures are updated using the Jacobi method with over-relaxation

        Parameters
        ----------
        omega: float
            over-relaxation parameter. Typical values lie between 1 < omega < 2.
            omega > 2 leads to unstable convergence. omega < 1 leads to slow convergence.
            rule of thumb: small values -> more stability,
                           large values -> faster convergence
            """
        if not self.red_black_slices:
            self.red_black_slices = self._get_red_black_slices(nw)

        k = nw.k_arrs
        for color_slices in (self.red_black_slices):
            for idx in color_slices:
                arr_new = (
                    arr[nw.u_slices['+x']][idx] * k['+x'][idx]
                    + arr[nw.u_slices['-x']][idx] * k['-x'][idx]
                    + arr[nw.u_slices['+y']][idx] * k['+y'][idx]
                    + arr[nw.u_slices['-y']][idx] * k['-y'][idx]
                    + arr[nw.u_slices['+z']][idx] * k['+z'][idx]
                    + arr[nw.u_slices['-z']][idx] * k['-z'][idx]
                )
                if rhs is not None:
                    arr_new += rhs[nw.u_slices['center']][idx]
                arr_new *= k['sum_inv'][idx]

                arr[nw.u_slices['center']][idx] += self.omega * (
                    arr_new - arr[nw.u_slices['center']][idx]
                )
        return arr

