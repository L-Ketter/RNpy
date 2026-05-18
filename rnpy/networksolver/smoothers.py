from numba import njit
from .basesolver import BaseSolver

class Jacobi(BaseSolver):
    """
    Solves the linear system using the Jacobi algorithm.

    The algorithm computes the new solution by updating all grid nodes simultaneously
    based only on values from the previous iteration.
    """
    def __init__(self, **kwargs):
        self.omega = kwargs.get("omega", None)

    def update_x(self, nw, x, rhs=None):
        k = nw.k_arrs
        x_new = nw.xp.zeros(
            (x[nw.u_slices['center']].shape), dtype=x.dtype
        ) if rhs is None else rhs[nw.u_slices['center']].copy()
        for d in nw.directions:
            x_new += x[nw.u_slices[d]] * k[d]
        x_new *= k['sum_inv']
        if self.omega:
            x_new = x[nw.u_slices['center']] + self.omega * (x_new - x[nw.u_slices['center']])
        x[nw.u_slices['center']] = x_new
        return x


@njit
def _update_x_SOR(x, k_arrs, rhs, omega):
    k_px, k_mx, k_py, k_my, k_pz, k_mz, k_sum_inv, k_sum = k_arrs
    nx, ny, nz = x.shape
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                rhs_val = 0.0
                if rhs is not None:
                    rhs_val = rhs[i, j, k]
                x_new = (
                    x[i+1, j, k] * k_px[i-1, j-1, k-1]
                    + x[i-1, j, k] * k_mx[i-1, j-1, k-1]
                    + x[i, j+1, k] * k_py[i-1, j-1, k-1]
                    + x[i, j-1, k] * k_my[i-1, j-1, k-1]
                    + x[i, j, k+1] * k_pz[i-1, j-1, k-1]
                    + x[i, j, k-1] * k_mz[i-1, j-1, k-1]
                    + rhs_val
                ) * k_sum_inv[i-1, j-1, k-1]
                x[i, j, k] += omega * (x_new - x[i, j, k])

class SOR(BaseSolver):
    """
    Solves the linear system using the SOR algorithm.

    The algorithm iterates over all nodes sequentially and updates each value
    in place. If no over-relaxation is applied (omega=1),
    the method is equivalent to the Gauss-Seidel method.

    Parameters
    ----------
    omega: float
        over-relaxation parameter. Typical values lie between 1 < omega < 2.
        omega > 2 leads to unstable convergence. omega < 1 leads to slow convergence.
        rule of thumb: small values -> more stability,
                       large values -> faster convergence
    """
    def __init__(self, omega=1.979):
        self.omega = omega

    def update_x(self, nw, x, rhs=None):
        k_arrs = tuple(k for k in nw.k_arrs.values())
        _update_x_SOR(x, k_arrs, rhs=rhs, omega=self.omega)
        return x


class SORRedBlack(BaseSolver):
    """
    Solves the linear system using the SOR redblack algorithm.

    The algorithm partitions the grid into black and red nodes via a checkerboard pattern,
    to get two independent sets of nodes. The update is performed in two steps:
    First, the new solution for the red nodes is computed based on the current
    solution, followed by an over-relaxed update.
    Black nodes are then updated using the newly computed red values,
    again applying the over-relaxation formula.

    Attributes
    ----------
    omega: float
        over-relaxation parameter. Typical values lie between 1 < omega < 2.
        omega > 2 leads to unstable convergence. omega < 1 leads to slow convergence.
        rule of thumb: small values -> more stability,
                       large values -> faster convergence
    """
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

    def update_x(self, nw, x, rhs=None):
        if not self.red_black_slices:
            self.red_black_slices = self._get_red_black_slices(nw)

        k = nw.k_arrs
        for color_slices in (self.red_black_slices):
            for idx in color_slices:
                x_new = (
                    x[nw.u_slices['+x']][idx] * k['+x'][idx]
                    + x[nw.u_slices['-x']][idx] * k['-x'][idx]
                    + x[nw.u_slices['+y']][idx] * k['+y'][idx]
                    + x[nw.u_slices['-y']][idx] * k['-y'][idx]
                    + x[nw.u_slices['+z']][idx] * k['+z'][idx]
                    + x[nw.u_slices['-z']][idx] * k['-z'][idx]
                )
                if rhs is not None:
                    x_new += rhs[nw.u_slices['center']][idx]
                x_new *= k['sum_inv'][idx]

                x[nw.u_slices['center']][idx] += self.omega * (
                    x_new - x[nw.u_slices['center']][idx]
                )
        return x

