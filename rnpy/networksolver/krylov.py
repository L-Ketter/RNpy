from . import _solverutils as su
from .basesolver import BaseSolver
from .. import networksolver as nws

class CG(BaseSolver):
    """
    Conjugate Gradient (CG) algorithm with optional Jacobi-type preconditioning.
    The algorithm iteratively updates the solution by calculating the search direction and step size based on the residuals.

    Attributes
    ----------
    it_pc: int
        If 0, no preconditioning is applied.
        If 1, a standard Jacobi preconditioning is applied.
        If >1, iterative Jacobi-type smoothing is used as an approximate preconditioner
    """
    def __init__(self, it_pc=1, **kwargs):
        self.initialized = False
        self.it_pc = it_pc
        self.pc = kwargs.get('pc', nws.Jacobi()) if it_pc > 0 else None

    def _get_inner_product(self, nw,a, b):
        return nw.xp.vdot(a, b)

    def _get_z(self, nw, r):
        if self.it_pc == 0:
            return r
        else:
            for _ in range(self.it_pc):
                if _ == 0:
                    z = r * nw.k_arrs['sum_inv']
                    if self.it_pc == 1:
                        return z
                elif _ == 1:
                    z_pad = nw.xp.pad(z, pad_width=1, mode='constant', constant_values=0)
                    r_pad = nw.xp.pad(r, pad_width=1, mode='constant', constant_values=0)
                    if nw.periodic:
                        r_pad = nw.apply_periodic(r_pad)
                        z_pad = nw.apply_periodic(z_pad)
                if _ >= 1:
                    z_pad = self.pc.update_x(nw, x=z_pad, rhs=r_pad)
            return z_pad[nw.u_slices['center']]

    def update_x(self, nw, x, rhs=None):
        if not self.initialized:
            self.r = su.get_r(nw, x=x, rhs=rhs)
            z = self._get_z(nw, self.r)
            self.p = nw.xp.pad(z.copy(), pad_width=1, mode='constant', constant_values=0)
            if nw.periodic:
                self.p = nw.apply_periodic(self.p)
            self.rs_old = self._get_inner_product(nw, self.r, z)
            self.initialized = True
        # calculate distance
        Ap = su.get_Ax(nw, self.p)
        alpha = self.rs_old / self._get_inner_product(nw, self.p[nw.u_slices['center']], Ap)
        # update solution and residual
        x[nw.u_slices['center']] += alpha * self.p[nw.u_slices['center']]
        if nw.periodic:
            x = nw.apply_periodic(x)
        self.r -= alpha * Ap
        # preconditioning
        z = self._get_z(nw, self.r)
        # calculate next search direction
        rs_new = self._get_inner_product(nw, self.r, z)
        beta = rs_new / self.rs_old
        self.rs_old = rs_new
        self.p[nw.u_slices['center']] = z + beta * self.p[nw.u_slices['center']]
        self.p = nw.apply_periodic(self.p) if nw.periodic else self.p
        return x

class COCG(CG):
    """
    Conjugate Orthogonal Conjugate Gradient (COCG) method.
    This method is a modification of the CG method, designed to handle complex symmetric matrices.

    Attributes
    ----------
    it_pc: int
        If 0, no preconditioning is applied.
        If 1, a standard Jacobi preconditioning is applied.
        If >1, iterative Jacobi-type smoothing is used as an approximate preconditioner
    """
    def _get_inner_product(self, nw,a, b):
        return nw.xp.sum(a * b)