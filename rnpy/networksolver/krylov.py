from . import _solverutils as su
from .basesolver import BaseSolver
from .smoothers import SORRedBlack
"""
class ConjugateGradient(BaseSolver):
    def __init__(self, **kwargs):
        self.initialized = False

    def update_arr(self, nw, arr, rhs=None):
        if not self.initialized:
            self.r = su.get_r(nw, arr, rhs=rhs)
            self.p = nw.xp.pad(
                self.r.copy(),
                pad_width=1,
                mode='constant',
                constant_values=0
            )
            self.rs_old = nw.xp.sum(self.r * self.r)
            self.initialized = True
        # calculate distance
        Ap = su.get_Ax(nw, self.p)
        alpha = self.rs_old / nw.xp.sum(self.p[nw.u_slices['center']] * Ap)
        # update solution and residual
        arr += alpha * self.p
        self.r -= alpha * Ap
        # calculate next search direction
        rs_new = nw.xp.sum(self.r * self.r)
        beta = rs_new / self.rs_old
        self.p[nw.u_slices['center']] = self.r + beta * self.p[nw.u_slices['center']]
        self.rs_old = rs_new
        return arr
"""
class ConjugateGradient(BaseSolver):
    def __init__(self, pc=True, **kwargs):
        self.initialized = False
        self.pc = pc

    def _getz(self, nw, r):
        if self.pc:
            return r * nw.k_arrs['sum_inv']
        return r

    def update_arr(self, nw, arr, rhs=None):
        if not self.initialized:
            self.r = su.get_r(nw, arr, rhs=rhs)
            z = self._getz(nw, self.r)
            self.p = nw.xp.pad(z.copy(), pad_width=1, mode='constant', constant_values=0)
            self.rs_old = nw.xp.vdot(self.r, z)
            self.initialized = True
        # calculate distance
        Ap = su.get_Ax(nw, self.p)
        alpha = self.rs_old / nw.xp.vdot(self.p[nw.u_slices['center']], Ap)
        # update solution and residual
        arr += alpha * self.p
        self.r -= alpha * Ap
        # preconditioning
        z = self._getz(nw, self.r)
        # calculate next search direction
        rs_new = nw.xp.vdot(self.r, z)
        beta = rs_new / self.rs_old
        self.rs_old = rs_new
        self.p[nw.u_slices['center']] = z + beta * self.p[nw.u_slices['center']]
        return arr