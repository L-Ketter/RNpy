from . import _solverutils as su
from .basesolver import BaseSolver
from .smoothers import SORRedBlack
from .. import networksolver as nws
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
    def __init__(self, it_pc=True, **kwargs):
        self.initialized = False
        self.it_pc = it_pc
        self.pc = nws.Jacobi() if it_pc > 0 else None

    def _getz(self, nw, r):
        if self.it_pc == 0:
            return r
        else:
            for _ in range(self.it_pc):
                if _ == 0:
                    z = r * nw.k_arrs['sum_inv']
                elif _ == 1:
                    z_pad = nw.xp.pad(z, pad_width=1, mode='constant', constant_values=0)
                    r_pad = nw.xp.pad(r, pad_width=1, mode='constant', constant_values=0)
                else:
                    z_pad = self.pc.update_arr(nw, arr=z_pad, rhs=r_pad)
            return z_pad[nw.u_slices['center']] if self.it_pc > 1 else z

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