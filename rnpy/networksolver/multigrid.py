import time
import os
import numpy as np
from .. import networksolver as nws
from .. import utils
import warnings
from numba import njit


class multigridSolver(nws.BaseSolver):
    def __init__(self, levels, smoother_iterations=50):
        self.lvls = levels
        self.smoother_iterations = smoother_iterations

    def _stepdown(self, res_fine, nw_coarse):
        res_coarse = self.restrict(res_fine)
        res_coarse = self.xp.pad(res_coarse, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        e_coarse = np.zeros(res_coarse.shape, dtype=self.nw_coarse.dtype)
        self._smooth(
            nw = nw_coarse,
            iterations=self.smoother_iterations,
            arr=e_coarse,
            rhs=res_coarse
        )
        return e_coarse

    def _stepup(self, e_coarse, u_fine, nw_fine):
        e_fine = self.prolongate(e_coarse)
        nw_fine.T_array += e_fine
        self._smooth(
            nw=nw_fine,
            iterations=self.smoother_iterations,
            arr=u_fine
        )
        return u_fine

    def _smooth(self, nw, iterations, arr, rhs=None, smoother=nws.SORRedBlackSolver(omega=1.979)):
        for _ in range(iterations):
            smoother.update_arr(nw, arr=arr, rhs=rhs)

    def _get_residual_arr(self, network, arr, rhs=None):
        #T = network.T_arr
        k = network.k_arrs
        res_arr = network.xp.zeros((arr[network.T_slices['center']].shape), dtype=arr.dtype)
        for direc in network.directions:
            res_arr += k[direc] * (arr[network.T_slices[direc]] - arr[network.T_slices['center']])
        if rhs is not None:
            res_arr -= rhs
        return res_arr

    def cycle(self):
        l0,l1,l2 = self.lvls
        self._smooth(
            l0,
            iterations=self.smoother_iterations,
            arr=l0.T_arr
        )
        r0 = self._get_residual_arr(l0, arr=l0.T_arr, rhs=None)
        r1 = self._restrict(r0)
        e1 = self.xp.zeros(r1.shape, dtype=l1.T_arr.dtype)
        r1 = self._smooth(
            l1,
            arr=e1,
            rhs=r1
        )













    def restrict(self, res_fine):
        res_coarse = res_fine[::2, ::2, ::2] + res_fine[1::2, ::2, ::2] + \
                     res_fine[::2, 1::2, ::2] + res_fine[::2, ::2, 1::2] + \
                     res_fine[1::2, 1::2, ::2] + res_fine[1::2, ::2, 1::2] + \
                     res_fine[::2, 1::2, 1::2] + res_fine[1::2, 1::2, 1::2]
        return res_coarse

    def prolongate(self, res_coarse):
        # use the simple injection method for prolongation
        res_fine = res_coarse.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
        return res_fine / 8

    def update_T_arr(self, network):
        self.cycle()





