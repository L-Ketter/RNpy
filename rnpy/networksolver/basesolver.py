import time
import warnings
from . import _solverutils as su
from ..network import Network as nw
from .. import utils

class BaseSolver:
    """
    Base class for solvers. It provides a common interface for all solvers and implements
    the main solving loop and logging functionality.
    """
    def solve(
            self,
            nw,
            res_max=1e-9,
            t_max=None,
            it_max=None,
            it_log=50,
            print_log=True
        ):
        if print_log:
            print("-"*70, "\nsolving the network ...")
        it, self.it_lst = 0, []
        t_0, self.t_lst = time.perf_counter(), []
        self.res_lst, self.k_effs = [], []
        converged = False
        while not converged:
            t = time.perf_counter()-t_0
            if it % it_log == 0:
                self._log_lists(it, t, res_max, nw, print_log=print_log)
            converged = self.res_lst[-1] < res_max
            it_max_reached = it_max is not None and it == it_max
            t_max_reached = t_max is not None and t > t_max
            if converged or it_max_reached or t_max_reached:
                if not converged:
                    reason = "t_max" if t_max_reached else "it_max"
                    warnings.warn(
                        f"The solver stopped because {reason} was reached "
                        f"(final residual: {self.res_lst[-1]:.2e} ==> {res_max:.2e})."
                    )
                if print_log:
                    print(f'\nFinished calculation in {utils.format_seconds(t)} (hh:mm:ss)', flush=True)
                return nw
            it += 1
            nw.u = self.update_x(nw, x=nw.u, rhs=None)

    def _log_lists(self, it, t, res_max, nw, print_log=True):
        """
        Updates the lists of iterations, kappas, and residuals with the current values.
        It also prints the current iteration, effective conductivity, and residual if `print_log` is True.

        Parameters
        ----------
        print_log : bool, optional
            If True, prints the current iteration, effective conductivity, and residual.
        """
        self.it_lst.append(nw._shuttle_to_cpu(it))
        self.t_lst.append(nw._shuttle_to_cpu(t))
        self.k_effs.append(nw._shuttle_to_cpu(nw.get_k_eff()))
        r_scaled_L1 = su.get_r_scaled_L1(nw, x=nw.u, rhs=None)
        self.res_lst.append(r_scaled_L1)
        if print_log:
            print(f"\riteration: {self.it_lst[-1]} | conductivity: {self.k_effs[-1]:.4f} | "
                  f"residual: {self.res_lst[-1]:.2e} ==> {res_max:.2e}{10*' '}", end="", flush=True)

    def get_log(self):
        return {
            'iterations': nw._shuttle_to_cpu(self.it_lst),
            'k_effs': nw._shuttle_to_cpu(self.k_effs),
            'residuals': nw._shuttle_to_cpu(self.res_lst),
            'times': nw._shuttle_to_cpu(self.t_lst)
        }