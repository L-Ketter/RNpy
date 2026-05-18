import time
from . import _solverutils as su
from ..network import Network as nw
from .. import utils
import warnings

class BaseSolver:
    """
    Base class for solvers. It provides a common interface for all solvers and implements
    the main solving loop and logging functionality.
    """
    def solve(
            self,
            nw,
            res_max=1e-9,
            it_max=1e7,
            it_log=50,
        ):
        it, self.it_lst = 0, []
        self.res_lst, self.k_effs = [], []
        print("-"*70, "\nsolving the network ...")
        t_start = time.time()
        while it <= it_max:
            if it % it_log == 0:
                self._log_lists(it, res_max, nw, print_output=True)
            if self.res_lst[-1] < res_max or it == it_max:
                if it == it_max:
                    warnings.warn(
                        f"maximum number of iterations ({it_max}) reached without convergence "
                        f"(final residual: {self.res_lst[-1]:.2e} ==> {res_max:.2e})."
                    )
                print(f'\nFinished calculation in {utils.format_seconds(time.time()-t_start)} (hh:mm:ss)')
                return nw
            it += 1
            nw.u = self.update_x(nw, x=nw.u, rhs=None)

    def _log_lists(self, it, res_max, nw, print_output=True):
        """
        Updates the lists of iterations, kappas, and residuals with the current values.
        It also prints the current iteration, effective conductivity, and residual if `print_output` is True.

        Parameters
        ----------
        print_output : bool, optional
            If True, prints the current iteration, effective conductivity, and residual.
        """
        self.it_lst.append(nw._shuttle_to_cpu(it))
        self.k_effs.append(nw._shuttle_to_cpu(nw.get_k_eff()))
        r_scaled_L1 = su.get_r_scaled_L1(nw, x=nw.u, rhs=None)
        self.res_lst.append(r_scaled_L1)
        if print_output:
            print(f"\riteration: {self.it_lst[-1]} | conductivity: {self.k_effs[-1]:.4f} | "
                  f"residual: {self.res_lst[-1]:.2e} ==> {res_max:.2e}{10*' '}", end="", flush=True)

    def get_log(self):
        return {
            'iterations': nw._shuttle_to_cpu(self.it_lst),
            'k_effs': nw._shuttle_to_cpu(self.k_effs),
            'residuals': nw._shuttle_to_cpu(self.res_lst)
        }