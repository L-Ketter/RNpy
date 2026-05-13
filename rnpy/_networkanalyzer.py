import numpy as np

def get_kappa(nw):
        """
        Calculated the effective conductivity. First the average flux density going through the network
        along the field direction is calculated. Next, considering the mean flux density,
        the temperature difference specified by T_l and T_r,
        as well as the spacial distance between the hotplates, the effective conductivity is calculated using Fouriers law.

        Returns
        -------
        kappa: float
            The effective conductivity calculated using Fouriers law.
        """
        Q_x = (
              - nw.k_x_arr / nw.dx
              * (nw.u[1:, 1:-1, 1:-1] - nw.u[0:-1, 1:-1, 1:-1])
        )
        kappa = - np.mean(Q_x)*nw.L_x / (nw.u_xN - nw.u_x0)
        return kappa

def get_u_center(nw):
    return nw.u[nw.u_slices["center"]]

def get_jloc(nw):
        """
        Note: When jloc is calculated with default values (dT=-1, Lx=1), jloc might need to be rescaled.
        To get the real jloc for your system, multiply it with dT_real/Lx_real. Often it is sufficient
        to just show normalized jloc. In such cases, simply divide jloc by its maximum entry.
        ------------------------------------------------------------------------------
        Builds an array storing the local flux densities going through each node.
        The calculation assumes steady state conditions (flux into a node equals flux out of a node).
        1) Add the sums of all flux terms in each direction.
        2) Divide by the distance between neighboring nodes to get real current densities.
        3) Due to steady state conditions, half of the calculated sum of absolute fluxes
        is going into the node and half is going out of the node. Hence, we divide by 2.
        """
        u_center = get_u_center(nw)
        jloc = nw.xp.zeros(u_center.shape, dtype=nw.dtype)
        for direc in nw.directions:
            u_direc = nw.u[nw.u_slices[direc]]
            jloc += nw.xp.abs(nw.k_arrs[direc] * (u_direc-u_center))
        jloc /= 2*nw.dx
        return jloc



