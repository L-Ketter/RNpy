def get_k_eff(nw):
        """
        First the average flux density along the field direction is calculated.
        Next, considering the mean flux density,
        the potential difference specified by u_xN and u_x0,
        as well as the spacial distance between the hotplates,
        the effective conductivity is calculated using Fouriers law.

        Returns
        -------
        k_eff: float
            The effective conductivity calculated using Fouriers law.
        """
        j_x = (
              - nw.k_x_arr / nw.dx
              * (nw.u[1:, 1:-1, 1:-1] - nw.u[0:-1, 1:-1, 1:-1])
        )
        k_eff = - np.mean(j_x)*nw.L_x / (nw.u_xN - nw.u_x0)
        return k_eff

def get_u_center(nw):
    return nw.u[nw.u_slices["center"]]

def get_jloc(nw):
        """
        Builds an array storing the flux map.
        The calculation assumes steady state conditions (flux into a node equals flux out of a node).
        1) Add the sums of all flux terms in each direction.
        2) Divide by the distance between neighboring nodes to get real flux densities.
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



