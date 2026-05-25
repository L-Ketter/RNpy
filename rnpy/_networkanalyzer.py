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
    j_x = get_j(nw, ax='x')
    k_eff = - nw.xp.mean(j_x)*nw.L_x / (nw.u_xN - nw.u_x0)
    return k_eff

def get_u_center(nw):
    return nw.u[nw.u_slices["center"]]

def get_j(nw, ax, centered=False):
    """
    Calculates the local flux density in the specified direction.
    """
    k = {'x': nw.k_x_arr, 'y': nw.k_y_arr, 'z': nw.k_z_arr}[ax]
    u1 = nw.u[nw.u_ax_slices[f'{ax}_1']]
    u0 = nw.u[nw.u_ax_slices[f'{ax}_0']]
    j = -k * (u1-u0) / nw.dx
    if centered:
        j = 0.5 * (j[nw.k_slices['+'+ax]] +j[nw.k_slices['-'+ax]])
    return j

def get_j_mag(nw):
    """
    Calculates the magnitude of the flux density vector at each node.
    """
    j_x = get_j(nw, ax='x', centered=True)
    j_y = get_j(nw, ax='y', centered=True)
    j_z = get_j(nw, ax='z', centered=True)
    return nw.xp.sqrt(j_x**2 + j_y**2 + j_z**2)

def get_j_act(nw):
    """
    Computes a scalar flux activity measure at each node based on
    absolute face flux contributions in a steady-state transport field.
    """
    u_center = get_u_center(nw)
    j_int= nw.xp.zeros(u_center.shape, dtype=nw.dtype)
    for direc in nw.directions:
        u_direc = nw.u[nw.u_slices[direc]]
        j_int += nw.xp.abs(nw.k_arrs[direc] * (u_direc-u_center))
    j_int /= 2*nw.dx
    return j_int



