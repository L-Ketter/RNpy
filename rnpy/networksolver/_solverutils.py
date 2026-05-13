def get_Ax(nw, arr):
    """
    Calculates the matrix-vector product A*x, where A is the system matrix representing the discretized PDE
    and x is the input array (arr). The function uses the conductivity values from the network (nw) to compute
    the product.

    Parameters
    ----------
    nw : Network
        The network object containing the conductivity values and other relevant information.
    arr : np.ndarray
        The input array representing the current solution estimate.

    Returns
    -------
    np.ndarray
        The result of the matrix-vector product A*x.
    """
    k = nw.k_arrs
    Ax = k['sum'].copy() * arr[nw.u_slices['center']]
    for d in nw.directions:
        Ax -= k[d] * arr[nw.u_slices[d]]
    return Ax

def get_r(nw, u, rhs=None):
    """
    Calculates the residual r = rhs - A*u, where A is the system matrix,
    u is the current solution estimate, and rhs is the right-hand side of the equation.
    If rhs is None, it calculates r = -A*u.

    Parameters
    ----------
    nw : Network
        The network object containing the conductivity values and other relevant information.
    u : np.ndarray
        The current solution estimate.
    rhs : np.ndarray, optional
        The right-hand side of the equation. If None, it is assumed to be zero.

    Returns
    -------
    np.ndarray
        The calculated residual r.
    """
    Au = get_Ax(nw, u)
    r = rhs - Au if rhs is not None else -Au
    return r

def get_r_scaled_L1(nw, arr, rhs=None):
    """
    Calculates the L1 norm of the residual scaled by the inverse of the sum of conductivities. This provides a measure of the average absolute residual per node, normalized by the local conductivity.

    Parameters
    ----------
    nw : Network
        The network object containing the conductivity values and other relevant information.
    arr : np.ndarray
        The input array representing the current solution estimate.
    rhs : np.ndarray, optional
        The right-hand side of the equation. If None, it is assumed to be zero.

    Returns
    -------
    np.ndarray
        The calculated scaled L1 norm of the residual.
    """
    r_scaled = get_r(nw, u=arr, rhs=rhs) * nw.k_arrs['sum_inv']
    r_scaled_L1 = nw.xp.mean(nw.xp.abs(r_scaled))
    return r_scaled_L1
