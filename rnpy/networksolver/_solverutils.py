def get_Ax(nw, arr):
    k = nw.k_arrs
    Ax = k['sum'].copy() * arr[nw.u_slices['center']]
    for d in nw.directions:
        Ax -= k[d] * arr[nw.u_slices[d]]
    return Ax

def get_r(nw, u, rhs=None):
    Au = get_Ax(nw, u)
    r = rhs - Au if rhs is not None else -Au
    return r

def get_r_scaled_L1(nw, arr, rhs=None):
    r_scaled = get_r(nw, u=arr, rhs=rhs) * nw.k_arrs['sum_inv']
    r_scaled_L1 = nw.xp.mean(nw.xp.abs(r_scaled))
    return r_scaled_L1
