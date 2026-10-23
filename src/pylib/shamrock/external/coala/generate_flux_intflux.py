import numpy as np

# -------------------------------
# functions for coagulation
# -------------------------------


def compute_flux_coag_k0(gij, tensor_tabflux_coag):
    """
    Function to compute the approximation of the coagulation flux with DG scheme k=0
    Flux defined at the right boundary of mass bins, i.e. flux[j] ~ F(m_{j+1/2})

    Parameters
    ----------
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux


    Returns
    -------
    flux : 1D array (dim = nbins), type -> float
       approximation of the coagulation flux at each bin

    """

    # use Einstein summation over all the pair interaction (l,m) for each j.
    # compared to the definition of tensor_tabflux_coag l <-> lp and m <-> l
    flux = np.einsum("jlm,l,m->j", tensor_tabflux_coag, gij, gij)

    return flux


def compute_flux_coag_k0_kdv(gij, tensor_tabflux_coag, dv):
    """
    Function to compute the approximation of the coagulation flux with DG scheme k=0
    Flux defined at the right boundary of mass bins, i.e. flux[j] ~ F(m_{j+1/2})
    Function for ballistic kernel with differential velocities dv

    Parameters
    ----------
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains


    Returns
    -------
    flux : 1D array (dim = nbins), type -> float
       approximation of the coagulation flux at each bin

    """

    # use Einstein summation over all the pair interaction (l,m) for each j.
    # compared to the definition of tensor_tabflux_coag l <-> lp and m <-> l
    flux = np.einsum("jlm,lm,l,m->j", tensor_tabflux_coag, dv, gij, gij)

    return flux


def compute_flux_coag(gij, tensor_tabflux_coag):
    """
    Function to compute the approximation of the coagulation flux with DG scheme k>0
    Flux defined at the right boundary of mass bins, i.e. flux[j] ~ F(m_{j+1/2})

    Parameters
    ----------
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate coagulation flux


    Returns
    -------
    flux : 1D array (dim = nbins), type -> float
       approximation of the coagulation flux at each bin

    """

    # test to use only einsum, does not work yet !
    # flux = np.einsum('jlmqp,lq,mp->j', tensor_tabflux_coag, gij, gij)
    # flux = 0.

    # create 4d array with element gij*gij
    arr_gij = np.einsum("ac,bd->abcd", gij, gij)
    flux = np.asarray(
        [np.sum(tensor_tabflux_coag[j, :, :, :, :] * arr_gij) for j in range(len(gij[:, 0]))]
    )

    return flux


def compute_flux_coag_kdv(gij, tensor_tabflux_coag, dv):
    """
    Function to compute the approximation of the coagulation flux with DG scheme k>0
    Flux defined at the right boundary of mass bins, i.e. flux[j] ~ F(m_{j+1/2})
    Function for ballistic kernel with differential velocities dv

    Parameters
    ----------
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains


    Returns
    -------
    flux : 1D array (dim = nbins), type -> float
       approximation of the coagulation flux at each bin

    """

    # test to use only einsum, does not work yet !
    # flux = np.einsum('jlmqp,lq,mp->j', tensor_tabflux_coag, gij, gij)
    # flux = 0.

    # create 4d array with element gij*gij
    # arr_gij_dv = np.einsum('ac,bd,ab->abcd',gij,gij,dv)

    # test
    nbins = len(gij[:, 0])
    kpol = len(gij[0, :]) - 1
    arr_gij_dv = np.zeros((nbins, nbins, kpol + 1, kpol + 1))
    for lp in range(nbins):
        for l in range(nbins):
            for ip in range(kpol + 1):
                for i in range(kpol + 1):
                    arr_gij_dv[lp, l, ip, i] = gij[lp, ip] * gij[l, i] * dv[lp, l]

    flux = np.asarray(
        [np.sum(tensor_tabflux_coag[j, :, :, :, :] * arr_gij_dv) for j in range(len(gij[:, 0]))]
    )

    return flux


def compute_intflux_coag(gij, tensor_tabintflux_coag):
    """
    Function to compute the approximation of the term including the integral of coagulation flux with DG scheme k>0

    Parameters
    ----------
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    tensor_tabintflux_coag : 6D array (dim = (nbins,kpol+1,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate the term including the intgegral of the coagulation flux


    Returns
    -------
    intflux : 2D array (dim = (nbins,kpol+1)), type -> float
       approximation of the term including the integral of coagulation flux at each bin

    """
    # create 4d array with element gij*gij
    arr_gij = np.einsum("ac,bd->abcd", gij, gij)

    nbins = len(gij[:, 0])
    dim_kpol = len(gij[0, :])
    intflux = np.zeros((nbins, dim_kpol))
    for j in range(nbins):
        for k in range(1, dim_kpol):
            intflux[j, k] = np.sum(tensor_tabintflux_coag[j, k, :, :, :, :] * arr_gij)

    return intflux


def compute_intflux_coag_kdv(gij, tensor_tabintflux_coag, dv):
    """
    Function to compute the approximation of the term including the integral of coagulation flux with DG scheme k>0
    Function for ballistic kernel with differential velocities dv

    Parameters
    ----------
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    tensor_tabintflux_coag : 6D array (dim = (nbins,kpol+1,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate the term including the integral of the coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains

    Returns
    -------
    intflux : 2D array (dim = (nbins,kpol+1)), type -> float
       approximation of the term including the integral of coagulation flux at each bin

    """
    # create 4d array with element gij*gij
    arr_gij_dv = np.einsum("ac,bd,ab->abcd", gij, gij, dv)

    nbins = len(gij[:, 0])
    dim_kpol = len(gij[0, :])
    intflux = np.zeros((nbins, dim_kpol))
    for j in range(nbins):
        for k in range(1, dim_kpol):
            intflux[j, k] = np.sum(tensor_tabintflux_coag[j, k, :, :, :, :] * arr_gij_dv)

    return intflux


# -------------------------------
# functions for fragmentation
# -------------------------------


def compute_flux_frag_k0(gij, tensor_tabflux_coag, tensor_tabflux_frag_T1, tensor_tabflux_frag_T2):
    """
    Function to compute the approximation of the fragmentation flux with DG scheme k=0, for simple collision kernels

    Frag flux is calculated from the expression (similar to coag) dg/dt + dF/dm = 0
    -> frag flux is defined from small to large masses
    -> frag flux is defined on the right boundary of bin j, i.e. flux(j) ~ F(m_{j+1/2})

    Parameters
    ----------
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation term in the fragmentation flux
    tensor_tabflux_frag_T1 : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate the term 1 for fragmentation flux
    tensor_tabflux_frag_T2 : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate the term 2 for fragmentation flux


    Returns
    -------
    flux : 1D array (dim = nbins), type -> float
       approximation of the fragmentation flux at each bin

    """

    # use Einstein summation over all the pair interaction (l,m) for each j.
    # compared to the definition of tensor_tabflux l <-> lp and m <-> l
    flux = (
        np.einsum("jlm,l,m->j", tensor_tabflux_coag, gij, gij)
        - np.einsum("jlm,l,m->j", tensor_tabflux_frag_T1, gij, gij)
        - np.einsum("jlm,l,m->j", tensor_tabflux_frag_T2, gij, gij)
    )

    return flux


def compute_flux_frag_k0_kdv(
    gij, tensor_tabflux_coag, tensor_tabflux_frag_T1, tensor_tabflux_frag_T2, dv
):
    """
    Function to compute the approximation of the coagulation flux with DG scheme k=0, for physical collision kernel

    Frag flux is calculated from the expression (similar to coag) dg/dt + dF/dm = 0
    -> frag flux is defined from small to large masses
    -> frag flux is defined on the right boundary of bin j, i.e. flux(j) ~ F(m_{j+1/2})

    Function for ballistic kernel with differential velocities dv

    Parameters
    ----------
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    tensor_tabflux_frag_T1 : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate the term 1 for fragmentation flux
    tensor_tabflux_frag_T2 : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate the term 2 for fragmentation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains


    Returns
    -------
    flux : 1D array (dim = nbins), type -> float
       approximation of the fragmentation flux at each bin

    """

    # use Einstein summation over all the pair interaction (l,m) for each j.
    # compared to the definition of tensor_tabflux l <-> lp and m <-> l
    flux = (
        np.einsum("jlm,lm,l,m->j", tensor_tabflux_coag, dv, gij, gij)
        - np.einsum("jlm,lm,l,m->j", tensor_tabflux_frag_T1, dv, gij, gij)
        - np.einsum("jlm,lm,l,m->j", tensor_tabflux_frag_T2, dv, gij, gij)
    )

    return flux


# -------------------------------
# functions for coagulation + fragmentation
# -------------------------------
def compute_flux_coag_frag_k0_kdv(
    gij,
    tensor_tabflux_coag,
    tensor_tabflux_frag_T1,
    tensor_tabflux_frag_T2,
    dv,
    proba_coag,
    proba_frag,
):
    """
    Function to compute the approximation of the coagulation flux with DG scheme k=0, for physical collision kernel

    Frag flux is calculated from the expression (similar to coag) dg/dt + dF/dm = 0
    -> frag flux is defined from small to large masses
    -> frag flux is defined on the right boundary of bin j, i.e. flux(j) ~ F(m_{j+1/2})

    Function for ballistic kernel with differential velocities dv

    Parameters
    ----------
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    tensor_tabflux_frag_T1 : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate the term 1 for fragmentation flux
    tensor_tabflux_frag_T2 : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate the term 2 for fragmentation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains
    proba_coag : 2D array (dim = (nbins,nbins)), type -> float
       probability to have coagulation (depending on v_frag and dv)
    proba_frag : 2D array (dim = (nbins,nbins)), type -> float
       probability to have fragmentation (depending on v_frag and dv)


    Returns
    -------
    flux : 1D array (dim = nbins), type -> float
       approximation of the fragmentation flux at each bin

    """

    # use Einstein summation over all the pair interaction (l,m) for each j.
    # compared to the definition of tensor_tabflux l <-> lp and m <-> l
    dv_coag = proba_coag * dv
    dv_frag = proba_frag * dv

    flux = (
        np.einsum("jlm,lm,l,m->j", tensor_tabflux_coag, dv_coag + dv_frag, gij, gij)
        - np.einsum("jlm,lm,l,m->j", tensor_tabflux_frag_T1, dv_frag, gij, gij)
        - np.einsum("jlm,lm,l,m->j", tensor_tabflux_frag_T2, dv_frag, gij, gij)
    )

    return flux
