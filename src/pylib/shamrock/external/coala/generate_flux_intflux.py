import numpy as np


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
