import numpy as np

from .generate_flux_intflux import compute_flux_coag_k0_kdv


def coala_source_term_k0(nbins, massgrid, rhodust, rhodust_eps, tensor_tabflux_coag, v_dust):
    """
    Function to compute the source for coagulation and fragmentation in continuity equation for piecewise constant approximation (see Lombart et al., 2021)
    Function for ballistic kernel with differential velocities dv
    Used to evaluate the source term, then hydro code applies time solver

    /!\ Only coagulation so far

    Parameters
    ----------
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    rhodust : 1D array (dim = nbins), type -> float
       dust density for each grain size
    rhodust_eps : scalar, type -> float
       threshold value for rhodust
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    v_dust : 1D array (dim = (nbins)), type -> float
       array of the dust velocities (could also be delta_v in monofluid since it is a delta)

    Returns
    -------
    S_coag : 1D array (dim = nbins), type -> float
        Source term for dust coagulation in continuity equation
        DG operator for piecewise constant approximation in each binls

    """

    # compute gij from rhodust for coala k=0
    gij = np.zeros(nbins)  # shape is 1D with k0
    for j in range(nbins):
        if rhodust[j] > rhodust_eps:
            gij[j] = rhodust[j] / (massgrid[j + 1] - massgrid[j])

    # dv_ij = v_dust_j - v_dust_i
    dv = v_dust[None, :] - v_dust[:, None]

    # compute flux for all dust bins
    flux = compute_flux_coag_k0_kdv(gij, tensor_tabflux_coag, dv)

    S_coag = np.zeros(nbins)
    S_coag[0] = -flux[0]
    S_coag[1:] = flux[:-1] - flux[1:]

    return S_coag


import time

from scipy.special import legendre

from .generate_tabflux_tabintflux import compute_coagtabflux_k0_numba
from .utils_polynomials import legendre_coeffs


def coala_precalc_tabflux_coag(K0, nbins, Q, massgrid):
    """
    Function to iterate coagulation solver to reach the time ndthydro x dthydro

    Function for ballistic kernel with differential velocities dv

    DG scheme k=0, piecewise constant approximation

    Parameters
    ----------
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    nbins : scalar, type -> integer
       number of dust bins
    Q : scalar, type -> integer
       number of points for Gauss-Legendre quadrature
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins


    Returns
    -------
    gij_init : 1D array (dim = nbins) or 2D array (dim = (nbins.kpol+1)), type -> float
       initial components of g on the polynomial basis
    gij : 1D array (dim = nbins) or 2D array (dim = (nbins.kpol+1)), type -> float
       evolved components of g on the polynomial basis
    time_coag : scalar, type -> float
       final time ndthydro x dthydro

    """

    kernel = 3
    kpol = 0

    vecnodes, vecweights = np.polynomial.legendre.leggauss(Q)

    # Legendre polynomial coefficients
    mat_coeffs_leg = np.zeros((kpol + 1, kpol + 1))
    mat_coeffs_leg = legendre_coeffs(kpol)

    start = time.time()
    tensor_tabflux_coag = np.zeros((nbins, nbins, nbins))

    compute_coagtabflux_k0_numba(
        kernel,
        K0,
        Q,
        vecnodes,
        vecweights,
        nbins,
        massgrid,
        mat_coeffs_leg,
        tensor_tabflux_coag,
    )

    finish = time.time()
    print("Tensor tabflux generated in %.5f s" % (finish - start))

    return tensor_tabflux_coag
