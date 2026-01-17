import sys

import numpy as np
from scipy.special import eval_legendre

from .utils_polynomials import *


# generate gij components for k=0 from g(x)=x*exp(-x)
def L2proj_k0(eps, nbins, N0, m0, massgrid, massbins, Q, vecnodes, vecweights):
    """
    Function to compute the initial gij coefficient from the initial function g(m) = m*N0/m0*exp(-m/m0)

    DG scheme k=0, piecewise constant approximation

    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    nbins : scalar, type -> integer
       number of dust bins
    N0 : scalar, type -> float
       initial total number density of grains
    m0 : scalar, type -> float
       initial mean mass of grains
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
       arithmetic mean value of massgrid for each mass bins
    Q : scalar, type -> integer
       number of points for Gauss-Legendre quadrature
    vecnodes : 1D array (dim = Q), type -> float
       nodes of the Legendre polynomials
    vecweights : 1D array (dim = Q), type -> float
       weights coefficients for the Gauss-Legendre polynomials


    Returns
    -------
    gij : 1D array (dim = nbins), type -> float
       initial components of g on the polynomial basis

    """

    mat_vecweights = np.outer(np.ones(nbins), vecweights)
    mat_xjalpha = np.outer(massbins, np.ones(Q)) + np.outer(
        (massgrid[1:] - massgrid[0:nbins]) / 2.0, vecnodes
    )

    term_sum = np.sum(mat_vecweights * N0 * mat_xjalpha / m0 * np.exp(-mat_xjalpha / m0), axis=1)

    gij = term_sum / coeff_norm_leg(0)
    gij[gij < eps] = eps

    return gij


def L2projDL_k0(eps, nbins, massgrid, massbins, Q, vecnodes, vecweights):
    """
    Function to compute the initial gij coefficient from the dimensionless initial function g(x)=x*exp(-x)

    DG scheme k=0, piecewise constant approximation

    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
       arithmetic mean value of massgrid for each mass bins
    Q : scalar, type -> integer
       number of points for Gauss-Legendre quadrature
    vecnodes : 1D array (dim = Q), type -> float
       nodes of the Legendre polynomials
    vecweights : 1D array (dim = Q), type -> float
       weights coefficients for the Gauss-Legendre polynomials


    Returns
    -------
    gij : 1D array (dim = nbins), type -> float
       initial components of g on the polynomial basis

    """
    mat_vecweights = np.outer(np.ones(nbins), vecweights)
    mat_xjalpha = np.outer(massbins, np.ones(Q)) + np.outer(
        (massgrid[1:] - massgrid[0:nbins]) / 2.0, vecnodes
    )

    term_sum = np.sum(mat_vecweights * mat_xjalpha * np.exp(-mat_xjalpha), axis=1)

    gij = term_sum / coeff_norm_leg(0)
    gij[gij < eps] = eps

    return gij


# genreate gij components for k>0 from g(x)=x*exp(-x)
def L2proj(eps, nbins, kpol, N0, m0, massgrid, massbins, Q, vecnodes, vecweights):
    """
    Function to compute the initial gij coefficient from the initial function g(m) = m*N0/m0*exp(-m/m0)

    DG scheme k>0, piecewise polynomial approximation

    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    nbins : scalar, type -> integer
       number of dust bins
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    N0 : scalar, type -> float
       initial total number density of grains
    m0 : scalar, type -> float
       initial mean mass of grains
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
       arithmetic mean value of massgrid for each mass bins
    Q : scalar, type -> integer
       number of points for Gauss-Legendre quadrature
    vecnodes : 1D array (dim = Q), type -> float
       nodes of the Legendre polynomials
    vecweights : 1D array (dim = Q), type -> float
       weights coefficients for the Gauss-Legendre polynomials


    Returns
    -------
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       initial components of g on the polynomial basis

    """

    gij = np.zeros((nbins, kpol + 1))
    for j in range(nbins):
        xj = massbins[j]
        hj = massgrid[j + 1] - massgrid[j]

        for k in range(kpol + 1):
            term_sum = 0.0
            for alpha in range(Q):
                xjalpha = xj + hj * vecnodes[alpha] / 2.0

                term_sum += (
                    vecweights[alpha]
                    * N0
                    * xjalpha
                    / m0
                    * np.exp(-xjalpha / m0)
                    * eval_legendre(k, vecnodes[alpha])
                )

            gij[j, k] = term_sum / coeff_norm_leg(k)

    if gij[:, 0][gij[:, 0] < 0.0].any():
        print("gij", gij)
        sys.exit()
    else:
        gij[:, 1:][gij[:, 0] < eps] = 0.0
        gij[:, 0][gij[:, 0] < eps] = eps

    return gij


def L2projDL(eps, nbins, kpol, massgrid, massbins, Q, vecnodes, vecweights):
    """
    Function to compute the initial gij coefficient from the dimensionless initial function g(x)=x*exp(-x)

    DG scheme k>0, piecewise polynomial approximation

    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    nbins : scalar, type -> integer
       number of dust bins
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
       arithmetic mean value of massgrid for each mass bins
    Q : scalar, type -> integer
       number of points for Gauss-Legendre quadrature
    vecnodes : 1D array (dim = Q), type -> float
       nodes of the Legendre polynomials
    vecweights : 1D array (dim = Q), type -> float
       weights coefficients for the Gauss-Legendre polynomials


    Returns
    -------
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       initial components of g on the polynomial basis

    """

    gij = np.zeros((nbins, kpol + 1))
    for j in range(nbins):
        xj = massbins[j]
        hj = massgrid[j + 1] - massgrid[j]

        for k in range(kpol + 1):
            term_sum = 0.0
            for alpha in range(Q):
                xjalpha = xj + hj * vecnodes[alpha] / 2.0

                term_sum += (
                    vecweights[alpha]
                    * xjalpha
                    * np.exp(-xjalpha)
                    * eval_legendre(k, vecnodes[alpha])
                )

            gij[j, k] = term_sum / coeff_norm_leg(k)

    if gij[:, 0][gij[:, 0] < 0.0].any():
        print("gij", gij)
        sys.exit()
    else:
        gij[:, 1:][gij[:, 0] < eps] = 0.0
        gij[:, 0][gij[:, 0] < eps] = eps

    return gij
