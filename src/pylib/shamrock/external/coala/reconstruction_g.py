import numpy as np

from .utils_polynomials import *


def recons_g(massgrid, massbins, kpol, gij, j, x):
    """
    Function to reconstruct the approximation of g using the components gij and the Legendre polynomial basis

    only for DG scheme k>0, piecewise polynomial approximation

    Parameters
    ----------
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
       arithmetic mean value of massgrid for each mass bins
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    j : scalar, type -> integer
       index of the bin for reconstruction
    x : scalar, type -> float
       mass value to evaluate the reconstruction of g


    Returns
    -------
    res : scalar, type -> float
       reconstruction of g evaluated at x

    """

    # to map bin j onto [-1,1]
    xij = 2.0 / (massgrid[j + 1] - massgrid[j]) * (x - massbins[j])
    if kpol == 0:
        res = np.polynomial.legendre.legval(xij, gij[j])
    else:
        res = np.polynomial.legendre.legval(xij, gij[j, :])

    return res
