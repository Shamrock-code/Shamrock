import sys

import numpy as np
from scipy.optimize import minimize_scalar

from .reconstruction_g import *


def minval_approx_g(nbins, kpol, massgrid, massbins, gij):
    """
    Function to compute the minimum value of the approximation of g in each bin

    only for DG scheme k>0, piecewise polynomial approximation

    Parameters
    ----------
    nbins : scalar, type -> integer
       number of dust bins
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
       arithmetic mean value of massgrid for each mass bins
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis


    Returns
    -------
    tab_minval_recons_g : 1D array (dim = nbins), type -> float
       minimum value of the approximation of in each bin

    """

    tab_minval_recons_g = np.zeros(nbins)

    if kpol == 1:
        for j in range(nbins):
            # print("j=",j,", recons_g(massgrid[j])=",recons_g(massgrid,massbins,kpol,gij,j,massgrid[j]))
            tab_minval_recons_g[j] = min(
                recons_g(massgrid, massbins, kpol, gij, j, massgrid[j]),
                recons_g(massgrid, massbins, kpol, gij, j, massgrid[j + 1]),
            )

    else:
        for j in range(nbins):
            func_pol = lambda x: recons_g(massgrid, massbins, kpol, gij, j, x)

            xjgridl = massgrid[j]
            xjgridr = massgrid[j + 1]

            # brute force
            # xgrid = np.linspace(xjgridl,xjgridr,num=100)
            # tab_minval_recons_g[j] = np.min(func_pol(xgrid))

            # scipy algorithm
            res = minimize_scalar(func_pol, bounds=(xjgridl, xjgridr), method="bounded")
            tab_minval_recons_g[j] = np.min(
                [res.fun, func_pol(xjgridl), func_pol(xjgridr)]
            )  # Accurate minimum value

    return tab_minval_recons_g


def gammafunction(eps, nbins, kpol, massgrid, massbins, gij):
    """
    Function to compute the limiter coefficient to ensure positivity of the numerical solution (Zhang and Shu 2010)

    only for DG scheme k>0, piecewise polynomial approximation

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
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis


    Returns
    -------
    tab_gamma : 1D array (dim = nbins), type -> float
       limiter coefficient in each bin

    """

    # Liu et al.2019
    min_approx_g = minval_approx_g(nbins, kpol, massgrid, massbins, gij)

    # print("min_approx_g=",min_approx_g)

    tab_gamma = np.asarray(
        [
            np.min([1.0, np.abs((eps - gij[j, 0]) / (gij[j, 0] - min_approx_g[j]))])
            if gij[j, 0] != min_approx_g[j]
            else 1.0
            for j in range(nbins)
        ]
    )

    return tab_gamma
