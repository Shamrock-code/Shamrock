import sys

import numpy as np
# from numba import njit
from scipy.special import legendre

from ..kernel_collision import *
from ..utils_polynomials import *


def func_coag_flux(kernel, K0, ai, aip, u, v, xilp, xil):
    """
    Function to evaluate the integrand of the coagulation flux, i.e. integrand of the double integral

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    ai : 1D array (dim = i+1), type -> float
       coefficients of polynomial of degree i
    aip : 1D array (dim = ip+1), type - > float
       coefficients of polynomial of degree ip
    u : scalar, type -> float
       mass variable (colliding grain of mass u)
    v : scalar, type -> float
       mass variable (colliding grain of mass v)
    xilp : scalar, type -> float
       variable mapping the mass bin lp in [-1,1], needed for Legendre polynomials
    xil : scalar, type -> float
       variable mapping the mass bin l in [-1,1], need for Legendre polynomials

    Returns
    -------
    func_coag_flux : scalar, type -> float
       integrand of cogulation flux evaluated at u,v,xilp,xil

    """

    return func_kernel(kernel, K0, u, v) * phi_pol(aip, xilp) * phi_pol(ai, xil) / v


def coagfluxfunction(
    kernel, K0, Q, vecnodes, vecweights, nbins, massgrid, mat_coeffs_leg, j, lp, l, ip, i
):
    """
    Function to evaluate the double integral depending only masses with Gauss-Legendre quadrature method.
    This function is used to calculate the array for the coagulation flux as precomputation.

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    Q : scalar, type -> integer
       number of points for Gauss-Legendre quadrature
    vecnodes : 1D array (dim = Q), type -> float
       nodes of the Legendre polynomials
    vecweights : 1D array (dim = Q), type -> float
       weights coefficients for the Gauss-Legendre polynomials
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    mat_coeffs_leg : 2D array (dmin = (kpol+1,kpol+1)), type -> float
       array containing on each line Legendre polynomial coefficients from degree 0 to kpol inclusive
       on each line coefficients are ordered from low to high orders
    j : scalar, type -> integer
       index corresponding to the mass of the new formed grain
    lp : scarlar, type -> integer
       index corresponding to the mass of one colliding grain
    l : scarlar, type -> integer
       index corresponding to the mass of the second colliding grain
    ip : scalar, type -> integer
       degree of polynomials for approximation in bin lp
    i : scalar, type -> integer
       degree of polynomials for approximation in bin l


    Returns
    -------
    coagfluxfunction : scalar, type -> float
       double integral for the coagulation flux evaluated at j,lp,l,ip,i

    """

    # borders, size and mean values of mass bins of indices j,lp,l
    xlgridr = massgrid[l + 1]
    xlgridl = massgrid[l]
    xlpgridl = massgrid[lp]
    xlpgridr = massgrid[lp + 1]

    hlp = xlpgridr - xlpgridl
    xlp = 0.5 * (xlpgridr + xlpgridl)

    hl = xlgridr - xlgridl
    xl = 0.5 * (xlgridr + xlgridl)

    xjgridr = massgrid[j + 1]

    # minimum and maximum value of mass range
    xmin = massgrid[0]
    xmax = massgrid[-1]

    # polynomials coefficients (low to high order)
    aip = mat_coeffs_leg[ip, : ip + 1]
    ai = mat_coeffs_leg[i, : i + 1]

    # initialisation double integral
    res = 0.0

    # gauss-legendre quadrature on integral on u
    for alpha_u in range(Q):
        ulp_alpha = xlp + 0.5 * hlp * vecnodes[alpha_u]
        xilp = vecnodes[alpha_u]

        # gauss-legendre quadrature on integral on v
        for alpha_v in range(Q):
            a_vl = np.max([xjgridr - ulp_alpha + xmin, xlgridl])
            b_vl = np.min([xmax - ulp_alpha + xmin, xlgridr])
            vl_alpha = 0.5 * (b_vl + a_vl) + 0.5 * (b_vl - a_vl) * vecnodes[alpha_v]
            xil = 2.0 * (vl_alpha - xl) / hl

            if xmax - ulp_alpha + xmin > xlgridl and xlgridr > xjgridr - ulp_alpha + xmin:
                res += (
                    0.25
                    * hlp
                    * (b_vl - a_vl)
                    * vecweights[alpha_u]
                    * vecweights[alpha_v]
                    * func_coag_flux(kernel, K0, ai, aip, ulp_alpha, vl_alpha, xilp, xil)
                )

    return res


# @njit
def func_coag_flux_numba(kernel, K0, ai, aip, u, v, xilp, xil):
    """
    Function to evaluate the integrand of the coagulation flux, i.e. integrand of the double integral
    Numba formalism

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    ai : 1D array (dim = i+1), type -> float
       coefficients of polynomial of degree i
    aip : 1D array (dim = ip+1), type - > float
       coefficients of polynomial of degree ip
    u : scalar, type -> float
       mass variable (colliding grain of mass u)
    v : scalar, type -> float
       mass variable (colliding grain of mass v)
    xilp : scalar, type -> float
       variable mapping the mass bin lp in [-1,1], needed for Legendre polynomials
    xil : scalar, type -> float
       variable mapping the mass bin l in [-1,1], need for Legendre polynomials

    Returns
    -------
    func_coag_flux : scalar, type -> float
       integrand of cogulation flux evaluated at u,v,xilp,xil

    """

    return func_kernel_numba(kernel, K0, u, v) * phi_pol(aip, xilp) * phi_pol(ai, xil) / v


# @njit
def coagfluxfunction_numba(
    kernel, K0, Q, vecnodes, vecweights, nbins, massgrid, mat_coeffs_leg, j, lp, l, ip, i
):
    """
    Function to evaluate the double integral depending only masses with Gauss-Legendre quadrature method.
    This function is used to calculate the array for the coagulation flux as precomputation.
    Numba formalism

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    Q : scalar, type -> integer
       number of points for Gauss-Legendre quadrature
    vecnodes : 1D array (dim = Q), type -> float
       nodes of the Legendre polynomials
    vecweights : 1D array (dim = Q), type -> float
       weights coefficients for the Gauss-Legendre polynomials
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    mat_coeffs_leg : 2D array (dmin = (kpol+1,kpol+1)), type -> float
       array containing on each line Legendre polynomial coefficients from degree 0 to kpol inclusive
       on each line coefficients are ordered from low to high orders
    j : scalar, type -> integer
       index corresponding to the mass of the new formed grain
    lp : scarlar, type -> integer
       index corresponding to the mass of one colliding grain
    l : scarlar, type -> integer
       index corresponding to the mass of the second colliding grain
    ip : scalar, type -> integer
       degree of polynomials for approximation in bin lp
    i : scalar, type -> integer
       degree of polynomials for approximation in bin l


    Returns
    -------
    coagfluxfunction : scalar, type -> float
       double integral for the coagulation flux evaluated at j,lp,l,ip,i

    """

    # borders, size and mean values of mass bins of indices j,lp,l
    xlgridr = massgrid[l + 1]
    xlgridl = massgrid[l]
    xlpgridl = massgrid[lp]
    xlpgridr = massgrid[lp + 1]

    hlp = xlpgridr - xlpgridl
    xlp = 0.5 * (xlpgridr + xlpgridl)

    hl = xlgridr - xlgridl
    xl = 0.5 * (xlgridr + xlgridl)

    xjgridr = massgrid[j + 1]

    # minimum and maximum value of mass range
    xmin = massgrid[0]
    xmax = massgrid[-1]

    # polynomials coefficients (low to high order)
    aip = mat_coeffs_leg[ip, : ip + 1]
    ai = mat_coeffs_leg[i, : i + 1]

    # initialisation double integral
    res = 0.0

    # gauss-legendre quadrature on integral on u
    for alpha_u in range(Q):
        ulp_alpha = xlp + 0.5 * hlp * vecnodes[alpha_u]
        xilp = vecnodes[alpha_u]

        # gauss-legendre quadrature on integral on v
        for alpha_v in range(Q):
            a_vl = max(xjgridr - ulp_alpha + xmin, xlgridl)
            b_vl = min(xmax - ulp_alpha + xmin, xlgridr)
            vl_alpha = 0.5 * (b_vl + a_vl) + 0.5 * (b_vl - a_vl) * vecnodes[alpha_v]
            xil = 2.0 * (vl_alpha - xl) / hl

            if xmax - ulp_alpha + xmin > xlgridl and xlgridr > xjgridr - ulp_alpha + xmin:
                res += (
                    0.25
                    * hlp
                    * (b_vl - a_vl)
                    * vecweights[alpha_u]
                    * vecweights[alpha_v]
                    * func_coag_flux_numba(kernel, K0, ai, aip, ulp_alpha, vl_alpha, xilp, xil)
                )

    return res


def func_coag_intflux(kernel, K0, ak, ai, aip, hj, u, v, xij, xilp, xil):
    """
    Function to evaluate the integrand of the term including the integral of the coagulation flux, i.e. integrand of the triple integral.
    See discontinuous galerkin scheme.

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    ak : 1D array (dim = kpol+1), type -> float
       coefficients of polynomial of degree k
    ai : 1D array (dim = kpol+1), type -> float
       coefficients of polynomial of degree i
    aip : 1D array (dim = kpol+1), type -> float
       coefficients of polynomial of degree ip
    hj : scalar, type -> float
       size of mass bin j
    u : scalar, type -> float
       mass variable (colliding grain of mass u)
    v : scalar, type -> float
       mass variable (colliding grain of mass v)
    xij : scalar, type -> float
       variable mapping the mass bin j in [-1,1], needed for Legendre polynomials
    xilp : scalar, type -> float
       variable mapping the mass bin lp in [-1,1], needed for Legendre polynomials
    xil : scalar, type -> float
       variable mapping the mass bin l in [-1,1], need for Legendre polynomials

    Returns
    -------
    func_coag_intflux : scalar, type -> float
       integrand of the triple integral evaluated at u,v,xij,xilp,xil

    """

    return (
        dphik(ak, hj, xij)
        * func_kernel(kernel, K0, u, v)
        * phi_pol(aip, xilp)
        * phi_pol(ai, xil)
        / v
    )


def coagintfluxfunction(
    kernel, K0, Q, vecnodes, vecweights, nbins, massgrid, mat_coeffs_leg, j, k, lp, l, ip, i
):
    """
    Function to evaluate the triple integral for coagulation (term in DG scheme) depending only masses with Gauss-Legendre quadrature method.
    This function is used to calculate the array for the term including the integral of the coagulation flux as precomputation.

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    Q : scalar, type -> integer
       number of points for Gauss-Legendre quadrature
    vecnodes : 1D array (dim = Q), type -> float
       nodes of the Legendre polynomials
    vecweights : 1D array (dim = Q), type -> float
       weights coefficients for the Gauss-Legendre polynomials
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    mat_coeffs_leg : 2D array (dmin = (kpol+1,kpol+1)), type -> float
       array containing on each line Legendre polynomial coefficients from degree 0 to kpol inclusive
       on each line coefficients are ordered from low to high orders
    j : scalar, type -> integer
       index corresponding to the mass of the new formed grain
    k : scalar, type -> integer
       degree of polynomials for approximation in bin j
    lp : scarlar, type -> integer
       index corresponding to the mass of one colliding grain
    l : scarlar, type -> integer
       index corresponding to the mass of the second colliding grain
    ip : scalar, type -> integer
       degree of polynomials for approximation in bin lp
    i : scalar, type -> integer
       degree of polynomials for approximation in bin l


    Returns
    -------
    coagintfluxfunction : scalar, type -> float
       triple integral for the term including the integral of coagulation flux evaluated at j,k,lp,l,ip,i

    """

    # borders, size and mean values of mass bins of indices j,lp,l
    xlgridr = massgrid[l + 1]
    xlgridl = massgrid[l]
    xlpgridl = massgrid[lp]
    xlpgridr = massgrid[lp + 1]

    xjgridr = massgrid[j + 1]
    xjgridl = massgrid[j]

    hlp = xlpgridr - xlpgridl
    xlp = 0.5 * (xlpgridr + xlpgridl)

    hl = xlgridr - xlgridl
    xl = 0.5 * (xlgridr + xlgridl)

    hj = xjgridr - xjgridl
    xj = 0.5 * (xjgridr + xjgridl)

    # minimum and maximum value of mass range
    xmin = massgrid[0]
    xmax = massgrid[-1]

    # polynomials coefficients (low to high order)
    aip = mat_coeffs_leg[ip, : ip + 1]
    ai = mat_coeffs_leg[i, : i + 1]
    ak = mat_coeffs_leg[k, : k + 1]

    res = 0.0

    # gauss-legendre quadrature on integral on x
    for alpha_x in range(Q):
        xj_alpha = xj + 0.5 * hj * vecnodes[alpha_x]
        xij = vecnodes[alpha_x]

        # gauss-legendre quadrature on integral on u
        for alpha_u in range(Q):
            a_ulp = np.max([xmin, xlpgridl])
            b_ulp = np.min([xj_alpha, xlpgridr])
            ulp_alpha = 0.5 * (b_ulp + a_ulp) + 0.5 * (b_ulp - a_ulp) * vecnodes[alpha_u]
            xilp = 2.0 * (ulp_alpha - xlp) / hlp

            # gauss-legendre quadrature on integral on v
            for alpha_v in range(Q):
                a_vl = np.max([xj_alpha - ulp_alpha + xmin, xlgridl])
                b_vl = np.min([xmax - ulp_alpha + xmin, xlgridr])
                vl_alpha = 0.5 * (b_vl + a_vl) + 0.5 * (b_vl - a_vl) * vecnodes[alpha_v]
                xil = 2.0 * (vl_alpha - xl) / hl

                if xmax - ulp_alpha + xmin > xlgridl and xlgridr > xj_alpha - ulp_alpha + xmin:
                    res += (
                        0.125
                        * hj
                        * (b_ulp - a_ulp)
                        * (b_vl - a_vl)
                        * vecweights[alpha_x]
                        * vecweights[alpha_u]
                        * vecweights[alpha_v]
                        * func_coag_intflux(
                            kernel, K0, ak, ai, aip, hj, ulp_alpha, vl_alpha, xij, xilp, xil
                        )
                    )

    return res


# @njit
def func_coag_intflux_numba(kernel, K0, ak, ai, aip, hj, u, v, xij, xilp, xil):
    dphik_val = dphik(ak, hj, xij)
    kernel_val = func_kernel_numba(kernel, K0, u, v)
    phi_aip_val = phi_pol(aip, xilp)
    phi_ai_val = phi_pol(ai, xil)

    return dphik_val * kernel_val * phi_aip_val * phi_ai_val / v


# @njit
def coagintfluxfunction_numba(
    kernel, K0, Q, vecnodes, vecweights, nbins, massgrid, mat_coeffs_leg, j, k, lp, l, ip, i
):
    """
    Function to evaluate the triple integral for coagulation (term in DG scheme) depending only masses with Gauss-Legendre quadrature method.
    This function is used to calculate the array for the term including the integral of the coagulation flux as precomputation.

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    Q : scalar, type -> integer
       number of points for Gauss-Legendre quadrature
    vecnodes : 1D array (dim = Q), type -> float
       nodes of the Legendre polynomials
    vecweights : 1D array (dim = Q), type -> float
       weights coefficients for the Gauss-Legendre polynomials
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    mat_coeffs_leg : 2D array (dmin = (kpol+1,kpol+1)), type -> float
       array containing on each line Legendre polynomial coefficients from degree 0 to kpol inclusive
       on each line coefficients are ordered from low to high orders
    j : scalar, type -> integer
       index corresponding to the mass of the new formed grain
    k : scalar, type -> integer
       degree of polynomials for approximation in bin j
    lp : scarlar, type -> integer
       index corresponding to the mass of one colliding grain
    l : scarlar, type -> integer
       index corresponding to the mass of the second colliding grain
    ip : scalar, type -> integer
       degree of polynomials for approximation in bin lp
    i : scalar, type -> integer
       degree of polynomials for approximation in bin l


    Returns
    -------
    coagintfluxfunction : scalar, type -> float
       triple integral for the term including the integral of coagulation flux evaluated at j,k,lp,l,ip,i

    """

    # borders, size and mean values of mass bins of indices j,lp,l
    xlgridr = massgrid[l + 1]
    xlgridl = massgrid[l]
    xlpgridl = massgrid[lp]
    xlpgridr = massgrid[lp + 1]

    xjgridr = massgrid[j + 1]
    xjgridl = massgrid[j]

    hlp = xlpgridr - xlpgridl
    xlp = 0.5 * (xlpgridr + xlpgridl)

    hl = xlgridr - xlgridl
    xl = 0.5 * (xlgridr + xlgridl)

    hj = xjgridr - xjgridl
    xj = 0.5 * (xjgridr + xjgridl)

    # minimum and maximum value of mass range
    xmin = massgrid[0]
    xmax = massgrid[-1]

    # polynomials coefficients (low to high order)
    aip = mat_coeffs_leg[ip, : ip + 1]
    ai = mat_coeffs_leg[i, : i + 1]
    ak = mat_coeffs_leg[k, : k + 1]

    res = 0.0

    # gauss-legendre quadrature on integral on x
    for alpha_x in range(Q):
        xj_alpha = xj + 0.5 * hj * vecnodes[alpha_x]
        xij = vecnodes[alpha_x]

        # gauss-legendre quadrature on integral on u
        for alpha_u in range(Q):
            a_ulp = max(xmin, xlpgridl)
            b_ulp = min(xj_alpha, xlpgridr)
            ulp_alpha = 0.5 * (b_ulp + a_ulp) + 0.5 * (b_ulp - a_ulp) * vecnodes[alpha_u]
            xilp = 2.0 * (ulp_alpha - xlp) / hlp

            # gauss-legendre quadrature on integral on v
            for alpha_v in range(Q):
                a_vl = max(xj_alpha - ulp_alpha + xmin, xlgridl)
                b_vl = min(xmax - ulp_alpha + xmin, xlgridr)
                vl_alpha = 0.5 * (b_vl + a_vl) + 0.5 * (b_vl - a_vl) * vecnodes[alpha_v]
                xil = 2.0 * (vl_alpha - xl) / hl

                if xmax - ulp_alpha + xmin > xlgridl and xlgridr > xj_alpha - ulp_alpha + xmin:
                    res += (
                        0.125
                        * hj
                        * (b_ulp - a_ulp)
                        * (b_vl - a_vl)
                        * vecweights[alpha_x]
                        * vecweights[alpha_u]
                        * vecweights[alpha_v]
                        * func_coag_intflux_numba(
                            kernel, K0, ak, ai, aip, hj, ulp_alpha, vl_alpha, xij, xilp, xil
                        )
                    )

    return res
