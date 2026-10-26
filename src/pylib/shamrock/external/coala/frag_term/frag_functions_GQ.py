import sys

import numpy as np
from numba import jit, njit, prange
from scipy.special import legendre

from ..kernel_collision import *
from ..utils_polynomials import *


@njit
def int_distrib_frag_generic(frac_mass_fragments, beta, xmin, x, u, v):
    """
    Function to evaluate the analytic integral on mass fragment of the distribution of fragment
    in the form b(w;u,v) = A.w^beta + delta(w - m_left)

    Parameters
    ----------
    frac_mass_fragments : scalar, type -> float
        mass fraction of total mass (u+v) represented by fragments, m_frag = frac_mass_fragments*(u+v)
    beta : scalar, type -> float
        coefficient for power law fragment distribution in mass range [xmin,mfrag]
    xmin : scalar, type -> float
        minimum mass of massgrid, here lower mass limit for integral of distrib of fragments
    x : scalar, type -> float
        upper mass limit for integral of distrib of fragments
    u : scalar, type -> float
        mass variable (colliding grain of mass u)
    v : scalar, type -> float
        mass variable (colliding grain of mass v)

    Returns
    -------
    int_distrib_frag_generic : scalar, type -> float
        integral over w of w.b(w;u,v)

    """

    m_frag = frac_mass_fragments * (u + v)
    m_left = u + v - m_frag

    # condition if m_left in [xmin,x] for delta function
    if xmin < m_left < x:
        res = (
            m_frag
            * (x ** (2.0 + beta) - xmin ** (2.0 + beta))
            / (m_frag ** (2.0 + beta) - xmin ** (2.0 + beta))
            + m_left
        )
    else:
        res = (
            m_frag
            * (x ** (2.0 + beta) - xmin ** (2.0 + beta))
            / (m_frag ** (2.0 + beta) - xmin ** (2.0 + beta))
        )

    return res


def func_frag_flux(kernel, K0, ai, aip, frac_mass_fragments, beta, xmin, x, u, v, xilp, xil):
    """
    Function to evaluate the integrand of the fragmentation flux, i.e. integrand of the double integral

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
    frac_mass_fragments : scalar, type -> float
        mass fraction of total mass (u+v) represented by fragments, m_frag = frac_mass_fragments*(u+v)
    beta : scalar, type -> float
        coefficient for power law fragment distribution in mass range [xmin,mfrag]
    xmin : scalar, type -> float
        minimum mass of massgrid, here lower mass limit for integral of distrib of fragments
    x : scalar, type -> float
        upper mass limit for integral of distrib of fragments
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
    func_frag_flux : scalar, type -> float
       integrand of fragmentation flux evaluated at u,v,xilp,xil

    """

    return (
        int_distrib_frag_generic(frac_mass_fragments, beta, xmin, x, u, v)
        * func_kernel(kernel, K0, u, v)
        * phi_pol(aip, xilp)
        * phi_pol(ai, xil)
        / (v * u)
    )


def fragfluxfunction_T1(
    kernel,
    K0,
    Q,
    vecnodes,
    vecweights,
    nbins,
    massgrid,
    mat_coeffs_leg,
    j,
    lp,
    l,
    ip,
    i,
    frac_mass_fragments,
    beta,
):
    """
    Function to evaluate the integral of the term 1 for fragmentation flux with Gauss-Legendre quadrature method.
    This function is used to calculate the array for the fragmentation flux as precomputation.

    Frag flux is calculated from the expression (similar to coag) dg/dt + dF/dm = 0
    -> frag flux is defined from small to large masses
    -> frag flux is defined on the right boundary of bin j

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
    frac_mass_fragments : scalar, type -> float
        mass fraction of total mass (u+v) represented by fragments, m_frag = frac_mass_fragments*(u+v)
    beta : scalar, type -> float
        coefficient for power law fragment distribution in mass range [xmin,mfrag]


    Returns
    -------
    fragfluxfunction : scalar, type -> float
       double integral for the fragulation flux evaluated at j,lp,l,ip,i

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

        a_vl = np.max([xjgridr - ulp_alpha + xmin, xlgridl])
        b_vl = np.min([xmax - ulp_alpha + xmin, xlgridr])

        if xmax - ulp_alpha + xmin > xlgridl and xlgridr > xjgridr - ulp_alpha + xmin:
            # gauss-legendre quadrature on integral on v
            for alpha_v in range(Q):
                vl_alpha = 0.5 * (b_vl + a_vl) + 0.5 * (b_vl - a_vl) * vecnodes[alpha_v]
                xil = 2.0 * (vl_alpha - xl) / hl

                res += (
                    0.25
                    * hlp
                    * (b_vl - a_vl)
                    * vecweights[alpha_u]
                    * vecweights[alpha_v]
                    * func_frag_flux(
                        kernel,
                        K0,
                        ai,
                        aip,
                        frac_mass_fragments,
                        beta,
                        xmin,
                        xjgridr,
                        ulp_alpha,
                        vl_alpha,
                        xilp,
                        xil,
                    )
                )

                res = 0.5 * res

    return res


def fragfluxfunction_T2(
    kernel,
    K0,
    Q,
    vecnodes,
    vecweights,
    nbins,
    massgrid,
    mat_coeffs_leg,
    j,
    lp,
    l,
    ip,
    i,
    frac_mass_fragments,
    beta,
):
    """
    Function to evaluate the integral of the term 2 for fragmentation flux with Gauss-Legendre quadrature method.
    This function is used to calculate the array for the fragmentation flux as precomputation.

    Frag flux is calculated from the expression (similar to coag) dg/dt + dF/dm = 0
    -> frag flux is defined from small to large masses
    -> frag flux is defined on the right boundary of bin j

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
    frac_mass_fragments : scalar, type -> float
        mass fraction of total mass (u+v) represented by fragments, m_frag = frac_mass_fragments*(u+v)
    beta : scalar, type -> float
        coefficient for power law fragment distribution in mass range [xmin,mfrag]


    Returns
    -------
    fragfluxfunction : scalar, type -> float
       double integral for the fragulation flux evaluated at j,lp,l,ip,i

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

        a_vl = xlgridl
        b_vl = np.min([xmax - ulp_alpha + xmin, xlgridr])

        if xmax - ulp_alpha + xmin > xlgridl and xlgridr > xmin:
            # gauss-legendre quadrature on integral on v
            for alpha_v in range(Q):
                vl_alpha = 0.5 * (b_vl + a_vl) + 0.5 * (b_vl - a_vl) * vecnodes[alpha_v]
                xil = 2.0 * (vl_alpha - xl) / hl

                res += (
                    0.25
                    * hlp
                    * (b_vl - a_vl)
                    * vecweights[alpha_u]
                    * vecweights[alpha_v]
                    * func_frag_flux(
                        kernel,
                        K0,
                        ai,
                        aip,
                        frac_mass_fragments,
                        beta,
                        xmin,
                        xjgridr,
                        ulp_alpha,
                        vl_alpha,
                        xilp,
                        xil,
                    )
                )

                res = 0.5 * res

    return res


@njit
def func_frag_flux_numba(kernel, K0, ai, aip, frac_mass_fragments, beta, xmin, x, u, v, xilp, xil):
    """
    Function to evaluate the integrand of the fragmentation flux, i.e. integrand of the double integral

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
    frac_mass_fragments : scalar, type -> float
        mass fraction of total mass (u+v) represented by fragments, m_frag = frac_mass_fragments*(u+v)
    beta : scalar, type -> float
        coefficient for power law fragment distribution in mass range [xmin,mfrag]
    xmin : scalar, type -> float
        minimum mass of massgrid, here lower mass limit for integral of distrib of fragments
    x : scalar, type -> float
        upper mass limit for integral of distrib of fragments
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
    func_frag_flux : scalar, type -> float
       integrand of fragmentation flux evaluated at u,v,xilp,xil

    """

    return (
        int_distrib_frag_generic(frac_mass_fragments, beta, xmin, x, u, v)
        * func_kernel_numba(kernel, K0, u, v)
        * phi_pol(aip, xilp)
        * phi_pol(ai, xil)
        / (v * u)
    )


@njit
def fragfluxfunction_T1_numba(
    kernel,
    K0,
    Q,
    vecnodes,
    vecweights,
    nbins,
    massgrid,
    mat_coeffs_leg,
    j,
    lp,
    l,
    ip,
    i,
    frac_mass_fragments,
    beta,
):
    """
    Function to evaluate the integral of the term 1 for fragmentation flux with Gauss-Legendre quadrature method.
    This function is used to calculate the array for the fragmentation flux as precomputation.

    Frag flux is calculated from the expression (similar to coag) dg/dt + dF/dm = 0
    -> frag flux is defined from small to large masses
    -> frag flux is defined on the right boundary of bin j

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
    frac_mass_fragments : scalar, type -> float
        mass fraction of total mass (u+v) represented by fragments, m_frag = frac_mass_fragments*(u+v)
    beta : scalar, type -> float
        coefficient for power law fragment distribution in mass range [xmin,mfrag]


    Returns
    -------
    fragfluxfunction : scalar, type -> float
       double integral for the fragulation flux evaluated at j,lp,l,ip,i

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

        a_vl = max(xjgridr - ulp_alpha + xmin, xlgridl)
        b_vl = min(xmax - ulp_alpha + xmin, xlgridr)

        if xmax - ulp_alpha + xmin > xlgridl and xlgridr > xjgridr - ulp_alpha + xmin:
            # gauss-legendre quadrature on integral on v
            for alpha_v in range(Q):
                vl_alpha = 0.5 * (b_vl + a_vl) + 0.5 * (b_vl - a_vl) * vecnodes[alpha_v]
                xil = 2.0 * (vl_alpha - xl) / hl

                res += (
                    0.5
                    * 0.25
                    * hlp
                    * (b_vl - a_vl)
                    * vecweights[alpha_u]
                    * vecweights[alpha_v]
                    * func_frag_flux_numba(
                        kernel,
                        K0,
                        ai,
                        aip,
                        frac_mass_fragments,
                        beta,
                        xmin,
                        xjgridr,
                        ulp_alpha,
                        vl_alpha,
                        xilp,
                        xil,
                    )
                )

    return res


@njit
def fragfluxfunction_T2_numba(
    kernel,
    K0,
    Q,
    vecnodes,
    vecweights,
    nbins,
    massgrid,
    mat_coeffs_leg,
    j,
    lp,
    l,
    ip,
    i,
    frac_mass_fragments,
    beta,
):
    """
    Function to evaluate the integral of the term 2 for fragmentation flux with Gauss-Legendre quadrature method.
    This function is used to calculate the array for the fragmentation flux as precomputation.

    Frag flux is calculated from the expression (similar to coag) dg/dt + dF/dm = 0
    -> frag flux is defined from small to large masses
    -> frag flux is defined on the right boundary of bin j

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
    frac_mass_fragments : scalar, type -> float
        mass fraction of total mass (u+v) represented by fragments, m_frag = frac_mass_fragments*(u+v)
    beta : scalar, type -> float
        coefficient for power law fragment distribution in mass range [xmin,mfrag]


    Returns
    -------
    fragfluxfunction : scalar, type -> float
       double integral for the fragulation flux evaluated at j,lp,l,ip,i

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

        a_vl = xlgridl
        b_vl = min(xmax - ulp_alpha + xmin, xlgridr)

        if xmax - ulp_alpha + xmin > xlgridl and xlgridr > xmin:
            # gauss-legendre quadrature on integral on v
            for alpha_v in range(Q):
                vl_alpha = 0.5 * (b_vl + a_vl) + 0.5 * (b_vl - a_vl) * vecnodes[alpha_v]
                xil = 2.0 * (vl_alpha - xl) / hl

                res += (
                    0.5
                    * 0.25
                    * hlp
                    * (b_vl - a_vl)
                    * vecweights[alpha_u]
                    * vecweights[alpha_v]
                    * func_frag_flux_numba(
                        kernel,
                        K0,
                        ai,
                        aip,
                        frac_mass_fragments,
                        beta,
                        xmin,
                        xjgridr,
                        ulp_alpha,
                        vl_alpha,
                        xilp,
                        xil,
                    )
                )

    return res
