import numpy as np
# from numba import njit
from scipy.special import legendre


def legendre_coeffs(kpol):
    """
    Returns a matrix (kpol+1,kpol+1) containing on each line the coefficients for each Legendre polynomial,
    from degree 0 to kpol inclusive. On each line coefficients are ordered from low to high orders.

    Parameters
    ----------
    kpol : scalar, type -> float
       degree of polynomials for approximation

    Returns
    ----------
    mat_coeffs_leg : 2D array (dmin = (kpol+1,kpol+1)), type -> float
       array containing the coefficients for Legendre polynomials up to degree kpol
    """
    mat_coeffs_leg = np.zeros((kpol + 1, kpol + 1))
    for k in range(kpol + 1):
        mat_coeffs_leg[k, : k + 1] = legendre(k).c[::-1]
    return mat_coeffs_leg


# @njit
def phi_pol(pol_coeffs, x):
    """
    Evaluate polynomial sum_{i=0}^{k} a_i x^i by Horner's method

    Parameters
    ----------
    pol_coeffs : 1D array (dim = k+1), type -> float
       coefficients of polynomial of order k sort from low to high orders

    x : scalar, type -> float
       value to evaluate the polynomial

    Returns
    ----------
    result : scalar, type -> float
       evaluation of polynomial of order k at x

    """

    result = 0.0
    for c in pol_coeffs[::-1]:
        result = result * x + c
    return result


# @njit
def polynomial_derivative_coeffs(k, pol_coeffs):
    """
    Compute coefficients for the derivative of polynomial with coeff pol_coeffs.

    Parameters
    ----------
    k : scalar, type -> integer
       order of polynomials
    pol_coeffs : 1D array (dim = k+1), type -> float
       coefficients of polynomial of order k

    Returns
    ----------
    dcoeffs: 1D array (dim = k), type -> float
       coefficients of the derivative of polynomial of order k

    """

    if k == 0:
        return np.zeros(1)
    dcoeffs = np.zeros(k)
    for i in range(1, k + 1):
        dcoeffs[i - 1] = i * pol_coeffs[i]
    return dcoeffs


# @njit
def dphik(ak, hj, xij):
    """
    Derivative of P_k(xij) with respect to x, where xij = 2/hj*(x-xj)

    Parameters
    ----------
    ak : 1D array (dim = k+1), type -> float
       coefficients of polynomial of order k, sorted from low to high order
    hj : scalar, type -> float
       width of the bin
    xij : scalar, type -> float
       variable mapping the mass bin j in [-1,1], needed for Legendre polynomials

    Returns
    ----------
       Evaluation at xij of the derivative of P_k(xij) with respect to x

    """
    k = len(ak) - 1
    d_coeffs = polynomial_derivative_coeffs(k, ak)
    return phi_pol(d_coeffs, xij) * (2.0 / hj)


def coeff_norm_leg(k):
    """
    Compute the normalisation coefficient of a Legendre polynomial with same order k

    Parameters
    ----------
    k : scalar, type -> integer
       degree of Legendre polynomials
    .

    Returns
    -------
    type -> float
      normalisation coefficient
    """
    return 2.0 / (2.0 * k + 1.0)
