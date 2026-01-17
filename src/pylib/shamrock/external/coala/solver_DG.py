import sys
import time

import numpy as np
from scipy.special import eval_legendre

from .generate_flux_intflux import *
from .limiter import *
from .utils_polynomials import *


# for analytical collision kernel
def compute_CFL_coag_k0(eps, nbins, massgrid, gij, tensor_tabflux_coag):
    """
    Function to compute coagulation CFL for DG scheme k=0 piecewise constant approximation

    CFL formulation from Filbet & Laurencot 2004, dt <= mean_g * dm/dF

    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux


    Returns
    -------
    CFL_k0 : scalar, type -> float
       CFL restriction for coagulation

    """

    tabdflux = np.zeros(nbins)
    tabdtCFL = np.zeros(nbins)

    flux = compute_flux_coag_k0(gij, tensor_tabflux_coag)

    if np.array_equal(flux, np.zeros(nbins)):
        # calculs continue with no coag
        CFL_k0 = 1e3

    else:
        tabdflux[0] = flux[0]
        tabdtCFL[0] = np.abs(gij[0] * (massgrid[1] - massgrid[0]) / (tabdflux[0]))

        for j in range(1, nbins):
            hj = massgrid[j + 1] - massgrid[j]
            tabdflux[j] = flux[j] - flux[j - 1]

            if gij[j] > eps:
                tabdtCFL[j] = np.abs(gij[j] * hj / tabdflux[j])
            else:
                tabdtCFL[j] = 1e3

        CFL_k0 = np.min(tabdtCFL)

    if CFL_k0 == 0.0:
        print("gij=", gij)
        print("hj = ", massgrid[1:] - massgrid[:nbins])
        print("tabdtCFL = ", tabdtCFL)
        print("CFL_k0 = ", CFL_k0)
        print("issue in CFL coagulation")
        sys.exit()

    return CFL_k0


def L_func_coag_k0(nbins, massgrid, gij, tensor_tabflux_coag):
    """
    Function to compute the DG operator L for piecewise constant approximation (see Lombart et al., 2021)
    It is used for the time solver

    Parameters
    ----------
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux


    Returns
    -------
    Lk0 : 1D array (dim = nbins), type -> float
       DG operator for piecewise constant approximation in each bin

    """

    flux = compute_flux_coag_k0(gij, tensor_tabflux_coag)

    Lk0 = np.zeros(nbins)
    Lk0[0] = -flux[0] / (massgrid[1] - massgrid[0])
    Lk0[1:] = -(flux[1:] - flux[0 : nbins - 1]) / (massgrid[2:] - massgrid[1:nbins])

    return Lk0


def solver_coag_k0(eps, nbins, massgrid, gij, tensor_tabflux_coag, dt):
    """
    Function to compute SSPRK order 3 time solver with piecewise constant approximation
    See Zhang & Shu 2010 and Lombart et al., 2021


    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    dt : scalar, type -> float
       timestep


    Returns
    -------
    gijnew : 1D array (dim = nbins), type -> float
       evolved components of g on the polynomial basis after 1 timestep

    """

    # SSPRK3 algo (Zhang & Shu 2010)
    # step 1
    Lk0 = L_func_coag_k0(nbins, massgrid, gij, tensor_tabflux_coag)

    gij1 = gij + dt * Lk0

    # limit gij1 to eps value
    if len(gij1[gij1 < 0.0]) > 0:
        print("gij", gij)
        print("dt*Lk0 = ", dt * Lk0)
        print("gij1 = ", gij1)
        sys.exit()
    else:
        gij1[gij1 < eps] = eps

    Lk0_1 = L_func_coag_k0(nbins, massgrid, gij1, tensor_tabflux_coag)

    gij2 = 3.0 * gij / 4.0 + (gij1 + dt * Lk0_1) / 4.0

    # limit gij2 to eps value
    if len(gij2[gij2 < 0.0]) > 0:
        print("gi1", gi1)
        print("dt*Lk0_1 = ", dt * Lk0_1)
        print("gij2 = ", gij2)
        sys.exit()
    else:
        gij2[gij2 < eps] = eps

    # step 3
    Lk0_2 = L_func_coag_k0(nbins, massgrid, gij2, tensor_tabflux_coag)

    gijnew = gij / 3.0 + 2.0 * (gij2 + dt * Lk0_2) / 3.0

    # limit gijnew to eps value
    if len(gijnew[gijnew < 0.0]) > 0:
        print("gij2", gij2)
        print("dt*Lk0_2 = ", dt * Lk0_2)
        print("gijnew = ", gijnew)
        sys.exit()
    else:
        gijnew[gijnew < eps] = eps

    return gijnew


# for ballistic collision kernel with non analytical dv
def compute_CFL_coag_k0_kdv(eps, nbins, massgrid, gij, tensor_tabflux_coag, dv):
    """
    Function to compute coagulation CFL for DG scheme k=0 piecewise constant approximation.

    Function for ballistic kernel with differential velocities dv

    CFL formulation from Filbet & Laurencot 2004, dt <= mean_g * dm/dF

    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains


    Returns
    -------
    CFL_k0 : scalar, type -> float
       CFL restriction for coagulation

    """

    tabdflux = np.zeros(nbins)
    tabdtCFL = np.zeros(nbins)

    flux = compute_flux_coag_k0_kdv(gij, tensor_tabflux_coag, dv)

    if np.array_equal(flux, np.zeros(nbins)):
        # calculs continue with no coag
        CFL_k0 = 1e3

    else:
        tabdflux[0] = flux[0]
        tabdtCFL[0] = np.abs(gij[0] * (massgrid[1] - massgrid[0]) / (tabdflux[0]))

        for j in range(1, nbins):
            hj = massgrid[j + 1] - massgrid[j]
            tabdflux[j] = flux[j] - flux[j - 1]

            if gij[j] > eps:
                tabdtCFL[j] = np.abs(gij[j] * hj / tabdflux[j])
            else:
                tabdtCFL[j] = 1e3

        # print("gij=",gij)
        # print("tabdtCFL =",tabdtCFL)

        CFL_k0 = np.min(tabdtCFL)

    if CFL_k0 == 0.0:
        print("gij=", gij)
        print("hj = ", massgrid[1:] - massgrid[:nbins])
        print("tabdtCFL = ", tabdtCFL)
        print("CFL_k0 = ", CFL_k0)
        print("issue in CFL coagulation")
        sys.exit()

    return CFL_k0


# for ballistic collision kernel with non analytical dv
def L_func_coag_k0_kdv(nbins, massgrid, gij, tensor_tabflux_coag, dv):
    """
    Function to compute the DG operator L for piecewise constant approximation (see Lombart et al., 2021)
    Function for ballistic kernel with differential velocities dv
    It is used for the time solver

    Parameters
    ----------
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains


    Returns
    -------
    Lk0 : 1D array (dim = nbins), type -> float
       DG operator for piecewise constant approximation in each bin

    """

    flux = compute_flux_coag_k0_kdv(gij, tensor_tabflux_coag, dv)

    Lk0 = np.zeros(nbins)
    Lk0[0] = -flux[0] / (massgrid[1] - massgrid[0])
    Lk0[1:] = -(flux[1:] - flux[0 : nbins - 1]) / (massgrid[2:] - massgrid[1:nbins])

    return Lk0


# for ballistic collision kernel with non analytical dv
def solver_coag_k0_kdv(eps, nbins, massgrid, gij, tensor_tabflux_coag, dv, dt):
    """
    Function to compute SSPRK order 3 time solver with piecewise constant approximation
    Function for ballistic kernel with differential velocities dv
    See Zhang & Shu 2010 and Lombart et al., 2021


    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains
    dt : scalar, type -> float
       timestep


    Returns
    -------
    gijnew : 1D array (dim = nbins), type -> float
       evolved components of g on the polynomial basis after 1 timestep

    """

    # SSPRK3 algo (Zhang & Shu 2010)
    # step 1
    Lk0 = L_func_coag_k0_kdv(nbins, massgrid, gij, tensor_tabflux_coag, dv)

    gij1 = gij + dt * Lk0

    # limit gij1 to eps value
    if len(gij1[gij1 < 0.0]) > 0:
        print("gij", gij)
        print("dt*Lk0 = ", dt * Lk0)
        print("gij1 = ", gij1)
        sys.exit()
    else:
        gij1[gij1 < eps] = eps

    Lk0_1 = L_func_coag_k0_kdv(nbins, massgrid, gij1, tensor_tabflux_coag, dv)

    gij2 = 3.0 * gij / 4.0 + (gij1 + dt * Lk0_1) / 4.0

    # limit gij2 to eps value
    if len(gij2[gij2 < 0.0]) > 0:
        print("gi1", gi1)
        print("dt*Lk0_1 = ", dt * Lk0_1)
        print("gij2 = ", gij2)
        sys.exit()
    else:
        gij2[gij2 < eps] = eps

    # step 3
    Lk0_2 = L_func_coag_k0_kdv(nbins, massgrid, gij2, tensor_tabflux_coag, dv)

    gijnew = gij / 3.0 + 2.0 * (gij2 + dt * Lk0_2) / 3.0

    # limit gijnew to eps value
    if len(gijnew[gijnew < 0.0]) > 0:
        print("gij2", gij2)
        print("dt*Lk0_2 = ", dt * Lk0_2)
        print("gijnew = ", gijnew)
        sys.exit()
    else:
        gijnew[gijnew < eps] = eps

    return gijnew


def compute_CFL_coag(eps, nbins, kpol, massgrid, gij, tensor_tabflux_coag):
    """
    Function to compute coagulation CFL for DG scheme k>0 piecewise polynomial approximation

    CFL formulation from Filbet & Laurencot 2004, dt <= mean_g * dm/dF

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
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate coagulation flux


    Returns
    -------
    CFL : scalar, type -> float
       CFL restriction for coagulation

    """

    tabdflux = np.zeros(nbins)
    tabdtCFL = np.zeros(nbins)

    flux = compute_flux_coag(gij, tensor_tabflux_coag)

    if np.array_equal(flux, np.zeros(nbins)):
        # calculs continue with no coag
        CFL_k0 = 1e3

    else:
        tabdflux[0] = flux[0]
        tabdtCFL[0] = np.abs(gij[0, 0] * (massgrid[1] - massgrid[0]) / (tabdflux[0]))

        for j in range(1, nbins):
            hj = massgrid[j + 1] - massgrid[j]
            tabdflux[j] = flux[j] - flux[j - 1]

            if gij[j, 0] > eps:
                tabdtCFL[j] = np.abs(gij[j, 0] * hj / tabdflux[j])
            else:
                tabdtCFL[j] = 1e3

        CFL = np.min(tabdtCFL)

    if CFL <= 0.0:
        print("gij[:,0]=", gij[:, 0])
        print("hj = ", massgrid[1:] - massgrid[: nbins + 1])
        print("tabdtCFL = ", tabdtCFL)
        print("CFL = ", CFL)
        print("issue in CFL")
        sys.exit()

    return CFL


def L_func_coag(nbins, kpol, massgrid, gij, tensor_tabflux_coag, tensor_tabintflux_coag):
    """
    Function to compute the DG operator L for piecewise polynomial approximation (see Lombart et al., 2021)
    It is used for the time solver

    Parameters
    ----------
    nbins : scalar, type -> integer
       number of dust bins
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate coagulation flux
    tensor_tabintflux_coag : 6D array (dim = (nbins,kpol+1,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate the term including the integral of coagulation flux


    Returns
    -------
    Lk : 2D array (dim = (nbins,kpol+1)), type -> float
       DG operator for piecewise polynomials approximation in each bin

    """

    flux = compute_flux_coag(gij, tensor_tabflux_coag)
    intflux = compute_intflux_coag(gij, tensor_tabintflux_coag)

    mat_intflux = intflux.reshape(nbins, kpol + 1)

    LegPright = eval_legendre(range(kpol + 1), 1.0)
    flux_right = np.outer(flux, LegPright)

    flux_left = np.zeros((nbins, kpol + 1))
    LegPleft = eval_legendre(range(kpol + 1), -1.0)
    flux_left[1:, :] = np.outer(flux[0 : nbins - 1], LegPleft)

    mat_coeff_norm = np.outer(
        2.0 / (massgrid[1:] - massgrid[0:nbins]), 1.0 / coeff_norm_leg(np.arange(kpol + 1))
    )

    Lk = mat_coeff_norm * (mat_intflux - (flux_right - flux_left))

    return Lk


def solver_coag(
    eps, nbins, kpol, massgrid, massbins, gij, tensor_tabflux_coag, tensor_tabintflux_coag, dt
):
    """
    Function to compute SSPRK order 3 time solver with piecewise polynomial approximation
    See Zhang & Shu 2010 and Lombart et al., 2021


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
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate coagulation flux
    tensor_tabintflux_coag : 6D array (dim = (nbins,kpol+1,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate the term including the integral of coagulation flux
    dt : scalar, type -> float
       timestep


    Returns
    -------
    gijnew : 2D array (dim = (nbins,kpol+1)), type -> float
       evolved components of g on the polynomial basis after 1 timestep

    """

    # SSPRK3 algo (Zhang & Shu 2010)
    # step 1
    Lk = L_func_coag(nbins, kpol, massgrid, gij, tensor_tabflux_coag, tensor_tabintflux_coag)

    gij1 = gij + dt * Lk

    # apply limiter coefficient
    tabgamma = gammafunction(eps, nbins, kpol, massgrid, massbins, gij1)
    gij1[:, 1:] = tabgamma.reshape(nbins, 1) * gij1[:, 1:]

    # limit to eps value
    if gij1[:, 0][gij1[:, 0] < 0.0].any():
        print("gij", gij)
        print("dt*Lk = ", dt * Lk)
        print("gij1 = ", gij1)
        sys.exit()
    else:
        gij1[:, 1:][gij1[:, 0] < eps] = 0.0
        gij1[:, 0][gij1[:, 0] < eps] = eps

    Lk_1 = L_func_coag(nbins, kpol, massgrid, gij1, tensor_tabflux_coag, tensor_tabintflux_coag)

    gij2 = 3.0 * gij / 4.0 + (gij1 + dt * Lk_1) / 4.0

    # apply limiter coefficient
    tabgamma = gammafunction(eps, nbins, kpol, massgrid, massbins, gij2)
    gij2[:, 1:] = tabgamma.reshape(nbins, 1) * gij2[:, 1:]

    # limit to eps value

    if gij2[:, 0][gij2[:, 0] < 0.0].any():
        print("gij", gij)
        print("dt*Lk_1 = ", dt * Lk_1)
        print("gij2 = ", gij2)
        sys.exit()
    else:
        gij2[:, 1:][gij2[:, 0] < eps] = 0.0
        gij2[:, 0][gij2[:, 0] < eps] = eps

    # step 3
    Lk_2 = L_func_coag(nbins, kpol, massgrid, gij2, tensor_tabflux_coag, tensor_tabintflux_coag)

    gijnew = gij / 3.0 + 2.0 * (gij2 + dt * Lk_2) / 3.0

    # apply limiter coefficient
    tabgamma = gammafunction(eps, nbins, kpol, massgrid, massbins, gijnew)
    gijnew[:, 1:] = tabgamma.reshape(nbins, 1) * gijnew[:, 1:]

    # limit to eps value

    if gijnew[:, 0][gijnew[:, 0] < 0.0].any():
        print("gij", gij)
        print("dt*Lk_1 = ", dt * Lk_1)
        print("gijnew = ", gijnew)
        sys.exit()
    else:
        gijnew[:, 1:][gijnew[:, 0] < eps] = 0.0
        gijnew[:, 0][gijnew[:, 0] < eps] = eps

    return gijnew


# for ballistic collision kernel with non analytical dv
def compute_CFL_coag_kdv(eps, nbins, kpol, massgrid, gij, tensor_tabflux_coag, dv):
    """
    Function to compute coagulation CFL for DG scheme k>0 piecewise polynomial approximation
    Function for ballistic kernel with differential velocities dv
    CFL formulation from Filbet & Laurencot 2004, dt <= mean_g * dm/dF

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
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains


    Returns
    -------
    CFL : scalar, type -> float
       CFL restriction for coagulation

    """
    tabdflux = np.zeros(nbins)
    tabdtCFL = np.zeros(nbins)

    flux = compute_flux_coag_kdv(gij, tensor_tabflux_coag, dv)

    if np.array_equal(flux, np.zeros(nbins)):
        # calculs continue with no coag
        CFL = 1e3

    else:
        tabdflux[0] = flux[0]
        tabdtCFL[0] = np.abs(gij[0, 0] * (massgrid[1] - massgrid[0]) / (tabdflux[0]))

        for j in range(1, nbins):
            hj = massgrid[j + 1] - massgrid[j]
            tabdflux[j] = flux[j] - flux[j - 1]

            if gij[j, 0] > eps:
                tabdtCFL[j] = np.abs(gij[j, 0] * hj / tabdflux[j])
            else:
                tabdtCFL[j] = 1e3

        CFL = np.min(tabdtCFL)

    if CFL <= 0.0:
        print("gij[:,0]=", gij[:, 0])
        print("hj = ", massgrid[1:] - massgrid[: nbins + 1])
        print("tabdtCFL = ", tabdtCFL)
        print("CFL = ", CFL)
        print("issue in CFL")
        sys.exit()

    return CFL


# for ballistic collision kernel with non analytical dv
def L_func_coag_kdv(nbins, kpol, massgrid, gij, tensor_tabflux_coag, tensor_tabintflux_coag, dv):
    """
    Function to compute the DG operator L for piecewise polynomial approximation (see Lombart et al., 2021)
    Function for ballistic kernel with differential velocities dv
    It is used for the time solver

    Parameters
    ----------
    nbins : scalar, type -> integer
       number of dust bins
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate coagulation flux
    tensor_tabintflux_coag : 6D array (dim = (nbins,kpol+1,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate the term including the integral of coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains

    Returns
    -------
    Lk : 2D array (dim = (nbins,kpol+1)), type -> float
       DG operator for piecewise polynomials approximation in each bin

    """

    flux = compute_flux_coag_kdv(gij, tensor_tabflux_coag, dv)
    intflux = compute_intflux_coag_kdv(gij, tensor_tabintflux_coag, dv)

    mat_intflux = intflux.reshape(nbins, kpol + 1)

    LegPright = eval_legendre(range(kpol + 1), 1.0)
    flux_right = np.outer(flux, LegPright)

    flux_left = np.zeros((nbins, kpol + 1))
    LegPleft = eval_legendre(range(kpol + 1), -1.0)
    flux_left[1:, :] = np.outer(flux[0 : nbins - 1], LegPleft)

    mat_coeff_norm = np.outer(
        2.0 / (massgrid[1:] - massgrid[0:nbins]), 1.0 / coeff_norm_leg(np.arange(kpol + 1))
    )

    Lk = mat_coeff_norm * (mat_intflux - (flux_right - flux_left))

    return Lk


# for ballistic collision kernel with non analytical dv
def solver_coag_kdv(
    eps, nbins, kpol, massgrid, massbins, gij, tensor_tabflux_coag, tensor_tabintflux_coag, dv, dt
):
    """
    Function to compute SSPRK order 3 time solver with piecewise constant approximation
    Function for ballistic kernel with differential velocities dv
    See Zhang & Shu 2010 and Lombart et al., 2021


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
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate coagulation flux
    tensor_tabintflux_coag : 6D array (dim = (nbins,kpol+1,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate the term including the integral of coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains
    dt : scalar, type -> float
       timestep


    Returns
    -------
    gijnew : 2D array (dim = (nbins,kpol+1)), type -> float
       evolved components of g on the polynomial basis after 1 timestep

    """

    # SSPRK3 algo (Zhang & Shu 2010)
    # step 1
    Lk = L_func_coag_kdv(
        nbins, kpol, massgrid, gij, tensor_tabflux_coag, tensor_tabintflux_coag, dv
    )

    gij1 = gij + dt * Lk

    # apply limiter coefficient
    tabgamma = gammafunction(eps, nbins, kpol, massgrid, massbins, gij1)
    gij1[:, 1:] = tabgamma.reshape(nbins, 1) * gij1[:, 1:]

    # limit to eps value
    if gij1[:, 0][gij1[:, 0] < 0.0].any():
        print("gij", gij)
        print("dt*Lk = ", dt * Lk)
        print("gij1 = ", gij1)
        print("Negative value ")
        sys.exit()
    else:
        gij1[:, 1:][gij1[:, 0] < eps] = 0.0
        gij1[:, 0][gij1[:, 0] < eps] = eps

    Lk_1 = L_func_coag_kdv(
        nbins, kpol, massgrid, gij1, tensor_tabflux_coag, tensor_tabintflux_coag, dv
    )

    gij2 = 3.0 * gij / 4.0 + (gij1 + dt * Lk_1) / 4.0

    # apply limiter coefficient
    tabgamma = gammafunction(eps, nbins, kpol, massgrid, massbins, gij2)
    gij2[:, 1:] = tabgamma.reshape(nbins, 1) * gij2[:, 1:]

    # limit to eps value

    if gij2[:, 0][gij2[:, 0] < 0.0].any():
        print("gij", gij)
        print("dt*Lk_1 = ", dt * Lk_1)
        print("gij2 = ", gij2)
        print("Negative value ")
        sys.exit()
    else:
        gij2[:, 1:][gij2[:, 0] < eps] = 0.0
        gij2[:, 0][gij2[:, 0] < eps] = eps

    # step 3
    Lk_2 = L_func_coag_kdv(
        nbins, kpol, massgrid, gij2, tensor_tabflux_coag, tensor_tabintflux_coag, dv
    )

    gijnew = gij / 3.0 + 2.0 * (gij2 + dt * Lk_2) / 3.0

    # apply limiter coefficient
    tabgamma = gammafunction(eps, nbins, kpol, massgrid, massbins, gijnew)
    gijnew[:, 1:] = tabgamma.reshape(nbins, 1) * gijnew[:, 1:]

    # limit to eps value

    if gijnew[:, 0][gijnew[:, 0] < 0.0].any():
        print("gij", gij)
        print("dt*Lk_1 = ", dt * Lk_1)
        print("gijnew = ", gijnew)
        print("Negative value ")
        sys.exit()
    else:
        gijnew[:, 1:][gijnew[:, 0] < eps] = 0.0
        gijnew[:, 0][gijnew[:, 0] < eps] = eps

    return gijnew
