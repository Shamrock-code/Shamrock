import sys

import numpy as np

# from progressbar import  Bar,Percentage,ProgressBar
from numba import jit, njit, prange

from .coag_term.coag_functions_GQ import *
from .frag_term.frag_functions_GQ import *
from .utils_polynomials import *

# -------------------------------
# functions for coagulation
# -------------------------------


def compute_coagtabflux_k0(
    kernel, K0, Q, vecnodes, vecweights, nbins, massgrid, mat_coeffs_leg, tensor_tabflux_coag
):
    """
    Function to precompute array depending only on massgrid to evaluate the coagulation flux
    DG scheme with piecewise constant approximation

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
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
        array to evaluate coagulation flux

    Returns
    -------
    filled array tensor_tabflux_coag

    """

    # display progress bar
    # bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=nbins).start()

    for j in range(nbins):
        for lp in range(j + 1):
            for l in range(nbins):
                res = coagfluxfunction(
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
                    0,
                    0,
                )

                if res != 0.0:
                    tensor_tabflux_coag[j, lp, l] = res

        # bar.update(j+1)

    # bar.finish()


@njit
def compute_coagtabflux_k0_numba(
    kernel, K0, Q, vecnodes, vecweights, nbins, massgrid, mat_coeffs_leg, tensor_tabflux_coag
):  # ,progress):
    """
    Function to precompute array depending only on massgrid to evaluate the coagulation flux
    DG scheme with piecewise constant approximation
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
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
        array to evaluate coagulation flux
    progress : used for numba progress bar

    Returns
    -------
    filled array tensor_tabflux_coag

    """

    for j in range(nbins):
        # progress.update(1) # update on each outer loop iteration
        for lp in range(j + 1):
            for l in range(nbins):
                res = coagfluxfunction_numba(
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
                    0,
                    0,
                )

                if res != 0.0:
                    tensor_tabflux_coag[j, lp, l] = res


def compute_coagtabflux(
    kernel, K0, Q, vecnodes, vecweights, nbins, kpol, massgrid, mat_coeffs_leg, tensor_tabflux_coag
):
    """
    Function to precompute array depending only on massgrid to evaluate the coagulation flux
    DG scheme with piecewise polynomial approximation

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
    kpol : scalar, type -> integer
        degree of polynomials for approximation
    massgrid : 1D array (dim = nbins+1), type -> float
        grid of masses given borders value of mass bins
    mat_coeffs_leg : 2D array (dmin = (kpol+1,kpol+1)), type -> float
        array containing on each line Legendre polynomial coefficients from degree 0 to kpol inclusive
        on each line coefficients are ordered from low to high orders
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
        array to evaluate coagulation flux


    Returns
    -------
    filled array tensor_tabflux_coag

    """

    # display progress bar
    # bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=nbins).start()

    for j in range(nbins):
        for lp in range(j + 1):
            for l in range(nbins):
                for ip in range(kpol + 1):
                    for i in range(kpol + 1):
                        res = coagfluxfunction(
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
                        )

                        if res != 0.0:
                            tensor_tabflux_coag[j, lp, l, ip, i] = res

        # bar.update(j+1)

    # bar.finish()


# @njit
def compute_coagtabflux_numba(
    kernel, K0, Q, vecnodes, vecweights, nbins, kpol, massgrid, mat_coeffs_leg, tensor_tabflux_coag
):  # ,progress):
    """
    Function to precompute array depending only on massgrid to evaluate the coagulation flux
    DG scheme with piecewise polynomial approximation
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
    kpol : scalar, type -> integer
        degree of polynomials for approximation
    massgrid : 1D array (dim = nbins+1), type -> float
        grid of masses given borders value of mass bins
    mat_coeffs_leg : 2D array (dmin = (kpol+1,kpol+1)), type -> float
        array containing on each line Legendre polynomial coefficients from degree 0 to kpol inclusive
        on each line coefficients are ordered from low to high orders
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
        array to evaluate coagulation flux


    Returns
    -------
    filled array tensor_tabflux_coag

    """

    for j in range(nbins):
        # progress.update(1) # update on each outer loop iteration
        for lp in range(j + 1):
            for l in range(nbins):
                for ip in range(kpol + 1):
                    for i in range(kpol + 1):
                        res = coagfluxfunction_numba(
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
                        )

                        if res != 0.0:
                            tensor_tabflux_coag[j, lp, l, ip, i] = res


def compute_coagtabintflux(
    kernel,
    K0,
    Q,
    vecnodes,
    vecweights,
    nbins,
    kpol,
    massgrid,
    mat_coeffs_leg,
    tensor_tabintflux_coag,
):
    """
    Function to precompute array depending only on massgrid to evaluate the term including the integral of coagulation flux
    DG scheme with piecewise polynomial approximation

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
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    mat_coeffs_leg : 2D array (dmin = (kpol+1,kpol+1)), type -> float
       array containing on each line Legendre polynomial coefficients from degree 0 to kpol inclusive
       on each line coefficients are ordered from low to high orders
    tensor_tabintflux_coag : 6D array (dim = (nbins,kpol+1,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate the term including the integral of the coagulation flux


    Returns
    -------
    filled array tensor_tabintflux_coag

    """

    # bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=nbins*(kpol+1)).start()

    for j in range(nbins):
        for k in range(1, kpol + 1):
            for lp in range(j + 1):
                for l in range(nbins):
                    for ip in range(kpol + 1):
                        for i in range(kpol + 1):
                            res = coagintfluxfunction(
                                kernel,
                                K0,
                                Q,
                                vecnodes,
                                vecweights,
                                nbins,
                                massgrid,
                                mat_coeffs_leg,
                                j,
                                k,
                                lp,
                                l,
                                ip,
                                i,
                            )

                            if res != 0.0:
                                tensor_tabintflux_coag[j, k, lp, l, ip, i] = res

        # bar.update(j+1)

    # bar.finish()


# @njit
def compute_coagtabintflux_numba(
    kernel,
    K0,
    Q,
    vecnodes,
    vecweights,
    nbins,
    kpol,
    massgrid,
    mat_coeffs_leg,
    tensor_tabintflux_coag,
):  # ,progress):
    """
    Function to precompute array depending only on massgrid to evaluate the term including the integral of coagulation flux
    DG scheme with piecewise polynomial approximation
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
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    mat_coeffs_leg : 2D array (dmin = (kpol+1,kpol+1)), type -> float
       array containing on each line Legendre polynomial coefficients from degree 0 to kpol inclusive
       on each line coefficients are ordered from low to high orders
    tensor_tabintflux_coag : 6D array (dim = (nbins,kpol+1,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate the term including the integral of the coagulation flux
    progress : used for numba progress bar


    Returns
    -------
       filled array tensor_tabintflux_coag


    """

    for j in range(nbins):
        # progress.update(1)  # update on each outer loop iteration
        for k in range(1, kpol + 1):
            for lp in range(j + 1):
                for l in range(nbins):
                    for ip in range(kpol + 1):
                        for i in range(kpol + 1):
                            res = coagintfluxfunction_numba(
                                kernel,
                                K0,
                                Q,
                                vecnodes,
                                vecweights,
                                nbins,
                                massgrid,
                                mat_coeffs_leg,
                                j,
                                k,
                                lp,
                                l,
                                ip,
                                i,
                            )

                            if res != 0.0:
                                tensor_tabintflux_coag[j, k, lp, l, ip, i] = res


# -------------------------------
# functions for fragmentation
# -------------------------------


def compute_fragtabflux_k0(
    kernel,
    K0,
    Q,
    vecnodes,
    vecweights,
    nbins,
    massgrid,
    mat_coeffs_leg,
    frac_mass_fragments,
    beta,
    tensor_tabflux_frag_T1,
    tensor_tabflux_frag_T2,
):
    """
    Function to precompute array depending only on massgrid to evaluate the fragmentation flux
    DG scheme with piecewise constant approximation

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
    frac_mass_fragments : scalar, type -> float
        mass fraction of total mass (u+v) represented by fragments, m_frag = frac_mass_fragments*(u+v)
    beta : scalar, type -> float
        coefficient for power law fragment distribution in mass range [xmin,mfrag]
    tensor_tabflux_frag_T1 : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate the term 1 for fragmentation flux
    tensor_tabflux_frag_T2 : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate the term 2 for fragmentation flux

    Returns
    -------
    filled arrays tensor_tabflux_frag_T1 and tensor_tabflux_frag_T2

    """

    # display progress bar
    # bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=nbins).start()

    for j in range(nbins):
        for lp in range(nbins):
            for l in range(nbins):
                # term 1
                if lp <= j:
                    res_T1 = fragfluxfunction_T1(
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
                        0,
                        0,
                        frac_mass_fragments,
                        beta,
                    )

                    if res_T1 != 0.0:
                        tensor_tabflux_frag_T1[j, lp, l] = res_T1

                # term 2
                if lp > j:
                    res_T2 = fragfluxfunction_T2(
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
                        0,
                        0,
                        frac_mass_fragments,
                        beta,
                    )

                    if res_T2 != 0.0:
                        tensor_tabflux_frag_T2[j, lp, l] = res_T2

        # bar.update(j+1)

    # bar.finish()


@njit
def compute_fragtabflux_k0_numba(
    kernel,
    K0,
    Q,
    vecnodes,
    vecweights,
    nbins,
    massgrid,
    mat_coeffs_leg,
    frac_mass_fragments,
    beta,
    tensor_tabflux_frag_T1,
    tensor_tabflux_frag_T2,
):  # ,progress):
    """
    Function to precompute array depending only on massgrid to evaluate the coagulation flux
    DG scheme with piecewise constant approximation
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
    frac_mass_fragments : scalar, type -> float
        mass fraction of total mass (u+v) represented by fragments, m_frag = frac_mass_fragments*(u+v)
    beta : scalar, type -> float
        coefficient for power law fragment distribution in mass range [xmin,mfrag]
    tensor_tabflux_frag_T1 : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate the term 1 for fragmentation flux
    tensor_tabflux_frag_T2 : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate the term 2 for fragmentation flux

    Returns
    -------
    filled arrays tensor_tabflux_frag_T1 and tensor_tabflux_frag_T2

    """

    for j in range(nbins):
        # progress.update(1) # update on each outer loop iteration
        for lp in range(nbins):
            for l in range(nbins):
                # term 1
                if lp <= j:
                    res_T1 = fragfluxfunction_T1_numba(
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
                        0,
                        0,
                        frac_mass_fragments,
                        beta,
                    )

                    if res_T1 != 0.0:
                        tensor_tabflux_frag_T1[j, lp, l] = res_T1

                # term 2
                if lp > j:
                    res_T2 = fragfluxfunction_T2_numba(
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
                        0,
                        0,
                        frac_mass_fragments,
                        beta,
                    )

                    if res_T2 != 0.0:
                        tensor_tabflux_frag_T2[j, lp, l] = res_T2
