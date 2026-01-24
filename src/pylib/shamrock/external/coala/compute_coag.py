import sys

from .solver_DG import *


def compute_coag_k0(eps, coeff_CFL, nbins, massgrid, gij, tensor_tabflux_coag, dthydro):
    """
    Function to compute coagulation solver for 1 hydro time-step

    DG scheme k=0, piecewise constant approximation

    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    coeff_CFL : scalar, type -> float
       timestep coefficient for stability of the SSPRK order 3 scheme
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    gij : 1D array (dim = nbins), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    dthydro : scalar, type -> float
       hydro timestep, used as timestep to reach for coagulation process


    Returns
    -------
    gij : 1D array (dim = nbins), type -> float
       evolved components of g on the polynomial basis after 1 hydro timestep
    nsub : scalar, type -> integer
       number of subcycling coagulation timestep to reach dthydro
    ndt : scalar, type -> integer
       number of hydro timestep, when coagulation CFL > dthydro

    """

    nsub = 0
    ndt = 0

    # evaluate coagulation CFL
    dtCFLsub = compute_CFL_coag_k0(eps, nbins, massgrid, gij, tensor_tabflux_coag)
    dtCFLsub = coeff_CFL * dtCFLsub
    # print("dtCFLsub=",dtCFLsub)

    # compare hydro timestep and coagulation CFL
    dt = min(dtCFLsub, dthydro)

    # coagulation subcycling timesteps
    if dt < dthydro:
        dtsub = 0.0
        while dtsub < dthydro and dthydro - dtsub > dtCFLsub:
            dtsub += dtCFLsub

            nsub += 1

            gij = solver_coag_k0(eps, nbins, massgrid, gij, tensor_tabflux_coag, dtCFLsub)

            dtCFLsub = compute_CFL_coag_k0(eps, nbins, massgrid, gij, tensor_tabflux_coag)
            dtCFLsub = coeff_CFL * dtCFLsub

            # print("gij=",gij)
            # sys.exit(-1)

        # last timestep to reach dthydro
        dtlast = dthydro - dtsub
        nsub += 1

        gij = solver_coag_k0(eps, nbins, massgrid, gij, tensor_tabflux_coag, dtlast)

    # when coagulation CFL > hydro timstep
    else:
        ndt += 1

        gij = solver_coag_k0(eps, nbins, massgrid, gij, tensor_tabflux_coag, dt)

    return gij, nsub, ndt


def compute_coag_k0_kdv(eps, coeff_CFL, nbins, massgrid, gij, tensor_tabflux_coag, dv, dthydro):
    """
    Function to compute coagulation solver for 1 hydro time-step

    DG scheme k=0, piecewise constant approximation

    Function for ballistic kernel with differential velocities dv

    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    coeff_CFL : scalar, type -> float
       timestep coefficient for stability of the SSPRK order 3 scheme
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
    dthydro : scalar, type -> float
       hydro timestep, used as timestep to reach for coagulation process


    Returns
    -------
    gij : 1D array (dim = nbins), type -> float
       evolved components of g on the polynomial basis after 1 hydro timestep
    nsub : scalar, type -> integer
       number of subcycling coagulation timestep to reach dthydro
    ndt : scalar, type -> integer
       number of hydro timestep, when coagulation CFL > dthydro

    """

    nsub = 0
    ndt = 0

    # evaluate coagulation CFL
    dtCFLsub = compute_CFL_coag_k0_kdv(eps, nbins, massgrid, gij, tensor_tabflux_coag, dv)
    dtCFLsub = coeff_CFL * dtCFLsub
    # print("dtCFLsub=",dtCFLsub)

    # compare hydro timestep and coagulation CFL
    dt = min(dtCFLsub, dthydro)

    # coagulation subcycling timesteps
    if dt < dthydro:
        dtsub = 0.0
        while dtsub < dthydro and dthydro - dtsub > dtCFLsub:
            dtsub += dtCFLsub

            nsub += 1

            gij = solver_coag_k0_kdv(eps, nbins, massgrid, gij, tensor_tabflux_coag, dv, dtCFLsub)

            dtCFLsub = compute_CFL_coag_k0_kdv(eps, nbins, massgrid, gij, tensor_tabflux_coag, dv)
            dtCFLsub = coeff_CFL * dtCFLsub

            # print("dtCFLsub=",dtCFLsub)

        # last timestep to reach dthydro
        dtlast = dthydro - dtsub
        nsub += 1

        gij = solver_coag_k0_kdv(eps, nbins, massgrid, gij, tensor_tabflux_coag, dv, dtlast)

    # when coagulation CFL > hydro timstep
    else:
        ndt += 1

        gij = solver_coag_k0_kdv(eps, nbins, massgrid, gij, tensor_tabflux_coag, dv, dt)

    return gij, nsub, ndt


def compute_coag(
    eps,
    coeff_CFL,
    nbins,
    kpol,
    massgrid,
    massbins,
    gij,
    tensor_tabflux_coag,
    tensor_tabintflux_coag,
    dthydro,
):
    """
    Function to compute coagulation solver for 1 hydro time-step

    DG scheme k>0, piecewise polynomial (order k) approximation

    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    coeff_CFL : scalar, type -> float
       timestep coefficient for stability of the SSPRK order 3 scheme
    nbins : scalar, type -> integer
       number of dust bins
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
       arithmetic mean of massgrid
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate coagulation flux
    tensor_tabintflux_coag : 6D array (dim = (nbins,kpol+1,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate term including the integral of the coagulation flux
    dthydro : scalar, type -> float
       hydro timestep, used as timestep to reach for coagulation process


    Returns
    -------
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       evolve components of g on the polynomial basis after 1 hydro timestep
    nsub : scalar, type -> integer
       number of subcycling coagulation timestep to reach dthydro
    ndt : scalar, type -> integer
       number of hydro timestep, when coagulation CFL > dthydro

    """

    nsub = 0
    ndt = 0

    # evaluate coagulation CFL
    dtCFLsub = compute_CFL_coag(eps, nbins, kpol, massgrid, gij, tensor_tabflux_coag)
    dtCFLsub = coeff_CFL * dtCFLsub
    # print("dtCFLsub = ",dtCFLsub)

    # compare hydro timestep and coagulation CFL
    dt = min(dtCFLsub, dthydro)

    # coagulation subcycling timesteps
    if dt < dthydro:
        dtsub = 0.0
        while dtsub < dthydro and dthydro - dtsub > dtCFLsub:
            dtsub += dtCFLsub

            nsub += 1

            gij = solver_coag(
                eps,
                nbins,
                kpol,
                massgrid,
                massbins,
                gij,
                tensor_tabflux_coag,
                tensor_tabintflux_coag,
                dtCFLsub,
            )

            dtCFLsub = compute_CFL_coag(eps, nbins, kpol, massgrid, gij, tensor_tabflux_coag)
            dtCFLsub = coeff_CFL * dtCFLsub
            # print("dtCFLsub = ",dtCFLsub)

        # last timestep to reach dthydro
        dtlast = dthydro - dtsub
        nsub += 1

        gij = solver_coag(
            eps,
            nbins,
            kpol,
            massgrid,
            massbins,
            gij,
            tensor_tabflux_coag,
            tensor_tabintflux_coag,
            dtlast,
        )

    # when coagulation CFL > hydro timstep
    else:
        ndt += 1

        gij = solver_coag(
            eps,
            nbins,
            kpol,
            massgrid,
            massbins,
            gij,
            tensor_tabflux_coag,
            tensor_tabintflux_coag,
            dt,
        )

    return gij, nsub, ndt


def compute_coag_kdv(
    eps,
    coeff_CFL,
    nbins,
    kpol,
    massgrid,
    massbins,
    gij,
    tensor_tabflux_coag,
    tensor_tabintflux_coag,
    dv,
    dthydro,
):
    """
    Function to compute coagulation solver for 1 hydro time-step

    DG scheme k>0, piecewise polynomial (order k) approximation

    Function for ballistic kernel with differential velocities dv

    Parameters
    ----------
    eps : scalar, type -> float
       minimum value for mass distribution approximation gij
    coeff_CFL : scalar, type -> float
       timestep coefficient for stability of the SSPRK order 3 scheme
    nbins : scalar, type -> integer
       number of dust bins
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
       arithmetic mean of massgrid
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       components of g on the polynomial basis
    tensor_tabflux_coag : 5D array (dim = (nbins,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate coagulation flux
    tensor_tabintflux_coag : 6D array (dim = (nbins,kpol+1,nbins,nbins,kpol+1,kpol+1)), type -> float
       array to evaluate term including the integral of the coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains
    dthydro : scalar, type -> float
       hydro timestep, used as timestep to reach for coagulation process


    Returns
    -------
    gij : 2D array (dim = (nbins,kpol+1)), type -> float
       evolve components of g on the polynomial basis after 1 hydro timestep
    nsub : scalar, type -> integer
       number of subcycling coagulation timestep to reach dthydro
    ndt : scalar, type -> integer
       number of hydro timestep, when coagulation CFL > dthydro

    """

    nsub = 0
    ndt = 0

    # evaluate coagulation CFL
    dtCFLsub = compute_CFL_coag_kdv(eps, nbins, kpol, massgrid, gij, tensor_tabflux_coag, dv)
    dtCFLsub = coeff_CFL * dtCFLsub
    # print("dtCFLsub = ",dtCFLsub)

    # compare hydro timestep and coagulation CFL
    dt = min(dtCFLsub, dthydro)

    # coagulation subcycling timesteps
    if dt < dthydro:
        dtsub = 0.0
        while dtsub < dthydro and dthydro - dtsub > dtCFLsub:
            dtsub += dtCFLsub

            nsub += 1

            gij = solver_coag_kdv(
                eps,
                nbins,
                kpol,
                massgrid,
                massbins,
                gij,
                tensor_tabflux_coag,
                tensor_tabintflux_coag,
                dv,
                dtCFLsub,
            )

            dtCFLsub = compute_CFL_coag_kdv(
                eps, nbins, kpol, massgrid, gij, tensor_tabflux_coag, dv
            )
            dtCFLsub = coeff_CFL * dtCFLsub
            # print("dtCFLsub = ",dtCFLsub)

        # last timestep to reach dthydro
        dtlast = dthydro - dtsub
        nsub += 1

        gij = solver_coag_kdv(
            eps,
            nbins,
            kpol,
            massgrid,
            massbins,
            gij,
            tensor_tabflux_coag,
            tensor_tabintflux_coag,
            dv,
            dtlast,
        )

    # when coagulation CFL > hydro timstep
    else:
        ndt += 1

        gij = solver_coag_kdv(
            eps,
            nbins,
            kpol,
            massgrid,
            massbins,
            gij,
            tensor_tabflux_coag,
            tensor_tabintflux_coag,
            dv,
            dt,
        )

    return gij, nsub, ndt
