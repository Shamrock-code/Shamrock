import sys

from .solver_DG import *


def compute_coag_frag_k0_kdv(
    eps,
    coeff_CFL,
    nbins,
    massgrid,
    gij,
    tensor_tabflux_coag,
    tensor_tabflux_frag_T1,
    tensor_tabflux_frag_T2,
    dv,
    proba_coag,
    proba_frag,
    dthydro,
):
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
    proba_coag : 2D array (dim = (nbins,nbins)), type -> float
       probability to have coagulation (depending on v_frag and dv)
    proba_frag : 2D array (dim = (nbins,nbins)), type -> float
       probability to have fragmentation (depending on v_frag and dv)
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
    dtCFLsub = compute_CFL_coag_frag_k0_kdv(
        eps,
        nbins,
        massgrid,
        gij,
        tensor_tabflux_coag,
        tensor_tabflux_frag_T1,
        tensor_tabflux_frag_T2,
        dv,
        proba_coag,
        proba_frag,
    )
    dtCFLsub *= coeff_CFL

    print("dtCFLsub=", dtCFLsub)
    #  sys.exit()

    # compare hydro timestep and coagulation CFL
    dt = min(dtCFLsub, dthydro)

    # coagulation subcycling timesteps
    if dt < dthydro:
        dtsub = 0.0
        while dtsub < dthydro and dthydro - dtsub > dtCFLsub:
            dtsub += dtCFLsub

            nsub += 1

            gij = solver_coag_frag_k0_kdv(
                eps,
                nbins,
                massgrid,
                gij,
                tensor_tabflux_coag,
                tensor_tabflux_frag_T1,
                tensor_tabflux_frag_T2,
                dv,
                proba_coag,
                proba_frag,
                dtCFLsub,
            )

            dtCFLsub = compute_CFL_coag_frag_k0_kdv(
                eps,
                nbins,
                massgrid,
                gij,
                tensor_tabflux_coag,
                tensor_tabflux_frag_T1,
                tensor_tabflux_frag_T2,
                dv,
                proba_coag,
                proba_frag,
            )
            dtCFLsub = coeff_CFL * dtCFLsub

            print("dtCFLsub=", dtCFLsub)

        # last timestep to reach dthydro
        dtlast = dthydro - dtsub
        nsub += 1

        gij = solver_coag_frag_k0_kdv(
            eps,
            nbins,
            massgrid,
            gij,
            tensor_tabflux_coag,
            tensor_tabflux_frag_T1,
            tensor_tabflux_frag_T2,
            dv,
            proba_coag,
            proba_frag,
            dtlast,
        )

    # when coagulation CFL > hydro timstep
    else:
        ndt += 1

        gij = solver_coag_frag_k0_kdv(
            eps,
            nbins,
            massgrid,
            gij,
            tensor_tabflux_coag,
            tensor_tabflux_frag_T1,
            tensor_tabflux_frag_T2,
            dv,
            proba_coag,
            proba_frag,
            dt,
        )

    return gij, nsub, ndt
