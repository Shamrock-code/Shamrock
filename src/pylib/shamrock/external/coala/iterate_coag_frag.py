import sys
import time

import numpy as np
from scipy.special import legendre, roots_legendre

from .compute_coag_frag import *
from .generate_flux_intflux import *

# import numba_progress
# from progressbar import  Bar,Percentage,ProgressBar
from .generate_tabflux_tabintflux import *
from .L2_proj import *
from .limiter import *
from .utils_polynomials import *


def iterate_coag_frag_kdv(
    kernel,
    K0,
    nbins,
    kpol,
    dthydro,
    ndthydro,
    coeff_CFL,
    Q_intGQ,
    Q_L2_proj,
    eps_gij,
    eps_rhodust,
    massgrid,
    massbins,
    coeff_pl,
    bin_scut,
    dv,
    proba_coag,
    proba_frag,
    frac_mass_fragments,
    beta,
    dtg,
    rho_gas,
):
    """
    Function to iterate coagulation + fragmentation solver to reach the time ndthydro x dthydro

    Function for ballistic kernel with differential velocities dv

    DG scheme k=0, piecewise constant approximation

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    nbins : scalar, type -> integer
       number of dust bins
    kpol : scalar, type -> integer
       degree of polynomials for approximation
    dthydro : scalar, type -> float
       hydro timestep, used as timestep to reach for coagulation process
    ndthydro : scalar, type -> integer
       number of hydro timestep
    coeff_CFL : scalar, type -> float
       timestep coefficient for stability of the SSPRK order 3 scheme
    Q_intGQ : scalar, type -> integer
       number of points for Gauss-Legendre quadrature for integrals to evaluate flux
    Q_L2_proj : scalar, type -> integer
       number of points for Gauss-Legendre quadrature for integral in initial condition
    eps_gij : scalar, type -> float
       minimum value for mass distribution approximation gij
    eps_rhodust : scalar, type -> float
       minimum value for dust mass distribution rhodust
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
       arithmetic mean value of massgrid for each mass bins
    coeff_pl : scalar, type -> float
       power law for initial distribution in size
    bin_scut : scalar, type -> integer
       index massbins from which initial distribution is cut
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains
    proba_coag : 2D array (dim = (nbins,nbins)), type -> float
       probability to have coagulation (depending on v_frag and dv)
    proba_frag : 2D array (dim = (nbins,nbins)), type -> float
       probability to have fragmentation (depending on v_frag and dv)
    frac_mass_fragments : scalar, type -> float
        mass fraction of total mass (u+v) represented by fragments, m_frag = frac_mass_fragments*(u+v)
    beta : scalar, type -> float
        coefficient for power law fragment distribution in mass range [xmin,mfrag]
    dtg : scalar, type -> float
       mass dust to gas ratio
    rho_gas : scalar, type -> float
       mass gas density


    Returns
    -------
    gij_init : 1D array (dim = nbins) or 2D array (dim = (nbins.kpol+1)), type -> float
       initial components of g on the polynomial basis
    gij : 1D array (dim = nbins) or 2D array (dim = (nbins.kpol+1)), type -> float
       evolved components of g on the polynomial basis
    time_coag_frag : scalar, type -> float
       final time ndthydro x dthydro

    """

    # vecnodes_intGQ, vecweights_intGQ = np.polynomial.legendre.leggauss(Q_intGQ)
    vecnodes_intGQ, vecweights_intGQ = roots_legendre(Q_intGQ)

    # print("vecnodes_intGQ = ",vecnodes_intGQ)
    # print("vecweights_intGQ = ",vecweights_intGQ)

    vecnodes_intGQ = vecnodes_intGQ.tolist()
    vecweights_intGQ = vecweights_intGQ.tolist()

    # Legendre polynomial coefficients
    mat_coeffs_leg = np.zeros((kpol + 1, kpol + 1))
    mat_coeffs_leg = legendre_coeffs(kpol)

    # print("mat_coeffs_leg=",mat_coeffs_leg)

    start = time.time()
    if kernel == 3:
        if kpol == 0:
            tensor_tabflux_coag = np.zeros((nbins, nbins, nbins))

            # original version
            # print("Computing coagtabflux k=0 ...")
            # compute_coagtabflux_k0(kernel,K0,Q,vecnodes,vecweights,nbins,massgrid,mat_coeffs_leg,tensor_tabflux_coag)
            # print("coagtabflux k=0 done")

            # numba version
            # with numba_progress.ProgressBar(total=nbins, desc="Precomputing coagtabflux k=%d with numba"%(kpol)) as progress:
            compute_coagtabflux_k0_numba(
                kernel,
                K0,
                Q_intGQ,
                vecnodes_intGQ,
                vecweights_intGQ,
                nbins,
                massgrid,
                mat_coeffs_leg,
                tensor_tabflux_coag,
            )  # ,progress)

            tensor_tabflux_frag_T1 = np.zeros((nbins, nbins, nbins))
            tensor_tabflux_frag_T2 = np.zeros((nbins, nbins, nbins))

            compute_fragtabflux_k0_numba(
                kernel,
                K0,
                Q_intGQ,
                vecnodes_intGQ,
                vecweights_intGQ,
                nbins,
                massgrid,
                mat_coeffs_leg,
                frac_mass_fragments,
                beta,
                tensor_tabflux_frag_T1,
                tensor_tabflux_frag_T2,
            )

        else:
            print("Functions for kpol > 0 are missing for fragmentation")
            sys.exit()
            tensor_tabflux_coag = np.zeros((nbins, nbins, nbins, kpol + 1, kpol + 1))

            # original version
            # print("Computing coagtabflux k=%d ..."%(kpol))
            # compute_coagtabflux(kernel,K0,Q,vecnodes,vecweights,nbins,kpol,massgrid,mat_coeffs_leg,tensor_tabflux_coag)
            # print("coagtabflux k=%d done"%(kpol))

            # numba version
            # with numba_progress.ProgressBar(total=nbins, desc="Precomputing coagtabflux k=%d with numba"%(kpol)) as progress:
            compute_coagtabflux_numba(
                kernel,
                K0,
                Q_intGQ,
                vecnodes_intGQ,
                vecweights_intGQ,
                nbins,
                kpol,
                massgrid,
                mat_coeffs_leg,
                tensor_tabflux_coag,
            )  # ,progress)

            tensor_tabintflux_coag = np.zeros((nbins, kpol + 1, nbins, nbins, kpol + 1, kpol + 1))

            # original version
            # print("Computing coagtabintflux k=%d ..."%(kpol))
            # compute_coagtabintflux(kernel,K0,Q,vecnodes,vecweights,nbins,kpol,massgrid,mat_coeffs_leg,tensor_tabintflux_coag)
            # print("coagtabintflux k=%d done"%(kpol))

            # with numba_progress.ProgressBar(total=nbins, desc="Precomputing coagtabintflux k=%d with numba"%(kpol)) as progress:
            compute_coagtabintflux_numba(
                kernel,
                K0,
                Q_intGQ,
                vecnodes_intGQ,
                vecweights_intGQ,
                nbins,
                kpol,
                massgrid,
                mat_coeffs_leg,
                tensor_tabintflux_coag,
            )  # ,progress)

    else:
        print("Need to choose a kernel = 3 for ballistic kernel with dv array.")
        sys.exit()

    finish = time.time()
    if kpol == 0:
        print("Tensor tabflux generated in %.5f s" % (finish - start))
    else:
        print("Tensor tabflux tabintflux generated in %.5f s" % (finish - start))

    # print("min tabflux_coag = ",np.min(tensor_tabflux_coag[tensor_tabflux_coag != 0.0]))
    # print("max tabflux_coag = ",np.max(tensor_tabflux_coag[tensor_tabflux_coag != 0.0]))
    # print("min tabflux_frag_T1 = ",np.min(tensor_tabflux_frag_T1[tensor_tabflux_frag_T1 != 0.0]))
    # print("max tabflux_frag_T1 = ",np.max(tensor_tabflux_frag_T1[tensor_tabflux_frag_T1 != 0.0]))
    # print("min tabflux_frag_T2 = ",np.min(tensor_tabflux_frag_T2[tensor_tabflux_frag_T2 != 0.0]))
    # print("max tabflux_frag_T2 = ",np.max(tensor_tabflux_frag_T2[tensor_tabflux_frag_T2 != 0.0]))

    # sys.exit()
    # generate gij component on Legendre polynomials basis
    # initial condition
    vecnodes_L2_proj, vecweights_L2_proj = np.polynomial.legendre.leggauss(Q_L2_proj)

    start = time.time()

    if kpol == 0:
        rhodust = init_power_law_rho_dust(
            nbins, massgrid, massbins, bin_scut, coeff_pl, dtg, rho_gas, eps_rhodust
        )

        gij = eps_gij * np.ones(nbins)
        for j in range(nbins):
            if rhodust[j] > eps_rhodust:
                gij[j] = rhodust[j] / (massgrid[j + 1] - massgrid[j])

    else:
        raise ValueError("need to add fonction for initial condition kpol > 0")
        # gij = L2projDL(eps, nbins, kpol, massgrid, massbins, Q, vecnodes, vecweights)

        # # print("gij before limiter",gij)

        # # apply scale limiter
        # tabgamma = gammafunction(eps, nbins, kpol, massgrid, massbins, gij)
        # # print("tabgamma = ",tabgamma)
        # gij[:, 1:] = np.transpose(tabgamma * np.transpose(gij[:, 1:]))

        # # sys.exit()

        # # limit to eps value
        # if gij[:, 0][gij[:, 0] < 0.0].any():
        #     print("gij", gij)
        #     sys.exit()
        # else:
        #     gij[:, 0][gij[:, 0] < eps] = eps
        #     gij[:, 1:][gij[:, 0] < eps] = 0.0

    finish = time.time()
    gij_init = np.copy(gij)
    print("rhodust =", rhodust)
    print("gij =", gij)
    print("gij generated in %.5f s" % (finish - start))

    # test flux kdv
    # flux_test = compute_flux_coag_kdv(gij,tensor_tabflux_coag,dv)
    # for i in range(nbins):
    #    print("i=",i,",flux =",flux_test[i])

    # intflux_test = compute_intflux_coag_kdv(gij,tensor_tabintflux_coag,dv)
    # for i in range(nbins):
    #    print("i=",i,",intflux =",intflux_test[i,:])

    # sys.exit()

    # total mass density
    if kpol == 0:
        M1_t0 = np.sum((massgrid[1:] - massgrid[0:nbins]) * gij)
    else:
        M1_t0 = np.sum((massgrid[1:] - massgrid[0:nbins]) * gij[:, 0])
    print("M1 t0 = ", M1_t0)

    tot_nsub = 0
    tot_ndt = 0

    print("Time solver in progress")
    # time solver DG
    start = time.time()
    time_coag_frag = 0.0

    if kernel == 3:
        # bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=ndthydro).start()
        if kpol == 0:
            for i in range(ndthydro):
                gij, nsub, ndt = compute_coag_frag_k0_kdv(
                    eps_gij,
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
                )

                tot_nsub += nsub
                tot_ndt += ndt
                time_coag_frag += dthydro
                # bar.update(i+1)
        else:
            print("Functions for kpol > 0 are missing for fragmentation")
            sys.exit()
            # for i in range(ndthydro):
            #     gij, nsub, ndt = compute_coag_kdv(
            #         eps,
            #         coeff_CFL,
            #         nbins,
            #         kpol,
            #         massgrid,
            #         massbins,
            #         gij,
            #         tensor_tabflux_coag,
            #         tensor_tabintflux_coag,
            #         dv,
            #         dthydro,
            #     )

            #     tot_nsub += nsub
            #     tot_ndt += ndt
            #     time_coag += dthydro
            #     # bar.update(i+1)

        # bar.finish()

    else:
        print("Need to choose a kernel = 3 for ballistic kernel with dv array.")
        sys.exit()

    finish = time.time()

    print()
    print("total nsub =", tot_nsub)
    print("total ndt =", tot_ndt)
    print("total number time-steps =", tot_ndt + tot_nsub)

    print()
    print("gij tend =", gij)

    if kpol == 0:
        M1_tend = np.sum((massgrid[1:] - massgrid[0:nbins]) * gij)
    else:
        M1_tend = np.sum((massgrid[1:] - massgrid[0:nbins]) * gij[:, 0])
    print("M1 t0 = ", M1_t0)
    print("M1 tend = ", M1_tend)
    print("diff M1 = ", M1_tend - M1_t0)

    print("Time solver in %.5f" % (finish - start))

    return gij_init, gij, time_coag_frag
