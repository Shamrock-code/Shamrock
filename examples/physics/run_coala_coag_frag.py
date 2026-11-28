"""
Using Coala within Shamrock to solve the Smoluchowski and fragmentation equations
==================================================================================

"""

# %%
# Imports
import os

import numpy as np
from matplotlib import pyplot as plt
from shamrock.external import coala

# %%
# Where is coala located?
print(f"coala path : {coala.__file__}")

# %%
# Parameters of the dust distribution & evolution
nbins = 20
Q_L2_proj = 5
Q_intGQ = 150
eps_gij = 1e-50
coeff_CFL = 0.3
t0 = 0.0

# physical parameters in cgs
rho_gas = 1e-14
Temp = 10.0
dtg = 1e-2
alpha_turb = 1.5
grav = 6.67e-8
t_ff = np.sqrt(3.0 * np.pi / (32.0 * grav * rho_gas))
yr = 31556926.0
kB = 1.38e-16
mu_gas = 23e-1
mh = 100749e-5 * 1660538921e-33

# dust parameters
rhograin = 2.3  # cgs
eps_rhodust = 1e-50
smax = 1e-1  # cgs
smin = 5e-7
scut = 250e-7
frac_mass_fragments = 1e-1
beta = -11.0 / 6.0
v_frag_0 = 1e2  # cgs

# power law for initial distribution in size
coeff_pl = -3.5  # mrn

# coefficient for kernel collision
K0 = np.pi * (4.0 / 3.0 * np.pi * rhograin) ** (-2.0 / 3.0)


def compute_dv_ormel(nbins, rho_gas, Temp, rhograin, grainsize, tdyn, alpha_turb):

    w_th = np.sqrt(8.0 * kB * Temp / (np.pi * mu_gas * mh))
    ts = rhograin * grainsize / (rho_gas * w_th)

    St = ts / tdyn
    cs = np.sqrt(kB * Temp / (mu_gas * mh))

    dv_ormel = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            dv_ormel[i, j] = np.sqrt(alpha_turb) * cs * max(np.sqrt(St[i]), np.sqrt(St[j]))

    return dv_ormel


def compute_v_frag(nbins, massmeanlog, v_frag_0, i, j):
    mu_ij = massmeanlog[i] / massmeanlog[j]
    v_frag = (1.0 + 1.0 / mu_ij) * np.sqrt(mu_ij) * 0.5 * v_frag_0

    return v_frag


# %%
# Function to run the tests for a given kernel
def run_kernel_case(kernel):

    cases = {
        "order k=0": {
            "kpol": 0,
        },
        # "order k=1": {
        #     "kpol": 1,
        # },
        # "order k=2": {
        #     "kpol": 2,
        # },
        # "order k=3": {
        #     "kpol": 3,
        # },
        # "order k=4": {
        #     "kpol": 4,
        # },
        # "order k=5": {
        #     "kpol": 5,
        # },
    }

    if kernel == 3:
        dthydro = 1e2 * yr
        n_tff = 1500
        ndthydro = np.floor(n_tff * t_ff / dthydro).astype(int)

        if ndthydro == 0:
            raise ValueError("need to reduce dthydro")

    else:
        raise ValueError("need to choose physical kernel kdv")

    print("Test coala for k_dv")

    sizegrid, sizebins, sizemeanlog = coala.init_sizegrid_log(nbins, smax, smin)
    massgrid = 4.0 * np.pi * rhograin * sizegrid**3 / 3.0
    massbins = 0.5 * (massgrid[1:] + massgrid[0:nbins])
    massmeanlog = np.sqrt(massgrid[1:] * massgrid[0:nbins])

    # compute bin_scut
    for j in range(nbins):
        if sizegrid[j] < scut <= sizegrid[j + 1]:
            bin_scut = j

    for case in cases:
        kpol = cases[case]["kpol"]
        print()
        print("Computing coala solver for k=%d" % (kpol))

        # Ormel dv model
        dv = compute_dv_ormel(nbins, rho_gas, Temp, rhograin, sizemeanlog, t_ff, alpha_turb)

        # proba frag from maxwell distrib for dv
        # constant v_frag
        # threshold_vfrag = v_frag_0

        # vfrag depending on energy, grain of same size break for dv > v_frag_0
        proba_frag = np.zeros((nbins, nbins))
        for i in range(nbins):
            for j in range(nbins):
                threshold_vfrag = compute_v_frag(nbins, massmeanlog, v_frag_0, i, j)

                if 1.5 * (threshold_vfrag / dv[i, j]) ** 2 <= 1e2:
                    proba_frag[i, j] = (1.5 * (threshold_vfrag / dv[i, j]) ** 2 + 1.0) * np.exp(
                        -1.5 * (threshold_vfrag / dv[i, j]) ** 2
                    )

        proba_coag = 1.0 - proba_frag

        dv_coag = proba_coag * dv
        dv_frag = proba_frag * dv

        # print("min dv = ",np.min(dv))
        # print("max dv = ",np.max(dv))
        # print("min dv_coag = ",np.min(dv_coag[dv_coag != 0.]))
        # print("max dv_coag = ",np.max(dv_coag))
        # print("min dv_frag = ",np.min(dv_frag[dv_frag != 0.0]))
        # print("max dv_frag = ",np.max(dv_frag))
        # sys.exit()

        gij_init, gij, time_coag_frag = coala.iterate_coag_frag_kdv(
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
        )

        cases[case]["massgrid"] = massgrid
        cases[case]["massbins"] = massbins
        cases[case]["gij_init"] = gij_init
        cases[case]["gij_end"] = gij
        cases[case]["time"] = [t0, time_coag_frag]

    # compute ref solutions when needed

    print()
    print("Computing coala solver for k_dv (k=0), ref solution")
    nbins_ref = 2 * nbins
    sizegrid_ref, sizebins_ref, sizemeanlog_ref = coala.init_sizegrid_log(nbins_ref, smax, smin)
    massgrid_ref = 4.0 * np.pi * rhograin * sizegrid_ref**3 / 3.0
    massbins_ref = 0.5 * (massgrid_ref[1:] + massgrid_ref[0:nbins_ref])
    massmeanlog_ref = np.sqrt(massgrid_ref[1:] * massgrid_ref[0:nbins_ref])

    # for j in range(nbins_ref):
    #     if (sizegrid_ref[j] < scut and scut <= sizegrid_ref[j+1]):
    #         bin_scut_ref = j

    for j in range(nbins_ref):
        if (
            massgrid_ref[j] < massgrid[bin_scut + 1]
            and massgrid[bin_scut + 1] <= massgrid_ref[j + 1]
        ):
            bin_scut_ref = j

    # Ormel dv model
    dv_ref = compute_dv_ormel(nbins_ref, rho_gas, Temp, rhograin, sizemeanlog_ref, t_ff, alpha_turb)

    # proba frag from maxwell distrib for dv
    # constant v_frag
    # threshold_vfrag_ref = v_frag_0

    # vfrag depending on energy, grain of same size break for dv > v_frag_0
    proba_frag_ref = np.zeros((nbins_ref, nbins_ref))
    for i in range(nbins_ref):
        for j in range(nbins_ref):
            threshold_vfrag_ref = compute_v_frag(nbins_ref, massmeanlog_ref, v_frag_0, i, j)
            proba_frag_ref[i, j] = (1.5 * (threshold_vfrag_ref / dv_ref[i, j]) ** 2 + 1.0) * np.exp(
                -1.5 * (threshold_vfrag_ref / dv_ref[i, j]) ** 2
            )

    proba_coag_ref = 1.0 - proba_frag_ref

    gij_init_ref, gij_ref, time_coag_frag_ref = coala.iterate_coag_frag_kdv(
        kernel,
        K0,
        nbins_ref,
        kpol,
        dthydro,
        ndthydro,
        coeff_CFL,
        Q_intGQ,
        Q_L2_proj,
        eps_gij,
        eps_rhodust,
        massgrid_ref,
        massbins_ref,
        coeff_pl,
        bin_scut_ref,
        dv_ref,
        proba_coag_ref,
        proba_frag_ref,
        frac_mass_fragments,
        beta,
        dtg,
        rho_gas,
    )

    # Plotting
    plt.rcParams["font.size"] = 16
    plt.rcParams["lines.linewidth"] = 3
    plt.rcParams["legend.columnspacing"] = 0.5

    savefig_options = dict(bbox_inches="tight")

    marker_style = dict(
        marker="o", markersize=8, markerfacecolor="white", linestyle="", markeredgewidth=2
    )

    str_kernel = "k_dv"

    massmin = cases["order k=0"]["massgrid"][0]
    massmax = cases["order k=0"]["massgrid"][-1]

    tend = cases["order k=0"]["time"][-1]
    plt.figure(1)

    plt.loglog(massbins_ref, gij_init_ref, "--", c="C0", alpha=0.5)
    plt.loglog(massbins_ref, gij_ref, "--", c="C0", label="ref %d bins" % (nbins_ref))

    plt.loglog(
        cases["order k=0"]["massbins"],
        cases["order k=0"]["gij_init"],
        markeredgecolor="black",
        label="gij init",
        **marker_style,
        alpha=0.5,
    )
    for case in cases:
        print("plotting case", case)

        # if cases[case]["gij_end"][0] is a scalar
        print("gij_end = ", type(cases[case]["gij_end"][0]))
        if isinstance(cases[case]["gij_end"][0], np.float64):
            print("gij_end is a scalar")
            plt.loglog(cases[case]["massbins"], cases[case]["gij_end"], label=case, **marker_style)
        else:
            print("gij_end is an array")
            plt.loglog(
                cases[case]["massbins"], cases[case]["gij_end"][:, 0], label=case, **marker_style
            )
        # print ("gij_end = ",type(cases[case]["gij_end"][0]))
        # plt.loglog(cases[case]["massbins"],cases[case]["gij_end"],markeredgecolor='black',label=case,**marker_style)

    plt.ylim(1.0e-15, 1.0e1)
    plt.xlim(massmin, massmax)
    plt.title(str_kernel + ", nbins=%d" % (nbins))
    plt.xlabel(r"mass ")
    plt.ylabel(r"mass density distribution g")
    plt.legend(loc="lower left", ncol=1)
    plt.tight_layout()


# %%
# Run the tests (kdv)
run_kernel_case(3)
plt.show()
