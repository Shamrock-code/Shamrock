import sys

sys.path.insert(0, "../src")

import coala

print(f"coala path : {coala.__file__}")


import os

import numpy as np

# setup parameters
# -------------------------------------------
# nbins     -> number of grain size bins
# massmax   -> maximal mass of grains to consider
# minmass   -> minimal mass of grains to consider
# kernel    -> chose among available kernels (see kernel_collision.py)
# K0        -> normalisation coefficient for simple kernels
# kpol      -> order of polynomials for approximation
# Q         -> number of Gauss points for Gauss-quadrature method
# eps       -> minimum value for mass density distribution
# coeff_CFL -> coefficient for SSPRK 3 time solver
# dthydro   -> hydro timestep
# ndthydro  -> number of dthydro

# -------------------------------------------

nbins = 20

massmax = 1e6
massmin = 1e-3

# available kernels
# 0 -> kconst
# 1 -> kadd
# 2 -> brownian motion kernel with analytical dv : K = sigma * dv
# 3 -> kdv with analytical cross section, need approximated dv (2D array)
kernel = 1
K0 = 1.0
kpol = 0
Q = 5
eps = 1e-20
coeff_CFL = 0.3


# define path for data
# pre defined values for dthydro and ndthydro
if kernel == 0:
    path_data = f"./data/kconst/{nbins}bins/Qcoag={Q}/"

    t0 = 0.0
    dthydro = 100
    ndthydro = 300

elif kernel == 1:
    path_data = "./data/kadd/%dbins/Qcoag=%d/" % (nbins, Q)

    t0 = 0.0
    dthydro = 1e-2
    ndthydro = 300

elif kernel == 2:
    # nbins=100 and kpol=1 as reference
    path_data = "./data/dv_brownian_ref/%dbins/" % (nbins)

    t0 = 0.0
    dthydro = 1e-1
    ndthydro = 500

elif kernel == 3:
    path_data = "./data/dv_brownian_approx/%dbins/Qcoag=%d/" % (nbins, Q)

    t0 = 0.0
    dthydro = 1e-1
    ndthydro = 500

else:
    print("need to choose a kernel")
    sys.exit()

os.makedirs(path_data, exist_ok=True)


# init grid
massgrid, massbins = coala.init_grid_log(nbins, massmax, massmin)

# print("massgrid = ",massgrid)


match kernel:
    case 0 | 1 | 2:
        gij_init, gij, time_coag = coala.iterate_coag(
            kernel, K0, nbins, kpol, dthydro, ndthydro, coeff_CFL, Q, eps, massgrid, massbins
        )

    case 3:
        # Brownian motion dv with constant approximation
        dv_Br = np.zeros((nbins, nbins))
        massmeanlog = np.sqrt(massgrid[0:nbins] * massgrid[1:])
        for i in range(nbins):
            for j in range(nbins):
                dv_Br[i, j] = np.sqrt(1.0 / massmeanlog[i] + 1.0 / massmeanlog[j])

        gij_init, gij, time_coag = coala.iterate_coag_kdv(
            kernel, K0, nbins, kpol, dthydro, ndthydro, coeff_CFL, Q, eps, massgrid, massbins, dv_Br
        )

    case _:
        print("Need to choose available kernel in kernel_collision.py.")


# print("massgrid=",massgrid)
np.savetxt(path_data + "massgrid_k%d.txt" % (kpol), massgrid)
np.savetxt(path_data + "massbins_k%d.txt" % (kpol), massbins)
np.savetxt(path_data + "gij_init_k%d.txt" % (kpol), gij_init)
np.savetxt(path_data + "gij_end_k%d.txt" % (kpol), gij)
np.savetxt(path_data + "time_k%d.txt" % (kpol), [t0, time_coag])

import numpy as np
from coala import *
from matplotlib import pyplot as plt

plt.rcParams["font.size"] = 16
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["legend.columnspacing"] = 0.5

savefig_options = dict(bbox_inches="tight")

marker_style = dict(
    marker="o", markersize=8, markerfacecolor="white", linestyle="", markeredgewidth=2
)


# parameters to read data

match kernel:
    case 0:
        path_data = "./data/kconst/" + str(nbins) + "bins/Qcoag=" + str(Q) + "/"
        str_kernel = "kconst"
    case 1:
        path_data = "./data/kadd/" + str(nbins) + "bins/Qcoag=" + str(Q) + "/"
        str_kernel = "kadd"
    case _:
        print("Need to choose a simple kernel in the list.")


def I(massgrid, j):
    res = np.logspace(np.log10(massgrid[j]), np.log10(massgrid[j + 1]), num=100)
    return res


try:
    massgrid_k0 = np.loadtxt(path_data + "massgrid_k0.txt")
    massbins_k0 = np.loadtxt(path_data + "massbins_k0.txt")

    gij_t0_k0 = np.loadtxt(path_data + "gij_init_k0.txt")
    gij_tend_k0 = np.loadtxt(path_data + "gij_end_k0.txt")
    time_k0 = np.loadtxt(path_data + "time_k0.txt")

    massmax = massgrid_k0[-1]
    massmin = massgrid_k0[0]
    tend = time_k0[-1]

except:
    print("Missing data for k=0")


try:
    massgrid_k1 = np.loadtxt(path_data + "massgrid_k1.txt")
    massbins_k1 = np.loadtxt(path_data + "massbins_k1.txt")

    gij_t0_k1 = np.loadtxt(path_data + "gij_init_k1.txt")
    gij_tend_k1 = np.loadtxt(path_data + "gij_end_k1.txt")
    time_k1 = np.loadtxt(path_data + "time_k1.txt")

    massmax = massgrid_k1[-1]
    massmin = massgrid_k1[0]
    tend = time_k1[-1]

except:
    print("Missing data for k=1")

try:
    massgrid_k2 = np.loadtxt(path_data + "massgrid_k2.txt")
    massbins_k2 = np.loadtxt(path_data + "massbins_k2.txt")

    gij_t0_k2 = np.loadtxt(path_data + "gij_init_k2.txt")
    gij_tend_k2 = np.loadtxt(path_data + "gij_end_k2.txt")
    time_k2 = np.loadtxt(path_data + "time_k2.txt")

    massmax = massgrid_k2[-1]
    massmin = massgrid_k2[0]
    tend = time_k2[-1]

except:
    print("Missing data for k=2")


x = np.logspace(np.log10(massmin), np.log10(massmax), num=100)


plt.figure(1)
plt.loglog(x, exact_sol_coag(kernel, x, 0.0), "--", c="C0", alpha=0.5)
plt.loglog(x, exact_sol_coag(kernel, x, tend), "--", c="C0", label="Analytic")

try:
    plt.loglog(massbins_k0, gij_t0_k0, markeredgecolor="black", **marker_style, alpha=0.5)
    plt.loglog(massbins_k0, gij_tend_k0, markeredgecolor="black", label="coala k=0", **marker_style)
except:
    print("Need data for k=0 for plot")

try:
    plt.loglog(massbins_k1, gij_t0_k1[:, 0], markeredgecolor="C1", **marker_style, alpha=0.5)
    plt.loglog(
        massbins_k1, gij_tend_k1[:, 0], markeredgecolor="C1", label="coala k=1", **marker_style
    )
except:
    print("Need data for k=1 for plot")

try:
    plt.loglog(massbins_k2, gij_t0_k2[:, 0], markeredgecolor="C2", **marker_style, alpha=0.5)
    plt.loglog(
        massbins_k2, gij_tend_k2[:, 0], markeredgecolor="C2", label="coala k=2", **marker_style
    )
except:
    print("Need data for k=2 for plot")


plt.ylim(1.0e-15, 1.0e1)
plt.xlim(massmin, massmax)
plt.title(str_kernel + ", nbins=%d" % (nbins))
plt.xlabel(r"mass ")
plt.ylabel(r"mass density distribution g")
plt.legend(loc="lower left", ncol=1)
plt.tight_layout()

# create plots directory
# os.makedirs(path_data, exist_ok=True)
# plt.savefig("./plots/"+str_kernel+"_loglog_%dbins_Q%d.pdf"%(nbins,Q),**savefig_options)

plt.show()
