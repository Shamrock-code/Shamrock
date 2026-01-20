"""
Using Coala within Shamrock to solve the Smoluchowski equation
==============================================================

"""

# %%
# Imports
import os

import numpy as np
import shamrock.external.coala as coala
from matplotlib import pyplot as plt

# %%
# Where is coala located?
print(f"coala path : {coala.__file__}")


# %%
# Parameters of the dust distribution & evolution
nbins = 20

massmax = 1e6
massmin = 1e-3

kernel = 1
K0 = 1.0
Q = 5
eps = 1e-20
coeff_CFL = 0.3
t0 = 0.0

cases = {
    "order k=0": {
        "kpol": 0,
    },
    "order k=1": {
        "kpol": 1,
    },
    "order k=2": {
        "kpol": 2,
    },
}

if kernel == 0:
    dthydro = 100
    ndthydro = 300
elif kernel == 1:
    dthydro = 1e-2
    ndthydro = 300
elif kernel == 2:
    dthydro = 1e-1
    ndthydro = 500
elif kernel == 3:
    dthydro = 1e-1
    ndthydro = 500
else:
    raise ValueError("need to choose a kernel")

massgrid, massbins = coala.init_grid_log(nbins, massmax, massmin)

for case in cases:
    kpol = cases[case]["kpol"]
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
                kernel,
                K0,
                nbins,
                kpol,
                dthydro,
                ndthydro,
                coeff_CFL,
                Q,
                eps,
                massgrid,
                massbins,
                dv_Br,
            )

        case _:
            print("Need to choose available kernel in kernel_collision.py.")

    cases[case]["massgrid"] = massgrid
    cases[case]["massbins"] = massbins
    cases[case]["gij_init"] = gij_init
    cases[case]["gij_end"] = gij
    cases[case]["time"] = [t0, time_coag]


# %%
# Plotting
plt.rcParams["font.size"] = 16
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["legend.columnspacing"] = 0.5

savefig_options = dict(bbox_inches="tight")

marker_style = dict(
    marker="o", markersize=8, markerfacecolor="white", linestyle="", markeredgewidth=2
)

match kernel:
    case 0:
        str_kernel = "kconst"
    case 1:
        str_kernel = "kadd"
    case _:
        print("Need to choose a simple kernel in the list.")


x = np.logspace(np.log10(massmin), np.log10(massmax), num=100)

tend = cases["order k=0"]["time"][-1]
plt.figure(1)
plt.loglog(x, coala.exact_sol_coag(kernel, x, 0.0), "--", c="C0", alpha=0.5)
plt.loglog(x, coala.exact_sol_coag(kernel, x, tend), "--", c="C0", label="Analytic")

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

plt.show()
