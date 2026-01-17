import sys

sys.path.insert(0, "../src")

import coala

print(f"coala path : {coala.__file__}")

import os

import numpy as np

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
    print("need to choose a kernel")
    sys.exit()

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

match kernel:
    case 0:
        str_kernel = "kconst"
    case 1:
        str_kernel = "kadd"
    case _:
        print("Need to choose a simple kernel in the list.")


def I(massgrid, j):
    res = np.logspace(np.log10(massgrid[j]), np.log10(massgrid[j + 1]), num=100)
    return res


x = np.logspace(np.log10(massmin), np.log10(massmax), num=100)

tend = cases["order k=0"]["time"][-1]
plt.figure(1)
plt.loglog(x, exact_sol_coag(kernel, x, 0.0), "--", c="C0", alpha=0.5)
plt.loglog(x, exact_sol_coag(kernel, x, tend), "--", c="C0", label="Analytic")

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


gij_t0_k0 = [
    1.90527518e-03,
    5.34978592e-03,
    1.49200031e-02,
    4.08236317e-02,
    1.05876532e-01,
    2.36603895e-01,
    3.53238069e-01,
    1.92536780e-01,
    1.28210758e-02,
    7.88742300e-06,
    5.05922357e-15,
    1.00000000e-20,
    1.00000000e-20,
    1.00000000e-20,
    1.00000000e-20,
    1.00000000e-20,
    1.00000000e-20,
    1.00000000e-20,
    1.00000000e-20,
    1.00000000e-20,
]

gij_tend_k0 = [
    9.46923595e-05,
    2.65043632e-04,
    7.32678756e-04,
    1.95693839e-03,
    4.76890398e-03,
    9.28871278e-03,
    1.15251017e-02,
    7.84524840e-03,
    4.18790882e-03,
    2.15758465e-03,
    1.11809027e-03,
    5.73274221e-04,
    2.79582274e-04,
    1.24810694e-04,
    4.80539799e-05,
    1.45278618e-05,
    3.02547109e-06,
    3.71308293e-07,
    2.26007997e-08,
    5.67256497e-10,
]

gij_t0_k1 = [
    [1.90527518e-03, 9.05725016e-04],
    [5.34978592e-03, 2.53498952e-03],
    [1.49200031e-02, 7.00550421e-03],
    [4.08236317e-02, 1.86714390e-02],
    [1.05876532e-01, 4.47777100e-02],
    [2.36603895e-01, 7.68481264e-02],
    [3.53238069e-01, 1.50847901e-02],
    [1.92536780e-01, -1.40538593e-01],
    [1.28210758e-02, -1.28210758e-02],
    [7.88742300e-06, -7.88742300e-06],
    [5.05922357e-15, -5.05921357e-15],
    [1.00000000e-20, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00],
]

gij_tend_k1 = [
    [9.46748867e-05, 4.49313443e-05],
    [2.64905195e-04, 1.24935394e-04],
    [7.31591213e-04, 3.38957363e-04],
    [1.94859122e-03, 8.57842302e-04],
    [4.70971706e-03, 1.78004402e-03],
    [8.97837560e-03, 2.02344363e-03],
    [1.09628859e-02, -3.93487186e-04],
    [8.44513302e-03, -1.48433949e-03],
    [5.14873253e-03, -1.06071864e-03],
    [3.03683434e-03, -6.13512395e-04],
    [1.77279721e-03, -3.39368817e-04],
    [1.00204189e-03, -2.09442170e-04],
    [5.04192232e-04, -1.41730972e-04],
    [1.86341177e-04, -8.79482101e-05],
    [3.25424160e-05, -2.82716732e-05],
    [1.23991390e-06, -1.23991390e-06],
    [1.01484523e-08, -1.01484523e-08],
    [2.08646952e-11, -2.08646952e-11],
    [6.75735584e-15, -6.75734584e-15],
    [3.27695741e-19, -3.17695741e-19],
]


gij_t0_k2 = [
    [1.90527518e-03, 9.05725016e-04, -5.49509960e-07],
    [5.34978592e-03, 2.53498952e-03, -4.34223153e-06],
    [1.49200031e-02, 7.00550421e-03, -3.39884114e-05],
    [4.08236317e-02, 1.86714390e-02, -2.59000879e-04],
    [1.05876532e-01, 4.47777100e-02, -1.82867015e-03],
    [2.36603895e-01, 7.68481264e-02, -1.03458457e-02],
    [3.53238069e-01, 1.50847901e-02, -2.89427354e-02],
    [1.92536780e-01, -1.40538593e-01, 1.94501341e-02],
    [1.28210758e-02, -2.10736118e-02, 1.68642939e-02],
    [7.88742300e-06, -1.01855598e-05, 1.31438063e-05],
    [5.05922357e-15, -6.34995441e-15, 8.54561595e-15],
    [1.00000000e-20, 0.00000000e00, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00, 0.00000000e00],
    [1.00000000e-20, 0.00000000e00, 0.00000000e00],
]

gij_tend_k2 = [
    [9.46740930e-05, 4.49306564e-05, -5.30725953e-08],
    [2.64898939e-04, 1.24930186e-04, -4.16260358e-07],
    [7.31542559e-04, 3.38921500e-04, -3.19048882e-06],
    [1.94822620e-03, 8.57669602e-04, -2.29199278e-05],
    [4.70720297e-03, 1.78056407e-03, -1.37297806e-04],
    [8.96373187e-03, 2.04516882e-03, -4.95078148e-04],
    [1.09001508e-02, -2.95227344e-04, -3.90237343e-04],
    [8.36656837e-03, -1.65208521e-03, 2.89923760e-04],
    [5.25686775e-03, -1.25407691e-03, 2.62553011e-04],
    [3.16148600e-03, -7.64386129e-04, 1.75976220e-04],
    [1.87293760e-03, -4.28005126e-04, 8.32321583e-05],
    [1.06532175e-03, -2.58924824e-04, 3.24894494e-05],
    [5.33049416e-04, -1.78936288e-04, 1.85477876e-05],
    [1.85170566e-04, -1.14608901e-04, 1.94401344e-05],
    [2.57319638e-05, -3.31284417e-05, 1.23865040e-05],
    [6.36584472e-07, -1.08681430e-06, 7.43906502e-07],
    [1.38512136e-08, -2.12361893e-08, 2.02956586e-08],
    [2.95845862e-10, -3.62377036e-10, 5.05016610e-10],
    [9.06693425e-13, -1.01043556e-12, 1.60078727e-12],
    [1.91516589e-16, -2.08222215e-16, 3.40579088e-16],
]
