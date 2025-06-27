"""
Shamrock math derivatives functions
=======================================

This example shows how to use Shamrock math derivatives functions
"""

# sphinx_gallery_multi_image = "single"

from math import *

import matplotlib.pyplot as plt
import numpy as np

from shamrock.math import *

# %%
# Compute the error associated to a derivative function


def err_plot(deriv_func, x, f, df, label):
    h = np.logspace(-16, 0, 100)
    err = []

    for i in range(len(h)):
        _err = deriv_func(x, h[i], f) - df(x)
        err.append(fabs(_err))
    plt.plot(h, err, "o", label=label)


def analysis(f, df, x0, label):
    plt.figure()

    # fmt: off
    err_plot(deriv_func=derivative_upwind, x=x0, f=f, df=df, label="derivative_upwind")
    err_plot(deriv_func=derivative_centered, x=x0, f=f, df=df, label="derivative_centered")
    err_plot(deriv_func=derivative_3point_forward, x=x0, f=f, df=df, label="derivative_3point_forward")
    err_plot(deriv_func=derivative_3point_backward, x=x0, f=f, df=df, label="derivative_3point_backward")
    err_plot(deriv_func=derivative_5point_midpoint, x=x0, f=f, df=df, label="derivative_5point_midpoint")
    # fmt: on

    plt.xscale("log")
    plt.yscale("log")

    ymin, ymax = plt.gca().get_ylim()
    plt.ylim(ymin, ymax)

    for i in range(1, 4):
        print(i, estim_deriv_step(i))
        plt.vlines(estim_deriv_step(i), 1e-50, 1e50, color="grey", alpha=0.3)

    plt.xlabel("h")
    plt.ylabel("error")
    plt.title(label)
    plt.legend()


# %%
# Exemple of analysis


def f1(x):
    return exp(x)


def df1(x):
    return exp(x)


analysis(f1, df1, 0, "exp(0)")
analysis(f1, df1, 100, "exp(100)")


def f2(x):
    return sin(x**2)


def df2(x):
    return cos(x**2) * 2 * x


analysis(f2, df2, 0, "sin(0)")
analysis(f2, df2, 100, "sin(100)")

plt.show()
