"""
Dustywave TVA dispersion relation
=======================================

This example shows how to derive the dustywave TVA dispersion relation using SymPy
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# %%
# Usefull symbols
omega = sp.symbols(r"\omega", complex=True)

k, cs, ts, eps = sp.symbols(
    r"k c_s t_s \epsilon",
    positive=True,
    real=True,
)

i = sp.I  # imaginary unit

a = k * cs  # s^-1
b = a**2 * ts * eps  # s^-1


# %%
# The perturbation matrix is M = K(omega = 0)
# The eigenfrequencies are the roots of det(M + i omega I) = 0
K = sp.Matrix(
    [
        [i * omega, 0, -i * a],
        [b * (1 - eps), i * omega - b, 0],
        [-i * a * (1 - eps), i * a, i * omega],
    ]
)  # s^-1


# %%
# Compute the determinant
det = sp.factor(K.det())
print(sp.latex(det))

# %%
# Let's remove the leading omega mode (omega = 0)
det /= i * omega
det = sp.simplify(det)
det = sp.collect(det, omega)
print(sp.latex(det))

# %%
# Find the roots of the dispersion relation
r1, r2 = sp.solve(sp.Eq(det, 0), omega)
print(sp.latex(r1))
print(sp.latex(r2))

print(r1)
print(r2)


# %%
# Function to plot the roots
def get_roots(_r1, _r2, k_list, eps_value, cs_value, ts_value):

    # Substitute all parameters
    r1_num = _r1.subs({cs: cs_value, ts: ts_value, eps: eps_value})

    r2_num = _r2.subs({cs: cs_value, ts: ts_value, eps: eps_value})

    r1_num_re = sp.re(r1_num)
    r1_num_im = sp.im(r1_num)
    r2_num_re = sp.re(r2_num)
    r2_num_im = sp.im(r2_num)

    # Lambdify only k remains
    r1_re_func = sp.lambdify(k, r1_num_re, modules="numpy")
    r1_im_func = sp.lambdify(k, r1_num_im, modules="numpy")
    r2_re_func = sp.lambdify(k, r2_num_re, modules="numpy")
    r2_im_func = sp.lambdify(k, r2_num_im, modules="numpy")

    # Evaluate
    r1_vals_re = r1_re_func(k_list)
    r1_vals_im = r1_im_func(k_list)
    r2_vals_re = r2_re_func(k_list)
    r2_vals_im = r2_im_func(k_list)

    def restore(lst):
        # if it is not a numpy array return a np.zeros_like(k_list)
        if not isinstance(lst, np.ndarray):
            return np.zeros_like(k_list)
        return lst

    r1_vals_re = restore(r1_vals_re)
    r1_vals_im = restore(r1_vals_im)
    r2_vals_re = restore(r2_vals_re)
    r2_vals_im = restore(r2_vals_im)

    return r1_vals_re, r1_vals_im, r2_vals_re, r2_vals_im


def get_roots_LP14(k_list, eps_value, cs_value, ts_value):
    _r1 = +cs * sp.sqrt(1 - eps) * k - i * ts * k**2 * cs**2 * eps / 2
    _r2 = -cs * sp.sqrt(1 - eps) * k - i * ts * k**2 * cs**2 * eps / 2

    return get_roots(_r1, _r2, k_list, eps_value, cs_value, ts_value)


def get_overroots_DCL26_simple(k_list, eps_value, cs_value, ts_value):
    _r1 = +cs * sp.sqrt(1 - eps) * k + i * k**2 * cs**2 * eps * ts * (-1 + 1) / 2
    _r2 = -cs * sp.sqrt(1 - eps) * k + i * k**2 * cs**2 * eps * ts * (-1 - 1) / 2

    return get_roots(_r1, _r2, k_list, eps_value, cs_value, ts_value)


def get_overroots_DCL26(k_list, eps_value, cs_value, ts_value):
    D = 4 * (1 - eps) - eps**2 * cs**2 * ts**2 * k**2

    sqrtD_real = sp.sqrt(sp.Max(D, 0))
    sqrtD_imag = sp.sqrt(sp.Max(-D, 0))

    _r1 = cs * k / 2 * (+sqrtD_real + i * (sqrtD_imag - eps * cs * k * ts))

    _r2 = cs * k / 2 * (-sqrtD_real + i * (-sqrtD_imag - eps * cs * k * ts))

    print(sp.latex(sp.Abs(_r1)))
    print(sp.latex(sp.Abs(_r2)))

    return get_roots(_r1, _r2, k_list, eps_value, cs_value, ts_value)


def plot_case(k_plot, eps_value, cs_value, ts_value):

    r1_vals_re, r1_vals_im, r2_vals_re, r2_vals_im = get_roots(
        r1, r2, k_plot, eps_value, cs_value, ts_value
    )

    r1_vals_re_LP14, r1_vals_im_LP14, r2_vals_re_LP14, r2_vals_im_LP14 = get_roots_LP14(
        k_plot, eps_value, cs_value, ts_value
    )

    (
        r1_vals_re_DCL26_simple,
        r1_vals_im_DCL26_simple,
        r2_vals_re_DCL26_simple,
        r2_vals_im_DCL26_simple,
    ) = get_overroots_DCL26_simple(k_plot, eps_value, cs_value, ts_value)
    r1_vals_re_DCL26, r1_vals_im_DCL26, r2_vals_re_DCL26, r2_vals_im_DCL26 = get_overroots_DCL26(
        k_plot, eps_value, cs_value, ts_value
    )

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex=True)

    # Real parts
    axs[0, 0].plot(k_plot, r1_vals_re, color="0", linewidth=2, label="Re($\omega_+$)")
    axs[0, 0].plot(k_plot, r2_vals_re, color="0", linewidth=2, label="Re($\omega_-$)")
    axs[0, 0].plot(k_plot, r1_vals_re_LP14, "--", label="Re($\omega_{+,LP14}$)")
    axs[0, 0].plot(k_plot, r2_vals_re_LP14, "--", label="Re($\omega_{-,LP14}$)")
    axs[0, 0].plot(
        k_plot, r1_vals_re_DCL26_simple, linestyle="dotted", label="Re($\omega_{+,approx}$)"
    )
    axs[0, 0].plot(
        k_plot, r2_vals_re_DCL26_simple, linestyle="dotted", label="Re($\omega_{-,approx}$)"
    )
    # axs[0,0].plot(k_plot, r1_vals_re_DCL26,"--", label="Re($r_1$) DCL26")
    # axs[0,0].plot(k_plot, r2_vals_re_DCL26,"--", label="Re($r_2$) DCL26")
    axs[0, 0].set_ylabel("Real part")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Imaginary parts
    axs[0, 1].plot(k_plot, r1_vals_im, color="0", linewidth=2, label="Im($\omega_+$)")
    axs[0, 1].plot(k_plot, r2_vals_im, color="0", linewidth=2, label="Im($\omega_-$)")
    axs[0, 1].plot(k_plot, r1_vals_im_LP14, "--", label="Im($\omega_{+,LP14}$)")
    axs[0, 1].plot(k_plot, r2_vals_im_LP14, "--", label="Im($\omega_{-,LP14}$)")
    axs[0, 1].plot(
        k_plot, r1_vals_im_DCL26_simple, linestyle="dotted", label="Im($\omega_{+,approx}$)"
    )
    axs[0, 1].plot(
        k_plot, r2_vals_im_DCL26_simple, linestyle="dotted", label="Im($\omega_{-,approx}$)"
    )
    # axs[0,1].plot(k_plot, r1_vals_im_DCL26,"--", label="Im($r_1$) DCL26")
    # axs[0,1].plot(k_plot, r2_vals_im_DCL26,"--", label="Im($r_2$) DCL26")
    axs[0, 1].set_xlabel("$k$")
    axs[0, 1].set_ylabel("Imaginary part")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Abs
    r1_vals_abs = np.sqrt(r1_vals_re**2 + r1_vals_im**2)
    r2_vals_abs = np.sqrt(r2_vals_re**2 + r2_vals_im**2)
    r1_vals_abs_LP14 = np.sqrt(r1_vals_re_LP14**2 + r1_vals_im_LP14**2)
    r2_vals_abs_LP14 = np.sqrt(r2_vals_re_LP14**2 + r2_vals_im_LP14**2)
    r1_vals_abs_DCL26_simple = np.sqrt(r1_vals_re_DCL26_simple**2 + r1_vals_im_DCL26_simple**2)
    r2_vals_abs_DCL26_simple = np.sqrt(r2_vals_re_DCL26_simple**2 + r2_vals_im_DCL26_simple**2)
    r1_vals_abs_DCL26 = np.sqrt(r1_vals_re_DCL26**2 + r1_vals_im_DCL26**2)
    r2_vals_abs_DCL26 = np.sqrt(r2_vals_re_DCL26**2 + r2_vals_im_DCL26**2)
    axs[1, 0].plot(k_plot, r1_vals_abs, color="0", linewidth=2, label="Abs($\omega_+$)")
    axs[1, 0].plot(k_plot, r2_vals_abs, color="0", linewidth=2, label="Abs($\omega_-$)")
    axs[1, 0].plot(k_plot, r1_vals_abs_LP14, "--", label="Abs($\omega_{+,LP14}$)")
    axs[1, 0].plot(k_plot, r2_vals_abs_LP14, "--", label="Abs($\omega_{-,LP14}$)")
    axs[1, 0].plot(
        k_plot, r1_vals_abs_DCL26_simple, linestyle="dotted", label="Abs($\omega_{+,approx}$)"
    )
    axs[1, 0].plot(
        k_plot, r2_vals_abs_DCL26_simple, linestyle="dotted", label="Abs($\omega_{-,approx}$)"
    )

    def approx(_k, _cs, _ts, _eps):
        print(type(_k), type(_cs), type(_ts), type(_eps))
        return _cs * _k * np.sqrt((1 - _eps) + (_k * _cs * _ts * _eps) ** 2)

    axs[1, 0].plot(
        k_plot,
        approx(k_plot, cs_value, ts_value, eps_value),
        "--",
        label=r"$max(\vert \omega_{\pm,approx} \vert)$",
    )

    # axs[1,0].plot(k_plot, r1_vals_abs_DCL26,"--", label="Abs($r_1$) DCL26")
    # axs[1,0].plot(k_plot, r2_vals_abs_DCL26,"--", label="Abs($r_2$) DCL26")
    axs[1, 0].set_xlabel("$k$")
    axs[1, 0].set_ylabel("Abs part")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # delta with max
    r_max = np.maximum(r1_vals_abs, r2_vals_abs)
    r_max_LP14 = np.maximum(r1_vals_abs_LP14, r2_vals_abs_LP14)
    r_max_DCL26_simple = np.maximum(r1_vals_abs_DCL26_simple, r2_vals_abs_DCL26_simple)
    r_max_DCL26 = np.maximum(r1_vals_abs_DCL26, r2_vals_abs_DCL26)
    axs[1, 1].plot(k_plot, (r_max_LP14 - r_max) / r_max, label="Analytic - LP14")
    axs[1, 1].plot(k_plot, (r_max_DCL26_simple - r_max) / r_max, label="Analytic - approx")
    # axs[1,1].plot(k_plot, (r_max_DCL26 - r_max) / r_max, label="Ana - DCL26")
    axs[1, 1].set_xlabel("$k$")
    axs[1, 1].set_ylabel("Abs(Ana) - Abs(Max model) / Abs(Ana)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.suptitle(f"eps = {eps_value}, cs = {cs_value}, ts = {ts_value}")

    plt.tight_layout()

    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(4, 6), sharex=True)

    # Abs
    r1_vals_abs = np.sqrt(r1_vals_re**2 + r1_vals_im**2)
    r2_vals_abs = np.sqrt(r2_vals_re**2 + r2_vals_im**2)
    r1_vals_abs_LP14 = np.sqrt(r1_vals_re_LP14**2 + r1_vals_im_LP14**2)
    r2_vals_abs_LP14 = np.sqrt(r2_vals_re_LP14**2 + r2_vals_im_LP14**2)
    r1_vals_abs_DCL26_simple = np.sqrt(r1_vals_re_DCL26_simple**2 + r1_vals_im_DCL26_simple**2)
    r2_vals_abs_DCL26_simple = np.sqrt(r2_vals_re_DCL26_simple**2 + r2_vals_im_DCL26_simple**2)
    r1_vals_abs_DCL26 = np.sqrt(r1_vals_re_DCL26**2 + r1_vals_im_DCL26**2)
    r2_vals_abs_DCL26 = np.sqrt(r2_vals_re_DCL26**2 + r2_vals_im_DCL26**2)
    (line_ana,) = axs[0].plot(
        k_plot, r1_vals_abs, color="0", linewidth=2, label=r"Abs($\omega_\pm$)"
    )
    axs[0].plot(k_plot, r2_vals_abs, color=line_ana.get_color(), linewidth=2)
    (line_lp14,) = axs[0].plot(k_plot, r1_vals_abs_LP14, "--", label=r"Abs($\omega_{\pm,LP14}$)")
    axs[0].plot(k_plot, r2_vals_abs_LP14, "--", color=line_lp14.get_color())
    (line_approx,) = axs[0].plot(
        k_plot,
        r1_vals_abs_DCL26_simple,
        label=r"Abs($\omega_{\pm,approx}$)",
        alpha=0.7,
    )
    axs[0].plot(
        k_plot,
        r2_vals_abs_DCL26_simple,
        color=line_approx.get_color(),
        alpha=0.7,
    )

    # axs[1,0].plot(k_plot, r1_vals_abs_DCL26,"--", label="Abs($r_1$) DCL26")
    # axs[1,0].plot(k_plot, r2_vals_abs_DCL26,"--", label="Abs($r_2$) DCL26")
    axs[0].set_xlabel("$k$")
    axs[0].set_ylabel("Abs part")
    axs[0].grid(True)
    axs[0].legend()

    # delta with max
    r_max = np.maximum(r1_vals_abs, r2_vals_abs)
    r_max_LP14 = np.maximum(r1_vals_abs_LP14, r2_vals_abs_LP14)
    r_max_DCL26_simple = np.maximum(r1_vals_abs_DCL26_simple, r2_vals_abs_DCL26_simple)
    r_max_DCL26 = np.maximum(r1_vals_abs_DCL26, r2_vals_abs_DCL26)
    delta_LP14 = (r_max_LP14 - r_max) / r_max
    delta_DCL26_simple = (r_max_DCL26_simple - r_max) / r_max
    axs[1].plot(k_plot, delta_LP14, label="LP14")
    axs[1].plot(k_plot, delta_DCL26_simple, label="approx")
    # axs[1,1].plot(k_plot, (r_max_DCL26 - r_max) / r_max, label="Ana - DCL26")
    axs[1].set_xlabel("$k$")
    axs[1].set_ylabel(
        r"$(\vert \omega_{ana} \vert_{\rm max} - \vert \omega_{model} \vert_{\rm max} )/ \vert \omega_{ana} \vert_{\rm max}$"
    )
    axs[1].grid(True)
    axs[1].legend()
    ymin, ymax = axs[1].get_ylim()
    axs[1].axhspan(ymin=min(ymin, 0) * 10, ymax=0, facecolor="red", alpha=0.15, zorder=0)
    axs[1].text(
        0.05,
        0.05,
        "CFL underestimated",
        transform=axs[1].transAxes,
        ha="left",
        va="bottom",
        color="red",
        fontsize=9,
    )
    plt.ylim(ymin, ymax)
    plt.suptitle(f"$\epsilon = {eps_value}, c_s = {cs_value}, t_s = {ts_value}$")

    plt.tight_layout()
    plt.savefig(f"_to_trash/dustywave_sympy_eps_{eps_value}_cs_{cs_value}_ts_{ts_value}.pdf")


# %%
# Plot the case eps = 0.5, cs = 1.0, ts = 1.0
k_plot = np.linspace(0, 5, 1000)
plot_case(k_plot, 0.5, 1.0, 1.0)
plt.show()


# %%
# Plot the case eps = 0.1, cs = 1.0, ts = 1.0
k_plot = np.linspace(0, 40, 1000)
plot_case(k_plot, 0.1, 1.0, 1.0)
plt.show()
