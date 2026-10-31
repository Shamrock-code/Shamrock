"""
Jeans instability characteristic timescales
===========================================

Characteristic timescales of the Jeans instability with the self-gravity of the RAMSES solver,
compared with the linear theory, as a function of :math:`\\lambda / \\lambda_{\\rm J}`.

A uniform periodic box with a small standing density perturbation
:math:`\\delta \\rho = A \\rho_0 \\cos(k x)` (gas at rest, isothermal) follows the dispersion
relation :math:`\\omega^2 = c_s^2 k^2 - 4 \\pi G \\rho_0`, with the Jeans length
:math:`\\lambda_{\\rm J} = \\sqrt{\\pi c_s^2 / (G \\rho_0)}`:

* for :math:`\\lambda < \\lambda_{\\rm J}` the perturbation oscillates, with the period
  :math:`T_{\\rm osc} = \\sqrt{\\pi / (G \\rho_0)} \\, \\lambda / \\sqrt{\\lambda_{\\rm J}^2 - \\lambda^2}`,
* for :math:`\\lambda > \\lambda_{\\rm J}` it collapses, with the e-folding time
  :math:`T_{\\rm col} = \\lambda / \\sqrt{4 \\pi G \\rho_0 (\\lambda^2 - \\lambda_{\\rm J}^2)}`.

The timescales are measured on the density of one cell, as in the analysis of the
``new_amr_plus_grav`` branch of Léodasce Sewanou: the mean period between zero crossings for
the oscillating modes, and the time for the perturbation to go from :math:`A` to
:math:`A \\cosh 1` for the collapsing ones.

The setup is the one of that analysis (:math:`\\lambda = 0.5`, :math:`A = 10^{-4}`, :math:`G = 1`,
32 cells per wavelength, CFL 0.5), with a periodic box of one wavelength.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

from shamrock.model_ramses import ExtForceCoupling

shamrock.enable_experimental_features()

# %%
# Parameters
# ^^^^^^^^^^

G = 1.0
rho0 = 1.0
Lambd = 0.5  # wavelength of the perturbation
A = 1e-4  # relative amplitude of the perturbation
gamma = 1.0000001  # isothermal

L = Lambd  # periodic box of one wavelength
sz = 1 << 1
base = 16  # 32 cells per wavelength

# sound speeds of the collapsing and of the oscillating modes
cs_col_list = [0.105, 0.128, 0.1548, 0.1768, 0.2042, 0.2262, 0.2518, 0.2664, 0.2737, 0.2792]
cs_osc_list = [0.45, 0.55, 0.65, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]

# G = 1 in the code units (unit_mass = 1 / G_SI kg), self-gravity takes G from the unit system
G_SI = shamrock.Constants(shamrock.UnitSystem()).G()
units = shamrock.UnitSystem(unit_time=1.0, unit_length=1.0, unit_mass=1.0 / G_SI)

# %%
# Linear theory
# ^^^^^^^^^^^^^

tff = np.sqrt(3 * np.pi / (32 * rho0 * G))  # free fall time


def jeans_length(cs):
    return np.sqrt(np.pi * cs * cs / (G * rho0))


def t_osc(cs):
    lj = jeans_length(cs)
    return np.sqrt(np.pi / (G * rho0)) * Lambd / np.sqrt(lj**2 - Lambd**2)


def t_col(cs):
    lj = jeans_length(cs)
    return np.sqrt(1 / (4 * np.pi * G * rho0)) * Lambd / np.sqrt(Lambd**2 - lj**2)


# %%
# Simulation
# ^^^^^^^^^^
# Self-gravity is applied as an external force, with the trapezoidal coupling. Each run records
# the density perturbation of the first cell (:math:`x_0 = \Delta x / 2`) until its timescale
# can be measured: past :math:`A \cosh 1` for a collapsing mode, after three zero crossings for
# an oscillating one.


def run_sim(cs, kind):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(L / (sz * base))
    cfg.set_Csafe(0.5)
    cfg.set_eos_gamma(gamma)
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(True)
    cfg.set_riemann_solver_hllc()
    cfg.set_units(units)
    cfg.set_gravity_mode_cg()
    cfg.set_self_gravity_Niter_max(500)
    cfg.set_self_gravity_tol(1e-6)
    cfg.set_ext_force_coupling(ExtForceCoupling.Trapezoidal)
    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base, base, base))

    def rho_map(rmin, rmax):
        x = 0.5 * (rmin[0] + rmax[0])
        return rho0 * (1.0 + A * np.cos(2 * np.pi * x / Lambd))

    def rhoe_map(rmin, rmax):
        press = cs * cs * rho_map(rmin, rmax) / gamma
        return press / (gamma - 1.0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", lambda rmin, rmax: (0.0, 0.0, 0.0))

    if kind == "col":

        def stop(t, d):
            return d > 1.5 * A * rho0 * np.cosh(1.0)

    else:
        omega = 2 * np.pi / t_osc(cs)

        def stop(t, d):
            return t > 2.6 * np.pi / omega  # 3 zero crossings of cos(omega t) + margin

    times, drho = [], []
    t, dt = 0.0, 0.0
    while True:
        data = ctx.collect_data()
        d = data["rho"][0] - rho0
        times.append(t)
        drho.append(d)
        if stop(t, d):
            break
        next_dt = model.evolve_once_override_time(t, dt)
        t += dt
        dt = next_dt

    return np.array(times), np.array(drho)


# %%
# Measured timescales
# ^^^^^^^^^^^^^^^^^^^


def collapse_time(t, rho):
    """Time for the perturbation to go from A to A cosh(1)"""
    time_of_rho = interp1d(rho, t, kind="linear", fill_value="extrapolate")
    return float(time_of_rho(A * rho0 * np.cosh(1))) - float(time_of_rho(A * rho0))


def zero_crossings(t, rho):
    """Interpolated zero-crossing times of rho(t)"""
    zt = []
    for i in range(len(rho) - 1):
        if rho[i] == 0:
            zt.append(t[i])
        elif rho[i] * rho[i + 1] < 0:
            alpha = -rho[i] / (rho[i + 1] - rho[i])
            zt.append(t[i] + alpha * (t[i + 1] - t[i]))
    return np.array(zt)


def mean_period(t, rho):
    """Mean period from every second zero crossing"""
    zt = zero_crossings(t, rho)
    return np.mean(zt[2:] - zt[:-2])


# %%
# Run the simulations
# ^^^^^^^^^^^^^^^^^^^

T_col_num = []
for cs in cs_col_list:
    t, rho = run_sim(cs, "col")
    T_col_num.append(collapse_time(t, rho))

T_osc_num = []
for cs in cs_osc_list:
    t, rho = run_sim(cs, "osc")
    T_osc_num.append(mean_period(t, rho))

cs_c = np.array(cs_col_list)
cs_o = np.array(cs_osc_list)
T_col_num = np.array(T_col_num)
T_osc_num = np.array(T_osc_num)

X_c = Lambd / jeans_length(cs_c)
X_o = Lambd / jeans_length(cs_o)
err_col = np.abs(T_col_num - t_col(cs_c)) / t_col(cs_c)
err_osc = np.abs(T_osc_num - t_osc(cs_o)) / t_osc(cs_o)

print(" cs       lambda/lJ   T_col(ana)  T_col(num)  err")
for c, x, tn, e in zip(cs_c, X_c, T_col_num, err_col):
    print(f" {c:<8} {x:.4f}     {t_col(c):.5f}     {tn:.5f}     {e:.3e}")
print(" cs       lambda/lJ   T_osc(ana)  T_osc(num)  err")
for c, x, tn, e in zip(cs_o, X_o, T_osc_num, err_osc):
    print(f" {c:<8} {x:.4f}     {t_osc(c):.5f}     {tn:.5f}     {e:.3e}")

# %%
# Plot
# ^^^^

mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.facecolor": "white",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
    }
)

SINGLE_COLUMN_WIDTH = 88 / 25.4  # inches (A&A column)
fontsize = 8
fs_label = fontsize + 1
fs_tick = fontsize - 0.5
fs_small = fontsize - 1
lw, ms = 1.0, 3.5

cc = np.sqrt(G * rho0 * Lambd**2 / np.pi)  # critical sound speed
cs_osc_curve = np.linspace(cc * 1.0005, 4.3, 600)
cs_col_curve = np.linspace(0.1, cc * 0.9995, 600)

fig, (ax, axr) = plt.subplots(
    2,
    1,
    figsize=(SINGLE_COLUMN_WIDTH, 4.7),
    sharex=True,
    layout="constrained",
    gridspec_kw={"height_ratios": [3, 2]},
)
fig.get_layout_engine().set(hspace=0.0, h_pad=0.02, w_pad=0.02)

# top panel: timescales
ax.plot(
    Lambd / jeans_length(cs_osc_curve),
    t_osc(cs_osc_curve) / tff,
    color="black",
    lw=lw,
    label="linear theory",
)
ax.plot(Lambd / jeans_length(cs_col_curve), t_col(cs_col_curve) / tff, color="black", lw=lw)
ax.plot(
    X_o,
    T_osc_num / tff,
    color="C0",
    ls="none",
    marker="o",
    markersize=ms,
    label=r"oscillating, period $T_{\rm osc}$",
)
ax.plot(
    X_c,
    T_col_num / tff,
    color="C1",
    ls="none",
    marker="s",
    markersize=ms,
    label=r"collapsing, $e$-folding time $T_{\rm col}$",
)
ax.set_ylim(0, 6.5)
ax.set_ylabel(r"$T \, / \, t_{\rm ff}$", fontsize=fs_label, labelpad=2)
ax.set_title(
    r"Jeans instability, $\lambda = 0.5$, $A = 10^{-4}$, 32 cells per $\lambda$",
    fontsize=fs_label,
    pad=4,
)
ax.text(0.5, 6.0, "stable", fontsize=fs_small, ha="center", va="center")
ax.text(1.95, 6.0, "unstable", fontsize=fs_small, ha="center", va="center")

# bottom panel: relative error
axr.set_yscale("log")
axr.plot(X_o, err_osc, color="C0", ls="none", marker="o", markersize=ms, label="oscillating")
axr.plot(
    X_c,
    err_col,
    color="C1",
    ls="none",
    marker="s",
    markersize=ms,
    label=r"collapsing, $\rho = A \to A\cosh 1$ timing",
)
axr.set_ylim(1e-3, 1.0)
axr.set_ylabel(r"$|T_{\rm num} - T_{\rm ana}| \, / \, T_{\rm ana}$", fontsize=fs_label, labelpad=2)
axr.set_xlabel(r"$\lambda \, / \, \lambda_{\rm J}$", fontsize=fs_label, labelpad=2)

legend_kw = dict(
    fontsize=fs_small,
    handlelength=1.6,
    handletextpad=0.4,
    borderpad=0.4,
    labelspacing=0.3,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    framealpha=1.0,
)
for a in (ax, axr):
    a.axvline(1.0, color="0.4", linewidth=1.0, linestyle="--", zorder=0)
    a.set_xlim(0, 2.9)
    a.grid(axis="y", which="major", linestyle=":", alpha=0.5, lw=0.5)
    a.grid(axis="x", which="major", linestyle=":", alpha=0.5, lw=0.5)
    a.tick_params(which="major", length=4, width=0.6, labelsize=fs_tick)
    a.tick_params(which="minor", length=2, width=0.5)
    for spine in a.spines.values():
        spine.set_linewidth(0.6)
leg = ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), **legend_kw)
leg.get_frame().set_linewidth(0.6)
leg = fig.legend(*axr.get_legend_handles_labels(), loc="outside lower center", **legend_kw)
leg.get_frame().set_linewidth(0.6)

plt.show()
