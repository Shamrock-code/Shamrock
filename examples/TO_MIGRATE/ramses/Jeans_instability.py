from math import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import shamrock

#####============================== matplot config start ===============================

lw, ms = 5, 7  # linewidth  #markersize
elw, cs = 0.75, 0.75  # elinewidth and capthick #capsize for errorbar specifically
fontsize = 35
tickwidth, ticksize = 1.5, 4
mpl.rcParams["axes.titlesize"] = fontsize * 1.5
mpl.rcParams["axes.labelsize"] = fontsize * 1.5
mpl.rcParams["xtick.major.size"] = ticksize
mpl.rcParams["ytick.major.size"] = ticksize
mpl.rcParams["xtick.major.width"] = tickwidth
mpl.rcParams["ytick.major.width"] = tickwidth
mpl.rcParams["xtick.minor.size"] = ticksize
mpl.rcParams["ytick.minor.size"] = ticksize
mpl.rcParams["xtick.minor.width"] = tickwidth
mpl.rcParams["ytick.minor.width"] = tickwidth
mpl.rcParams["lines.linewidth"] = lw
mpl.rcParams["lines.markersize"] = ms
mpl.rcParams["lines.markeredgewidth"] = 1.15
mpl.rcParams["lines.dash_joinstyle"] = "bevel"
mpl.rcParams["markers.fillstyle"] = "top"
mpl.rcParams["lines.dashed_pattern"] = 6.4, 1.6, 1, 1.6
mpl.rcParams["xtick.labelsize"] = fontsize
mpl.rcParams["ytick.labelsize"] = fontsize
mpl.rcParams["legend.fontsize"] = fontsize
mpl.rcParams["grid.linewidth"] = 8
mpl.rcParams["font.weight"] = "normal"
mpl.rcParams["font.serif"] = "latex"

####============================ matplot config end ===================


shamrock.enable_experimental_features()


def run_sim(rhog, vg, etot, cs, times, lembda=0.5, rho0=1, amp=1e-2):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 32

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    center = (base * scale_fact, base * scale_fact, base * scale_fact)
    xc, yc, zc = center
    cfg.set_Csafe(0.5)
    cfg.set_eos_gamma(1.0000001)
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(True)
    cfg.set_gravity_mode_cg()
    cfg.set_riemann_solver_hllc()

    cfg.set_self_gravity_G_values(True, 1.0)
    cfg.set_self_gravity_Niter_max(500)
    cfg.set_self_gravity_tol(1e-6)
    # cfg.set_self_gravity_happy_breakdown_tol(1e-6)
    cfg.set_coupling_gravity_mode_ramses_like()

    model.set_solver_config(cfg)
    model.init_scheduler(int(50000), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    # ================= Fields maps  =========================

    def pertubation(x, A) -> float:
        return A * cos((2 * np.pi * x) / lembda)

    A_rho = amp

    gamma = 1.0000001

    ### Gas maps
    def rho_map(rmin, rmax) -> float:
        x_mn, y_mn, z_mn = rmin
        x_mx, y_mx, z_mx = rmax

        x = 0.5 * (x_mn + x_mx)
        y = 0.5 * (y_mn + y_mx)
        z = 0.5 * (z_mn + z_mx)
        return rho0 * (1.0 + pertubation(x, A_rho))

    def rhovel_map(rmin, rmax) -> tuple[float, float, float]:

        return (0, 0, 0)

    def rhoe_map(rmin, rmax) -> float:
        x, y, z = rmin
        rho = rho_map(rmin, rmax)
        vx = 0
        press = (cs * cs * rho) / gamma
        rhoeint = press / (gamma - 1.0)
        rhoekin = 0.5 * rho * (vx * vx + 0.0)
        return rhoeint + rhoekin

    def phi_map(rmin, rmax):
        rho = rho_map(rmin, rmax)
        return 0

    def phi_old_map(rmin, rmax):
        return 0

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)
    model.set_field_value_lambda_f64("phi", phi_map)
    model.set_field_value_lambda_f64("phi_old", phi_old_map)

    def convert_to_cell_coords(dic):

        cmin = dic["cell_min"]
        cmax = dic["cell_max"]

        xmin = []
        ymin = []
        zmin = []
        xmax = []
        ymax = []
        zmax = []

        for i in range(len(cmin)):
            m, M = cmin[i], cmax[i]

            mx, my, mz = m
            Mx, My, Mz = M

            for j in range(8):
                a, b = model.get_cell_coords(((mx, my, mz), (Mx, My, Mz)), j)

                x, y, z = a
                xmin.append(x)
                ymin.append(y)
                zmin.append(z)

                x, y, z = b
                xmax.append(x)
                ymax.append(y)
                zmax.append(z)

        dic["xmin"] = np.array(xmin)
        dic["ymin"] = np.array(ymin)
        dic["zmin"] = np.array(zmin)
        dic["xmax"] = np.array(xmax)
        dic["ymax"] = np.array(ymax)
        dic["zmax"] = np.array(zmax)

        return dic

    freq = 500
    dt = 0.000
    t = 0
    tend = 0.2
    a = None
    b = None
    c = None
    mask = None
    for i in range(100000):
        dic = ctx.collect_data()

        if shamrock.sys.world_rank() == 0:
            dic = convert_to_cell_coords(dic)

            vg_i = dic["rhovel"][0][0] / dic["rho"][0]
            rg_i = dic["rho"][0]
            e_i = dic["rhoetot"][0]
            a = dic["rho"] - rho0

            b = 0.5 * (dic["xmin"] + dic["xmax"])

            c = dic["rhovel"][:, 0] / (dic["rho"])

            mask = (dic["zmin"] == 0) & (dic["ymin"] == 0)

            rhog.append(rg_i - rho0)
            vg.append(vg_i)
            etot.append(e_i)

        next_dt = model.evolve_once_override_time(t, dt)

        t += dt

        times.append(t)
        dt = next_dt

        if tend < t + next_dt:
            dt = tend - t
        if t == tend:
            break

    return a, c, b, mask


# ================= Analytical ==============
def plot_analytical_solution_osc_times(A, k, rho0, Lambd, w_lambd, times, x_pos):
    density = [A * rho0 * np.cos((2 * pi * x_pos) / Lambd) * np.cos(w_lambd * t) for t in times]
    velocity = [
        ((A * w_lambd) / k) * np.sin((2 * np.pi * x_pos) / Lambd) * np.sin(w_lambd * t)
        for t in times
    ]
    return density, velocity


def plot_analytical_solution_osc_snapshots(A, k, rho0, Lambd, w_lambd, positions, t_last):
    density = [
        A * rho0 * np.cos((2 * pi * x) / Lambd) * np.cos(w_lambd * t_last) for x in positions
    ]
    velocity = [
        ((A * w_lambd) / k) * np.sin((2 * np.pi * x) / Lambd) * np.sin(w_lambd * t_last)
        for x in positions
    ]
    return density, velocity


def plot_analytical_solution_col_times(A, k, rho0, Lambd, gam_lambd, times, x_pos):
    density = [A * rho0 * np.cos((2 * pi * x_pos) / Lambd) * np.cosh(gam_lambd * t) for t in times]
    velocity = [
        -((A * gam_lambd) / k) * np.sin((2 * np.pi * x_pos) / Lambd) * np.sinh(gam_lambd * t)
        for t in times
    ]
    return density, velocity


def plot_analytical_solution_col_snapshots(A, k, rho0, Lambd, gam_lambd, positions, t_last):
    density = [
        A * rho0 * np.cos((2 * pi * x) / Lambd) * np.cosh(gam_lambd * t_last) for x in positions
    ]
    velocity = [
        -((A * gam_lambd) / k) * np.sin((2 * np.pi * x) / Lambd) * np.sinh(gam_lambd * t_last)
        for x in positions
    ]
    return density, velocity


# ====================================================

# ================ post treatment =========

## ===== get numerical results ========
times = []
rg_num = []
vg_num = []
etot_num = []

cs = 2
lembda = 0.5
amp = 1e-4
rho0 = 1
L = 1.0
G = 1.0


rho_last, vel_last, X, mask = run_sim(rg_num, vg_num, etot_num, cs, times, lembda, rho0, amp)


if shamrock.sys.world_rank() == 0:
    # get indexes X at (Y,Z)=(0,0)
    ind = np.where(mask)[0]

    # extract only X, density and velocity
    X = X[ind]
    rho_last = rho_last[ind]
    vel_last = vel_last[ind]

    times = np.array(times)
    x0 = X[0]
    t_last = times[-1]
    k = (2 * np.pi) / (L * lembda)  # wave number
    Lambd_jeans = np.sqrt((np.pi * cs * cs) / (G * rho0))  # Jeans length
    dens_in_time = None
    vel_in_time = None
    dens_fix_time = None
    vel_fix_time = None

    if lembda < Lambd_jeans:
        w_lambd = 2 * np.pi * cs * np.sqrt((1.0 / (lembda**2) - 1.0 / (Lambd_jeans**2)))
        dens_in_time, vel_in_time = plot_analytical_solution_osc_times(
            amp, k, rho0, lembda, w_lambd, times, x0
        )
        dens_fix_time, vel_fix_time = plot_analytical_solution_osc_snapshots(
            amp, k, rho0, lembda, w_lambd, X, t_last
        )

    elif lembda > Lambd_jeans:
        gam_lambd = 2 * np.pi * cs * np.sqrt((1.0 / (Lambd_jeans**2) - 1.0 / (lembda**2)))
        dens_in_time, vel_in_time = plot_analytical_solution_col_times(
            amp, k, rho0, lembda, gam_lambd, times, x0
        )
        dens_fix_time, vel_fix_time = plot_analytical_solution_col_snapshots(
            amp, k, rho0, lembda, gam_lambd, X, t_last
        )

    datas_times = np.stack((times, rg_num, dens_in_time, vg_num, vel_in_time)).T
    np.savetxt(
        f"Jeans-instablity-test-datas-times-for-{shamrock.sys.world_size()}-amp-{amp}-cs-{cs}-rho0-{rho0}-lambda-{lembda}-{len(X)}--{len(rg_num)}-cfl-{0.5}-ramses",
        datas_times,
    )

    datas_spaces = np.stack((X, rho_last, dens_fix_time, vel_last, vel_fix_time)).T
    np.savetxt(
        f"Jeans-instablity-test-datas-spaces-for-{shamrock.sys.world_size()}-amp-{amp}-cs-{cs}-rho0-{rho0}-lambda-{lembda}-{len(X)}--{len(rg_num)}-cfl-{0.5}-ramses",
        datas_spaces,
    )

    # datas_extrema = np.stack((maxima_pos, maxima , minima_pos, minima)).T
    # np.savetxt(f"Jeans-instablity-test-datas-extremas-for-{shamrock.sys.world_size()}-amp-{amp}-cs-{cs}-rho0-{rho0}-lambda-{lembda}-{len(X)}--{len(rg_num)}", datas_extrema)

    fig, axs = plt.subplots(2, 2, figsize=(40, 35))
    plt.subplots_adjust(wspace=0.25)
    axs[0][0].plot(times, rg_num, "co", label="$\\rho_{num}$")
    axs[0][0].plot(times, dens_in_time, "k--", label="$\\rho_{ana}$")
    axs[0][0].set_xlabel("Time", fontsize=fontsize, fontweight="bold")
    axs[0][0].set_ylabel("Density in time", fontsize=fontsize, fontweight="bold")
    axs[0][0].legend(prop={"weight": "bold"}, loc="best")

    axs[0][1].plot(times, vg_num, "co", label="$v_{num}$")
    axs[0][1].plot(times, vel_in_time, "k--", label="$v_{ana}$")
    axs[0][1].set_xlabel("Time", fontsize=fontsize, fontweight="bold")
    axs[0][1].set_ylabel("Velocity in time ", fontsize=fontsize, fontweight="bold")
    axs[0][1].legend(prop={"weight": "bold"}, loc="best")

    axs[1][0].plot(X, rho_last, "co", label="$\\rho_{num}$")
    axs[1][0].plot(X, dens_fix_time, "k--", label="$\\rho_{ana}$")
    axs[1][0].set_xlabel(r"$\mathbf{x}$", fontsize=fontsize, fontweight="bold")
    axs[1][0].set_ylabel(
        "Density at $t_{final}$ = " + f"{t_last} ", fontsize=fontsize, fontweight="bold"
    )
    axs[1][0].legend(prop={"weight": "bold"}, loc="best")

    axs[1][1].plot(X, vel_last, "co", label="$v_{num}$")
    axs[1][1].plot(X, vel_fix_time, "k--", label="$v_{ana}$")
    axs[1][1].set_xlabel(r"$\mathbf{x}$", fontsize=fontsize, fontweight="bold")
    axs[1][1].set_ylabel(
        "Velocity at $t_{final}$ = " + f"{t_last}", fontsize=fontsize, fontweight="bold"
    )
    axs[1][1].legend(prop={"weight": "bold"}, loc="best")

    fig.text(
        0.5,
        0.90,
        "Time Evolution of $X_{0}$  ",
        ha="center",
        fontsize=fontsize + 2,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.49,
        "Spatial Profiles (in X-direction) ",
        ha="center",
        fontsize=fontsize + 2,
        fontweight="bold",
    )

    plt.legend(prop={"weight": "bold"})
    plt.savefig(
        f"Jeans_instability-{shamrock.sys.world_size()}-amp-{amp}-cs-{cs}-rho0-{rho0}-lambda-{lembda}-{len(X)}--{len(rg_num)}-V-Check.pdf",
        format="pdf",
    )
