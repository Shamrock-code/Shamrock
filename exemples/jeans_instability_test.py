from math import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import shamrock

#####============================== matplot config start ===============================

lw, ms = 3, 8  # linewidth #markersize

elw, cs = 0.75, 0.75  # linewidth and capthick #capsize for errorbar specifically

fontsize = 20

tickwidth, ticksize = 1.5, 7

mpl.rcParams["axes.titlesize"] = fontsize

mpl.rcParams["axes.labelsize"] = fontsize

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

mpl.rcParams["xtick.labelsize"] = fontsize * 0.75

mpl.rcParams["ytick.labelsize"] = fontsize * 0.75

mpl.rcParams["legend.fontsize"] = fontsize * 0.5


mpl.rcParams["font.weight"] = "normal"

mpl.rcParams["font.serif"] = "Times New Roman"

####============================ matplot config end ===================


def run_sim(
    rhog,
    vg,
    cs,
    times,
    lembda=2.0,
    rho0=1.0,
):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 8

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    center = (base * scale_fact, base * scale_fact, base * scale_fact)
    xc, yc, zc = center
    cfg.set_Csafe(0.5)
    cfg.set_eos_gamma(1.00000001)
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(True)
    cfg.set_gravity_mode_cg()

    cfg.set_self_gravity_G_values(True, 1.0)
    cfg.set_self_gravity_Niter_max(10)
    cfg.set_self_gravity_tol(1e-10)
    cfg.set_self_gravity_happy_breakdown_tol(1e-6)
    model.set_solver_config(cfg)

    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    # ================= Fields maps  =========================

    def pertubation(x, A) -> float:
        return A * cos((2 * np.pi * x) / lembda)

    A_rho = 1e-2

    gamma = 1.0000001

    ### Gas maps
    def rho_map(rmin, rmax) -> float:
        x_mn, y_mn, z_mn = rmin
        x_mx, y_mx, z_mx = rmax

        x = 0.5 * (x_mn + x_mx)
        y = 0.5 * (y_mn + y_mx)
        z = 0.5 * (z_mn + z_mx)
        return rho0 * (8.0 + pertubation(x, A_rho))

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

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)
    model.set_field_value_lambda_f64("phi", phi_map)

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

    freq = 15
    dt = 0.000
    t = 0
    tend = 2
    for i in range(2):

        # if i % freq == 0:
        model.dump_vtk("test" + str(i // freq) + ".vtk")

        model.evolve_once_override_time(t, dt)

        dic = ctx.collect_data()

        if shamrock.sys.world_rank() == 0:
            dic = convert_to_cell_coords(dic)

        vg_i = dic["rhovel"][0][0] / dic["rho"][0]
        rg_i = dic["rho"][0]

        rhog.append(rg_i)
        vg.append(vg_i)

        next_dt = model.evolve_once_override_time(t, dt)

        t += dt

        times.append(t)
        dt = next_dt

        if tend < t + next_dt:
            dt = tend - t
        if t == tend:
            break


# ================ post treatment =========

## ===== get numerical results ========
times = []
rg_num = []
vg_num = []
cs = 1.0

run_sim(rg_num, vg_num, cs, times)

fig, axs = plt.subplots(1, 2, figsize=(25, 10))
plt.subplots_adjust(wspace=0.25)
axs[0].plot(times, rg_num, "co", label="Gas-num")
axs[0].set_xlabel("Time", fontsize=15, fontweight="bold")
axs[0].set_ylabel("Density", fontsize=15, fontweight="bold")

axs[1].plot(times, vg_num, "co", label="Gas-num")
axs[1].set_xlabel("Time", fontsize=15, fontweight="bold")
axs[1].set_ylabel("Velocity", fontsize=15, fontweight="bold")

plt.legend(prop={"weight": "bold"})
plt.savefig(f"Jeans_instability-{shamrock.sys.world_size()}-{len(rg_num)}.png", format="png")
