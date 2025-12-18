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


def run_sim(times, x0, normalized_rd_num, normalized_rg_num, normalized_vd_num, normalized_vg_num):
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
    cfg.set_Csafe(0.5)
    cfg.set_eos_gamma(1.0000001)
    cfg.set_dust_mode_hb(4)
    # cfg.set_drag_mode_irk1(True)
    cfg.set_drag_mode_expo(True)
    cfg.set_face_time_interpolation(True)

    # =================== set drag coefficients for 2 fluids dust =========
    # cfg.set_alpha_values(2.5)          # ts = 0.4

    # =================== set drag coefficients for 5 fluids dust =========
    cfg.set_alpha_values(float(1.0 / 0.1))  # ts = 0.1
    cfg.set_alpha_values(float(1.0 / 0.215443))  # ts = 0.215443
    cfg.set_alpha_values(float(1.0 / 0.464159))  # ts = 0.464159
    cfg.set_alpha_values(1.0)  # ts = 1.0

    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    # ================= Fields maps  =========================

    def pertubation(x, A, Re, Im, L) -> float:
        return A * (Re * cos(2 * x * pi / L) - Im * sin(2 * x * pi / L))

    """
    ##  2 fluids test setup
    rhog_0    = 1.00000
    rhod_0    = 2.240000
    L         = 1
    A_rho     = 1e-4
    A_vel     = 1e-4
    Re_rho    = 1.0
    Im_rho    = 0.0
    Re_vel    = -0.7019594018594713
    Im_vel    = -0.30492431884998994
    Re_rd     = 0.16525079505682766
    Im_rd     = -1.247800745895827
    Re_vd     = -0.22164470614182466
    Im_vd     = 0.3685341424583546
    cs        = 1.0
    gamma     = 1.0000001
    ### Gas maps
    def rho_map(rmin,rmax)->float:
        x,y,z = rmin
        return rhog_0 + pertubation(x,A_rho,Re_rho,Im_rho,L)
    def rhovel_map(rmin, rmax)->tuple[float,float,float]:
        x,y,z = rmin
        rho = rhog_0 + pertubation(x,A_rho,Re_rho,Im_rho,L)
        vx  = pertubation(x,A_vel,Re_vel,Im_vel,L)
        return (rho*vx, 0, 0)
    def rhoe_map (rmin, rmax)->float:
        x,y,z   = rmin
        rho     = rhog_0 + pertubation(x,A_rho,Re_rho,Im_rho,L)
        vx      = pertubation(x,A_vel,Re_vel,Im_vel,L)
        press   = (cs * cs * rho) / gamma
        rhoeint = press / (gamma - 1.0)
        rhoekin = 0.5 * rho * (vx *vx + 0.0)
        return (rhoeint + rhoekin)
    ### Dust maps
    def rho_d_map(rmin,rmax)->float:
        x,y,z = rmin
        return rhod_0 + pertubation(x,A_rho,Re_rd,Im_rd,L)
    def rhovel_d_map(rmin, rmax)->tuple[float,float,float]:
        x,y,z = rmin
        rho = rhod_0 + pertubation(x,A_rho,Re_rd,Im_rd,L)
        vx  = pertubation(x,A_vel,Re_vd,Im_vd,L)
        return (rho*vx, 0, 0)"""

    ##  5 fluids test setup
    L = 1
    A_rho = 1e-4
    A_vel = 1e-4

    rhog_0 = 1.000000
    Re_rho = 1.0
    Im_rho = 0.0

    Re_vel = -0.874365
    Im_vel = -0.145215

    rhod_1 = 0.100000
    rhod_2 = 0.233333
    rhod_3 = 0.366667
    rhod_4 = 0.500000

    Re_rd_1 = 0.080588
    Im_rd_1 = -0.048719
    Re_rd_2 = 0.09160
    Im_rd_2 = -0.134955
    Re_rd_3 = 0.030927
    Im_rd_3 = -0.136799
    Re_rd_4 = 0.001451
    Im_rd_4 = -0.090989

    Re_vd_1 = -0.775380
    Im_vd_1 = 0.308952
    Re_vd_2 = -0.427268
    Im_vd_2 = 0.448704
    Re_vd_3 = -0.127928
    Im_vd_3 = 0.313967
    Re_vd_4 = -0.028963
    Im_vd_4 = 0.158693

    cs = 1.0
    gamma = 1.0000001

    ### Gas maps
    def rho_map(rmin, rmax) -> float:
        x, y, z = rmin
        return rhog_0 + pertubation(x, A_rho, Re_rho, Im_rho, L)

    def rhovel_map(rmin, rmax) -> tuple[float, float, float]:
        x, y, z = rmin
        rho = rhog_0 + pertubation(x, A_rho, Re_rho, Im_rho, L)
        vx = pertubation(x, A_vel, Re_vel, Im_vel, L)
        return (rho * vx, 0, 0)

    def rhoe_map(rmin, rmax) -> float:
        x, y, z = rmin
        rho = rhog_0 + pertubation(x, A_rho, Re_rho, Im_rho, L)
        vx = pertubation(x, A_vel, Re_vel, Im_vel, L)
        press = (cs * cs * rho) / gamma
        rhoeint = press / (gamma - 1.0)
        rhoekin = 0.5 * rho * (vx * vx + 0.0)
        return rhoeint + rhoekin

    ### Dusts maps

    def rho_d_1_map(rmin, rmax) -> float:
        x, y, z = rmin
        return rhod_1 + pertubation(x, A_rho, Re_rd_1, Im_rd_1, L)

    def rhovel_d_1_map(rmin, rmax) -> tuple[float, float, float]:
        x, y, z = rmin
        rho = rhod_1 + pertubation(x, A_rho, Re_rd_1, Im_rd_1, L)
        vx = pertubation(x, A_vel, Re_vd_1, Im_vd_1, L)
        return (rho * vx, 0, 0)

    def rho_d_2_map(rmin, rmax) -> float:
        x, y, z = rmin
        return rhod_2 + pertubation(x, A_rho, Re_rd_2, Im_rd_2, L)

    def rhovel_d_2_map(rmin, rmax) -> tuple[float, float, float]:
        x, y, z = rmin
        rho = rhod_2 + pertubation(x, A_rho, Re_rd_2, Im_rd_2, L)
        vx = pertubation(x, A_vel, Re_vd_2, Im_vd_2, L)
        return (rho * vx, 0, 0)

    def rho_d_3_map(rmin, rmax) -> float:
        x, y, z = rmin
        return rhod_3 + pertubation(x, A_rho, Re_rd_3, Im_rd_3, L)

    def rhovel_d_3_map(rmin, rmax) -> tuple[float, float, float]:
        x, y, z = rmin
        rho = rhod_3 + pertubation(x, A_rho, Re_rd_3, Im_rd_3, L)
        vx = pertubation(x, A_vel, Re_vd_3, Im_vd_3, L)
        return (rho * vx, 0, 0)

    def rho_d_4_map(rmin, rmax) -> float:
        x, y, z = rmin
        return rhod_4 + pertubation(x, A_rho, Re_rd_4, Im_rd_4, L)

    def rhovel_d_4_map(rmin, rmax) -> tuple[float, float, float]:
        x, y, z = rmin
        rho = rhod_4 + pertubation(x, A_rho, Re_rd_4, Im_rd_4, L)
        vx = pertubation(x, A_vel, Re_vd_4, Im_vd_4, L)
        return (rho * vx, 0, 0)

    # ============ set init fields values for gas =============
    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    # ============ set init fields values for dusts [2 fluid case] =============
    # model.set_field_value_lambda_f64("rho_dust", rho_d_map)
    # model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_d_map)

    # ============ set init fields values for dusts [5 fluid case] =============
    model.set_field_value_lambda_f64("rho_dust", rho_d_1_map, 0)
    model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_d_1_map, 0)
    model.set_field_value_lambda_f64("rho_dust", rho_d_2_map, 1)
    model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_d_2_map, 1)
    model.set_field_value_lambda_f64("rho_dust", rho_d_3_map, 2)
    model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_d_3_map, 2)
    model.set_field_value_lambda_f64("rho_dust", rho_d_4_map, 3)
    model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_d_4_map, 3)

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
    for i in range(1000000):

        if i % freq == 0:
            model.dump_vtk("test" + str(i // freq) + ".vtk")

            dic_i = convert_to_cell_coords(ctx.collect_data())

            # vg_i = dic_i["rhovel"][0][0] / dic_i["rho"][0]
            # rg_i = dic_i["rho"][0]
            # rd_i = dic_i["rho_dust"][0]
            # vd_i = dic_i["rhovel_dust"][0][0] / dic_i["rho_dust"][0]
            # x0 = dic_i["xmin"][0]
            # normalized_rg_num.append((rg_i - rhog_0)/A_rho)
            # normalized_rd_num.append((rd_i - rhod_0)/(A_rho * rhod_0))
            # normalized_vg_num.append(vg_i/A_vel)
            # normalized_vd_num.append(vd_i/A_vel)

            vg_i = dic_i["rhovel"][0][0] / dic_i["rho"][0]
            rg_i = dic_i["rho"][0]
            rd_1_i = dic_i["rho_dust"][0]
            vd_1_i = dic_i["rhovel_dust"][0][0] / dic_i["rho_dust"][0]
            rd_2_i = dic_i["rho_dust"][1]
            vd_2_i = dic_i["rhovel_dust"][1][0] / dic_i["rho_dust"][1]
            rd_3_i = dic_i["rho_dust"][2]
            vd_3_i = dic_i["rhovel_dust"][2][0] / dic_i["rho_dust"][2]
            rd_4_i = dic_i["rho_dust"][3]
            vd_4_i = dic_i["rhovel_dust"][3][0] / dic_i["rho_dust"][3]

            x0 = dic_i["xmin"][0]

            normalized_rg_num.append((rg_i - rhog_0) / A_rho)
            normalized_vg_num.append(vg_i / A_vel)

            normalized_rd_num[0].append((rd_1_i - rhod_1) / (A_rho * rhod_1))
            normalized_vd_num[0].append(vd_1_i / A_vel)
            normalized_rd_num[1].append((rd_2_i - rhod_2) / (A_rho * rhod_2))
            normalized_vd_num[1].append(vd_2_i / A_vel)
            normalized_rd_num[2].append((rd_3_i - rhod_3) / (A_rho * rhod_3))
            normalized_vd_num[2].append(vd_3_i / A_vel)
            normalized_rd_num[3].append((rd_4_i - rhod_4) / (A_rho * rhod_4))
            normalized_vd_num[3].append(vd_4_i / A_vel)

        next_dt = model.evolve_once_override_time(t, dt)

        t += dt

        if i % freq == 0:
            times.append(t)
        dt = next_dt

        if tend < t + next_dt:
            dt = tend - t
        if t == tend:
            break


# ================ post treatment =========

## ===== get numerical results ========
times = []
# normalized_rd_num    = []
# normalized_vd_num    = []
normalized_rg_num = []
normalized_vg_num = []
normalized_rd_num = [[], [], [], []]
normalized_vd_num = [[], [], [], []]
x0 = 0
# rhod_0    = 2.240000

rhod_1 = 0.100000
rhod_2 = 0.233333
rhod_3 = 0.366667
rhod_4 = 0.500000

run_sim(times, x0, normalized_rd_num, normalized_rg_num, normalized_vd_num, normalized_vg_num)

## ========= get analytical values ========

from cmath import *


## analytical function =============
def analytical_values(t, w, x, delta):
    res = 0.0 + 0.0j
    res = delta * exp(-t * w) * exp(pi * x * (2j))
    return res.real, res.imag


"""
## 2 fluid gas and dust analytical solutions
w = 1.9158960 - 4.410541j
norm_rg_re = [analytical_values(t,w,x0,1.0 + 0.0j)[0] for t in times]
norm_rg_im = [analytical_values(t,w,x0,1.0 + 0.0j)[1] for t in times]
norm_vg_re = [analytical_values(t,w,x0,-0.701960 - 0.304924j)[0] for t in times]
norm_vg_im = [analytical_values(t,w,x0,-0.701960 - 0.304924j)[1] for t in times]
norm_rd_re = [(1.0/rhod_0) * analytical_values(t,w,x0,0.165251 - 1.247801j)[0] for t in times]
norm_rd_im = [analytical_values(t,w,x0,0.165251 - 1.247801j)[1] for t in times]
norm_vd_re = [analytical_values(t,w,x0,-0.221645 + 0.368534j)[0] for t in times]
norm_vd_im = [analytical_values(t,w,x0,-0.221645 + 0.368534j)[1] for t in times]"""


# ## 5 fluid gas and dust analytical solutions
w = 0.912414 - 5.493800j
norm_rg_re = [analytical_values(t, w, x0, 1.0 + 0.0j)[0] for t in times]
norm_rg_im = [analytical_values(t, w, x0, 1.0 + 0.0j)[1] for t in times]
norm_vg_re = [analytical_values(t, w, x0, -0.874365 - 0.145215j)[0] for t in times]
norm_vg_im = [analytical_values(t, w, x0, -0.874365 - 0.145215j)[1] for t in times]

norm_rd_1_re = [
    (1.0 / rhod_1) * analytical_values(t, w, x0, 0.080588 - 0.048719j)[0] for t in times
]
norm_rd_1_im = [
    (1.0 / rhod_1) * analytical_values(t, w, x0, 0.080588 - 0.048719j)[1] for t in times
]
norm_vd_1_im = [analytical_values(t, w, x0, -0.775380 + 0.308952j)[1] for t in times]
norm_vd_1_re = [analytical_values(t, w, x0, -0.775380 + 0.308952j)[0] for t in times]

norm_rd_2_re = [
    (1.0 / rhod_2) * analytical_values(t, w, x0, 0.0916074536315816 - 0.13495523475722326j)[0]
    for t in times
]
norm_rd_2_im = [
    (1.0 / rhod_2) * analytical_values(t, w, x0, 0.0916074536315816 - 0.13495523475722326j)[1]
    for t in times
]
norm_vd_2_re = [analytical_values(t, w, x0, -0.427268 + 0.448704j)[0] for t in times]
norm_vd_2_im = [analytical_values(t, w, x0, -0.427268 + 0.448704j)[1] for t in times]

norm_rd_3_re = [
    (1.0 / rhod_3) * analytical_values(t, w, x0, 0.030927 - 0.136799j)[0] for t in times
]
norm_rd_3_im = [
    (1.0 / rhod_3) * analytical_values(t, w, x0, 0.030927 - 0.136799j)[1] for t in times
]
norm_vd_3_re = [analytical_values(t, w, x0, -0.127928 + 0.313967j)[0] for t in times]
norm_vd_3_im = [analytical_values(t, w, x0, -0.127928 + 0.313967j)[1] for t in times]

norm_rd_4_re = [
    (1.0 / rhod_4) * analytical_values(t, w, x0, 0.001451 - 0.090989j)[0] for t in times
]
norm_rd_4_im = [
    (1.0 / rhod_4) * analytical_values(t, w, x0, 0.001451 - 0.090989j)[1] for t in times
]
norm_vd_4_re = [analytical_values(t, w, x0, -0.028963 + 0.158693j)[0] for t in times]
norm_vd_4_im = [analytical_values(t, w, x0, -0.028963 + 0.158693j)[1] for t in times]

"""
# =============== plots ==================
## 2 fluids

fig, axs = plt.subplots(1,2,figsize=(25,10))
plt.subplots_adjust(wspace=0.25)
axs[0].plot(times, normalized_rd_num, 'bo', lw = 3, label="Dust-num")
axs[0].plot(times, normalized_rg_num, 'r*', lw = 3, label="Gas-num")
axs[0].plot(times, norm_rd_re, 'b', lw = 1, label="Dust-ana" )
axs[0].plot(times, norm_rg_re, 'r', lw = 1, label="Gas-ana")
axs[0].set_xlabel('Time', fontsize=15,fontweight='bold')
axs[0].set_ylabel('Normalized Density', fontsize=15, fontweight='bold')
axs[1].plot(times, normalized_vd_num, 'bo', lw = 3, label="Dust-num")
axs[1].plot(times, normalized_vg_num, 'r*', lw = 3, label="Gas-num")
axs[1].plot(times, norm_vd_re, 'b', lw = 1, label="Dust-ana" )
axs[1].plot(times, norm_vg_re, 'r', lw = 1, label="Gas-ana")
axs[1].set_xlabel('Time', fontsize=15,fontweight='bold')
axs[1].set_ylabel('Normalized Velocity', fontsize=15, fontweight='bold')
plt.legend(prop={'weight' : 'bold'})
plt.savefig("dusty_wave_test_2fluids.png")"""


## 5 fluids
fig, axs = plt.subplots(1, 2, figsize=(25, 10))
plt.subplots_adjust(wspace=0.25)
axs[0].plot(times, normalized_rd_num[0], "bo", lw=3, label="Dust1-num")
axs[0].plot(times, normalized_rd_num[1], "ro", lw=3, label="Dust2-num")
axs[0].plot(times, normalized_rd_num[2], "go", lw=3, label="Dust3-num")
axs[0].plot(times, normalized_rd_num[3], "co", lw=3, label="Dust4-num")
axs[0].plot(times, normalized_rg_num, "m*", lw=3, label="Gas-num")
axs[0].plot(times, norm_rd_1_re, "k", lw=1, label="Dust1-ana")
axs[0].plot(times, norm_rd_2_re, "k", lw=1, label="Dust2-ana")
axs[0].plot(times, norm_rd_3_re, "k", lw=1, label="Dust3-ana")
axs[0].plot(times, norm_rd_4_re, "k", lw=1, label="Dust4-ana")
axs[0].plot(times, norm_rg_re, "k", lw=1, label="Gas-ana")
axs[0].set_xlabel("Time", fontsize=15, fontweight="bold")
axs[0].set_ylabel("Normalized Density", fontsize=15, fontweight="bold")

axs[1].plot(times, normalized_vd_num[0], "bo", lw=3, label="Dust1-num")
axs[1].plot(times, normalized_vd_num[1], "ro", lw=3, label="Dust2-num")
axs[1].plot(times, normalized_vd_num[2], "go", lw=3, label="Dust3-num")
axs[1].plot(times, normalized_vd_num[3], "co", lw=3, label="Dust4-num")
axs[1].plot(times, normalized_vg_num, "m*", lw=3, label="Gas-num")
axs[1].plot(times, norm_vd_1_re, "k", lw=1, label="Dust1-ana")
axs[1].plot(times, norm_vd_2_re, "k", lw=1, label="Dust2-ana")
axs[1].plot(times, norm_vd_3_re, "k", lw=1, label="Dust3-ana")
axs[1].plot(times, norm_vd_4_re, "k", lw=1, label="Dust4-ana")
axs[1].plot(times, norm_vg_re, "k", lw=1, label="Gas-ana")
axs[1].set_xlabel("Time", fontsize=15, fontweight="bold")
axs[1].set_ylabel("Normalized Velocity", fontsize=15, fontweight="bold")

plt.legend(prop={"weight": "bold"})
plt.savefig("dusty_wave_test_5fluids_new.png")
