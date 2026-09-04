"""
Testing dusty wave with Godunov
==============================

CI test for dusty wave with Godunov
"""

from math import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import shamrock


def run_sim(times, x0, normalized_rd_num, normalized_rg_num, normalized_vd_num, normalized_vg_num):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 16

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    cfg.set_Csafe(0.5)
    cfg.set_eos_gamma(1.0000001)
    cfg.set_dust_mode_hb(4)
    cfg.set_drag_mode_irk1(True)
    cfg.set_face_time_interpolation(False)

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

    def perturbation(x, A, Re, Im, L):
        return A * (Re * cos(2 * x * pi / L) - Im * sin(2 * x * pi / L))

    """   ##  2 fluids test setup
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
        return rhog_0 + perturbation(x,A_rho,Re_rho,Im_rho,L)
    def rhovel_map(rmin, rmax)->tuple[float,float,float]:
        x,y,z = rmin
        rho = rhog_0 + perturbation(x,A_rho,Re_rho,Im_rho,L)
        vx  = perturbation(x,A_vel,Re_vel,Im_vel,L)
        return (rho*vx, 0, 0)
    def rhoe_map (rmin, rmax)->float:
        x,y,z   = rmin
        rho     = rhog_0 + perturbation(x,A_rho,Re_rho,Im_rho,L)
        vx      = perturbation(x,A_vel,Re_vel,Im_vel,L)
        press   = (cs * cs * rho) / gamma
        rhoeint = press / (gamma - 1.0)
        rhoekin = 0.5 * rho * (vx *vx + 0.0)
        return (rhoeint + rhoekin)
    ### Dust maps
    def rho_d_map(rmin,rmax)->float:
        x,y,z = rmin
        return rhod_0 + perturbation(x,A_rho,Re_rd,Im_rd,L)
    def rhovel_d_map(rmin, rmax)->tuple[float,float,float]:
        x,y,z = rmin
        rho = rhod_0 + perturbation(x,A_rho,Re_rd,Im_rd,L)
        vx  = perturbation(x,A_vel,Re_vd,Im_vd,L)
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
    def rho_map(rmin, rmax):
        x, y, z = rmin
        return rhog_0 + perturbation(x, A_rho, Re_rho, Im_rho, L)

    def rhovel_map(rmin, rmax):
        x, y, z = rmin
        rho = rhog_0 + perturbation(x, A_rho, Re_rho, Im_rho, L)
        vx = perturbation(x, A_vel, Re_vel, Im_vel, L)
        return (rho * vx, 0, 0)

    def rhoe_map(rmin, rmax):
        x, y, z = rmin
        rho = rhog_0 + perturbation(x, A_rho, Re_rho, Im_rho, L)
        vx = perturbation(x, A_vel, Re_vel, Im_vel, L)
        press = (cs * cs * rho) / gamma
        rhoeint = press / (gamma - 1.0)
        rhoekin = 0.5 * rho * (vx * vx + 0.0)
        return rhoeint + rhoekin

    ### Dusts maps

    def rho_d_1_map(rmin, rmax):
        x, y, z = rmin
        return rhod_1 + perturbation(x, A_rho, Re_rd_1, Im_rd_1, L)

    def rhovel_d_1_map(rmin, rmax):
        x, y, z = rmin
        rho = rhod_1 + perturbation(x, A_rho, Re_rd_1, Im_rd_1, L)
        vx = perturbation(x, A_vel, Re_vd_1, Im_vd_1, L)
        return (rho * vx, 0, 0)

    def rho_d_2_map(rmin, rmax):
        x, y, z = rmin
        return rhod_2 + perturbation(x, A_rho, Re_rd_2, Im_rd_2, L)

    def rhovel_d_2_map(rmin, rmax):
        x, y, z = rmin
        rho = rhod_2 + perturbation(x, A_rho, Re_rd_2, Im_rd_2, L)
        vx = perturbation(x, A_vel, Re_vd_2, Im_vd_2, L)
        return (rho * vx, 0, 0)

    def rho_d_3_map(rmin, rmax):
        x, y, z = rmin
        return rhod_3 + perturbation(x, A_rho, Re_rd_3, Im_rd_3, L)

    def rhovel_d_3_map(rmin, rmax):
        x, y, z = rmin
        rho = rhod_3 + perturbation(x, A_rho, Re_rd_3, Im_rd_3, L)
        vx = perturbation(x, A_vel, Re_vd_3, Im_vd_3, L)
        return (rho * vx, 0, 0)

    def rho_d_4_map(rmin, rmax):
        x, y, z = rmin
        return rhod_4 + perturbation(x, A_rho, Re_rd_4, Im_rd_4, L)

    def rhovel_d_4_map(rmin, rmax):
        x, y, z = rmin
        rho = rhod_4 + perturbation(x, A_rho, Re_rd_4, Im_rd_4, L)
        vx = perturbation(x, A_vel, Re_vd_4, Im_vd_4, L)
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
            dic_i = convert_to_cell_coords(ctx.collect_data())

            vg_i = dic_i["rhovel"][0][0] / dic_i["rho"][0]
            rg_i = dic_i["rho"][0]
            rd_i = dic_i["rho_dust"][0]
            vd_i = dic_i["rhovel_dust"][0][0] / dic_i["rho_dust"][0]
            x0 = dic_i["xmin"][0]
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

# =============== plots ==================
"""## 2 fluids

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
if False:
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
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
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Normalized Density")

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
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Normalized Velocity")

    axs[0].legend()
    axs[1].legend()

    plt.savefig("dusty_wave_test_5fluids_new.png")

print(f"rdnum0 = {normalized_rd_num[0]}")
print(f"rdnum1 = {normalized_rd_num[1]}")
print(f"rdnum2 = {normalized_rd_num[2]}")
print(f"rdnum3 = {normalized_rd_num[3]}")
print(f"rgnum = {normalized_rg_num}")
print(f"vdnum0 = {normalized_vd_num[0]}")
print(f"vdnum1 = {normalized_vd_num[1]}")
print(f"vdnum2 = {normalized_vd_num[2]}")
print(f"vdnum3 = {normalized_vd_num[3]}")
print(f"vgnum = {normalized_vg_num}")


rdnum0_ref = [
    (0.805880000000203),
    (0.8689243084003538),
    (0.7900129335228211),
    (0.579842660818064),
    (0.29382255223864284),
    (-0.009579635265599684),
    (-0.2699129708974568),
    (-0.44097593524111683),
    (-0.5233132377824301),
    (-0.5165562695155779),
    (-0.3957957995251448),
    (-0.22238856309669816),
    (-0.02839787172775887),
    (0.1400617212848898),
    (0.2618040653878939),
    (0.33209865967220864),
    (0.34296700406055697),
    (0.28252577559534364),
    (0.1790292769601742),
    (0.06110935456549171),
    (-0.04812964529282703),
    (-0.13405926669318724),
    (-0.1878277827480068),
    (-0.2071717358359537),
    (-0.18043883381196887),
    (-0.12034051173892556),
]
rdnum1_ref = [
    (0.39257198938890125),
    (0.5542035334949146),
    (0.6031567492567251),
    (0.5482164911435082),
    (0.40222603606897905),
    (0.202507489777572),
    (-0.003854689335971584),
    (-0.17772545608071375),
    (-0.30161100033309823),
    (-0.3585764603122679),
    (-0.34146345457712785),
    (-0.259003282436169),
    (-0.13790812223515236),
    (-0.008080715656867257),
    (0.10729383764142078),
    (0.1956018641605447),
    (0.24939622662494695),
    (0.245120314988042),
    (0.20208434364850558),
    (0.13229918411272942),
    (0.05244194715942338),
    (-0.023243113507741813),
    (-0.08439859109483745),
    (-0.1244774019744991),
    (-0.13269876990324073),
    (-0.1113807124659012),
]
rdnum2_ref = [
    (0.08434628695750471),
    (0.2106234613201813),
    (0.2923363573167843),
    (0.30864096360977344),
    (0.27245075904708016),
    (0.19074432915103703),
    (0.08645304943399448),
    (-0.01637639475285132),
    (-0.10489780718285861),
    (-0.16585631000055187),
    (-0.18501291152811425),
    (-0.1694332337064602),
    (-0.12155929928720144),
    (-0.058182530789561544),
    (0.008476227016025293),
    (0.06856968168158446),
    (0.11502438561047071),
    (0.13455897109281761),
    (0.12906816952956998),
    (0.10474245977410819),
    (0.06807939951884429),
    (0.02678667992286499),
    (-0.012378468438113136),
    (-0.044354519793374216),
    (-0.06084145813054332),
    (-0.06208091553502869),
]
rdnum3_ref = [
    (0.0029020000003043833),
    (0.06956148160597309),
    (0.11948308370257621),
    (0.14159549809544814),
    (0.13520241968878466),
    (0.10570784073848927),
    (0.06114704406590121),
    (0.01264495409492028),
    (-0.03277416772529129),
    (-0.06816840717571715),
    (-0.08535165213263696),
    (-0.0856839191698544),
    (-0.06862646604433031),
    (-0.04194936800927529),
    (-0.010991112641134393),
    (0.019496359267012764),
    (0.04541139830394059),
    (0.060006677793378316),
    (0.06237712183887467),
    (0.05545930156891643),
    (0.041067031470998216),
    (0.022670521220291562),
    (0.003700465589506763),
    (-0.013302308820017927),
    (-0.02398277228921053),
    (-0.02718012055091812),
]
rgnum_ref = [
    (0.9999999999998899),
    (0.8688349495966641),
    (0.5910913966999942),
    (0.24642370436867012),
    (-0.10192983860179972),
    (-0.39677041998587015),
    (-0.5616261642304998),
    (-0.6124042747168712),
    (-0.5885544361916573),
    (-0.4245818736348106),
    (-0.2131800131599526),
    (0.01727539234330777),
    (0.2046455065318753),
    (0.3241370146622735),
    (0.37981812422982486),
    (0.3829670523258777),
    (0.3063493229782388),
    (0.16514219502106187),
    (0.0255392870274207),
    (-0.09547997896319771),
    (-0.18404092464896493),
    (-0.23295745767759612),
    (-0.24418904712386613),
    (-0.21372740393488243),
    (-0.12348088156310943),
    (-0.040919532890981714),
]
vdnum0_ref = [
    (-0.77538),
    (-0.7811863832371356),
    (-0.655634806352653),
    (-0.43169254399674284),
    (-0.15922820267233143),
    (0.11119734803476938),
    (0.32465204189803887),
    (0.4472854797978536),
    (0.4878569679290291),
    (0.44229533435117224),
    (0.31363825298352144),
    (0.14441002689956045),
    (-0.029593491257600843),
    (-0.17025129899839478),
    (-0.26183545779520245),
    (-0.3030856597168068),
    (-0.29443245106849475),
    (-0.22154787440916435),
    (-0.11833662986689451),
    (-0.01014938478558113),
    (0.0833139693250616),
    (0.15054830694123533),
    (0.18632856107205514),
    (0.19103403522975843),
    (0.15275810096488854),
    (0.09193118808117595),
]
vdnum1_ref = [
    (-0.427268),
    (-0.5309603274304748),
    (-0.5382456189794439),
    (-0.45029078119101523),
    (-0.2937570630108705),
    (-0.1027932432118918),
    (0.08122475595458416),
    (0.22378911020494335),
    (0.314573041851117),
    (0.3428856624396502),
    (0.30212121403359005),
    (0.21202510111100373),
    (0.09484809107731261),
    (-0.02128165101749134),
    (-0.11765423361006018),
    (-0.1846028142444747),
    (-0.21630476462378567),
    (-0.20001699393340805),
    (-0.14943812056695036),
    (-0.07993746406694753),
    (-0.00676383818367264),
    (0.05807764256928899),
    (0.1061956413713949),
    (0.1333782902593811),
    (0.13002136115021),
    (0.10324977747134896),
]
vdnum2_ref = [
    (-0.127928),
    (-0.22433822837479792),
    (-0.27602066368693423),
    (-0.27220579966736363),
    (-0.22027553130274014),
    (-0.13444010386451855),
    (-0.03563601519086175),
    (0.055330261638462566),
    (0.1278913971933085),
    (0.171607716618186),
    (0.17762726797509779),
    (0.15120346580553842),
    (0.10097293974503266),
    (0.04079487861874067),
    (-0.018036818208876402),
    (-0.06770663216025845),
    (-0.10216645884513763),
    (-0.11175538977702502),
    (-0.0997782867939847),
    (-0.07210466726083371),
    (-0.03602133578673551),
    (0.0016061683967248394),
    (0.03494913627324335),
    (0.05993591866377568),
    (0.06975885269913283),
    (0.06552765574853019),
]
vdnum3_ref = [
    (-0.028963),
    (-0.08274555765707187),
    (-0.11865660363392633),
    (-0.1287763663115185),
    (-0.11443714294025117),
    (-0.0807517851227285),
    (-0.03677593860760486),
    (0.007553788906515882),
    (0.046460171634521506),
    (0.07393195401516559),
    (0.08429278412529245),
    (0.07855092055571575),
    (0.05964847759717853),
    (0.03352120599673748),
    (0.005463917800689098),
    (-0.020469258259941704),
    (-0.04082120950356733),
    (-0.0501601208683192),
    (-0.04902162281095555),
    (-0.039421490555584554),
    (-0.02435697477446975),
    (-0.006944696795611556),
    (0.009917329517998728),
    (0.023968809216562907),
    (0.031493696169100985),
    (0.032345825856686224),
]
vgnum_ref = [
    (-0.874365),
    (-0.7099220326528427),
    (-0.4261561487342454),
    (-0.10465046845077873),
    (0.1984771258117866),
    (0.4358660029177983),
    (0.5451490601695725),
    (0.5515280703079309),
    (0.4980829289275977),
    (0.3209684559041537),
    (0.11971531599291403),
    (-0.08504474565534141),
    (-0.23807427811275042),
    (-0.32250378785552075),
    (-0.34814620922752715),
    (-0.3273923454140501),
    (-0.24192131786791657),
    (-0.10359979718599996),
    (0.021859623163015863),
    (0.1226700469403289),
    (0.18918124142234669),
    (0.21809383370794294),
    (0.2136510930151699),
    (0.1738734039625043),
    (0.08392518783273606),
    (0.008437875034811467),
]

rd0_diff = [abs(normalized_rd_num[0][i] - rdnum0_ref[i]) for i in range(len(normalized_rd_num[0]))]
rd1_diff = [abs(normalized_rd_num[1][i] - rdnum1_ref[i]) for i in range(len(normalized_rd_num[1]))]
rd2_diff = [abs(normalized_rd_num[2][i] - rdnum2_ref[i]) for i in range(len(normalized_rd_num[2]))]
rd3_diff = [abs(normalized_rd_num[3][i] - rdnum3_ref[i]) for i in range(len(normalized_rd_num[3]))]
rg_diff = [abs(normalized_rg_num[i] - rgnum_ref[i]) for i in range(len(normalized_rg_num))]
vd0_diff = [abs(normalized_vd_num[0][i] - vdnum0_ref[i]) for i in range(len(normalized_vd_num[0]))]
vd1_diff = [abs(normalized_vd_num[1][i] - vdnum1_ref[i]) for i in range(len(normalized_vd_num[1]))]
vd2_diff = [abs(normalized_vd_num[2][i] - vdnum2_ref[i]) for i in range(len(normalized_vd_num[2]))]
vd3_diff = [abs(normalized_vd_num[3][i] - vdnum3_ref[i]) for i in range(len(normalized_vd_num[3]))]
vg_diff = [abs(normalized_vg_num[i] - vgnum_ref[i]) for i in range(len(normalized_vg_num))]

print(f"rd0_diff = {rd0_diff} with len = {len(rd0_diff)} \n")
print(f"rd1_diff = {rd1_diff} with len = {len(rd1_diff)} \n")
print(f"rd2_diff = {rd2_diff} with len = {len(rd2_diff)} \n")
print(f"rd3_diff = {rd3_diff} with len = {len(rd3_diff)} \n")
print(f"rg_diff = {rg_diff} with len = {len(rg_diff)} \n")
print(f"vd0_diff = {vd0_diff} with len = {len(vd0_diff)} \n")
print(f"vd1_diff = {vd1_diff} with len = {len(vd1_diff)} \n")
print(f"vd2_diff = {vd2_diff} with len = {len(vd2_diff)} \n")
print(f"vd3_diff = {vd3_diff} with len = {len(vd3_diff)} \n")
print(f"vg_diff = {vg_diff} with len = {len(vg_diff)} \n")


"""
CI results:

rd0_diff = 9.71445146547012e-12 > 1e-12
rd1_diff = 5.947632143732395e-12 > 1e-12
rd2_diff = 3.027880030037622e-12 > 1e-12
rd3_diff = 4.440892098500626e-12 > 1e-12
rg_diff = 1.1102230246251565e-11 > 1e-12
vd0_diff = 3.4523772729500024e-12 > 1e-12
vd1_diff = 2.9127256162553294e-12 > 1e-12
vd2_diff = 2.0037860259947138e-12 > 1e-12
vd3_diff = 1.157754447866921e-12 > 1e-12
vg_diff = 6.4616922923477205e-12 > 1e-12

"""


test_pass = True
rd0_max_pass = 1e-11
rd1_max_pass = 1e-11
rd2_max_pass = 1e-11
rd3_max_pass = 1e-11
rg_max_pass = 1e-10
vd0_max_pass = 1e-11
vd1_max_pass = 1e-11
vd2_max_pass = 1e-11
vd3_max_pass = 1e-11
vg_max_pass = 1e-11

err_log = ""
if np.max(rd0_diff) > rd0_max_pass:
    err_log += f"rd0_diff = {np.max(rd0_diff)} > {rd0_max_pass} \n"
    test_pass = False

if np.max(rd1_diff) > rd1_max_pass:
    err_log += f"rd1_diff = {np.max(rd1_diff)} > {rd1_max_pass} \n"
    test_pass = False

if np.max(rd2_diff) > rd2_max_pass:
    err_log += f"rd2_diff = {np.max(rd2_diff)} > {rd2_max_pass} \n"
    test_pass = False

if np.max(rd3_diff) > rd3_max_pass:
    err_log += f"rd3_diff = {np.max(rd3_diff)} > {rd3_max_pass} \n"
    test_pass = False

if np.max(rg_diff) > rg_max_pass:
    err_log += f"rg_diff = {np.max(rg_diff)} > {rg_max_pass} \n"
    test_pass = False

if np.max(vd0_diff) > vd0_max_pass:
    err_log += f"vd0_diff = {np.max(vd0_diff)} > {vd0_max_pass} \n"
    test_pass = False

if np.max(vd1_diff) > vd1_max_pass:
    err_log += f"vd1_diff = {np.max(vd1_diff)} > {vd1_max_pass} \n"
    test_pass = False

if np.max(vd2_diff) > vd2_max_pass:
    err_log += f"vd2_diff = {np.max(vd2_diff)} > {vd2_max_pass} \n"
    test_pass = False

if np.max(vd3_diff) > vd3_max_pass:
    err_log += f"vd3_diff = {np.max(vd3_diff)} > {vd3_max_pass} \n"
    test_pass = False

if np.max(vg_diff) > vg_max_pass:
    err_log += f"vg_diff = {np.max(vg_diff)} > {vg_max_pass} \n"
    test_pass = False

if test_pass == False:
    exit("Test did not pass L2 margins : \n" + err_log)
