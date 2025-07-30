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

    def pertubation(x, A, Re, Im, L):
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
    def rho_map(rmin, rmax):
        x, y, z = rmin
        return rhog_0 + pertubation(x, A_rho, Re_rho, Im_rho, L)

    def rhovel_map(rmin, rmax):
        x, y, z = rmin
        rho = rhog_0 + pertubation(x, A_rho, Re_rho, Im_rho, L)
        vx = pertubation(x, A_vel, Re_vel, Im_vel, L)
        return (rho * vx, 0, 0)

    def rhoe_map(rmin, rmax):
        x, y, z = rmin
        rho = rhog_0 + pertubation(x, A_rho, Re_rho, Im_rho, L)
        vx = pertubation(x, A_vel, Re_vel, Im_vel, L)
        press = (cs * cs * rho) / gamma
        rhoeint = press / (gamma - 1.0)
        rhoekin = 0.5 * rho * (vx * vx + 0.0)
        return rhoeint + rhoekin

    ### Dusts maps

    def rho_d_1_map(rmin, rmax):
        x, y, z = rmin
        return rhod_1 + pertubation(x, A_rho, Re_rd_1, Im_rd_1, L)

    def rhovel_d_1_map(rmin, rmax):
        x, y, z = rmin
        rho = rhod_1 + pertubation(x, A_rho, Re_rd_1, Im_rd_1, L)
        vx = pertubation(x, A_vel, Re_vd_1, Im_vd_1, L)
        return (rho * vx, 0, 0)

    def rho_d_2_map(rmin, rmax):
        x, y, z = rmin
        return rhod_2 + pertubation(x, A_rho, Re_rd_2, Im_rd_2, L)

    def rhovel_d_2_map(rmin, rmax):
        x, y, z = rmin
        rho = rhod_2 + pertubation(x, A_rho, Re_rd_2, Im_rd_2, L)
        vx = pertubation(x, A_vel, Re_vd_2, Im_vd_2, L)
        return (rho * vx, 0, 0)

    def rho_d_3_map(rmin, rmax):
        x, y, z = rmin
        return rhod_3 + pertubation(x, A_rho, Re_rd_3, Im_rd_3, L)

    def rhovel_d_3_map(rmin, rmax):
        x, y, z = rmin
        rho = rhod_3 + pertubation(x, A_rho, Re_rd_3, Im_rd_3, L)
        vx = pertubation(x, A_vel, Re_vd_3, Im_vd_3, L)
        return (rho * vx, 0, 0)

    def rho_d_4_map(rmin, rmax):
        x, y, z = rmin
        return rhod_4 + pertubation(x, A_rho, Re_rd_4, Im_rd_4, L)

    def rhovel_d_4_map(rmin, rmax):
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
    (0.8689243077925067),
    (0.7900129344901029),
    (0.5798426664191392),
    (0.2938225564075303),
    (-0.009579637037793187),
    (-0.26991296780826124),
    (-0.4409759276943758),
    (-0.5233132250329064),
    (-0.5165562572517768),
    (-0.395795789913389),
    (-0.2223885476271281),
    (-0.028397848157724056),
    (0.14006175439035262),
    (0.2618041011495653),
    (0.332098698928307),
    (0.34296704644193315),
    (0.2825258215571891),
    (0.1790293225792383),
    (0.061109398673264696),
    (-0.04812960177763558),
    (-0.13405922108661317),
    (-0.18782773463371646),
    (-0.2071716863658035),
    (-0.18043878479007122),
    (-0.12034046096565108),
]
rdnum1_ref = [
    (0.39257198938890125),
    (0.5542035347986355),
    (0.6031567548213295),
    (0.5482164998591679),
    (0.4022260439388855),
    (0.2025074960796827),
    (-0.003854689554844436),
    (-0.17772546738121425),
    (-0.3016110220931041),
    (-0.35857649286484655),
    (-0.3414634939575878),
    (-0.25900332734197895),
    (-0.13790817100097535),
    (-0.008080767335840339),
    (0.10729378337641737),
    (0.19560180599151578),
    (0.24939616919818247),
    (0.24512025887094602),
    (0.20208428768247946),
    (0.13229913026881834),
    (0.05244189399592139),
    (-0.023243167866717804),
    (-0.08439864242052121),
    (-0.12447745395442235),
    (-0.13269882229236105),
    (-0.11138076254971943),
]
rdnum2_ref = [
    (0.08434628695750471),
    (0.21062346808748914),
    (0.2923363733569692),
    (0.30864098822793734),
    (0.2724507899268962),
    (0.19074436436526077),
    (0.08645309262667734),
    (-0.01637634427660739),
    (-0.10489775335475349),
    (-0.16585624855733302),
    (-0.18501284828027997),
    (-0.16943316656628848),
    (-0.1215592315308565),
    (-0.058182463429868644),
    (0.00847630200294344),
    (0.06856975476699509),
    (0.11502446471833112),
    (0.13455905428225787),
    (0.12906825376968398),
    (0.10474254094395367),
    (0.06807948226772825),
    (0.026786767934201317),
    (-0.01237838059633799),
    (-0.0443544400890218),
    (-0.060841380837895906),
    (-0.06208083850580668),
]
rdnum3_ref = [
    (0.0029020000003043833),
    (0.06956148347114777),
    (0.11948309374787414),
    (0.14159551296577533),
    (0.13520243927978015),
    (0.10570786835417678),
    (0.061147073666667495),
    (0.012644984879184307),
    (-0.03277412613522657),
    (-0.06816835842027302),
    (-0.08535160297640232),
    (-0.08568386849483467),
    (-0.06862641456328866),
    (-0.041949313090983154),
    (-0.010991046288655326),
    (0.019496433840693328),
    (0.045411472970879885),
    (0.06000675009110168),
    (0.06237719263779695),
    (0.055459369678878545),
    (0.041067093610180905),
    (0.022670576287353583),
    (0.0037005204323037333),
    (-0.01330224985274242),
    (-0.023982707549885518),
    (-0.027180050371500286),
]
rgnum_ref = [
    (0.9999999999998899),
    (0.8688349497520953),
    (0.5910913976636678),
    (0.24642370692662396),
    (-0.10192983466050798),
    (-0.39677041619112785),
    (-0.561626161229567),
    (-0.6124042725308421),
    (-0.5885544339423454),
    (-0.4245818746517749),
    (-0.21318001503400907),
    (0.01727538923690375),
    (0.20464550396281922),
    (0.3241370138651334),
    (0.37981812456289177),
    (0.3829670537625063),
    (0.306349327250377),
    (0.16514219895125137),
    (0.025539290697818018),
    (-0.0954799764940617),
    (-0.18404092327450883),
    (-0.23295745713025617),
    (-0.24418904680079123),
    (-0.21372740465763762),
    (-0.12348088227143172),
    (-0.04091953305862539),
]
vdnum0_ref = [
    (-0.77538),
    (-0.7811863758910647),
    (-0.6556347650488735),
    (-0.4316925391136899),
    (-0.15922820758957576),
    (0.11119734045036302),
    (0.3246520280748058),
    (0.4472854623218433),
    (0.48785697131173245),
    (0.442295322836427),
    (0.3136382584280283),
    (0.14441004037288763),
    (-0.02959348025296491),
    (-0.17025129133723313),
    (-0.26183545523685964),
    (-0.303085651136941),
    (-0.2944324560056733),
    (-0.22154786434163998),
    (-0.11833663676346025),
    (-0.010149388077418078),
    (0.0833139633620755),
    (0.15054829766921585),
    (0.186328556406826),
    (0.19103403517545617),
    (0.15275810723488148),
    (0.09193119286409025),
]
vdnum1_ref = [
    (-0.427268),
    (-0.5309603454208828),
    (-0.5382456751876933),
    (-0.45029083801954034),
    (-0.29375712072072363),
    (-0.102793273979704),
    (0.08122473286418207),
    (0.2237890933638007),
    (0.31457303920617646),
    (0.34288566207080834),
    (0.30212119329923925),
    (0.2120250892944267),
    (0.09484808545876959),
    (-0.02128164859184025),
    (-0.11765422929628072),
    (-0.18460283281708043),
    (-0.216304773382997),
    (-0.20001698763524023),
    (-0.1494381161874679),
    (-0.07993747333413012),
    (-0.006763840063493263),
    (0.05807764072894879),
    (0.10619563228299671),
    (0.13337829216323221),
    (0.13002135329619113),
    (0.10324978155771455),
]
vdnum2_ref = [
    (-0.127928),
    (-0.22433822994326136),
    (-0.27602066610296094),
    (-0.2722057937325647),
    (-0.2202755345324273),
    (-0.1344401094732441),
    (-0.035636026605325694),
    (0.05533025280153579),
    (0.1278913854203787),
    (0.17160772283680298),
    (0.17762725605765986),
    (0.15120348961722455),
    (0.10097293229240813),
    (0.04079488263339769),
    (-0.018036812005879706),
    (-0.06770662850093334),
    (-0.10216646251901662),
    (-0.1117553730186458),
    (-0.09977828418787547),
    (-0.07210467925796142),
    (-0.03602133435843739),
    (0.0016061697720409604),
    (0.03494914200641798),
    (0.059935919787940335),
    (0.06975884068164784),
    (0.06552764505311985),
]
vdnum3_ref = [
    (-0.028963),
    (-0.08274556735514085),
    (-0.11865659880671561),
    (-0.12877637836886333),
    (-0.11443713845813151),
    (-0.08075177828758495),
    (-0.036775947979712675),
    (0.007553784808566575),
    (0.04646016756736207),
    (0.0739319638991898),
    (0.08429279118423574),
    (0.07855091155478137),
    (0.05964847012637353),
    (0.033521208058918936),
    (0.0054639253661799714),
    (-0.02046924840749944),
    (-0.040821194729435316),
    (-0.05016010622677305),
    (-0.049021615176800745),
    (-0.03942149457236898),
    (-0.024356990241931917),
    (-0.006944707917118663),
    (0.009917318467300065),
    (0.023968795687178763),
    (0.031493681350258824),
    (0.03234581246322336),
]
vgnum_ref = [
    (-0.874365),
    (-0.709922032828806),
    (-0.426156149324301),
    (-0.10465047095432616),
    (0.198477120454305),
    (0.4358659968677081),
    (0.5451490549239694),
    (0.5515280691563401),
    (0.4980829279270518),
    (0.32096845998912177),
    (0.11971532182104656),
    (-0.0850447382848111),
    (-0.238074271749472),
    (-0.32250378341554387),
    (-0.3481462054848407),
    (-0.3273923438079883),
    (-0.24192132112753134),
    (-0.10359979997242741),
    (0.02185961990209998),
    (0.12267004267841061),
    (0.18918123690563873),
    (0.21809383083453066),
    (0.21365109098794338),
    (0.17387340496791126),
    (0.0839251890575102),
    (0.008437877113669988),
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

rd1_diff = 1.6653373124952964e-11 > 9.526221144423298e-12
rd2_diff = 7.569696605647103e-12 > 4.551825249226861e-12
rg_diff = 1.1102230246251565e-11 > 6.671338147750939e-12
vd0_diff = 3.779143664672802e-12 > 2.8379045771739676e-12
vd1_diff = 2.7466917629226373e-12 > 1.9998527270356682e-12
vd2_diff = 1.936215077158465e-12 > 1.0505010186786967e-12
vd3_diff = 1.220287759728933e-12 > 1e-12
vg_diff = 7.118580030995858e-12 > 5.104230592916915e-12

"""


test_pass = True
rd0_max_pass = 1.5265566588595902e-11 + 1e-14
rd1_max_pass = 1.6653373124952964e-11 + 1e-14
rd2_max_pass = 7.569696605647103e-12 + 1e-14
rd3_max_pass = 1.1102230246251565e-11 + 1e-14
rg_max_pass = 1.1102230246251565e-11 + 1e-14
vd0_max_pass = 3.779143664672802e-12 + 1e-14
vd1_max_pass = 2.7466917629226373e-12 + 1e-14
vd2_max_pass = 1.936215077158465e-12 + 1e-14
vd3_max_pass = 1.220287759728933e-12 + 1e-14
vg_max_pass = 7.118580030995858e-12 + 1e-14

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
