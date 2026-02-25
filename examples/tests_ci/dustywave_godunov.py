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
    (0.8689243084003538),
    (0.7900129335228211),
    (0.5798426608166762),
    (0.2938225522344795),
    (-0.0095796352711508),
    (-0.2699129709030079),
    (-0.44097593524666795),
    (-0.523313237789369),
    (-0.5165562695239045),
    (-0.3957957995390226),
    (-0.22238856311196373),
    (-0.028397871740248878),
    (0.1400617212723998),
    (0.2618040653740161),
    (0.3320986596555553),
    (0.3429670040452914),
    (0.28252577558285363),
    (0.1790292769476842),
    (0.0611093545530017),
    (-0.04812964530531704),
    (-0.1340592667029017),
    (-0.18782778275910902),
    (-0.2071717358484437),
    (-0.18043883382723444),
    (-0.12034051175280334),
]
rdnum1_ref = [
    (0.39257198938890125),
    (0.5542035334949146),
    (0.6031567492591042),
    (0.5482164911435082),
    (0.4022260360677895),
    (0.20250748977638247),
    (-0.0038546893395401635),
    (-0.17772545608547186),
    (-0.30161100033904586),
    (-0.35857646032059465),
    (-0.34146345458664407),
    (-0.2590032824468747),
    (-0.13790812224466856),
    (-0.008080715668762522),
    (0.10729383762714646),
    (0.1956018641415123),
    (0.24939622660472502),
    (0.24512031496782005),
    (0.20208434362590458),
    (0.1322991840889389),
    (0.052441947135632855),
    (-0.023243113530342814),
    (-0.08439859111743846),
    (-0.1244774019971001),
    (-0.13269876992703125),
    (-0.11138071248969174),
]
rdnum2_ref = [
    (0.08434628695750471),
    (0.2106234613201813),
    (0.29233635731829827),
    (0.30864096360977344),
    (0.27245075904708016),
    (0.19074432915103703),
    (0.08645304943248056),
    (-0.0163763947558792),
    (-0.10489780718740042),
    (-0.16585631000509368),
    (-0.18501291153417),
    (-0.16943323371251598),
    (-0.1215592992932572),
    (-0.05818253079713124),
    (0.008476227008455597),
    (0.06856968167401477),
    (0.11502438559835919),
    (0.1345589710807061),
    (0.12906816951745848),
    (0.10474245976199668),
    (0.06807939950673277),
    (0.026786679910753477),
    (-0.01237846845022465),
    (-0.044354519808513605),
    (-0.06084145814568271),
    (-0.06208091555016808),
]
rdnum3_ref = [
    (0.0029020000003043833),
    (0.06956148160597309),
    (0.11948308370257621),
    (0.14159549809544814),
    (0.13520241968878466),
    (0.10570784073626882),
    (0.06114704406146032),
    (0.012644954090479388),
    (-0.032774167729732184),
    (-0.06816840718126826),
    (-0.08535165213818807),
    (-0.08568391917540552),
    (-0.06862646604988143),
    (-0.041949368014826405),
    (-0.010991112647795731),
    (0.019496359253690088),
    (0.045411398288397464),
    (0.06000667777783519),
    (0.06237712182333155),
    (0.05545930155337331),
    (0.041067031455455094),
    (0.02267052120474844),
    (0.0037004655739636405),
    (-0.01330230883556105),
    (-0.023982772304753652),
    (-0.02718012056646124),
]
rgnum_ref = [
    (0.9999999999998899),
    (0.8688349495944436),
    (0.5910913966955533),
    (0.24642370436422922),
    (-0.10192983860846105),
    (-0.39677041999586216),
    (-0.5616261642438225),
    (-0.6124042747268632),
    (-0.5885544362038697),
    (-0.4245818736492435),
    (-0.21318001317105484),
    (0.01727539233220554),
    (0.20464550652965485),
    (0.3241370146556122),
    (0.3798181242165022),
    (0.382967052312555),
    (0.30634932297157746),
    (0.16514219501662097),
    (0.02553928702075936),
    (-0.09547997896985905),
    (-0.18404092465562627),
    (-0.2329574576809268),
    (-0.24418904713052747),
    (-0.21372740394154377),
    (-0.1234808815664401),
    (-0.04091953289209194),
]
vdnum0_ref = [
    (-0.77538),
    (-0.781186383236733),
    (-0.6556348063526707),
    (-0.431692543997261),
    (-0.15922820267203422),
    (0.11119734803542475),
    (0.32465204189864033),
    (0.4472854797974426),
    (0.48785696792567634),
    (0.4422953343494822),
    (0.31363825298202574),
    (0.1444100268983182),
    (-0.029593491256808366),
    (-0.1702512989966378),
    (-0.26183545779312434),
    (-0.3030856597147541),
    (-0.29443245106934907),
    (-0.22154787441090817),
    (-0.11833662986595124),
    (-0.010149384784945393),
    (0.08331396932387314),
    (0.15054830693968646),
    (0.18632856107039308),
    (0.1910340352281365),
    (0.15275810096450626),
    (0.09193118807910677),
]
vdnum1_ref = [
    (-0.427268),
    (-0.5309603274302681),
    (-0.5382456189793695),
    (-0.45029078119126137),
    (-0.2937570630107788),
    (-0.10279324321150173),
    (0.08122475595499018),
    (0.22378911020485165),
    (0.3145730418492861),
    (0.3428856624382193),
    (0.3021212140322564),
    (0.2120251011096913),
    (0.09484809107712948),
    (-0.021281651016827028),
    (-0.11765423360886178),
    (-0.18460281424303243),
    (-0.2163047646237174),
    (-0.20001699393414982),
    (-0.14943812056652284),
    (-0.07993746406650025),
    (-0.006763838184138619),
    (0.05807764256833735),
    (0.10619564137021366),
    (0.1333782902580678),
    (0.13002136114951282),
    (0.10324977746982779),
]
vdnum2_ref = [
    (-0.127928),
    (-0.2243382283746973),
    (-0.2760206636868744),
    (-0.27220579966746783),
    (-0.22027553130270333),
    (-0.13444010386431185),
    (-0.03563601519062676),
    (0.055330261638473016),
    (0.12789139719241827),
    (0.17160771661733515),
    (0.17762726797422304),
    (0.15120346580458535),
    (0.100972939744592),
    (0.04079487861877566),
    (-0.018036818208465574),
    (-0.06770663215960371),
    (-0.10216645884504397),
    (-0.11175538977735708),
    (-0.09977828679379383),
    (-0.07210466726058655),
    (-0.036021335786910226),
    (0.0016061683962408043),
    (0.034949136272573345),
    (0.05993591866295704),
    (0.06975885269855503),
    (0.06552765574752095),
]
vdnum3_ref = [
    (-0.028963),
    (-0.08274555765702404),
    (-0.11865660363389191),
    (-0.12877636631156245),
    (-0.11443714294023342),
    (-0.08075178512262489),
    (-0.036775938607481656),
    (0.007553788906539163),
    (0.046460171634107),
    (0.07393195401472996),
    (0.08429278412481772),
    (0.0785509205551686),
    (0.05964847759684788),
    (0.03352120599663045),
    (0.005463917800776763),
    (-0.02046925825971172),
    (-0.04082120950357363),
    (-0.05016012086852671),
    (-0.04902162281092094),
    (-0.03942149055550854),
    (-0.024356974774582756),
    (-0.006944696795883246),
    (0.009917329517621492),
    (0.02396880921609092),
    (0.03149369616872048),
    (0.03234582585608311),
]
vgnum_ref = [
    (-0.874365),
    (-0.7099220326507727),
    (-0.4261561487361488),
    (-0.10465046845171813),
    (0.19847712581412003),
    (0.4358660029173281),
    (0.5451490601709348),
    (0.5515280703053866),
    (0.49808292892149175),
    (0.32096845590611023),
    (0.11971531598819532),
    (-0.08504474565502744),
    (-0.23807427810913015),
    (-0.32250378785285144),
    (-0.3481462092267303),
    (-0.3273923454130979),
    (-0.24192131787380006),
    (-0.10359979718690686),
    (0.021859623166806383),
    (0.12267004693966344),
    (0.18918124141677414),
    (0.21809383370807792),
    (0.21365109301271507),
    (0.1738734039620283),
    (0.08392518783167185),
    (0.0084378750306759),
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
rd0_max_pass = 1e-12
rd1_max_pass = 1e-12
rd2_max_pass = 1e-12
rd3_max_pass = 1e-12
rg_max_pass = 1e-12
vd0_max_pass = 1e-12
vd1_max_pass = 1e-12
vd2_max_pass = 1e-12
vd3_max_pass = 1e-12
vg_max_pass = 1e-12

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
