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
def run_sim(times, vg_num, vd1_num, vd2_num):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 2

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    cfg.set_Csafe(
        0.44
    )  #  TODO : remember to add possibility of different CFL for fluids(e.g Csafe_gas and Csafe_dust ...)
    cfg.set_eos_gamma(1.4)  # set adiabatic index gamma , here adiabatic EOS
    cfg.set_dust_mode_dhll(2)  # enable dust config
    cfg.set_drag_mode_irk1(True)  # enable drag config
    cfg.set_face_time_interpolation(False)

    # ======= set drag coefficients for test B ========
    cfg.set_alpha_values(100)  # ts := 0.01
    cfg.set_alpha_values(500)  # ts := 0.002

    """
    #======= set drag coefficients for test C ========
    cfg.set_alpha_values(0.5)          # ts  := 2
    cfg.set_alpha_values(1)            # ts  := 1
    """

    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    # ============= Fileds maps for gas ==============

    def rho_map(rmin, rmax):
        return 1  # 1 is the initial density

    def rhovel_map(rmin, rmax):
        return (1, 0, 0)  # vg_x:=1, vg_y:=0, vg_z:=0

    def rhoe_map(rmin, rmax):
        cs_0 = 1.4
        gamma = 1.4
        rho_0 = 1
        press = (cs_0 * rho_0) / gamma
        rhoeint = press / (gamma - 1.0)
        rhoekin = 0.5 * rho_0  # vg_x:=1, vg_y:=0, vg_z:=0
        return rhoeint + rhoekin

    # =========== Fields maps for dust in test B ============
    def b_rho_d_1_map(rmin, rmax):
        return 1  # rho_d_1 := 1

    def b_rho_d_2_map(rmin, rmax):
        return 1  # rho_d_2 := 1

    def b_rhovel_d_1_map(rmin, rmax):
        return (2, 0, 0)  # vd_1_x:=2, vd_1_y:=0, vd_1_z:=0

    def b_rhovel_d_2_map(rmin, rmax):
        return (0.5, 0, 0)  # vd_2_x:=0.5, vd_2_y:=0, vd_2_z:=0

    # =========== Fields maps for dust in test C ============
    def c_rho_d_1_map(rmin, rmax):
        return 10  # rho_d_1 := 10

    def c_rho_d_2_map(rmin, rmax):
        return 100  # rho_d_2 := 100

    def c_rhovel_d_1_map(rmin, rmax):
        return (20, 0, 0)  # vd_1_x:=2, vd_1_y:=0, vd_1_z:=0

    def c_rhovel_d_2_map(rmin, rmax):
        return (50, 0, 0)  # vd_2_x:=0.5, vd_2_y:=0, vd_2_z:=0

    # ============ set init fields values for gas =============

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    # ============ set init fields values for dusts in test B ==========
    model.set_field_value_lambda_f64("rho_dust", b_rho_d_1_map, 0)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_1_map, 0)
    model.set_field_value_lambda_f64("rho_dust", b_rho_d_2_map, 1)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_2_map, 1)

    # ============ set init fields values for dusts in test C ==========
    # model.set_field_value_lambda_f64("rho_dust", c_rho_d_1_map,0)
    # model.set_field_value_lambda_f64_3("rhovel_dust", c_rhovel_d_1_map,0)
    # model.set_field_value_lambda_f64("rho_dust", c_rho_d_2_map,1)
    # model.set_field_value_lambda_f64_3("rhovel_dust", c_rhovel_d_2_map,1)

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

    dt = 0.005  # b_dt := 0.005 and c_dt := 0.05
    t = 0
    tend = 0.05  # b_tend := 0.05 and c_tend := 0.3
    freq = 1

    for i in range(13):
        if i % freq == 0:
            model.dump_vtk("colid_test" + str(i // freq) + ".vtk")

        dic_i = convert_to_cell_coords(ctx.collect_data())
        vg_num.append(dic_i["rhovel"][0][0] / dic_i["rho"][0])
        vd1_num.append(dic_i["rhovel_dust"][0][0] / dic_i["rho_dust"][0])
        vd2_num.append(dic_i["rhovel_dust"][1][0] / dic_i["rho_dust"][1])
        model.evolve_once_override_time(dt * float(i), dt)
        t = dt * i
        times.append(t)

        if t > tend:
            break


# ============== post treatment ===================


## ========= analytical function for velocity =====
def analytical_velocity(t, Vcom, c1, c2, lambda1, lambda2):
    return Vcom + c1 * exp(lambda1 * t) + c2 * exp(lambda2 * t)


## =========Test A setup=========
Vcom_A = 1.16666666666666
lambda1_A = -0.63397459621556
lambda2_A = -2.36602540378444
cg1_A = -0.22767090063074
cg2_A = 0.06100423396407
cd11_A = 0.84967936855889
cd12_A = -0.01634603522555
cd21_A = -0.62200846792815
cd22_A = -0.04465819873852

## =========Test B setup=========
Vcom_B = 1.166666666666667
lambda1_B = -141.742430504416
lambda2_B = -1058.25756949558
cg1_B = -0.35610569612832
cg2_B = 0.18943902946166
cd11_B = 0.85310244713865
cd12_B = -0.01976911380532
cd21_B = -0.49699675101033
cd22_B = -0.16966991565634

## =========Test C setup=========
Vcom_C = 0.63963963963963
lambda1_C = -0.52370200744224
lambda2_C = -105.976297992557
cg1_C = -0.06458203330249
cg2_C = 0.42494239366285
cd11_C = 1.36237475791577
cd12_C = -0.00201439755542
cd21_C = -0.13559165545855
cd22_C = -0.00404798418109


## ===== get numerical results ==================
times = []
vg_num = []
vd1_num = []
vd2_num = []

run_sim(times, vg_num, vd1_num, vd2_num)

## =========== get analytical results =========
test_type = "B"
if test_type == "A":
    vg_anal = [analytical_velocity(t, Vcom_A, cg1_A, cg2_A, lambda1_A, lambda2_A) for t in times]
    vd1_anal = [analytical_velocity(t, Vcom_A, cd11_A, cd12_A, lambda1_A, lambda2_A) for t in times]
    vd2_anal = [analytical_velocity(t, Vcom_A, cd21_A, cd22_A, lambda1_A, lambda2_A) for t in times]

elif test_type == "B":
    vg_anal = [analytical_velocity(t, Vcom_B, cg1_B, cg2_B, lambda1_B, lambda2_B) for t in times]
    vd1_anal = [analytical_velocity(t, Vcom_B, cd11_B, cd12_B, lambda1_B, lambda2_B) for t in times]
    vd2_anal = [analytical_velocity(t, Vcom_B, cd21_B, cd22_B, lambda1_B, lambda2_B) for t in times]

elif test_type == "C":
    vg_anal = [analytical_velocity(t, Vcom_C, cg1_C, cg2_C, lambda1_C, lambda2_C) for t in times]
    vd1_anal = [analytical_velocity(t, Vcom_C, cd11_C, cd12_C, lambda1_C, lambda2_C) for t in times]
    vd2_anal = [analytical_velocity(t, Vcom_C, cd21_C, cd22_C, lambda1_C, lambda2_C) for t in times]


print(f"times = {times} with len = {len(times)}\n")
print(f" vg_num = {vg_num} with len = {len(vg_num)} \n")
print(f" vd1_num = {vd1_num} with len = {len(vd1_num)}  \n")
print(f" vd2_num = {vd2_num} with len = {len(vd2_num)} \n")


# ============ plots ======================
fig, axs = plt.subplots(1, 3, figsize=(25, 10))
plt.subplots_adjust(wspace=0.25)
axs[0].plot(times, vg_num, lw=2, label="vg_num")
axs[1].plot(times, vd1_num, lw=2, label="vd1_num")
axs[2].plot(times, vd2_num, lw=2, label="vd2_num")

axs[0].plot(times, vg_anal, "k--", lw=2, label="vg_ana")
axs[1].plot(times, vd1_anal, "k--", lw=2, label="vd1_ana")
axs[2].plot(times, vd2_anal, "k--", lw=2, label="vd2_ana")

axs[0].set_xlabel("Time", fontsize=15, fontweight="bold")
axs[1].set_xlabel("Time", fontsize=15, fontweight="bold")
axs[2].set_xlabel("Time", fontsize=15, fontweight="bold")
axs[0].set_ylabel("Velocity", fontsize=15, fontweight="bold")

axs[0].set_title("$V_{g}$", fontsize=15, fontweight="bold")
axs[1].set_title("$V_{d,1}$", fontsize=15, fontweight="bold")
axs[2].set_title("$V_{d,2}$", fontsize=15, fontweight="bold")

plt.legend(prop={"weight": "bold"})
plt.savefig("dusty_collision_test_B.png")
# plt.savefig("dusty_collision_test_C.png")
