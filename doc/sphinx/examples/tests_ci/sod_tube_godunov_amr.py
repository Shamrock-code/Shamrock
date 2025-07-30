"""
Testing Sod tube with Godunov with AMR enabled
==============================================

CI test for Sod tube with Godunov with AMR enabled
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")


multx = 4
multy = 1
multz = 1

cell_size = 1 << 2  # refinement is limited to cell_size = 2
base = 16

cfg = model.gen_default_config()
scale_fact = 2 / (cell_size * base * multx)
cfg.set_scale_factor(scale_fact)

gamma = 1.4
cfg.set_eos_gamma(gamma)
# cfg.set_riemann_solver_rusanov()
cfg.set_riemann_solver_hll()

# cfg.set_slope_lim_none()
# cfg.set_slope_lim_vanleer_f()
# cfg.set_slope_lim_vanleer_std()
# cfg.set_slope_lim_vanleer_sym()
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)
mass_crit = 0.0000001 * 5 * 2 * 1.2
cfg.set_amr_mode_density_based(crit_mass=mass_crit)
model.set_solver_config(cfg)


model.init_scheduler(int(1e7), 1)
model.make_base_grid(
    (0, 0, 0), (cell_size, cell_size, cell_size), (base * multx, base * multy, base * multz)
)

# without face time interpolation
# 0.07979993131348424 (0.17970690984930585, 0.0, 0.0) 0.12628776652228088

# with face time interpolation
# 0.07894793711859852 (0.17754462339166546, 0.0, 0.0) 0.12498304725061045


kx, ky, kz = 2 * np.pi, 0, 0
delta_rho = 1e-2


def rho_map(rmin, rmax):

    x, y, z = rmin
    if x < 1:
        return 1
    else:
        return 0.125


etot_L = 1.0 / (gamma - 1)
etot_R = 0.1 / (gamma - 1)


def rhoetot_map(rmin, rmax):

    rho = rho_map(rmin, rmax)

    x, y, z = rmin
    if x < 1:
        return etot_L
    else:
        return etot_R


def rhovel_map(rmin, rmax):
    rho = rho_map(rmin, rmax)

    return (0, 0, 0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

t_target = 0.245

# for i in range(1000):
#    model.dump_vtk(f"test{i:04d}.vtk")
#    model.timestep()

model.evolve_until(t_target)

# model.evolve_once()
xref = 1.0
xrange = 0.5
sod = shamrock.phys.SodTube(gamma=gamma, rho_1=1, P_1=1, rho_5=0.125, P_5=0.1)
sodanalysis = model.make_analysis_sodtube(sod, (1, 0, 0), t_target, xref, -xrange, xrange)


#################
### Plot
#################
# do plot or not
if True:

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

    dic = convert_to_cell_coords(ctx.collect_data())

    X = []
    dX = []
    rho = []
    rhovelx = []
    rhoetot = []

    for i in range(len(dic["xmin"])):

        X.append(dic["xmin"][i])
        dX.append(dic["xmax"][i] - dic["xmin"][i])
        rho.append(dic["rho"][i])
        rhovelx.append(dic["rhovel"][i][0])
        rhoetot.append(dic["rhoetot"][i])

    X = np.array(X)
    dX = np.array(dX)
    rho = np.array(rho)
    rhovelx = np.array(rhovelx)
    rhoetot = np.array(rhoetot)

    vx = rhovelx / rho

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), dpi=125)

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    l = -np.log2(dX / np.max(dX)) + 1

    ax1.scatter(X, rho, rasterized=True, label="rho")
    ax1.scatter(X, vx, rasterized=True, label="v")
    ax1.scatter(X, (rhoetot - 0.5 * rho * (vx**2)) * (gamma - 1), rasterized=True, label="P")
    ax2.scatter(X, l, rasterized=True, color="purple", label="AMR level")
    # plt.scatter(X,rhoetot, rasterized=True,label="rhoetot")
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    ax1.grid()

    #### add analytical soluce
    arr_x = np.linspace(xref - xrange, xref + xrange, 1000)

    arr_rho = []
    arr_P = []
    arr_vx = []

    for i in range(len(arr_x)):
        x_ = arr_x[i] - xref

        _rho, _vx, _P = sod.get_value(t_target, x_)
        arr_rho.append(_rho)
        arr_vx.append(_vx)
        arr_P.append(_P)

    ax1.plot(arr_x, arr_rho, color="black", label="analytic")
    ax1.plot(arr_x, arr_vx, color="black")
    ax1.plot(arr_x, arr_P, color="black")

    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlim(0.5, 1.5)
    ax2.set_ylabel("AMR level")
    plt.title(r"$m_{crit}=" + str(mass_crit) + "$")
    plt.savefig("sod_tube.pdf")
    plt.savefig("sod_tube.png")
    #######
    plt.show()

#################
### Test CD
#################
rho, v, P = sodanalysis.compute_L2_dist()
print(rho, v, P)
vx, vy, vz = v

# normally :
# rho 0.07979993131348424
# v (0.17970690984930585, 0.0, 0.0)
# P 0.12628776652228088

test_pass = True
pass_rho = 0.07913442601255971 + 1e-7
pass_vx = 0.17762998971731672 + 1e-7
pass_vy = 0
pass_vz = 0
pass_P = 0.12516172582510562 + 1e-7

err_log = ""

if rho > pass_rho:
    err_log += ("error on rho is too high " + str(rho) + ">" + str(pass_rho)) + "\n"
    test_pass = False
if vx > pass_vx:
    err_log += ("error on vx is too high " + str(vx) + ">" + str(pass_vx)) + "\n"
    test_pass = False
if vy > pass_vy:
    err_log += ("error on vy is too high " + str(vy) + ">" + str(pass_vy)) + "\n"
    test_pass = False
if vz > pass_vz:
    err_log += ("error on vz is too high " + str(vz) + ">" + str(pass_vz)) + "\n"
    test_pass = False
if P > pass_P:
    err_log += ("error on P is too high " + str(P) + ">" + str(pass_P)) + "\n"
    test_pass = False

if test_pass == False:
    exit("Test did not pass L2 margins : \n" + err_log)
