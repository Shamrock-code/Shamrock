import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")


multx = 1
multy = 1
multz = 1
max_amr_lev = 2
cell_size = 1 << max_amr_lev  # refinement is limited to cell_size = 2
base = 16

cfg = model.gen_default_config()
scale_fact = 1 / (cell_size * base * multx)
cfg.set_scale_factor(scale_fact)

gamma = 1.4
cfg.set_eos_gamma(gamma)
cfg.set_boundary_condition("x", "reflective")
cfg.set_boundary_condition("y", "reflective")
cfg.set_boundary_condition("z", "reflective")
cfg.set_riemann_solver_hllc()
smooth_crit = 0.3
# cfg.set_amr_mode_slope_based(crit_smooth=smooth_crit)
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)
mass_crit = 1e-6 * 5 * 2 * 2
# cfg.set_amr_mode_density_based(crit_mass=mass_crit)


err_min = 0.01
err_max = 0.02
# cfg.set_amr_mode_pseudo_gradient_based(error_min=err_min, error_max=err_max)


crit_refin = 0.1
crit_coars = 0.2
cfg.set_amr_mode_second_order_derivative_based(crit_min=crit_refin, crit_max=crit_coars)
model.set_solver_config(cfg)


model.init_scheduler(int(1e7), 1)
model.make_base_grid(
    (0, 0, 0), (cell_size, cell_size, cell_size), (base * multx, base * multy, base * multz)
)

# without face time interpolation
# 0.07979993131348424 (0.17970690984930585, 0.0, 0.0) 0.12628776652228088

# with face time interpolation
# 0.07894793711859852 (0.17754462339166546, 0.0, 0.0) 0.12498304725061045


def rho_map(rmin, rmax):

    x, y, z = rmin
    if x < 0.5:
        return 1
    else:
        return 0.125


etot_L = 1.0 / (gamma - 1)
etot_R = 0.1 / (gamma - 1)


def rhoetot_map(rmin, rmax):

    rho = rho_map(rmin, rmax)

    x, y, z = rmin
    if x < 0.5:
        return etot_L
    else:
        return etot_R


def rhovel_map(rmin, rmax):
    rho = rho_map(rmin, rmax)

    return (0, 0, 0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)


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


t_target = 0.245

dt = 0
t = 0
freq = 1
dX0 = []
for i in range(10000):

    if i % freq == 0:
        model.dump_vtk(f"test{i:04d}.vtk")
    next_dt = model.evolve_once_override_time(t, dt)
    if i == 0:
        dic0 = convert_to_cell_coords(ctx.collect_data())
        dX0.append(dic0["xmax"][i] - dic0["xmin"][i])

    t += dt
    dt = next_dt

    if t_target < t + next_dt:
        dt = t_target - t
    if t == t_target:
        break

# for i in range(1000):
# model.dump_vtk(f"test{i:04d}.vtk")
#    model.timestep()

# model.evolve_until(t_target)

# model.evolve_once()
xref = 0.5
xrange = 0.5
sod = shamrock.phys.SodTube(gamma=gamma, rho_1=1, P_1=1, rho_5=0.125, P_5=0.1)
sodanalysis = model.make_analysis_sodtube(sod, (1, 0, 0), t_target, xref, 0.0, 1.0)


#################
### Plot
#################
# do plot or not
if True:

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
    dX0 = np.array(dX0)
    rho = np.array(rho)
    rhovelx = np.array(rhovelx)
    rhoetot = np.array(rhoetot)

    vx = rhovelx / rho

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), dpi=125)

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    l = -np.log2(dX / np.max(dX0)) + 1

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

    # ax1.set_ylim(-0.1, 1.1)
    # ax1.set_xlim(0.5, 1.5)
    ax2.set_ylabel("AMR level")
    # plt.title(r"$m_{crit}=" + str(mass_crit) + "$")
    # plt.title(r"$smooth_{crit}=" + str(smooth_crit) + "$")
    # plt.title(f"err_min={err_min} --- err_max = {err_max} -- max_amr_lev ={max_amr_lev}")
    plt.title(f"crit_ref={crit_refin} --- crit_coars = {crit_coars} -- max_amr_lev ={max_amr_lev}")
    # plt.savefig(f"sod_tube-mass-{mass_crit}-base-{base}-tf-{t_target}-reflective_mass.pdf")
    # plt.savefig(f"sod_tube-mass-{smooth_crit}-base-{base}-tf-{t_target}-reflective_slope_based.pdf")
    # plt.savefig(
    #     f"sod_tube-err_min-{err_min}-err_max-{err_max}-base-{base}-max_amr-{max_amr_lev}-tf-{t_target}-reflective_pseudo_gradient_based.pdf"
    # )
    plt.savefig(
        f"sod_tube-crit_ref-{crit_refin}-crit_coars-{crit_coars}-base-{base}-max_amr-{max_amr_lev}-tf-{t_target}-reflective_second_order_derivative_based_no_interpolation.pdf"
    )

    plt.savefig("sod_tube.png")
    #######
    plt.show()

# #################
# ### Test CD
# #################
# rho, v, P = sodanalysis.compute_L2_dist()
# print(rho, v, P)
# vx, vy, vz = v

# # normally :
# # rho 0.07979993131348424
# # v (0.17970690984930585, 0.0, 0.0)
# # P 0.12628776652228088

# test_pass = True
# pass_rho = 0.07979993131348424 + 1e-7
# pass_vx = 0.17970690984930585 + 1e-7
# pass_vy = 1e-09
# pass_vz = 1e-09
# pass_P = 0.12628776652228088 + 1e-7

# err_log = ""

# if rho > pass_rho:
#     err_log += ("error on rho is too high " + str(rho) + ">" + str(pass_rho)) + "\n"
#     test_pass = False
# if vx > pass_vx:
#     err_log += ("error on vx is too high " + str(vx) + ">" + str(pass_vx)) + "\n"
#     test_pass = False
# if vy > pass_vy:
#     err_log += ("error on vy is too high " + str(vy) + ">" + str(pass_vy)) + "\n"
#     test_pass = False
# if vz > pass_vz:
#     err_log += ("error on vz is too high " + str(vz) + ">" + str(pass_vz)) + "\n"
#     test_pass = False
# if P > pass_P:
#     err_log += ("error on P is too high " + str(P) + ">" + str(pass_P)) + "\n"
#     test_pass = False

# if test_pass == False:
#     exit("Test did not pass L2 margins : \n" + err_log)
