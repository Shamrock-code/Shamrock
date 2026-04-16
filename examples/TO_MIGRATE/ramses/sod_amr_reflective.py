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
max_amr_lev = 3
cell_size = 2 << max_amr_lev  # refinement is limited to cell_size = 2
base = 16

cfg = model.gen_default_config()
scale_fact = 1 / (cell_size * base * multx)
cfg.set_scale_factor(scale_fact)

gamma = 1.4
cfg.set_eos_gamma(gamma)
cfg.set_Csafe(0.3)
cfg.set_boundary_condition("x", "reflective")
cfg.set_boundary_condition("y", "reflective")
cfg.set_boundary_condition("z", "reflective")
cfg.set_riemann_solver_hllc()


cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)

err_min = 0.25
err_max = 0.15

cfg.set_amr_mode_pseudo_gradient_based(error_min=err_min, error_max=err_max)

mass_crit = 1e-6 * 5 * 2 * 2
# cfg.set_amr_mode_density_based(crit_mass=mass_crit)


crit_refin = 0.1
crit_coars = 0.2
# cfg.set_amr_mode_second_order_derivative_based(crit_min=crit_refin, crit_max=crit_coars)
model.set_solver_config(cfg)


model.init_scheduler(int(1e7), 1)
model.make_base_grid(
    (0, 0, 0), (cell_size, cell_size, cell_size), (base * multx, base * multy, base * multz)
)


def rho_map(rmin, rmax):

    x, y, z = rmin
    if y < 0.5:
        return 1
    else:
        return 0.125


etot_L = 1.0 / (gamma - 1)
etot_R = 0.1 / (gamma - 1)


def rhoetot_map(rmin, rmax):
    x, y, z = rmin
    if y < 0.5:
        return etot_L
    else:
        return etot_R


def rhovel_map(rmin, rmax):
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
freq = 10
dX0 = 0
for i in range(50):
    next_dt = model.evolve_once_override_time(t, dt)
    if i == 0:
        dic0 = convert_to_cell_coords(ctx.collect_data())
        dX0 = dic0["ymax"][0] - dic0["ymin"][0]

    t += dt
    dt = next_dt

    if i % freq == 0:
        model.dump_vtk(f"test{i:04d}.vtk")

    if t_target < t + next_dt:
        dt = t_target - t
    if t == t_target:
        model.dump_vtk(f"test{i:04d}.vtk")
        break

xref = 0.5
xrange = 0.5
sod = shamrock.phys.SodTube(gamma=gamma, rho_1=1, P_1=1, rho_5=0.125, P_5=0.1)
sodanalysis = model.make_analysis_sodtube(sod, (0, 1, 0), t_target, xref, 0.0, 1.0)


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

    for i in range(len(dic["ymin"])):
        X.append(dic["ymin"][i])
        dX.append(dic["ymax"][i] - dic["ymin"][i])
        rho.append(dic["rho"][i])
        rhovelx.append(dic["rhovel"][i][1])
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

    l_0 = np.log2(base * 2)

    l = -np.log2(dX / max(dX0, dX.max())) + l_0

    ax1.scatter(X, rho, rasterized=True, s=12 * np.ones(X.shape), label="rho")
    ax1.scatter(X, vx, rasterized=True, s=12 * np.ones(X.shape), label="v")
    ax1.scatter(
        X,
        (rhoetot - 0.5 * rho * (vx**2)) * (gamma - 1),
        rasterized=True,
        s=12 * np.ones(X.shape),
        label="P",
    )
    idx = np.argsort(X)
    ax2.plot(X[idx], l[idx], color="purple", marker="D", linewidth=2.0, ls="-.", label="AMR level")
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

    ax1.plot(arr_x, arr_rho, ls="--", lw=2.0, color="black", label="analytic")
    ax1.plot(arr_x, arr_vx, ls="--", lw=2.0, color="black")
    ax1.plot(arr_x, arr_P, ls="--", lw=2.0, color="black")
    ax2.set_ylabel("AMR level")
    plt.title(f"Threshold = {err_max}, derefinement factor = {err_min}")
    plt.savefig("sod_tube_3_1_baryonic_density_check_density_5.png")
    #######
