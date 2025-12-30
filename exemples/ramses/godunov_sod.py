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

sz = 1 << 1
base = 32

cfg = model.gen_default_config()
scale_fact = 2 / (sz * base * multx)
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
model.set_solver_config(cfg)


model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

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

# model.evolve_once(0,0.1)
freq = 50
dt = 0.0000
t = 0
tend = 0.245

for i in range(1):

    if i % freq == 0:
        model.dump_vtk("test" + str(i // freq) + ".vtk")

    next_dt = model.evolve_once_override_time(t, dt)

    t += dt
    dt = next_dt

    if tend < t + next_dt:
        dt = tend - t
    if t == tend:
        break


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
rho = []
rhovelx = []
rhoetot = []

for i in range(len(dic["xmin"])):

    X.append(dic["xmin"][i] - 0.5)
    rho.append(dic["rho"][i])
    rhovelx.append(dic["rhovel"][i][0])
    rhoetot.append(dic["rhoetot"][i])

X = np.array(X)
rho = np.array(rho)
rhovelx = np.array(rhovelx)
rhoetot = np.array(rhoetot)

vx = rhovelx / rho

plt.plot(X, rho, ".", label="rho")
plt.plot(X, vx, ".", label="v")
plt.plot(X, (rhoetot - 0.5 * rho * (vx**2)) * (gamma - 1), ".", label="P")
# plt.plot(X,rhoetot,'.',label="rhoetot")
plt.legend()
plt.grid()
plt.ylim(0, 1.1)
plt.xlim(0, 1)
plt.title("t=" + str(t))
plt.show()
