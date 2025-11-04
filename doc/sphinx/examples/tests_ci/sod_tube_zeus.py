"""
Testing Sod tube with Zeus
=========================

CI test for Sod tube with Zeus
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

multx = 4
multy = 1
multz = 1

sz = 1 << 1
base = 32
gamma = 1.4


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Zeus(context=ctx, vector_type="f64_3", grid_repr="i64_3")


cfg = model.gen_default_config()
scale_fact = 2 / (sz * base * multx)
cfg.set_scale_factor(scale_fact)

cfg.set_eos_gamma(gamma)
cfg.set_consistent_transport(True)
cfg.set_van_leer(True)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))


def rho_map(rmin, rmax):
    x, y, z = rmin
    if x < 1:
        return 1
    else:
        return 0.125


eint_L = 1.0 / (gamma - 1)
eint_R = 0.1 / (gamma - 1)


def eint_map(rmin, rmax):
    x, y, z = rmin
    if x < 1:
        return eint_L
    else:
        return eint_R


def vel_map(rmin, rmax):
    return (0, 0, 0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("eint", eint_map)
model.set_field_value_lambda_f64_3("vel", vel_map)

t_target = 0.245


# model.evolve_once(0,0.1)
freq = 50
dt = 0.0010
t = 0
for i in range(701):
    model.evolve_once(i * dt, dt)
    t = i * dt
    if i * dt >= t_target:
        break

# model.evolve_until(t_target)

# model.evolve_once()
xref = 1.0
xrange = 0.5
sod = shamrock.phys.SodTube(gamma=gamma, rho_1=1, P_1=1, rho_5=0.125, P_5=0.1)
sodanalysis = model.make_analysis_sodtube(sod, (1, 0, 0), t_target, xref, -xrange, xrange)


#################
### Plot
#################
# do plot or not
if False:

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
    velx = []
    P = []

    for i in range(len(dic["xmin"])):

        X.append(dic["xmin"][i] - 0.5)
        rho.append(dic["rho"][i])
        velx.append(dic["vel"][i][0])
        P.append(dic["eint"][i] * (gamma - 1))

    X = np.array(X)
    rho = np.array(rho)
    velx = np.array(velx)
    P = np.array(P)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), dpi=125)

    plt.scatter(X, rho, rasterized=True, label="rho")
    plt.scatter(X, velx, rasterized=True, label="v")
    plt.scatter(X, P, rasterized=True, label="P")
    # plt.scatter(X,rhoetot, rasterized=True,label="rhoetot")
    plt.legend()
    plt.grid()

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

    plt.plot(arr_x, arr_rho, color="black", label="analytic")
    plt.plot(arr_x, arr_vx, color="black")
    plt.plot(arr_x, arr_P, color="black")
    plt.ylim(-0.1, 1.1)
    plt.xlim(0.5, 1.5)
    #######
    plt.show()

#################
### Test CD
#################
rho, v, P = sodanalysis.compute_L2_dist()
vx, vy, vz = v

if shamrock.sys.world_rank() == 0:
    print("L2 norm : rho = ", rho, " v = ", v, " P = ", P)

test_pass = True
pass_rho = 0.08027925640209972 + 1e-7
pass_vx = 0.18526690716374897 + 1e-7
pass_vy = 1e-09
pass_vz = 1e-09
pass_P = 0.1263222182067176 + 1e-7

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
