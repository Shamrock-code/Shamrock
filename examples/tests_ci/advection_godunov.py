"""
Godunov advection test
======================

This test is used to test the Godunov advection setup
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

tmax = 1.0
do_plot = True

multx = 1
multy = 1
multz = 1

sz = 1 << 1
base = 32

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

cfg = model.gen_default_config()
scale_fact = 1 / (sz * base * multx)
cfg.set_scale_factor(scale_fact)
cfg.set_eos_gamma(1.000001)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

kx, ky, kz = 2 * np.pi, 0, 0
delta_rho = 0
delta_v = 1e-5


def rho_map(rmin, rmax):
    x_min, y_min, z_min = rmin
    x_max, y_max, z_max = rmax

    x = (x_min + x_max) / 2

    if x < 0.6 and x > 0.4:
        return 2

    return 1.0


def rhoe_map(rmin, rmax):
    rho = rho_map(rmin, rmax)
    return 1.0 * rho


def rhovel_map(rmin, rmax):
    rho = rho_map(rmin, rmax)
    return (1 * rho, 0 * rho, 0 * rho)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoe_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

model.evolve_until(tmax)


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


def get_l2_distance(a, b):
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2)) / np.sqrt(len(a))


X = []

rho = []
rhovelx = []
rhovely = []
rhovelz = []
rhoetot = []

rho_ref = []
rhovelx_ref = []
rhovely_ref = []
rhovelz_ref = []
rhoetot_ref = []

for i in range(len(dic["xmin"])):
    X.append(dic["xmin"][i])
    rho.append(dic["rho"][i])
    rhovelx.append(dic["rhovel"][i][0])
    rhovely.append(dic["rhovel"][i][1])
    rhovelz.append(dic["rhovel"][i][2])
    rhoetot.append(dic["rhoetot"][i])

    cell_min = (dic["xmin"][i], dic["ymin"][i], dic["zmin"][i])
    cell_max = (dic["xmax"][i], dic["ymax"][i], dic["zmax"][i])

    rho_ref.append(rho_map(cell_min, cell_max))
    rhovelx_ref.append(rhovel_map(cell_min, cell_max)[0])
    rhovely_ref.append(rhovel_map(cell_min, cell_max)[1])
    rhovelz_ref.append(rhovel_map(cell_min, cell_max)[2])
    rhoetot_ref.append(rhoe_map(cell_min, cell_max))

if do_plot:
    plt.plot(X, rho, ".", label="rho")
    plt.plot(X, rho_ref, ".", label="rho_ref")

    plt.plot(X, rhovelx, ".", label="rhovelx")
    plt.plot(X, rhovelx_ref, ".", label="rhovelx_ref")

    plt.legend()
    plt.grid()
    # plt.ylim(0.9,2.5)
    # plt.xlim(0,1)
    plt.title("t=" + str(tmax))
    plt.show()

l2_rho = get_l2_distance(rho, rho_ref)
l2_rhovelx = get_l2_distance(rhovelx, rhovelx_ref)
l2_rhovely = get_l2_distance(rhovely, rhovely_ref)
l2_rhovelz = get_l2_distance(rhovelz, rhovelz_ref)
l2_rhoetot = get_l2_distance(rhoetot, rhoetot_ref)

print(f"rho: {l2_rho}")
print(f"rhovelx: {l2_rhovelx}")
print(f"rhovely: {l2_rhovely}")
print(f"rhovelz: {l2_rhovelz}")
print(f"rhoetot: {l2_rhoetot}")

expected_l2_rho = 0.12026000336984567
expected_l2_rhovelx = 0.12026008009555872
expected_l2_rhovely = 0
expected_l2_rhovelz = 0
expected_l2_rhoetot = 0.12026008010012906

test_pass = True
pass_rho = 1.5e-15
pass_rhovelx = 1.5e-15
pass_rhovely = 0
pass_rhovelz = 0
pass_rhoetot = 1.5e-15

err_log = ""

if np.abs(l2_rho - expected_l2_rho) > pass_rho:
    err_log += f"error on rho is too far from expected value: {l2_rho} != {expected_l2_rho}, delta: {np.abs(l2_rho - expected_l2_rho)}\n"
    test_pass = False
if np.abs(l2_rhovelx - expected_l2_rhovelx) > pass_rhovelx:
    err_log += f"error on rhovelx is too far from expected value: {l2_rhovelx} != {expected_l2_rhovelx}, delta: {np.abs(l2_rhovelx - expected_l2_rhovelx)}\n"
    test_pass = False
if np.abs(l2_rhovely - expected_l2_rhovely) > pass_rhovely:
    err_log += f"error on rhovely is too far from expected value: {l2_rhovely} != {expected_l2_rhovely}, delta: {np.abs(l2_rhovely - expected_l2_rhovely)}\n"
    test_pass = False
if np.abs(l2_rhovelz - expected_l2_rhovelz) > pass_rhovelz:
    err_log += f"error on rhovelz is too far from expected value: {l2_rhovelz} != {expected_l2_rhovelz}, delta: {np.abs(l2_rhovelz - expected_l2_rhovelz)}\n"
    test_pass = False
if np.abs(l2_rhoetot - expected_l2_rhoetot) > pass_rhoetot:
    err_log += f"error on rhoetot is too far from expected value: {l2_rhoetot} != {expected_l2_rhoetot}, delta: {np.abs(l2_rhoetot - expected_l2_rhoetot)}\n"
    test_pass = False

if test_pass == False:
    exit("Test did not pass L2 margins : \n" + err_log)
