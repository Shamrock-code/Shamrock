import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

tmax = 1


def run_sim(ctp=False, vanleer=False, label="none"):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Zeus(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 32

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    cfg.set_eos_gamma(1.0)
    cfg.set_consistent_transport(ctp)
    cfg.set_van_leer(vanleer)
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    kx, ky, kz = 2 * np.pi, 0, 0
    delta_rho = 0
    delta_v = 1e-4

    def rho_map(rmin, rmax):

        x, y, z = rmin

        if x < 0.6 and x > 0.4:
            return 2

        return 1.0

    def eint_map(rmin, rmax):

        return 1.0

    def vel_map(rmin, rmax):

        x, y, z = rmin
        return (1, 0, 0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("eint", eint_map)
    model.set_field_value_lambda_f64_3("vel", vel_map)

    # model.evolve_once(0,0.1)
    freq = 20
    t = 0

    dt = 1 / 128
    for i in range(1000):

        if i % freq == 0:
            model.dump_vtk("test" + str(i // freq) + ".vtk")

        model.evolve_once(float(i), dt)
        t = dt * i

        if t >= tmax:
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
    velx = []
    eint = []

    for i in range(len(dic["xmin"])):

        X.append(dic["xmin"][i])
        rho.append(dic["rho"][i])
        velx.append(dic["vel"][i][0])

    plt.plot(X, rho, ".", label=label)


run_sim(ctp=False, vanleer=False, label="no consistent transp donor cell")
run_sim(ctp=False, vanleer=True, label="no consistent transp van leer")

run_sim(ctp=True, vanleer=False, label="consistent transp donor cell")
run_sim(ctp=True, vanleer=True, label="consistent transp van leer")

plt.legend()
plt.grid()
plt.ylim(0.9, 2.5)
plt.xlim(0, 1)
plt.title("t=" + str(tmax))
plt.show()
