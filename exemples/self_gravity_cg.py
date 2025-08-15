import matplotlib.pyplot as plt
import numpy as np

import shamrock


def run_sim(X, rho, phi, phi_ana, Lx=1, Ly=1, Lz=1, rho0=2, G=1, A=1, phi0=0):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 32

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    cfg.set_riemann_solver_hllc()
    cfg.set_eos_gamma(1.4)
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(False)
    # cfg.set_gravity_mode_cg()
    cfg.set_gravity_mode_pcg()
    cfg.set_self_gravity_G_values(True, 1.0)
    cfg.set_self_gravity_Niter_max(200)
    cfg.set_self_gravity_tol(1e-8)

    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    def rho_map(rmin, rmax):
        x_mn, y_mn, z_mn = rmin
        x_mx, y_mx, z_mx = rmax

        x = 0.5 * (x_mn + x_mx)
        y = 0.5 * (y_mn + y_mx)
        z = 0.5 * (z_mn + z_mx)

        res = rho0 + A * np.sin((2 * np.pi * x) / Lx) * np.sin((2 * np.pi * y) / Ly) * np.sin(
            (2 * np.pi * z) / Lz
        )
        return res

    def rhoe_map(rmin, rmax):
        rho = rho_map(rmin, rmax)
        return 1.0 * rho

    def rhovel_map(rmin, rmax):
        rho = rho_map(rmin, rmax)
        return (1 * rho, 0 * rho, 0 * rho)

    def phi_map(rmin, rmax):
        rho = rho_map(rmin, rmax)
        return 0
        # * rho

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)
    model.set_field_value_lambda_f64("phi", phi_map)

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

    freq = 50
    dt = 0.0000
    t = 0
    tend = 0.245

    for k in range(1):
        # if k % freq == 0:
        #     model.dump_vtk("test" + str(k) + ".vtk")

        model.evolve_once_override_time(t, dt)

        dic = convert_to_cell_coords(ctx.collect_data())

        cc = -(4.0 * np.pi * G) / (3 * (2 * np.pi) ** 2)

        tmp = (dic["rho"] - rho0) * cc

        for i in range(len(dic["xmin"])):
            X.append(dic["ymin"][i])
            rho.append(dic["rho"][i])
            phi.append(dic["phi"][i])
            phi_ana.append(tmp[i])

        t = dt * k
        # dt = next_dt

        # if tend < t + next_dt:
        #     dt = tend - t
        if t > tend:
            break


X = []
rho = []
phi = []
phi_ana = []


run_sim(X, rho, phi, phi_ana)
# plt.plot(X, rho, ".", label="rho")
plt.plot(X, phi, ".", label="phi-num")
plt.plot(X, phi_ana, ".", label="phi-ana")
plt.legend()
plt.savefig("with-ghost-64-pcg.png", format="png")
