import matplotlib.pyplot as plt
import numpy as np

import shamrock


def run_sim(X, Y, Z, rho, phi, G=1, rho_0=1, r0=0.25):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 8

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    center = (base * scale_fact, base * scale_fact, base * scale_fact)
    xc, yc, zc = center

    cfg.set_riemann_solver_hllc()
    cfg.set_eos_gamma(1.4)
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(False)
    cfg.set_gravity_mode_cg()
    # cfg.set_gravity_mode_pcg()
    # cfg.set_gravity_mode_bicgstab()
    cfg.set_self_gravity_G_values(True, 1.0)
    cfg.set_self_gravity_Niter_max(100)
    cfg.set_self_gravity_tol(1e-20)
    cfg.set_self_gravity_happy_breakdown_tol(1e-6)
    model.set_solver_config(cfg)

    model.init_scheduler(int(5000000), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    def rho_map(rmin, rmax):
        x_mn, y_mn, z_mn = rmin
        x_mx, y_mx, z_mx = rmax
        x0 = 0.5 * (x_mn + x_mx)
        y0 = 0.5 * (y_mn + y_mx)
        z0 = 0.5 * (z_mn + z_mx)
        # x = x_mn- xc
        # y = y_mn - yc
        # z = z_mn - zc
        y = y0 - yc
        x = x0 - xc
        z = z0 - zc
        r = np.sqrt(x * x + y * y + z * z)
        res = 0.0
        if r <= r0:
            rr = r * (1.0 / rho_0)
            cc = 1 - (rr * rr)
            res = rho_0 * (cc * cc)
        else:
            res = 0.0
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

        dic = ctx.collect_data()

        if shamrock.sys.world_rank() == 0:
            dic = convert_to_cell_coords(dic)

            for i in range(len(dic["xmin"])):
                # X.append(0.5 * (dic["xmin"][i] + dic["xmax"][i]))
                # Y.append(0.5 * (dic["ymin"][i] + dic["ymax"][i]))
                # Z.append(0.5 * (dic["zmin"][i] + dic["zmax"][i]))
                X.append(dic["xmin"][i])
                Y.append(dic["ymin"][i])
                Z.append(dic["zmin"][i])
                rho.append(dic["rho"][i])
                phi.append(dic["phi"][i])

        t = dt * k
        # dt = next_dt

        # if tend < t + next_dt:
        #     dt = tend - t

        if t > tend:
            break

    return xc, yc, zc


r0 = 0.25
rho0 = 1.0
X = []
Y = []
Z = []
rho = []
phi = []

x_c, y_c, z_c = run_sim(X, Y, Z, rho, phi)


def analytical_phi(r0, rho0, X, Y, Z, x_c, y_c, z_c):
    M = (32.0 * np.pi * rho0 * (r0**3)) / 105.0

    x = X - x_c
    y = Y - y_c
    z = Z - z_c

    r = np.sqrt(np.pow(x, 2) + np.pow(y, 2) + np.pow(z, 2))
    res = []

    N = X.shape[0]

    for i in range(N):
        if r[i] <= r0:
            res.append(
                (-2.0 / 3.0) * np.pi * rho0 * (r0**2)
                + 4
                * np.pi
                * rho0
                * (
                    (1.0 / 6) * (r[i] ** 2)
                    - (1.0 / 10.0) * (r[i] ** 4) / (r0**2)
                    + (1.0 / 42.0) * (r[i] ** 6) / (r[i] ** 4)
                )
            )
        else:
            res.append(-M / r[i])

    # res[res <= r0] = (-2./3.)*np.pi*rho0*(r0**2) + 4*np.pi*rho0*((1./6)*(r**2) -(1./10.)*(r**4)/(r0**2) + (1./42.)*(r**6)/(r**4))
    # res[res > r0] = -M/r

    return res


ana = analytical_phi(r0, rho0, np.array(X), np.array(Y), np.array(Z), x_c, y_c, z_c)
diff = np.array(phi) - np.array(ana)

plt.plot(np.array(X), np.array(phi), "+", label="phi-num")
plt.plot(np.array(X), np.array(ana), "-", label="phi-ana")
plt.plot(np.array(X), diff, "*", label="diff")
plt.legend()
plt.savefig("pluto-test-plots-center-based-1.png", format="png")
