import matplotlib.pyplot as plt
import numpy as np

import shamrock


def run_sim(X, Y, Z, rho, phi, phi_ana, Lx=1, Ly=1, Lz=1, rho0=2, G=1, A=1, phi0=1):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 16

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    cfg.set_riemann_solver_hllc()
    cfg.set_eos_gamma(1.4)
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(False)
    cfg.set_gravity_mode_cg()
    # cfg.set_gravity_mode_pcg()
    # cfg.set_gravity_mode_bicgstab()
    cfg.set_self_gravity_G_values(True, 1.0)
    cfg.set_self_gravity_Niter_max(1500)
    cfg.set_self_gravity_tol(1e-6)
    cfg.set_self_gravity_happy_breakdown_tol(1e-6)
    cfg.set_coupling_gravity_mode_ramses_like()

    model.set_solver_config(cfg)
    model.init_scheduler(int(4000000), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    def rho_map(rmin, rmax):
        x_mn, y_mn, z_mn = rmin
        x_mx, y_mx, z_mx = rmax

        x = 0.5 * (x_mn + x_mx)
        y = 0.5 * (y_mn + y_mx)
        z = 0.5 * (z_mn + z_mx)

        res = (
            np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z)
            + 0.5 * np.cos(4 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z)
            + (1 / 3) * np.cos(2 * np.pi * x) * np.cos(4 * np.pi * y) * np.cos(6 * np.pi * z)
        )

        return 2 + res

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
    # dt = 0.01226171192153859
    t = 0
    tend = 0.245
    Max_iter = 1

    for k in range(Max_iter):
        # if k % freq == 0:
        #     model.dump_vtk("test" + str(k) + ".vtk")

        next_dt = model.evolve_once_override_time(t, dt)

        t += dt
        dt = next_dt

        if tend < t + next_dt:
            dt = tend - t

        dic = ctx.collect_data()

        if (shamrock.sys.world_rank() == 0) and (k == Max_iter - 1):
            dic = convert_to_cell_coords(dic)
            tmp = dic["rho"] * (4.0 * np.pi * G)

            for i in range(len(dic["xmin"])):

                X.append(0.5 * (dic["xmin"][i] + dic["xmax"][i]))
                Y.append(0.5 * (dic["ymin"][i] + dic["ymax"][i]))
                Z.append(0.5 * (dic["zmin"][i] + dic["zmax"][i]))

                rho.append(dic["rho"][i])
                phi.append(dic["phi"][i])
                phi_ana.append(tmp[i])

        if t >= tend:
            break


X = []
Y = []
Z = []
rho = []
phi = []
phi_ana = []


run_sim(X, Y, Z, rho, phi, phi_ana, A=1)


# =============Analytical phi ============
def analytic_phi(X, Y, Z, Lx, Ly, Lz, G, A=1, phi_0=0):
    res = -(G / np.pi) * (
        (1 / 3) * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.cos(2 * np.pi * Z)
        + (1 / 12) * np.cos(4 * np.pi * X) * np.cos(2 * np.pi * Y) * np.cos(2 * np.pi * Z)
        + (1 / 42) * np.cos(2 * np.pi * X) * np.cos(4 * np.pi * Y) * np.cos(6 * np.pi * Z)
    )

    return res


# ============ L2 DIFF ============
def l2_diff(f1, f2):
    return np.sqrt(np.sum((f1 - f2) ** 2)) / f1.size


# ============ L1 DIFF ============
def l1_diff(f1, f2):
    return np.sum(np.abs(f1 - f2)) / f1.size


# ============ LINF DIFF ============
def linf_diff(f1, f2):
    return np.max(np.abs(f1 - f2))


# ============ L2 Norm ============
def l2_(f):
    return np.sqrt(np.sum((f) ** 2)) / f.size


# ============ L1 Norm ============
def l1_(f):
    return np.sum(np.abs(f)) / f.size


# ============ LINF Norm ============
def linf_(f):
    return np.max(np.abs(f))


ana = analytic_phi(np.array(X), np.array(Y), np.array(Z), Lx=1, Ly=1, Lz=1, G=1)
# print("===================================")
# for i in range(len(X)):
#     print(f"[{i}]: {ana[i]} -- {phi_ana[i]} -- {phi[i]} \n")
# print("===================================")

diff = np.array(phi) - ana
l1_dif = l1_diff(np.array(phi), ana)
l2_dif = l2_diff(np.array(phi), ana)
linf_dif = linf_diff(np.array(phi), ana)

l1 = l1_diff(np.array(phi), ana) / l1_(ana)
l2 = l2_diff(np.array(phi), ana) / l2_(ana)
linf = linf_diff(np.array(phi), ana) / linf_(ana)

print("============= Errors to analalytical solution ===========")
print(f"L1-NORM : {l1}  , \t L2-NORM : {l2}   , \t LINF-NORM : {linf} \n")


plt.plot(np.array(X), ana, ".", label="phi-ana")
plt.plot(np.array(X), phi, "+", label="phi-num")
plt.plot(np.array(X), diff, "*", label="diff")
plt.legend()
plt.show()
plt.savefig(f"Poisson-solver-convergence-test-{len(X)}.png", format="png")
