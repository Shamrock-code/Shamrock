import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


def get_mass(R, rho):
    return rho * (4.0 * np.pi / 3.0) * (R**3)


si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=1,  # [s]
    unit_length=1,  # [m]
    unit_mass=1,  # [Kg]
)
ucte = shamrock.Constants(codeu)
G = ucte.G()
kb = ucte.kb()

T0 = 10  # [K]
R0 = 7.07e16 * 1e-2  # [cm -> m]

rho0 = 1.38e-18 * 1e3  # [g/cm^3 -> kg/m^3]
M0 = get_mass(R0, rho0)  # [kg]
mu = 2.3  # molecular gas
m_H = 1.6735e-27  # [kg]
E_th0 = (3.0 * M0 * kb * T0) / (2 * mu * m_H)  # [J]
E_grav0 = (-3.0 * G * M0**2) / (5.0 * R0)  # [J]
alpha0 = E_th0 / np.abs(E_grav0)

t_ff = np.sqrt((3.0 * np.pi) / (32.0 * G * rho0))  # [s]
cs_sqr = (kb * T0) / (mu * m_H)
lamb_J = np.sqrt((cs_sqr * np.pi) / (G * rho0))  # [m]
print(f"Jeans length = {lamb_J}\n")
print(f"sound speed  = {np.sqrt(cs_sqr)}\n")
print(f"alpha = {alpha0}\n")
print(f"free fall time = {t_ff / (3600 * 24 * 365)} years \n")
N_J = 10  # N_J points per Jeans length
L0 = 4 * R0  # [m]
min_reso = (L0 * N_J) / (lamb_J)
print(f"min reso = {min_reso}\n")
gamma = 5.0 / 3.0


def run_sim():

    shamrock.enable_experimental_features()
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 32

    cfg = model.gen_default_config()
    scale_fact = L0 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)

    cfg.set_Csafe(0.8)
    cfg.set_eos_gamma(gamma)
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(True)
    cfg.set_gravity_mode_cg()
    cfg.set_riemann_solver_hllc()

    cfg.set_self_gravity_G_values(True, 1.0)
    cfg.set_self_gravity_Niter_max(100)
    cfg.set_self_gravity_tol(1e-6)
    cfg.set_coupling_gravity_mode_ramses_like()

    model.set_solver_config(cfg)
    model.init_scheduler(int(50000), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    ### Gas maps
    def rho_map(rmin, rmax) -> float:
        x_mn, y_mn, z_mn = rmin
        x_mx, y_mx, z_mx = rmax

        x = 0.5 * (x_mn + x_mx) - 0.5 * L0
        y = 0.5 * (y_mn + y_mx) - 0.5 * L0
        z = 0.5 * (z_mn + z_mx) - 0.5 * L0
        r = np.sqrt(x**2 + y**2 + z**2)

        if r < R0:
            return rho0
        else:
            return rho0 / 100

    def rhovel_map(rmin, rmax):
        return (0.0, 0.0, 0.0)

    def rhoe_map(rmin, rmax):
        rho = rho_map(rmin, rmax)
        # P = cs_sqr * rho
        return cs_sqr / (gamma - 1.0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    tmax = 1.5 * t_ff
    t = 0
    dt = 0
    freq = 1
    dX0 = []
    for i in range(100):
        next_dt = model.evolve_once_override_time(t, dt)

        t += dt
        dt = next_dt

        if i % freq == 0:
            model.dump_vtk(f"sphe_collapse{i:04d}.vtk")

        if tmax < t + next_dt:
            dt = tmax - t
        if t == tmax:
            model.dump_vtk(f"sphe_collapse{i:04d}.vtk")
            break
