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
m_H = ucte.proton_mass()  # [kg]


# T0 = 10.0  # [K]
T0 = 10.747
R0 = 7.07e16 * 1e-2  # [cm -> m]

rho0 = 1.38e-18 * 1e3  # [g/cm^3 -> kg/m^3]
M0 = get_mass(R0, rho0)  # [kg]
mu = 2.3  # molecular gas
# m_H = 1.6735e-27  # [kg]

print(f"proton-mass = {m_H} \n")
E_th0 = (3.0 * M0 * kb * T0) / (2 * mu * m_H)  # [J]
E_grav0 = (-3.0 * G * M0**2) / (5.0 * R0)  # [J]
alpha0 = E_th0 / np.abs(E_grav0)

t_ff = np.sqrt((3.0 * np.pi) / (32.0 * G * rho0))  # [s]
cs_sqr = (kb * T0) / (mu * m_H)
lamb_J = np.sqrt((cs_sqr * np.pi) / (G * rho0))  # [m]
print(f"kb = {kb}\n")
print(f"G value from set-up = {G}\n")
print(f"Jeans length = {lamb_J}\n")
print(f"sound speed  = {np.sqrt(cs_sqr)}\n")
print(f"alpha = {alpha0}\n")
print(f"ss = {(3600 * 24 * 365)}\n")
print(f"free fall time = {t_ff / (3600 * 24 * 365)} years \n")
N_J = 16  # N_J points per Jeans length
L0 = 4 * R0  # [m]
min_reso = (L0 * N_J) / (lamb_J)
print(f"min reso = {min_reso}\n")
gamma = 5.0 / 3.0

rho_c = 2.7e-11 * 1e3 # [g/cm^3 -> kg/m^3]


def run_sim():

    shamrock.enable_experimental_features()
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    max_amr_lev = 16
    sz = 2 << max_amr_lev

    base = 32

    cfg = model.gen_default_config()
    scale_fact = L0 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)

    cfg.set_Csafe(0.3)
    cfg.set_eos_gamma(gamma)
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(True)
    ########

    cfg.set_gravity_mode_cg()

    ####
    # cfg.set_gravity_mode_bicgstab()
    cfg.set_riemann_solver_hllc()

    cfg.set_self_gravity_G_values(True, G)
    cfg.set_self_gravity_Niter_max(200)
    cfg.set_self_gravity_tol(1e-6)
    cfg.set_coupling_gravity_mode_ramses_like()

    # err_min = 0.25
    # err_max = 0.10
    # cfg.set_amr_mode_pseudo_gradient_based(error_min=err_min, error_max=err_max)
    cfg.set_amr_mode_jeans_length_based(N_jeans=N_J, T_init=T0)

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
        rhov = rhovel_map(rmin, rmax)
        Ekin = 0.5 * (rhov[0]**2)/rho
        x = rho / rho_c
        P = cs_sqr * rho * (1. + x**(2./3.))
        Eint = P / (gamma - 1.0) 
        return  Ekin + Eint

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    tmax = 1.012 * t_ff
    t = 0
    dt = 0
    freq = 50
    dX0 = []
    for i in range(int(1e7)):
        next_dt = model.evolve_once_override_time(t, dt)

        t += dt
        dt = next_dt

        if i % freq == 0:
            model.dump_vtk(f"_iso_collapse_{t/t_ff:5f}.vtk")

        if tmax < t + next_dt:
            dt = tmax - t
        if t == tmax:
            model.dump_vtk(f"_iso_collapse{t/t_ff:5f}.vtk")
            break


run_sim()
