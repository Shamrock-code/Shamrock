import matplotlib.pyplot as plt

import shamrock

shamrock.enable_experimental_features()
import numpy as np

rho = 1
epsilon_0 = 0.5
cs_g = 1
ts = 1

delta_rho_0 = 0.01
delta_v_0 = 0.01

bmin = (-0.6, -0.6 / 4, -0.6 / 4)
bmax = (0.6, 0.6 / 4, 0.6 / 4)

N_target = 1e3
scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)


def func_rho_t(r):
    return rho


def func_rho_d(r):
    return func_rho_t(r) * 0.5


def func_rho_g(r):
    return func_rho_t(r) - func_rho_d(r)


def func_s(r):
    rho_t = func_rho_t(r)
    rho_d = func_rho_d(r)
    eps = rho_d / rho_t
    return np.sqrt(rho_t * eps)


xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

part_vol = vol_b / N_target

# lattice volume
HCP_PACKING_DENSITY = 0.74
part_vol_lattice = HCP_PACKING_DENSITY * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)

pmass = -1

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_dust_mode_monofluid_tvi(1)
cfg.set_dust_drag_constant([ts])
cfg.set_boundary_periodic()
cfg.set_eos_isothermal(cs_g)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8), 1)


bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
setup.apply_setup(gen, insert_step=scheduler_split_val)


def vel_func(r):
    global mm, MM
    x, y, z = r

    f = 2 * np.pi / (xM - xm)

    vel = delta_v_0 * np.sin(x * f)

    return (vel, 0.0, 0.0)


model.set_field_value_lambda_f64("s_j", func_s, 0)
model.set_field_value_lambda_f64_3("vxyz", vel_func)

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho * vol_b

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

model.timestep()

t_list = []
rho_g_ampl_list = []
rho_d_ampl_list = []
rho_g_phi_list = []
rho_d_phi_list = []
rho_g_offset_list = []
rho_d_offset_list = []

i = 0
while model.get_time() < 2:
    ################################
    # x field
    ################################
    def custom_getter_x(size: int, dic_out: dict) -> np.array:
        return dic_out["xyz"][:, 0]

    x_field = model.compute_field("custom", "f64", custom_getter_x)

    ################################
    # x field
    ################################
    def custom_getter_vx(size: int, dic_out: dict) -> np.array:
        return dic_out["vxyz"][:, 0]

    vx_field = model.compute_field("custom", "f64", custom_getter_vx)

    ################################
    # rho field
    ################################
    pmass = model.get_particle_mass()
    hfact = model.get_hfact()

    def internal_rho(size: int, h: np.array) -> np.array:
        return pmass * (hfact / h) ** 3

    def custom_getter_rho(size: int, dic_out: dict) -> np.array:
        return internal_rho(size, dic_out["hpart"])

    rho_field = model.compute_field("custom", "f64", custom_getter_rho)

    ################################
    # rho_(g+d) field
    ################################
    def internal_eps(size: int, s: np.array, rho: np.array) -> np.array:
        return (s**2) / rho

    def custom_getter_rho_g(size: int, dic_out: dict) -> np.array:
        rho = internal_rho(size, dic_out["hpart"])
        eps = internal_eps(size, dic_out["s_j"], rho)
        return (1 - eps) * rho

    def custom_getter_rho_d(size: int, dic_out: dict) -> np.array:
        rho = internal_rho(size, dic_out["hpart"])
        eps = internal_eps(size, dic_out["s_j"], rho)
        return eps * rho

    rho_g_field = model.compute_field("custom", "f64", custom_getter_rho_g)
    rho_d_field = model.compute_field("custom", "f64", custom_getter_rho_d)

    x_data = np.asarray(x_field.collect_data())
    rho_g_data = np.asarray(rho_g_field.collect_data())
    rho_d_data = np.asarray(rho_d_field.collect_data())

    from scipy.linalg import lstsq

    k = 2 * np.pi / (xM - xm)

    def fit_sine_wave(x, y):
        design = np.column_stack([np.ones_like(x), np.sin(k * x), np.cos(k * x)])
        offset, a, b = lstsq(design, y)[0]
        ampl = np.hypot(a, b)
        phi = np.arctan2(b, a)
        if phi < 0:
            phi += np.pi
            ampl = -ampl
        return offset, ampl, phi

    def sine_model(x, offset, ampl, phi):
        return offset + ampl * np.sin(phi + k * x)

    offset_g, ampl_g, phi_g = fit_sine_wave(x_data, rho_g_data)
    offset_d, ampl_d, phi_d = fit_sine_wave(x_data, rho_d_data)

    if model.get_time() == 0:
        phi_g = np.pi / 2
        phi_d = np.pi / 2

    x_fit = np.linspace(xm, xM, 256)
    rho_g_fit = sine_model(x_fit, offset_g, ampl_g, phi_g)
    rho_d_fit = sine_model(x_fit, offset_d, ampl_d, phi_d)

    t_list.append(model.get_time())
    rho_g_ampl_list.append(ampl_g)
    rho_d_ampl_list.append(ampl_d)
    rho_g_phi_list.append(phi_g)
    rho_d_phi_list.append(phi_d)
    rho_g_offset_list.append(offset_g)
    rho_d_offset_list.append(offset_d)

    print(f"rho_g fit: offset={offset_g:.6g}, ampl={ampl_g:.6g}, phi={phi_g:.6g} rad")
    print(f"rho_d fit: offset={offset_d:.6g}, ampl={ampl_d:.6g}, phi={phi_d:.6g} rad")

    plt.plot(x_data, rho_g_data, ".", label="rho_g")
    plt.plot(x_data, rho_d_data, ".", label="rho_d")
    plt.plot(
        x_fit,
        rho_g_fit,
        "-",
        label=rf"rho_g fit: $A={ampl_g:.3g}$, $\phi={phi_g:.3g}$",
    )
    plt.plot(
        x_fit,
        rho_d_fit,
        "-",
        label=rf"rho_d fit: $A={ampl_d:.3g}$, $\phi={phi_d:.3g}$",
    )
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.close()

    model.evolve_until(model.get_time() + 1e-1)

    model.do_vtk_dump(f"dump_dustywave_tvi_{i:04}.vtk", False)
    i += 1

plt.figure(dpi=150)
plt.subplot(2, 1, 1)
plt.plot(t_list, rho_g_ampl_list, label="rho_g_ampl")
plt.plot(t_list, rho_d_ampl_list, label="rho_d_ampl")
plt.subplot(2, 1, 2)
plt.plot(t_list, rho_g_phi_list, label="rho_g_phi")
plt.plot(t_list, rho_d_phi_list, label="rho_d_phi")
plt.show()
