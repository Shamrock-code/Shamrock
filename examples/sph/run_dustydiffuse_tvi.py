import matplotlib.pyplot as plt

import shamrock

shamrock.enable_experimental_features()
import numpy as np

rho = 1
epsilon_0 = 0.1
cs_g = 1
ts = 0.1
rc = 0.25


bmin = (-0.5, -0.5, -0.5)
bmax = (0.5, 0.5, 0.5)

N_target = 1e4


def func_rho_t(r):
    return rho


def func_eps(pos):
    r = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
    return epsilon_0 * max(0, 1 - (r / rc) ** 2)


def func_s(r):
    rho_t = func_rho_t(r)
    eps = func_eps(r)
    return np.sqrt(rho_t * eps)


def get_field_results(model):
    ################################
    # r field
    ################################
    def custom_getter_r(size: int, dic_out: dict) -> np.array:
        return np.sqrt(
            dic_out["xyz"][:, 0] ** 2 + dic_out["xyz"][:, 1] ** 2 + dic_out["xyz"][:, 2] ** 2
        )

    r_field = model.compute_field("custom", "f64", custom_getter_r)

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

    r_data = np.asarray(r_field.collect_data())
    rho_data = np.asarray(rho_field.collect_data())
    rho_g_data = np.asarray(rho_g_field.collect_data())
    rho_d_data = np.asarray(rho_d_field.collect_data())

    return r_data, rho_data, rho_g_data, rho_d_data


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
cfg.set_dust_mode_monofluid_tvi(nvar=1, pure_diffusion_mode=True)
cfg.set_dust_drag_constant([ts])
cfg.set_boundary_periodic()
cfg.set_eos_isothermal(cs_g)
cfg.print_status()
model.set_solver_config(cfg)

scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)

model.init_scheduler(scheduler_split_val, scheduler_merge_val)


bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
setup.apply_setup(gen, insert_step=scheduler_split_val)


model.set_field_value_lambda_f64("s_j", func_s, 0)

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho * vol_b

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

model.timestep()

t_list = [0.0, 0.1, 0.3, 1, 3, 10]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

i = 0
for tst in t_list:
    model.evolve_until(tst)
    r_data, rho_data, rho_g_data, rho_d_data = get_field_results(model)

    eps = rho_d_data / rho_data

    axs[0].plot(r_data, rho_data, ".", label="rho")
    axs[0].plot(r_data, rho_g_data, ".", label="rho_g")
    axs[0].plot(r_data, rho_d_data, ".", label="rho_d")
    axs[0].set_xlabel("r")
    axs[0].set_ylabel("density")
    axs[0].set_xlim(0, 0.5)
    axs[0].set_ylim(0, 1.1)
    axs[0].legend()
    axs[1].plot(r_data, eps, ".", label="eps")
    axs[1].set_xlabel("r")
    axs[1].set_ylabel("epsilon")
    axs[1].set_xlim(0, 0.5)
    axs[1].set_ylim(0, 0.11)
    axs[1].legend()
    plt.suptitle(f"t = {model.get_time():.2f}")
    plt.tight_layout()
    plt.savefig(f"dump_dustydiffuse_tvi_{i:04}.png")
    # plt.close()

    model.do_vtk_dump(f"dump_dustydiffuse_tvi_{i:04}.vtk", False)
    i += 1
