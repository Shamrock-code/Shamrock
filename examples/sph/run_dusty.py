"""
Testing Sod tube with SPH
=========================

CI test for Sod tube with SPH
"""

import matplotlib.pyplot as plt

import shamrock

shamrock.enable_experimental_features()
import numpy as np

gamma = 1.4
rho = 1


def func_rho_t(r):
    return rho


def func_rho_d_j(r, idust):
    r_ = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    return (0.1 / ndust) * max(1 - (r_ / rc) ** 2, 0)


def func_rho_g(r):
    return rho - sum([func_rho_d_j(r, i) for i in range(ndust)])


def func_s_j(r, idust):
    rho_t = func_rho_t(r)
    rho_d_j = [func_rho_d_j(r, i) for i in range(ndust)]
    eps_j = rho_d_j[idust] / rho_t
    return np.sqrt(rho_t * eps_j)


cs_g = 1


def uint_g(r):
    rho_g = func_rho_g(r)
    P = rho_g * cs_g * cs_g / gamma
    return P / ((gamma - 1) * rho_g)


ndust = 1
rc = 0.25
stopping_times = [1e-1]


bmin = (-0.6, -0.6, -0.6)
bmax = (0.6, 0.6, 0.6)

N_target = 1e4
scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)

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
cfg.set_dust_mode_monofluid_tvi(ndust)
cfg.set_dust_stopping_times(stopping_times)
cfg.set_boundary_periodic()
cfg.set_eos_isothermal(1)
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

for i in range(ndust):

    def func_s(r):
        return func_s_j(r, i)

    model.set_field_value_lambda_f64("s_j", func_s, i)

model.set_field_value_lambda_f64("uint", uint_g)

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho * vol_b
print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)


model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

model.timestep()

tnext = 0
for j in range(10):
    if j > 0:
        tnext += 0.1
        model.evolve_until(tnext)

    dic = ctx.collect_data()
    print(dic["s_j"])

    print(dic["xyz"].shape)

    x = dic["xyz"][:, 0]
    y = dic["xyz"][:, 1]
    z = dic["xyz"][:, 2]
    s_j = dic["s_j"].reshape(-1, ndust)
    ds_j_dt = dic["ds_j_dt"].reshape(-1, ndust)
    cs = dic["soundspeed"]

    print(s_j)

    r = np.sqrt(x * x + y * y + z * z)

    hpart = dic["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
    axs[0].scatter(r, rho, label="rho")
    axs[0].legend()
    for i in range(ndust):
        axs[1].scatter(r, s_j[:, i] ** 2 / rho, label=f"eps_j_{i}")
    axs[1].legend()

    axs[2].scatter(r, cs, label="soundspeed")
    axs[2].legend()

    for i in range(ndust):
        axs[3].scatter(r, ds_j_dt[:, i], label=f"ds_j_dt_{i}")
    axs[3].legend()

    for k in range(4):
        axs[k].set_xlim(-0.1, 1.1)
    plt.savefig(f"mono_{j}.png")
    plt.close()
