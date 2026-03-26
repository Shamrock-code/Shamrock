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
rho_i = 1
central_mass = 1
R0 = 1
H_r_0 = 0.05

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()


def kep_profile(r):
    return (G * central_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


H = H_r_0 * R0
cs = H * omega_k(R0)
box = 8 * H

print(f"cs = {cs}")
print(f"H = {H}")


def scaling_rho(r):
    x, y, z = r

    loc_h = H / (2**0.5)
    gaussian = np.exp(-(y**2) / (2 * loc_h * loc_h)) / (loc_h * np.sqrt(2 * np.pi))
    return gaussian


def func_rho_t(r):
    return rho_i * scaling_rho(r)


def func_rho_d_j(r, idust):
    return (0.1 / ndust) * rho_i * scaling_rho(r)


def func_rho_g(r):
    return rho_i * scaling_rho(r) - sum([func_rho_d_j(r, i) for i in range(ndust)])


cs_g = 1


def uint_g(r):
    rho_g = func_rho_g(r)
    P = rho_g * cs_g * cs_g / gamma
    return P / ((gamma - 1) * rho_g)


ndust = 4
rc = 0.25
stokes = np.logspace(-3, 0, ndust)
stopping_times = stokes / omega_k(R0) 
print(stopping_times, omega_k(R0))
from scipy.special import erfinv

bmin = (-box / 4, -box, -box / 4)
bmax = (box / 4, box, box / 4)

N_target = 1e5
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
cfg.add_ext_force_vertical_disc_potential(central_mass=1, R0=1)
cfg.add_ext_force_velocity_dissipation(eta=10)
cfg.set_boundary_periodic()
cfg.set_units(codeu)
cfg.set_eos_isothermal(cs)
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

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho_i * vol_b
print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)

# Correct the barycenter
analysis_barycenter = shamrock.model_sph.analysisBarycenter(model=model)
barycenter, disc_mass = analysis_barycenter.get_barycenter()

if shamrock.sys.world_rank() == 0:
    print(f"disc barycenter = {barycenter}")

model.apply_position_offset((-barycenter[0], -barycenter[1], -barycenter[2]))


def f_remap(r):
    x, y, z = r

    rn = max(abs(yM), abs(ym))
    print(y, H, H * erfinv(y / rn))
    y = H * erfinv(y / rn)

    y = min(y, yM)
    y = max(y, ym)
    return (x, y, z)


model.remap_positions(f_remap)
model.set_field_value_lambda_f64("uint", uint_g)


model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

model.timestep()

def compute_sj_new(patchdata):
    hpart = patchdata["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    epsilon_target = 0.1 / ndust
    s = np.sqrt(rho * epsilon_target)

    return s






# TODO: add function to modify fields e.g. get rho and do stuff according to it

tnext = 0
for j in range(1000):
    if j > 0:
        tnext += 0.1
        model.evolve_until(tnext)

        if(j == 20):
            for k in range(ndust):
                model.overwrite_field_value_f64("s_j",compute_sj_new,k)

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

    r = y

    hpart = dic["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    print("compute original rho")
    estimated_rho = [func_rho_t(dic["xyz"][kk]) for kk in range(len(dic["xyz"]))]

    sz = 2

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    time = model.get_time()
    fig.suptitle(f"t = {time:.2f}")

    fig.subplots_adjust(left=0.07, right=0.98, wspace=0.4)

    axs[0].scatter(y, rho, label="gas", s=sz)
    for i in range(ndust):
        axs[0].scatter(y, s_j[:, i] ** 2, label=f"dust {i} ts = {stopping_times[i]:.2f}", s=sz)
    # axs[0].scatter(y,estimated_rho)
    axs[0].set_ylabel(r"$\rho$")
    axs[0].set_xlabel(r"$y$")

    axs[0].set_yscale('log')
    axs[0].legend()
    for i in range(ndust):
        axs[1].scatter(y, s_j[:, i] ** 2 / rho, label=f"dust {i} ts = {stopping_times[i]:.2f}", s=sz)
    axs[1].set_ylabel(r"$\epsilon_j$")
    axs[1].set_xlabel(r"$y$")
    axs[1].legend()

    for i in range(ndust):
        axs[2].scatter(y, ds_j_dt[:, i], label=f"dust {i} ts = {stopping_times[i]:.2f}", s=sz)
    axs[2].set_ylabel(r"$\frac{d s_j}{dt}$")
    axs[2].set_xlabel(r"$y$")
    axs[2].legend()

    plt.savefig(f"mono_{j}.png")
    model.do_vtk_dump(f"dump_stratif_{j}.vtk", True)
    plt.close()
