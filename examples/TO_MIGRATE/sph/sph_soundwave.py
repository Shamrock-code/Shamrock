import matplotlib.pyplot as plt
import numpy as np

import shamrock

gamma = 5.0 / 3.0
rho_g = 1
target_tot_u = 1


dr = 0.01

bmin = (-0.6, -0.6, -0.6)
bmax = (0.6, 0.6, 0.6)
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
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e6), 1)


bmin, bmax = model.get_ideal_fcc_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

model.resize_simulation_box(bmin, bmax)
model.add_cube_fcc_3d(dr, bmin, bmax)

vol_b = (xM - xm) * (yM - ym) * (zM - zm)

totmass = rho_g * vol_b
# print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)


model.set_value_in_a_box("uint", "f64", 0.9, bmin, bmax)

kx, ky, kz = 2 * np.pi / (xM - xm), 0, 0
delta_v = 1e-5


def vel_func(r):
    x, y, z = r

    return (0 + delta_v * np.cos(kx * x + ky * y + kz * z), 0, 0)


model.set_field_value_lambda_f64_3("vxyz", vel_func)


# print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass * model.get_sum("uint", "f64")
# print("total u :",tot_u)

# a = input("continue ?")


model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)


# for i in range(9):
#    model.evolve(5e-4, False, False, "", False)


t_sum = 0
t_target = 2

i_dump = 0
dt_dump = 1e-1
next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target:
    fname = "dump_{:04}.phfile".format(i_dump)

    model.evolve_until(next_dt_target)
    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    i_dump += 1

    next_dt_target += dt_dump


import numpy as np

dic = ctx.collect_data()

r = np.sqrt(dic["xyz"][:, 0] ** 2 + dic["xyz"][:, 1] ** 2 + dic["xyz"][:, 2] ** 2)
vr = np.sqrt(dic["vxyz"][:, 0] ** 2 + dic["vxyz"][:, 1] ** 2 + dic["vxyz"][:, 2] ** 2)


hpart = dic["hpart"]
uint = dic["uint"]

gamma = 5.0 / 3.0

rho = pmass * (model.get_hfact() / hpart) ** 3
P = (gamma - 1) * rho * uint


plt.style.use("custom_style.mplstyle")
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 6), dpi=125)

axs[0, 0].scatter(r, vr, c="black", s=1, label="v")
axs[1, 0].scatter(r, uint, c="black", s=1, label="u")
axs[0, 1].scatter(r, rho, c="black", s=1, label="rho")
axs[1, 1].scatter(r, P, c="black", s=1, label="P")


axs[0, 0].set_ylabel(r"$v$")
axs[1, 0].set_ylabel(r"$u$")
axs[0, 1].set_ylabel(r"$\rho$")
axs[1, 1].set_ylabel(r"$P$")

axs[0, 0].set_xlabel("$r$")
axs[1, 0].set_xlabel("$r$")
axs[0, 1].set_xlabel("$r$")
axs[1, 1].set_xlabel("$r$")

axs[0, 0].set_xlim(0, 0.55)
axs[1, 0].set_xlim(0, 0.55)
axs[0, 1].set_xlim(0, 0.55)
axs[1, 1].set_xlim(0, 0.55)

plt.tight_layout()
plt.show()
