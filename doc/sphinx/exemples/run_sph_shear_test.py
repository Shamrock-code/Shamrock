"""
Shearing box in SPH
========================

This simple example shows how to run an unstratified shearing box simulaiton
"""

# sphinx_gallery_multi_image = "single"

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

from math import exp

import matplotlib.pyplot as plt
import numpy as np

# %%
# Initialize context & attach a SPH model to it
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")


# %%
# Setup parameters
gamma = 5.0 / 3.0
rho = 1
uint = 1

dr = 0.02
bmin = (-0.6, -0.6, -0.1)
bmax = (0.6, 0.6, 0.1)
pmass = -1

bmin, bmax = model.get_ideal_fcc_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

Omega_0 = 1
eta = 0.00
q = 3.0 / 2.0

shear_speed = -q * Omega_0 * (xM - xm)


# %%
# Generate the config & init the scheduler
cfg = model.gen_default_config()
# cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_shearing_periodic((1, 0, 0), (0, 1, 0), shear_speed)
cfg.set_eos_adiabatic(gamma)
cfg.add_ext_force_shearing_box(Omega_0=Omega_0, eta=eta, q=q)
cfg.set_units(shamrock.UnitSystem())
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)

model.resize_simulation_box(bmin, bmax)


# %%
# Add the particles & set fields values
# Note that every field that are not mentionned are set to zero
model.add_cube_fcc_3d(dr, bmin, bmax)

vol_b = (xM - xm) * (yM - ym) * (zM - zm)

totmass = rho * vol_b
# print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint", "f64", 1, bmin, bmax)
# model.set_value_in_a_box("vxyz","f64_3", (-10,0,0) , bmin,bmax)

pen_sz = 0.1

mm = 1
MM = 0


def vel_func(r):
    global mm, MM
    x, y, z = r

    s = (x - (xM + xm) / 2) / (xM - xm)
    vel = (shear_speed) * s

    mm = min(mm, vel)
    MM = max(MM, vel)

    return (0, vel, 0.0)
    # return (1,0,0)


model.set_field_value_lambda_f64_3("vxyz", vel_func)
# print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass * model.get_sum("uint", "f64")
# print("total u :",tot_u)

print(f"v_shear = {shear_speed} | dv = {MM - mm}")


model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

# %%
dump_folder = "_to_trash"
import os

os.system("mkdir -p " + dump_folder)


# %%
# Perform the plot
def plot():
    dic = ctx.collect_data()
    fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
    fig.suptitle("t = {:.2f}".format(model.get_time()))
    axs[0].scatter(dic["xyz"][:, 0], dic["xyz"][:, 1], s=1)
    axs[1].scatter(dic["xyz"][:, 0], dic["vxyz"][:, 1], s=1)

    axs[0].set_ylabel("y")
    axs[1].set_ylabel("vy")
    axs[1].set_xlabel("x")

    plt.tight_layout()
    plt.show()


# %%
# Performing the timestep loop
model.timestep()

dt_stop = 0.1
for i in range(2):
    t_target = i * dt_stop
    # skip if the model is already past the target
    if model.get_time() > t_target:
        continue

    model.evolve_until(i * dt_stop)

    # Dump name is "dump_xxxx.sham" where xxxx is the timestep
    model.do_vtk_dump(dump_folder + "/dump_{:04}.vtk".format(i), True)
    plot()
