"""
Uniform box in SPH
==================

This simple example shows a uniform density box in SPH, it is also used to test that the
smoothing length iteration find the correct value
"""

# sphinx_gallery_multi_image = "single"

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Setup parameters

gamma = 5.0 / 3.0
rho_g = 1
target_tot_u = 1

bmin = (-0.6, -0.6, -0.6)
bmax = (0.6, 0.6, 0.6)

N_target = 1e4
scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)


# %%
# Deduced quantities


xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

part_vol = vol_b / N_target

# lattice volume
part_vol_lattice = 0.74 * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * 3.1416)) ** (1.0 / 3.0)

pmass = -1

# %%
# Setup

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(scheduler_split_val, scheduler_merge_val)


bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
setup.apply_setup(gen, insert_step=scheduler_split_val)


xc, yc, zc = model.get_closest_part_to((0, 0, 0))

if shamrock.sys.world_rank() == 0:
    print("closest part to (0,0,0) is in :", xc, yc, zc)


vol_b = (xM - xm) * (yM - ym) * (zM - zm)

totmass = rho_g * vol_b
# print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint", "f64", 0, bmin, bmax)

tot_u = pmass * model.get_sum("uint", "f64")
if shamrock.sys.world_rank() == 0:
    print("total u :", tot_u)


model.set_particle_mass(pmass)


tot_u = pmass * model.get_sum("uint", "f64")

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

# %%
# Single timestep to iterate the smoothing length
model.timestep()

# %%
# Recover data
dat = ctx.collect_data()

# %%
# Test h value
import numpy as np

min_hpart = np.min(dat["hpart"])
max_hpart = np.max(dat["hpart"])
mean_hpart = np.mean(dat["hpart"])

print(f"hpart min={min_hpart} max={max_hpart} delta={max_hpart-min_hpart}")

assert np.abs(max_hpart - min_hpart) < 1e-9, "hpart delta is too large"

expected_h = 0.06688944619517681

assert np.abs(min_hpart - expected_h) < 1e-9, "hpart is off the expected value"

# %%
# Plot particle distrib
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(dpi=120)
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim3d(bmin[0], bmax[0])
ax.set_ylim3d(bmin[1], bmax[1])
ax.set_zlim3d(bmin[2], bmax[2])

cm = matplotlib.colormaps["viridis"]
sc = ax.scatter(
    dat["xyz"][:, 0],
    dat["xyz"][:, 1],
    dat["xyz"][:, 2],
    s=1,
    vmin=mean_hpart - 1e-10,
    vmax=mean_hpart + 1e-10,
    c=dat["hpart"],
    cmap=cm,
)
plt.colorbar(sc)
plt.show()
