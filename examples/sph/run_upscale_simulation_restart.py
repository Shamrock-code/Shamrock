"""
Init a simulation from an upscaled simulation
=============================================
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

bmin = (-0.6, -0.6, -0.6)
bmax = (0.6, 0.6, 0.6)

N_target = 1e4
scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)

# %%
# Deduced quantities
import numpy as np

xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

part_vol = vol_b / N_target

# lattice volume
HCP_PACKING_DENSITY = 0.74
part_vol_lattice = HCP_PACKING_DENSITY * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)

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

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint", "f64", 0, bmin, bmax)

tot_u = pmass * model.get_sum("uint", "f64")
if shamrock.sys.world_rank() == 0:
    print("total u :", tot_u)

model.set_particle_mass(pmass)

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)


# %%
# Single timestep to iterate the smoothing length
model.timestep()
model.dump("init.sham")

# here we can dump and load it into another context i we want like so
ctx_data_source = shamrock.Context()
ctx_data_source.pdata_layout_new()
model_data_source = shamrock.get_Model_SPH(
    context=ctx_data_source, vector_type="f64_3", sph_kernel="M4"
)
model_data_source.load_from_dump("init.sham")

# trigger rebalancing
model_data_source.set_dt(0.0)
model_data_source.timestep()

# reset dt to 0 for the init of the next simulation
model_data_source.set_dt(0.0)

cfg = model_data_source.get_current_config()
cfg.print_status()

# now we feed the old context to the new model
ctx_new = shamrock.Context()
ctx_new.pdata_layout_new()

model_new = shamrock.get_Model_SPH(context=ctx_new, vector_type="f64_3", sph_kernel="M4")
model_new.set_solver_config(cfg)
model_new.init_scheduler(scheduler_split_val, scheduler_merge_val)
model_new.resize_simulation_box(bmin, bmax)

setup = model_new.get_setup()
gen = setup.make_generator_from_context(ctx_data_source)
split_part = setup.make_modifier_split_part(parent=gen, n_split=2, seed=42)
setup.apply_setup(split_part, insert_step=scheduler_split_val)

model_new.timestep()


# %%
# Recover data
dat = ctx_new.collect_data()

# %%
# Test h value
import numpy as np

min_hpart = np.min(dat["hpart"])
max_hpart = np.max(dat["hpart"])
mean_hpart = np.mean(dat["hpart"])

print(f"hpart min={min_hpart} max={max_hpart} delta={max_hpart - min_hpart}")

assert min_hpart < 0.04, "hpart min is too large"


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
    c=dat["hpart"],
    cmap=cm,
)
plt.colorbar(sc)
plt.show()
