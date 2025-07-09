"""
Tracking particles by id in SPH
===============================

This simple example shows how to track particles by id
"""

# sphinx_gallery_multi_image = "single"

# %%
# Start shamrock with experimental features
import shamrock

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


def is_in_sphere(pt):
    x, y, z = pt
    return (x**2 + y**2 + z**2) < 1


# %%
# Setup parameters

dr = 0.1
pmass = 1

C_cour = 0.3
C_force = 0.25

bsize = 4

# %%
# Setup

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
# cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
# cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_particle_tracking(True)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(1.00001)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)

bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)
model.resize_simulation_box(bmin, bmax)

model.set_particle_mass(pmass)

setup = model.get_setup()
lat = setup.make_generator_lattice_hcp(dr, (-bsize, -bsize, -bsize), (bsize, bsize, bsize))

thesphere = setup.make_modifier_filter(parent=lat, filter=is_in_sphere)

offset_sphere = setup.make_modifier_offset(
    parent=thesphere, offset_position=(3.0, 3.0, 3.0), offset_velocity=(-1.0, -1.0, -1.0)
)

setup.apply_setup(offset_sphere)

model.set_value_in_a_box("uint", "f64", 1, bmin, bmax)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

model.change_htolerance(1.3)
model.timestep()
model.change_htolerance(1.1)


# %%
# Init history table
part_history = []

data = ctx.collect_data()
print(data)

part_history = [{} for i in range(len(data["xyz"]))]
for i in range(len(data["xyz"])):
    part_history[data["part_id"][i]] = {
        "x": [data["xyz"][i][0]],
        "y": [data["xyz"][i][1]],
        "z": [data["xyz"][i][2]],
    }


def append_to_history():
    data = ctx.collect_data()

    for i in range(len(data["xyz"])):
        part_history[data["part_id"][i]]["x"].append(data["xyz"][i][0])
        part_history[data["part_id"][i]]["y"].append(data["xyz"][i][1])
        part_history[data["part_id"][i]]["z"].append(data["xyz"][i][2])


# %%
# Run the sim
for i in range(5):
    model.timestep()
    append_to_history()

# %%
# Plot particles history
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for i in range(len(part_history)):
    ax.plot(part_history[i]["x"], part_history[i]["y"], part_history[i]["z"], ".-", lw=0.2)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(part_history[0]["x"], part_history[0]["y"], part_history[0]["z"], ".-", lw=0.5, label="0")
ax.plot(part_history[1]["x"], part_history[1]["y"], part_history[1]["z"], ".-", lw=0.5, label="1")
ax.plot(part_history[2]["x"], part_history[2]["y"], part_history[2]["z"], ".-", lw=0.5, label="2")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)

ax.legend()

plt.show()
