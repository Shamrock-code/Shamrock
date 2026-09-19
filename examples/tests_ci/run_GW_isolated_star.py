"""
Isolated star GW emmission
=================================

Tests that the GW emission from an isolated star is indeed 0.
"""

# sphinx_gallery_multi_image = "single"

import os

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

shamrock.enable_experimental_features()


def is_in_sphere(pt):
    x, y, z = pt
    return (x**2 + y**2 + z**2) < 1


# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=1,
    unit_length=sicte.solar_radius(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

# %%
# Setup parameters

dr = 0.1
pmass = 1

C_cour = 0.3
C_force = 0.25

bsize = 4

render_gif = True

dump_folder = "_to_trash"
sim_name = "isolated_star"

if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)

# %%
# Setup
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
cfg.set_units(codeu)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
# cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
cfg.set_eos_polytropic(1e-12, 5.0 / 3.0)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_self_gravity_fmm(order=1, opening_angle=0.5, reduction_level=3)
cfg.set_softening_plummer(epsilon=1e-9)
cfg.set_boundary_free()
cfg.compute_GW(True)

# %%
# The important part to enable killing
cfg.add_kill_sphere(center=(0.0, 0.0, 0.0), radius=1.0)

# %%
# Rest of the setup
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

setup.apply_setup(thesphere)

model.set_value_in_a_box("uint", "f64", 1, bmin, bmax)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

model.change_htolerance(1.3)
model.timestep()
model.change_htolerance(1.1)

# %%
# Show the current solvergraph
shamrock.utils.plot.DotGraph(model.get_solver_dot_graph())

####################################################
# Draw utilities
####################################################
import matplotlib.pyplot as plt
import numpy as np


def plot_state(iplot):
    pos = ctx.collect_data()["xyz"]

    if shamrock.sys.world_rank() == 0:
        X = pos[:, 0]
        Y = pos[:, 1]
        Z = pos[:, 2]

        plt.cla()

        ax.set_xlim3d(bmin[0], bmax[0])
        ax.set_ylim3d(bmin[1], bmax[1])
        ax.set_zlim3d(bmin[2], bmax[2])

        ax.scatter(X, Y, Z, s=1)

        ax.minorticks_off()

        ax.set_title(f"t = {model.get_time():.2f} ")

        plt.savefig(os.path.join(dump_folder, f"{sim_name}_{iplot:04}.png"))


####################################################
# Run the simulation
####################################################
nstop = 28  # To be increased when empty simulations will be fixed
dt_stop = 0.1

t_stop = [i * dt_stop for i in range(nstop + 1)]

# Init MPL
fig = plt.figure(dpi=120)
ax = fig.add_subplot(111, projection="3d")

iplot = 0
istop = 0
for ttarg in t_stop:
    model.evolve_until(ttarg)
    model.do_vtk_dump(os.path.join(dump_folder, f"{sim_name}_{iplot:04}.vtk"), True)

    # if do_plots:
    plot_state(iplot)

    iplot += 1
    istop += 1

plt.close(fig)
