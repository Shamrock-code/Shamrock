"""
Hard Wall Boundary Test
=======================

This example demonstrates the hard wall boundary condition in Shamrock
where particles reflect off a wall defined by a ghost mask.
"""

import os

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()


####################################################
# Setup parameters
####################################################
nx = 8
ny = 8
nz = 8

uuzero = 10.0
rhozero = 1.0
C_cour = 0.3
C_force = 0.25

t_target = 0.3
dt_dump = 0.01

render_gif = True  # whether to create a GIF animation
gif_fps = 10  # frames per second for the GIF

# Folder structure (following the production example)
sim_folder = "_to_trash/hardwall_test/"
dump_folder = os.path.join(sim_folder, "dump")
analysis_folder = os.path.join(sim_folder, "analysis")
plot_folder = os.path.join(analysis_folder, "plots")

dump_prefix = "dump_"  # prefix for VTK files
plot_prefix = "plot_"  # prefix for PNG files

# Create directories (only on rank 0)
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)


def ghost_map(r):
    """Define which particles are inside the wall boundary."""
    x, y, z = r
    return 1 if x > 0 else 0


def vel_func(r):
    """Velocity function for the particles."""
    x, y, z = r
    return (0.5, 0.5, 0.0)


####################################################
# Setup the simulation
####################################################

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(5.0 / 3.0)
cfg.use_wall(True)
cfg.print_status()
model.set_solver_config(cfg)

# Set scheduler criteria to effectively disable patch splitting and merging
crit_split = int(1e7)
crit_merge = 1
model.init_scheduler(crit_split, crit_merge)

# Compute box size
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, nx, ny, nz)
print(f"Initial dim: x: {xs}\ty: {ys}\tz: {zs}")
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, nx, ny, nz)
print(f"Final dim: x: {xs}\ty: {ys}\tz: {zs}")

bmin = (-xs / 2, -ys / 2, -zs / 2)
bmax = (xs / 2, ys / 2, zs / 2)
model.resize_simulation_box(bmin, bmax)

model.add_cube_fcc_3d(dr, bmin, bmax)
model.set_value_in_a_box("uint", "f64", uuzero, bmin, bmax)

# Set velocity field
model.set_field_value_lambda_f64_3("vxyz", vel_func)

# Set ghost mask for wall boundary
model.set_field_value_lambda_u32("ghost_mask", ghost_map)

# Compute mass
vol_b = xs * ys * zs
totmass = rhozero * vol_b
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

print("Total mass :", totmass)
print("Current part mass :", pmass)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

model.timestep()
cfg.print_status()

####################################################
# Draw utilities
####################################################

import matplotlib.pyplot as plt


def get_vtk_dump_path(idump):
    """Return full path for VTK dump file."""
    return os.path.join(dump_folder, f"{dump_prefix}{idump:07}.vtk")


def draw_aabb(ax, aabb, color, alpha):
    """Draw an axis-aligned bounding box."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    xmin, ymin, zmin = aabb.lower
    xmax, ymax, zmax = aabb.upper

    points = [
        aabb.lower,
        (aabb.lower[0], aabb.lower[1], aabb.upper[2]),
        (aabb.lower[0], aabb.upper[1], aabb.lower[2]),
        (aabb.lower[0], aabb.upper[1], aabb.upper[2]),
        (aabb.upper[0], aabb.lower[1], aabb.lower[2]),
        (aabb.upper[0], aabb.lower[1], aabb.upper[2]),
        (aabb.upper[0], aabb.upper[1], aabb.lower[2]),
        aabb.upper,
    ]

    faces = [
        [points[0], points[1], points[3], points[2]],
        [points[4], points[5], points[7], points[6]],
        [points[0], points[1], points[5], points[4]],
        [points[2], points[3], points[7], points[6]],
        [points[0], points[2], points[6], points[4]],
        [points[1], points[3], points[7], points[5]],
    ]

    edges = [
        [points[0], points[1]],
        [points[0], points[2]],
        [points[0], points[4]],
        [points[1], points[3]],
        [points[1], points[5]],
        [points[2], points[3]],
        [points[2], points[6]],
        [points[3], points[7]],
        [points[4], points[5]],
        [points[4], points[6]],
        [points[5], points[7]],
        [points[6], points[7]],
    ]

    collection = Poly3DCollection(faces, alpha=alpha, color=color)
    ax.add_collection3d(collection)

    edge_collection = Line3DCollection(edges, color="k", alpha=alpha)
    ax.add_collection3d(edge_collection)


def plot_state(iplot):
    """Plot the current state of the simulation."""
    data = ctx.collect_data()
    pos = data["xyz"]

    if shamrock.sys.world_rank() == 0:
        X = pos[:, 0]
        Y = pos[:, 1]
        Z = pos[:, 2]

        # Filter particles by ghost mask to show wall region
        ghost_mask = data["ghost_mask"]
        wall_mask = ghost_mask == 1

        plt.cla()

        patch_list = ctx.get_patch_list_global()

        ax.set_xlim3d(bmin[0], bmax[0])
        ax.set_ylim3d(bmin[1], bmax[1])
        ax.set_zlim3d(bmin[2], bmax[2])

        ptransf = model.get_patch_transform()
        for p in patch_list:
            draw_aabb(ax, ptransf.to_obj_coord(p), "blue", 0.1)

        # Plot particles with wall particles in a different color
        ax.scatter(X[~wall_mask], Y[~wall_mask], Z[~wall_mask], c="red", s=5, label="Fluid")
        if np.any(wall_mask):
            ax.scatter(X[wall_mask], Y[wall_mask], Z[wall_mask], c="green", s=5, label="Wall")
        ax.legend()

        # Draw the wall plane at x=0 for reference
        yy, zz = np.meshgrid(np.linspace(bmin[1], bmax[1], 10), np.linspace(bmin[2], bmax[2], 10))
        xx = np.zeros_like(yy)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color="gray")

        # Save plot with correct path and prefix
        plot_path = os.path.join(plot_folder, f"{plot_prefix}{iplot:04}.png")
        plt.savefig(plot_path)


####################################################
# Run the simulation
####################################################

# Initial dump
model.do_vtk_dump(get_vtk_dump_path(0), True)

i_dump = 1
t_sum = 0

# Create figure for plotting
fig = plt.figure(dpi=120)
ax = fig.add_subplot(111, projection="3d")

# Plot initial state
plot_state(0)

while t_sum < t_target:
    model.evolve_until(t_sum + dt_dump)
    model.do_vtk_dump(get_vtk_dump_path(i_dump), True)
    plot_state(i_dump)
    t_sum += dt_dump
    i_dump += 1

plt.close(fig)

####################################################
# Convert PNG sequence to GIF animation
####################################################

import matplotlib.animation as animation
from shamrock.utils.plot import show_image_sequence

# Build glob pattern for all plot images
glob_str = os.path.join(plot_folder, f"{plot_prefix}*.png")

# Create the animation
ani = show_image_sequence(glob_str, render_gif=render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # Save the animation as a GIF file
    gif_path = os.path.join(plot_folder, "hard_wall_animation.gif")
    # Use pillow writer; adjust fps as needed
    ani.save(gif_path, writer="pillow", fps=gif_fps)
    print(f"Animation saved to {gif_path}")

    # Optionally display the animation
    plt.show()
