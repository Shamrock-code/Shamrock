"""
Sphere advection with multiple patch
====================================

This simple example demonstrate how Shamrock decompose the simulation domain
"""

import shamrock

# Particle tracking is an experimental feature
shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


####################################################
# Setup parameters
####################################################
dr = 0.05
pmass = 1

C_cour = 0.3
C_force = 0.25

bsize = 4

render_gif = True

dump_folder = "_to_trash"
sim_name = "sphere_domain_decomp"


def is_in_sphere(pt):
    x, y, z = pt
    return (x**2 + y**2 + z**2) < 1


import os

# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)

####################################################
# Setup the simulation
####################################################

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

model.init_scheduler(int(500), 50)
# model.init_scheduler(int(300), 50)

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

####################################################
# Draw utilities
####################################################

import matplotlib.pyplot as plt
import numpy as np


def draw_aabb(ax, aabb, color, alpha):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    xmin, ymin, zmin = aabb.lower()
    xmax, ymax, zmax = aabb.upper()

    points = [
        aabb.lower(),
        (aabb.lower()[0], aabb.lower()[1], aabb.upper()[2]),
        (aabb.lower()[0], aabb.upper()[1], aabb.lower()[2]),
        (aabb.lower()[0], aabb.upper()[1], aabb.upper()[2]),
        (aabb.upper()[0], aabb.lower()[1], aabb.lower()[2]),
        (aabb.upper()[0], aabb.lower()[1], aabb.upper()[2]),
        (aabb.upper()[0], aabb.upper()[1], aabb.lower()[2]),
        aabb.upper(),
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

    pos = ctx.collect_data()["xyz"]

    if shamrock.sys.world_rank() == 0:
        X = pos[:, 0]
        Y = pos[:, 1]
        Z = pos[:, 2]

        plt.cla()

        patch_list = ctx.get_patch_list_global()

        ax.set_xlim3d(bmin[0], bmax[0])
        ax.set_ylim3d(bmin[1], bmax[1])
        ax.set_zlim3d(bmin[2], bmax[2])

        ptransf = model.get_patch_transform()
        for p in patch_list:
            draw_aabb(ax, ptransf.to_obj_coord(p), "blue", 0.1)

        ax.scatter(X, Y, Z, c="red", s=1)

        plt.savefig(os.path.join(dump_folder, f"{sim_name}_{iplot:04}.png"))


####################################################
# Run the simulation
####################################################
nstop = 80  # 100 normally
dt_stop = 0.1

t_stop = [i * dt_stop for i in range(nstop + 1)]

# Init MPL
fig = plt.figure(dpi=120)
ax = fig.add_subplot(111, projection="3d")

iplot = 0
istop = 0
for ttarg in t_stop:

    model.evolve_until(ttarg)

    # if do_plots:
    plot_state(iplot)

    iplot += 1
    istop += 1

plt.close(fig)


####################################################
# Convert PNG sequence to Image sequence in mpl
####################################################
import matplotlib.animation as animation


def show_image_sequence(glob_str):

    if render_gif and shamrock.sys.world_rank() == 0:

        import glob

        files = sorted(glob.glob(glob_str))

        from PIL import Image

        image_array = []
        for my_file in files:
            image = Image.open(my_file)
            image_array.append(image)

        img = Image.open(files[0])
        pixel_x, pixel_y = img.size

        # Create the figure and axes objects
        # Remove axes, ticks, and frame & set aspect ratio
        dpi = 200
        fig = plt.figure(dpi=dpi)
        plt.gca().set_position((0, 0, 1, 1))
        plt.gcf().set_size_inches(pixel_x / dpi, pixel_y / dpi)
        plt.axis("off")

        # Set the initial image with correct aspect ratio
        im = plt.imshow(image_array[0], animated=True, aspect="auto")

        def update(i):
            im.set_array(image_array[i])
            return (im,)

        # Create the animation object
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(image_array),
            interval=50,
            blit=True,
            repeat_delay=10,
        )

        return ani


# If the animation is not returned only a static image will be shown in the doc
glob_str = os.path.join(dump_folder, f"{sim_name}_*.png")
ani = show_image_sequence(glob_str)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    # writer = animation.PillowWriter(fps=15,
    #                                 metadata=dict(artist='Me'),
    #                                 bitrate=1800)
    # ani.save('scatter.gif', writer=writer)

    # Show the animation
    plt.show()
