"""
Killing sphere for SPH simulation
=================================

This simple example shows how to setup a killing sphere for sph simulations
"""

# sphinx_gallery_multi_image = "single"

import shamrock

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


render_gif = True

dump_folder = "_to_trash"
sim_name = "/kill_particle_sphere_"

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
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(1.00001)
cfg.add_kill_sphere(center=(0.0, 0.0, 0.0), radius=4.0)
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
    parent=thesphere, offset_position=(0.0, 0.0, 0.0), offset_velocity=(-1.0, -1.0, -1.0)
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

        ax.set_title(f"t = {model.get_time():.2f} ")

        plt.savefig(dump_folder + sim_name + f"_{iplot:04}.png")


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
ani = show_image_sequence(dump_folder + sim_name + "_*.png")

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    # writer = animation.PillowWriter(fps=15,
    #                                 metadata=dict(artist='Me'),
    #                                 bitrate=1800)
    # ani.save('scatter.gif', writer=writer)

    # Show the animation
    plt.show()
