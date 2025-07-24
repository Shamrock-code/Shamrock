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


render_gif = True

dump_folder = "_to_trash"
sim_name = "sph_shear_test"

import os

# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.system("mkdir -p " + dump_folder)

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

print(f"v_shear = {shear_speed} | dv = {MM-mm}")


model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

# %%
# Perform the plot

from math import exp

import matplotlib.pyplot as plt
import numpy as np


def plot(iplot):
    dic = ctx.collect_data()
    fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
    fig.suptitle("t = {:.2f}".format(model.get_time()))
    axs[0].scatter(dic["xyz"][:, 0], dic["xyz"][:, 1], s=1)
    axs[1].scatter(dic["xyz"][:, 0], dic["vxyz"][:, 1], s=1)

    axs[0].set_ylabel("y")
    axs[1].set_ylabel("vy")
    axs[1].set_xlabel("x")

    axs[0].set_xlim(xm - 0.1, xM + 0.1)
    axs[0].set_ylim(ym - 0.1, yM + 0.1)

    axs[1].set_xlim(xm - 0.1, xM + 0.1)
    axs[1].set_ylim(shear_speed * 0.7, -shear_speed * 0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(dump_folder, f"{sim_name}_{iplot:04}.png"))
    plt.close(fig)


# %%
# Performing the timestep loop
model.timestep()

dt_stop = 0.02
for i in range(20):

    t_target = i * dt_stop
    # skip if the model is already past the target
    if model.get_time() > t_target:
        continue

    model.evolve_until(i * dt_stop)

    # Dump name is "dump_xxxx.sham" where xxxx is the timestep
    model.do_vtk_dump(os.path.join(dump_folder, f"{sim_name}_{i:04}.vtk"), True)
    plot(i)


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
