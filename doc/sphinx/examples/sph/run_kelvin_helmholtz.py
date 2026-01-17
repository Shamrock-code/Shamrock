"""
Kelvin-Helmholtz instability in SPH
===================================

This simple example shows how to setup a Kelvin-Helmholtz instability in SPH

.. warning::
    This test is shown at low resolution to avoid smashing our testing time,
    the instability starts to appear for resol > 64 with M6 kernel

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
import numpy as np

kernel = "M6"  # SPH kernel to use
resol = 32  # number of particles in the x & y direction
thick = 6  # number of particles in the z direction

# CFLs
C_cour = 0.3
C_force = 0.25

gamma = 1.4

vslip = 1  # slip speed between the two layers

rho_1 = 1

fact = 2 / 3
rho_2 = rho_1 / (fact**3)

P_1 = 3.5
P_2 = 3.5

render_gif = True

dump_folder = "_to_trash"
sim_name = "kh_sph"

u_1 = P_1 / ((gamma - 1) * rho_1)
u_2 = P_2 / ((gamma - 1) * rho_2)

print("Mach number 1 :", vslip / np.sqrt(gamma * P_1 / rho_1))
print("Mach number 2 :", vslip / np.sqrt(gamma * P_2 / rho_2))


import os

# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)


# %%
# Configure the solver
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

# Set scheduler criteria to effectively disable patch splitting and merging.
crit_split = int(1e9)
crit_merge = 1
model.init_scheduler(crit_split, crit_merge)

# %%
# Setup the simulation

# Compute box size
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, resol, thick)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, resol, thick)

model.resize_simulation_box((-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))

# rho1 domain
y_interface = ys / 4
model.add_cube_fcc_3d(dr, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, -y_interface, zs / 2))
model.add_cube_fcc_3d(dr, (-xs / 2, y_interface, -zs / 2), (xs / 2, ys / 2, zs / 2))

# rho 2 domain
model.add_cube_fcc_3d(dr * fact, (-xs / 2, -y_interface, -zs / 2), (xs / 2, y_interface, zs / 2))

model.set_value_in_a_box(
    "uint", "f64", u_1, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, -y_interface, zs / 2)
)
model.set_value_in_a_box(
    "uint", "f64", u_1, (-xs / 2, y_interface, -zs / 2), (xs / 2, ys / 2, zs / 2)
)

model.set_value_in_a_box(
    "uint", "f64", u_2, (-xs / 2, -y_interface, -zs / 2), (xs / 2, y_interface, zs / 2)
)


# the velocity function to trigger KH
def vel_func(r):
    x, y, z = r

    ampl = 0.01
    n = 2
    pert = np.sin(2 * np.pi * n * x / (xs))

    sigma = 0.05 / (2**0.5)
    gauss1 = np.exp(-((y - y_interface) ** 2) / (2 * sigma * sigma))
    gauss2 = np.exp(-((y + y_interface) ** 2) / (2 * sigma * sigma))
    pert *= gauss1 + gauss2

    # Alternative formula (See T. Tricco paper)
    # interf_sz = ys/32
    # vx = np.arctan(y/interf_sz)/np.pi

    vx = 0
    if np.abs(y) > y_interface:
        vx = vslip / 2
    else:
        vx = -vslip / 2

    return (vx, ampl * pert, 0)


model.set_field_value_lambda_f64_3("vxyz", vel_func)

vol_b = xs * ys * zs

totmass = (rho_1 * vol_b / 2) + (rho_2 * vol_b / 2)

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

print("Total mass :", totmass)
print("Current part mass :", pmass)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

model.timestep()


# %%
# Plotting functions
import copy

import matplotlib
import matplotlib.pyplot as plt


def plot_state(iplot):
    pixel_x = 1080
    pixel_y = 1080
    radius = 0.5
    center = (0.0, 0.0, 0.0)
    aspect = pixel_x / pixel_y
    pic_range = [-radius * aspect, radius * aspect, -radius, radius]
    delta_x = (radius * 2 * aspect, 0.0, 0.0)
    delta_y = (0.0, radius * 2, 0.0)

    def _render(field, field_type, center):
        # Helper to reduce code duplication
        return model.render_cartesian_slice(
            field,
            field_type,
            center=center,
            delta_x=delta_x,
            delta_y=delta_y,
            nx=pixel_x,
            ny=pixel_y,
        )

    arr_rho = _render("rho", "f64", center)
    arr_alpha = _render("alpha_AV", "f64", center)
    arr_vel = _render("vxyz", "f64_3", center)

    vy_range = np.abs(arr_vel[:, :, 1]).max()

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))
    my_cmap.set_bad(color="black")

    my_cmap2 = copy.copy(matplotlib.colormaps.get_cmap("nipy_spectral"))
    my_cmap2.set_bad(color="black")

    # rho plot
    fig = plt.figure(dpi=200)
    im0 = plt.imshow(arr_rho, cmap=my_cmap, origin="lower", extent=pic_range, vmin=1, vmax=3)

    cbar0 = plt.colorbar(im0, extend="both")
    cbar0.set_label(r"$\rho$ [code unit]")

    plt.title("t = {:0.3f} [code unit]".format(model.get_time()))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(dump_folder, f"{sim_name}_rho_{iplot:04}.png"))

    plt.close(fig)

    # alpha plot
    fig = plt.figure(dpi=200)
    im0 = plt.imshow(
        arr_alpha, cmap=my_cmap2, origin="lower", norm="log", extent=pic_range, vmin=1e-6, vmax=1
    )

    cbar0 = plt.colorbar(im0, extend="both")
    cbar0.set_label(r"$\alpha_{AV}$ [code unit]")

    plt.title("t = {:0.3f} [code unit]".format(model.get_time()))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(dump_folder, f"{sim_name}_alpha_{iplot:04}.png"))

    plt.close(fig)

    # vy plot
    fig = plt.figure(dpi=200)
    im1 = plt.imshow(
        arr_vel[:, :, 1],
        cmap=my_cmap,
        origin="lower",
        extent=pic_range,
        vmin=-vy_range,
        vmax=vy_range,
    )

    cbar1 = plt.colorbar(im1, extend="both")
    cbar1.set_label(r"$v_y$ [code unit]")

    plt.title("t = {:0.3f} [code unit]".format(model.get_time()))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(dump_folder, f"{sim_name}_vy_{iplot:04}.png"))

    plt.close(fig)


# %%
# Running the simulation

t_sum = 0
t_target = 1

plot_state(0)

i_dump = 1
dt_dump = 0.02

while t_sum < t_target:
    model.evolve_until(t_sum + dt_dump)

    plot_state(i_dump)

    t_sum += dt_dump
    i_dump += 1


####################################################
# Convert PNG sequence to Image sequence in mpl
####################################################

import matplotlib.animation as animation
from shamrock.utils.plot import show_image_sequence

# If the animation is not returned only a static image will be shown in the doc

# %%
# Rho plot
glob_str = os.path.join(dump_folder, f"{sim_name}_rho_*.png")
ani = show_image_sequence(glob_str, render_gif=render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # Show the animation
    plt.show()

# %%
# Vy plot
glob_str = os.path.join(dump_folder, f"{sim_name}_vy_*.png")
ani = show_image_sequence(glob_str, render_gif=render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # Show the animation
    plt.show()

# %%
# alpha plot
glob_str = os.path.join(dump_folder, f"{sim_name}_alpha_*.png")
ani = show_image_sequence(glob_str, render_gif=render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # Show the animation
    plt.show()
