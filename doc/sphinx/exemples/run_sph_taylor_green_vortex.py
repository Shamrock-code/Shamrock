"""
Taylor green vortex in SPH
==========================

This simple example shows a SPH simulation of a Taylor green vortex
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

gamma = 1.4

rho_g = 1

Mach = 0.1
P_g = (Mach**-2) * rho_g / gamma

print(f"Mach number : {1 / np.sqrt(gamma * P_g / rho_g)}")

u_g = P_g / ((gamma - 1) * rho_g)

resol_per_green = 128
vortex_size = 1

L_green = vortex_size / (2 * np.pi)

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
# Set the CFL
cfg.set_cfl_cour(0.1)
cfg.set_cfl_force(0.1)

# Set the solver config to be the one stored in cfg
model.set_solver_config(cfg)

# Print the solver config
model.get_current_config().print_status()

# We want the patches to split above 10^8 part and merge if smaller than 1 part (e.g. disable patch)
model.init_scheduler(int(1e8), 1)


resol = resol_per_green
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, resol, resol)
dr = 1 / xs

print("Box size : ", xs, ys, zs)
(xs, ys, zs) = (1.0, 1.0, dr * 12)

model.resize_simulation_box((-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))


setup = model.get_setup()
gen1 = setup.make_generator_lattice_hcp(dr, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))
setup.apply_setup(gen1)

model.set_value_in_a_box("uint", "f64", u_g, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))


def vel_func(r):
    x, y, z = r

    x += L_green * np.pi

    vx = np.cos(x / L_green) * np.sin(y / L_green)
    vy = -np.sin(x / L_green) * np.cos(y / L_green)
    vz = 0

    return (vx, vy, vz)


model.set_field_value_lambda_f64_3("vxyz", vel_func)


vol_b = xs * ys * zs

totmass = rho_g * vol_b

print("Total mass :", totmass)


pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)


dump_folder = "_to_trash"
import os

os.system("mkdir -p " + dump_folder)

current_fig = None

cnt_plot = 0


def plot():
    global current_fig
    if current_fig is not None:
        plt.close(current_fig)

    pixel_x = 1080
    pixel_y = 1080
    radius = 0.5
    center = (0.0, 0.0, 0.0)
    aspect = pixel_x / pixel_y
    pic_range = [-radius * aspect, radius * aspect, -radius, radius]
    delta_x = (radius * 2 * aspect, 0.0, 0.0)
    delta_y = (0.0, radius * 2, 0.0)

    arr_vel = model.render_cartesian_slice(
        "vxyz",
        "f64_3",
        center=(0.0, 0.0, 0.0),
        delta_x=delta_x,
        delta_y=delta_y,
        nx=pixel_x,
        ny=pixel_y,
    )

    v_norm = np.sqrt(arr_vel[:, :, 0] ** 2 + arr_vel[:, :, 1] ** 2 + arr_vel[:, :, 2] ** 2)

    import copy

    import matplotlib

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    fig_width = 6
    fig_height = fig_width / aspect
    current_fig = plt.figure(figsize=(fig_width, fig_height))
    res = plt.imshow(v_norm, cmap=my_cmap, origin="lower", extent=pic_range)

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\sqrt{vx^2 + vy^2 + vz^2}$ [code unit]")

    plt.title("t = {:0.3f} [code unit]".format(model.get_time()))
    plt.xlabel("x")
    plt.ylabel("y")
    global cnt_plot
    plt.savefig(dump_folder + f"/test_{cnt_plot}.png")
    cnt_plot += 1


model.timestep()

dt_stop = 0.001
for i in range(1):
    t_target = i * dt_stop
    # skip if the model is already past the target
    if model.get_time() > t_target:
        continue

    model.evolve_until(i * dt_stop)

    # Dump name is "dump_xxxx.sham" where xxxx is the timestep
    model.do_vtk_dump(dump_folder + "/dump_{:04}.vtk".format(i), True)
    plot()
