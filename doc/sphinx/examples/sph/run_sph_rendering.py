"""
Using Shamrock SPH rendering module
===================================

This example demonstrates how to use the Shamrock SPH rendering module to render the density field or the velocity field of a SPH simulation.

"""

# %%
# Runscript (runscript.py)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The runscript is the actual simulation with on the fly analysis & rolling dumps


import glob
import json
import os  # for makedirs

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Setup units

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.second(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()
c = ucte.c()

# %%
# List parameters

# Resolution
Npart = 100000

# Domain decomposition parameters
scheduler_split_val = int(1.0e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Disc parameter
center_mass = 1e6  # [sol mass]
disc_mass = 0.001  # [sol mass]
Rg = G * center_mass / (c * c)  # [au]
rin = 4.0 * Rg  # [au]
rout = 10 * rin  # [au]
r0 = rin  # [au]

H_r_0 = 0.01
q = 0.75
p = 3.0 / 2.0

Tin = 2 * np.pi * np.sqrt(rin * rin * rin / (G * center_mass))
if shamrock.sys.world_rank() == 0:
    print(" Orbital period : ", Tin, " [seconds]")

# Sink parameters
center_racc = rin / 2.0  # [au]
inclination = 30.0 * np.pi / 180.0


# Viscosity parameter
alpha_AV = 1.0e-3 / 0.08
alpha_u = 1.0
beta_AV = 2.0

# Integrator parameters
C_cour = 0.3
C_force = 0.25


# Disc profiles
def sigma_profile(r):
    sigma_0 = 1.0  # We do not care as it will be renormalized
    return sigma_0 * (r / r0) ** (-p)


def kep_profile(r):
    return (G * center_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    cs_in = (H_r_0 * r0) * omega_k(r0)
    return ((r / r0) ** (-q)) * cs_in


# %%
# Utility functions and quantities deduced from the base one

# Deduced quantities
pmass = disc_mass / Npart

bsize = rout * 2
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)

cs0 = cs_profile(r0)


def rot_profile(r):
    return ((kep_profile(r) ** 2) - (2 * p + q) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    # fact = (2.**0.5) * 3. # factor taken from phantom, to fasten thermalizing
    fact = 1.0
    return fact * H


# %%
# Start the context
# The context holds the data of the code
# We then init the layout of the field (e.g. the list of fields used by the solver)

ctx = shamrock.Context()
ctx.pdata_layout_new()

# %%
# Attach a SPH model to the context

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

# Generate the default config
cfg = model.gen_default_config()
cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)

# cfg.add_ext_force_point_mass(center_mass, center_racc)

cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box
cfg.add_ext_force_lense_thirring(
    central_mass=center_mass,
    Racc=rin,
    a_spin=0.9,
    dir_spin=(np.sin(inclination), np.cos(inclination), 0.0),
)

cfg.set_units(codeu)
cfg.set_particle_mass(pmass)
# Set the CFL
cfg.set_cfl_cour(C_cour)
cfg.set_cfl_force(C_force)

# On a chaotic disc, we disable to two stage search to avoid giant leaves
cfg.set_tree_reduction_level(6)
cfg.set_two_stage_search(False)

# Enable this to debug the neighbor counts
# cfg.set_show_neigh_stats(True)

# Standard way to set the smoothing length (e.g. Price et al. 2018)
cfg.set_smoothing_length_density_based()

# Standard density based smoothing lenght but with a neighbor count limit
# Use it if you have large slowdowns due to giant particles
# I recommend to use it if you have a circumbinary discs as the issue is very likely to happen
# cfg.set_smoothing_length_density_based_neigh_lim(500)

# Set the solver config to be the one stored in cfg
model.set_solver_config(cfg)

# Print the solver config
model.get_current_config().print_status()

# Init the scheduler & fields
model.init_scheduler(scheduler_split_val, scheduler_merge_val)

# Set the simulation box size
model.resize_simulation_box(bmin, bmax)

# Create the setup

setup = model.get_setup()
gen_disc = setup.make_generator_disc_mc(
    part_mass=pmass,
    disc_mass=disc_mass,
    r_in=rin,
    r_out=rout,
    sigma_profile=sigma_profile,
    H_profile=H_profile,
    rot_profile=rot_profile,
    cs_profile=cs_profile,
    random_seed=666,
    init_h_factor=0.06,
)

# Print the dot graph of the setup
print(gen_disc.get_dot())

# Apply the setup
setup.apply_setup(gen_disc)

model.change_htolerances(coarse=1.3, fine=1.1)
model.timestep()
model.change_htolerances(coarse=1.1, fine=1.1)

for i in range(5):
    model.timestep()


# %%
# Usual cartesian rendering

ext = rout * 1.5
center = (0.0, 0.0, 0.0)
delta_x = (ext * 2, 0, 0.0)
delta_y = (0.0, ext * 2, 0.0)
nx = 1024
ny = 1024
nr = 1024
ntheta = 1024

arr_rho = model.render_cartesian_column_integ(
    "rho",
    "f64",
    center=center,
    delta_x=delta_x,
    delta_y=delta_y,
    nx=nx,
    ny=ny,
)

arr_vxyz = model.render_cartesian_column_integ(
    "vxyz",
    "f64_3",
    center=center,
    delta_x=delta_x,
    delta_y=delta_y,
    nx=nx,
    ny=ny,
)


def plot_rho_integ(metadata, arr_rho):

    ext = metadata["extent"]

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho, cmap=my_cmap, origin="lower", extent=ext, norm="log", vmin=1e-6, vmax=1e-2
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [seconds]")

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\int \rho \, \mathrm{d}z$ [code unit]")


def plot_vz_integ(metadata, arr_vz):
    ext = metadata["extent"]

    # if you want an adaptive colorbar
    v_ext = np.max(arr_vz)
    v_ext = max(v_ext, np.abs(np.min(arr_vz)))
    # v_ext = 1e-6

    res = plt.imshow(arr_vz, cmap="seismic", origin="lower", extent=ext, vmin=-v_ext, vmax=v_ext)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [seconds]")

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\int v_z \, \mathrm{d}z$ [code unit]")


metadata = {"extent": [-ext, ext, -ext, ext], "time": model.get_time()}

dpi = 200

plt.figure(num=1, clear=True, dpi=dpi)
plot_rho_integ(metadata, arr_rho)

plt.figure(num=1, clear=True, dpi=dpi)
plot_vz_integ(metadata, arr_vxyz[:, :, 2])

# %%
# Cylindrical rendering


def make_cylindrical_coords(nr, ntheta):
    """
    Generate a list of positions in cylindrical coordinates (r, theta)
    spanning [0, ext*2] x [-pi, pi] for use with the rendering module.

    Returns:
        list: List of [x, y, z] coordinate lists
    """

    # Create the cylindrical coordinate grid
    r_vals = np.linspace(0, ext, nr)
    theta_vals = np.linspace(-np.pi, np.pi, ntheta)

    # Create meshgrid
    r_grid, theta_grid = np.meshgrid(r_vals, theta_vals)

    # Convert to Cartesian coordinates (z = 0 for a disc in the xy-plane)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    z_grid = np.zeros_like(r_grid)

    # Flatten and stack to create list of positions
    positions = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    return [tuple(pos) for pos in positions]


def positions_to_rays(positions):
    rays = []
    for position in positions:
        x, y, z = position
        rays.append(shamrock.math.Ray_f64_3((x, y, z), (0.0, 0.0, 1.0)))
    return rays


positions_cylindrical = make_cylindrical_coords(nr, ntheta)
rays_cylindrical = positions_to_rays(positions_cylindrical)


arr_rho_cylindrical = model.render_column_integ("rho", "f64", rays_cylindrical)

arr_rho_pos = model.render_slice("rho", "f64", positions_cylindrical)


def plot_rho_integ_cylindrical(metadata, arr_rho_cylindrical):
    ext = metadata["extent"]

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    arr_rho_cylindrical = np.array(arr_rho_cylindrical).reshape(nr, ntheta)

    res = plt.imshow(
        arr_rho_cylindrical,
        cmap=my_cmap,
        origin="lower",
        extent=ext,
        norm="log",
        vmin=1e-6,
        vmax=1e-2,
        aspect="auto",
    )
    plt.xlabel("r")
    plt.ylabel(r"$\theta$")
    plt.title(f"t = {metadata['time']:0.3f} [seconds]")
    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\int \rho \, \mathrm{d}z$ [code unit]")


def plot_rho_slice_cylindrical(metadata, arr_rho_pos):
    ext = metadata["extent"]

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    arr_rho_pos = np.array(arr_rho_pos).reshape(nr, ntheta)

    res = plt.imshow(
        arr_rho_pos, cmap=my_cmap, origin="lower", extent=ext, norm="log", vmin=1e-8, aspect="auto"
    )
    plt.xlabel("r")
    plt.ylabel(r"$\theta$")
    plt.title(f"t = {metadata['time']:0.3f} [seconds]")
    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\rho$ [code unit]")


metadata = {"extent": [0, ext, -np.pi, np.pi], "time": model.get_time()}

plt.figure(num=1, clear=True, dpi=dpi)
plot_rho_integ_cylindrical(metadata, arr_rho_cylindrical)

plt.figure(num=1, clear=True, dpi=dpi)
plot_rho_slice_cylindrical(metadata, arr_rho_pos)

plt.show()
