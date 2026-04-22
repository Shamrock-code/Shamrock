"""
Orszag-Tang vortex in SPH
=========================

This example demonstrates how to setup and run an Orszag-Tang vortex
simulation using smoothed particle magnetohydrodynamics (SPMHD).

The simulation models:

- An ideal MHD fluid with adiabatic equation of state
- Periodic boundary conditions
- A sinusoidal velocity and magnetic field initialization to trigger
  the vortex instability
- On-the-fly rendering of density, magnetic field, and velocity fields

On a cluster or laptop, one can run the code as follows:

.. code-block:: bash

    mpirun <your parameters> ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript runscript.py

"""

# %%
# Setup and imports

import os

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# List parameters

kernel = "M4"

# CFLs
C_cour = 0.3
C_force = 0.25

# Grid resolution
nx = 64  # 512
ny = 64  # 590
nz = 3

# Domain bounds
xymin = -0.5
xmin = xymin
ymin = xymin

# Physical parameters
gamma = 5 / 3
betazero = 10.0 / 3.0
machzero = 1.0
vzero = 1.0
bzero = 1.0 / np.sqrt(4.0 * np.pi)

# Derived quantities
przero = 0.5 * bzero**2 * betazero
rhozero = gamma * przero * machzero
gam1 = gamma - 1.0
uuzero = przero / (gam1 * rhozero)

render_gif = True

dump_folder = ""
sim_name = "orztang"
analysis_folder = dump_folder  # store analysis data alongside dumps

# %%
# Configure the solver
# Set up the context, unit system, and SPH model with ideal MHD physics

ctx = shamrock.Context()
ctx.pdata_layout_new()

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=1.0,
    unit_length=1.0,
    unit_mass=1.2566370621219e-06,  # cgs mess
)
ucte = shamrock.Constants(codeu)
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)

cfg = model.gen_default_config()
cfg.set_units(codeu)

mu_0 = ucte.mu_0()
print(f"mu_0: {mu_0}")

cfg.set_artif_viscosity_None()  # artificial viscosity terms are computed in the MHD solver
cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
model.set_solver_config(cfg)

cfg.print_status()

crit_split = int(1e7)
crit_merge = 1
model.init_scheduler(crit_split, crit_merge)

# %%
# Setup the simulation

# Compute box size
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, nx, ny, nz)
print(f"Initial dim: x: {xs}\ty: {ys}\tz: {zs}")
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, nx, ny, nz)
print(f"Final dim: x: {xs}\ty: {ys}\tz: {zs}")

model.resize_simulation_box((-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))

model.add_cube_fcc_3d(dr, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))
model.set_value_in_a_box(
    "uint", "f64", uuzero, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2)
)

# %%
# Initial condition profiles


def vel_func(r):
    x, y, z = r
    vx = -vzero * np.sin(2.0 * np.pi * (y - ymin))
    vy = vzero * np.sin(2.0 * np.pi * (x - xmin))
    return (vx, vy, 0.0)


def mag_func(r):
    x, y, z = r
    Bx = -bzero * np.sin(2.0 * np.pi * (y - ymin)) / rhozero
    By = bzero * np.sin(4.0 * np.pi * (x - xmin)) / rhozero
    return (Bx, By, 0.0)


model.set_field_value_lambda_f64_3("vxyz", vel_func)
model.set_field_value_lambda_f64_3("B/rho", mag_func)

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

# %%
# On-the-fly analysis objects

import matplotlib.pyplot as plt
from shamrock.utils.analysis import (
    SliceByPlot,
    SliceDensityPlot,
    SliceVzPlot,
)

# Render domain half-extent (the box runs from -0.5 to 0.5)
ext_r = 0.5

density_slice_plot = SliceDensityPlot(
    model,
    ext_r=ext_r,
    nx=1080,
    ny=1080,
    ex=(1, 0, 0),
    ey=(0, 1, 0),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="rho_slice",
)

v_y_slice_plot = SliceVzPlot(
    model,
    ext_r=ext_r,
    nx=1080,
    ny=1080,
    ex=(1, 0, 0),
    ey=(0, 1, 0),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="vy_slice",
    do_normalization=False,
)

by_slice_plot = SliceByPlot(
    model,
    ext_r=ext_r,
    nx=1080,
    ny=1080,
    ex=(1, 0, 0),
    ey=(0, 1, 0),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="By_slice",
    do_normalization=False,
)


def analysis(ianalysis):
    density_slice_plot.analysis_save(ianalysis)
    v_y_slice_plot.analysis_save(ianalysis)
    by_slice_plot.analysis_save(ianalysis)


# %%
# Run the simulation

t_sum = 0.0
t_target = 1.0
dt_dump = 0.025

analysis(0)
model.do_vtk_dump(f"{sim_name}_{0:05}.vtk", True)

i_dump = 1

while t_sum < t_target:
    model.evolve_until(t_sum + dt_dump)
    model.do_vtk_dump(f"{sim_name}_{i_dump:05}.vtk", True)
    analysis(i_dump)
    t_sum += dt_dump
    i_dump += 1

# %%
# Plot generation

slice_render_kwargs = {
    "x_unit": "m",
    "y_unit": "m",
    "time_unit": "second",
    "x_label": "x",
    "y_label": "y",
}

density_slice_plot.render_all(
    **slice_render_kwargs,
    field_unit="kg.m^-2",
    field_label="$\\int \\rho \\, \\mathrm{{d}} z$",
    vmin=1e-7,
    vmax=5e-7,
    cmap="gist_heat",
    cmap_bad_color="black",
)

v_y_slice_plot.render_all(
    **slice_render_kwargs,
    field_unit="m.s^-1",
    field_label="$\\mathrm{v}_y$",
    vmin=-1e-2,
    vmax=1e-2,
    cmap="seismic",
    cmap_bad_color="black",
)

by_slice_plot.render_all(
    **slice_render_kwargs,
    field_unit="T",
    field_label=r"$B_y$",
    vmin=-1e-6,
    vmax=1e-6,
    cmap="seismic",
    cmap_bad_color="black",
)

# %%
# Make gif for the doc


# %%
# Density gif
if render_gif:
    ani = density_slice_plot.render_gif(gif_filename="rho_slice.gif", save_animation=True, fps=8)
    if ani is not None:
        plt.show()

# %%
# vy velocity gif
if render_gif:
    ani = v_y_slice_plot.render_gif(gif_filename="vy_slice.gif", save_animation=True, fps=8)
    if ani is not None:
        plt.show()

# %%
# By magnetic field gif
if render_gif:
    ani = by_slice_plot.render_gif(gif_filename="By_slice.gif", save_animation=True, fps=8)
    if ani is not None:
        plt.show()
