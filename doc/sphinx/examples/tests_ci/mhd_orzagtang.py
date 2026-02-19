"""
Comparing Orzag-Tang vortex with 1 patch with Phantom
==============================================

Restart a Sedov blast simulation from a Phantom dump using 1 patch, run it in Shamrock
and compare the results with the original Phantom dump.
This test is used to check that the Shamrock solver is able to reproduce the
same results as Phantom.
"""


import shamrock
import matplotlib.pyplot as plt
import numpy as np

################################################
################### PARAMETERS #################
################################################

################### box parameters ############
dr = 0.02

# size taken from phantom dump
nx = 128
ny = 148
nz = 12

xm = -0.5
ym = -0.5
bmin = (-0.5, -0.5, -2*np.sqrt(6)/nx)
bmax = (0.5, 0.5, 2*np.sqrt(6)/nx)

############# initial conditions ##########
pmass = -1
B0 = 1. / np.sqrt(4*np.pi)
v0 = 1.
cs0 = 1.
gamma = 5./3.
M0 = v0 / cs0
beta0 = 10./3.
P0 = 0.5 * B0 * B0 * beta0
rho0 = gamma * P0 * M0

C_cour = 0.3
C_force = 0.25

################################################
################# unit system ##################
################################################
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=1.0,
    unit_length=1.0,
    unit_mass=1.2566370621219e-06,
)
ucte = shamrock.Constants(codeu)

mu_0 = ucte.mu_0()
ctx = shamrock.Context()
ctx.pdata_layout_new()

################################################
#################### config ####################
################################################

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
cfg.set_units(codeu)
cfg.set_artif_viscosity_None()
cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)


################################################
############### size of the box ################
################################################

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, nx, ny, nz)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, nx, ny, nz)
model.resize_simulation_box(bmin, bmax)
model.add_cube_fcc_3d(dr, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))

vol_b = xs * ys * zs
totmass = (rho0 * vol_b)
print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)

################################################
############## initial conditions ##############
################################################


def B_func(r):
    x, y, z = r
    xp = x - xm
    yp = y - ym
    Bx = -B0 * np.sin(2*np.pi * yp) 
    By = B0 * np.sin(4*np.pi * xp) 
    Bz = 0.0 
    return (Bx, By, Bz)


model.set_field_value_lambda_f64_3("B/rho", B_func)


def vel_func(r):
    x, y, z = r
    xp = x - xm
    yp = y - ym
    vx = -v0 * np.sin(2*np.pi * yp) 
    vy =  v0 * np.sin(2*np.pi * xp) 
    vz = 0.0 
    return (vx, vy, vz)


model.set_field_value_lambda_f64_3("vxyz", vel_func)


def u_func(r):

    u = P0 / ((gamma - 1) * rho0)
    return u


model.set_field_value_lambda_f64("uint", u_func)


print("Current part mass :", pmass)
model.set_particle_mass(pmass)


#tot_u = pmass * model.get_sum("uint", "f64")

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

t_sum = 0
t_target = 1.
i_dump = 0
dt_dump = 0.1 * t_target
next_dt_target = t_sum + dt_dump


while next_dt_target <= t_target:

    model.do_vtk_dump(f"orztest/test_{i_dump:05}.vtk", True)
    model.evolve_until(next_dt_target)

    i_dump += 1

    next_dt_target += dt_dump
