"""
Testing Orszag-Tang vortex (3D thin box) with SPMHD
=====================================================

CI test for the 3D (thin box) Orszag-Tang vortex, following the setup
described in the Phantom code paper (Price et al. 2018, Section 5.7.3):
a uniform density, periodic box with a sinusoidal velocity and magnetic
field perturbation that develops into a system of interacting MHD shocks.

"""

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Parameters (Price et al. 2018, Sec 5.7.3)

kernel = "M4"

C_cour = 0.3
C_force = 0.25

nx = 32
ny = 32
nz = 3

xymin = -0.5
xmin = xymin
ymin = xymin

gamma = 5.0 / 3.0
betazero = 10.0 / 3.0
machzero = 1.0
vzero = 1.0
bzero = 1.0 / np.sqrt(4.0 * np.pi)

przero = 0.5 * bzero**2 * betazero
rhozero = gamma * przero * machzero
gam1 = gamma - 1.0
uuzero = przero / (gam1 * rhozero)

t_target = 0.5

# %%
# Configure the solver

ctx = shamrock.Context()
ctx.pdata_layout_new()

si = shamrock.UnitSystem()
codeu = shamrock.UnitSystem(unit_time=1.0, unit_length=1.0, unit_mass=1.2566370621219e-06)
ucte = shamrock.Constants(codeu)
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)

cfg = model.gen_default_config()
cfg.set_units(codeu)

mu_0 = ucte.mu_0()

cfg.set_artif_viscosity_None()
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

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, nx, ny, nz)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, nx, ny, nz)

model.resize_simulation_box((-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))
model.add_cube_fcc_3d(dr, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))
model.set_value_in_a_box(
    "uint", "f64", uuzero, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2)
)


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

# %%
# Run the simulation

model.evolve_until(t_target)

# %%
# Diagnostics

kinetic_energy = shamrock.model_sph.analysisEnergyKinetic(model=model).get_kinetic_energy()
total_momentum = shamrock.model_sph.analysisTotalMomentum(model=model).get_total_momentum()

data = ctx.collect_data()
hpart = data["hpart"]
hfact = model.get_hfact()
rho = pmass * (hfact / hpart) ** 3
B_on_rho = data["B/rho"]

magnetic_energy = 0.5 * pmass * np.sum(rho * np.sum(B_on_rho**2, axis=1)) / mu_0

rho_max = np.max(rho)
rho_min = np.min(rho)

momentum_norm = np.sqrt(total_momentum[0] ** 2 + total_momentum[1] ** 2 + total_momentum[2] ** 2)

print(f"kinetic_energy   = {kinetic_energy}")
print(f"magnetic_energy  = {magnetic_energy}")
print(f"momentum_norm    = {momentum_norm}")
print(f"rho_max          = {rho_max}")
print(f"rho_min          = {rho_min}")

test_pass = True
err_log = ""


expect_kinetic_energy = 0.0021959332605647032
expect_magnetic_energy = 0.003053061265398151
expect_rho_max = 0.34210660332016646
expect_rho_min = 0.14228949962922874

tol = 0.2  # relative tolerance
max_momentum_norm = 1e-3


def float_equal(val1, val2, prec):
    return abs(val1 - val2) < prec


if not float_equal(kinetic_energy, expect_kinetic_energy, tol * expect_kinetic_energy):
    err_log += "error on the kinetic energy is outside of tolerances:\n"
    err_log += f"  expected error = {expect_kinetic_energy} +- {tol * expect_kinetic_energy}\n"
    err_log += f"  obtained error = {kinetic_energy}\n"
    test_pass = False

if not float_equal(magnetic_energy, expect_magnetic_energy, tol * expect_magnetic_energy):
    err_log += "error on the magnetic energy is outside of tolerances:\n"
    err_log += f"  expected error = {expect_magnetic_energy} +- {tol * expect_magnetic_energy}\n"
    err_log += f"  obtained error = {magnetic_energy}\n"
    test_pass = False

if momentum_norm > max_momentum_norm:
    err_log += "total momentum norm is outside of tolerances:\n"
    err_log += f"  expected |p| < {max_momentum_norm}\n"
    err_log += f"  obtained |p| = {momentum_norm}\n"
    test_pass = False

if not float_equal(rho_max, expect_rho_max, tol * expect_rho_max):
    err_log += "error on rho_max is outside of tolerances:\n"
    err_log += f"  expected error = {expect_rho_max} +- {tol * expect_rho_max}\n"
    err_log += f"  obtained error = {rho_max}\n"
    test_pass = False

if not float_equal(rho_min, expect_rho_min, tol * expect_rho_min):
    err_log += "error on rho_min is outside of tolerances:\n"
    err_log += f"  expected error = {expect_rho_min} +- {tol * expect_rho_min}\n"
    err_log += f"  obtained error = {rho_min}\n"
    test_pass = False

if test_pass == False:
    exit("Test did not pass L2 margins : \n" + err_log)
