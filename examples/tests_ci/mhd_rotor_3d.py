"""
Testing the MHD rotor problem (3D) with SPMHD
===============================================

CI test for the 3D MHD rotor problem (Balsara & Spicer 1999; Toth 2000's
'first rotor problem'), following the setup described in the Phantom code
paper (Price et al. 2018, Section 5.6.4): a rapidly rotating dense disc of
material immersed in a uniform ambient medium threaded by a uniform
magnetic field, testing the propagation of rotational discontinuities.

"""

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
kernel = "M4"

C_cour = 0.3
C_force = 0.25

nx = 32
nz_thin = 12

gamma = 1.4

R_disc = 0.1
rho_disc = 10.0
rho_ambient = 1.0
P0 = 1.0
v0 = 2.0
Bx0 = 5.0 / np.sqrt(4.0 * np.pi)

fact = (rho_disc / rho_ambient) ** (1.0 / 3.0)

u_disc = P0 / ((gamma - 1) * rho_disc)
u_ambient = P0 / ((gamma - 1) * rho_ambient)

t_target = 0.15

# %%

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

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, nx, nx, nz_thin)
dr = 1.0 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, nx, nx, nz_thin)

box_min = (-xs / 2, -ys / 2, -zs / 2)
box_max = (xs / 2, ys / 2, zs / 2)

model.resize_simulation_box(box_min, box_max)

dr_disc = dr / fact
disc_box_min = (-R_disc, -R_disc, -zs / 2)
disc_box_max = (R_disc, R_disc, zs / 2)


def in_disc(r):
    x, y, z = r
    return (x * x + y * y) < R_disc * R_disc


def out_disc(r):
    x, y, z = r
    return (x * x + y * y) >= R_disc * R_disc


setup = model.get_setup()
gen_ambient = setup.make_generator_lattice_hcp(dr, box_min, box_max)
gen_ambient_masked = setup.make_modifier_filter(parent=gen_ambient, filter=out_disc)
gen_disc = setup.make_generator_lattice_hcp(dr_disc, disc_box_min, disc_box_max)
gen_disc_masked = setup.make_modifier_filter(parent=gen_disc, filter=in_disc)
comb = setup.make_combiner_add(gen_ambient_masked, gen_disc_masked)
setup.apply_setup(comb)

# Internal energy: ambient value everywhere, then overwrite inside the disc
model.set_value_in_a_box("uint", "f64", u_ambient, box_min, box_max)
model.set_value_in_sphere("uint", "f64", u_disc, (0, 0, 0), R_disc)


def vel_func(r):
    x, y, z = r
    r2 = x * x + y * y
    if r2 >= R_disc * R_disc:
        return (0.0, 0.0, 0.0)
    rcyl = np.sqrt(max(r2, 1e-30))
    vx = -v0 * y / rcyl
    vy = v0 * x / rcyl
    return (vx, vy, 0.0)


def mag_func(r):
    x, y, z = r
    rho_local = rho_disc if (x * x + y * y) < R_disc * R_disc else rho_ambient
    return (Bx0 / rho_local, 0.0, 0.0)


model.set_field_value_lambda_f64_3("vxyz", vel_func)
model.set_field_value_lambda_f64_3("B/rho", mag_func)

vol_b = xs * ys * zs
vol_disc = np.pi * R_disc * R_disc * zs
totmass = rho_ambient * (vol_b - vol_disc) + rho_disc * vol_disc
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

print("Total mass :", totmass)
print("Current part mass :", pmass)
print("Total particle count :", model.get_total_part_count())

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

model.timestep()

# %%

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

expect_kinetic_energy = 0.06164962101274802
expect_magnetic_energy = 0.33097235991190854
expect_rho_max = 6.7826140281526515
expect_rho_min = 0.21436775933952779

tol = 0.2  # relative tolerance
max_momentum_norm = 1e-2


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
