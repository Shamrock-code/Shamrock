"""
Testing 3D circularly polarised Alfven wave with SPMHD
========================================================

CI test for the 3D circularly polarised Alfven wave (Toth 2000, 3D setup
of Gardiner & Stone 2008), following the description given in the Phantom
code paper (Price et al. 2018, Section 5.6.2 / 5.7.1).

"""

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
gamma = 5.0 / 3.0
rho0 = 1.0
P0 = 0.1

wavelength = 1.0
amplitude = 0.1

sin_a = 2.0 / 3.0
cos_a = np.sqrt(1.0 - sin_a**2)
sin_b = 2.0 / np.sqrt(5.0)
cos_b = np.sqrt(1.0 - sin_b**2)

# Rotation matrix columns: r (propagation direction), e2, e3 (transverse)
r_hat = np.array([cos_a * cos_b, cos_a * sin_b, sin_a])
e2_hat = np.array([-sin_b, cos_b, 0.0])
e3_hat = np.array([-sin_a * cos_b, -sin_a * sin_b, cos_a])
rotmat = np.column_stack([r_hat, e2_hat, e3_hat])

# Alfven speed set by B1 = 1, rho = 1 -> v_A = 1, period = wavelength / v_A = 1
v_A = 1.0
n_periods = 5
t_target = n_periods * wavelength / v_A

resol = 32

# %%

ctx = shamrock.Context()
ctx.pdata_layout_new()

si = shamrock.UnitSystem()
codeu = shamrock.UnitSystem(unit_time=1.0, unit_length=1.0, unit_mass=1.2566370621219e-06)
ucte = shamrock.Constants(codeu)

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
cfg.set_units(codeu)
mu_0 = ucte.mu_0()
cfg.set_artif_viscosity_None()
cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

crit_split = int(1e7)
crit_merge = 1
model.init_scheduler(crit_split, crit_merge)


# %%
def best_transverse_counts(model, xcnt, target_ratio=0.5, search_frac=0.18):
    lo = max(2, int(xcnt * (target_ratio - search_frac)))
    hi = int(xcnt * (target_ratio + search_frac)) + 1
    lo += lo % 2
    best = None
    for ycnt in range(lo, hi, 2):
        for zcnt in range(lo, hi, 2):
            xs, ys, zs = model.get_box_dim_fcc_3d(1, xcnt, ycnt, zcnt)
            score = abs(ys / xs - target_ratio) + abs(zs / xs - target_ratio)
            if best is None or score < best[0]:
                best = (score, ycnt, zcnt)
    return best[1], best[2]


ycnt, zcnt = best_transverse_counts(model, resol)

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, ycnt, zcnt)
dr = 3.0 * wavelength / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, ycnt, zcnt)
print(f"Box dims: xs={xs} ys={ys} zs={zs} (target ys=zs={xs / 2})")

box_min = (-xs / 2, -ys / 2, -zs / 2)
box_max = (xs / 2, ys / 2, zs / 2)

model.resize_simulation_box(box_min, box_max)
model.add_cube_fcc_3d(dr, box_min, box_max)

gam1 = gamma - 1.0
uuzero = P0 / (gam1 * rho0)
model.set_value_in_a_box("uint", "f64", uuzero, box_min, box_max)


def wave_frame_coords(r):
    x, y, z = r
    x1 = x * r_hat[0] + y * r_hat[1] + z * r_hat[2]
    return x1


def vel_func(r):
    x1 = wave_frame_coords(r)
    v1 = 0.0
    v2 = amplitude * np.sin(2.0 * np.pi * x1 / wavelength)
    v3 = amplitude * np.cos(2.0 * np.pi * x1 / wavelength)
    vx, vy, vz = rotmat @ np.array([v1, v2, v3])
    return (vx, vy, vz)


def mag_func(r):
    x1 = wave_frame_coords(r)
    B1 = 1.0
    B2 = amplitude * np.sin(2.0 * np.pi * x1 / wavelength)
    B3 = amplitude * np.cos(2.0 * np.pi * x1 / wavelength)
    Bx, By, Bz = rotmat @ np.array([B1, B2, B3])
    # field is stored as B/rho in SPMHD
    return (Bx / rho0, By / rho0, Bz / rho0)


model.set_field_value_lambda_f64_3("vxyz", vel_func)
model.set_field_value_lambda_f64_3("B/rho", mag_func)

vol_b = xs * ys * zs
totmass = rho0 * vol_b
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

print("Total mass :", totmass)
print("Current part mass :", pmass)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

model.timestep()

# %%
# Run the simulation for n_periods periods

model.evolve_until(t_target)

# %%
# Compare the transverse field component B2 against the exact solution.
#
# Because t_target is an integer number of wave periods, the exact
# (undamped) solution coincides with the initial condition:
#   B2_exact(x1) = amplitude * sin(2*pi*x1/wavelength)

data = ctx.collect_data()

xyz = data["xyz"]
B_on_rho = data["B/rho"]

x1 = xyz @ r_hat
B2 = (
    B_on_rho @ e2_hat
)  # B/rho projected on the transverse axis (rho = 1 at t=0, close to 1 during the test)

B2_exact = amplitude * np.sin(2.0 * np.pi * x1 / wavelength)

l2_err_B2 = np.sqrt(np.mean((B2 - B2_exact) ** 2))

print(f"L2 error on B2 : {l2_err_B2}")

test_pass = True
err_log = ""

expect_l2_err_B2 = 0.07122916744802753
tol = 0.35  # too generous for now


def float_equal(val1, val2, prec):
    return abs(val1 - val2) < prec


if not float_equal(l2_err_B2, expect_l2_err_B2, tol * expect_l2_err_B2):
    err_log += "error on the L2 norm of B2 is outside of tolerances:\n"
    err_log += f"  expected error = {expect_l2_err_B2} +- {tol * expect_l2_err_B2}\n"
    err_log += f"  obtained error = {l2_err_B2} (relative error = {(l2_err_B2 - expect_l2_err_B2) / expect_l2_err_B2})\n"
    test_pass = False

if test_pass == False:
    exit("Test did not pass L2 margins : \n" + err_log)
