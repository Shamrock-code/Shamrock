#!/usr/bin/env python3
"""
SR-GSPH Problem 6: 2D Sod Problem (Kitajima et al. 2025)

2D Sod shock tube test - same physics as Problem 1 but in 2D.
Initial conditions:
  Left:  (P, n, vx) = (1.0, 1.0, 0)
  Right: (P, n, vx) = (0.1, 0.125, 0)
  gamma = 5/3, t_end = 0.35
"""
import sys
from pathlib import Path

import numpy as np

import shamrock

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from kitajima_plotting import plot_kitajima_4panel

# Kitajima parameters (same as Problem 1)
gamma = 5.0 / 3.0
n_L, n_R = 1.0, 0.125
P_L, P_R = 1.0, 0.1
u_L = P_L / ((gamma - 1) * n_L)
u_R = P_R / ((gamma - 1) * n_R)
t_target = 0.35

# Resolution - higher y resolution for 2D
resol_x = 80
resol_yz = 16  # More particles in y for proper 2D

print("SR 2D Sod Test (Kitajima Problem 6)")

ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="TGauss3")

cfg = model.gen_default_config()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.set_c_smooth(2.0)
model.set_solver_config(cfg)
model.set_physics_sr(c_speed=1.0)
model.init_scheduler(int(1e8), 1)

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol_x, resol_yz, resol_yz)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol_x, resol_yz, resol_yz)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
hfact = model.get_hfact()

# Add particles
model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

init_data = ctx.collect_data()
xyz_init = np.array(init_data["xyz"])
n_left = np.sum(xyz_init[:, 0] < 0)
n_right = np.sum(xyz_init[:, 0] >= 0)
V_per_particle = xs * ys * zs / n_left

nu_L = n_L * V_per_particle
nu_R = n_R * V_per_particle
model.set_field_in_box("pmass", "f64", nu_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("pmass", "f64", nu_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

vol_total = 2 * xs * ys * zs
totmass = n_L * xs * ys * zs + n_R * xs * ys * zs
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

h_init = hfact * V_per_particle ** (1 / 3)
model.set_field_in_box("hpart", "f64", h_init, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.set_cfl_cour(0.3)
model.set_cfl_force(0.3)

print(f"  Particles: {n_left + n_right}, domain: [{-xs:.3f}, {xs:.3f}]")
print(f"  Running to t={t_target}...")
model.evolve_until(t_target)

# Collect data
data = ctx.collect_data()
physics = model.collect_physics_data()

points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])

x = points[:, 0]
vx = velocities[:, 0]

# Compute Lorentz factor from velocity (c=1)
v2 = np.sum(velocities**2, axis=1)
gamma_lor = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-10))

# physics["N_labframe"] contains LAB-FRAME N from kernel summation
# Convert to REST-FRAME n = N/γ for comparison
N_sim = np.array(physics["N_labframe"])
n_sim = N_sim / gamma_lor
P_sim = np.array(physics["pressure"])

# Reference solution (1D exact)
x_ref = np.linspace(-0.5, 0.5, 100)
P_ref = np.where(x_ref < 0, P_L, P_R)
n_ref = np.where(x_ref < 0, n_L, n_R)
vx_ref = np.zeros_like(x_ref)

print("\nResults:")
print(f"  P range: [{np.min(P_sim):.4f}, {np.max(P_sim):.4f}]")
print(f"  n range: [{np.min(n_sim):.4f}, {np.max(n_sim):.4f}]")
print(f"  vx range: [{np.min(vx):.4f}, {np.max(vx):.4f}]")

# Plot
plot_kitajima_4panel(
    x,
    P_sim,
    n_sim,
    vx,
    hpart,
    x_ref,
    P_ref,
    n_ref,
    vx_ref,
    "sr_2d_sod_problem6.png",
    f"SR 2D Sod (t={t_target})",
    h0=h_init,
)

# Basic validation
test_pass = True
errors = []

if np.min(P_sim) < 0:
    test_pass = False
    errors.append("Negative pressure")

if np.min(n_sim) < 0:
    test_pass = False
    errors.append("Negative density")

if not np.all(np.isfinite(vx)):
    test_pass = False
    errors.append("NaN in velocity")

if test_pass:
    print("\n" + "=" * 50)
    print("SR 2D Sod Problem 6: PASSED")
    print("=" * 50)
else:
    print("\n" + "=" * 50)
    print("SR 2D Sod Problem 6: FAILED")
    for e in errors:
        print(f"  - {e}")
    print("=" * 50)
    exit(1)
