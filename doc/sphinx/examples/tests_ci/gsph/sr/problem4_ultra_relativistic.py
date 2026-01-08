"""
Problem 4: Ultra-Relativistic Shock (Kitajima et al. 2025, Section 3.1.4)
=========================================================================

High-velocity shock collision test (γ ≈ 7).

Initial conditions:
    Left:  (P, n, v^x, v^t) = (1.0, 1.0, 0.99, 0)
    Right: (P, n, v^x, v^t) = (1.0, 1.0, 0, 0)

Kitajima setup:
    - 1000 particles each side
    - t = 0.3, γ = 5/3
"""

import sys
from pathlib import Path
import numpy as np
import shamrock

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from kitajima_plotting import plot_kitajima_4panel

# Kitajima Problem 4 parameters
gamma = 5.0 / 3.0
n_L, n_R = 1.0, 1.0
P_L, P_R = 1.0, 1.0
v_L = 0.99  # Ultra-relativistic velocity (γ ≈ 7)
u_L = P_L / ((gamma - 1) * n_L)
u_R = P_R / ((gamma - 1) * n_R)
t_target = 0.3

# Resolution (quasi-1D)
resol_x = 80  # Reduced for faster testing
resol_yz = 4

print("SR Ultra-Relativistic Shock Test (Kitajima Problem 4)")
print(f"  v_left = {v_L}, γ_left = {1/np.sqrt(1 - v_L**2):.2f}")

ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="TGauss3")

cfg = model.gen_default_config()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()  # PERIODIC - FREE has issues
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

model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Set velocity on left side
model.set_field_in_box("vxyz", "f64_3", (v_L, 0.0, 0.0), (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))

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

h_init = hfact * V_per_particle**(1/3)
model.set_field_in_box("hpart", "f64", h_init, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.set_cfl_cour(0.15)
model.set_cfl_force(0.15)

print(f"  Particles: {n_left + n_right}, domain: [{-xs:.3f}, {xs:.3f}]")
print(f"  Running to t={t_target}...")
model.evolve_until(t_target)

# Collect data - use direct physics from solver
data = ctx.collect_data()
physics = model.collect_physics_data()

points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])

x = points[:, 0]
vx = velocities[:, 0]

# Direct values from solver
n_sim = np.array(physics["density"])
P_sim = np.array(physics["pressure"])

# Compute lorentz factor from velocity
v2 = np.sum(velocities**2, axis=1)
gamma_lor = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-10))

gamma_max = np.max(gamma_lor)
v_max = np.max(np.abs(vx))

print(f"\nResults:")
print(f"  γ_max = {gamma_max:.4f} (expected ~7 from initial v=0.99)")
print(f"  |v|_max = {v_max:.6f}")
print(f"  P range: [{np.min(P_sim):.4e}, {np.max(P_sim):.4e}]")
print(f"  n range: [{np.min(n_sim):.4e}, {np.max(n_sim):.4e}]")

# No exact solution available, use zeros for reference line
x_ref = np.linspace(-0.5, 0.5, 100)
P_ref = np.ones_like(x_ref)  # Initial pressure
n_ref = np.ones_like(x_ref)  # Initial density
vx_ref = np.zeros_like(x_ref)

# Plot
plot_kitajima_4panel(x, P_sim, n_sim, vx, hpart,
                     x_ref, P_ref, n_ref, vx_ref,
                     "sr_ultra_relativistic_problem4.png",
                     f"SR Ultra-Relativistic Shock (t={t_target}, v_L={v_L})")

# Regression test
test_pass = True
errors = []

if not np.isfinite(gamma_max):
    test_pass = False
    errors.append("NaN in Lorentz factor")

if gamma_max < 2.0:
    test_pass = False
    errors.append(f"γ_max too low: {gamma_max:.4f}")

if np.min(P_sim) <= 0 or np.max(P_sim) <= 0:
    test_pass = False
    errors.append(f"Invalid pressure range")

if test_pass:
    print("\n" + "=" * 50)
    print("SR Ultra-Relativistic Problem 4: PASSED")
    print("=" * 50)
else:
    print("\n" + "=" * 50)
    print("SR Ultra-Relativistic Problem 4: FAILED")
    for e in errors:
        print(f"  - {e}")
    print("=" * 50)
    exit(1)
