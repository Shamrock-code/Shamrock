"""
Problem 2: Standard Relativistic Blast Wave (Kitajima et al. 2025, Section 3.1.2)
================================================================================

CI test matching arXiv:2510.18251v1 Figure 4

Initial conditions:
    Left:  (P, n, v^x, v^t) = (40/3, 10, 0, 0)
    Right: (P, n, v^x, v^t) = (1e-6, 1, 0, 0)
    
Kitajima setup:
    - Equal baryon: 5000 left, 500 right (10:1 ratio = n_L/n_R)
    - Different baryon: 2750 vs 2750 (volume-based for right side)
    - t = 0.4, γ = 5/3
"""

import sys
from pathlib import Path
import numpy as np
import shamrock

# Add this directory to path for local imports
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

SRRP_PATH = THIS_DIR.parent.parent.parent.parent.parent.parent.parent / "docs/papers/sg-gsph/srrp"
sys.path.insert(0, str(SRRP_PATH))
from srrp.Solver import Solver
from srrp.State import State

from kitajima_plotting import plot_kitajima_4panel, compute_L2_errors

# Kitajima Problem 2 parameters
gamma = 5.0 / 3.0
n_L, n_R = 10.0, 1.0
P_L, P_R = 40.0 / 3.0, 1e-6
u_L = P_L / ((gamma - 1) * n_L)
u_R = P_R / ((gamma - 1) * n_R)
t_target = 0.4

# Resolution (quasi-1D) - reduced for fast testing
resol_x = 80
resol_yz = 4

# Solve exact Riemann problem
solver = Solver()
stateL = State(rho=n_L, vx=0.0, vt=0.0, pressure=P_L)
stateR = State(rho=n_R, vx=0.0, vt=0.0, pressure=P_R)
wavefan = solver.solve(stateL, stateR, gamma)

print("SR Standard Blast Wave Test (Kitajima Problem 2)")
print(f"  Star state: P*={wavefan.states[1].pressure:.6f}, v*={wavefan.states[1].vx:.6f}")

ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="TGauss3")

cfg = model.gen_default_config()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()  # PERIODIC - FREE has issues with extreme pressure ratio
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

# Uniform spacing, density from pmass
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

h_init = hfact * V_per_particle**(1/3)
model.set_field_in_box("hpart", "f64", h_init, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.set_cfl_cour(0.3)
model.set_cfl_force(0.3)

print(f"  Particles: {n_left + n_right}, domain: [{-xs:.3f}, {xs:.3f}]")
print(f"  Running to t={t_target}...")
model.evolve_until(t_target)

# Collect data - use direct physics from solver (no post-processing)
data = ctx.collect_data()
physics = model.collect_physics_data()

points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])

x = points[:, 0]
vx = velocities[:, 0]

# Direct values from solver (primitive recovery)
n_sim = np.array(physics["density"])
P_sim = np.array(physics["pressure"])

# Compute lorentz factor from velocity
v2 = np.sum(velocities**2, axis=1)
gamma_lor = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-10))

# Exact solution
x_exact = np.linspace(-0.5, 0.5, 1000)
xi = x_exact / t_target
state_exact = wavefan.getState(xi)
P_exact = state_exact.pressure
n_exact = state_exact.rho
vx_exact = state_exact.vx

# Compute errors
err_P = compute_L2_errors(x, P_sim, x_exact, P_exact)
err_n = compute_L2_errors(x, n_sim, x_exact, n_exact)
err_vx = compute_L2_errors(x, vx, x_exact, vx_exact)

print(f"\nL2 errors: rho={err_n:.6e}, vx={err_vx:.6e}, P={err_P:.6e}")

# Plot
plot_kitajima_4panel(x, P_sim, n_sim, vx, hpart,
                     x_exact, P_exact, n_exact, vx_exact,
                     "sr_blast_problem2.png", f"SR Blast Wave (t={t_target})")

# Regression test - tolerances tuned for extreme pressure ratio
expect_n = 0.35
expect_vx = 0.35
expect_P = 0.35
tol = 0.5

test_pass = (err_n < expect_n * (1 + tol) and
             err_vx < expect_vx * (1 + tol) and
             err_P < expect_P * (1 + tol))

if test_pass:
    print("\n" + "=" * 50)
    print("SR Blast Problem 2: PASSED")
    print("=" * 50)
else:
    print("\n" + "=" * 50)
    print("SR Blast Problem 2: FAILED")
    print(f"  err_n={err_n:.4f} (expect < {expect_n * (1 + tol):.4f})")
    print(f"  err_vx={err_vx:.4f} (expect < {expect_vx * (1 + tol):.4f})")
    print(f"  err_P={err_P:.4f} (expect < {expect_P * (1 + tol):.4f})")
    print("=" * 50)
    exit(1)
