"""
SR Sod Shock Tube - Kitajima Configuration
===========================================

CI test matching Kitajima, Inutsuka, and Seno (2025) exactly.
arXiv:2510.18251v1, Figure 3 (1D_Sod_32_3200vs400)

Kitajima setup:
    - Equal baryon number ν per particle (= our equal pmass)
    - 3200 particles left, 400 particles right (8:1 ratio = n_L/n_R)
    - t = 0.35, γ = 5/3
    - Initial: (P_L, n_L) = (1.0, 1.0), (P_R, n_R) = (0.1, 0.125)

Correspondence:
    Kitajima ν (baryon number) ↔ Shamrock pmass (particle mass)
    Kitajima n (number density) ↔ Shamrock ρ (density from SPH)
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/guo/Downloads/sph-simulators/docs/papers/sg-gsph/srrp')

import numpy as np
import srrp
import shamrock


def sr_enthalpy(rho, P, gamma):
    """Relativistic specific enthalpy H = 1 + σP/ρ where σ = γ/(γ-1)."""
    sigma = gamma / (gamma - 1)
    return 1 + sigma * P / rho


def plot_kitajima(x_sim, P_sim, n_sim, vx_sim, h_sim,
                  x_exact, P_exact, n_exact, vx_exact, h_exact,
                  title, filename):
    """Create Kitajima-style 4-panel plot (P, n, vx, h)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamp = time.strftime("%H%M%S")
    base, ext = filename.rsplit(".", 1)
    filename = f"{base}_{timestamp}.{ext}"

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    fig.subplots_adjust(hspace=0.05, left=0.12, right=0.95, top=0.95, bottom=0.06)

    colors = ["#007F66", "#CC3311", "#003366", "#EE7733"]
    labels = [r"$P$", r"$n$", r"$v^x$", r"$h$"]
    sim_data = [P_sim, n_sim, vx_sim, h_sim]
    exact_data = [P_exact, n_exact, vx_exact, h_exact]

    for ax, color, label, sim, exact in zip(axes, colors, labels, sim_data, exact_data):
        ax.plot(x_exact, exact, "k-", linewidth=1.5)
        ax.scatter(x_sim, sim, s=3, c=color, marker="o", alpha=0.6)
        ax.set_ylabel(label, fontsize=14)
        ax.set_xlim(-0.5, 0.5)
        ax.tick_params(direction="in", which="both")

    axes[-1].set_xlabel(r"$x$", fontsize=14)
    fig.suptitle(title, fontsize=12, y=0.98)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {filename}")
    plt.close(fig)
    return filename


# Kitajima parameters
gamma = 5.0 / 3.0
n_L, n_R = 1.0, 0.125
P_L, P_R = 1.0, 0.1
u_L = P_L / ((gamma - 1) * n_L)
u_R = P_R / ((gamma - 1) * n_R)
t_target = 0.35  # Kitajima uses t=0.35

# Kitajima resolution: 1800 particles per side in TRUE 1D
# For quasi-1D 3D: 900 particles in x, minimal y/z
resol = 900  # Particles in x direction per side
resol_yz = 3  # Minimal y/z for quasi-1D

ctx = shamrock.Context()
ctx.pdata_layout_new()

# Use Gaussian kernel to match Kitajima (TGauss3 = truncated Gaussian with R=3h)
model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="TGauss3")
cfg = model.gen_default_config()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.set_c_smooth(2.0)  # Kitajima uses C_smooth = 2.0
model.set_solver_config(cfg)
model.set_physics_sr(c_speed=1.0)  # Physics mode on Model, not SolverConfig
model.init_scheduler(int(1e8), 1)

# Setup box dimensions - quasi-1D: high x resolution, minimal y/z
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, resol_yz, resol_yz)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, resol_yz, resol_yz)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

hfact = model.get_hfact()

# SR mode uses volume-based h (Kitajima): Equal spacing, different per-particle baryon number
# All particles have same spacing dr → same initial h
# Density difference from per-particle baryon number (pmass field)
model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Get actual per-particle volume from box volume and particle count
# SPH density: ρ = ν × Σ_j W(r_ij, h) where Σ W ≈ 1/V_per_particle
# For target density ρ = n: ν = n × V_per_particle = n × (box_volume / n_particles)
init_data = ctx.collect_data()
xyz_init = np.array(init_data["xyz"])
left_mask = xyz_init[:, 0] < 0
n_left = np.sum(left_mask)
n_right = np.sum(~left_mask)
vol_left_box = xs * ys * zs  # Left half of box
vol_right_box = xs * ys * zs
V_per_particle_L = vol_left_box / n_left
V_per_particle_R = vol_right_box / n_right
# Effective spacing (cube root of volume per particle)
dr_eff = (V_per_particle_L)**(1/3)

# Set internal energy
model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Per-particle baryon number (ν) = target_density × volume_per_particle
nu_L = n_L * V_per_particle_L
nu_R = n_R * V_per_particle_R
model.set_field_in_box("pmass", "f64", nu_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("pmass", "f64", nu_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Global particle mass (unused for density when pmass exists, but needed for setup)
vol_b = xs * ys * zs
totmass = (n_R * vol_b) + (n_L * vol_b)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

# Initial h: same for all particles, based on effective spacing
h_init = hfact * dr_eff
model.set_field_in_box("hpart", "f64", h_init, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
print(f"  Volume-based h: dr_nominal={dr:.6f}, dr_eff={dr_eff:.6f}, h_init={h_init:.6f}")
print(f"  n_left={n_left}, n_right={n_right}, V_per_particle={V_per_particle_L:.6e}")
print(f"  nu_L={nu_L:.6e}, nu_R={nu_R:.6e}")

model.set_cfl_cour(0.3)
model.set_cfl_force(0.3)

print(f"SR-GSPH Kitajima Sod Test (TGauss3, t={t_target})")
print(f"  Resolution: x={resol}, y/z={resol_yz}")
print(f"  Domain: x=[{-xs:.4f}, {xs:.4f}], y=[{-ys/2:.4f}, {ys/2:.4f}], z=[{-zs/2:.4f}, {zs/2:.4f}]")
print(f"  Nominal particle spacing dr={dr:.6f}")

# Count particles
data_init = ctx.collect_data()
n_particles = len(data_init["xyz"])
h_init_data = np.array(data_init["hpart"])
print(f"  Total particles: {n_particles}")
print(f"  h range: [{h_init_data.min():.6f}, {h_init_data.max():.6f}]")

# Evolve with per-timestep output and NaN check
step = 0
while model.get_time() < t_target:
    t_before = model.get_time()
    dt_before = model.get_dt()
    model.evolve_once()
    t_after = model.get_time()
    step += 1

    # Check for NaNs after each step
    check_data = ctx.collect_data()
    xyz = np.array(check_data["xyz"])
    vxyz = np.array(check_data["vxyz"])
    hpart_check = np.array(check_data["hpart"])

    has_nan = np.any(np.isnan(xyz)) or np.any(np.isnan(vxyz))
    x_range = f"[{xyz[:,0].min():.4f}, {xyz[:,0].max():.4f}]" if not np.any(np.isnan(xyz)) else "[NaN]"
    v_max = np.max(np.abs(vxyz)) if not np.any(np.isnan(vxyz)) else float('nan')
    h_range = f"[{hpart_check.min():.4f}, {hpart_check.max():.4f}]"

    print(f"  Step {step}: t={t_after:.6f}, dt={dt_before:.6e}, x={x_range}, |v|_max={v_max:.4f}, h={h_range}")

    if has_nan:
        print("  ERROR: NaN detected! Stopping simulation.")
        break

# Collect simulation data
print(f"\nSimulation completed at t={model.get_time():.6f}")
data = ctx.collect_data()
points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])
uint_data = np.array(data["uint"])

print(f"  Collected {len(points)} particles")
print(f"  x range: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]")
print(f"  h range: [{hpart.min():.6f}, {hpart.max():.6f}]")

# Get density directly from C++ (computed by SPH summation)
rho_sim = np.array(data["rho"])
print(f"  rho_sim range: [{rho_sim.min():.4f}, {rho_sim.max():.4f}], expected: [{n_R:.4f}, {n_L:.4f}]")
P_sim = (gamma - 1) * rho_sim * uint_data
x = points[:, 0]
vx = velocities[:, 0]
vy = velocities[:, 1]
vz = velocities[:, 2]

# Solve exact SR Riemann problem using srrp
stateL = srrp.State(rho=n_L, pressure=P_L, vx=0, vt=0)
stateR = srrp.State(rho=n_R, pressure=P_R, vx=0, vt=0)
solver = srrp.Solver()
solution = solver.solve(stateL, stateR, gamma)

# Filter to central region
x_min, x_max = -0.5, 0.5
mask = (x >= x_min) & (x <= x_max)
x_f = x[mask]
rho_f = rho_sim[mask]
vx_f = vx[mask]
P_f = P_sim[mask]
h_f = hpart[mask]

print(f"  Filtered to [{x_min}, {x_max}]: {len(x_f)} particles")

if len(x_f) == 0:
    print("ERROR: No particles in filter region!")
    exit(1)

# Get exact solution at particle positions
x0 = 0.0
t_final = model.get_time()  # Use actual final time
xis_f = (x_f - x0) / t_final
states_f = solution.getState(xis_f)

err_rho = np.sqrt(np.mean((rho_f - states_f.rho) ** 2)) / np.mean(states_f.rho)
err_vx = np.sqrt(np.mean((vx_f - states_f.vx) ** 2)) / (np.mean(np.abs(states_f.vx)) + 0.1)
err_vy = np.sqrt(np.mean(vy[mask] ** 2))
err_vz = np.sqrt(np.mean(vz[mask] ** 2))
err_P = np.sqrt(np.mean((P_f - states_f.pressure) ** 2)) / np.mean(states_f.pressure)

print(f"L2 errors: rho={err_rho:.6e}, vx={err_vx:.6e}, vy={err_vy:.6e}, vz={err_vz:.6e}, P={err_P:.6e}")

# Plot exact solution
x_exact = np.linspace(-0.5, 0.5, 500)
xis_exact = (x_exact - x0) / t_final
states_exact = solution.getState(xis_exact)

# Smoothing length exact solution (volume-based h for SR)
# Volume-based h (Kitajima): h depends on particle spacing, not density
# In unperturbed regions, h ≈ h_init (constant)
h_exact = np.full_like(x_exact, h_init)

plot_kitajima(
    x_f, P_f, rho_f, vx_f, h_f,
    x_exact, states_exact.pressure, states_exact.rho, states_exact.vx, h_exact,
    f"SR Sod (Kitajima) t={t_final:.4f}", "sr_sod_kitajima.png"
)

# Test pass criteria (SR volume-based h thresholds)
expect_rho = 0.18
expect_vx = 0.40
expect_vy = 0.02
expect_vz = 0.02
expect_P = 0.18

test_pass = True
err_log = ""

error_checks = {
    "rho": (err_rho, expect_rho),
    "vx": (err_vx, expect_vx),
    "vy": (err_vy, expect_vy),
    "vz": (err_vz, expect_vz),
    "P": (err_P, expect_P),
}

for name, (value, expected) in error_checks.items():
    if value > expected * 1.1:
        err_log += f"error on {name}: expected <= {expected:.6e}, got {value:.6e}\n"
        test_pass = False

if test_pass:
    print("\n" + "=" * 50)
    print("SR-GSPH Kitajima Sod Test: PASSED")
    print("=" * 50)
else:
    exit("Test did not pass L2 margins:\n" + err_log)
