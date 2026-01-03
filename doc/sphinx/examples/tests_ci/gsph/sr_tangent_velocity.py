"""
Special Relativistic Tangent Velocity Test with SR-GSPH
========================================================

CI test for Kitajima et al. (2025) arXiv:2510.18251v1 Section 3.1.5
Riemann Problem 5: 1D Shock Tube With Tangential Velocity

Key Physics:
- In SR, tangent velocity v^t affects the solution via the Lorentz factor
- The K-invariant K = γHv_t is conserved across shocks
- γ = 1/√(1 - v_x² - v_t²) includes both velocity components

Initial Conditions (Kitajima Problem 5):
    Left:  (P, n, v^x, v^t) = (1000, 1.0, 0, 0.9)
    Right: (P, n, v^x, v^t) = (0.01, 1.0, 0, 0.9)

IMPORTANT Implementation Notes (from reference sr_tangent_velocity.cpp):
1. Baryon number per particle: ν = γ × n × dx (accounts for lab-frame spacing)
2. Lorentz factor: γ = 1/√(1 - v_x² - v_t²) (includes tangent velocity!)
3. For 3D shamrock: tangent velocity maps to y-component of vxyz
4. Enthalpy: H = 1 + γ_eos/(γ_eos-1) × P/n (in natural units c=1)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import shamrock


# =============================================================================
# Physical Constants and Parameters
# =============================================================================

gamma_eos = 5.0 / 3.0  # Adiabatic index
c_speed = 1.0          # Speed of light (natural units)

# Kitajima Problem 5 (Section 3.1.5) - Strong blast with tangent velocity
P_L, P_R = 1000.0, 0.01   # Pressure
n_L, n_R = 1.0, 1.0       # Rest-frame number density
vx_L, vx_R = 0.0, 0.0     # Normal velocity
vt_L, vt_R = 0.9, 0.9     # Tangent velocity (same on both sides)

# Resolution - N particles per side (reference uses 800+800)
N_per_side = 400

# Simulation time - run until shock reaches ~x=0.2
t_target = 0.1


# =============================================================================
# Compute Derived Quantities (Following Reference Implementation)
# =============================================================================

def compute_lorentz_factor(vx, vt, c=1.0):
    """Lorentz factor including tangent velocity: γ = 1/√(1 - v_x² - v_t²)"""
    v2 = vx**2 + vt**2
    if v2 >= c**2:
        raise ValueError(f"Superluminal velocity: v² = {v2} >= c² = {c**2}")
    return 1.0 / np.sqrt(1.0 - v2 / c**2)


def compute_enthalpy(P, n, gamma_eos, c=1.0):
    """Specific enthalpy: H = 1 + (γ/(γ-1)) × P/n (natural units)"""
    return 1.0 + (gamma_eos / (gamma_eos - 1.0)) * P / n


def compute_internal_energy(P, n, gamma_eos):
    """Internal energy per unit mass: u = P / ((γ-1) × n)"""
    return P / ((gamma_eos - 1.0) * n)


def compute_sound_speed(P, n, H, gamma_eos, c=1.0):
    """Relativistic sound speed: c_s = √(γ(γ-1)(H-1)/(γH)) × c"""
    # From cs² = (γ-1)(H-1)/H × c²
    cs2 = (gamma_eos - 1.0) * (H - 1.0) / H * c**2
    return np.sqrt(max(cs2, 0.0))


# Compute initial state quantities
gamma_L = compute_lorentz_factor(vx_L, vt_L, c_speed)
gamma_R = compute_lorentz_factor(vx_R, vt_R, c_speed)

H_L = compute_enthalpy(P_L, n_L, gamma_eos, c_speed)
H_R = compute_enthalpy(P_R, n_R, gamma_eos, c_speed)

u_L = compute_internal_energy(P_L, n_L, gamma_eos)
u_R = compute_internal_energy(P_R, n_R, gamma_eos)

cs_L = compute_sound_speed(P_L, n_L, H_L, gamma_eos, c_speed)
cs_R = compute_sound_speed(P_R, n_R, H_R, gamma_eos, c_speed)

# K-invariant: K = γ × H × v_t (should be conserved across shocks)
K_L = gamma_L * H_L * vt_L
K_R = gamma_R * H_R * vt_R


print("=" * 70)
print("SR-GSPH Tangent Velocity Test (Kitajima et al. 2025, Section 3.1.5)")
print("=" * 70)
print(f"\nLeft state:")
print(f"  P = {P_L}, n = {n_L}, v_x = {vx_L}, v_t = {vt_L}")
print(f"  γ = {gamma_L:.4f}, H = {H_L:.4f}, u = {u_L:.4f}, c_s = {cs_L:.4f}")
print(f"  K-invariant: K = γHv_t = {K_L:.4f}")
print(f"\nRight state:")
print(f"  P = {P_R}, n = {n_R}, v_x = {vx_R}, v_t = {vt_R}")
print(f"  γ = {gamma_R:.4f}, H = {H_R:.4f}, u = {u_R:.4f}, c_s = {cs_R:.4f}")
print(f"  K-invariant: K = γHv_t = {K_R:.4f}")
print(f"\nResolution: {N_per_side} particles per side")
print(f"Target time: t = {t_target}")


# =============================================================================
# Setup Shamrock Simulation
# =============================================================================

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
cfg = model.gen_default_config()

# Configure solver
cfg.set_riemann_hllc()
cfg.set_reconstruct_piecewise_constant()  # First order (diffusive but stable)
# IMPORTANT: Use periodic boundaries for 3D simulation of 1D shock tube
# This ensures particles at y/z edges have correct neighbors
# The x-domain is large enough that shocks don't wrap around by t=0.4
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma_eos)
# IMPORTANT: Disable iterative smoothing length!
# Reference implementation skips h iteration on first timestep to preserve initial h.
# Since we set h explicitly (h = dr), we don't want shamrock to change it.
cfg.set_sr(c_speed=c_speed, iterative_sml=False)

cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)


# =============================================================================
# Particle Setup (Following Reference Implementation Principles)
# =============================================================================

# Domain: x ∈ [-1, 1], y and z thin slabs
# For 3D shamrock, we use HCP packing but the physics is essentially 1D

# Get box dimensions for target resolution
# Use 8,8 layers in y/z for proper 3D resolution with periodic boundaries
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, N_per_side, 8, 8)
dr = 1.0 / xs  # Particle spacing
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, N_per_side, 8, 8)

# Resize simulation box
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Add particles for left and right regions
model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Set internal energy (determines pressure via EOS)
model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Set velocity: (v_x, v_t, 0) where v_t maps to y-component
# This is the key for tangent velocity test in 3D
model.set_field_in_box("vxyz", "f64_3", (vx_L, vt_L, 0.0), (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("vxyz", "f64_3", (vx_R, vt_R, 0.0), (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# CRITICAL: Set initial smoothing length explicitly!
# Reference implementation (sr_tangent_velocity.cpp line 219): p_i.sml = dx
# This is ESSENTIAL - reference skips h iteration on first timestep to preserve this value!
# Without this, shamrock's 3D iterative h calculation gives wrong values.
h_init = dr  # Smoothing length = particle spacing (as in reference)
model.set_field_in_box("hpart", "f64", h_init, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
print(f"Initial smoothing length: h = {h_init:.6f}")

# Compute particle mass (baryon number per particle)
# Reference: ν = γ × n × dx for proper baryon number accounting
# The SPH sum gives lab-frame density N = γ × n
# So total "mass" (baryon number) = γ × n × Volume
vol_total = 2 * xs * ys * zs
# Since γ_L = γ_R for this test (same tangent velocity), mass is uniform
# CRITICAL: Include Lorentz factor for correct lab-frame density!
totmass = gamma_L * n_L * vol_total

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

# CFL parameters (from Kitajima paper)
# IMPORTANT: Kitajima only uses sound CFL, NOT force CFL!
# "we find that a simpler time step criterion... works well"
# Δt = C_CFL * min_i(h_i / c_{s,i})
model.set_cfl_cour(0.15)
model.set_cfl_force(1e10)  # Disable force CFL by setting very large value

print(f"\nParticle mass: {pmass:.6e}")
print(f"hfact: {hfact:.4f}")
print(f"Domain: x ∈ [{-xs:.4f}, {xs:.4f}]")


# =============================================================================
# SR Exact Riemann Solver (for analytic comparison)
# =============================================================================

def sr_exact_riemann_simple(P_L, n_L, vx_L, vt_L, P_R, n_R, vx_R, vt_R, gamma_eos):
    """
    Simplified SR Riemann solver for plotting comparison.
    Returns approximate star state values.
    """
    # For strong blast (P_L >> P_R), the star pressure is close to geometric mean
    # This is approximate - full solver would iterate
    P_star = np.sqrt(P_L * P_R) * 2  # Rough estimate

    # Star velocity (approximate)
    H_L = compute_enthalpy(P_L, n_L, gamma_eos)
    H_R = compute_enthalpy(P_R, n_R, gamma_eos)
    cs_L = compute_sound_speed(P_L, n_L, H_L, gamma_eos)
    cs_R = compute_sound_speed(P_R, n_R, H_R, gamma_eos)

    # Simple estimate based on pressure jump
    vx_star = 0.5 * (vx_L + vx_R) + 0.3  # Rough estimate for this problem

    # Tangent velocity from K-invariant
    gamma_star = compute_lorentz_factor(vx_star, vt_L, 1.0)
    H_star = compute_enthalpy(P_star, n_L, gamma_eos)
    K = gamma_L * H_L * vt_L
    vt_star = K / (gamma_star * H_star)

    return P_star, vx_star, vt_star


# =============================================================================
# Run Simulation
# =============================================================================

print(f"\nRunning SR-GSPH simulation to t={t_target}...")
model.evolve_until(t_target)


# =============================================================================
# Collect Results
# =============================================================================

print("\nCollecting results...")
data = ctx.collect_data()

points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])
uint_data = np.array(data["uint"])

x = points[:, 0]
vx = velocities[:, 0]
vy = velocities[:, 1]  # Tangent velocity (y-component)
vz = velocities[:, 2]

# Compute Lorentz factor from full velocity
v_mag_sq = vx**2 + vy**2 + vz**2
gamma_lorentz = 1.0 / np.sqrt(1.0 - np.clip(v_mag_sq, 0, 0.9999))

# Rest-frame density from SPH sum
# N_lab = pmass × Σ W (SPH sum gives lab-frame density)
# n = N_lab / γ (rest-frame density)
N_lab = pmass * (hfact / hpart) ** 3
n_sim = N_lab / gamma_lorentz

# Pressure from EOS: P = (γ-1) × n × u
P_sim = (gamma_eos - 1.0) * n_sim * uint_data

# Enthalpy: H = 1 + γ_eos × u (in natural units)
H_sim = 1.0 + gamma_eos * uint_data

# K-invariant: K = γ × H × v_t
K_sim = gamma_lorentz * H_sim * vy


# =============================================================================
# Generate Plots (Kitajima Style - 2x3 Grid)
# =============================================================================

output_dir = "/Users/guo/Downloads/sph-simulators/simulations_data/sr_tangent"
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'SR-GSPH Tangent Velocity Test (Kitajima Problem 5)\n' +
             rf'$P_L={P_L}$, $P_R={P_R}$, $v_t={vt_L}$, $\gamma_L={gamma_L:.3f}$, $t={t_target}$',
             fontsize=14)

# Row 1: Primary variables
# Density
ax = axes[0, 0]
ax.plot(x, n_sim, 'b.', ms=1, alpha=0.3, label='SR-GSPH')
ax.axhline(y=n_L, color='r', ls='--', alpha=0.5, label=f'Initial n={n_L}')
ax.set_xlabel('x')
ax.set_ylabel('Rest-Frame Density n')
ax.set_title('Density')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(x.min(), x.max())
ax.grid(True, alpha=0.3)

# Pressure
ax = axes[0, 1]
ax.semilogy(x, P_sim, 'b.', ms=1, alpha=0.3, label='SR-GSPH')
ax.axhline(y=P_L, color='r', ls='--', alpha=0.5, label=f'$P_L$={P_L}')
ax.axhline(y=P_R, color='g', ls='--', alpha=0.5, label=f'$P_R$={P_R}')
ax.set_xlabel('x')
ax.set_ylabel('Pressure P')
ax.set_title('Pressure (log scale)')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(x.min(), x.max())
ax.grid(True, alpha=0.3)

# Normal Velocity
ax = axes[0, 2]
ax.plot(x, vx, 'b.', ms=1, alpha=0.3, label='SR-GSPH')
ax.axhline(y=0, color='gray', ls='-', alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel(r'Normal Velocity $v^x$')
ax.set_title('Normal Velocity')
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(x.min(), x.max())
ax.grid(True, alpha=0.3)

# Row 2: SR-specific quantities
# Tangent Velocity
ax = axes[1, 0]
ax.plot(x, vy, 'b.', ms=1, alpha=0.3, label='SR-GSPH')
ax.axhline(y=vt_L, color='r', ls='--', alpha=0.7, label=f'Initial $v_t$={vt_L}')
ax.set_xlabel('x')
ax.set_ylabel(r'Tangent Velocity $v^t$')
ax.set_title('Tangent Velocity (K-invariant test)')
ax.legend(loc='lower right', fontsize=8)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0.4, 1.0)
ax.grid(True, alpha=0.3)

# Lorentz Factor
ax = axes[1, 1]
ax.plot(x, gamma_lorentz, 'b.', ms=1, alpha=0.3, label='SR-GSPH')
ax.axhline(y=gamma_L, color='r', ls='--', alpha=0.5, label=f'Initial $\gamma$={gamma_L:.3f}')
ax.set_xlabel('x')
ax.set_ylabel(r'Lorentz Factor $\gamma$')
ax.set_title('Lorentz Factor')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(x.min(), x.max())
ax.grid(True, alpha=0.3)

# K-invariant
ax = axes[1, 2]
ax.plot(x, K_sim, 'b.', ms=1, alpha=0.3, label='SR-GSPH')
ax.axhline(y=K_L, color='r', ls='--', alpha=0.7, label=f'Initial K={K_L:.2f}')
ax.set_xlabel('x')
ax.set_ylabel(r'K-invariant $K = \gamma H v_t$')
ax.set_title('K-invariant (should be conserved)')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(x.min(), x.max())
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(output_dir, "sr_tangent_velocity_test.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")


# =============================================================================
# Print Statistics
# =============================================================================

print("\n" + "=" * 70)
print("Simulation Statistics")
print("=" * 70)
print(f"Particles: {len(x)}")
print(f"x range: [{x.min():.4f}, {x.max():.4f}]")
print(f"\nVelocity:")
print(f"  v_x range: [{vx.min():.4f}, {vx.max():.4f}]")
print(f"  v_t range: [{vy.min():.4f}, {vy.max():.4f}]")
print(f"\nDensity & Pressure:")
print(f"  n range: [{n_sim.min():.4f}, {n_sim.max():.4f}]")
print(f"  P range: [{P_sim.min():.4f}, {P_sim.max():.4f}]")
print(f"\nRelativistic quantities:")
print(f"  γ range: [{gamma_lorentz.min():.4f}, {gamma_lorentz.max():.4f}]")
print(f"  K range: [{K_sim.min():.4f}, {K_sim.max():.4f}]")

# Tangent velocity statistics
vt_mean = np.mean(vy)
vt_std = np.std(vy)
vt_err = abs(vt_mean - vt_L) / vt_L * 100

# K-invariant statistics
K_mean = np.mean(K_sim)
K_std = np.std(K_sim)
K_err = abs(K_mean - K_L) / K_L * 100

print(f"\n--- K-invariant Conservation ---")
print(f"Initial K = {K_L:.4f}")
print(f"Final K: mean = {K_mean:.4f}, std = {K_std:.4f}")
print(f"K error: {K_err:.2f}%")

print(f"\n--- Tangent Velocity Preservation ---")
print(f"Initial v_t = {vt_L}")
print(f"Final v_t: mean = {vt_mean:.4f}, std = {vt_std:.4f}")
print(f"v_t error: {vt_err:.2f}%")

print("\n" + "=" * 70)
print("SR-GSPH Tangent Velocity Test: COMPLETE")
print("=" * 70)
