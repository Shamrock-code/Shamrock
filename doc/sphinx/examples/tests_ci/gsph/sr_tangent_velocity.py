"""
Special Relativistic Tangent Velocity Test with SR-GSPH
========================================================

CI test for Kitajima et al. (2025) arXiv:2510.18251v1 Section 3.1.5
Riemann Problem 5: 1D Shock Tube With Tangential Velocity

Key Physics:
- In SR, tangent velocity v^t affects the solution via the Lorentz factor
- The K-invariant K = gamma*H*v_t is conserved across shocks
- gamma = 1/sqrt(1 - v_x^2 - v_t^2) includes both velocity components

Initial Conditions (Kitajima Problem 5):
    Left:  (P, n, v^x, v^t) = (1000, 1.0, 0, 0.9)
    Right: (P, n, v^x, v^t) = (0.01, 1.0, 0, 0.9)
"""

import numpy as np

import shamrock


# =============================================================================
# Physical Constants and Parameters
# =============================================================================

gamma_eos = 5.0 / 3.0
c_speed = 1.0

P_L, P_R = 1000.0, 0.01
n_L, n_R = 1.0, 1.0
vx_L, vx_R = 0.0, 0.0
vt_L, vt_R = 0.9, 0.9

N_per_side = 400
t_target = 0.1


def compute_lorentz_factor(vx, vt, c=1.0):
    """Lorentz factor including tangent velocity."""
    v2 = vx**2 + vt**2
    if v2 >= c**2:
        raise ValueError(f"Superluminal velocity: v^2 = {v2} >= c^2 = {c**2}")
    return 1.0 / np.sqrt(1.0 - v2 / c**2)


def compute_enthalpy(P, n, gamma_eos, c=1.0):
    """Specific enthalpy: H = 1 + (gamma/(gamma-1)) * P/n (natural units)"""
    return 1.0 + (gamma_eos / (gamma_eos - 1.0)) * P / n


def compute_internal_energy(P, n, gamma_eos):
    """Internal energy per unit mass: u = P / ((gamma-1) * n)"""
    return P / ((gamma_eos - 1.0) * n)


gamma_L = compute_lorentz_factor(vx_L, vt_L, c_speed)
gamma_R = compute_lorentz_factor(vx_R, vt_R, c_speed)

H_L = compute_enthalpy(P_L, n_L, gamma_eos, c_speed)
H_R = compute_enthalpy(P_R, n_R, gamma_eos, c_speed)

u_L = compute_internal_energy(P_L, n_L, gamma_eos)
u_R = compute_internal_energy(P_R, n_R, gamma_eos)

K_L = gamma_L * H_L * vt_L


print("SR-GSPH Tangent Velocity Test (Kitajima Problem 5)")
print(f"Left: P={P_L}, n={n_L}, v_x={vx_L}, v_t={vt_L}, gamma={gamma_L:.4f}")
print(f"Right: P={P_R}, n={n_R}, v_x={vx_R}, v_t={vt_R}, gamma={gamma_R:.4f}")


# =============================================================================
# Setup Simulation
# =============================================================================

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="TGauss3")
cfg = model.gen_default_config()

cfg.set_riemann_hllc()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma_eos)
cfg.set_sr(c_speed=c_speed)

model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, N_per_side, 8, 8)
dr = 1.0 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, N_per_side, 8, 8)

model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.set_field_in_box("vxyz", "f64_3", (vx_L, vt_L, 0.0), (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("vxyz", "f64_3", (vx_R, vt_R, 0.0), (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

h_init = dr
model.set_field_in_box("hpart", "f64", h_init, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

vol_total = 2 * xs * ys * zs
totmass = gamma_L * n_L * vol_total

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.15)
model.set_cfl_force(1e10)

print(f"Running simulation to t={t_target}...")
model.evolve_until(t_target)


# =============================================================================
# Collect and Analyze Results
# =============================================================================

data = ctx.collect_data()

points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])
uint_data = np.array(data["uint"])

x = points[:, 0]
vx = velocities[:, 0]
vy = velocities[:, 1]
vz = velocities[:, 2]

v_mag_sq = vx**2 + vy**2 + vz**2
gamma_lorentz = 1.0 / np.sqrt(1.0 - np.clip(v_mag_sq, 0, 0.9999))

N_lab = pmass * (hfact / hpart) ** 3
n_sim = N_lab / gamma_lorentz

P_sim = (gamma_eos - 1.0) * n_sim * uint_data
H_sim = 1.0 + gamma_eos * uint_data
K_sim = gamma_lorentz * H_sim * vy

print(f"Particles: {len(x)}")

# Compute K-invariant and tangent velocity errors
K_mean = np.mean(K_sim)
K_std = np.std(K_sim)
K_err = abs(K_mean - K_L) / K_L

vt_mean = np.mean(vy)
vt_std = np.std(vy)

# Error metrics
err_vz = np.sqrt(np.mean(vz**2))
K_conservation_err = K_std / K_L

print(f"K-invariant: mean={K_mean:.4f} (init={K_L:.4f}), std={K_std:.4f}")
print(f"v_t: mean={vt_mean:.4f} (init={vt_L}), std={vt_std:.4f}")
print(f"v_z RMS: {err_vz:.6e} (should be ~0)")
print(f"K conservation error: {K_conservation_err:.6e}")


# =============================================================================
# Regression Testing
# =============================================================================

error_checks = {
    "K_rel_err": (K_err, 0.1),
    "K_std_rel": (K_conservation_err, 0.15),
    "vz_rms": (err_vz, 1e-3),
}

test_pass = True
err_log = ""

for name, (value, threshold) in error_checks.items():
    if value > threshold:
        err_log += f"error on {name}: expected < {threshold:.1e}, got {value:.6e}\n"
        test_pass = False

if test_pass:
    print("SR-GSPH Tangent Velocity: PASS")
else:
    print("SR-GSPH Tangent Velocity: FAIL")
    print(err_log)
    exit(1)
