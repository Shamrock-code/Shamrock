"""
Strong Relativistic Blast Wave with SR-GSPH
============================================

CI test for the strong relativistic blast wave problem.
Based on Kitajima, Inutsuka, and Seno (2025) - arXiv:2510.18251v1 Section 3.1.3

Initial conditions (rest-frame, natural units c=1):
    n_L = 1.0,      n_R = 1.0
    P_L = 1000.0,   P_R = 0.01
    v_L = 0,        v_R = 0

This is a strong shock test with pressure ratio of 10^5.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import brentq

import shamrock


def sr_enthalpy(P, rho, gamma):
    """Compute specific enthalpy h = 1 + eps + P/rho (c=1)."""
    return 1.0 + (gamma / (gamma - 1.0)) * P / rho


def sr_sound_speed(P, rho, h, gamma):
    """Compute relativistic sound speed."""
    return np.sqrt(gamma * P / (rho * h))


def sr_shock_velocity(P_star, P_a, rho_a, v_a, gamma, is_left):
    """Solve shock jump conditions (Taub adiabat) following Pons et al. (2000)."""
    h_a = sr_enthalpy(P_a, rho_a, gamma)

    A = (gamma - 1.0) * (P_a - P_star) / (gamma * P_star)
    B = h_a * (P_a - P_star) / rho_a

    qa = 1.0 + A
    qb = -A
    qc = B - h_a**2

    discriminant = qb**2 - 4 * qa * qc
    if discriminant < 0:
        return v_a, rho_a

    h_star = (-qb + np.sqrt(discriminant)) / (2 * qa)
    rho_star = (gamma / (gamma - 1.0)) * P_star / (h_star - 1.0)

    denom_j = h_star / rho_star - h_a / rho_a
    if abs(denom_j) < 1e-15:
        return v_a, rho_star

    j2 = -(P_star - P_a) / denom_j
    if j2 <= 0:
        return v_a, rho_star

    j = np.sqrt(j2)

    gamma_a = 1.0 / np.sqrt(1.0 - v_a**2) if abs(v_a) < 0.9999 else 1e4
    N_a = rho_a * gamma_a

    term = j**2 + N_a**2 * (1.0 - v_a**2)
    sqrt_term = np.sqrt(term)
    denom_Vs = N_a**2 + j**2

    if is_left:
        V_s = (N_a**2 * v_a - j * sqrt_term) / denom_Vs
    else:
        V_s = (N_a**2 * v_a + j * sqrt_term) / denom_Vs

    gamma_s = 1.0 / np.sqrt(max(1.0 - V_s**2, 1e-10))
    j_signed = -j if is_left else j

    if abs(j_signed) < 1e-15:
        return v_a, rho_star

    num = h_a * gamma_a * v_a + gamma_s * (P_star - P_a) / j_signed
    den = h_a * gamma_a + (P_star - P_a) * (gamma_s * v_a / j_signed + 1.0 / N_a)

    if abs(den) < 1e-15:
        return v_a, rho_star

    v_star = num / den
    return v_star, rho_star


def sr_rarefaction_velocity(P_star, P_a, rho_a, v_a, gamma, is_left):
    """Compute velocity across isentropic rarefaction."""
    K_entropy = P_a / (rho_a**gamma)
    rho_star = (P_star / K_entropy) ** (1.0 / gamma)

    u_a = P_a / ((gamma - 1.0) * rho_a)
    h_a = 1.0 + u_a + P_a / rho_a
    cs_a = np.sqrt(gamma * P_a / (rho_a * h_a))

    u_star = P_star / ((gamma - 1.0) * rho_star)
    h_star = 1.0 + u_star + P_star / rho_star
    cs_star = np.sqrt(gamma * P_star / (rho_star * h_star))

    sign = 1.0 if is_left else -1.0
    sqrt_gm1 = np.sqrt(gamma - 1.0)

    term_v = (1.0 + v_a) / (1.0 - v_a)
    term_ca = (sqrt_gm1 + cs_a) / (sqrt_gm1 - cs_a)
    term_cb = (sqrt_gm1 + cs_star) / (sqrt_gm1 - cs_star)

    exponent = sign * 2.0 / sqrt_gm1
    base = term_ca / term_cb

    A_val = term_v * (base**exponent)
    v_star = (A_val - 1.0) / (A_val + 1.0)

    return v_star, rho_star


def sr_wave_curve(P, P_state, rho_state, v_state, gamma, is_left):
    """Compute velocity on wave curve at pressure P."""
    if P > P_state:
        v_star, _ = sr_shock_velocity(P, P_state, rho_state, v_state, gamma, is_left)
    else:
        v_star, _ = sr_rarefaction_velocity(P, P_state, rho_state, v_state, gamma, is_left)
    return v_star


def sr_solve_riemann(P_L, rho_L, v_L, P_R, rho_R, v_R, gamma):
    """Solve the SR Riemann problem to find P* and v*."""
    def residual(P):
        v_L_star = sr_wave_curve(P, P_L, rho_L, v_L, gamma, True)
        v_R_star = sr_wave_curve(P, P_R, rho_R, v_R, gamma, False)
        return v_L_star - v_R_star

    P_min = min(P_L, P_R) * 1e-6
    P_max = max(P_L, P_R) * 1e6

    try:
        P_star = brentq(residual, P_min, P_max, xtol=1e-12)
    except ValueError:
        P_star = 0.5 * (P_L + P_R)

    v_star = sr_wave_curve(P_star, P_L, rho_L, v_L, gamma, True)

    if P_star > P_L:
        _, rho_L_star = sr_shock_velocity(P_star, P_L, rho_L, v_L, gamma, True)
    else:
        _, rho_L_star = sr_rarefaction_velocity(P_star, P_L, rho_L, v_L, gamma, True)

    if P_star > P_R:
        _, rho_R_star = sr_shock_velocity(P_star, P_R, rho_R, v_R, gamma, False)
    else:
        _, rho_R_star = sr_rarefaction_velocity(P_star, P_R, rho_R, v_R, gamma, False)

    return P_star, v_star, rho_L_star, rho_R_star


def sr_characteristic_speed(v, cs, sign):
    """Relativistic characteristic speed."""
    return (v + sign * cs) / (1.0 + sign * v * cs)


def sr_sample_rarefaction(xi, P_a, rho_a, v_a, gamma, is_left):
    """Sample state inside rarefaction fan."""
    K_entropy = P_a / (rho_a**gamma)

    def residual(P):
        rho = (P / K_entropy) ** (1.0 / gamma)
        u = P / ((gamma - 1.0) * rho)
        h = 1.0 + u + P / rho
        cs = np.sqrt(gamma * P / (rho * h))

        sign = 1.0 if is_left else -1.0
        sqrt_gm1 = np.sqrt(gamma - 1.0)

        u_a = P_a / ((gamma - 1.0) * rho_a)
        h_a = 1.0 + u_a + P_a / rho_a
        cs_a = np.sqrt(gamma * P_a / (rho_a * h_a))

        term_v = (1.0 + v_a) / (1.0 - v_a)
        term_ca = (sqrt_gm1 + cs_a) / (sqrt_gm1 - cs_a)
        term_c = (sqrt_gm1 + cs) / (sqrt_gm1 - cs)

        exponent = sign * 2.0 / sqrt_gm1
        A_val = term_v * ((term_ca / term_c) ** exponent)
        v = (A_val - 1.0) / (A_val + 1.0)

        char_sign = -1 if is_left else +1
        char = sr_characteristic_speed(v, cs, char_sign)
        return char - xi

    try:
        P = brentq(residual, P_a * 1e-6, P_a, xtol=1e-10)
    except ValueError:
        return rho_a, v_a, P_a

    rho = (P / K_entropy) ** (1.0 / gamma)
    u = P / ((gamma - 1.0) * rho)
    h = 1.0 + u + P / rho
    cs = np.sqrt(gamma * P / (rho * h))

    sign = 1.0 if is_left else -1.0
    sqrt_gm1 = np.sqrt(gamma - 1.0)

    u_a = P_a / ((gamma - 1.0) * rho_a)
    h_a = 1.0 + u_a + P_a / rho_a
    cs_a = np.sqrt(gamma * P_a / (rho_a * h_a))

    term_v = (1.0 + v_a) / (1.0 - v_a)
    term_ca = (sqrt_gm1 + cs_a) / (sqrt_gm1 - cs_a)
    term_c = (sqrt_gm1 + cs) / (sqrt_gm1 - cs)

    exponent = sign * 2.0 / sqrt_gm1
    A_val = term_v * ((term_ca / term_c) ** exponent)
    v = (A_val - 1.0) / (A_val + 1.0)

    return rho, v, P


def sr_sample_solution(x, t, x0, P_L, rho_L, v_L, P_R, rho_R, v_R, gamma, star_state):
    """Sample the exact SR Riemann solution."""
    P_star, v_star, rho_L_star, rho_R_star = star_state

    if t <= 0:
        return (rho_L, v_L, P_L) if x < x0 else (rho_R, v_R, P_R)

    xi = (x - x0) / t

    h_L = sr_enthalpy(P_L, rho_L, gamma)
    cs_L = sr_sound_speed(P_L, rho_L, h_L, gamma)
    h_R = sr_enthalpy(P_R, rho_R, gamma)
    cs_R = sr_sound_speed(P_R, rho_R, h_R, gamma)

    h_L_star = sr_enthalpy(P_star, rho_L_star, gamma)
    cs_L_star = sr_sound_speed(P_star, rho_L_star, h_L_star, gamma)
    h_R_star = sr_enthalpy(P_star, rho_R_star, gamma)
    cs_R_star = sr_sound_speed(P_star, rho_R_star, h_R_star, gamma)

    if P_star > P_L:
        gamma_L = 1.0 / np.sqrt(1.0 - v_L**2) if abs(v_L) < 0.9999 else 1e4
        N_L = rho_L * gamma_L
        j2 = -(P_star - P_L) / (h_L_star / rho_L_star - h_L / rho_L)
        j = np.sqrt(max(j2, 0))
        term = j**2 + N_L**2 * (1.0 - v_L**2)
        denom = N_L**2 + j**2
        lambda_L_head = (N_L**2 * v_L - j * np.sqrt(term)) / denom
        lambda_L_tail = lambda_L_head
    else:
        lambda_L_head = sr_characteristic_speed(v_L, cs_L, -1)
        lambda_L_tail = sr_characteristic_speed(v_star, cs_L_star, -1)

    if P_star > P_R:
        gamma_R = 1.0 / np.sqrt(1.0 - v_R**2) if abs(v_R) < 0.9999 else 1e4
        N_R = rho_R * gamma_R
        j2 = -(P_star - P_R) / (h_R_star / rho_R_star - h_R / rho_R)
        j = np.sqrt(max(j2, 0))
        term = j**2 + N_R**2 * (1.0 - v_R**2)
        denom = N_R**2 + j**2
        lambda_R_head = (N_R**2 * v_R + j * np.sqrt(term)) / denom
        lambda_R_tail = lambda_R_head
    else:
        lambda_R_head = sr_characteristic_speed(v_star, cs_R_star, +1)
        lambda_R_tail = sr_characteristic_speed(v_R, cs_R, +1)

    if xi < lambda_L_head:
        return rho_L, v_L, P_L
    elif xi < lambda_L_tail:
        return sr_sample_rarefaction(xi, P_L, rho_L, v_L, gamma, True)
    elif xi < v_star:
        return rho_L_star, v_star, P_star
    elif xi < lambda_R_head:
        return rho_R_star, v_star, P_star
    elif xi < lambda_R_tail:
        return sr_sample_rarefaction(xi, P_R, rho_R, v_R, gamma, False)
    else:
        return rho_R, v_R, P_R


# =============================================================================
# Simulation Parameters (Kitajima et al. 2025, Section 3.1.3)
# =============================================================================
gamma = 5.0 / 3.0
n_L, n_R = 1.0, 1.0
P_L, P_R = 1000.0, 0.01
u_L = P_L / ((gamma - 1) * n_L)
u_R = P_R / ((gamma - 1) * n_R)

resol = 100

print("Setting up SR-GSPH Strong Shock Test...")
print(f"  Left state:  P={P_L}, n={n_L}, u={u_L:.4f}")
print(f"  Right state: P={P_R}, n={n_R}, u={u_R:.4f}")
print(f"  Pressure ratio: {P_L/P_R:.0e}")

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
cfg = model.gen_default_config()
cfg.set_riemann_hllc()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.set_sr(c_speed=1.0)

cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Both sides have same density (n_L = n_R = 1.0)
model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

vol_b = xs * ys * zs
totmass = n_L * vol_b * 2
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.2)
model.set_cfl_force(0.2)

# Paper uses t=0.16 for strong shock
t_target = 0.16
print(f"Running SR-GSPH Strong Shock (M4, HLLC, t={t_target})...")
model.evolve_until(t_target)

# Collect results
print("Collecting results...")
data = ctx.collect_data()
points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])
uint_data = np.array(data["uint"])

rho_sim = pmass * (hfact / hpart) ** 3
P_sim = (gamma - 1) * rho_sim * uint_data

x = points[:, 0]
vx = velocities[:, 0]

print("SR-GSPH Strong Shock completed.")
print(f"Particles: {len(x)}")
print(f"x: [{x.min():.4f}, {x.max():.4f}]")
print(f"vx: [{vx.min():.4f}, {vx.max():.4f}]")
print(f"rho: [{rho_sim.min():.4f}, {rho_sim.max():.4f}]")
print(f"P: [{P_sim.min():.4f}, {P_sim.max():.4f}]")

# Compute exact solution
print("\nComputing exact Riemann solution...")
x0 = 0.0

P_star, v_star, rho_L_star, rho_R_star = sr_solve_riemann(
    P_L, n_L, 0.0, P_R, n_R, 0.0, gamma
)
star_state = (P_star, v_star, rho_L_star, rho_R_star)

print(f"  Star state:")
print(f"    P* = {P_star:.6f}")
print(f"    v* = {v_star:.6f}")
print(f"    rho_L* = {rho_L_star:.6f}")
print(f"    rho_R* = {rho_R_star:.6f}")

x_exact = np.linspace(x.min(), x.max(), 500)
rho_exact = np.zeros_like(x_exact)
v_exact = np.zeros_like(x_exact)
p_exact = np.zeros_like(x_exact)

for i, xi in enumerate(x_exact):
    rho_exact[i], v_exact[i], p_exact[i] = sr_sample_solution(
        xi, t_target, x0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state
    )

print("Exact solution computed successfully")

# Create Kitajima-style plots
output_dir = "/Users/guo/Downloads/sph-simulators/simulations_data/sr_strong_shock"
os.makedirs(output_dir, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
fig.subplots_adjust(hspace=0.05)

sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
P_sorted = P_sim[sort_idx]
rho_sorted = rho_sim[sort_idx]
vx_sorted = vx[sort_idx]
h_sorted = hpart[sort_idx]

# Panel 1: Pressure (green) - log scale for strong shock
ax = axes[0]
ax.plot(x_exact, p_exact, 'k-', linewidth=1.2)
ax.plot(x_sorted, P_sorted, 'o', color='#2E8B57', markersize=2, markeredgewidth=0)
ax.set_ylabel(r'$P$')
ax.set_yscale('log')
ax.set_xlim(-0.5, 0.5)
ax.tick_params(labelbottom=False)

# Panel 2: Density (red)
ax = axes[1]
ax.plot(x_exact, rho_exact, 'k-', linewidth=1.2)
ax.plot(x_sorted, rho_sorted, 'o', color='#CD5C5C', markersize=2, markeredgewidth=0)
ax.set_ylabel(r'$n$')
ax.tick_params(labelbottom=False)

# Panel 3: Velocity (blue)
ax = axes[2]
ax.plot(x_exact, v_exact, 'k-', linewidth=1.2)
ax.plot(x_sorted, vx_sorted, 'x', color='#4169E1', markersize=2, markeredgewidth=0.5)
ax.set_ylabel(r'$v^x$')
ax.tick_params(labelbottom=False)

# Panel 4: Smoothing length (orange)
ax = axes[3]
ax.plot(x_sorted, h_sorted, '^', color='#DAA520', markersize=2, markeredgewidth=0)
ax.set_ylabel(r'$h$')
ax.set_xlabel(r'$x$')

plt.tight_layout()
plot_path = os.path.join(output_dir, "sr_strong_shock.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")

print("\n" + "=" * 50)
print("SR-GSPH Strong Shock Test Complete")
print("=" * 50)
