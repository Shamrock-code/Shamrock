"""
Special Relativistic Sod Shock Tube with SR-GSPH
=================================================

CI test for the relativistic Sod shock tube problem.
Based on Kitajima, Inutsuka, and Seno (2025) - arXiv:2510.18251v1

Initial conditions (rest-frame, natural units c=1):
    n_L = 1.0,      n_R = 0.125
    P_L = 1.0,      P_R = 0.1
    v_L = 0,        v_R = 0

Computes L2 error against exact SR Riemann solution.
"""

import numpy as np
from scipy.optimize import brentq

import shamrock


def sr_enthalpy(P, rho, gamma):
    """Compute specific enthalpy h = 1 + eps + P/rho (c=1)."""
    return 1.0 + (gamma / (gamma - 1.0)) * P / rho


def sr_sound_speed(P, rho, h, gamma):
    """Compute relativistic sound speed."""
    return np.sqrt(gamma * P / (rho * h))


def sr_shock_velocity(P_star, P_a, rho_a, v_a, gamma, is_left):
    """Solve shock jump conditions (Taub adiabat)."""
    h_a = sr_enthalpy(P_a, rho_a, gamma)
    A = (gamma - 1.0) * (P_a - P_star) / (gamma * P_star)
    B = h_a * (P_a - P_star) / rho_a

    qa, qb, qc = 1.0 + A, -A, B - h_a**2
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
    denom_Vs = N_a**2 + j**2
    V_s = (
        (N_a**2 * v_a - j * np.sqrt(term)) / denom_Vs
        if is_left
        else (N_a**2 * v_a + j * np.sqrt(term)) / denom_Vs
    )

    gamma_s = 1.0 / np.sqrt(max(1.0 - V_s**2, 1e-10))
    j_signed = -j if is_left else j

    if abs(j_signed) < 1e-15:
        return v_a, rho_star

    num = h_a * gamma_a * v_a + gamma_s * (P_star - P_a) / j_signed
    den = h_a * gamma_a + (P_star - P_a) * (gamma_s * v_a / j_signed + 1.0 / N_a)

    if abs(den) < 1e-15:
        return v_a, rho_star
    return num / den, rho_star


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

    A_val = term_v * ((term_ca / term_cb) ** (sign * 2.0 / sqrt_gm1))
    return (A_val - 1.0) / (A_val + 1.0), rho_star


def sr_wave_curve(P, P_state, rho_state, v_state, gamma, is_left):
    """Compute velocity on wave curve at pressure P."""
    if P > P_state:
        v_star, _ = sr_shock_velocity(P, P_state, rho_state, v_state, gamma, is_left)
    else:
        v_star, _ = sr_rarefaction_velocity(
            P, P_state, rho_state, v_state, gamma, is_left
        )
    return v_star


def sr_solve_riemann(P_L, rho_L, v_L, P_R, rho_R, v_R, gamma):
    """Solve the SR Riemann problem using Brent's method."""

    def residual(P):
        v_L_star = sr_wave_curve(P, P_L, rho_L, v_L, gamma, True)
        v_R_star = sr_wave_curve(P, P_R, rho_R, v_R, gamma, False)
        return v_L_star - v_R_star

    P_min, P_max = min(P_L, P_R) * 1e-6, max(P_L, P_R) * 1e6
    P_star = brentq(residual, P_min, P_max, xtol=1e-12)
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


def sr_sample_solution(x, t, x0, P_L, rho_L, v_L, P_R, rho_R, v_R, gamma, star_state):
    """Sample the exact SR Riemann solution at position x and time t."""
    P_star, v_star, rho_L_star, rho_R_star = star_state
    if t <= 0:
        return (rho_L, v_L, P_L) if x < x0 else (rho_R, v_R, P_R)

    xi = (x - x0) / t
    h_L, h_R = sr_enthalpy(P_L, rho_L, gamma), sr_enthalpy(P_R, rho_R, gamma)
    cs_L, cs_R = sr_sound_speed(P_L, rho_L, h_L, gamma), sr_sound_speed(
        P_R, rho_R, h_R, gamma
    )
    h_L_star = sr_enthalpy(P_star, rho_L_star, gamma)
    cs_L_star = sr_sound_speed(P_star, rho_L_star, h_L_star, gamma)

    # Left wave
    if P_star > P_L:
        gamma_L = 1.0 / np.sqrt(1.0 - v_L**2) if abs(v_L) < 0.9999 else 1e4
        N_L = rho_L * gamma_L
        j2 = -(P_star - P_L) / (h_L_star / rho_L_star - h_L / rho_L)
        j = np.sqrt(max(j2, 0))
        term = j**2 + N_L**2 * (1.0 - v_L**2)
        lambda_L_head = lambda_L_tail = (N_L**2 * v_L - j * np.sqrt(term)) / (
            N_L**2 + j**2
        )
    else:
        lambda_L_head = sr_characteristic_speed(v_L, cs_L, -1)
        lambda_L_tail = sr_characteristic_speed(v_star, cs_L_star, -1)

    # Right wave
    h_R_star = sr_enthalpy(P_star, rho_R_star, gamma)
    cs_R_star = sr_sound_speed(P_star, rho_R_star, h_R_star, gamma)
    if P_star > P_R:
        gamma_R = 1.0 / np.sqrt(1.0 - v_R**2) if abs(v_R) < 0.9999 else 1e4
        N_R = rho_R * gamma_R
        j2 = -(P_star - P_R) / (h_R_star / rho_R_star - h_R / rho_R)
        j = np.sqrt(max(j2, 0))
        term = j**2 + N_R**2 * (1.0 - v_R**2)
        lambda_R_head = lambda_R_tail = (N_R**2 * v_R + j * np.sqrt(term)) / (
            N_R**2 + j**2
        )
    else:
        lambda_R_head = sr_characteristic_speed(v_star, cs_R_star, +1)
        lambda_R_tail = sr_characteristic_speed(v_R, cs_R, +1)

    if xi < lambda_L_head:
        return rho_L, v_L, P_L
    elif xi < lambda_L_tail:
        return rho_L_star, v_star, P_star  # Inside rarefaction approx
    elif xi < v_star:
        return rho_L_star, v_star, P_star
    elif xi < lambda_R_head:
        return rho_R_star, v_star, P_star
    elif xi < lambda_R_tail:
        return rho_R_star, v_star, P_star
    else:
        return rho_R, v_R, P_R


# =============================================================================
# Simulation setup
# =============================================================================
gamma = 5.0 / 3.0
n_L, n_R = 1.0, 0.125
P_L, P_R = 1.0, 0.1
u_L = P_L / ((gamma - 1) * n_L)
u_R = P_R / ((gamma - 1) * n_R)
resol = 100
fact = (n_L / n_R) ** (1.0 / 3.0)

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="TGauss3")
cfg = model.gen_default_config()
cfg.set_riemann_hll()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.set_sr(c_speed=1.0)
model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr * fact, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

vol_b = xs * ys * zs
totmass = (n_R * vol_b) + (n_L * vol_b)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.3)
model.set_cfl_force(0.3)

t_target = 0.4
print(f"SR-GSPH Sod Shock Tube Test (TGauss3, t={t_target})")
model.evolve_until(t_target)

# =============================================================================
# Collect results and compute L2 errors
# =============================================================================
data = ctx.collect_data()
points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])
uint_data = np.array(data["uint"])

rho_sim = pmass * (hfact / hpart) ** 3
P_sim = (gamma - 1) * rho_sim * uint_data
x, vx, vy, vz = points[:, 0], velocities[:, 0], velocities[:, 1], velocities[:, 2]

# Compute exact solution
P_star, v_star, rho_L_star, rho_R_star = sr_solve_riemann(
    P_L, n_L, 0.0, P_R, n_R, 0.0, gamma
)
star_state = (P_star, v_star, rho_L_star, rho_R_star)

# Sample exact solution at particle positions
x_min, x_max = -0.5, 0.5
mask = (x >= x_min) & (x <= x_max)
x_f, rho_f, vx_f, P_f = x[mask], rho_sim[mask], vx[mask], P_sim[mask]
vy_f, vz_f = vy[mask], vz[mask]

rho_ana = np.array(
    [
        sr_sample_solution(
            xi, t_target, 0.0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state
        )[0]
        for xi in x_f
    ]
)
vx_ana = np.array(
    [
        sr_sample_solution(
            xi, t_target, 0.0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state
        )[1]
        for xi in x_f
    ]
)
P_ana = np.array(
    [
        sr_sample_solution(
            xi, t_target, 0.0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state
        )[2]
        for xi in x_f
    ]
)

# Compute L2 errors
err_rho = np.sqrt(np.mean((rho_f - rho_ana) ** 2)) / np.mean(rho_ana)
err_vx = np.sqrt(np.mean((vx_f - vx_ana) ** 2)) / (np.mean(np.abs(vx_ana)) + 0.1)
err_vy = np.sqrt(np.mean(vy_f**2))
err_vz = np.sqrt(np.mean(vz_f**2))
err_P = np.sqrt(np.mean((P_f - P_ana) ** 2)) / np.mean(P_ana)

print(
    f"L2 errors: rho={err_rho:.6e}, vx={err_vx:.6e}, vy={err_vy:.6e}, vz={err_vz:.6e}, P={err_P:.6e}"
)

# =============================================================================
# Regression test with strict tolerances
# =============================================================================
expect_rho = 0.2331247
expect_vx = 0.4112091
expect_vy = 0.001167533
expect_vz = 8.164793e-05
expect_P = 0.3264808
tol = 1e-8

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
    if abs(value - expected) > tol * expected + 1e-10:
        err_log += f"error on {name}: expected {expected:.6e}, got {value:.6e}\n"
        test_pass = False

if test_pass:
    print("\n" + "=" * 50)
    print("SR-GSPH Sod Shock Tube Test: PASSED")
    print("=" * 50)
else:
    exit("Test did not pass L2 margins:\n" + err_log)
