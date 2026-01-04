"""
SR-GSPH All Tests with Plots
============================

Runs all 3 SR-GSPH tests (Sod, Strong Shock, Tangent Velocity) to t=0.4
and generates plots comparing numerical and exact solutions.
Based on Kitajima et al. (2025) arXiv:2510.18251v1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os

import shamrock


# =============================================================================
# SR Riemann Solver Helper Functions
# =============================================================================


def sr_enthalpy(P, rho, gamma):
    return 1.0 + (gamma / (gamma - 1.0)) * P / rho


def sr_sound_speed(P, rho, h, gamma):
    return np.sqrt(gamma * P / (rho * h))


def sr_shock_velocity(P_star, P_a, rho_a, v_a, gamma, is_left):
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
    if P > P_state:
        v_star, _ = sr_shock_velocity(P, P_state, rho_state, v_state, gamma, is_left)
    else:
        v_star, _ = sr_rarefaction_velocity(
            P, P_state, rho_state, v_state, gamma, is_left
        )
    return v_star


def sr_solve_riemann(P_L, rho_L, v_L, P_R, rho_R, v_R, gamma):
    def residual(P):
        v_L_star = sr_wave_curve(P, P_L, rho_L, v_L, gamma, True)
        v_R_star = sr_wave_curve(P, P_R, rho_R, v_R, gamma, False)
        return v_L_star - v_R_star

    P_min = min(P_L, P_R) * 1e-6
    P_max = max(P_L, P_R) * 1e6
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
    return (v + sign * cs) / (1.0 + sign * v * cs)


def sr_sample_rarefaction(xi, P_a, rho_a, v_a, gamma, is_left):
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
    except:
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
# Common parameters
# =============================================================================
gamma = 5.0 / 3.0
t_target = 0.4
output_dir = "/Users/guo/Downloads/sph-simulators/simulations_data/sr_tests"
os.makedirs(output_dir, exist_ok=True)


def run_sr_test(name, P_L, n_L, P_R, n_R, resol=100):
    """Run SR shock tube test and return results."""
    print(f"\n{'='*60}")
    print(f"Running {name} test (t={t_target})")
    print(f"{'='*60}")

    u_L = P_L / ((gamma - 1) * n_L)
    u_R = P_R / ((gamma - 1) * n_R)

    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_GSPH(
        context=ctx, vector_type="f64_3", sph_kernel="TGauss3"
    )
    cfg = model.gen_default_config()
    cfg.set_riemann_hll()
    cfg.set_reconstruct_piecewise_constant()
    cfg.set_boundary_periodic()
    cfg.set_eos_adiabatic(gamma)
    cfg.set_sr(c_speed=1.0)

    model.set_solver_config(cfg)
    model.init_scheduler(int(1e8), 1)

    (xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 8, 8)
    dr = 1 / xs
    (xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 8, 8)
    model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

    if n_L != n_R:
        dr_L = dr * (n_R / n_L) ** (1 / 3)
        dr_R = dr
        model.add_cube_hcp_3d(dr_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
        model.add_cube_hcp_3d(dr_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
    else:
        model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
        model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

    model.set_field_in_box(
        "uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2)
    )
    model.set_field_in_box(
        "uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2)
    )

    vol_b = xs * ys * zs
    totmass = n_L * vol_b + n_R * vol_b
    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)
    hfact = model.get_hfact()

    model.set_cfl_cour(0.2)
    model.set_cfl_force(0.2)

    print(f"Evolving to t={t_target}...")
    model.evolve_until(t_target)

    data = ctx.collect_data()
    points = np.array(data["xyz"])
    velocities = np.array(data["vxyz"])
    hpart = np.array(data["hpart"])
    uint_data = np.array(data["uint"])

    rho_sim = pmass * (hfact / hpart) ** 3
    P_sim = (gamma - 1) * rho_sim * uint_data
    x = points[:, 0]
    vx = velocities[:, 0]

    # Compute exact solution
    star_state = sr_solve_riemann(P_L, n_L, 0.0, P_R, n_R, 0.0, gamma)
    x_exact = np.linspace(x.min(), x.max(), 500)
    rho_exact, v_exact, P_exact = [], [], []
    for xi in x_exact:
        r, v, p = sr_sample_solution(
            xi, t_target, 0.0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state
        )
        rho_exact.append(r)
        v_exact.append(v)
        P_exact.append(p)

    return {
        "x": x,
        "vx": vx,
        "rho": rho_sim,
        "P": P_sim,
        "x_exact": x_exact,
        "rho_exact": np.array(rho_exact),
        "v_exact": np.array(v_exact),
        "P_exact": np.array(P_exact),
        "star_state": star_state,
    }


# =============================================================================
# Run all 3 tests
# =============================================================================

# Test 1: Sod tube (Section 3.1.1)
print("\n" + "=" * 70)
print("SR-GSPH All Tests at t=0.4 (Kitajima et al. 2025)")
print("=" * 70)

sod = run_sr_test("Sod Tube", P_L=1.0, n_L=1.0, P_R=0.1, n_R=0.125, resol=100)
strong = run_sr_test("Strong Shock", P_L=1000.0, n_L=1.0, P_R=0.01, n_R=1.0, resol=100)

# Test 3: Tangent velocity (simplified - just run standard shock with v_t)
print("\n" + "=" * 60)
print("Running Tangent Velocity test (t=0.4)")
print("=" * 60)

gamma_eos = gamma
P_L_t, P_R_t = 1000.0, 0.01
n_L_t, n_R_t = 1.0, 1.0
vt = 0.9
gamma_L = 1.0 / np.sqrt(1.0 - vt**2)

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="TGauss3")
cfg = model.gen_default_config()
cfg.set_riemann_hll()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma_eos)
cfg.set_sr(c_speed=1.0)

model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)

N_per_side = 100
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, N_per_side, 8, 8)
dr = 1.0 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, N_per_side, 8, 8)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

u_L_t = P_L_t / ((gamma_eos - 1.0) * n_L_t)
u_R_t = P_R_t / ((gamma_eos - 1.0) * n_R_t)

model.set_field_in_box(
    "uint", "f64", u_L_t, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2)
)
model.set_field_in_box(
    "uint", "f64", u_R_t, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2)
)
model.set_field_in_box(
    "vxyz", "f64_3", (0.0, vt, 0.0), (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2)
)

vol_total = 2 * xs * ys * zs
totmass = gamma_L * n_L_t * vol_total
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.15)
model.set_cfl_force(1e10)

print(f"Evolving to t={t_target}...")
model.evolve_until(t_target)

data = ctx.collect_data()
points_t = np.array(data["xyz"])
velocities_t = np.array(data["vxyz"])
hpart_t = np.array(data["hpart"])
uint_t = np.array(data["uint"])

x_t = points_t[:, 0]
vx_t = velocities_t[:, 0]
vy_t = velocities_t[:, 1]

v_mag_sq = vx_t**2 + vy_t**2
gamma_lorentz = 1.0 / np.sqrt(1.0 - np.clip(v_mag_sq, 0, 0.9999))

N_lab = pmass * (hfact / hpart_t) ** 3
n_sim_t = N_lab / gamma_lorentz
P_sim_t = (gamma_eos - 1.0) * n_sim_t * uint_t


# =============================================================================
# Generate plots
# =============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle(
    f"SR-GSPH Tests at t={t_target} (Kitajima et al. 2025)\nTGauss3 kernel, HLL Riemann solver",
    fontsize=14,
)

# Row 1: Sod tube
sort_idx = np.argsort(sod["x"])
ax = axes[0, 0]
ax.plot(sod["x_exact"], sod["rho_exact"], "k-", lw=1.5, label="Exact")
ax.plot(sod["x"][sort_idx], sod["rho"][sort_idx], "b.", ms=1, alpha=0.3)
ax.set_ylabel("Density n")
ax.set_title("Sod Tube")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(sod["x_exact"], sod["v_exact"], "k-", lw=1.5)
ax.plot(sod["x"][sort_idx], sod["vx"][sort_idx], "b.", ms=1, alpha=0.3)
ax.set_ylabel("Velocity vx")

ax = axes[0, 2]
ax.plot(sod["x_exact"], sod["P_exact"], "k-", lw=1.5)
ax.plot(sod["x"][sort_idx], sod["P"][sort_idx], "b.", ms=1, alpha=0.3)
ax.set_ylabel("Pressure P")

ax = axes[0, 3]
ax.text(
    0.5,
    0.5,
    f"P* = {sod['star_state'][0]:.4f}\nv* = {sod['star_state'][1]:.4f}",
    ha="center",
    va="center",
    fontsize=12,
    transform=ax.transAxes,
)
ax.set_title("Star State")
ax.axis("off")

# Row 2: Strong shock
sort_idx = np.argsort(strong["x"])
ax = axes[1, 0]
ax.plot(strong["x_exact"], strong["rho_exact"], "k-", lw=1.5, label="Exact")
ax.plot(strong["x"][sort_idx], strong["rho"][sort_idx], "r.", ms=1, alpha=0.3)
ax.set_ylabel("Density n")
ax.set_title("Strong Shock")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(strong["x_exact"], strong["v_exact"], "k-", lw=1.5)
ax.plot(strong["x"][sort_idx], strong["vx"][sort_idx], "r.", ms=1, alpha=0.3)
ax.set_ylabel("Velocity vx")

ax = axes[1, 2]
ax.semilogy(strong["x_exact"], strong["P_exact"], "k-", lw=1.5)
ax.semilogy(strong["x"][sort_idx], strong["P"][sort_idx], "r.", ms=1, alpha=0.3)
ax.set_ylabel("Pressure P (log)")

ax = axes[1, 3]
ax.text(
    0.5,
    0.5,
    f"P* = {strong['star_state'][0]:.4f}\nv* = {strong['star_state'][1]:.4f}",
    ha="center",
    va="center",
    fontsize=12,
    transform=ax.transAxes,
)
ax.set_title("Star State")
ax.axis("off")

# Row 3: Tangent velocity
sort_idx = np.argsort(x_t)
ax = axes[2, 0]
ax.plot(x_t[sort_idx], n_sim_t[sort_idx], "g.", ms=1, alpha=0.3)
ax.set_ylabel("Density n")
ax.set_xlabel("x")
ax.set_title(f"Tangent Velocity (vt={vt})")

ax = axes[2, 1]
ax.plot(x_t[sort_idx], vx_t[sort_idx], "g.", ms=1, alpha=0.3)
ax.set_ylabel("Normal Velocity vx")
ax.set_xlabel("x")

ax = axes[2, 2]
ax.plot(x_t[sort_idx], vy_t[sort_idx], "g.", ms=1, alpha=0.3)
ax.axhline(y=vt, color="r", ls="--", lw=1, label=f"Initial vt={vt}")
ax.set_ylabel("Tangent Velocity vy")
ax.set_xlabel("x")
ax.legend(fontsize=8)

ax = axes[2, 3]
H_sim = 1.0 + gamma_eos * uint_t
K_sim = gamma_lorentz * H_sim * vy_t
H_L = 1.0 + (gamma_eos / (gamma_eos - 1.0)) * P_L_t / n_L_t
K_init = gamma_L * H_L * vt
ax.plot(x_t[sort_idx], K_sim[sort_idx], "g.", ms=1, alpha=0.3)
ax.axhline(y=K_init, color="r", ls="--", lw=1, label=f"Initial K={K_init:.2f}")
ax.set_ylabel("K-invariant")
ax.set_xlabel("x")
ax.set_title("K = gamma*H*vt")
ax.legend(fontsize=8)

plt.tight_layout()
plot_path = os.path.join(output_dir, "sr_all_tests_t04.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {plot_path}")

print("\n" + "=" * 70)
print("All SR-GSPH tests completed successfully!")
print("=" * 70)
