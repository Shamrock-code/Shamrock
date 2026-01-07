"""
Exact Special Relativistic Riemann Solver
==========================================

Common module for SR Riemann problem solver used by all SR-GSPH CI tests.
Based on Kitajima, Inutsuka, and Seno (2025) - arXiv:2510.18251v1

Provides:
- Exact solution via iterative root-finding (Brent's method)
- Shock jump conditions via Taub adiabat
- Isentropic rarefaction waves with self-similar sampling
- Solution sampling at arbitrary (x, t)
"""

import numpy as np
from scipy.optimize import brentq


def sr_enthalpy(P, rho, gamma):
    """Specific enthalpy h = 1 + eps + P/rho (c=1)."""
    return 1.0 + (gamma / (gamma - 1.0)) * P / rho


def sr_sound_speed(P, rho, h, gamma):
    """Relativistic sound speed."""
    return np.sqrt(gamma * P / (rho * h))


def sr_characteristic_speed(v, cs, sign):
    """Relativistic characteristic speed (v ± cs)/(1 ± v*cs)."""
    return (v + sign * cs) / (1.0 + sign * v * cs)


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
        v_star, _ = sr_rarefaction_velocity(
            P, P_state, rho_state, v_state, gamma, is_left
        )
    return v_star


def sr_solve_riemann(P_L, rho_L, v_L, P_R, rho_R, v_R, gamma):
    """Solve the SR Riemann problem to find P* and v*."""

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


def sr_sample_rarefaction(xi, P_a, rho_a, v_a, gamma, is_left):
    """Sample state inside rarefaction fan at self-similar coordinate xi = x/t."""
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

    P = brentq(residual, P_a * 1e-6, P_a, xtol=1e-10)

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
    """Sample the exact SR Riemann solution at position x and time t."""
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


def sr_sample_solution_array(x_arr, t, x0, P_L, rho_L, v_L, P_R, rho_R, v_R, gamma):
    """Sample exact solution at array of positions. Returns (rho, v, P) arrays."""
    star_state = sr_solve_riemann(P_L, rho_L, v_L, P_R, rho_R, v_R, gamma)
    rho = np.zeros_like(x_arr)
    v = np.zeros_like(x_arr)
    P = np.zeros_like(x_arr)
    for i, x in enumerate(x_arr):
        rho[i], v[i], P[i] = sr_sample_solution(
            x, t, x0, P_L, rho_L, v_L, P_R, rho_R, v_R, gamma, star_state
        )
    return rho, v, P, star_state


def sr_enthalpy(rho, P, gamma, c=1.0):
    """
    Compute relativistic specific enthalpy H.

    H = 1 + u/c² + P/(ρc²) = 1 + γP/((γ-1)ρc²)

    For ideal gas EOS: u = P/((γ-1)ρ)

    Args:
        rho: Rest-frame density
        P: Pressure
        gamma: Adiabatic index
        c: Speed of light (default 1.0 for natural units)

    Returns:
        H: Relativistic specific enthalpy
    """
    c2 = c * c
    return 1.0 + gamma * P / ((gamma - 1.0) * rho * c2)


def plot_kitajima_style(x_sim, P_sim, n_sim, vx_sim, H_sim, h_sim,
                        x_exact, P_exact, n_exact, vx_exact, H_exact, h_exact,
                        title, filename, xlim=(-0.5, 0.5)):
    """Create Kitajima-style 5-panel plot (P, n, vx, H, h)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time

    # Add timestamp to filename
    timestamp = time.strftime("%H%M%S")
    base, ext = filename.rsplit(".", 1)
    filename = f"{base}_{timestamp}.{ext}"

    # 5 panels with Kitajima-like aspect ratio (wider than tall per panel)
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    fig.subplots_adjust(hspace=0.05, left=0.12, right=0.95, top=0.95, bottom=0.06)

    ax1 = axes[0]
    ax1.plot(x_exact, P_exact, "k-", linewidth=1.5, label="Exact")
    ax1.scatter(x_sim, P_sim, s=2, c="#007F66", marker="v", alpha=0.6)
    ax1.set_ylabel(r"$P$", fontsize=14)
    ax1.tick_params(labelbottom=False)

    ax2 = axes[1]
    ax2.plot(x_exact, n_exact, "k-", linewidth=1.5, label="Exact")
    ax2.scatter(x_sim, n_sim, s=2, c="#CC3311", marker="v", alpha=0.6)
    ax2.set_ylabel(r"$n$", fontsize=14)
    ax2.tick_params(labelbottom=False)

    ax3 = axes[2]
    ax3.plot(x_exact, vx_exact, "k-", linewidth=1.5, label="Exact")
    ax3.scatter(x_sim, vx_sim, s=2, c="#003366", marker="x", alpha=0.6)
    ax3.set_ylabel(r"$v^x$", fontsize=14)
    ax3.tick_params(labelbottom=False)

    ax4 = axes[3]
    ax4.plot(x_exact, H_exact, "k-", linewidth=1.5)
    ax4.scatter(x_sim, H_sim, s=2, c="#EE7733", marker="^", alpha=0.6)
    ax4.set_ylabel(r"$H$", fontsize=14)
    ax4.tick_params(labelbottom=False)

    ax5 = axes[4]
    ax5.plot(x_exact, h_exact, "k-", linewidth=1.5)
    ax5.scatter(x_sim, h_sim, s=2, c="#9933CC", marker="s", alpha=0.6)
    ax5.set_ylabel(r"$h$", fontsize=14)
    ax5.set_xlabel(r"$x$", fontsize=14)

    for ax in axes:
        ax.set_xlim(xlim)
        ax.tick_params(direction="in", which="both")

    fig.suptitle(title, fontsize=12, y=0.98)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved Kitajima-style plot to {filename}")
    plt.close(fig)
