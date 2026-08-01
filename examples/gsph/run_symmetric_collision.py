"""
Symmetric gas collision with GSPH (exact Riemann solver + Inutsuka V2)
========================================================================

Two identical bodies of gas move toward each other at equal and opposite
speed (:math:`\\rho_L=\\rho_R=1`, :math:`u_L=+1`, :math:`u_R=-1`,
:math:`p_L=p_R=1`). By symmetry the contact stays fixed at :math:`u^*=0`
for all time, which makes this equivalent to each half of the gas hitting
a rigid, reflecting wall at x=0. This is the standard "wall collision"
test used to check that a scheme does not produce spurious post-shock
heating at a symmetry plane -- a known pathology of artificial-viscosity
SPH that Riemann-solver formulations such as GSPH are designed to avoid.
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Analytic exact Riemann solver (Toro 2009), reimplemented here in pure
# Python since only the Sod-tube (zero-velocity) case has a bound analytic
# helper in shamrock.phys.


def _phi_shock(p, rho_K, p_K, gamma):
    A_K = 2.0 / ((gamma + 1.0) * rho_K)
    B_K = (gamma - 1.0) / (gamma + 1.0) * p_K
    return (p - p_K) * np.sqrt(A_K / (p + B_K))


def _phi_raref(p, rho_K, p_K, gamma):
    c_K = np.sqrt(gamma * p_K / rho_K)
    return (2.0 * c_K / (gamma - 1.0)) * ((p / p_K) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)


def _phi(p, wave_type, rho_K, p_K, gamma):
    return (
        _phi_shock(p, rho_K, p_K, gamma)
        if wave_type == "shock"
        else _phi_raref(p, rho_K, p_K, gamma)
    )


def exact_riemann_profile(x, t, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma):
    """Sample the exact 1D Riemann solution (rho, u, p) at positions x, time t."""
    v_lr_0 = u_L - u_R

    def f_ss(p):
        return _phi_shock(p, rho_L, p_L, gamma) + _phi_shock(p, rho_R, p_R, gamma)

    def f_rs(p):
        return _phi_raref(p, rho_L, p_L, gamma) + _phi_shock(p, rho_R, p_R, gamma)

    def f_sr(p):
        return _phi_shock(p, rho_L, p_L, gamma) + _phi_raref(p, rho_R, p_R, gamma)

    if p_L > p_R:
        if v_lr_0 > f_ss(p_L):
            pattern = "ss"
        elif v_lr_0 > f_rs(p_R):
            pattern = "rs"
        else:
            pattern = "rr"
    else:
        if v_lr_0 > f_ss(p_R):
            pattern = "ss"
        elif v_lr_0 > f_sr(p_L):
            pattern = "sr"
        else:
            pattern = "rr"

    left_type = "shock" if pattern in ("ss", "sr") else "raref"
    right_type = "shock" if pattern in ("ss", "rs") else "raref"

    def f_total(p):
        return _phi(p, left_type, rho_L, p_L, gamma) + _phi(p, right_type, rho_R, p_R, gamma)

    if pattern == "ss":
        p_plus, p_minus = p_L + p_R, min(p_L, p_R)
        for _ in range(300):
            if f_total(p_plus) - v_lr_0 > 0:
                break
            p_plus *= 10.0
    elif pattern == "rs":
        p_plus, p_minus = p_L, p_R
    elif pattern == "sr":
        p_plus, p_minus = p_R, p_L
    else:
        p_plus, p_minus = min(p_L, p_R), 0.0

    p_star = 0.5 * (p_plus + p_minus)
    for _ in range(200):
        p_star = 0.5 * (p_plus + p_minus)
        r = f_total(p_star) - v_lr_0
        if abs(r) < 1e-12:
            break
        if r > 0:
            p_plus = p_star
        else:
            p_minus = p_star

    phi_L = _phi(p_star, left_type, rho_L, p_L, gamma)
    phi_R = _phi(p_star, right_type, rho_R, p_R, gamma)
    u_star = 0.5 * (u_L + u_R) - 0.5 * (phi_L - phi_R)

    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)

    if left_type == "shock":
        rho_star_L = rho_L * (
            (p_star / p_L + (gamma - 1) / (gamma + 1))
            / ((gamma - 1) / (gamma + 1) * (p_star / p_L) + 1)
        )
        left = {
            "type": "shock",
            "S": u_L
            - c_L * np.sqrt((gamma + 1) / (2 * gamma) * (p_star / p_L) + (gamma - 1) / (2 * gamma)),
        }
    else:
        rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)
        left = {
            "type": "raref",
            "S_H": u_L - c_L,
            "S_T": u_star - c_L * (p_star / p_L) ** ((gamma - 1) / (2 * gamma)),
        }

    if right_type == "shock":
        rho_star_R = rho_R * (
            (p_star / p_R + (gamma - 1) / (gamma + 1))
            / ((gamma - 1) / (gamma + 1) * (p_star / p_R) + 1)
        )
        right = {
            "type": "shock",
            "S": u_R
            + c_R * np.sqrt((gamma + 1) / (2 * gamma) * (p_star / p_R) + (gamma - 1) / (2 * gamma)),
        }
    else:
        rho_star_R = rho_R * (p_star / p_R) ** (1.0 / gamma)
        right = {
            "type": "raref",
            "S_H": u_R + c_R,
            "S_T": u_star + c_R * (p_star / p_R) ** ((gamma - 1) / (2 * gamma)),
        }

    rho_out, u_out, p_out = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for i, xi_pos in enumerate(x):
        xi = xi_pos / t
        if xi <= u_star:
            if left["type"] == "shock":
                rho, u, p = (rho_L, u_L, p_L) if xi < left["S"] else (rho_star_L, u_star, p_star)
            elif xi < left["S_H"]:
                rho, u, p = rho_L, u_L, p_L
            elif xi > left["S_T"]:
                rho, u, p = rho_star_L, u_star, p_star
            else:
                b = 2 / (gamma + 1) + (gamma - 1) / ((gamma + 1) * c_L) * (u_L - xi)
                rho, u, p = (
                    rho_L * b ** (2 / (gamma - 1)),
                    2 / (gamma + 1) * (c_L + (gamma - 1) / 2 * u_L + xi),
                    p_L * b ** (2 * gamma / (gamma - 1)),
                )
        else:
            if right["type"] == "shock":
                rho, u, p = (rho_R, u_R, p_R) if xi > right["S"] else (rho_star_R, u_star, p_star)
            elif xi > right["S_H"]:
                rho, u, p = rho_R, u_R, p_R
            elif xi < right["S_T"]:
                rho, u, p = rho_star_R, u_star, p_star
            else:
                b = 2 / (gamma + 1) - (gamma - 1) / ((gamma + 1) * c_R) * (u_R - xi)
                rho, u, p = (
                    rho_R * b ** (2 / (gamma - 1)),
                    2 / (gamma + 1) * (-c_R + (gamma - 1) / 2 * u_R + xi),
                    p_R * b ** (2 * gamma / (gamma - 1)),
                )
        rho_out[i], u_out[i], p_out[i] = rho, u, p

    return rho_out, u_out, p_out, p_star, u_star


# %%
# Setup parameters: symmetric collision (equivalent to a wall-collision test)

gamma = 1.4
rho_L, rho_R = 1.0, 1.0
u_L, u_R = 1.0, -1.0
P_L, P_R = 1.0, 1.0
uint0 = P_L / ((gamma - 1) * rho_L)
resol = 64

# %%
# Setup the solver: GSPH with the exact Riemann solver and the Inutsuka V2
# effective volume/face force formulation.

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
cfg = model.gen_default_config()
cfg.set_riemann_exact()
cfg.set_force_inutsuka_v2()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)

# %%
# Setup the initial conditions: uniform density/pressure, opposite bulk
# velocities on each half (converging flow, no density or pressure jump).

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", uint0, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
model.set_field_in_box(
    "vxyz", "f64_3", (u_L, 0.0, 0.0), (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2)
)
model.set_field_in_box(
    "vxyz", "f64_3", (u_R, 0.0, 0.0), (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2)
)

vol_b = xs * ys * zs
totmass = 2 * rho_L * vol_b
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

# %%
# Run the simulation

t_target = 0.2
model.evolve_until(t_target)

# %%
# Collect the particle data

data = ctx.collect_data()

points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])
uint = np.array(data["uint"])

x = points[:, 0]
vx = velocities[:, 0]
rho = pmass * (hfact / hpart) ** 3
P = (gamma - 1) * rho * uint

# %%
# Analytic exact-Riemann solution for comparison

arr_x = np.linspace(-0.45, 0.45, 1000)
arr_rho, arr_vx, arr_P, p_star, u_star = exact_riemann_profile(
    arr_x, t_target, rho_L, u_L, P_L, rho_R, u_R, P_R, gamma
)
print(f"Symmetric collision: p* = {p_star:.6g} (p*/p_L = {p_star / P_L:.4g}), u* = {u_star:.6g}")

# %%
# Plot the particle data against the analytic solution

fig, ax = plt.subplots(figsize=(9, 6), dpi=125)

ax.scatter(x, rho, rasterized=True, s=6 * np.ones(x.shape), label="density")
ax.scatter(x, vx, rasterized=True, s=6 * np.ones(x.shape), label="velocity")
ax.scatter(x, P, rasterized=True, s=6 * np.ones(x.shape), label="pressure")

ax.plot(arr_x, arr_rho, ls="--", lw=2.0, color="black", label="analytic")
ax.plot(arr_x, arr_vx, ls="--", lw=2.0, color="black")
ax.plot(arr_x, arr_P, ls="--", lw=2.0, color="black")

ax.set_xlim(-0.45, 0.45)
ax.set_xlabel("x")
ax.set_title(f"GSPH symmetric collision / wall test (exact solver + Inutsuka V2), t={t_target}")
ax.legend(loc=0)
ax.grid(alpha=0.3)

if shamrock.sys.world_rank() == 0:
    plt.show()
