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

import shamrock
from sr_riemann_exact import (
    sr_solve_riemann,
    sr_sample_solution,
    plot_kitajima_style,
)

gamma = 5.0 / 3.0
n_L, n_R = 1.0, 1.0
P_L, P_R = 1000.0, 0.01
u_L = P_L / ((gamma - 1) * n_L)
u_R = P_R / ((gamma - 1) * n_R)

resol = 100

print("Setting up SR-GSPH Strong Shock Test...")

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="TGauss3")
cfg = model.gen_default_config()
cfg.set_riemann_hll()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.set_sr(c_speed=1.0)
cfg.set_use_grad_h(True)

model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

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

t_target = 0.16
print(f"Running SR-GSPH Strong Shock (TGauss3, HLL, t={t_target})...")
model.evolve_until(t_target)

data = ctx.collect_data()
points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])
uint_data = np.array(data["uint"])

N_sim = pmass * (hfact / hpart) ** 3
x = points[:, 0]
vx = velocities[:, 0]
vy = velocities[:, 1]
vz = velocities[:, 2]

v2_sim = vx**2 + vy**2 + vz**2
gamma_sim = 1.0 / np.sqrt(np.maximum(1.0 - v2_sim, 1e-10))
n_sim = N_sim / gamma_sim

P_sim = (gamma - 1) * n_sim * uint_data

print(f"Particles: {len(x)}")
print(f"Max simulation Lorentz factor: {np.max(gamma_sim):.3f}")

x0 = 0.0
P_star, v_star, rho_L_star, rho_R_star = sr_solve_riemann(
    P_L, n_L, 0.0, P_R, n_R, 0.0, gamma
)
star_state = (P_star, v_star, rho_L_star, rho_R_star)

print(f"Star state: P*={P_star:.6f}, v*={v_star:.6f}")

n_ana = np.zeros_like(x)
vx_ana = np.zeros_like(x)
P_ana = np.zeros_like(x)

for i in range(len(x)):
    n_ana[i], vx_ana[i], P_ana[i] = sr_sample_solution(
        x[i], t_target, x0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state
    )

x_fine = np.linspace(-1.0, 1.0, 2000)
n_ana_fine = np.zeros_like(x_fine)
vx_ana_fine = np.zeros_like(x_fine)
P_ana_fine = np.zeros_like(x_fine)

for i in range(len(x_fine)):
    n_ana_fine[i], vx_ana_fine[i], P_ana_fine[i] = sr_sample_solution(
        x_fine[i], t_target, x0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state
    )

h_ana_fine = (pmass / n_ana_fine) ** (1.0 / 3.0) * hfact

err_rho = np.sqrt(np.mean((n_sim - n_ana) ** 2)) / np.mean(n_ana)
err_vx = np.sqrt(np.mean((vx - vx_ana) ** 2)) / max(np.mean(np.abs(vx_ana)), 1e-10)
err_vy = np.sqrt(np.mean(vy**2))
err_vz = np.sqrt(np.mean(vz**2))
err_P = np.sqrt(np.mean((P_sim - P_ana) ** 2)) / np.mean(P_ana)

print(f"L2 errors: rho={err_rho:.6e}, vx={err_vx:.6e}, vy={err_vy:.6e}, P={err_P:.6e}")

plot_kitajima_style(
    x, P_sim, n_sim, vx, hpart,
    x_fine, P_ana_fine, n_ana_fine, vx_ana_fine, h_ana_fine,
    f"SR Strong Shock t={t_target}", "sr_strong_shock_kitajima.png",
    xlim=(-1.0, 1.0)
)

expect_rho = 0.08
expect_vx = 0.12
expect_vy = 1e-10
expect_vz = 1e-10
expect_P = 0.20

error_checks = {
    "err_rho": (err_rho, expect_rho, 1e-1),
    "err_vx": (err_vx, expect_vx, 1e-1),
    "err_vy": (err_vy, expect_vy, 1e-4),
    "err_vz": (err_vz, expect_vz, 1e-4),
    "err_P": (err_P, expect_P, 1e-1),
}

test_pass = True
err_log = ""

for name, (value, expected, loose_tol) in error_checks.items():
    if value > loose_tol:
        err_log += f"error on {name}: expected < {loose_tol:.1e}, got {value:.6e}\n"
        test_pass = False

if test_pass:
    print("SR-GSPH Strong Shock: PASS")
else:
    print("SR-GSPH Strong Shock: FAIL")
    print(err_log)
