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

import shamrock
from sr_riemann_exact import (
    sr_solve_riemann,
    sr_sample_solution,
    plot_kitajima_style,
)

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

data = ctx.collect_data()
points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])
uint_data = np.array(data["uint"])

rho_sim = pmass * (hfact / hpart) ** 3
P_sim = (gamma - 1) * rho_sim * uint_data
x, vx, vy, vz = points[:, 0], velocities[:, 0], velocities[:, 1], velocities[:, 2]

P_star, v_star, rho_L_star, rho_R_star = sr_solve_riemann(
    P_L, n_L, 0.0, P_R, n_R, 0.0, gamma
)
star_state = (P_star, v_star, rho_L_star, rho_R_star)

x_min, x_max = -0.5, 0.5
mask = (x >= x_min) & (x <= x_max)
x_f, rho_f, vx_f, P_f = x[mask], rho_sim[mask], vx[mask], P_sim[mask]
h_f = hpart[mask]

rho_ana = np.array([
    sr_sample_solution(xi, t_target, 0.0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state)[0]
    for xi in x_f
])
vx_ana = np.array([
    sr_sample_solution(xi, t_target, 0.0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state)[1]
    for xi in x_f
])
P_ana = np.array([
    sr_sample_solution(xi, t_target, 0.0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state)[2]
    for xi in x_f
])

err_rho = np.sqrt(np.mean((rho_f - rho_ana) ** 2)) / np.mean(rho_ana)
err_vx = np.sqrt(np.mean((vx_f - vx_ana) ** 2)) / (np.mean(np.abs(vx_ana)) + 0.1)
err_vy = np.sqrt(np.mean(vy[mask] ** 2))
err_vz = np.sqrt(np.mean(vz[mask] ** 2))
err_P = np.sqrt(np.mean((P_f - P_ana) ** 2)) / np.mean(P_ana)

print(f"L2 errors: rho={err_rho:.6e}, vx={err_vx:.6e}, vy={err_vy:.6e}, vz={err_vz:.6e}, P={err_P:.6e}")

x_exact = np.linspace(-0.5, 0.5, 500)
rho_exact = np.array([
    sr_sample_solution(xi, t_target, 0.0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state)[0]
    for xi in x_exact
])
vx_exact = np.array([
    sr_sample_solution(xi, t_target, 0.0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state)[1]
    for xi in x_exact
])
P_exact = np.array([
    sr_sample_solution(xi, t_target, 0.0, P_L, n_L, 0.0, P_R, n_R, 0.0, gamma, star_state)[2]
    for xi in x_exact
])
h_exact = (pmass / rho_exact) ** (1.0 / 3.0) * hfact

plot_kitajima_style(
    x_f, P_f, rho_f, vx_f, h_f,
    x_exact, P_exact, rho_exact, vx_exact, h_exact,
    f"SR Sod Shock Tube t={t_target}", "sr_sod_tube_kitajima.png"
)

expect_rho = 0.2331247
expect_vx = 0.4112091
expect_vy = 0.001167533
expect_vz = 8.164854e-05
expect_P = 0.3343444
tol = 1e-5

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
