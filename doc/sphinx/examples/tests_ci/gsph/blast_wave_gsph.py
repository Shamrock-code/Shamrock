"""
Testing Extreme Blast Wave with GSPH
====================================

CI test for the extreme blast wave problem from Inutsuka 2002 (Section 4.3).
This is a severe test with Mach number ~10^5.

Initial conditions (from Inutsuka 2002):
    rho_L = 1,      rho_R = 1
    P_L   = 3000,   P_R   = 1e-7
    v_L   = 0,      v_R   = 0
"""

import numpy as np

import shamrock

gamma = 1.4
rho_L, rho_R = 1.0, 1.0
P_L, P_R = 3000.0, 1e-7
u_L = P_L / ((gamma - 1) * rho_L)
u_R = P_R / ((gamma - 1) * rho_R)
resol = 100

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
cfg = model.gen_default_config()
cfg.set_riemann_hllc()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
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
totmass = (rho_R * vol_b) + (rho_L * vol_b)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.3)
model.set_cfl_force(0.3)

t_target = 0.015
print(f"GSPH Extreme Blast Wave Test (M4, HLLC, t={t_target})")
model.evolve_until(t_target)

sod = shamrock.phys.SodTube(gamma=gamma, rho_1=rho_L, P_1=P_L, rho_5=rho_R, P_5=P_R)


def compute_L2_errors(ctx, sod, t, x_min, x_max):
    """Compute L2 errors using ctx.collect_data() (no pyvista dependency)."""
    data = ctx.collect_data()
    points = np.array(data["xyz"])
    velocities = np.array(data["vxyz"])
    hpart = np.array(data["hpart"])
    uint = np.array(data["uint"])

    rho_sim = pmass * (hfact / hpart) ** 3
    P_sim = (gamma - 1) * rho_sim * uint

    x, vx, vy, vz = points[:, 0], velocities[:, 0], velocities[:, 1], velocities[:, 2]
    mask = (x >= x_min) & (x <= x_max)
    x_f, rho_f, vx_f, vy_f, vz_f, P_f = (
        x[mask],
        rho_sim[mask],
        vx[mask],
        vy[mask],
        vz[mask],
        P_sim[mask],
    )

    if len(x_f) == 0:
        raise RuntimeError("No particles in analysis region")

    rho_ana, vx_ana, P_ana = np.zeros(len(x_f)), np.zeros(len(x_f)), np.zeros(len(x_f))
    for i, xi in enumerate(x_f):
        rho_ana[i], vx_ana[i], P_ana[i] = sod.get_value(t, xi)

    rho_norm = max(np.mean(rho_ana), 1e-10)
    vx_norm = max(np.mean(np.abs(vx_ana)), 0.1)
    P_norm = max(np.mean(P_ana), 1e-10)

    err_rho = np.sqrt(np.mean((rho_f - rho_ana) ** 2)) / rho_norm
    err_vx = np.sqrt(np.mean((vx_f - vx_ana) ** 2)) / vx_norm
    err_vy = np.sqrt(np.mean(vy_f**2))
    err_vz = np.sqrt(np.mean(vz_f**2))
    err_P = np.sqrt(np.mean((P_f - P_ana) ** 2)) / P_norm
    return err_rho, (err_vx, err_vy, err_vz), err_P


rho, v, P = compute_L2_errors(ctx, sod, t_target, -0.5, 0.5)
vx, vy, vz = v

print("current errors :")
print(f"err_rho = {rho}")
print(f"err_vx = {vx}")
print(f"err_vy = {vy}")
print(f"err_vz = {vz}")
print(f"err_P = {P}")

# Expected L2 error values (calibrated from CI run with M4 kernel)
expect_rho = 10.688658207003348
expect_vx = 1.0420471749025182
expect_vy = 0.11766417324542999
expect_vz = 0.0027436730451881886
expect_P = 1.6660643954434153

tol = 1e-8

test_pass = True
err_log = ""

error_checks = {
    "rho": (rho, expect_rho),
    "vx": (vx, expect_vx),
    "vy": (vy, expect_vy),
    "vz": (vz, expect_vz),
    "P": (P, expect_P),
}

for name, (value, expected) in error_checks.items():
    if abs(value - expected) > tol * expected:
        err_log += f"error on {name} is outside of tolerances:\n"
        err_log += f"  expected error = {expected} +- {tol*expected}\n"
        err_log += f"  obtained error = {value} (relative error = {(value - expected)/expected})\n"
        test_pass = False

if test_pass:
    print("\n" + "=" * 50)
    print("GSPH Extreme Blast Wave Test: PASSED")
    print("=" * 50)
else:
    exit("Test did not pass L2 margins : \n" + err_log)
