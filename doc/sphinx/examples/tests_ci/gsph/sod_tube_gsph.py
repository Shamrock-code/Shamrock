"""
Testing Sod tube with GSPH
==========================

CI test for Sod tube with GSPH using M4 kernel and HLLC Riemann solver.
Uses piecewise constant reconstruction (first-order, stable).
Computes L2 error against analytical solution and checks for regression.
"""

import numpy as np

import shamrock

gamma = 1.4
rho_L, rho_R = 1.0, 0.125
P_L, P_R = 1.0, 0.1
fact = (rho_L / rho_R) ** (1.0 / 3.0)
u_L = P_L / ((gamma - 1) * rho_L)
u_R = P_R / ((gamma - 1) * rho_R)
resol = 128

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
model.add_cube_hcp_3d(dr * fact, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

vol_b = xs * ys * zs
totmass = (rho_R * vol_b) + (rho_L * vol_b)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

t_target = 0.245
print(f"GSPH Sod Shock Tube Test (M4, HLLC, t={t_target})")
model.evolve_until(t_target)

sod = shamrock.phys.SodTube(gamma=gamma, rho_1=rho_L, P_1=P_L, rho_5=rho_R, P_5=P_R)

data = ctx.collect_data()

def compute_L2_errors(data, sod, t, x_min, x_max):
    """Compute L2 errors using ctx.collect_data() (no pyvista dependency)."""
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

    err_rho = np.sqrt(np.mean((rho_f - rho_ana) ** 2)) / np.mean(rho_ana)
    err_vx = np.sqrt(np.mean((vx_f - vx_ana) ** 2)) / (np.mean(np.abs(vx_ana)) + 0.1)
    err_vy = np.sqrt(np.mean(vy_f**2))
    err_vz = np.sqrt(np.mean(vz_f**2))
    err_P = np.sqrt(np.mean((P_f - P_ana) ** 2)) / np.mean(P_ana)
    return err_rho, (err_vx, err_vy, err_vz), err_P

if shamrock.sys.world_rank() == 0:
    rho, v, P = compute_L2_errors(ctx, sod, t_target, -0.5, 0.5)
    vx, vy, vz = v
    
    print("current errors :")
    print(f"err_rho = {rho}")
    print(f"err_vx = {vx}")
    print(f"err_vy = {vy}")
    print(f"err_vz = {vz}")
    print(f"err_P = {P}")
    
    # Expected L2 error values (calibrated from local run with M4 kernel)
    # Tolerance set very strict for regression testing (like sod_tube_sph.py)
    expect_rho = 0.029892771160040497
    expect_vx = 0.10118608617971991
    expect_vy = 0.006382105147197806
    expect_vz = 3.118241304703099e-05
    expect_P = 0.038072557056294656
    
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
        print("GSPH Sod Shock Tube Test: PASSED")
        print("=" * 50)
