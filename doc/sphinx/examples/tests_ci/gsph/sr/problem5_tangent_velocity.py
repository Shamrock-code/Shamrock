"""
Problem 5: 1D Shock Tube with Tangential Velocity (Kitajima et al. 2025, Section 3.1.5)
=======================================================================================

CI test matching arXiv:2510.18251v1 Figure 7

Initial conditions (strong blast with tangential velocity):
    Left:  (P, n, v^x, v^t) = (1000, 1, 0, v_t)
    Right: (P, n, v^x, v^t) = (0.01, 1, 0, v_t)

Test cases:
    - v_t = 0.0  (reference)
    - v_t = 0.9  (highly relativistic)
    - v_t = 0.99 (ultra-relativistic)

Kitajima setup:
    - 1600 particles each side
    - t = 0.16, γ = 5/3
"""

import sys
from pathlib import Path

import numpy as np

import shamrock

# Add this directory to path for local imports
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

SRRP_PATH = THIS_DIR.parent.parent.parent.parent.parent.parent.parent / "docs/papers/sg-gsph/srrp"
sys.path.insert(0, str(SRRP_PATH))
from kitajima_plotting import compute_L2_errors, plot_kitajima_4panel
from srrp.Solver import Solver
from srrp.State import State


def run_tangent_test(v_t, save_suffix=""):
    """Run tangential velocity test with given v_t."""

    # Kitajima Problem 5 parameters (same as Problem 3 but with tangential velocity)
    gamma = 5.0 / 3.0
    n_L, n_R = 1.0, 1.0
    P_L, P_R = 1000.0, 0.01
    u_L = P_L / ((gamma - 1) * n_L)
    u_R = P_R / ((gamma - 1) * n_R)
    t_target = 0.16

    # Resolution (quasi-1D) - target ~18k particles
    resol_x = 160  # ~1600 particles per side
    resol_yz = 8

    # Solve exact Riemann problem (with tangential velocity)
    solver = Solver()
    stateL = State(rho=n_L, vx=0.0, vt=v_t, pressure=P_L)
    stateR = State(rho=n_R, vx=0.0, vt=v_t, pressure=P_R)
    wavefan = solver.solve(stateL, stateR, gamma)

    print(f"\nSR Tangential Velocity Test v_t={v_t}")
    print(f"  Star state: P*={wavefan.states[1].pressure:.6f}, v*={wavefan.states[1].vx:.6f}")

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="TGauss3")

    cfg = model.gen_default_config()
    cfg.set_reconstruct_piecewise_constant()
    cfg.set_boundary_periodic()  # PERIODIC - FREE has issues with extreme pressure ratio
    cfg.set_eos_adiabatic(gamma)
    cfg.set_c_smooth(2.0)
    model.set_solver_config(cfg)
    model.set_physics_sr(c_speed=1.0)
    model.init_scheduler(int(1e8), 1)

    (xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol_x, resol_yz, resol_yz)
    dr = 1 / xs
    (xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol_x, resol_yz, resol_yz)
    model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
    hfact = model.get_hfact()

    # Uniform spacing
    model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
    model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

    model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
    model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

    # Set tangential velocity in y-direction
    model.set_field_in_box(
        "vxyz", "f64_3", (0.0, v_t, 0.0), (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2)
    )

    init_data = ctx.collect_data()
    xyz_init = np.array(init_data["xyz"])
    n_left = np.sum(xyz_init[:, 0] < 0)
    n_right = np.sum(xyz_init[:, 0] >= 0)
    V_per_particle = xs * ys * zs / n_left

    nu_L = n_L * V_per_particle
    nu_R = n_R * V_per_particle
    model.set_field_in_box("pmass", "f64", nu_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
    model.set_field_in_box("pmass", "f64", nu_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

    vol_total = 2 * xs * ys * zs
    totmass = n_L * xs * ys * zs + n_R * xs * ys * zs
    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)

    h_init = hfact * V_per_particle ** (1 / 3)
    model.set_field_in_box("hpart", "f64", h_init, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

    model.set_cfl_cour(0.2)
    model.set_cfl_force(0.2)

    print(f"  Particles: {n_left + n_right}")
    print(f"  Running to t={t_target}...")
    model.evolve_until(t_target)

    # Collect data - use direct physics from solver
    data = ctx.collect_data()
    physics = model.collect_physics_data()

    points = np.array(data["xyz"])
    velocities = np.array(data["vxyz"])
    hpart = np.array(data["hpart"])

    x = points[:, 0]
    vx = velocities[:, 0]

    # Direct values from solver
    n_sim = np.array(physics["N_labframe"])
    P_sim = np.array(physics["pressure"])

    # Compute lorentz factor from velocity
    v2 = np.sum(velocities**2, axis=1)
    gamma_lor = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-10))

    # Exact solution
    x_exact = np.linspace(-0.5, 0.5, 1000)
    xi = x_exact / t_target
    state_exact = wavefan.getState(xi)
    P_exact = state_exact.pressure
    n_exact = state_exact.rho
    vx_exact = state_exact.vx

    # Compute errors
    err_P = compute_L2_errors(x, P_sim, x_exact, P_exact)
    err_n = compute_L2_errors(x, n_sim, x_exact, n_exact)
    err_vx = compute_L2_errors(x, vx, x_exact, vx_exact)

    print(f"  L2 errors: rho={err_n:.6e}, vx={err_vx:.6e}, P={err_P:.6e}")

    # Plot
    filename = f"sr_tangent_vt{v_t}{save_suffix}.png"
    plot_kitajima_4panel(
        x,
        P_sim,
        n_sim,
        vx,
        hpart,
        x_exact,
        P_exact,
        n_exact,
        vx_exact,
        filename,
        f"SR Tangent v_t={v_t} (t={t_target})",
    )

    return err_n, err_vx, err_P


# Run all tangential velocity tests
print("=" * 60)
print("Problem 5: Tangential Velocity Tests (Kitajima Section 3.1.5)")
print("=" * 60)

# Test cases from Kitajima paper
test_cases = [0.0, 0.9, 0.99]
results = {}

for v_t in test_cases:
    err_n, err_vx, err_P = run_tangent_test(v_t)
    results[v_t] = (err_n, err_vx, err_P)

# Summary and regression check
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

test_pass = True
# Allow progressively larger errors for higher tangential velocities
tolerances = {0.0: 0.5, 0.9: 0.6, 0.99: 0.8}

for v_t, (err_n, err_vx, err_P) in results.items():
    tol = tolerances[v_t]
    expect = 0.35  # Base expectation
    passed = err_n < expect * (1 + tol) and err_vx < expect * (1 + tol)
    status = "PASS" if passed else "FAIL"
    print(f"  v_t={v_t:4.2f}: err_n={err_n:.4f}, err_vx={err_vx:.4f}, err_P={err_P:.4f} [{status}]")
    if not passed:
        test_pass = False

if test_pass:
    print("\n" + "=" * 50)
    print("SR Tangent Velocity Problem 5: ALL PASSED")
    print("=" * 50)
else:
    print("\n" + "=" * 50)
    print("SR Tangent Velocity Problem 5: SOME FAILED")
    print("=" * 50)
    exit(1)
