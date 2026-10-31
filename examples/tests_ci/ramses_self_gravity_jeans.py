"""
Testing the RAMSES solver self-gravity with the Jeans instability
=================================================================

CI test for the self-gravity of the RAMSES solver, applied through the external forces with the
trapezoidal coupling (``cfg.set_ext_force_coupling(ExtForceCoupling.Trapezoidal)``).

A uniform periodic box with a small standing density perturbation
:math:`\\delta \\rho = A \\rho_0 \\cos(k x)` (gas at rest, isentropic) follows the linear
dispersion relation :math:`\\omega^2 = c_s^2 k^2 - 4 \\pi G \\rho_0`:

* above the Jeans length (:math:`c_s k < \\sqrt{4 \\pi G \\rho_0}`) the perturbation grows as
  :math:`\\cosh(\\gamma t)`, with :math:`\\gamma^2 = 4 \\pi G \\rho_0 - c_s^2 k^2`,
* below it, it oscillates as :math:`\\cos(\\omega t)`, slower than a sound wave.

The growth rate (with both Poisson solvers) and the oscillation frequency are checked against the
linear theory. The gravitational constant comes from the unit system, set so that :math:`G = 1`.
The box is split into several patches, so that the ghost exchanges of the Poisson solvers are
tested between patches (and between ranks when run with MPI).
"""

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

from shamrock.model_ramses import ExtForceCoupling

shamrock.enable_experimental_features()

gamma = 5.0 / 3.0
rho0 = 1.0
amp = 1e-3
k = 2 * np.pi  # one wavelength in the unit box

# G = 1 in the code units (unit_mass = 1 / G_SI kg)
G_SI = shamrock.Constants(shamrock.UnitSystem()).G()
units = shamrock.UnitSystem(unit_time=1.0, unit_length=1.0, unit_mass=1.0 / G_SI)
G = shamrock.Constants(units).G()
four_pi_G_rho0 = 4 * np.pi * G * rho0

sz = 1 << 1
base = 8  # 16 cells per wavelength
block_size = 8  # cells per AMR block


def run(cs, solver, dt, nsteps):
    """Return the times and the amplitudes of the cos(k x) mode of the density"""

    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(1 / (sz * base))
    cfg.set_eos_gamma(gamma)
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_riemann_solver_hllc()
    cfg.set_face_time_interpolation(True)
    cfg.set_units(units)
    if solver == "cg":
        cfg.set_gravity_mode_cg()
    else:
        cfg.set_gravity_mode_bicgstab()
    # well above the round-off floor of the (singular, periodic) Poisson problem: when the
    # perturbation is small (e.g. near a zero of cos(omega t)) a tolerance at that floor is not
    # reached and the CG iterations diverge (see the TODO in NodeCGLoop)
    cfg.set_self_gravity_tol(1e-8)
    cfg.set_self_gravity_Niter_max(500)
    cfg.set_ext_force_coupling(ExtForceCoupling.Trapezoidal)
    model.set_solver_config(cfg)

    # split the 512 blocks into several patches
    model.init_scheduler(64, 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base, base, base))

    P0 = cs * cs * rho0 / gamma

    def rho_map(rmin, rmax):
        x = 0.5 * (rmin[0] + rmax[0])
        return rho0 * (1.0 + amp * np.cos(k * x))

    def rhoetot_map(rmin, rmax):
        # isentropic perturbation
        return P0 * (rho_map(rmin, rmax) / rho0) ** gamma / (gamma - 1.0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64_3("rhovel", lambda rmin, rmax: (0.0, 0.0, 0.0))
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64("phi", lambda rmin, rmax: 0.0)
    model.set_field_value_lambda_f64("phi_old", lambda rmin, rmax: 0.0)

    def mode_amplitude():
        data = ctx.collect_data()
        if shamrock.sys.world_rank() != 0:
            return 0.0, 0.0
        rho = np.array(data["rho"])
        x = np.array(
            [
                0.5 * (rmin[0] + rmax[0])
                for cmin, cmax in zip(data["cell_min"], data["cell_max"])
                for rmin, rmax in (
                    model.get_cell_coords((tuple(map(int, cmin)), tuple(map(int, cmax))), lid)
                    for lid in range(block_size)
                )
            ]
        )

        a = 2 * np.mean((rho - rho0) * np.cos(k * x)) / (rho0 * amp)
        p_tot = np.sum(np.array(data["rhovel"]), axis=0)
        return a, float(np.max(np.abs(p_tot)))

    times = [0.0]
    a0, _ = mode_amplitude()
    amps = [a0]
    p_max = 0.0
    for i in range(nsteps):
        model.evolve_once_override_time(dt * float(i), dt)
        a, p = mode_amplitude()
        times.append(dt * (i + 1))
        amps.append(a)
        p_max = max(p_max, p)

    # every rank must own patches, otherwise the exchanges between ranks are not tested
    owners = [p.node_owner_id for p in ctx.get_patch_list_global()]
    patches_per_rank = np.bincount(owners, minlength=shamrock.sys.world_size())
    if shamrock.sys.world_rank() == 0:
        print(f"patches per rank: {patches_per_rank.tolist()}")
    assert len(owners) > 1, "the box is not split into several patches"
    assert np.all(patches_per_rank > 0), f"a rank owns no patch ({patches_per_rank.tolist()})"

    return np.array(times), np.array(amps), p_max


def test_unstable(solver):
    cs = 0.3
    growth_th = np.sqrt(four_pi_G_rho0 - cs * cs * k * k)

    dt, nsteps = 0.01, 50
    t, a, p_max = run(cs, solver, dt, nsteps)

    if shamrock.sys.world_rank() != 0:
        return

    # measured growth rate from a(t) = a(0) cosh(growth t) at the end of the run
    growth = np.arccosh(a[-1] / a[0]) / t[-1]
    rel_err = abs(growth - growth_th) / growth_th

    print(
        f"Jeans unstable ({solver}): a(t_end)/a(0) = {a[-1] / a[0]:.6f} "
        f"(theory {np.cosh(growth_th * t[-1]):.6f}), growth rate = {growth:.6f} "
        f"(theory {growth_th:.6f}), rel err = {rel_err:.3e}, max |total momentum| = {p_max:.3e}"
    )

    assert abs(a[0] - 1.0) < 1e-2, f"bad initial mode amplitude ({a[0]})"
    # the first step only sees the pressure in the density update (the face predictor has no
    # gravity source term, as in the reference scheme), so the growth starts at the second step
    assert np.all(np.diff(a[1:]) > 0), f"{solver}: the perturbation does not grow monotonically"
    assert rel_err < 0.05, f"{solver}: growth rate {growth} too far from theory {growth_th}"
    assert p_max < 1e-12, f"{solver}: the total momentum is not conserved ({p_max})"

    return growth


def test_stable():
    cs = 1.0
    omega_th = np.sqrt(cs * cs * k * k - four_pi_G_rho0)
    omega_sound = cs * k

    dt, nsteps = 0.01, 40
    t, a, p_max = run(cs, "cg", dt, nsteps)

    if shamrock.sys.world_rank() != 0:
        return

    # first zero of a(t) = cos(omega t), at t = pi / (2 omega)
    i = int(np.argmax(a < 0))
    assert i > 0, "the perturbation does not oscillate"
    t_zero = t[i - 1] + (t[i] - t[i - 1]) * a[i - 1] / (a[i - 1] - a[i])
    omega = np.pi / (2 * t_zero)
    rel_err = abs(omega - omega_th) / omega_th

    print(
        f"Jeans stable: omega = {omega:.6f} (theory {omega_th:.6f}, sound wave {omega_sound:.6f}), "
        f"rel err = {rel_err:.3e}, max |total momentum| = {p_max:.3e}"
    )

    assert rel_err < 0.05, f"oscillation frequency {omega} too far from theory {omega_th}"
    assert p_max < 1e-12, f"the total momentum is not conserved ({p_max})"


growth_cg = test_unstable("cg")
growth_bicgstab = test_unstable("bicgstab")
test_stable()

if shamrock.sys.world_rank() == 0:
    # both solvers converge to the same potential (tol = 1e-10)
    rel_diff = abs(growth_cg - growth_bicgstab) / growth_cg
    print(f"growth rate CG vs BiCGSTAB: rel diff = {rel_diff:.3e}")
    assert rel_diff < 1e-6, f"CG and BiCGSTAB give different growth rates ({rel_diff})"

    print("ramses_self_gravity_jeans: OK")
