"""
Testing the external forces coupling modes in RAMSES solver
===========================================================

CI test for the two time quadratures of the external forces source term
(``cfg.set_ext_force_coupling``), with a constant gravitational acceleration:

* free fall of a uniform periodic box (gas + one dust species): the momentum must follow
  :math:`\\rho \\mathbf{g} t` exactly in both modes. The trapezoidal rule must conserve the
  internal energy, while the midpoint rule has the known
  :math:`-\\rho g^2 \\Delta t^2 / 2` drift per step (see the TODO in ``ApplyExtForceGas.cpp``).
  This is checked with and without the time interpolation of the face states.
* one step on a non uniform moving state: for a constant acceleration both modes give the same momentum
  update, and their energy differs by :math:`\\Delta t^2 \\rho^{n+1/2} g^2 / 2`.
"""

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

from shamrock.model_ramses import ExtForceCoupling

gamma = 1.4
g = np.array([0.0, -1.0, 0.0])
g_norm2 = float(np.dot(g, g))

sz = 1 << 1
base = 4


def zero_vel(rmin, rmax):
    return (0.0, 0.0, 0.0)


def run(
    mode,
    rho_map,
    rho_dust_map,
    dt,
    nsteps,
    vel_map=zero_vel,
    vel_dust_map=zero_vel,
    face_time_interpolation=True,
):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(1 / (sz * base))
    cfg.set_eos_gamma(gamma)
    cfg.set_dust_mode_dhll(1)
    cfg.set_face_time_interpolation(face_time_interpolation)
    cfg.add_ext_force_constant_gravity(g=tuple(g))
    cfg.set_ext_force_coupling(mode)
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base, base, base))

    model.set_field_value_lambda_f64("rho", rho_map)

    def rhovel_map(rmin, rmax):
        return tuple(rho_map(rmin, rmax) * np.array(vel_map(rmin, rmax)))

    def rhoetot_map(rmin, rmax):
        v = np.array(vel_map(rmin, rmax))
        return 0.5 * rho_map(rmin, rmax) * float(np.dot(v, v)) + 1.0 / (gamma - 1.0)

    def rhovel_dust_map(rmin, rmax):
        return tuple(rho_dust_map(rmin, rmax) * np.array(vel_dust_map(rmin, rmax)))

    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64("rho_dust", rho_dust_map, 0)
    model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_dust_map, 0)

    data_init = ctx.collect_data()

    for i in range(nsteps):
        model.evolve_once_override_time(dt * float(i), dt)

    return data_init, ctx.collect_data()


def internal_energy(data):
    rho = np.array(data["rho"])
    rhov = np.array(data["rhovel"])
    return np.array(data["rhoetot"]) - 0.5 * np.sum(rhov * rhov, axis=1) / rho


def max_rel_err(a, b, scale):
    return float(np.max(np.abs(np.array(a) - np.array(b)))) / scale


def test_free_fall():
    rho0, rho_d0 = 1.5, 0.7
    dt, nsteps = 0.01, 20
    t = dt * nsteps

    cases = [
        (mode, face_interp)
        for mode in [ExtForceCoupling.Midpoint, ExtForceCoupling.Trapezoidal]
        for face_interp in [True, False]
    ]
    for mode, face_interp in cases:
        data_init, data = run(
            mode,
            lambda rmin, rmax: rho0,
            lambda rmin, rmax: rho_d0,
            dt,
            nsteps,
            face_time_interpolation=face_interp,
        )

        # the collected data is only complete on rank 0
        if shamrock.sys.world_rank() != 0:
            continue

        # no pressure gradient: the gas and the dust are in free fall
        err_rhov = max_rel_err(data["rhovel"], [rho0 * g * t], rho0 * t)
        err_rhov_d = max_rel_err(data["rhovel_dust"], [rho_d0 * g * t], rho_d0 * t)

        de_int = internal_energy(data) - internal_energy(data_init)
        if mode == ExtForceCoupling.Trapezoidal:
            de_int_expected = 0.0
        else:
            de_int_expected = -nsteps * rho0 * g_norm2 * dt**2 / 2
        err_e_int = float(np.max(np.abs(de_int - de_int_expected))) / (1.0 / (gamma - 1.0))

        print(
            f"free fall {mode}, face interp {face_interp}: err rhovel = {err_rhov:.3e}, err rhovel_dust = {err_rhov_d:.3e}, "
            f"err e_int = {err_e_int:.3e}"
        )

        assert err_rhov < 1e-13, (
            f"{mode}, face interp {face_interp}: gas momentum is not rho g t ({err_rhov})"
        )
        assert err_rhov_d < 1e-13, (
            f"{mode}, face interp {face_interp}: dust momentum is not rho_d g t ({err_rhov_d})"
        )
        assert err_e_int < 1e-12, (
            f"{mode}, face interp {face_interp}: unexpected internal energy change ({err_e_int})"
        )


def test_modes_one_step():
    def rho_map(rmin, rmax):
        x, y, z = (np.array(rmin) + np.array(rmax)) / 2
        return 1.0 + 0.3 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

    def rho_dust_map(rmin, rmax):
        x, y, z = (np.array(rmin) + np.array(rmax)) / 2
        return 0.5 + 0.2 * np.cos(2 * np.pi * z)

    # a non uniform velocity, so that the flux derivatives (dt rho, dt rhov) are not zero
    def vel_map(rmin, rmax):
        x, y, z = (np.array(rmin) + np.array(rmax)) / 2
        return (0.3 * np.sin(2 * np.pi * y), 0.2 * np.cos(2 * np.pi * x), 0.1)

    def vel_dust_map(rmin, rmax):
        x, y, z = (np.array(rmin) + np.array(rmax)) / 2
        return (0.1, 0.2 * np.sin(2 * np.pi * z), 0.3 * np.cos(2 * np.pi * x))

    dt = 0.01

    args = (rho_map, rho_dust_map, dt, 1, vel_map, vel_dust_map)
    data_init, data_mid = run(ExtForceCoupling.Midpoint, *args)
    _, data_trap = run(ExtForceCoupling.Trapezoidal, *args)

    # the collected data is only complete on rank 0
    if shamrock.sys.world_rank() != 0:
        return

    # check that the setup does exercise the flux derivatives
    assert not np.array_equal(np.array(data_init["rho"]), np.array(data_mid["rho"]))
    assert not np.array_equal(np.array(data_init["rho_dust"]), np.array(data_mid["rho_dust"]))

    # the density update does not depend on the external forces
    assert np.array_equal(np.array(data_mid["rho"]), np.array(data_trap["rho"]))

    rhov_scale = float(np.max(np.abs(np.array(data_mid["rhovel"]))))
    rhov_d_scale = float(np.max(np.abs(np.array(data_mid["rhovel_dust"]))))
    err_rhov = max_rel_err(data_mid["rhovel"], data_trap["rhovel"], rhov_scale)
    err_rhov_d = max_rel_err(data_mid["rhovel_dust"], data_trap["rhovel_dust"], rhov_d_scale)

    rho_half = 0.5 * (np.array(data_init["rho"]) + np.array(data_mid["rho"]))
    de_expected = dt**2 * rho_half * g_norm2 / 2
    de = np.array(data_trap["rhoetot"]) - np.array(data_mid["rhoetot"])
    err_de = float(np.max(np.abs(de - de_expected) / de_expected))

    print(
        f"one step midpoint vs trapezoidal: err rhovel = {err_rhov:.3e}, "
        f"err rhovel_dust = {err_rhov_d:.3e}, err energy difference = {err_de:.3e}"
    )

    assert err_rhov < 1e-13, f"modes give different gas momentum ({err_rhov})"
    assert err_rhov_d < 1e-13, f"modes give different dust momentum ({err_rhov_d})"
    assert err_de < 1e-8, f"unexpected energy difference between the modes ({err_de})"


test_free_fall()
test_modes_one_step()

if shamrock.sys.world_rank() == 0:
    print("ramses_ext_force_coupling: OK")
