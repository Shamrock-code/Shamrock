"""
CI test: SPH barotropic equation of state
==========================================

The barotropic equation of state

.. math::
    P = c_{s,0}^2 \\rho \\left[1 + \\left(\\frac{\\rho}{\\rho_0}\\right)^{\\gamma - 1}\\right]

is isothermal well below :math:`\\rho_0` and a polytrope
:math:`P = K \\rho^\\gamma` with :math:`K = c_{s,0}^2 \\rho_0^{1-\\gamma}` well above it.

This test evolves the same small periodic setup (a velocity perturbation in a
uniform box) with

* ``set_eos_barotropic`` with :math:`\\rho_0 \\gg \\rho`, compared to
  ``set_eos_isothermal(cs0)``,
* ``set_eos_barotropic`` with :math:`\\rho_0 \\ll \\rho`, compared to
  ``set_eos_polytropic(K, gamma)``,

and checks that both pairs of runs agree. It also checks that invalid
barotropic parameters are rejected.
"""

import numpy as np

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

pmass = 1e-3
dr = 0.05
bmin = (-0.6, -0.6, -0.6)
bmax = (0.6, 0.6, 0.6)
t_end = 0.05

# HCP neighbours are 2 dr apart, the number density is sqrt(2) / (2 dr)^3
rho_mean = pmass * np.sqrt(2) / (2 * dr) ** 3


def run(set_eos):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_default_config()
    cfg.set_self_gravity_none()
    cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
    set_eos(cfg)
    cfg.set_particle_mass(pmass)
    cfg.set_boundary_periodic()
    model.set_solver_config(cfg)

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)

    model.init_scheduler(1000, 1)
    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    setup.apply_setup(setup.make_generator_lattice_hcp(dr, bmin, bmax))

    L = bmax[0] - bmin[0]

    def vel_func(r):
        x, y, z = r
        return (0.05 * np.sin(2 * np.pi * x / L), 0.0, 0.0)

    model.set_field_value_lambda_f64_3("vxyz", vel_func)
    model.set_value_in_a_box("uint", "f64", 1.0, bmin, bmax)

    model.timestep()
    model.evolve_until(t_end)

    vxyz = np.asarray(ctx.collect_data()["vxyz"])
    # sorted speeds do not depend on the particle ordering
    return np.sort(np.linalg.norm(vxyz, axis=1))


def compare(name, ref, got, rtol):
    if ref.shape != got.shape:
        raise AssertionError(f"{name}: shape mismatch {ref.shape} vs {got.shape}")
    err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    print(f"{name}: max relative speed difference = {err:.3e} (tolerance {rtol:.1e})")
    if not err < rtol:
        raise AssertionError(f"{name}: barotropic run differs from the reference: {err:.3e}")


gamma = 5.0 / 3.0

# %%
# Isothermal limit: rho0 >> rho

cs0 = 1.0
v_iso = run(lambda cfg: cfg.set_eos_isothermal(cs0))
v_baro_iso = run(lambda cfg: cfg.set_eos_barotropic(cs0=cs0, rho0=1e30, gamma0=gamma))
compare("isothermal limit", v_iso, v_baro_iso, 1e-10)

# %%
# Adiabatic limit: rho0 << rho, choosing cs0 such that K = 1 (the sound speed stays ~1)
#
# The barotropic pressure differs from the polytrope by a relative (rho0 / rho)^(gamma - 1)

K = 1.0
rho0 = 1e-12 * rho_mean
cs0_poly = np.sqrt(K * rho0 ** (gamma - 1))

v_poly = run(lambda cfg: cfg.set_eos_polytropic(K, gamma))
v_baro_poly = run(lambda cfg: cfg.set_eos_barotropic(cs0=cs0_poly, rho0=rho0, gamma0=gamma))
compare("adiabatic limit", v_poly, v_baro_poly, 1e-6)

# %%
# Invalid parameters must be rejected when the config is set

for bad in [
    dict(cs0=-1.0, rho0=1.0, gamma0=gamma),
    dict(cs0=1.0, rho0=0.0, gamma0=gamma),
    dict(cs0=1.0, rho0=1.0, gamma0=1.0),
]:
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
    cfg = model.gen_default_config()
    cfg.set_eos_barotropic(**bad)
    try:
        model.set_solver_config(cfg)
    except ValueError as e:
        print(f"rejected {bad} as expected: {type(e).__name__}")
    else:
        raise AssertionError(f"invalid barotropic parameters {bad} were accepted")

print("barotropic EOS test passed")
