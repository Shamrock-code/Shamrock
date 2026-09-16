"""
CI test: rendering a precomputed Field matches rendering by name
==================================================================

Creates a tiny SPH gas setup, then for each render entry point
(render_slice, render_column_integ, render_azymuthal_integ,
render_cartesian_slice, render_cartesian_column_integ) checks that
rendering a Field object obtained from model.compute_field(...) gives
the exact same result as rendering by field name directly. Also checks
that a Field derived with shamrock.map_fields_f64 renders correctly.
"""

import numpy as np

import shamrock


def build_model():
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_default_config()
    cfg.set_self_gravity_none()
    cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
    cfg.set_eos_isothermal(1.0)
    cfg.set_particle_mass(1e-3)
    cfg.set_boundary_periodic()
    model.set_solver_config(cfg)

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)

    model.init_scheduler(1000, 1)

    dr = 0.05
    bmin = (-0.6, -0.6, -0.6)
    bmax = (0.6, 0.6, 0.6)
    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen)

    def vel_func(r):
        x, y, z = r
        return (0.1 * y, -0.1 * x, 0.0)

    model.set_field_value_lambda_f64_3("vxyz", vel_func)

    model.timestep()

    return ctx, model


def check_equal(what, ref, got):
    ref = np.asarray(ref)
    got = np.asarray(got)
    if ref.shape != got.shape:
        raise RuntimeError(f"{what}: shape mismatch, ref={ref.shape}, got={got.shape}")
    if not np.array_equal(ref, got):
        abs_diff = np.abs(ref - got)
        raise RuntimeError(
            f"{what}: field-based render does not match name-based render\n"
            f"  max abs diff={np.max(abs_diff)}"
        )
    if shamrock.sys.world_rank() == 0:
        print(f"{what}: OK")


def main():
    ctx, model = build_model()

    positions = [
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.0, 0.1, 0.0),
        (0.2, 0.1, 0.0),
        (-0.15, -0.1, 0.05),
    ]

    rays = [shamrock.math.Ray_f64_3(pos, (0.0, 0.0, 1.0)) for pos in positions]

    ring_rays = [
        shamrock.math.RingRay_f64_3((0.0, 0.0, z), r, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        for z, r in [(-0.1, 0.05), (0.0, 0.1), (0.1, 0.15)]
    ]

    center = (0.0, 0.0, 0.0)
    delta_x = (0.5, 0.0, 0.0)
    delta_y = (0.0, 0.5, 0.0)
    nx = ny = 8

    # ---- scalar field (rho, Field_f64) ----
    rho_field = model.compute_field("rho", "f64")

    check_equal(
        "render_slice(rho)",
        model.render_slice("rho", "f64", positions),
        model.render_slice(rho_field, positions),
    )

    check_equal(
        "render_column_integ(rho)",
        model.render_column_integ("rho", "f64", rays),
        model.render_column_integ(rho_field, rays),
    )

    check_equal(
        "render_azymuthal_integ(rho)",
        model.render_azymuthal_integ("rho", "f64", ring_rays),
        model.render_azymuthal_integ(rho_field, ring_rays),
    )

    check_equal(
        "render_cartesian_slice(rho)",
        model.render_cartesian_slice(
            "rho", "f64", center=center, delta_x=delta_x, delta_y=delta_y, nx=nx, ny=ny
        ),
        model.render_cartesian_slice(
            rho_field, center=center, delta_x=delta_x, delta_y=delta_y, nx=nx, ny=ny
        ),
    )

    check_equal(
        "render_cartesian_column_integ(rho)",
        model.render_cartesian_column_integ(
            "rho", "f64", center=center, delta_x=delta_x, delta_y=delta_y, nx=nx, ny=ny
        ),
        model.render_cartesian_column_integ(
            rho_field, center=center, delta_x=delta_x, delta_y=delta_y, nx=nx, ny=ny
        ),
    )

    # ---- vector field (vxyz, Field_f64_3) ----
    vxyz_field = model.compute_field("vxyz", "f64_3")

    check_equal(
        "render_slice(vxyz)",
        model.render_slice("vxyz", "f64_3", positions),
        model.render_slice(vxyz_field, positions),
    )

    check_equal(
        "render_cartesian_slice(vxyz)",
        model.render_cartesian_slice(
            "vxyz", "f64_3", center=center, delta_x=delta_x, delta_y=delta_y, nx=nx, ny=ny
        ),
        model.render_cartesian_slice(
            vxyz_field, center=center, delta_x=delta_x, delta_y=delta_y, nx=nx, ny=ny
        ),
    )

    # ---- derived field via shamrock.map_fields_f64 ----
    def scale_by_two(size, x):
        return 2.0 * x

    rho_x2_field = shamrock.map_fields_f64(scale_by_two, x=rho_field)

    base = np.asarray(model.render_slice(rho_field, positions))
    derived = np.asarray(model.render_slice(rho_x2_field, positions))
    if not np.allclose(derived, 2.0 * base, rtol=1e-12, atol=1e-18):
        raise RuntimeError(
            "render_slice(map_fields_f64(2*rho)) does not match 2*render_slice(rho)\n"
            f"  base={base}\n"
            f"  derived={derived}"
        )
    if shamrock.sys.world_rank() == 0:
        print("render_slice(map_fields_f64 derived field): OK")

    if shamrock.sys.world_rank() == 0:
        print("run_sph_render_compute_field: OK")


main()
