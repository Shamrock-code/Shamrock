"""
CI test: Field.collect_data returns arrays that line up across Fields
=====================================================================

Two Fields built with different histories (one from model.compute_field, one
derived from it with shamrock.map_fields_f64) hold their patches in hash maps
that can iterate in different orders. collect_data must still concatenate
the patches in the same (patch id) order, so that element i of both arrays
refers to the same particle.
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

    # low split criterion so that every rank holds several patches
    model.init_scheduler(1000, 1)

    dr = 0.05
    bmin = (-0.6, -0.6, -0.6)
    bmax = (0.6, 0.6, 0.6)
    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen)

    model.timestep()

    return ctx, model


def main():
    ctx, model = build_model()

    rho_field = model.compute_field("rho", "f64")

    def scale_by_two(size, x):
        return 2.0 * x

    rho_x2_field = shamrock.map_fields_f64(scale_by_two, x=rho_field)

    rho = np.asarray(rho_field.collect_data())
    rho_x2 = np.asarray(rho_x2_field.collect_data())

    if rho.shape != rho_x2.shape:
        raise RuntimeError(f"shape mismatch: rho={rho.shape}, rho_x2={rho_x2.shape}")

    # multiplying by two is exact, so any mismatch means the patches are not aligned
    if not np.array_equal(rho_x2, 2.0 * rho):
        bad = np.count_nonzero(rho_x2 != 2.0 * rho)
        raise RuntimeError(
            "map_fields_f64(2*rho).collect_data() is not aligned with rho.collect_data()\n"
            f"  {bad} / {rho.size} mismatching elements"
        )

    if shamrock.sys.world_rank() == 0:
        print(f"collect_data alignment ({rho.size} values): OK")
        print("run_field_collect_data_alignment: OK")


main()
