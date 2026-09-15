"""
CI test: external forces point mass accretion
=============================================

Creates a tiny SPH setup with a few point mass external forces with an
accretion radius, evolves, dumps, reloads into a fresh context, evolves
again, and checks the resulting particle count and field sums.
"""

import numpy as np

import shamrock

DUMP_NAME = "ext_forces_accretion_test.sham"

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()


def build_model_with_ext_forces():
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_default_config()
    cfg.set_self_gravity_none()
    cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
    cfg.set_eos_isothermal(1.0)
    cfg.set_particle_mass(1e-3)
    cfg.set_boundary_periodic()
    cfg.set_show_cfl_detail(True)
    cfg.set_units(codeu)

    cfg.add_ext_force_paczynski_wiita(1.0, (0.1, 0.0, 0.0), 0.15)
    cfg.add_ext_force_paczynski_wiita(0.5, (-0.2, 0.1, 0.0), 0.15)
    cfg.add_ext_force_paczynski_wiita(0.25, (0.0, -0.15, 0.05), 0.15)

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
        return (10.0, 0.0, 0.0)

    model.set_field_value_lambda_f64_3("vxyz", vel_func)

    return ctx, model


def main():
    ctx, model = build_model_with_ext_forces()

    for _ in range(5):
        model.timestep()

    model.dump(DUMP_NAME)

    del model
    del ctx

    ctx2 = shamrock.Context()
    ctx2.pdata_layout_new()
    model2 = shamrock.get_Model_SPH(context=ctx2, vector_type="f64_3", sph_kernel="M4")
    model2.load_from_dump(DUMP_NAME)

    for _ in range(5):
        model2.timestep()

    if shamrock.sys.world_rank() == 0:
        print("run_test_ext_forces_accretion: OK")

    dic = ctx2.collect_data()

    if shamrock.sys.world_rank() > 0:
        return

    assert 2262 == len(dic["xyz"])

    sum_pos = np.sum(dic["xyz"], axis=0)
    sum_vel = np.sum(dic["vxyz"], axis=0)
    sum_acc = np.sum(dic["axyz"], axis=0)
    sum_hpart = np.sum(dic["hpart"], axis=0)

    dat = np.concatenate([sum_pos, sum_vel, sum_acc, np.atleast_1d(sum_hpart)])
    print("Current sums: ", [float(dat[i]) for i in range(len(dat))])

    ref_sums = [
        111.13342631354446,
        -15.790263346174845,
        -1.0024165122154156,
        22523.573252726077,
        0.16482394866530714,
        25.96527810306144,
        -25326.18011836384,
        -69.26477834526008,
        6203.398030417404,
        245.61542121817263,
    ]

    mismatch = False
    for i in range(len(dat)):
        if not np.isclose(dat[i], ref_sums[i], rtol=1e-12, atol=1e-18):
            abs_diff = np.abs(dat[i] - ref_sums[i])
            rel_diff = abs_diff / np.abs(ref_sums[i])
            print(f"sum[{i}] mismatch: got {dat[i]}, expected {ref_sums[i]}")
            print(f"  max abs diff={np.max(abs_diff)}")
            print(f"  max rel diff={np.max(rel_diff)}")
            mismatch = True
    if mismatch:
        raise RuntimeError("sums mismatch")


main()
