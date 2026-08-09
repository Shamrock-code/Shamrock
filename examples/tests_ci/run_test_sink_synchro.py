"""
CI test: sink state stays synchronized across MPI ranks through dump/reload
===========================================================================

Creates a tiny SPH setup with a few sinks and gas particles, checks sink sync,
evolves, dumps, reloads into a fresh context, and checks again.
"""

import numpy as np

import shamrock

DUMP_NAME = "sink_sync_test.sham"


def check_sinks_are_in_sync(ctx, model):
    s = str(model.get_sinks())
    # Collective: every rank must call this
    hist = shamrock.algs.all_string_histogram([s], delimiter="\n", hash_based=False)
    if len(hist) != 1:
        raise RuntimeError(f"sinks not in sync across ranks: {hist}")
    key, count = next(iter(hist.items()))
    if count != shamrock.sys.world_size():
        raise RuntimeError(
            f"expected count={shamrock.sys.world_size()}, got {count} for key={key!r}"
        )
    shamrock.sys.mpi_barrier()
    if shamrock.sys.world_rank() == 0:
        print("Sinks are in sync !")


si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()


def build_model_with_sinks():
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
    model.set_solver_config(cfg)

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)
    model.set_eta_sink(1.0)

    model.init_scheduler(1000, 1)

    # Very coarse HCP cube -> handful of SPH particles
    dr = 0.05
    bmin = (-0.6, -0.6, -0.6)
    bmax = (0.6, 0.6, 0.6)
    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen)

    eng = shamrock.algs.gen_seed(42)

    def vel_func(r):
        vx, vy, vz = shamrock.algs.mock_gaussian_f64_3(eng)
        return (10 * vx, 10 * vy, 10 * vz)

    model.set_field_value_lambda_f64_3("vxyz", vel_func)

    # A few sinks (must be added after init_scheduler, on all ranks)
    model.add_sink(1.0, (0.1, 0.0, 0.0), (0.0, 0.05, 0.0), 0.15)
    model.add_sink(0.5, (-0.2, 0.1, 0.0), (0.0, -0.03, 0.0), 0.15)
    model.add_sink(0.25, (0.0, -0.15, 0.05), (0.02, 0.0, 0.0), 0.15)

    return ctx, model


def check_ref_dataset(sinks):
    ref_dataset = [
        {
            "pos": (0.09996994962091943, -1.844475903942043e-05, -1.3115452028029389e-06),
            "velocity": (-0.15426123492821367, -0.0793787943797117, 0.018385275951724726),
            "sph_acceleration": (-7.951438161324546, 6.117833793490561, -2.7134097770298666),
            "ext_acceleration": (-356.95762815769365, -173.34543864966082, 79.66138320385133),
            "mass": 1.019,
            "angular_momentum": (
                0.004109434275900465,
                -0.0027544889436795033,
                -0.000884944960634745,
            ),
            "accretion_radius": 0.15,
        },
        {
            "pos": (-0.19955791618491797, 0.10011689796389439, 1.200280532951476e-05),
            "velocity": (0.4170728907949309, -0.18351545167455108, 0.02598010535939589),
            "sph_acceleration": (37.2061738581171, -16.704992124629435, -1.7072951100430274),
            "ext_acceleration": (444.56416301139825, -206.4655412692829, 15.72075772382885),
            "mass": 0.523,
            "angular_momentum": (
                -4.752029366157232e-05,
                0.00549233813270157,
                -0.0028782596789702175,
            ),
            "accretion_radius": 0.15,
        },
        {
            "pos": (-0.0014005543135792629, -0.15082555116219917, 0.05026364054807148),
            "velocity": (0.29183513471029215, 0.6043203640193001, -0.36936631799129205),
            "sph_acceleration": (2.0715680965753904, 17.62344096570381, -6.02150962224265),
            "ext_acceleration": (487.85414809564486, 1058.0686991369491, -332.3305047371264),
            "mass": 0.269,
            "angular_momentum": (-0.0008540268554936789, -0.00886551897978765, 0.00320269934509115),
            "accretion_radius": 0.15,
        },
    ]

    errors = []

    if len(sinks) != len(ref_dataset):
        errors.append(f"sink count mismatch: got {len(sinks)}, expected {len(ref_dataset)}")
    else:
        for i, (got_sink, ref_sink) in enumerate(zip(sinks, ref_dataset)):
            for key, ref_val in ref_sink.items():
                got_val = got_sink[key]
                got_arr = np.asarray(got_val, dtype=float)
                ref_arr = np.asarray(ref_val, dtype=float)
                if not np.all(np.isclose(got_arr, ref_arr, rtol=1e-8)):
                    abs_diff = np.abs(got_arr - ref_arr)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rel_diff = np.where(ref_arr != 0, abs_diff / np.abs(ref_arr), abs_diff)
                    errors.append(
                        f"sink[{i}].{key} mismatch:\n"
                        f"  got={got_val}\n"
                        f"  ref={ref_val}\n"
                        f"  max abs diff={np.max(abs_diff)}\n"
                        f"  max rel diff={np.max(rel_diff)}"
                    )

    for err in errors:
        print(err)

    if errors:
        raise RuntimeError(f"check_ref_dataset failed with {len(errors)} error(s)")

    if shamrock.sys.world_rank() == 0:
        print("check_ref_dataset: OK")


def main():
    ctx, model = build_model_with_sinks()

    check_sinks_are_in_sync(ctx, model)

    for _ in range(5):
        model.timestep()
    check_sinks_are_in_sync(ctx, model)

    sinks_before_dump = str(model.get_sinks())
    model.dump(DUMP_NAME)

    del model
    del ctx

    ctx2 = shamrock.Context()
    ctx2.pdata_layout_new()
    model2 = shamrock.get_Model_SPH(context=ctx2, vector_type="f64_3", sph_kernel="M4")
    model2.load_from_dump(DUMP_NAME)

    sinks_after_reload = str(model2.get_sinks())
    if sinks_before_dump != sinks_after_reload:
        raise RuntimeError(
            "sink content changed across dump/reload:\n"
            f"  before={sinks_before_dump!r}\n"
            f"  after ={sinks_after_reload!r}"
        )

    check_sinks_are_in_sync(ctx2, model2)

    for _ in range(5):
        model2.timestep()
    check_sinks_are_in_sync(ctx2, model2)

    if shamrock.sys.world_rank() == 0:
        print("run_test_sink_synchro: OK")

    check_ref_dataset(model2.get_sinks())


main()
