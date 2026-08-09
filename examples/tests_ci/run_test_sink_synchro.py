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
        return (10.0, 0.0, 0.0)

    model.set_field_value_lambda_f64_3("vxyz", vel_func)

    # A few sinks (must be added after init_scheduler, on all ranks)
    model.add_sink(1.0, (0.1, 0.0, 0.0), (0.0, 0.05, 0.0), 0.15)
    model.add_sink(0.5, (-0.2, 0.1, 0.0), (0.0, -0.03, 0.0), 0.15)
    model.add_sink(0.25, (0.0, -0.15, 0.05), (0.02, 0.0, 0.0), 0.15)

    return ctx, model


def check_ref_dataset(sinks):
    if shamrock.sys.world_rank() == 0:
        print("Current sinks:")
        print(sinks)

    ref_dataset = [
        {
            "pos": (0.09877660057755726, -0.0003699820152028823, 0.00029285849440530886),
            "velocity": (-0.7909594697171893, -0.42835509420057705, 0.2157916322019174),
            "sph_acceleration": (-2.7881186060647103, 6.194080856496935, -3.386302543973038),
            "ext_acceleration": (-371.6829482929875, -190.87247862152324, 85.76563352228864),
            "mass": 1.0219999999999998,
            "angular_momentum": (
                1.5355132242207996e-05,
                -3.404279116936264e-05,
                -0.001391721046508032,
            ),
            "accretion_radius": 0.15,
        },
        {
            "pos": (-0.19723897477916802, 0.0992597386368148, 5.457513126197254e-05),
            "velocity": (1.7840284522444, -0.6462230101076988, 0.039967403747977054),
            "sph_acceleration": (40.132674771440605, -16.361891265380414, -1.2376352720798245),
            "ext_acceleration": (456.6529453658143, -211.77602552231514, 16.18781052807078),
            "mass": 0.523,
            "angular_momentum": (
                1.4588956579124888e-08,
                -1.6035141770323982e-07,
                -0.0009268830439052644,
            ),
            "accretion_radius": 0.15,
        },
        {
            "pos": (0.0019199655817502001, -0.1466495904953547, 0.048938432104203365),
            "velocity": (2.1365106397204268, 3.0220342305086376, -0.9498166179408467),
            "sph_acceleration": (4.433172833108257, 16.09220042839757, -6.362794011098254),
            "ext_acceleration": (522.3351212189343, 1132.705683330991, -355.99519394799995),
            "mass": 0.27,
            "angular_momentum": (
                4.4469346449927724e-05,
                0.0006693090866850239,
                0.0020383710492193246,
            ),
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
                rtol = 1e-14 if key == "sph_acceleration" else 1e-15
                if not np.all(np.isclose(got_arr, ref_arr, rtol=rtol, atol=1e-18)):
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
