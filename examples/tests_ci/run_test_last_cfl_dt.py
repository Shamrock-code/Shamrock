"""
CI test: last CFL dt stored in synchronized edges
=================================================

Checks that the SPH solver stores the last CFL timestep in the synchronized
edges and that ``model.get_last_cfl_dts()`` exposes it:

- without CFL detail, only the global ``"cfl_dt"`` edge exists,
- with CFL detail, one ``"cfl_dt_<type>"`` edge per enabled CFL condition
  exists as well, and ``"cfl_dt"`` is their minimum,
- the values are identical on every rank and survive a dump/reload.
"""

import shamrock

DUMP_NAME = "last_cfl_dt_test.sham"


def build_model(show_cfl_detail):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_default_config()
    cfg.set_self_gravity_none()
    cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
    cfg.set_eos_isothermal(1.0)
    cfg.set_particle_mass(1e-3)
    cfg.set_boundary_periodic()
    cfg.set_show_cfl_detail(show_cfl_detail)
    model.set_solver_config(cfg)

    model.init_scheduler(1000, 1)

    dr = 0.1
    bmin = (-0.6, -0.6, -0.6)
    bmax = (0.6, 0.6, 0.6)
    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen)

    model.set_field_value_lambda_f64_3("vxyz", lambda r: (1.0, 0.0, 0.0))

    model.add_sink(1.0, (0.1, 0.0, 0.0), (0.0, 0.05, 0.0), 0.05)
    model.add_sink(0.5, (-0.2, 0.1, 0.0), (0.0, -0.03, 0.0), 0.05)

    return ctx, model


def check_in_sync(cfl):
    s = str(sorted(cfl.items()))
    # Collective: every rank must call this
    hist = shamrock.algs.all_string_histogram([s], delimiter="\n", hash_based=False)
    if len(hist) != 1 or next(iter(hist.values())) != shamrock.sys.world_size():
        raise RuntimeError(f"last cfl dt not in sync across ranks: {hist}")


def check_detail_off():
    ctx, model = build_model(show_cfl_detail=False)

    cfl = model.get_last_cfl_dts()
    if cfl != {"cfl_dt": 0.0}:
        raise RuntimeError(f"expected only 'cfl_dt' = 0 before the first step, got {cfl}")

    model.timestep()  # first step only computes the initial dt
    model.timestep()

    cfl = model.get_last_cfl_dts()
    if shamrock.sys.world_rank() == 0:
        print("last cfl dt (detail off):", cfl)

    if set(cfl.keys()) != {"cfl_dt"}:
        raise RuntimeError(f"expected only 'cfl_dt' without cfl detail, got {cfl}")
    if cfl["cfl_dt"] != model.get_dt():
        raise RuntimeError(f"cfl_dt={cfl['cfl_dt']} != next dt={model.get_dt()}")

    check_in_sync(cfl)


def check_detail_on():
    ctx, model = build_model(show_cfl_detail=True)

    expected = {"cfl_dt", "cfl_dt_courant", "cfl_dt_force", "cfl_dt_sink_sink"}

    cfl = model.get_last_cfl_dts()
    if cfl != {k: 0.0 for k in expected}:
        raise RuntimeError(f"expected {sorted(expected)} = 0 before the first step, got {cfl}")

    model.timestep()
    model.timestep()

    cfl = model.get_last_cfl_dts()
    if shamrock.sys.world_rank() == 0:
        print("last cfl dt (detail on):", cfl)

    if set(cfl.keys()) != expected:
        raise RuntimeError(f"expected keys {sorted(expected)}, got {sorted(cfl.keys())}")

    detail_min = min(v for k, v in cfl.items() if k != "cfl_dt")
    if cfl["cfl_dt"] != detail_min:
        raise RuntimeError(f"cfl_dt={cfl['cfl_dt']} != min of per-type cfl dt={detail_min}")
    if cfl["cfl_dt"] != model.get_dt():
        raise RuntimeError(f"cfl_dt={cfl['cfl_dt']} != next dt={model.get_dt()}")

    check_in_sync(cfl)

    model.dump(DUMP_NAME)

    del model
    del ctx

    ctx2 = shamrock.Context()
    ctx2.pdata_layout_new()
    model2 = shamrock.get_Model_SPH(context=ctx2, vector_type="f64_3", sph_kernel="M4")
    model2.load_from_dump(DUMP_NAME)

    cfl_reload = model2.get_last_cfl_dts()
    if cfl_reload != cfl:
        raise RuntimeError(
            f"last cfl dt changed across dump/reload:\n  before={cfl}\n  after ={cfl_reload}"
        )

    check_in_sync(cfl_reload)


check_detail_off()
check_detail_on()

if shamrock.sys.world_rank() == 0:
    print("last cfl dt test: OK")
