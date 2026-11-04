"""
CI test: fallback stream injection (SPH and GSPH)
=================================================

Injects a gas stream following a tidal disruption fallback rate, with SPH (into
a periodic box of background gas) and with GSPH (into vacuum), and checks that:

- the injected mass follows the integral of the fallback rate, within one wall;
- the injected particles have unique ids, and the boundary particles sit
  upstream of the injection point, inside the stream cross-section;
- the injection is deterministic (two identical runs give bitwise identical
  particles) and survives a dump/reload (the injection state is replayed from
  the time of the dump);
- with several MPI ranks, the SPH stream is inserted into patches of more than
  one rank: the injection point sits at a corner of the patches, so the born
  walls and the boundary walls fall on both sides of the rank boundary.
"""

import numpy as np

import shamrock

shamrock.enable_experimental_features()

# allow running the script with python directly, not only through shamrock --rscript
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

DUMP_NAME = "fallback_stream_injection_test.sham"

gamma = 5.0 / 3.0
resol = 16
rho_bg = 1.0
u_bg = 0.05
pmass_vacuum = 1.7e-4

# the SPH background is cleared around the injection point: injecting on top of existing gas
# would bury its particles in the dense boundary walls, where the smoothing length iteration
# diverges
channel_end = 0.15
channel_radius = 0.1

# fallback rate mdot_peak * ((t - t_disrupt) / t_mb)^mdot_index, starting at t_disrupt + t_mb = 0
# the injection point is the corner shared by the 8 patches of the box, so that the stream is
# split between patches (and ranks): born walls downstream (x > 0), boundary walls upstream
inject_pos = (0.0, 0.0, 0.0)
inject_vel = (2.0, 0.0, 0.0)
stream_radius = 0.05
u_inject = 1e-2
mdot_peak = 0.5
t_mb = 0.1
t_disrupt = -0.1
mdot_index = -5.0 / 3.0

# default number of boundary walls for the M4 kernel: ceil(Rkern * hfact) + 1
n_boundary_walls = 4

t_mid = 0.05
t_end = 0.1

INJECTED_BIT = np.uint64(1) << np.uint64(62)
BOUNDARY_BIT = np.uint64(1) << np.uint64(61)


def mdot(t):
    """Fallback rate at time t"""
    return mdot_peak * ((t - t_disrupt) / t_mb) ** mdot_index


def injected_mass(t):
    """Mass supplied by the fallback rate between its start and t"""
    tau = (t - t_disrupt) / t_mb
    return mdot_peak * t_mb * (tau ** (mdot_index + 1) - 1) / (mdot_index + 1)


def get_model(solver, ctx):
    if solver == "SPH":
        return shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
    return shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")


def build_model(solver):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = get_model(solver, ctx)

    cfg = model.gen_default_config()
    if solver == "SPH":
        cfg.set_artif_viscosity_VaryingCD10(
            alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
        )
        cfg.set_particle_tracking(True)
    else:
        cfg.set_riemann_hllc()
        cfg.set_reconstruct_piecewise_constant()
    cfg.set_boundary_periodic()
    cfg.set_eos_adiabatic(gamma)
    cfg.add_injection_fallback_stream(
        inject_pos=inject_pos,
        inject_vel=inject_vel,
        stream_radius=stream_radius,
        u_inject=u_inject,
        mdot_peak=mdot_peak,
        t_mb=t_mb,
        t_disrupt=t_disrupt,
        mdot_index=mdot_index,
    )
    model.set_solver_config(cfg)

    # several patches, whose boundaries cross the stream axis (y = z = 0)
    model.init_scheduler(1000, 1)

    (xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, resol, resol)
    dr = 1 / xs
    (xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, resol, resol)
    bmin = (-xs / 2, -ys / 2, -zs / 2)
    bmax = (xs / 2, ys / 2, zs / 2)
    model.resize_simulation_box(bmin, bmax)

    if solver == "SPH":

        def outside_channel(r):
            x, y, z = r
            return x > channel_end or y * y + z * z > channel_radius**2

        setup = model.get_setup()
        gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
        background = setup.make_modifier_filter(parent=gen, filter=outside_channel)
        setup.apply_setup(background)

        model.set_value_in_a_box("uint", "f64", u_bg, bmin, bmax)

        channel_volume = np.pi * channel_radius**2 * (channel_end - bmin[0])
        model.set_particle_mass(
            model.total_mass_to_part_mass(rho_bg * (xs * ys * zs - channel_volume))
        )
    else:
        # the stream is injected into vacuum
        model.set_particle_mass(pmass_vacuum)

    return ctx, model


def sorted_by_id(data):
    order = np.argsort(data["part_id"], kind="stable")
    return {key: np.asarray(val)[order] for key, val in data.items()}


def check_identical(data_a, data_b, label):
    a = sorted_by_id(data_a)
    b = sorted_by_id(data_b)
    if a.keys() != b.keys():
        raise RuntimeError(f"{label}: different fields {a.keys()} vs {b.keys()}")
    for key in a:
        if a[key].tobytes() != b[key].tobytes():
            raise RuntimeError(f"{label}: field {key} differs")
    print(f"{label}: bitwise identical ({len(a['part_id'])} particles)")


def check_injection(data, model, t):
    part_id = np.asarray(data["part_id"], dtype=np.uint64)
    xyz = np.asarray(data["xyz"])

    is_boundary = (part_id & BOUNDARY_BIT) != 0
    is_live = ((part_id & INJECTED_BIT) != 0) & ~is_boundary

    n_live = int(np.sum(is_live))
    n_boundary = int(np.sum(is_boundary))
    print(f"t = {t}: {n_live} injected particles, {n_boundary} boundary particles")

    injected_ids = part_id[is_live | is_boundary]
    assert np.unique(injected_ids).size == injected_ids.size, "duplicated particle ids"
    assert n_live > 0, "no particle was injected"
    assert n_boundary > 0, "no boundary particle"

    # the born walls hold the mass supplied by the rate, minus the wall still accumulating
    pmass = model.get_particle_mass()
    mass_deficit = injected_mass(t) - n_live * pmass
    wall_mass = n_boundary / n_boundary_walls * pmass
    print(f"injected mass deficit = {mass_deficit}, wall mass ~ {wall_mass}")
    assert mass_deficit >= -1e-12 * injected_mass(t), "more mass injected than supplied"
    assert mass_deficit < 1.5 * wall_mass, "the injected mass lags behind the rate"

    # the boundary walls wait upstream of the injection point, inside the stream, within the
    # length that the stream covers while the rate supplies the boundary walls
    bnd = xyz[is_boundary]
    wall_spacing = inject_vel[0] * wall_mass / mdot(t)
    upstream_dist = inject_pos[0] - bnd[:, 0]
    print(f"boundary walls upstream distance in [{upstream_dist.min()}, {upstream_dist.max()}]")
    assert np.all(upstream_dist > 0), "boundary particle downstream of the injection point"
    assert np.all(upstream_dist < (n_boundary_walls + 1) * wall_spacing), (
        "boundary particle too far upstream"
    )
    assert np.all(np.hypot(bnd[:, 1], bnd[:, 2]) < stream_radius), (
        "boundary particle outside of the stream"
    )


def check_stream_ranks(ctx, model, data):
    """Count the injected particles in the patches of each rank, and require several ranks"""
    transform = model.get_patch_transform()
    patches = [(p.node_owner_id, transform.to_obj_coord(p)) for p in ctx.get_patch_list_global()]

    if shamrock.sys.world_rank() > 0:
        return

    part_id = np.asarray(data["part_id"], dtype=np.uint64)
    xyz = np.asarray(data["xyz"])[(part_id & (INJECTED_BIT | BOUNDARY_BIT)) != 0]

    count_per_rank = {}
    for owner, box in patches:
        inside = np.all((xyz >= np.array(box.lower)) & (xyz < np.array(box.upper)), axis=1)
        count_per_rank[owner] = count_per_rank.get(owner, 0) + int(np.sum(inside))
    print(f"injected particles per rank: {count_per_rank}")

    ranks_with_stream = [rank for rank, count in count_per_rank.items() if count > 0]
    assert len(ranks_with_stream) > 1, (
        "the stream lies on a single rank, the multi-rank insertion is not tested"
    )


def run(solver, dump_at_mid):
    ctx, model = build_model(solver)

    model.evolve_until(t_mid)

    if dump_at_mid:
        model.dump(DUMP_NAME)
        del model
        del ctx

        ctx = shamrock.Context()
        ctx.pdata_layout_new()
        model = get_model(solver, ctx)
        model.load_from_dump(DUMP_NAME)

    model.evolve_until(t_end)

    return ctx, model, ctx.collect_data()


def test_solver(solver):
    ctx_ref, model_ref, data_ref = run(solver, dump_at_mid=False)
    _, _, data_again = run(solver, dump_at_mid=False)
    _, _, data_restart = run(solver, dump_at_mid=True)

    # the GSPH stream goes into vacuum and stays in a single patch
    if solver == "SPH" and shamrock.sys.world_size() > 1:
        check_stream_ranks(ctx_ref, model_ref, data_ref)

    if shamrock.sys.world_rank() > 0:
        return

    print(f"----- {solver} -----")
    check_injection(data_ref, model_ref, t_end)
    check_identical(data_ref, data_again, f"{solver} rerun")
    check_identical(data_ref, data_restart, f"{solver} dump/reload")


test_solver("SPH")
test_solver("GSPH")

if shamrock.sys.world_rank() == 0:
    print("run_fallback_stream_injection: OK")
