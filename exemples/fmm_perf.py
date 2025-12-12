import matplotlib.pyplot as plt
import numpy as np

import shamrock

# Particle tracking is an experimental feature

shamrock.enable_experimental_features()

shamrock.tree.set_impl_clbvh_dual_tree_traversal("parallel_select", "")


def run_benchmark(ctx, model):

    res_rates = []
    res_cnts = []

    model.timestep()

    data = ctx.collect_data()

    for i in range(5):
        model.timestep()

        tmp_res_rate, tmp_res_cnt = (
            model.solver_logs_last_rate(),
            model.solver_logs_last_obj_count(),
        )
        res_rates.append(tmp_res_rate)
        res_cnts.append(tmp_res_cnt)

    res_rate, res_cnt = max(res_rates), res_cnts[0]

    return data, res_rate, res_cnt


def cub_collapse(config_func, Npart):

    si = shamrock.UnitSystem()
    sicte = shamrock.Constants(si)
    codeu = shamrock.UnitSystem(
        unit_time=sicte.year(),
        unit_length=sicte.au(),
        unit_mass=sicte.sol_mass(),
    )
    ucte = shamrock.Constants(codeu)

    gamma = 5.0 / 3.0
    rho_g = 100
    initial_u = 10

    sphere_radius = 0.1
    sim_radius = 0.5

    bmin = (-sim_radius, -sim_radius, -sim_radius)
    bmax = (sim_radius, sim_radius, sim_radius)

    init_part_bmin = (-sphere_radius, -sphere_radius, -sphere_radius)
    init_part_bmax = (sphere_radius, sphere_radius, sphere_radius)

    scheduler_split_val = int(2e7)
    scheduler_merge_val = int(1)

    N_target = Npart
    xm, ym, zm = init_part_bmin
    xM, yM, zM = init_part_bmax
    vol_b = (xM - xm) * (yM - ym) * (zM - zm)

    if shamrock.sys.world_rank() == 0:
        print("Npart", Npart)
        print("scheduler_split_val", scheduler_split_val)
        print("scheduler_merge_val", scheduler_merge_val)
        print("N_target", N_target)
        print("vol_b", vol_b)

    part_vol = vol_b / N_target

    # lattice volume
    part_vol_lattice = 0.74 * part_vol

    dr = (part_vol_lattice / ((4.0 / 3.0) * 3.1416)) ** (1.0 / 3.0)

    pmass = -1

    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_default_config()
    # cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
    # cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )
    cfg.set_boundary_periodic()
    cfg.set_eos_adiabatic(gamma)

    config_func(cfg)

    cfg.set_units(codeu)
    cfg.print_status()
    model.set_solver_config(cfg)
    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, init_part_bmin, init_part_bmax)

    # On aurora /2 was correct to avoid out of memory
    setup.apply_setup(gen, insert_step=int(scheduler_split_val / 2))

    vol_b = (xM - xm) * (yM - ym) * (zM - zm)
    totmass = rho_g * vol_b
    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)

    model.set_value_in_a_box("uint", "f64", initial_u, bmin, bmax)

    model.set_cfl_cour(1e-6)
    model.set_cfl_force(1e-6)

    return run_benchmark(ctx, model)


def add_data_to_collect(collected_data, none_case, ref_case=None):

    # substract the none case from the collected data
    collected_data["axyz"] = collected_data["axyz"] - none_case["axyz"]

    collected_data["r"] = np.sqrt(
        collected_data["xyz"][:, 0] ** 2
        + collected_data["xyz"][:, 1] ** 2
        + collected_data["xyz"][:, 2] ** 2
    )
    collected_data["ar"] = np.sqrt(
        collected_data["axyz"][:, 0] ** 2
        + collected_data["axyz"][:, 1] ** 2
        + collected_data["axyz"][:, 2] ** 2
    )

    if ref_case is not None:
        collected_data["axyz_delta"] = collected_data["axyz"] - ref_case["axyz"]

        tmp = np.linalg.norm(collected_data["axyz_delta"], axis=1)
        tmp = tmp / np.linalg.norm(ref_case["axyz"], axis=1)
        collected_data["rel_error"] = tmp

    return collected_data


def none_config(cfg):
    cfg.set_self_gravity_none()
    return cfg


def direct_config(cfg):
    cfg.set_self_gravity_direct()
    return cfg


def direct_safe_config(cfg):
    cfg.set_self_gravity_direct(reference_mode=True)
    return cfg


def mm1_config(cfg):
    cfg.set_self_gravity_mm(order=1, opening_angle=0.5)
    return cfg


def mm2_config(cfg):
    cfg.set_self_gravity_mm(order=2, opening_angle=0.5)
    return cfg


def mm3_config(cfg):
    cfg.set_self_gravity_mm(order=3, opening_angle=0.5)
    return cfg


def mm4_config(cfg):
    cfg.set_self_gravity_mm(order=4, opening_angle=0.5)
    return cfg


def mm5_config(cfg):
    cfg.set_self_gravity_mm(order=5, opening_angle=0.5)
    return cfg


def fmm1_config(cfg):
    cfg.set_self_gravity_fmm(order=1, opening_angle=0.5)
    return cfg


def fmm2_config(cfg):
    cfg.set_self_gravity_fmm(order=2, opening_angle=0.5)
    return cfg


def fmm3_config(cfg):
    cfg.set_self_gravity_fmm(order=3, opening_angle=0.5)
    return cfg


def fmm4_config(cfg):
    cfg.set_self_gravity_fmm(order=4, opening_angle=0.5)
    return cfg


def fmm5_config(cfg):
    cfg.set_self_gravity_fmm(order=5, opening_angle=0.5)
    return cfg


def sfmm1_config(cfg):
    cfg.set_self_gravity_sfmm(order=1, opening_angle=0.5)
    return cfg


def sfmm2_config(cfg):
    cfg.set_self_gravity_sfmm(order=2, opening_angle=0.5)
    return cfg


def sfmm3_config(cfg):
    cfg.set_self_gravity_sfmm(order=3, opening_angle=0.5)
    return cfg


def sfmm4_config(cfg):
    cfg.set_self_gravity_sfmm(order=4, opening_angle=0.5)
    return cfg


def sfmm5_config(cfg):
    cfg.set_self_gravity_sfmm(order=5, opening_angle=0.5)
    return cfg


configs = {
    "none": none_config,
    "direct_safe": direct_safe_config,
    "direct": direct_config,
    "mm4": mm4_config,
    "fmm4": fmm4_config,
    "sfmm4": sfmm4_config,
}

no_sg_config = "none"
reference_sg_config = "direct_safe"

Nparts = np.logspace(3, 6, 10).astype(int).tolist()

perf_curves = {}
for config in configs.keys():
    if config in ["none"]:
        continue

    perf_curves[config] = {"Npart": [], "grav_rate": [], "grav_hydro_cost": []}

for Npart in Nparts:

    results = {}
    for config in configs.keys():

        if config in ["direct", "direct_safe"] and Npart > 2e4:
            continue

        results[config] = cub_collapse(configs[config], Npart)

    # TODO compute error relative to reference_sg_config

    _, no_sg_rate, no_sg_cnt = results[no_sg_config]
    no_sg_tstep = no_sg_cnt / no_sg_rate

    for config in results.keys():
        if config in ["none"]:
            continue

        _, rate, cnt = results[config]

        tstep = cnt / rate

        grav_tstep = tstep - no_sg_tstep
        grav_rate = cnt / grav_tstep

        ratio = no_sg_rate / grav_rate

        perf_curves[config]["Npart"].append(cnt)
        perf_curves[config]["grav_rate"].append(grav_rate)
        perf_curves[config]["grav_hydro_cost"].append(ratio)

plt.figure()
plt.title("Gravitational rate")
for config in perf_curves.keys():
    plt.plot(perf_curves[config]["Npart"], perf_curves[config]["grav_rate"], label=config)

plt.yscale("log")
plt.xscale("log")
plt.xlabel("Npart")
plt.ylabel("Gravitational rate [part.s-1]")

plt.legend()
plt.savefig("fmm_perf1.pdf")


plt.figure()
plt.title("Time cost relative to pure hydro")
for config in perf_curves.keys():
    plt.plot(perf_curves[config]["Npart"], perf_curves[config]["grav_hydro_cost"], label=config)

plt.xscale("log")
plt.xlabel("Npart")
plt.ylabel("Hydro rate / Gravitational rate")

plt.legend()
plt.show()

plt.savefig("fmm_perf2.pdf")

plt.show()
