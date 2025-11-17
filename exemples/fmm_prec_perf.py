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


def cub_collapse(theta_crit, config_mode, Npart):

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

    if config_mode == "none":
        cfg.set_self_gravity_none()
    elif config_mode == "direct":
        cfg.set_self_gravity_direct()
    elif config_mode == "direct_safe":
        cfg.set_self_gravity_direct(reference_mode=True)
    elif config_mode == "mm1":
        cfg.set_self_gravity_mm(order=1, opening_angle=theta_crit)
    elif config_mode == "mm2":
        cfg.set_self_gravity_mm(order=2, opening_angle=theta_crit)
    elif config_mode == "mm3":
        cfg.set_self_gravity_mm(order=3, opening_angle=theta_crit)
    elif config_mode == "mm4":
        cfg.set_self_gravity_mm(order=4, opening_angle=theta_crit)
    elif config_mode == "mm5":
        cfg.set_self_gravity_mm(order=5, opening_angle=theta_crit)
    elif config_mode == "fmm1":
        cfg.set_self_gravity_fmm(order=1, opening_angle=theta_crit)
    elif config_mode == "fmm2":
        cfg.set_self_gravity_fmm(order=2, opening_angle=theta_crit)
    elif config_mode == "fmm3":
        cfg.set_self_gravity_fmm(order=3, opening_angle=theta_crit)
    elif config_mode == "fmm4":
        cfg.set_self_gravity_fmm(order=4, opening_angle=theta_crit)
    elif config_mode == "fmm5":
        cfg.set_self_gravity_fmm(order=5, opening_angle=theta_crit)
    elif "sfmm1" in config_mode:
        cfg.set_self_gravity_sfmm(
            order=1, opening_angle=theta_crit, leaf_lowering="lowering" in config_mode
        )
    elif "sfmm2" in config_mode:
        cfg.set_self_gravity_sfmm(
            order=2, opening_angle=theta_crit, leaf_lowering="lowering" in config_mode
        )
    elif "sfmm3" in config_mode:
        cfg.set_self_gravity_sfmm(
            order=3, opening_angle=theta_crit, leaf_lowering="lowering" in config_mode
        )
    elif "sfmm4" in config_mode:
        cfg.set_self_gravity_sfmm(
            order=4, opening_angle=theta_crit, leaf_lowering="lowering" in config_mode
        )
    elif "sfmm5" in config_mode:
        cfg.set_self_gravity_sfmm(
            order=5, opening_angle=theta_crit, leaf_lowering="lowering" in config_mode
        )
    else:
        raise ValueError(f"Invalid case name: {config_mode}")

    # otherwise the MM precision at very low angle does
    # not match the direct mode
    cfg.set_softening_plummer(epsilon=1e-15)

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

        # if(np.mean(tmp) > 1e-3):
        #    print(f"failed rel_error = {tmp}")
        #    plt.figure(figsize=(10, 5))
        #    plt.scatter(collected_data["r"], collected_data["rel_error"], s=1)
        #    plt.legend()
        #    plt.show()

    return collected_data


base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
extended_colors = base_colors * 3
color_cycle = extended_colors


def get_linestyle(key):
    if "lowering" in key:
        return "dashdot"
    elif "sfmm" in key:
        return "dotted"
    elif "fmm" in key:
        return "dashed"
    elif "mm" in key:
        return "solid"
    else:
        return "solid"


Nparts = np.logspace(3, 5, 3).astype(int).tolist()
# Nparts = [1000, 10000]
configs = [
    "mm1",
    # "mm2",
    # "mm3",
    # "mm4",
    "mm5",
    "fmm1",
    # "fmm2",
    # "fmm3",
    # "fmm4",
    "fmm5",
    "sfmm1",
    # "sfmm2",
    # "sfmm3",
    # "sfmm4",
    "sfmm5",
    "lowering_sfmm1",
    # "lowering_sfmm2",
    # "lowering_sfmm3",
    # "lowering_sfmm4",
    "lowering_sfmm5",
]
# configs = ["mm5", "fmm5", "sfmm5"]
theta_vals = [
    1e-3,
    1e-2,
    1e-1,
    # 0.1,
    # 0.12915497,
    # 0.16681005,
    # 0.21544347,
    # 0.27825594,
    # 0.35938137,
    # 0.46415888,
    0.5,
    # 0.59948425,
    # 0.77426368,
    1.0,
]

no_sg_config = "none"
reference_sg_config = "direct_safe"

curve_precisions = {}
curve_rates = {}

for npart in Nparts:

    prec_npart = {}

    data_none, none_rate, none_cnt = cub_collapse(0, "none", npart)
    data_direct_safe, direct_safe_rate, direct_safe_cnt = cub_collapse(0, "direct_safe", npart)
    data_direct_safe = add_data_to_collect(data_direct_safe, data_none)

    data_direct, direct_rate, direct_cnt = cub_collapse(0, "direct", npart)

    key_rate_none = ("none", get_linestyle("none"), color_cycle[0])
    key_rate_direct_safe = ("direct_safe", get_linestyle("direct_safe"), color_cycle[1])
    key_rate_direct = ("direct", get_linestyle("direct"), color_cycle[2])

    if not key_rate_none in curve_rates:
        curve_rates[key_rate_none] = []
    if not key_rate_direct_safe in curve_rates:
        curve_rates[key_rate_direct_safe] = []
    if not key_rate_direct in curve_rates:
        curve_rates[key_rate_direct] = []

    curve_rates[key_rate_none].append(none_rate)
    curve_rates[key_rate_direct_safe].append(direct_safe_rate)
    curve_rates[key_rate_direct].append(direct_rate)

    data_direct = add_data_to_collect(data_direct, data_none, data_direct_safe)

    avg_rel_error_direct = np.mean(data_direct["rel_error"])
    print(f"avg_rel_error_direct = {avg_rel_error_direct}")

    # plt.figure(figsize=(10, 5))
    # plt.scatter(data_direct["r"], data_direct["ar"] - data_direct_safe["ar"], s=1, label="direct")
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.scatter(data_direct["r"], data_direct["rel_error"], s=1, label="direct")
    # plt.legend()
    # plt.show()

    for itheta, theta in enumerate(theta_vals):

        key_prec_direct = ("direct", get_linestyle("direct"), color_cycle[2])
        if not key_prec_direct in prec_npart:
            prec_npart[key_prec_direct] = []
        prec_npart[key_prec_direct].append(avg_rel_error_direct)

        for config in configs:
            data, rate, cnt = cub_collapse(theta, config, npart)

            data = add_data_to_collect(data, data_none, data_direct_safe)

            avg_rel_error = np.max(data["rel_error"])

            if theta == 0.5:
                key = (f"{config}", get_linestyle(config), color_cycle[int(config[-1]) + 2])
                if not key in curve_rates:
                    curve_rates[key] = []
                curve_rates[key].append(rate)

            key = (f"{config}", get_linestyle(config), color_cycle[int(config[-1]) + 2])
            if not key in prec_npart:
                prec_npart[key] = []
            prec_npart[key].append(avg_rel_error)

            print(f"avg_rel_error = {avg_rel_error}")

            # plt.figure(figsize=(10, 5))
            # plt.scatter(data["r"], data["rel_error"], s=1, label=key)
            # plt.legend()
            # plt.show()

    curve_precisions[npart] = prec_npart


plt.figure(figsize=(10, 5))
for key in curve_rates.keys():
    (label, linestyle, color) = key
    plt.plot(Nparts, curve_rates[key], label=label, linestyle=linestyle, color=color)
plt.legend()
plt.xlabel("Npart")
plt.ylabel("rate")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()

for npart in Nparts:
    plt.figure(figsize=(10, 5))
    plt.title(f"Npart = {npart}")
    for key in curve_precisions[npart].keys():
        (label, linestyle, color) = key
        plt.plot(
            theta_vals, curve_precisions[npart][key], label=label, linestyle=linestyle, color=color
        )
    plt.xlabel("theta")
    plt.ylabel("precision")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
plt.show()
