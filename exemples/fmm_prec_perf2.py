import os
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import shamrock

# Particle tracking is an experimental feature

shamrock.enable_experimental_features()

shamrock.tree.set_impl_clbvh_dual_tree_traversal("parallel_select", "")


def save_bench_result(filename, data, res_rate, res_cnt):
    # save as pickle
    import pickle

    with open(filename, "wb") as f:
        pickle.dump({"data": data, "res_rate": res_rate, "res_cnt": res_cnt}, f)


def load_bench_result(filename):
    # load from pickle
    import pickle

    with open(filename, "rb") as f:
        tmp = pickle.load(f)
        return tmp["data"], tmp["res_rate"], tmp["res_cnt"]


def config_to_filename(theta_crit, config_mode, Npart, reduction_level):
    if config_mode in ["none", "direct", "direct_safe"]:
        return f"/tmp/bench_result_{config_mode}_{Npart}.pkl"

    return f"/tmp/bench_result_{theta_crit:.3f}_{config_mode}_{Npart}_{reduction_level}.pkl"


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


def cub_collapse(theta_crit, config_mode, Npart, reduction_level):

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
        cfg.set_self_gravity_mm(order=1, opening_angle=theta_crit, reduction_level=reduction_level)
    elif config_mode == "mm2":
        cfg.set_self_gravity_mm(order=2, opening_angle=theta_crit, reduction_level=reduction_level)
    elif config_mode == "mm3":
        cfg.set_self_gravity_mm(order=3, opening_angle=theta_crit, reduction_level=reduction_level)
    elif config_mode == "mm4":
        cfg.set_self_gravity_mm(order=4, opening_angle=theta_crit, reduction_level=reduction_level)
    elif config_mode == "mm5":
        cfg.set_self_gravity_mm(order=5, opening_angle=theta_crit, reduction_level=reduction_level)
    elif config_mode == "fmm1":
        cfg.set_self_gravity_fmm(order=1, opening_angle=theta_crit, reduction_level=reduction_level)
    elif config_mode == "fmm2":
        cfg.set_self_gravity_fmm(order=2, opening_angle=theta_crit, reduction_level=reduction_level)
    elif config_mode == "fmm3":
        cfg.set_self_gravity_fmm(order=3, opening_angle=theta_crit, reduction_level=reduction_level)
    elif config_mode == "fmm4":
        cfg.set_self_gravity_fmm(order=4, opening_angle=theta_crit, reduction_level=reduction_level)
    elif config_mode == "fmm5":
        cfg.set_self_gravity_fmm(order=5, opening_angle=theta_crit, reduction_level=reduction_level)
    elif "sfmm1" in config_mode:
        cfg.set_self_gravity_sfmm(
            order=1,
            opening_angle=theta_crit,
            leaf_lowering="lowering" in config_mode,
            reduction_level=reduction_level,
        )
    elif "sfmm2" in config_mode:
        cfg.set_self_gravity_sfmm(
            order=2,
            opening_angle=theta_crit,
            leaf_lowering="lowering" in config_mode,
            reduction_level=reduction_level,
        )
    elif "sfmm3" in config_mode:
        cfg.set_self_gravity_sfmm(
            order=3,
            opening_angle=theta_crit,
            leaf_lowering="lowering" in config_mode,
            reduction_level=reduction_level,
        )
    elif "sfmm4" in config_mode:
        cfg.set_self_gravity_sfmm(
            order=4,
            opening_angle=theta_crit,
            leaf_lowering="lowering" in config_mode,
            reduction_level=reduction_level,
        )
    elif "sfmm5" in config_mode:
        cfg.set_self_gravity_sfmm(
            order=5,
            opening_angle=theta_crit,
            leaf_lowering="lowering" in config_mode,
            reduction_level=reduction_level,
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

    data, res_rate, res_cnt = run_benchmark(ctx, model)

    save_bench_result(
        config_to_filename(theta_crit, config_mode, Npart, reduction_level), data, res_rate, res_cnt
    )

    return data, res_rate, res_cnt


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


def bench_entry(theta_crit, config_mode, Npart, reduction_level):
    filename = config_to_filename(theta_crit, config_mode, Npart, reduction_level)
    if os.path.exists(filename):
        print(f"Loading benchmark result from {filename}")
        return load_bench_result(filename)
    else:
        print(f"Running benchmark for {theta_crit:.3f}, {config_mode}, {Npart}, {reduction_level}")
        return cub_collapse(theta_crit, config_mode, Npart, reduction_level)


Nparts = np.logspace(3, 5.5, 10).astype(int).tolist()

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

configs_all = ["none", "direct", "direct_safe"] + configs.copy()

theta_vals = [
    0.1,
    0.12915497,
    0.16681005,
    0.21544347,
    0.27825594,
    0.35938137,
    0.46415888,
    0.5,
    0.59948425,
    0.77426368,
    1.0,
]

# matplotlib style
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["xtick.top"] = True

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["ytick.minor.width"] = 1
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["ytick.right"] = True

plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["lines.linewidth"] = 1.0

plt.rcParams["legend.frameon"] = True

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.05

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 15
plt.rcParams["mathtext.fontset"] = "dejavuserif"

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"


# perf curves
for reduction_level in [1, 2, 3, 4, 5]:
    theta_perf_curve = 0.5

    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    extended_colors = base_colors * 3
    color_cycle = extended_colors

    true_cnt = {config: [] for config in configs_all}
    true_rate = {config: [] for config in configs_all}

    config_plot = {
        "none": ("none", "solid", color_cycle[0]),
        "direct": ("direct", "solid", color_cycle[1]),
        "direct_safe": ("direct_safe", "solid", color_cycle[2]),
    }

    order_list = []
    for config in configs_all:

        if "mm" in config:
            order = int(config[-1])
        elif "fmm" in config:
            order = int(config[-1])
        elif "sfmm" in config:
            order = int(config[-1])
        elif "lowering_sfmm" in config:
            order = int(config[-1])
        else:
            continue

        order_list.append(order)

    order_list = list(set(order_list))
    order_list.sort()

    for i, order in enumerate(order_list):
        config_plot[f"mm{order}"] = (f"mm{order}", "solid", color_cycle[3 + i])
        config_plot[f"fmm{order}"] = (f"fmm{order}", "dashed", color_cycle[3 + i])
        config_plot[f"sfmm{order}"] = (f"sfmm{order}", "dotted", color_cycle[3 + i])
        config_plot[f"lowering_sfmm{order}"] = (
            f"lowering_sfmm{order}",
            "dashdot",
            color_cycle[3 + i],
        )

    for config in configs_all:
        for Npart in Nparts:
            if config in ["direct", "direct_safe"] and Npart > 2e4:
                continue
            _, rate, cnt = bench_entry(theta_perf_curve, config, Npart, reduction_level)
            true_cnt[config].append(cnt)
            true_rate[config].append(rate)

    plt.figure(figsize=(10, 5))
    for config in configs_all:
        (label, linestyle, color) = config_plot[config]
        plt.plot(true_cnt[config], true_rate[config], label=label, linestyle=linestyle, color=color)
    plt.legend(loc="upper left")
    plt.xlabel("Npart")
    plt.ylabel("rate")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"reduction_level = {reduction_level}")
    plt.tight_layout()
    plt.savefig(f"fmm_prec_perf2_reduction_level_{reduction_level}.pdf")

    plt.figure(figsize=(10, 5))
    for config in configs_all:
        (label, linestyle, color) = config_plot[config]
        rate = true_rate[config]
        slowdown_factor = np.array(true_rate["none"][: len(rate)]) / rate
        plt.plot(true_cnt[config], slowdown_factor, label=label, linestyle=linestyle, color=color)
    plt.legend(loc="upper left")
    plt.xlabel("Npart")
    plt.ylabel("none rate / rate (slowdown factor)")
    plt.xscale("log")
    plt.title(f"reduction_level = {reduction_level}")
    plt.ylim(0.9, 5)
    plt.tight_layout()
    plt.savefig(f"fmm_prec_perf2_reduction_level_{reduction_level}_slowdown.pdf")

# precision plot
reduc_level_prec_plot = 3
Npart_precision_plot = 5e4

none_case, none_rate, none_cnt = bench_entry(0, "none", Npart_precision_plot, reduc_level_prec_plot)
direct_safe_case, direct_safe_rate, direct_safe_cnt = bench_entry(
    0, "direct_safe", Npart_precision_plot, reduc_level_prec_plot
)

direct_safe_case = add_data_to_collect(direct_safe_case, none_case)

precision_data = {config: [] for config in configs}
precision_data["direct"] = []

perf_data = {config: [] for config in configs}
perf_data["direct"] = []

for config in precision_data.keys():
    for theta_crit in theta_vals:
        case, rate, cnt = bench_entry(
            theta_crit, config, Npart_precision_plot, reduc_level_prec_plot
        )
        case = add_data_to_collect(case, none_case, direct_safe_case)
        mask = case["r"] > 1e-2  # avoid div by zero in center
        avg_rel_error = np.max(case["rel_error"][mask])
        if False:  # debug plot
            plt.figure()
            plt.scatter(
                case["r"][mask], case["rel_error"][mask], s=1, label=f"{config} {theta_crit:.3f}"
            )
            plt.legend(loc="upper left")
            plt.xlabel("r")
            plt.ylabel("relative error")
            plt.xscale("log")
            plt.yscale("log")
            plt.tight_layout()
            plt.show()
        precision_data[config].append(avg_rel_error)
        perf_data[config].append(rate)

config_plot = {
    "direct": ("direct", "solid", color_cycle[0]),
}

for i, order in enumerate(order_list):
    config_plot[f"mm{order}"] = (f"mm{order}", "solid", color_cycle[1 + i])
    config_plot[f"fmm{order}"] = (f"fmm{order}", "dashed", color_cycle[1 + i])
    config_plot[f"sfmm{order}"] = (f"sfmm{order}", "dotted", color_cycle[1 + i])
    config_plot[f"lowering_sfmm{order}"] = (f"lowering_sfmm{order}", "dashdot", color_cycle[1 + i])


plt.figure(figsize=(10, 5))
for config in precision_data.keys():
    (label, linestyle, color) = config_plot[config]
    plt.plot(theta_vals, precision_data[config], label=label, linestyle=linestyle, color=color)
plt.legend()
plt.xlabel("theta")
plt.ylabel("precision")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-8, 100)
plt.tight_layout()
plt.savefig("fmm_prec_perf2_precision.pdf")


plt.figure(figsize=(10, 5))
for config in perf_data.keys():
    (label, linestyle, color) = config_plot[config]
    plt.plot(
        precision_data[config], perf_data[config], label=label, linestyle=linestyle, color=color
    )
plt.legend()
plt.xlabel("precision")
plt.ylabel("rate")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-8, 100)
plt.tight_layout()
plt.savefig("fmm_prec_perf2_rate.pdf")

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

for config in perf_data.keys():
    (label, linestyle, color) = config_plot[config]

    # Create arrays for the 3D plot with log10 values
    precision = np.array(precision_data[config])
    rate = np.array(perf_data[config])
    theta = np.array(theta_vals[: len(precision)])  # Match length in case of missing data

    # Take log10 of precision and rate
    log_precision = np.log10(precision)
    slowdown = none_rate / rate

    # Plot lines connecting the points
    ax.plot(
        log_precision, slowdown, theta, label=label, linestyle=linestyle, color=color, linewidth=2
    )

    # Add scatter points for better visibility
    ax.scatter(log_precision, slowdown, theta, color=color, s=50, alpha=0.7)

ax.set_xlabel(r"$\log_{10}$(Precision)", fontsize=12)
ax.set_ylabel(r"Slowdown factor", fontsize=12)
ax.set_zlabel(r"$\theta$", fontsize=12)
ax.set_xlim(-6, 0)
ax.set_ylim(1, 3)
ax.legend(loc="best")
ax.set_title("Precision vs Rate vs Opening Angle", fontsize=14)
plt.tight_layout()
plt.savefig("fmm_prec_perf2_3d.pdf")
plt.show()
