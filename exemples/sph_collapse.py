import matplotlib.pyplot as plt
import numpy as np

import shamrock

# Particle tracking is an experimental feature

shamrock.enable_experimental_features()

shamrock.tree.set_impl_clbvh_dual_tree_traversal("parallel_select", "")

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

Npart = 1e4

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


def run_case(case_name):
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

    if case_name == "none":
        cfg.set_self_gravity_none()
    elif case_name == "direct":
        cfg.set_self_gravity_direct()
    elif case_name == "direct_safe":
        cfg.set_self_gravity_direct(reference_mode=True)
    elif case_name == "mm5":
        cfg.set_self_gravity_mm(order=5, opening_angle=0.5)
    elif case_name == "mm4":
        cfg.set_self_gravity_mm(order=4, opening_angle=0.5)
    elif case_name == "mm3":
        cfg.set_self_gravity_mm(order=3, opening_angle=0.5)
    elif case_name == "fmm3":
        cfg.set_self_gravity_fmm(order=3, opening_angle=0.5)
    elif case_name == "fmm4":
        cfg.set_self_gravity_fmm(order=4, opening_angle=0.5)
    elif case_name == "fmm5":
        cfg.set_self_gravity_fmm(order=5, opening_angle=0.5)
    elif case_name == "sfmm3":
        cfg.set_self_gravity_sfmm(order=3, opening_angle=0.5)
    elif case_name == "sfmm4":
        cfg.set_self_gravity_sfmm(order=4, opening_angle=0.5)
    elif case_name == "sfmm5":
        cfg.set_self_gravity_sfmm(order=5, opening_angle=0.5)
    else:
        raise ValueError(f"Invalid case name: {case_name}")
    cfg.set_softening_plummer(epsilon=1e-9)

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

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)

    model.timestep()

    data = ctx.collect_data()

    # for i in range(10):
    #    model.do_vtk_dump(f"dump_{case_name}_{i:04}.vtk", True)
    #    for i in range(10):
    #        model.timestep()

    return data


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


data_none = run_case("none")

data_direct_safe = run_case("direct_safe")
data_direct_safe = add_data_to_collect(data_direct_safe, data_none)

data_direct = run_case("direct")
data_direct = add_data_to_collect(data_direct, data_none, data_direct_safe)

data_mm5 = run_case("mm5")
data_mm5 = add_data_to_collect(data_mm5, data_none, data_direct_safe)

data_mm4 = run_case("mm4")
data_mm4 = add_data_to_collect(data_mm4, data_none, data_direct_safe)

data_mm3 = run_case("mm3")
data_mm3 = add_data_to_collect(data_mm3, data_none, data_direct_safe)

data_fmm3 = run_case("fmm3")
data_fmm3 = add_data_to_collect(data_fmm3, data_none, data_direct_safe)

data_fmm4 = run_case("fmm4")
data_fmm4 = add_data_to_collect(data_fmm4, data_none, data_direct_safe)

data_fmm5 = run_case("fmm5")
data_fmm5 = add_data_to_collect(data_fmm5, data_none, data_direct_safe)

data_sfmm3 = run_case("sfmm3")
data_sfmm3 = add_data_to_collect(data_sfmm3, data_none, data_direct_safe)

data_sfmm4 = run_case("sfmm4")
data_sfmm4 = add_data_to_collect(data_sfmm4, data_none, data_direct_safe)

data_sfmm5 = run_case("sfmm5")
data_sfmm5 = add_data_to_collect(data_sfmm5, data_none, data_direct_safe)

plt.figure()

plt.scatter(data_direct_safe["r"], data_direct_safe["ar"], s=1, label="direct_safe")
plt.scatter(data_direct["r"], data_direct["ar"], s=1, label="direct")
plt.scatter(data_mm5["r"], data_mm5["ar"], s=1, label="mm order 5")
plt.scatter(data_mm4["r"], data_mm4["ar"], s=1, label="mm order 4")
plt.scatter(data_mm3["r"], data_mm3["ar"], s=1, label="mm order 3")
plt.scatter(data_fmm3["r"], data_fmm3["ar"], s=1, label="fmm order 3")
plt.scatter(data_fmm4["r"], data_fmm4["ar"], s=1, label="fmm order 4")
plt.scatter(data_fmm5["r"], data_fmm5["ar"], s=1, label="fmm order 5")
plt.scatter(data_sfmm3["r"], data_sfmm3["ar"], s=1, label="sfmm order 3")
plt.scatter(data_sfmm4["r"], data_sfmm4["ar"], s=1, label="sfmm order 4")
plt.scatter(data_sfmm5["r"], data_sfmm5["ar"], s=1, label="sfmm order 5")
plt.legend()

plt.xlabel("$r$")
plt.ylabel("$a_r$")


def compute_binned_average(r_values, err_values, bin_size):
    r = np.asarray(r_values)
    err = np.asarray(err_values)
    valid = np.isfinite(r) & np.isfinite(err) & (r >= 0.0)
    if not np.any(valid):
        return np.array([]), np.array([])

    idx = np.floor(r[valid] / bin_size).astype(int)
    counts = np.bincount(idx)
    sums = np.bincount(idx, weights=err[valid])
    mask = counts > 0

    centers = (np.arange(len(counts)) + 0.5) * bin_size
    averages = sums / counts
    return centers[mask], averages[mask]


plt.figure()
cases = (
    ("direct", data_direct),
    ("mm order 5", data_mm5),
    ("mm order 4", data_mm4),
    ("mm order 3", data_mm3),
    ("fmm order 3", data_fmm3),
    ("fmm order 4", data_fmm4),
    ("fmm order 5", data_fmm5),
    ("sfmm order 3", data_sfmm3),
    ("sfmm order 4", data_sfmm4),
    ("sfmm order 5", data_sfmm5),
)

for label, dataset in cases:
    plt.scatter(dataset["r"], dataset["rel_error"], s=1, label=label, alpha=0.05)
    r_avg, err_avg = compute_binned_average(dataset["r"], dataset["rel_error"], dr)
    plt.plot(r_avg, err_avg, label=f"{label}")

plt.yscale("log")
plt.legend()
plt.xlabel("$r$")
plt.ylabel("relative error")
plt.show()
