"""
Test the precision of SG methods in SPH
============================

Test that all methodes give expected deviantion compare to the reference mode
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# %%
# Shamrock init

# Self-gravity is still an experimental feature
shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Parameters of the test

# tolerances for the test for each quantity [min, max] outside = fail
# here the challenge is to find the right tolerances for many combinations of hardware
TOL_HPC_CUBE = {
    "direct": {
        "max_rel_delta": [0.0, 1e-15],
        "avg_rel_delta": [0.0, 1e-15],
        "min_rel_delta": [0.0, 1e-16],
        "std_rel_delta": [0.0, 1e-16],
    },
    "mm1": {
        "max_rel_delta": [0.10237644408204995 - 2e-6, 0.10237644408204995 + 2e-6],
        "avg_rel_delta": [0.02854797753451957 - 1e-5, 0.02854797753451957 + 1e-5],
        "min_rel_delta": [0.0011855528704246662 - 2e-6, 0.0011855528704246662 + 2e-6],
        "std_rel_delta": [0.012057071282465973 - 1e-5, 0.012057071282465973 + 1e-5],
    },
    "mm2": {
        "max_rel_delta": [0.09870604133669476 - 2e-7, 0.09870604133669476 + 2e-7],
        "avg_rel_delta": [0.02809153086497972 - 4e-6, 0.02809153086497972 + 4e-6],
        "min_rel_delta": [0.0007660948531784951 - 2e-7, 0.0007660948531784951 + 2e-7],
        "std_rel_delta": [0.011728053659573516 - 3e-6, 0.011728053659573516 + 3e-6],
    },
    "mm3": {
        "max_rel_delta": [0.053440736724222254 - 1e-2, 0.053440736724222254 + 1e-2],
        "avg_rel_delta": [0.016082633182446748 - 1e-5, 0.016082633182446748 + 1e-5],
        "min_rel_delta": [0.0006196291223064161 - 1e-5, 0.0006196291223064161 + 1e-5],
        "std_rel_delta": [0.007676767861132326 - 1e-5, 0.007676767861132326 + 1e-5],
    },
    "mm4": {
        "max_rel_delta": [0.05833850788436308 - 4e-7, 0.05833850788436308 + 4e-7],
        "avg_rel_delta": [0.014909667170457708 - 4e-6, 0.014909667170457708 + 4e-6],
        "min_rel_delta": [0.000900026203337091 - 4e-7, 0.000900026203337091 + 4e-7],
        "std_rel_delta": [0.007051020680900702 - 1e-5, 0.007051020680900702 + 1e-5],
    },
    "mm5": {
        "max_rel_delta": [0.03130228236510434 - 4e-8, 0.03130228236510434 + 4e-8],
        "avg_rel_delta": [0.005541631318650798 - 2e-6, 0.005541631318650798 + 2e-6],
        "min_rel_delta": [0.00011105937841131173 - 4e-8, 0.00011105937841131173 + 4e-8],
        "std_rel_delta": [0.003575474772831352 - 4e-7, 0.003575474772831352 + 4e-7],
    },
    "fmm1": {
        "max_rel_delta": [0.09457251908559332 - 1e-15, 0.09457251908559332 + 1e-15],
        "avg_rel_delta": [0.025849750275017017 - 1.5e-7, 0.025849750275017017 + 1.5e-7],
        "min_rel_delta": [0.0009107123147511316 - 1e-15, 0.0009107123147511316 + 1e-15],
        "std_rel_delta": [0.014771871930712405 - 5e-7, 0.014771871930712405 + 5e-7],
    },
    "fmm2": {
        "max_rel_delta": [0.0637884157532678 - 2e-15, 0.0637884157532678 + 2e-15],
        "avg_rel_delta": [0.011974487570609172 - 1e-6, 0.011974487570609172 + 1e-6],
        "min_rel_delta": [0.0003606563787779815 - 1e-15, 0.0003606563787779815 + 1e-15],
        "std_rel_delta": [0.005685736194766207 - 1e-6, 0.005685736194766207 + 1e-6],
    },
    "fmm3": {
        "max_rel_delta": [0.041451600220479765 - 1e-14, 0.041451600220479765 + 1e-14],
        "avg_rel_delta": [0.00590471908748555 - 1e-7, 0.00590471908748555 + 1e-7],
        "min_rel_delta": [0.00011353264639102378 - 1e-14, 0.00011353264639102378 + 1e-14],
        "std_rel_delta": [0.003551982999913884 - 1e-6, 0.003551982999913884 + 1e-6],
    },
    "fmm4": {
        "max_rel_delta": [0.027041008234275913 - 1e-15, 0.027041008234275913 + 1e-15],
        "avg_rel_delta": [0.004870267930794465 - 1e-15, 0.004870267930794465 + 1e-15],
        "min_rel_delta": [9.263728961868266e-05 - 1e-15, 9.263728961868266e-05 + 1e-15],
        "std_rel_delta": [0.0028526970191818392 - 1e-15, 0.0028526970191818392 + 1e-15],
    },
    "fmm5": {
        "max_rel_delta": [0.013154949137016961 - 1e-15, 0.013154949137016961 + 1e-15],
        "avg_rel_delta": [0.001933834336767182 - 1e-15, 0.001933834336767182 + 1e-15],
        "min_rel_delta": [3.7696982571244474e-05 - 1e-15, 3.7696982571244474e-05 + 1e-15],
        "std_rel_delta": [0.0014635310344450105 - 1e-15, 0.0014635310344450105 + 1e-15],
    },
}


# %%
# Helper functions for this test


# helper to run one case (SG config & setup)
def run_case(setup_func, setup_name, sg_setup_func):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_default_config()

    sg_setup_func(cfg)

    setup_func(model, cfg)

    model.timestep()

    data = ctx.collect_data()

    return data


# Compare the SG method to the reference and the one without SG and return error metrics
def compare_sg_methods_data(no_sg_data, reference_data, data_to_comp, sat_relative_error=1):
    a_sg = data_to_comp["axyz"] - no_sg_data["axyz"]

    a_sg_ref = reference_data["axyz"] - no_sg_data["axyz"]
    delta_sg = a_sg - a_sg_ref

    delta_sg_norm = np.linalg.norm(delta_sg, axis=1)
    rel_delta_norm = delta_sg_norm / (np.max(np.linalg.norm(a_sg_ref, axis=1)))

    return delta_sg, rel_delta_norm, data_to_comp["xyz"]


# Compute error related quantities and check if they are within the tolerances
def check_print_errors(rel_delta, setup_name, method_name, tols):
    max_rel_delta = np.max(np.abs(rel_delta))
    if shamrock.sys.world_rank() == 0:
        print(f"max relative error {method_name}: {max_rel_delta} for {setup_name}")
    avg_rel_delta = np.mean(np.abs(rel_delta))
    if shamrock.sys.world_rank() == 0:
        print(f"avg relative error {method_name}: {avg_rel_delta} for {setup_name}")
    min_rel_delta = np.min(np.abs(rel_delta))
    if shamrock.sys.world_rank() == 0:
        print(f"min relative error {method_name}: {min_rel_delta} for {setup_name}")
    std_rel_delta = np.std(np.abs(rel_delta))
    if shamrock.sys.world_rank() == 0:
        print(f"std relative error {method_name}: {std_rel_delta} for {setup_name}")

    delta_max_tol = (tols["max_rel_delta"][1] + tols["max_rel_delta"][0]) / 2 - max_rel_delta
    delta_avg_tol = (tols["avg_rel_delta"][1] + tols["avg_rel_delta"][0]) / 2 - avg_rel_delta
    delta_min_tol = (tols["min_rel_delta"][1] + tols["min_rel_delta"][0]) / 2 - min_rel_delta
    delta_std_tol = (tols["std_rel_delta"][1] + tols["std_rel_delta"][0]) / 2 - std_rel_delta

    to_raise = []

    if max_rel_delta > tols["max_rel_delta"][1] or max_rel_delta < tols["max_rel_delta"][0]:
        to_raise.append(
            f"max relative error {method_name} is out of tolerance for {setup_name}: {max_rel_delta} not in [{tols['max_rel_delta'][0]}, {tols['max_rel_delta'][1]}], delta = {delta_max_tol}"
        )
    if avg_rel_delta > tols["avg_rel_delta"][1] or avg_rel_delta < tols["avg_rel_delta"][0]:
        to_raise.append(
            f"avg relative error {method_name} is out of tolerance for {setup_name}: {avg_rel_delta} not in [{tols['avg_rel_delta'][0]}, {tols['avg_rel_delta'][1]}], delta = {delta_avg_tol}"
        )
    if min_rel_delta > tols["min_rel_delta"][1] or min_rel_delta < tols["min_rel_delta"][0]:
        to_raise.append(
            f"min relative error {method_name} is out of tolerance for {setup_name}: {min_rel_delta} not in [{tols['min_rel_delta'][0]}, {tols['min_rel_delta'][1]}], delta = {delta_min_tol}"
        )
    if std_rel_delta > tols["std_rel_delta"][1] or std_rel_delta < tols["std_rel_delta"][0]:
        to_raise.append(
            f"std relative error {method_name} is out of tolerance for {setup_name}: {std_rel_delta} not in [{tols['std_rel_delta'][0]}, {tols['std_rel_delta'][1]}], delta = {delta_std_tol}"
        )

    if len(to_raise) > 0:
        print(f"Errors for {setup_name} {method_name}:")
        for to_raise_item in to_raise:
            print(to_raise_item)

    for to_raise_item in to_raise:
        raise ValueError(to_raise_item)


# Compare the SG method to the reference and the one without SG and return error metrics
def compare_sg_methods(setup_func, setup_name, tols):

    def sg_case_none(cfg):
        cfg.set_self_gravity_none()

    def sg_case_reference(cfg):
        cfg.set_self_gravity_direct(reference_mode=True)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_direct(cfg):
        cfg.set_self_gravity_direct(reference_mode=False)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_mm1(cfg):
        cfg.set_self_gravity_mm(order=1, opening_angle=0.5, reduction_level=3)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_mm2(cfg):
        cfg.set_self_gravity_mm(order=2, opening_angle=0.5, reduction_level=3)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_mm3(cfg):
        cfg.set_self_gravity_mm(order=3, opening_angle=0.5, reduction_level=3)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_mm4(cfg):
        cfg.set_self_gravity_mm(order=4, opening_angle=0.5, reduction_level=3)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_mm5(cfg):
        cfg.set_self_gravity_mm(order=5, opening_angle=0.5, reduction_level=3)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_fmm1(cfg):
        cfg.set_self_gravity_fmm(order=1, opening_angle=0.5, reduction_level=3)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_fmm2(cfg):
        cfg.set_self_gravity_fmm(order=2, opening_angle=0.5, reduction_level=3)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_fmm3(cfg):
        cfg.set_self_gravity_fmm(order=3, opening_angle=0.5, reduction_level=3)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_fmm4(cfg):
        cfg.set_self_gravity_fmm(order=4, opening_angle=0.5, reduction_level=3)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_fmm5(cfg):
        cfg.set_self_gravity_fmm(order=5, opening_angle=0.5, reduction_level=3)
        cfg.set_softening_plummer(epsilon=1e-9)

    no_sg_data = run_case(setup_func, setup_name, sg_case_none)
    reference_data = run_case(setup_func, setup_name, sg_case_reference)

    direct_data = run_case(setup_func, setup_name, sg_case_direct)
    mm1_data = run_case(setup_func, setup_name, sg_case_mm1)
    mm2_data = run_case(setup_func, setup_name, sg_case_mm2)
    mm3_data = run_case(setup_func, setup_name, sg_case_mm3)
    mm4_data = run_case(setup_func, setup_name, sg_case_mm4)
    mm5_data = run_case(setup_func, setup_name, sg_case_mm5)
    fmm1_data = run_case(setup_func, setup_name, sg_case_fmm1)
    fmm2_data = run_case(setup_func, setup_name, sg_case_fmm2)
    fmm3_data = run_case(setup_func, setup_name, sg_case_fmm3)
    fmm4_data = run_case(setup_func, setup_name, sg_case_fmm4)
    fmm5_data = run_case(setup_func, setup_name, sg_case_fmm5)

    delta_sg_direct, rel_delta_direct, xyz_direct = compare_sg_methods_data(
        no_sg_data, reference_data, direct_data
    )

    delta_sg_mm1, rel_delta_mm1, xyz_mm1 = compare_sg_methods_data(
        no_sg_data, reference_data, mm1_data
    )
    delta_sg_mm2, rel_delta_mm2, xyz_mm2 = compare_sg_methods_data(
        no_sg_data, reference_data, mm2_data
    )
    delta_sg_mm3, rel_delta_mm3, xyz_mm3 = compare_sg_methods_data(
        no_sg_data, reference_data, mm3_data
    )
    delta_sg_mm4, rel_delta_mm4, xyz_mm4 = compare_sg_methods_data(
        no_sg_data, reference_data, mm4_data
    )
    delta_sg_mm5, rel_delta_mm5, xyz_mm5 = compare_sg_methods_data(
        no_sg_data, reference_data, mm5_data
    )

    delta_sg_fmm1, rel_delta_fmm1, xyz_fmm1 = compare_sg_methods_data(
        no_sg_data, reference_data, fmm1_data
    )
    delta_sg_fmm2, rel_delta_fmm2, xyz_fmm2 = compare_sg_methods_data(
        no_sg_data, reference_data, fmm2_data
    )
    delta_sg_fmm3, rel_delta_fmm3, xyz_fmm3 = compare_sg_methods_data(
        no_sg_data, reference_data, fmm3_data
    )
    delta_sg_fmm4, rel_delta_fmm4, xyz_fmm4 = compare_sg_methods_data(
        no_sg_data, reference_data, fmm4_data
    )
    delta_sg_fmm5, rel_delta_fmm5, xyz_fmm5 = compare_sg_methods_data(
        no_sg_data, reference_data, fmm5_data
    )

    check_print_errors(rel_delta_direct, setup_name, "direct", tols["direct"])
    check_print_errors(rel_delta_mm1, setup_name, "mm1", tols["mm1"])
    check_print_errors(rel_delta_mm2, setup_name, "mm2", tols["mm2"])
    check_print_errors(rel_delta_mm3, setup_name, "mm3", tols["mm3"])
    check_print_errors(rel_delta_mm4, setup_name, "mm4", tols["mm4"])
    check_print_errors(rel_delta_mm5, setup_name, "mm5", tols["mm5"])
    check_print_errors(rel_delta_fmm1, setup_name, "fmm1", tols["fmm1"])
    check_print_errors(rel_delta_fmm2, setup_name, "fmm2", tols["fmm2"])
    check_print_errors(rel_delta_fmm3, setup_name, "fmm3", tols["fmm3"])
    check_print_errors(rel_delta_fmm4, setup_name, "fmm4", tols["fmm4"])
    check_print_errors(rel_delta_fmm5, setup_name, "fmm5", tols["fmm5"])

    return (
        {
            "direct": delta_sg_direct,
            "mm1": delta_sg_mm1,
            "mm2": delta_sg_mm2,
            "mm3": delta_sg_mm3,
            "mm4": delta_sg_mm4,
            "mm5": delta_sg_mm5,
            "fmm1": delta_sg_fmm1,
            "fmm2": delta_sg_fmm2,
            "fmm3": delta_sg_fmm3,
            "fmm4": delta_sg_fmm4,
            "fmm5": delta_sg_fmm5,
        },
        {
            "direct": rel_delta_direct,
            "mm1": rel_delta_mm1,
            "mm2": rel_delta_mm2,
            "mm3": rel_delta_mm3,
            "mm4": rel_delta_mm4,
            "mm5": rel_delta_mm5,
            "fmm1": rel_delta_fmm1,
            "fmm2": rel_delta_fmm2,
            "fmm3": rel_delta_fmm3,
            "fmm4": rel_delta_fmm4,
            "fmm5": rel_delta_fmm5,
        },
        {
            "direct": xyz_direct,
            "mm1": xyz_mm1,
            "mm2": xyz_mm2,
            "mm3": xyz_mm3,
            "mm4": xyz_mm4,
            "mm5": xyz_mm5,
            "fmm1": xyz_fmm1,
            "fmm2": xyz_fmm2,
            "fmm3": xyz_fmm3,
            "fmm4": xyz_fmm4,
            "fmm5": xyz_fmm5,
        },
    )


# Plot the 3D delta of the SG method
def plot3d_delta_sg(delta_sg_norm, xyz, case_name, method_name):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    dat = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=delta_sg_norm, s=1, cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{method_name} - {case_name} relative error")
    ax.set_aspect("equal")
    fig.colorbar(dat)

    return fig


# %%
# Setup for the test


def setup_cube_hcp(model, cfg):

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

    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )
    cfg.set_boundary_periodic()
    cfg.set_eos_adiabatic(gamma)

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


# %%
# Run the tests
# ^^^^^^^^^^^^^

delta_sg_dict, rel_delta_dict, xyz_dict = compare_sg_methods(
    setup_cube_hcp, "cube_hcp", TOL_HPC_CUBE
)

# %%
fig = plot3d_delta_sg(rel_delta_dict["direct"], xyz_dict["direct"], "cube_hcp", "direct")
plt.show()

# %%
fig = plot3d_delta_sg(rel_delta_dict["mm1"], xyz_dict["mm1"], "cube_hcp", "mm1")
plt.show()

# %%
fig = plot3d_delta_sg(rel_delta_dict["mm2"], xyz_dict["mm2"], "cube_hcp", "mm2")
plt.show()

# %%
fig = plot3d_delta_sg(rel_delta_dict["mm3"], xyz_dict["mm3"], "cube_hcp", "mm3")
plt.show()

# %%
fig = plot3d_delta_sg(rel_delta_dict["mm4"], xyz_dict["mm4"], "cube_hcp", "mm4")
plt.show()

# %%
fig = plot3d_delta_sg(rel_delta_dict["mm5"], xyz_dict["mm5"], "cube_hcp", "mm5")
plt.show()

# %%
fig = plot3d_delta_sg(rel_delta_dict["fmm1"], xyz_dict["fmm1"], "cube_hcp", "fmm1")
plt.show()

# %%
fig = plot3d_delta_sg(rel_delta_dict["fmm2"], xyz_dict["fmm2"], "cube_hcp", "fmm2")
plt.show()

# %%
fig = plot3d_delta_sg(rel_delta_dict["fmm3"], xyz_dict["fmm3"], "cube_hcp", "fmm3")
plt.show()

# %%
fig = plot3d_delta_sg(rel_delta_dict["fmm4"], xyz_dict["fmm4"], "cube_hcp", "fmm4")
plt.show()

# %%
fig = plot3d_delta_sg(rel_delta_dict["fmm5"], xyz_dict["fmm5"], "cube_hcp", "fmm5")
plt.show()
