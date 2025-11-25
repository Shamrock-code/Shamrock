"""
Test the precision of SG methods in SPH
============================

Test that all methodes give expected deviantion compare to the reference mode
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# Particle tracking is an experimental feature
shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


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


def compare_sg_methods_data(no_sg_data, reference_data, data_to_comp, sat_relative_error=1e-10):
    a_sg = data_to_comp["axyz"] - no_sg_data["axyz"]
    a_sg_ref = reference_data["axyz"] - no_sg_data["axyz"]
    delta_sg = a_sg - a_sg_ref

    delta_sg_norm = np.linalg.norm(delta_sg, axis=1)
    rel_delta_norm = delta_sg_norm / (np.linalg.norm(a_sg_ref, axis=1) + sat_relative_error)

    return delta_sg, rel_delta_norm, data_to_comp["xyz"]

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

    if max_rel_delta > tols["max_rel_delta"][1] or max_rel_delta < tols["max_rel_delta"][0]:
        raise ValueError(f"max relative error {method_name} is out of tolerance for {setup_name}: {max_rel_delta} not in [{tols['max_rel_delta'][0]}, {tols['max_rel_delta'][1]}]")
    if avg_rel_delta > tols["avg_rel_delta"][1] or avg_rel_delta < tols["avg_rel_delta"][0]:
        raise ValueError(f"avg relative error {method_name} is out of tolerance for {setup_name}: {avg_rel_delta} not in [{tols['avg_rel_delta'][0]}, {tols['avg_rel_delta'][1]}]")
    if min_rel_delta > tols["min_rel_delta"][1] or min_rel_delta < tols["min_rel_delta"][0]:
        raise ValueError(f"min relative error {method_name} is out of tolerance for {setup_name}: {min_rel_delta} not in [{tols['min_rel_delta'][0]}, {tols['min_rel_delta'][1]}]")
    if std_rel_delta > tols["std_rel_delta"][1] or std_rel_delta < tols["std_rel_delta"][0]:
        raise ValueError(f"std relative error {method_name} is out of tolerance for {setup_name}: {std_rel_delta} not in [{tols['std_rel_delta'][0]}, {tols['std_rel_delta'][1]}]")

    

    

def compare_sg_methods(setup_func, setup_name, tols):

    def sg_case_none(cfg):
        cfg.set_self_gravity_none()

    def sg_case_reference(cfg):
        cfg.set_self_gravity_direct(reference_mode=True)
        cfg.set_softening_plummer(epsilon=1e-9)

    def sg_case_direct(cfg):
        cfg.set_self_gravity_direct(reference_mode=False)
        cfg.set_softening_plummer(epsilon=1e-9)

    no_sg_data = run_case(setup_func, setup_name, sg_case_none)
    reference_data = run_case(setup_func, setup_name, sg_case_reference)

    direct_data = run_case(setup_func, setup_name, sg_case_direct)

    delta_sg_direct, rel_delta_direct, xyz_direct = compare_sg_methods_data(
        no_sg_data, reference_data, direct_data
    )

    check_print_errors(rel_delta_direct, setup_name, "direct", tols["direct"])

    return delta_sg_direct, rel_delta_direct, xyz_direct


def plot3d_delta_sg(delta_sg_norm, xyz, case_name):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    dat = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=delta_sg_norm, s=1, cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"reference - {case_name} relative error")
    ax.set_aspect("equal")
    fig.colorbar(dat)

    return fig


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

tol_hcp_cube = {
    "direct": {
        "max_rel_delta": [0., 1e-20],
        "avg_rel_delta": [0., 1e-20],
        "min_rel_delta": [0., 1e-20],
        "std_rel_delta": [0., 1e-20]
    }
}

delta_sg_direct, rel_delta_direct, xyz_direct = compare_sg_methods(setup_cube_hcp, "cube_hcp", tol_hcp_cube)

fig = plot3d_delta_sg(rel_delta_direct, xyz_direct, "cube_hcp")
plt.show()
