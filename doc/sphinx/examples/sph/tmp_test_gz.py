"""
Uniform box in SPH
==================

This simple example shows a uniform density box in SPH, it is also used to test that the
smoothing length iteration find the correct value
"""

# sphinx_gallery_multi_image = "single"

from time import sleep

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


results = {}


def run_case(N_target, split_fact):
    # %%
    # Setup parameters

    gamma = 5.0 / 3.0
    rho_g = 1

    bmin = (-0.6, -0.6, -0.6)
    bmax = (0.6, 0.6, 0.6)

    scheduler_split_val = int(N_target / split_fact)
    scheduler_merge_val = int(1)

    # %%
    # Deduced quantities
    import numpy as np

    xm, ym, zm = bmin
    xM, yM, zM = bmax
    vol_b = (xM - xm) * (yM - ym) * (zM - zm)

    part_vol = vol_b / N_target

    # lattice volume
    HCP_PACKING_DENSITY = 0.74
    part_vol_lattice = HCP_PACKING_DENSITY * part_vol

    dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)

    pmass = -1

    # %%
    # Setup

    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )
    cfg.set_boundary_periodic()
    cfg.set_eos_adiabatic(gamma)
    cfg.print_status()
    model.set_solver_config(cfg)
    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
    xm, ym, zm = bmin
    xM, yM, zM = bmax

    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen, insert_step=scheduler_split_val)

    vol_b = (xM - xm) * (yM - ym) * (zM - zm)

    totmass = rho_g * vol_b

    pmass = model.total_mass_to_part_mass(totmass)

    model.set_value_in_a_box("uint", "f64", 0, bmin, bmax)

    tot_u = pmass * model.get_sum("uint", "f64")
    if shamrock.sys.world_rank() == 0:
        print("total u :", tot_u)

    model.set_particle_mass(pmass)

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)

    # %%
    # Single timestep to iterate the smoothing length
    model.change_htolerances(coarse=1.3, fine=1.1)
    model.timestep()
    model.change_htolerances(coarse=1.1, fine=1.05)

    model.timestep()
    model.timestep()

    model.solver_logs_last_rate()
    model.solver_logs_last_obj_count()

    ret_last_rate = model.solver_logs_last_rate()
    ret_last_obj_count = model.solver_logs_last_obj_count()
    ret_patch_count = len(ctx.get_patch_list_global())

    del model
    del ctx

    return ret_last_rate, ret_last_obj_count, ret_patch_count


sleep(5)

for i in range(3):
    N_target, split_fact = 1e6, 8
    run_case(N_target, split_fact)

sleep(5)

for N_target in [1e3, 1e4, 1e5, 1e6]:
    for split_fact in [0.5, 1, 8]:

        rate, cnt, patch_count = run_case(N_target, split_fact)

        if not N_target in results:
            results[N_target] = {
                "rate": [],
                "cnt": [],
                "tstep": [],
                "patch_count": [],
            }
        results[N_target]["rate"].append(rate)
        results[N_target]["cnt"].append(cnt)
        results[N_target]["tstep"].append(cnt / rate)
        results[N_target]["patch_count"].append(patch_count)


if True and shamrock.sys.world_rank() == 0:
    import json

    print(json.dumps(results, indent=4))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5), dpi=200)
    for N_target in results:
        plt.plot(
            results[N_target]["patch_count"],
            results[N_target]["rate"],
            "x-",
            label=f"N_target = {N_target:.1e}",
        )
    plt.xlabel("patch count")
    plt.ylabel("rate")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("rate vs patch count")
    plt.legend()
    plt.savefig("rate_vs_patch_count.png")

    plt.figure(figsize=(8, 5), dpi=200)
    for N_target in results:
        plt.plot(
            results[N_target]["patch_count"],
            results[N_target]["tstep"],
            "x-",
            label=f"N_target = {N_target:.1e}",
        )
    plt.xlabel("patch count")
    plt.ylabel("tstep")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("tstep vs patch count")
    plt.legend()
    plt.savefig("tstep_vs_patch_count.png")
    plt.show()
