"""
Weak scale test for SPH
=======================

This example tests the weak scalability performance of the SPH solver.
It is based on a Sedov blast wave simulation, the number of particles is scaled
with the number of processes. Run it only if you have enough memory available.
"""

from statistics import mean, stdev

import shamrock

result_text = ""

for N_target_base in [32e6]:
    shamrock.backends.reset_mem_info_max()

    gamma = 5.0 / 3.0
    rho_g = 1
    target_tot_u = 1

    bmin = (-0.6, -0.6, -0.6)
    bmax = (0.6, 0.6, 0.6)

    compute_multiplier = shamrock.sys.world_size()
    # compute_multiplier = 12
    scheduler_split_val = int(2e7)
    scheduler_merge_val = int(1)

    N_target = N_target_base * compute_multiplier
    xm, ym, zm = bmin
    xM, yM, zM = bmax
    vol_b = (xM - xm) * (yM - ym) * (zM - zm)

    if shamrock.sys.world_rank() == 0:
        print("N_target_base", N_target_base)
        print("compute_multiplier", compute_multiplier)
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
    cfg.print_status()
    model.set_solver_config(cfg)
    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
    xm, ym, zm = bmin
    xM, yM, zM = bmax

    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)

    # Kind of optimized for Aurora
    setup.apply_setup(
        gen,
        gen_step=int(scheduler_split_val / 8),
        insert_step=int(scheduler_split_val * 2),
        msg_count_limit=1024,
        rank_comm_size_limit=int(scheduler_split_val) * 2,
        max_msg_size=int(scheduler_split_val / 8),
        do_setup_log=False,
    )

    xc, yc, zc = model.get_closest_part_to((0, 0, 0))

    if shamrock.sys.world_rank() == 0:
        print("closest part to (0,0,0) is in :", xc, yc, zc)

    vol_b = (xM - xm) * (yM - ym) * (zM - zm)

    totmass = rho_g * vol_b
    # print("Total mass :", totmass)

    pmass = model.total_mass_to_part_mass(totmass)

    model.set_value_in_a_box("uint", "f64", 0, bmin, bmax)

    rinj = 16 * dr
    u_inj = 1
    model.add_kernel_value("uint", "f64", u_inj, (0, 0, 0), rinj)

    tot_u = pmass * model.get_sum("uint", "f64")
    if shamrock.sys.world_rank() == 0:
        print("total u :", tot_u)

    # print("Current part mass :", pmass)
    model.set_particle_mass(pmass)

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)

    shamrock.backends.reset_mem_info_max()

    # converge smoothing length and compute initial dt
    model.timestep()

    # Now run the actual benchmark for 5 steps
    res_rates = []
    res_cnts = []
    res_system_metrics = []
    res_mpi_timers = []

    """
    Here we insert callbacks to measure solver MPI usage by fetching the timers twice at the begining and end of the step
    """
    before_mpi_timers, after_mpi_timers = None, None

    def callback_before_mpi_timer():
        global before_mpi_timers
        # print(shamrock.sys.world_rank(), "register before_mpi_timers")
        before_mpi_timers = shamrock.comm.get_timers()

    def callback_after_mpi_timer():
        global after_mpi_timers
        # print(shamrock.sys.world_rank(), "register after_mpi_timers")
        after_mpi_timers = shamrock.comm.get_timers()

    model.add_timestep_callback(
        step_begin=callback_before_mpi_timer, step_end=callback_after_mpi_timer
    )

    for i in range(10):
        if shamrock.sys.world_rank() == 0:
            print("running step ", i + 1, "/", 10, " ...")

        shamrock.sys.mpi_barrier()

        # To replay the same step
        model.set_next_dt(0.0)
        model.timestep()

        if shamrock.sys.world_rank() == 0:
            print("collecting results ...")

        tmp_res_rate, tmp_res_cnt, tmp_system_metrics = (
            model.solver_logs_last_rate(),
            model.solver_logs_last_obj_count(),
            model.solver_logs_last_system_metrics(),
        )
        res_rates.append(tmp_res_rate)
        res_cnts.append(tmp_res_cnt)
        res_system_metrics.append(tmp_system_metrics)
        res_mpi_timers.append(shamrock.comm.mpi_timers_delta(before_mpi_timers, after_mpi_timers))

        if shamrock.sys.world_rank() == 0:
            print("sleeping 1 second ...")

        import time

        time.sleep(1)

        if shamrock.sys.world_rank() == 0:
            print("done sleeping 1 second ...")

    # result is the best rate of the 5 steps
    res_rate, res_cnt = max(res_rates), res_cnts[0]

    # index of the max rate
    max_rate_index = res_rates.index(max(res_rates))
    max_rate_system_metrics = res_system_metrics[max_rate_index]
    max_mpi_timers = res_mpi_timers[max_rate_index]
    step_time = res_cnt / res_rate

    if shamrock.sys.world_rank() == 0:
        result_text += f"--- final score for N_target_base={N_target_base} ---"
        result_text += f"world size  : {shamrock.sys.world_size()}\n"
        result_text += f"result rate : {res_rate}\n"
        result_text += f"result cnt  : {res_cnt}\n"
        result_text += f"cnt/rank    : {res_cnt / shamrock.sys.world_size()}\n"
        result_text += f"result rate per rank : {res_rate / shamrock.sys.world_size()}\n"
        result_text += f"rates infos : max={max(res_rates)}, min={min(res_rates)}, mean={mean(res_rates)}, stddev={stdev(res_rates)}\n"
        result_text += f"res_rates = {res_rates}\n"
        result_text += f"res_cnts = {res_cnts}\n"
        result_text += f"step time = {step_time}\n"

        dic_out = {
            "world_size": shamrock.sys.world_size(),
            "rate": res_rate,
            "cnt": res_cnt,
            "step_time": step_time,
            "mpi_timers": max_mpi_timers,
        }

        # print the system metrics
        metrics_duration = max_rate_system_metrics["duration"]
        result_text += "system metrics:\n"
        for key, value in max_rate_system_metrics.items():
            if not key == "duration":
                result_text += f"{key}: {value} J\n"
                dic_out[key] = value

        for key, value in max_rate_system_metrics.items():
            if not key == "duration":
                result_text += f"avg power {key} / step time : {value / metrics_duration} W\n"
                dic_out[f"power_{key}"] = value / metrics_duration

        dic_out["system_metric_duration"] = metrics_duration

        result_text += f"dic_out = {dic_out}\n"

        print("current results:")
        print(result_text)

if shamrock.sys.world_rank() == 0:
    print(result_text)
