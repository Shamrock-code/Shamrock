"""
SPH benchmark for an accretion disc
===================================

This example tests the performance of the SPH solver for a locally isothermal accretion disc
orbiting a central sink particle. It is the disc counterpart of the homogeneous density box
benchmark: the resolution is automatically adapted to the available memory and number of
processes, and the same step is replayed several times to measure the solver rate.
"""

import datetime
import json
import math
import time
from statistics import mean, stdev

import shamrock

device_properties = shamrock.sys.get_compute_device_properties()

microbench_results = shamrock.sys.get_microbench_results(allow_run=True)
if len(microbench_results) == 0:
    print("no microbench results, please run with --benchmark-mpi")
    raise ValueError("no microbench results")

memory_gb = device_properties["global_mem_size"] / (1e9)

N_target_base = 2 ** int(math.log2(memory_gb * 1e6 / 1.5))
print(f"N_target_base = {N_target_base}")
print(f"memory_gb = {memory_gb}")
print(f"device_properties = {device_properties}")

N_target_base = min(N_target_base, 2**25)

if device_properties["type"] == "CPU":
    N_target_base = min(N_target_base, 2**23)

shamrock.backends.reset_mem_info_max()

compute_multiplier = shamrock.sys.world_size()
# compute_multiplier = 12
scheduler_split_val = int(2e7)
scheduler_merge_val = 1

N_target = N_target_base * compute_multiplier

# %%
# Disc parameters (same disc as the dusty disc perf test, without the dust)
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),  # year
    unit_length=sicte.au(),  # astro unit
    unit_mass=sicte.sol_mass(),
)

center_mass = 1.0  # sol mass
center_racc = 8.0  # au

disc = shamrock.utils.disc_setup.StandardDisc(
    units=codeu,
    center_mass=center_mass,
    disc_mass=0.05,  # sol mass
    rin=10.0,  # au
    rout=150.0,  # au
    H_r_0=0.1,
    q=0.5,
    p=3.0 / 2.0,
    r0=10.0,
    rotation="subkeplerian_3d",
    inner_tapering=True,
)

bsize = disc.rout * 2
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)

if shamrock.sys.world_rank() == 0:
    print("N_target_base", N_target_base)
    print("compute_multiplier", compute_multiplier)
    print("scheduler_split_val", scheduler_split_val)
    print("scheduler_merge_val", scheduler_merge_val)
    print("N_target", N_target)

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_eos_locally_isothermalLP07(cs0=disc.cs0(), q=disc.q, r0=disc.r0)
cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box
cfg.set_units(codeu)
cfg.set_particle_mass(disc.part_mass(N_target))
cfg.set_cfl_cour(0.1)
cfg.set_cfl_force(0.1)
cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(scheduler_split_val, scheduler_merge_val)

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = disc.make_generator(setup, N_target, random_seed=666)

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

model.add_sink(center_mass, (0, 0, 0), (0, 0, 0), center_racc)

shamrock.backends.reset_mem_info_max()

# converge smoothing length and compute initial dt
# (the disc is generated with a rough guess of h, hence the larger coarse tolerance)
model.change_htolerances(coarse=1.3, fine=1.1)
model.timestep()
model.change_htolerances(coarse=1.1, fine=1.1)

# Now run the actual benchmark for 10 steps
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
    before_mpi_timers = shamrock.comm.get_timers()


def callback_after_mpi_timer():
    global after_mpi_timers
    after_mpi_timers = shamrock.comm.get_timers()


model.add_timestep_callback(step_begin=callback_before_mpi_timer, step_end=callback_after_mpi_timer)

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

    time.sleep(1)

    if shamrock.sys.world_rank() == 0:
        print("done sleeping 1 second ...")

# result is the best rate of the 10 steps
res_rate, res_cnt = max(res_rates), res_cnts[0]

# index of the max rate
max_rate_index = res_rates.index(max(res_rates))
max_rate_system_metrics = res_system_metrics[max_rate_index]
max_mpi_timers = res_mpi_timers[max_rate_index]
step_time = res_cnt / res_rate

if shamrock.sys.world_rank() == 0:
    result_text = ""
    result_text += f"--- final score for N_target_base={N_target_base} ---\n"
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
        "device_properties": device_properties,
        "microbench_results": shamrock.sys.get_microbench_results(),
        "shamrock_version": shamrock.version_string(),
        "shamrock_compiler_id_string": shamrock.get_compiler_id_string(),
        "shamrock_compile_flags": shamrock.get_compile_arg(),
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

    result_text += "---------submit this result--------\n"
    result_text += f"{json.dumps(dic_out, indent=4)}\n"
    result_text += "-----------------------------------\n"

    print("current results:")
    print(result_text)
