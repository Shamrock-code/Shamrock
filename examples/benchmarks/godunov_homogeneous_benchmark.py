"""
AMR benchmark for homogeneous density box
=========================================

This example tests the the performance of the AMR solver for a homogeneous density box,
the resolution is automatically adapted to the available memory and number of processes.
"""

import datetime
import json
import math
from statistics import mean, stdev

import shamrock

device_properties = shamrock.sys.get_compute_device_properties()

microbench_results = shamrock.sys.get_microbench_results(allow_run=True)
if len(microbench_results) == 0:
    print("no microbench results, please run with --benchmark-mpi")
    raise ValueError("no microbench results")

memory_gb = device_properties["global_mem_size"] / (1e9)

sz = 1 << 1

N_target_blocks = 2 ** (math.floor(math.log2(memory_gb * 1e6 / (1.5 * 8))))


print(f"N_target_blocks = {N_target_blocks}")
print(f"memory_gb = {memory_gb}")
print(f"device_properties = {device_properties}")

N_target_blocks = min(N_target_blocks, 2**22)

if device_properties["type"] == "CPU":
    N_target_blocks = min(N_target_blocks, 2**20)

# make_base_grid takes a per-axis block count, not a total, so convert the
# memory-based total block budget above into a per-axis size (kept as a power
# of two so the base grid splits evenly across the 3 axes).
N_per_axis = 2 ** (int(math.log2(N_target_blocks)) // 3)

shamrock.backends.reset_mem_info_max()


compute_multiplier = shamrock.sys.world_size()
# compute_multiplier = 12
scheduler_split_val = int(2e7)
scheduler_merge_val = 1


if shamrock.sys.world_rank() == 0:
    print("N_target_block", N_target_blocks)
    print("N_per_axis", N_per_axis)
    print("scheduler_split_val", scheduler_split_val)
    print("scheduler_merge_val", scheduler_merge_val)
    print("N_target", N_per_axis**3 * 8)


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

multx = 1
multy = 1
multz = 1


cfg = model.gen_default_config()
scale_fact = 1 / (sz * N_per_axis * multx)
cfg.set_scale_factor(scale_fact)
cfg.set_riemann_solver_hllc()
cfg.set_eos_gamma(1.66667)
cfg.set_slope_lim_vanleer_sym()
cfg.set_face_time_interpolation(True)
model.init_scheduler(scheduler_split_val, scheduler_merge_val)
model.make_base_grid(
    (0, 0, 0),
    (sz, sz, sz),
    (N_per_axis * multx, N_per_axis * multy, N_per_axis * multz),
)


def rho_map(rmin, rmax):
    x, y, z = rmin
    if x > 0.25 and x < 0.75:
        return 2
    return 1.0


def rhoe_map(rmin, rmax):
    rho = rho_map(rmin, rmax)
    return 1.0 * rho


def rhovel_map(rmin, rmax):
    rho = rho_map(rmin, rmax)
    return (1 * rho, 0 * rho, 0 * rho)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoe_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)


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
    result_text = ""
    result_text += f"--- final score for N_target_block={N_target_blocks} ---"
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
