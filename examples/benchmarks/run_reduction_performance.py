"""
reduction performance benchmarks
================================

This example benchmarks the reduction performance for the different algorithms available in Shamrock
"""

# sphinx_gallery_multi_image = "single"

import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Recover microbenchmark results
microbench_results = shamrock.sys.get_microbench_results(allow_run=True)
if len(microbench_results) == 0:
    print("no microbench results, please run with --benchmark-mpi")
    raise ValueError("no microbench results")


# %%
# Main benchmark functions
def benchmark_f32(N, nb_repeat=10):
    times = []
    for i in range(nb_repeat):
        buf = shamrock.backends.DeviceBuffer_f32()
        buf.resize(N)
        buf.fill(0)
        times.append(shamrock.algs.benchmark_reduction_sum(buf, N))
    return min(times), max(times), sum(times) / nb_repeat


def benchmark_f64(N, nb_repeat=10):
    times = []
    for i in range(nb_repeat):
        buf = shamrock.backends.DeviceBuffer_f64()
        buf.resize(N)
        buf.fill(0)
        times.append(shamrock.algs.benchmark_reduction_sum(buf, N))
    return min(times), max(times), sum(times) / nb_repeat


# %%
# Run the performance test for all parameters
def run_performance_sweep():
    # Define parameter ranges
    # logspace as array
    particle_counts = np.logspace(2, 7, 20).astype(int).tolist()

    # Initialize results matrix
    results_f32 = []
    results_f64 = []

    print(f"Particle counts: {particle_counts}")

    total_runs = len(particle_counts)
    current_run = 0

    for i, N in enumerate(particle_counts):
        current_run += 1

        print(
            f"[{current_run:2d}/{total_runs}] Running N={N:5d}...",
            end=" ",
        )

        start_time = time.time()
        min_time, max_time, mean_time = benchmark_f32(N)
        results_f32.append(min_time)
        min_time, max_time, mean_time = benchmark_f64(N)
        results_f64.append(min_time)
        elapsed = time.time() - start_time

        print(f"mean={mean_time:.3f}s (took {elapsed:.1f}s)")

    return particle_counts, results_f32, results_f64


# %%
# List current implementation
if not shamrock.algs.is_impl_set_reduction():
    shamrock.algs.autoselect_impl_reduction()
current_impl = shamrock.algs.get_current_impl_reduction()

print(current_impl)

# %%
# List all implementations available
all_default_impls = shamrock.algs.get_default_impl_list_reduction()

print(all_default_impls)

# %%
# Run the performance benchmarks for all implementations

dic_bench = {}
for impl in all_default_impls:
    shamrock.algs.set_impl_reduction(impl)

    impl_name = json.loads(impl)["implementation"]

    print(f"Running reduction performance benchmarks for {impl}...")

    # Run the performance sweep
    particle_counts, results_f32, results_f64 = run_performance_sweep()

    dic_bench[impl_name] = {
        "particle_counts": particle_counts,
        "results_f32": results_f32,
        "results_f64": results_f64,
    }


# %%
# Plot results (time)

print_ref = True
for label, item in dic_bench.items():
    if print_ref:
        Nobj = np.array(item["particle_counts"])
        Time100M = Nobj / 1e8
        plt.plot(
            item["particle_counts"],
            Time100M,
            color="grey",
            linestyle="-",
            alpha=0.7,
            label="100M obj/sec",
        )
        print_ref = False

    (line,) = plt.plot(item["particle_counts"], item["results_f64"], "--.", label=label + " (f64)")
    plt.plot(
        item["particle_counts"],
        item["results_f32"],
        ":",
        color=line.get_color(),
        label=label + " (f32)",
    )


plt.xlabel("Number of elements")
plt.ylabel("Time (s)")
plt.title("reduction performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend(fontsize=10)
plt.show()

# %%
# Plot results (bandwidth)

peak_bw_f32 = microbench_results["saxpy_f32"]
peak_bw_f64 = microbench_results["saxpy_f64"]

for label, item in dic_bench.items():
    Nobj = np.array(item["particle_counts"])
    last_x = item["particle_counts"][-1]

    Bytes_f64 = 8 * Nobj  # 1 read f64 (sizeof = 8)
    BW_f64 = Bytes_f64 / np.array(item["results_f64"])
    (line,) = plt.plot(item["particle_counts"], BW_f64, "--.", label=label + " (f64)")
    plt.text(
        last_x,
        BW_f64[-1],
        f"{BW_f64[-1] / 1e9:.2f} GB.s^-1",
        color=line.get_color(),
        va="bottom",
        ha="right",
    )

    Bytes_f32 = 4 * Nobj  # 1 read f32 (sizeof = 4)
    BW_f32 = Bytes_f32 / np.array(item["results_f32"])
    plt.plot(item["particle_counts"], BW_f32, ":", color=line.get_color(), label=label + " (f32)")
    plt.text(
        last_x,
        BW_f32[-1],
        f"{BW_f32[-1] / 1e9:.2f} GB.s^-1",
        color=line.get_color(),
        va="bottom",
        ha="right",
    )

plt.axhline(
    y=peak_bw_f64,
    color="black",
    linestyle=":",
    label=f"microbenchmark peak BW f64 ({peak_bw_f64 / 1e9:.2f} GB.s^-1)",
)
plt.axhline(
    y=peak_bw_f32,
    color="black",
    linestyle="--",
    label=f"microbenchmark peak BW f32 ({peak_bw_f32 / 1e9:.2f} GB.s^-1)",
)

plt.xlabel("Number of elements")
plt.ylabel("Bandwidth (B.s^-1)")
plt.title("reduction performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend(fontsize=10)
plt.show()
