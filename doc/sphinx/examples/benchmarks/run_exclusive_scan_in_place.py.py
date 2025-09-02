"""
is_all_true performance benchmarks
=================================

This example benchmarks the is_all_true performance for the different algorithms available in Shamrock
"""

# sphinx_gallery_multi_image = "single"

import random
import time

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


# %%
# Main benchmark functions
def benchmark_u32(N, nb_repeat=10):
    times = []
    for i in range(nb_repeat):
        buf = shamrock.backends.DeviceBuffer_u32()
        buf.resize(N)
        buf.fill(0)
        times.append(shamrock.algs.benchmark_scan_exclusive_sum_in_place(buf, N))
    return min(times), max(times), sum(times) / nb_repeat


# %%
# Run the performance test for all parameters
def run_performance_sweep():

    # Define parameter ranges
    # logspace as array
    particle_counts = np.logspace(2, 7, 20).astype(int).tolist()

    # Initialize results matrix
    results_u32 = []

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
        min_time, max_time, mean_time = benchmark_u32(N)
        results_u32.append(min_time)
        elapsed = time.time() - start_time

        print(f"mean={mean_time:.3f}s (took {elapsed:.1f}s)")

    return particle_counts, results_u32


# %%
# List all implementations available
all_algs = shamrock.algs.get_impl_list_scan_exclusive_sum_in_place()

print(all_algs)

# %%
# Run the performance benchmarks for all implementations
results = {}

for algname in all_algs:
    shamrock.algs.set_impl_scan_exclusive_sum_in_place(algname, "")

    print(f"Running reduction performance benchmarks for {algname}...")

    # Run the performance sweep
    particle_counts, results_u32 = run_performance_sweep()

    (line,) = plt.plot(particle_counts, results_u32, "--.", label=algname + " (u32)")


Nobj = np.array(particle_counts)
Time100M = Nobj / 1e8
plt.plot(particle_counts, Time100M, color="grey", linestyle="-", alpha=0.7, label="100M obj/sec")


plt.xlabel("Number of elements")
plt.ylabel("Time (s)")
plt.title("scan exclusive sum in place performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend()
plt.show()
