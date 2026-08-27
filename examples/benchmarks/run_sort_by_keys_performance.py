"""
sort by keys performance benchmarks
====================================

This example benchmarks the sort by keys (general, non power-of-2 length) performance for
the different algorithms available in Shamrock
"""

# sphinx_gallery_multi_image = "single"

import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np

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
# Main benchmark functions
def benchmark_u32(N, nb_repeat=10, max_cumulated_time=2.0):
    random.seed(111)

    times = []
    cumulated_time = 0.0
    for _ in range(nb_repeat):
        keys = shamrock.algs.mock_buffer_u32(random.randint(0, 1000000), N, 0, 1000000)

        values = shamrock.backends.DeviceBuffer_u32()
        values.resize(N)
        values.copy_from_stdvec(list(range(N)))

        t = shamrock.algs.benchmark_sort_by_keys(keys, values, N)
        times.append(t)
        cumulated_time += t

        if cumulated_time > max_cumulated_time:
            break
    return min(times), max(times), sum(times) / len(times)


# %%
# Run the performance test for all parameters
def run_performance_sweep():
    # Define parameter ranges
    # logspace as array, deliberately not restricted to powers of 2
    particle_counts = np.logspace(2, 7, 20).astype(int).tolist()

    # Initialize results matrix
    results_u32 = []

    print(f"Particle counts: {particle_counts}")

    total_runs = len(particle_counts)
    current_run = 0

    for _, N in enumerate(particle_counts):
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
# List current implementation
if not shamrock.algs.is_impl_set_sort_by_keys():
    shamrock.algs.autoselect_impl_sort_by_keys()

current_impl = shamrock.algs.get_current_impl_sort_by_keys()

print(current_impl)

# %%
# List all implementations available
all_default_impls = shamrock.algs.get_default_impl_list_sort_by_keys()

print(all_default_impls)

# %%
# Run the performance benchmarks for all implementations

for impl in all_default_impls:
    shamrock.algs.set_impl_sort_by_keys(impl)

    impl_name = json.loads(impl)["implementation"]

    print(f"Running sort by keys performance benchmarks for {impl}...")

    # Run the performance sweep
    particle_counts, results_u32 = run_performance_sweep()

    plt.plot(particle_counts, results_u32, "--.", label=impl_name + " (u32)")


Nobj = np.array(particle_counts)
Time100M = Nobj / 1e8
plt.plot(particle_counts, Time100M, color="grey", linestyle="-", alpha=0.7, label="100M obj/sec")


plt.xlabel("Number of elements")
plt.ylabel("Time (s)")
plt.title("sort by keys performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend()
plt.show()
