"""
sort by keys performance benchmarks
====================================

This example benchmarks the sort by keys (general, non power-of-2 length) performance for
the different algorithms available in Shamrock, as well as the sort_by_key_pow2_len
(power-of-2 length only) performance in a second figure.
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

results_by_impl = {}

for impl in all_default_impls:
    shamrock.algs.set_impl_sort_by_keys(impl)

    impl_name = json.loads(impl)["implementation"]

    print(f"Running sort by keys performance benchmarks for {impl}...")

    # Run the performance sweep
    particle_counts, results_u32 = run_performance_sweep()

    results_by_impl[impl_name] = (particle_counts, results_u32)


# %%
# Main benchmark function for sort_by_key_pow2_len (power-of-2 length only)
def benchmark_u32_pow2_len(N, nb_repeat=10, max_cumulated_time=2.0):
    random.seed(111)

    times = []
    cumulated_time = 0.0
    for _ in range(nb_repeat):
        keys = shamrock.algs.mock_buffer_u32(random.randint(0, 1000000), N, 0, 1000000)

        values = shamrock.backends.DeviceBuffer_u32()
        values.resize(N)
        values.copy_from_stdvec(list(range(N)))

        t = shamrock.algs.benchmark_sort_by_key_pow2_len(keys, values, N)
        times.append(t)
        cumulated_time += t

        if cumulated_time > max_cumulated_time:
            break
    return min(times), max(times), sum(times) / len(times)


# %%
# Run the performance test for all parameters, restricted to power-of-2 lengths
def run_performance_sweep_pow2_len():
    # power-of-2 lengths only, spanning roughly the same range as the general benchmark above
    particle_counts = [2**k for k in range(7, 24)]

    results_u32 = []

    print(f"Particle counts (pow2 len): {particle_counts}")

    total_runs = len(particle_counts)
    current_run = 0

    for _, N in enumerate(particle_counts):
        current_run += 1

        print(
            f"[{current_run:2d}/{total_runs}] Running N={N:8d}...",
            end=" ",
        )

        start_time = time.time()
        min_time, max_time, mean_time = benchmark_u32_pow2_len(N)
        results_u32.append(min_time)
        elapsed = time.time() - start_time

        print(f"mean={mean_time:.3f}s (took {elapsed:.1f}s)")

    return particle_counts, results_u32


# %%
# List current implementation
if not shamrock.algs.is_impl_set_sort_by_key_pow2_len():
    shamrock.algs.autoselect_impl_sort_by_key_pow2_len()

current_impl_pow2_len = shamrock.algs.get_current_impl_sort_by_key_pow2_len()

print(current_impl_pow2_len)

# %%
# List all implementations available
all_default_impls_pow2_len = shamrock.algs.get_default_impl_list_sort_by_key_pow2_len()

print(all_default_impls_pow2_len)

# %%
# Run the performance benchmarks for all implementations (power-of-2 lengths only)

results_by_impl_pow2_len = {}

for impl in all_default_impls_pow2_len:
    shamrock.algs.set_impl_sort_by_key_pow2_len(impl)

    impl_name = json.loads(impl)["implementation"]

    print(f"Running sort by key (pow2 len) performance benchmarks for {impl}...")

    # Run the performance sweep
    particle_counts_pow2, results_u32_pow2 = run_performance_sweep_pow2_len()

    results_by_impl_pow2_len[impl_name] = (particle_counts_pow2, results_u32_pow2)


# %%
# Plot the sort by keys performance benchmarks (first figure)

for impl_name, (particle_counts, results_u32) in results_by_impl.items():
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

# %%
# Plot the sort by keys performance benchmarks (bandwidth)
# Note: no microbenchmark peak-bandwidth reference here, since a sort will never
# reach the raw memory-bandwidth ceiling.

for impl_name, (particle_counts, results_u32) in results_by_impl.items():
    Nobj = np.array(particle_counts)
    Bytes = 2 * 4 * Nobj  # 2 u32 moved per element, key + value (sizeof = 4)
    BW = Bytes / np.array(results_u32)
    (line,) = plt.plot(particle_counts, BW, "--.", label=impl_name + " (u32)")

    last_x = particle_counts[-1]
    plt.text(
        last_x,
        BW[-1],
        f"{BW[-1] / 1e9:.2f} GB.s^-1",
        color=line.get_color(),
        va="bottom",
        ha="right",
    )

plt.xlabel("Number of elements")
plt.ylabel("Bandwidth (B.s^-1)")
plt.title("sort by keys performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend()
plt.show()

# %%
# Plot the sort by key (power-of-2 length) performance benchmarks (second figure)

plt.figure()

for impl_name, (particle_counts_pow2, results_u32_pow2) in results_by_impl_pow2_len.items():
    plt.plot(particle_counts_pow2, results_u32_pow2, "--.", label=impl_name + " (u32)")

Nobj_pow2 = np.array(particle_counts_pow2)
Time100M_pow2 = Nobj_pow2 / 1e8
plt.plot(
    particle_counts_pow2,
    Time100M_pow2,
    color="grey",
    linestyle="-",
    alpha=0.7,
    label="100M obj/sec",
)

plt.xlabel("Number of elements")
plt.ylabel("Time (s)")
plt.title("sort by key (power-of-2 length) performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend()
plt.show()

# %%
# Plot the sort by key (power-of-2 length) performance benchmarks (bandwidth)
# Note: no microbenchmark peak-bandwidth reference here, since a sort will never
# reach the raw memory-bandwidth ceiling.

for impl_name, (particle_counts_pow2, results_u32_pow2) in results_by_impl_pow2_len.items():
    Nobj_pow2 = np.array(particle_counts_pow2)
    Bytes_pow2 = 2 * 4 * Nobj_pow2  # 2 u32 moved per element, key + value (sizeof = 4)
    BW_pow2 = Bytes_pow2 / np.array(results_u32_pow2)
    (line,) = plt.plot(particle_counts_pow2, BW_pow2, "--.", label=impl_name + " (u32)")

    last_x_pow2 = particle_counts_pow2[-1]
    plt.text(
        last_x_pow2,
        BW_pow2[-1],
        f"{BW_pow2[-1] / 1e9:.2f} GB.s^-1",
        color=line.get_color(),
        va="bottom",
        ha="right",
    )

plt.xlabel("Number of elements")
plt.ylabel("Bandwidth (B.s^-1)")
plt.title("sort by key (power-of-2 length) performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend()
plt.show()

# %%
# Plot both benchmarks overlaid (third figure), using different markers to distinguish the
# generic (non power-of-2 length) and the pow2_len-only implementations

plt.figure()

for impl_name, (particle_counts, results_u32) in results_by_impl.items():
    plt.plot(particle_counts, results_u32, "--.", label=impl_name + " (generic)")

for impl_name, (particle_counts_pow2, results_u32_pow2) in results_by_impl_pow2_len.items():
    plt.plot(particle_counts_pow2, results_u32_pow2, "--x", label=impl_name + " (pow2_len)")

Nobj_all = np.array(sorted(set(particle_counts) | set(particle_counts_pow2)))
Time100M_all = Nobj_all / 1e8
plt.plot(Nobj_all, Time100M_all, color="grey", linestyle="-", alpha=0.7, label="100M obj/sec")

plt.xlabel("Number of elements")
plt.ylabel("Time (s)")
plt.title("sort by keys (general vs pow2 len)")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend()
plt.show()
