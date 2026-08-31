"""
in place ex-scan performance benchmarks
=======================================

This example benchmarks the scan exclusive sum in place performance for the different algorithms available in Shamrock
"""

# sphinx_gallery_multi_image = "single"

import json
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
# Recover microbenchmark results
microbench_results = shamrock.sys.get_microbench_results(allow_run=True)
if len(microbench_results) == 0:
    print("no microbench results, please run with --benchmark-mpi")
    raise ValueError("no microbench results")


# %%
# Main benchmark functions
def benchmark_u32(N, nb_repeat=10):
    times = []
    for _ in range(nb_repeat):
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
if not shamrock.algs.is_impl_set_scan_exclusive_sum_in_place():
    shamrock.algs.autoselect_impl_scan_exclusive_sum_in_place()

current_impl = shamrock.algs.get_current_impl_scan_exclusive_sum_in_place()

print(current_impl)

# %%
# List all implementations available
all_default_impls = shamrock.algs.get_default_impl_list_scan_exclusive_sum_in_place()

print(all_default_impls)

# %%
# Run the performance benchmarks for all implementations

dic_bench = {}
for impl in all_default_impls:
    shamrock.algs.set_impl_scan_exclusive_sum_in_place(impl)

    impl_name = json.loads(impl)["implementation"]

    print(f"Running ex-scan in place performance benchmarks for {impl}...")

    # Run the performance sweep
    particle_counts, results_u32 = run_performance_sweep()

    label = impl_name + " (u32)"

    dic_bench[label] = {"particle_counts": particle_counts, "results_u32": results_u32}


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

    plt.plot(item["particle_counts"], item["results_u32"], "--.", label=label)


plt.xlabel("Number of elements")
plt.ylabel("Time (s)")
plt.title("ex-scan in place performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend()
plt.show()

# %%
# Plot results (bandwidth)

peak_bw = microbench_results["saxpy_f32"]

for label, item in dic_bench.items():
    Nobj = np.array(item["particle_counts"])
    Bytes = 2 * 4 * Nobj  # 1 read, 1 write u32 (sizeof = 4)
    BW = Bytes / item["results_u32"]
    (line,) = plt.plot(item["particle_counts"], BW, "--.", label=label)

    last_x = item["particle_counts"][-1]
    last_bw_GBs = BW[-1] / 1e9
    plt.text(
        last_x,
        BW[-1],
        f"{last_bw_GBs:.2f} GB.s^-1",
        color=line.get_color(),
        va="bottom",
        ha="right",
    )

plt.axhline(
    y=peak_bw,
    color="black",
    linestyle=":",
    label=f"microbenchmark peak BW ({peak_bw / 1e9:.2f} GB.s^-1)",
)

plt.xlabel("Number of elements")
plt.ylabel("Bandwidth (B.s^-1)")
plt.title("ex-scan in place performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend()
plt.show()
