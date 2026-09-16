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
from shamrock.utils.plot import make_std_bench_plot

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

    dic_bench[impl_name] = {"particle_counts": particle_counts, "results_u32": results_u32}


# %%
# Plot results (time)

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plot_data = {}
for i, (label, item) in enumerate(dic_bench.items()):
    color = color_cycle[i % len(color_cycle)]
    plot_data[label + " (u32)"] = {
        "x": item["particle_counts"],
        "y": item["results_u32"],
        "color": color,
        "label": label + " (u32)",
        "linestyle": "--",
        "marker": ".",
    }


def before_plot(ax_plot):
    particle_counts = next(iter(dic_bench.values()))["particle_counts"]
    Nobj = np.array(particle_counts)
    Time100M = Nobj / 1e8
    ax_plot.plot(
        particle_counts,
        Time100M,
        color="grey",
        linestyle="-",
        alpha=0.7,
        label="100M obj/sec",
    )


make_std_bench_plot(
    plot_data,
    xlabel="Number of elements",
    ylabel="Time (s)",
    title="ex-scan in place performance benchmarks",
    end_label_fmt=lambda y: f"{y:.2e} s",
    before_plot_func=before_plot,
)
plt.show()


# %%
# Plot results (bandwidth)

peak_bw = microbench_results["saxpy_f32"]

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plot_data = {}
for i, (label, item) in enumerate(dic_bench.items()):
    color = color_cycle[i % len(color_cycle)]
    Nobj = np.array(item["particle_counts"])

    Bytes = 2 * 4 * Nobj  # 1 read, 1 write u32 (sizeof = 4)
    BW = Bytes / np.array(item["results_u32"])
    plot_data[label + " (u32)"] = {
        "x": item["particle_counts"],
        "y": BW,
        "color": color,
        "label": label + " (u32)",
        "linestyle": "--",
        "marker": "x",
    }


def before_plot(ax_plot):
    ax_plot.axhline(
        y=peak_bw,
        color="black",
        linestyle=":",
        label="microbenchmark peak BW f32",
    )


make_std_bench_plot(
    plot_data,
    xlabel="Number of elements",
    ylabel="Bandwidth (B.s^-1)",
    title="ex-scan in place performance benchmarks",
    end_label_fmt=lambda y: f"{y / 1e9:.2f} GB.s^-1",
    before_plot_func=before_plot,
)
plt.show()
