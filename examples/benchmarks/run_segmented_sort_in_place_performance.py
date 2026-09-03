"""
segmented sort in place performance benchmarks
================================

This example benchmarks the segmented sort in place performance for the different algorithms available in Shamrock
"""

# sphinx_gallery_multi_image = "single"

import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
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
# Main benchmark functions
def benchmark_u32_balanced(N, slice_size, nb_repeat=10):
    assert N % slice_size == 0, "N must be divisible by slice_size"

    random.seed(111)

    times = []
    for _ in range(nb_repeat):
        buf = shamrock.algs.mock_buffer_u32(random.randint(0, 1000000), N, 0, 1000000)

        offsets_values = [i * slice_size for i in range((N // slice_size) + 1)]

        offsets = shamrock.backends.DeviceBuffer_u32()
        offsets.resize(len(offsets_values))
        offsets.copy_from_stdvec(offsets_values)

        times.append(shamrock.algs.benchmark_segmented_sort_in_place(buf, offsets))
    return min(times), max(times), sum(times) / nb_repeat


# %%
# Run performance sweep
particle_counts = 2**20  # = 1.048.576


def run_performance_sweep_balanced():
    # Define parameter ranges
    # logspace as array

    slice_sizes = [2**i for i in range(14)]

    # Initialize results matrix
    results_u32_balanced = []

    print(f"Particle counts: {particle_counts}")
    print(f"Slice sizes: {slice_sizes}")

    total_runs = len(slice_sizes)
    current_run = 0

    for i, slice_size in enumerate(slice_sizes):
        current_run += 1

        print(
            f"[{current_run:2d}/{total_runs}] Running N={particle_counts:5d}, slice_size={slice_size:5d}...",
            end=" ",
        )

        start_time = time.time()
        min_time, max_time, mean_time = benchmark_u32_balanced(particle_counts, slice_size)
        results_u32_balanced.append(mean_time)

        elapsed = time.time() - start_time

        print(f"mean={mean_time:.3f}s (took {elapsed:.1f}s)")

    return particle_counts, slice_sizes, results_u32_balanced


# %%
# List current implementation
current_impl = shamrock.algs.get_current_impl_segmented_sort_in_place()

print(current_impl)

# %%
# List all implementations available
all_default_impls = shamrock.algs.get_default_impl_list_segmented_sort_in_place()

print(all_default_impls)

# %%
# Run the performance benchmarks for all implementations

dic_bench = {}
for impl in all_default_impls:
    shamrock.algs.set_impl_segmented_sort_in_place(impl)

    impl_name = json.loads(impl)["implementation"]

    print(f"Running segmented sort in place performance benchmarks for {impl}...")

    # Run the performance sweep
    particle_counts, slice_sizes, results_u32_balanced = run_performance_sweep_balanced()

    dic_bench[impl_name] = {
        "slice_sizes": slice_sizes,
        "results_u32_balanced": results_u32_balanced,
    }


# %%
# Plot results

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plot_data = {}
for i, (label, item) in enumerate(dic_bench.items()):
    color = color_cycle[i % len(color_cycle)]
    plot_data[label + " (u32)"] = {
        "x": item["slice_sizes"],
        "y": item["results_u32_balanced"],
        "color": color,
        "label": label + " (u32)",
        "linestyle": "--",
        "marker": ".",
    }


def before_plot(ax_plot):
    slice_sizes = next(iter(dic_bench.values()))["slice_sizes"]
    Time100M = particle_counts / 1e8
    ax_plot.plot(
        slice_sizes,
        [Time100M for _ in slice_sizes],
        color="grey",
        linestyle="-",
        alpha=0.7,
        label="100M obj/sec",
    )


make_std_bench_plot(
    plot_data,
    xlabel="Slice size",
    ylabel="Time (s)",
    title="segmented sort in place benchmarks, N=" + str(particle_counts),
    end_label_fmt=lambda y: f"{y:.2e} s",
    before_plot_func=before_plot,
)
plt.show()
