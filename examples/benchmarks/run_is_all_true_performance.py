"""
is_all_true performance benchmarks
=================================

This example benchmarks the is_all_true performance for the different algorithms available in Shamrock
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


def benchmark_is_all_true_random(N, nb_repeat=10):
    times = []
    for i in range(nb_repeat):
        random.seed(111)
        buf = shamrock.algs.mock_buffer_u8(random.randint(0, 1000000), N, 0, 1)
        times.append(shamrock.algs.benchmark_is_all_true(buf, N))
    return min(times), max(times), sum(times) / nb_repeat


def benchmark_is_all_true_ones(N, nb_repeat=10):
    times = []
    for i in range(nb_repeat):
        buf = shamrock.backends.DeviceBuffer_u8()
        buf.resize(N)
        buf.fill(1)
        times.append(shamrock.algs.benchmark_is_all_true(buf, N))
    return min(times), max(times), sum(times) / nb_repeat


def benchmark_is_all_true_zeros(N, nb_repeat=10):
    times = []
    for i in range(nb_repeat):
        buf = shamrock.backends.DeviceBuffer_u8()
        buf.resize(N)
        buf.fill(0)
        times.append(shamrock.algs.benchmark_is_all_true(buf, N))
    return min(times), max(times), sum(times) / nb_repeat


# %%
# Run the performance test for all parameters
def run_performance_sweep():
    # Define parameter ranges
    # logspace as array
    particle_counts = np.logspace(2, 7, 20).astype(int).tolist()

    # Initialize results matrix
    results_random = []
    results_ones = []
    results_zeros = []

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
        min_time, max_time, mean_time = benchmark_is_all_true_random(N)
        results_random.append(mean_time)
        min_time, max_time, mean_time = benchmark_is_all_true_ones(N)
        results_ones.append(mean_time)
        min_time, max_time, mean_time = benchmark_is_all_true_zeros(N)
        results_zeros.append(mean_time)
        elapsed = time.time() - start_time

        print(f"mean={mean_time:.3f}s (took {elapsed:.1f}s)")

    return particle_counts, results_random, results_ones, results_zeros


# %%
# List current implementation
if not shamrock.algs.is_impl_set_is_all_true():
    shamrock.algs.autoselect_impl_is_all_true()

current_impl = shamrock.algs.get_current_impl_is_all_true()

print(current_impl)

# %%
# List all implementations available
all_default_impls = shamrock.algs.get_default_impl_list_is_all_true()

print(all_default_impls)

# %%
# Run the performance benchmarks for all implementations

dic_bench = {}
for impl in all_default_impls:
    shamrock.algs.set_impl_is_all_true(impl)

    impl_name = json.loads(impl)["implementation"]

    print(f"Running is_all_true performance benchmarks for {impl}...")

    # Run the performance sweep
    particle_counts, results_random, results_ones, results_zeros = run_performance_sweep()

    dic_bench[impl_name] = {
        "particle_counts": particle_counts,
        "results_random": results_random,
        "results_ones": results_ones,
        "results_zeros": results_zeros,
    }


# %%
# Plot results

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plot_data = {}
for i, (label, item) in enumerate(dic_bench.items()):
    color = color_cycle[i % len(color_cycle)]
    plot_data[label + " (random set)"] = {
        "x": item["particle_counts"],
        "y": item["results_random"],
        "color": color,
        "label": label + " (random set)",
        "linestyle": "--",
        "marker": None,
    }
    plot_data[label + " (all ones)"] = {
        "x": item["particle_counts"],
        "y": item["results_ones"],
        "color": color,
        "label": label + " (all ones)",
        "linestyle": "--",
        "marker": "+",
    }
    plot_data[label + " (all zeros)"] = {
        "x": item["particle_counts"],
        "y": item["results_zeros"],
        "color": color,
        "label": label + " (all zeros)",
        "linestyle": "--",
        "marker": "o",
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
    title="is_all_true performance benchmarks",
    end_label_fmt=lambda y: f"{y:.2e} s",
    before_plot_func=before_plot,
)
plt.show()
