"""
DTT performance benchmarks
=================================

This example benchmarks the DTT performance for the different algorithms available in Shamrock
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
bounding_box = shamrock.math.AABB_f64_3((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def benchmark_dtt_core(N, theta_crit, compression_level, nb_repeat=10):
    times = []
    random.seed(111)
    for i in range(nb_repeat):
        positions = shamrock.algs.mock_buffer_f64_3(
            random.randint(0, 1000000), N, bounding_box.lower, bounding_box.upper
        )
        tree = shamrock.tree.CLBVH_u64_f64_3()
        tree.rebuild_from_positions(positions, bounding_box, compression_level)
        times.append(shamrock.tree.benchmark_clbvh_dual_tree_traversal(tree, theta_crit) * 1000)
    return times


def benchmark_dtt(N, theta_crit, compression_level, nb_repeat=10):
    times = benchmark_dtt_core(N, theta_crit, compression_level, nb_repeat)
    return min(times), max(times), sum(times) / nb_repeat


# %%
# Run the performance test for all parameters
def run_performance_sweep(compression_level, threshold_run):

    # Define parameter ranges
    # logspace as array
    particle_counts = np.logspace(2, 7, 10).astype(int).tolist()
    theta_crits = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Initialize results matrix
    results_mean = np.zeros((len(theta_crits), len(particle_counts)))
    results_min = np.zeros((len(theta_crits), len(particle_counts)))
    results_max = np.zeros((len(theta_crits), len(particle_counts)))

    print(f"Particle counts: {particle_counts}")
    print(f"Theta_crit values: {theta_crits}")
    print(f"Compression level: {compression_level}")

    total_runs = len(particle_counts) * len(theta_crits)
    current_run = 0

    for i, theta_crit in enumerate(theta_crits):
        for j, N in enumerate(particle_counts):
            current_run += 1

            # Check if we should skip this benchmark based on the criterion
            criterion_value = N / (theta_crit**3)
            if criterion_value > threshold_run:
                print(
                    f"[{current_run:2d}/{total_runs}] Skipping N={N:5d}, theta_crit={theta_crit:.1f} (N/theta³={criterion_value:.0f} > 100000)"
                )
                results_mean[i, j] = np.nan
                results_min[i, j] = np.nan
                results_max[i, j] = np.nan
                continue

            print(
                f"[{current_run:2d}/{total_runs}] Running N={N:5d}, theta_crit={theta_crit:.1f} (N/theta³={criterion_value:.0f})...",
                end=" ",
            )

            start_time = time.time()
            min_time, max_time, mean_time = benchmark_dtt(N, theta_crit, compression_level)
            elapsed = time.time() - start_time

            results_mean[i, j] = mean_time
            results_min[i, j] = min_time
            results_max[i, j] = max_time

            print(f"mean={mean_time:.3f}ms (took {elapsed:.1f}s)")

    return particle_counts, theta_crits, results_mean, results_min, results_max


# %%
# Create checkerboard plot with execution times and relative performance to reference algorithm
def create_checkerboard_plot(
    particle_counts,
    theta_crits,
    results_data,
    compression_level,
    algname,
    max_axis_value,
    reference_data,
):
    """Create checkerboard plot with execution times"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate relative performance compared to reference algorithm
    # results_data / reference_data gives the ratio (>1 means slower, <1 means faster)
    relative_performance = results_data / reference_data

    # Create the heatmap with relative performance values
    # Create a masked array to handle NaN values (skipped benchmarks) as white
    masked_relative = np.ma.masked_invalid(relative_performance)

    # Use a diverging colormap: red for better performance (<1), green for worse (>1)
    # RdYlGn_r (reversed) has green for high values (worse) and red for low values (better)
    cmap = plt.cm.RdYlGn_r.copy()  # Green for >1 (slower), Red for <1 (faster)
    cmap.set_bad(color="white")  # Set NaN values to white

    # Set the color scale limits for relative performance
    vmin = 0.5
    vmax = 1.5

    im = ax.imshow(
        masked_relative, cmap=cmap, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax
    )

    # Set ticks and labels
    ax.set_xticks(range(len(particle_counts)))
    ax.set_yticks(range(len(theta_crits)))
    ax.set_xticklabels([f"{N//1000}k" if N >= 1000 else str(N) for N in particle_counts])
    ax.set_yticklabels([f"{theta:.1f}" for theta in theta_crits])

    # Add labels
    ax.set_xlabel("Particle Count")
    ax.set_ylabel("Theta Critical")
    ax.set_title(
        f"Dual Tree Traversal Performance\n(Colors: Relative to Reference, Text: Absolute Time in ms)\ncompression level = {compression_level} algorithm = {algname}",
        pad=20,
    )

    # Add text annotations showing the values
    for i in range(len(theta_crits)):
        for j in range(len(particle_counts)):
            value = results_data[i, j]

            if np.isnan(value):
                # For skipped benchmarks, show "SKIPPED" in black on white background
                # ax.text(j, i, 'SKIPPED', ha='center', va='center',
                #       color='black', fontweight='bold', fontsize=8)
                pass
            else:
                perf = relative_performance[i, j]
                text_color = "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}ms\n{perf:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=10,
                )

    # Add colorbar for relative performance
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Relative performance (time / reference time)")
    cbar.ax.tick_params(labelsize=10)

    # Add custom tick labels for better interpretation
    tick_positions = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    cbar.set_ticks([pos for pos in tick_positions if vmin <= pos <= vmax])

    # Improve layout
    plt.tight_layout()

    # Add grid for better readability
    ax.set_xticks(np.arange(len(particle_counts)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(theta_crits)) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1, alpha=0.3)

    return fig, ax


# %%
# List current implementation
current_impl = shamrock.tree.get_current_impl_clbvh_dual_tree_traversal_impl()

print(current_impl)

# %%
# List all implementations available
all_default_impls = shamrock.tree.get_default_impl_list_clbvh_dual_tree_traversal()

print(all_default_impls)

# %%
# Run the performance benchmarks for all implementations
results = {}

for default_impl in all_default_impls:
    shamrock.tree.set_impl_clbvh_dual_tree_traversal(default_impl.impl_name, default_impl.params)

    print(f"Running DTT performance benchmarks for {default_impl.impl_name}...")

    compression_level = 4

    threshold_run = 100000
    # Run the performance sweep
    particle_counts, theta_crits, results_mean, results_min, results_max = run_performance_sweep(
        compression_level, threshold_run
    )

    results[default_impl.impl_name + " " + default_impl.params] = {
        "particle_counts": particle_counts,
        "theta_crits": theta_crits,
        "results_mean": results_mean,
        "results_min": results_min,
        "results_max": results_max,
        "name": default_impl.impl_name + " " + default_impl.params,
    }

# %%
# Plot the performance benchmarks for all implementations
dump_folder = "_to_trash"

import os

# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)

ref_key = "reference "
largest_refalg_value = np.nanmax(results[ref_key]["results_min"])

i = 0
# iterate over the results
for k, v in results.items():

    # Get the results for this algorithm
    particle_counts = v["particle_counts"]
    theta_crits = v["theta_crits"]
    results_min = v["results_min"]

    # Get reference algorithm results for comparison
    reference_min = results[ref_key]["results_min"]

    # Create and display the plot
    fig, ax = create_checkerboard_plot(
        particle_counts,
        theta_crits,
        results_min,
        compression_level,
        v["name"],
        largest_refalg_value,
        reference_min,
    )

    plt.savefig(f"{dump_folder}/benchmark-dtt-performance-{i}.pdf")
    i += 1

plt.show()
