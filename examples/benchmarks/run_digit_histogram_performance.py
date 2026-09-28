"""
digit histogram performance benchmarks
======================================

This example benchmarks the digit histogram primitive (the upfront histogram of every radix
digit place of a key buffer, as used by the Onesweep radix sort) for the different digit sizes
available in Shamrock.
"""

# sphinx_gallery_multi_image = "single"

import random
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
# Digit sizes (in bits) supported by the digit histogram
radix_bits_list = [1, 2, 4, 8]


# %%
# Check the result against numpy on a small buffer
def check_digit_histogram(N, radix_bits):
    keys = shamrock.algs.mock_buffer_u32(42, N, 0, 2**32 - 1)
    hist = np.array(shamrock.algs.digit_histogram(keys, radix_bits, N).copy_to_stdvec())

    keys_np = np.array(keys.copy_to_stdvec(), dtype=np.uint64)
    nbuckets = 2**radix_bits
    npasses = 32 // radix_bits
    expected = np.concatenate(
        [
            np.bincount((keys_np >> (p * radix_bits)) & (nbuckets - 1), minlength=nbuckets)
            for p in range(npasses)
        ]
    )
    return np.array_equal(hist, expected), hist


for radix_bits in radix_bits_list:
    ok, _ = check_digit_histogram(100003, radix_bits)
    print(f"radix_bits={radix_bits} : result matches numpy = {ok}")


# %%
# Main benchmark function
def benchmark_u32(N, radix_bits, nb_repeat=10, max_cumulated_time=2.0):
    random.seed(111)

    times = []
    cumulated_time = 0.0
    for _ in range(nb_repeat):
        keys = shamrock.algs.mock_buffer_u32(random.randint(0, 1000000), N, 0, 2**32 - 1)

        t = shamrock.algs.benchmark_digit_histogram(keys, radix_bits, N)
        times.append(t)
        cumulated_time += t

        if cumulated_time > max_cumulated_time:
            break
    return min(times), max(times), sum(times) / len(times)


# %%
# Run the performance test for all parameters
def run_performance_sweep(radix_bits):
    # logspace as array, deliberately not restricted to powers of 2
    particle_counts = np.logspace(2, 7, 20).astype(int).tolist()

    results_u32 = []

    print(f"Particle counts: {particle_counts}")

    total_runs = len(particle_counts)
    current_run = 0

    for _, N in enumerate(particle_counts):
        current_run += 1

        print(
            f"[{current_run:2d}/{total_runs}] Running N={N:8d}...",
            end=" ",
        )

        start_time = time.time()
        min_time, max_time, mean_time = benchmark_u32(N, radix_bits)
        results_u32.append(min_time)
        elapsed = time.time() - start_time

        print(f"mean={mean_time:.3e}s (took {elapsed:.1f}s)")

    return particle_counts, results_u32


# %%
# Run the performance benchmarks for all digit sizes

results_by_radix = {}

for radix_bits in radix_bits_list:
    print(f"Running digit histogram performance benchmarks for radix_bits={radix_bits}...")

    particle_counts, results_u32 = run_performance_sweep(radix_bits)

    results_by_radix[f"radix_bits={radix_bits}"] = (particle_counts, results_u32)


# %%
# Plot the digit histogram performance benchmarks

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plot_data = {}
for i, (name, (particle_counts, results_u32)) in enumerate(results_by_radix.items()):
    plot_data[name] = {
        "x": particle_counts,
        "y": results_u32,
        "color": color_cycle[i % len(color_cycle)],
        "label": name + " (u32)",
        "linestyle": "--",
        "marker": ".",
    }


def before_plot(ax_plot):
    particle_counts = next(iter(results_by_radix.values()))[0]
    Nobj = np.array(particle_counts)
    Time1G = Nobj / 1e9
    ax_plot.plot(
        particle_counts, Time1G, color="grey", linestyle="-", alpha=0.7, label="1G obj/sec"
    )


make_std_bench_plot(
    plot_data,
    xlabel="Number of elements",
    ylabel="Time (s)",
    title="digit histogram performance benchmarks",
    end_label_fmt=lambda y: f"{y:.2e} s",
    before_plot_func=before_plot,
)
plt.show()

# %%
# Plot the digit histogram performance benchmarks (bandwidth)
# The keys are read only once whatever the number of digit places, the histogram itself being
# negligible in size.

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plot_data = {}
for i, (name, (particle_counts, results_u32)) in enumerate(results_by_radix.items()):
    Nobj = np.array(particle_counts)
    Bytes = 4 * Nobj  # 1 u32 key read per element (sizeof = 4)
    BW = Bytes / np.array(results_u32)
    plot_data[name] = {
        "x": particle_counts,
        "y": BW,
        "color": color_cycle[i % len(color_cycle)],
        "label": name + " (u32)",
        "linestyle": "--",
        "marker": ".",
    }

make_std_bench_plot(
    plot_data,
    xlabel="Number of elements",
    ylabel="Bandwidth (B.s^-1)",
    title="digit histogram performance benchmarks",
    end_label_fmt=lambda y: f"{y / 1e9:.2f} GB.s^-1",
)
plt.show()

# %%
# Plot the histograms of the 4 digit places of 8 bits of uniformly distributed keys

_, hist = check_digit_histogram(1000000, 8)

plt.figure()
for p in range(4):
    plt.plot(hist[p * 256 : (p + 1) * 256], label=f"digit place {p} (bits {8 * p}-{8 * p + 7})")
plt.xlabel("digit value")
plt.ylabel("count")
plt.title("digit histogram (radix_bits=8, 1M uniform u32 keys)")
plt.legend()
plt.show()
