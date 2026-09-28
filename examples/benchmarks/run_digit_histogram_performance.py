"""
digit histogram performance benchmarks
======================================

This example benchmarks the digit histogram primitive (the upfront histogram of every radix
digit place of a key buffer, as used by the Onesweep radix sort) for the different digit sizes
available in Shamrock, on 3 key distributions:

* ``zeros`` : every key is 0, so every digit place of every key lands in the same bin (worst case
  for the atomic contention on the histogram bins)
* ``shuffled iota`` : a random permutation of ``0 .. N-1``, the low digit places are uniform while
  the high ones are concentrated on the few lowest bins
* ``random`` : uniformly distributed keys over the full u32 range, every digit place is uniform
"""

# sphinx_gallery_multi_image = "single"

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
# Key distributions to benchmark
key_cases = ["zeros", "shuffled iota", "random"]


def make_keys(case, N, seed=111):
    if case == "zeros":
        keys = shamrock.backends.DeviceBuffer_u32()
        keys.resize(N)
        keys.fill(0)
        return keys
    elif case == "shuffled iota":
        rng = np.random.default_rng(seed)
        keys = shamrock.backends.DeviceBuffer_u32()
        keys.resize(N)
        keys.copy_from_stdvec(rng.permutation(N).astype(np.uint32).tolist())
        return keys
    elif case == "random":
        return shamrock.algs.mock_buffer_u32(seed, N, 0, 2**32 - 1)
    else:
        raise ValueError(f"unknown key case {case}")


# %%
# Check the result against numpy
def reference_digit_histogram(keys, radix_bits):
    keys_np = np.array(keys.copy_to_stdvec(), dtype=np.uint64)
    nbuckets = 2**radix_bits
    npasses = 32 // radix_bits
    return np.concatenate(
        [
            np.bincount((keys_np >> (p * radix_bits)) & (nbuckets - 1), minlength=nbuckets)
            for p in range(npasses)
        ]
    )


def compute_digit_histogram(keys, radix_bits):
    N = keys.get_size()
    return np.array(shamrock.algs.digit_histogram(keys, radix_bits, N).copy_to_stdvec())


for case in key_cases:
    keys = make_keys(case, 100003)
    for radix_bits in radix_bits_list:
        ok = np.array_equal(
            compute_digit_histogram(keys, radix_bits), reference_digit_histogram(keys, radix_bits)
        )
        print(f"{case:>13s}, radix_bits={radix_bits} : result matches numpy = {ok}")


# %%
# Main benchmark function
def benchmark_u32(keys, radix_bits, nb_repeat=10, max_cumulated_time=2.0):
    N = keys.get_size()

    times = []
    cumulated_time = 0.0
    for _ in range(nb_repeat):
        t = shamrock.algs.benchmark_digit_histogram(keys, radix_bits, N)
        times.append(t)
        cumulated_time += t

        if cumulated_time > max_cumulated_time:
            break
    return min(times), max(times), sum(times) / len(times)


# %%
# Run the performance test for all parameters
# (the keys of a given case and size are generated once and reused for every digit size)
def run_performance_sweep():
    # logspace as array, deliberately not restricted to powers of 2
    particle_counts = np.logspace(2, 7, 20).astype(int).tolist()

    results = {case: {radix_bits: [] for radix_bits in radix_bits_list} for case in key_cases}

    print(f"Particle counts: {particle_counts}")

    total_runs = len(particle_counts)

    for current_run, N in enumerate(particle_counts, start=1):
        for case in key_cases:
            keys = make_keys(case, N)
            for radix_bits in radix_bits_list:
                print(
                    f"[{current_run:2d}/{total_runs}] Running N={N:8d}, {case:>13s}, "
                    f"radix_bits={radix_bits}...",
                    end=" ",
                )

                start_time = time.time()
                min_time, max_time, mean_time = benchmark_u32(keys, radix_bits)
                results[case][radix_bits].append(min_time)
                elapsed = time.time() - start_time

                print(f"mean={mean_time:.3e}s (took {elapsed:.1f}s)")

    return particle_counts, results


# %%
# Run the performance benchmarks for all key distributions and digit sizes

particle_counts, results = run_performance_sweep()


# %%
# Plot the digit histogram performance benchmarks, one figure per key distribution

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def before_plot(ax_plot):
    Nobj = np.array(particle_counts)
    Time1G = Nobj / 1e9
    ax_plot.plot(
        particle_counts, Time1G, color="grey", linestyle="-", alpha=0.7, label="1G obj/sec"
    )


for case in key_cases:
    plot_data = {}
    for i, radix_bits in enumerate(radix_bits_list):
        plot_data[f"radix_bits={radix_bits}"] = {
            "x": particle_counts,
            "y": results[case][radix_bits],
            "color": color_cycle[i % len(color_cycle)],
            "label": f"radix_bits={radix_bits} (u32)",
            "linestyle": "--",
            "marker": ".",
        }

    make_std_bench_plot(
        plot_data,
        xlabel="Number of elements",
        ylabel="Time (s)",
        title=f"digit histogram performance benchmarks ({case} keys)",
        end_label_fmt=lambda y: f"{y:.2e} s",
        before_plot_func=before_plot,
    )
    plt.show()

# %%
# Plot the digit histogram performance benchmarks (bandwidth), one figure per key distribution.
# The keys are read only once whatever the number of digit places, the histogram itself being
# negligible in size.

for case in key_cases:
    plot_data = {}
    for i, radix_bits in enumerate(radix_bits_list):
        Nobj = np.array(particle_counts)
        Bytes = 4 * Nobj  # 1 u32 key read per element (sizeof = 4)
        BW = Bytes / np.array(results[case][radix_bits])
        plot_data[f"radix_bits={radix_bits}"] = {
            "x": particle_counts,
            "y": BW,
            "color": color_cycle[i % len(color_cycle)],
            "label": f"radix_bits={radix_bits} (u32)",
            "linestyle": "--",
            "marker": ".",
        }

    make_std_bench_plot(
        plot_data,
        xlabel="Number of elements",
        ylabel="Bandwidth (B.s^-1)",
        title=f"digit histogram performance benchmarks ({case} keys)",
        end_label_fmt=lambda y: f"{y / 1e9:.2f} GB.s^-1",
    )
    plt.show()

# %%
# Plot the histograms of the 4 digit places of 8 bits for each key distribution

N_plot = 1000000

fig, axs = plt.subplots(1, len(key_cases), figsize=(15, 4), layout="constrained")
for ax, case in zip(axs, key_cases):
    hist = compute_digit_histogram(make_keys(case, N_plot), 8)
    for p in range(4):
        ax.plot(hist[p * 256 : (p + 1) * 256], label=f"digit place {p} (bits {8 * p}-{8 * p + 7})")
    ax.set_xlabel("digit value")
    ax.set_ylabel("count")
    ax.set_yscale("symlog")
    ax.set_title(f"{case} keys")
axs[0].legend()
fig.suptitle(f"digit histogram (radix_bits=8, {N_plot} u32 keys)")
plt.show()
