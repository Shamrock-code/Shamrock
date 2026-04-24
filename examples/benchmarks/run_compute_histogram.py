"""
Compute histogram performance benchmarks
=================================

This example benchmarks the compute histogram performance for the different algorithms available in Shamrock
"""

# sphinx_gallery_multi_image = "single"
# sphinx_gallery_thumbnail_number = 2

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
impl_control = shamrock.algs.compute_histogram_impl()

print(impl_control.get_alg_name())

# %%
impl_control.was_configured()

# %%
default_config = impl_control.get_default_config()
print(f"Current config: {impl_control.get_config()}")
print(f"Default config: {default_config}")
print(f"Available configs: {impl_control.get_avail_configs()}")

# %%
bin_edges = np.linspace(0, 1, 2049)
bin_edge_inf = bin_edges[:-1]
bin_edge_sup = bin_edges[1:]
rng = np.random.default_rng()
positions = rng.random(int(1e6))

bin_edge_inf_f32 = bin_edge_inf.astype(np.float32)
bin_edge_sup_f32 = bin_edge_sup.astype(np.float32)
positions_f32 = positions.astype(np.float32)

buf_bin_edge_inf = shamrock.backends.DeviceBuffer_f64()
buf_bin_edge_sup = shamrock.backends.DeviceBuffer_f64()
buf_positions = shamrock.backends.DeviceBuffer_f64()

buf_bin_edge_inf.resize(len(bin_edge_inf))
buf_bin_edge_sup.resize(len(bin_edge_sup))
buf_positions.resize(len(positions))

buf_bin_edge_inf.copy_from_stdvec(bin_edge_inf)
buf_bin_edge_sup.copy_from_stdvec(bin_edge_sup)
buf_positions.copy_from_stdvec(positions)

buf_bin_edge_inf_f32 = shamrock.backends.DeviceBuffer_f32()
buf_bin_edge_sup_f32 = shamrock.backends.DeviceBuffer_f32()
buf_positions_f32 = shamrock.backends.DeviceBuffer_f32()

buf_bin_edge_inf_f32.resize(len(bin_edge_inf_f32))
buf_bin_edge_sup_f32.resize(len(bin_edge_sup_f32))
buf_positions_f32.resize(len(positions_f32))

buf_bin_edge_inf_f32.copy_from_stdvec(bin_edge_inf_f32)
buf_bin_edge_sup_f32.copy_from_stdvec(bin_edge_sup_f32)
buf_positions_f32.copy_from_stdvec(positions_f32)

# %%
results_f64 = {}
results_f32 = {}
avail_configs = impl_control.get_avail_configs()
for config in avail_configs:
    impl_control.set_config(config)
    time_f64 = shamrock.algs.benchmark_compute_histogram_basic_f64(
        buf_bin_edge_inf, buf_bin_edge_sup, buf_positions
    )
    time_f32 = shamrock.algs.benchmark_compute_histogram_basic_f32(
        buf_bin_edge_inf_f32, buf_bin_edge_sup_f32, buf_positions_f32
    )
    print(f"Config: {config}, Time f64: {time_f64 * 1000}ms, Time f32: {time_f32 * 1000}ms")
    results_f64[config] = time_f64 * 1000
    results_f32[config] = time_f32 * 1000

# %%
# plot the histogram
result = shamrock.algs.compute_histogram_basic_f64(
    buf_bin_edge_inf, buf_bin_edge_sup, buf_positions
)
plt.plot(result.copy_to_stdvec())
plt.show()

# %%
# plot the results
plt.figure(layout="constrained")

configs = list(results_f64.keys())
vals_f64 = [results_f64[c] for c in configs]
vals_f32 = [results_f32[c] for c in configs]
x = np.arange(len(configs))
bar_w = 0.35
plt.bar(x - bar_w / 2, vals_f64, bar_w, label="f64")
plt.bar(x + bar_w / 2, vals_f32, bar_w, label="f32")
plt.xticks(x, configs, rotation=45, ha="right")
for tick_label, cfg in zip(plt.gca().get_xticklabels(), configs):
    if cfg == default_config:
        tick_label.set_color("red")

plt.ylabel("Time (ms)")
plt.yscale("log")

_ymin, _ymax = plt.gca().get_ylim()
_ymin = 10 ** int(np.floor(np.log10(_ymin)))
_ymax = 10 ** int(np.ceil(np.log10(_ymax)))
plt.ylim(_ymin, _ymax * 1.1)

plt.title("Compute histogram performance benchmarks")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
