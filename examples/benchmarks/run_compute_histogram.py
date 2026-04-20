"""
Compute histogram performance benchmarks
=================================

This example benchmarks the compute histogram performance for the different algorithms available in Shamrock
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
impl_control = shamrock.algs.compute_histogram_impl()

print(impl_control.get_alg_name())

# %%
impl_control.was_configured()

# %%
print(f"Current config: {impl_control.get_config()}")
print(f"Default config: {impl_control.get_default_config()}")
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
plt.plot(list(results_f64.keys()), list(results_f64.values()), "--.", label="f64")
plt.plot(list(results_f32.keys()), list(results_f32.values()), "--.", label="f32")
plt.xlabel("Config")
plt.ylabel("Time (ms)")
plt.ylim(0, max(max(results_f64.values()), max(results_f32.values())) * 1.1)
plt.title("Compute histogram performance benchmarks")
plt.legend()
plt.show()
