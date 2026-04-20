"""
Sph homogeneous benchmarks results
==================================

Show the results on various devices
"""

# sphinx_gallery_multi_image = "single"

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

results = [
    {
        "device_properties": {
            "vendor": "Unknown",
            "backend": "CUDA",
            "type": "GPU",
            "name": "NVIDIA GeForce RTX 3070",
            "platform": "CUDA (platform 0)",
            "global_mem_size": 8187674624,
            "global_mem_cache_line_size": 128,
            "global_mem_cache_size": 4194304,
            "local_mem_size": 49152,
            "max_compute_units": 46,
            "max_mem_alloc_size_dev": 8187674624,
            "max_mem_alloc_size_host": 66755719168,
            "mem_base_addr_align": 8,
        },
        "microbench_results": {
            "vector_allgather_u64_128": 4.007318413143612e-06,
            "vector_allgather_u64_8": 3.996427886344854e-06,
            "vector_allgather_u64_1": 4.008516935910047e-06,
            "fma_chains_f64_4": 368106547044.4124,
            "fma_chains_f32_4": 21830490235423.76,
            "fma_chains_f64_3": 369085609092.1387,
            "fma_chains_f32_3": 21317731323152.094,
            "p2p_bandwidth": 24197121395.81985,
            "saxpy_f32": 391405748413.5871,
            "saxpy_f64_2": 405820818398.0056,
            "vector_allgather_u64_1024": 4.093421080646039e-06,
            "saxpy_f64": 401132890345.5902,
            "saxpy_f32_3": 387517897478.1941,
            "saxpy_f64_3": 402289112353.3202,
            "saxpy_f32_4": 405433239724.79285,
            "fma_chains_f32": 20613866730745.523,
            "saxpy_f64_4": 401698347029.11176,
            "vector_allgather_u64_150": 4.005194448900971e-06,
            "saxpy_f32_2": 396474134612.07086,
            "fma_chains_f64": 367921828994.0285,
            "vector_allgather_u64_64": 3.99664353942146e-06,
            "fma_chains_f32_2": 21262970189336.14,
            "fma_chains_f64_2": 368348632671.6383,
        },
        "shamrock_version": "2025.10.0+git.da5ddecb5.patch-2026-04-19-17-05.dirty",
        "world_size": 1,
        "rate": 1934860.6025199948,
        "cnt": 4254912,
        "step_time": 2.199079352,
        "mpi_timers": {
            "MPI_Gatherv": 0.0,
            "MPI_Gather": 0.0,
            "MPI_File_close": 0.0,
            "MPI_File_read": 0.0,
            "MPI_File_write": 0.0,
            "MPI_File_write_all": 0.0,
            "MPI_File_set_view": 0.0,
            "total": 0.00018202600000805091,
            "MPI_Type_size": 0.0,
            "MPI_Probe": 0.0,
            "MPI_File_open": 0.0,
            "MPI_File_read_at": 0.0,
            "MPI_Isend": 0.0,
            "MPI_Allgatherv": 7.820000000435812e-06,
            "MPI_Recv": 0.0,
            "MPI_Test": 0.0,
            "MPI_Irecv": 0.0,
            "MPI_Allreduce": 9.998500000207855e-05,
            "MPI_File_write_at": 0.0,
            "MPI_Allgather": 7.361100000480292e-05,
            "MPI_Send": 0.0,
            "MPI_Get_count": 0.0,
            "MPI_Wait": 0.0,
            "MPI_Exscan": 0.0,
            "MPI_Waitall": 6.100000007336348e-07,
            "MPI_Barrier": 0.0,
        },
        "system_metric_duration": 2.1991319939999983,
    },
    {
        "device_properties": {
            "vendor": "Unknown",
            "backend": "CUDA",
            "type": "GPU",
            "name": "NVIDIA H100",
            "platform": "NVIDIA CUDA BACKEND",
            "global_mem_size": 99951443968,
            "global_mem_cache_line_size": 128,
            "global_mem_cache_size": 62914560,
            "local_mem_size": 232448,
            "max_compute_units": 132,
            "max_mem_alloc_size_dev": 99951443968,
            "max_mem_alloc_size_host": 2147483647,
            "mem_base_addr_align": 512,
        },
        "microbench_results": {
            "vector_allgather_u64_128": 6.873259037797696e-06,
            "vector_allgather_u64_8": 6.820455053884564e-06,
            "vector_allgather_u64_1": 6.792518983908679e-06,
            "fma_chains_f64_4": 32435874177603.85,
            "fma_chains_f32_4": 63383438474733.88,
            "fma_chains_f64_3": 24517378871793.74,
            "fma_chains_f32_3": 45468758517977.305,
            "p2p_bandwidth": 41734403486.237175,
            "saxpy_f32": 1938827077878.6948,
            "saxpy_f64_2": 2199302138792.81,
            "vector_allgather_u64_1024": 7.073777887819018e-06,
            "saxpy_f64": 2183468667282.5955,
            "saxpy_f32_3": 2162405688856.9053,
            "saxpy_f64_3": 2118545725750.071,
            "saxpy_f32_4": 2202036910418.7134,
            "fma_chains_f32": 59517431746102.67,
            "saxpy_f64_4": 2194535962050.334,
            "vector_allgather_u64_150": 6.89201385252324e-06,
            "saxpy_f32_2": 2178860694575.6865,
            "fma_chains_f64": 31684392660914.688,
            "vector_allgather_u64_64": 6.9135875561659815e-06,
            "fma_chains_f32_2": 60808447798297.055,
            "fma_chains_f64_2": 32181021255401.95,
        },
        "shamrock_version": "2025.10.0+git.588f7f53.patch-2026-04-19-17-05",
        "world_size": 1,
        "rate": 25435254.10148032,
        "cnt": 33848064,
        "step_time": 1.3307539160000001,
        "mpi_timers": {
            "MPI_Gatherv": 0.0,
            "MPI_Gather": 0.0,
            "MPI_File_close": 0.0,
            "MPI_File_read": 0.0,
            "MPI_File_write": 0.0,
            "MPI_File_write_all": 0.0,
            "MPI_File_set_view": 0.0,
            "total": 0.00038459299998550023,
            "MPI_Type_size": 0.0,
            "MPI_Probe": 0.0,
            "MPI_File_open": 0.0,
            "MPI_File_read_at": 0.0,
            "MPI_Isend": 0.0,
            "MPI_Allgatherv": 5.862099999376369e-05,
            "MPI_Recv": 0.0,
            "MPI_Test": 0.0,
            "MPI_Irecv": 0.0,
            "MPI_Allreduce": 0.00018985099998047872,
            "MPI_File_write_at": 0.0,
            "MPI_Allgather": 0.00013524100000950057,
            "MPI_Send": 0.0,
            "MPI_Get_count": 0.0,
            "MPI_Wait": 0.0,
            "MPI_Exscan": 0.0,
            "MPI_Waitall": 8.800000017572529e-07,
            "MPI_Barrier": 0.0,
        },
        "system_metric_duration": 1.3308230069999993,
    },
    {
        "device_properties": {
            "vendor": "Unknown",
            "backend": "OpenMP",
            "type": "CPU",
            "name": "Intel(R) Core(TM) Ultra 9 285K",
            "platform": "OpenMP (platform 0)",
            "global_mem_size": 66755719168,
            "global_mem_cache_line_size": 64,
            "global_mem_cache_size": 1,
            "local_mem_size": 18446744073709551615,
            "max_compute_units": 24,
            "max_mem_alloc_size_dev": 18446744073709551615,
            "max_mem_alloc_size_host": 66755719168,
            "mem_base_addr_align": 8,
        },
        "microbench_results": {
            "vector_allgather_u64_128": 3.4481665804647515e-06,
            "vector_allgather_u64_8": 3.415846666211485e-06,
            "vector_allgather_u64_1": 3.442007262583832e-06,
            "fma_chains_f64_4": 120501340061.55841,
            "fma_chains_f32_4": 155915459979.13483,
            "fma_chains_f64_3": 106471602660.81873,
            "fma_chains_f32_3": 107563566658.63593,
            "p2p_bandwidth": 16255391592.959614,
            "saxpy_f32": 111732091511.64706,
            "saxpy_f64_2": 94490913314.98209,
            "vector_allgather_u64_1024": 3.51229938185649e-06,
            "saxpy_f64": 106747022294.61467,
            "saxpy_f32_3": 105340630683.61382,
            "saxpy_f64_3": 104065079379.84784,
            "saxpy_f32_4": 101466760543.36261,
            "fma_chains_f32": 38934763201.56768,
            "saxpy_f64_4": 98500857426.4876,
            "vector_allgather_u64_150": 3.4460049965453817e-06,
            "saxpy_f32_2": 102242689953.04973,
            "fma_chains_f64": 39981735757.206894,
            "vector_allgather_u64_64": 3.4367292941147245e-06,
            "fma_chains_f32_2": 81955275730.48764,
            "fma_chains_f64_2": 64999850415.96892,
        },
        "shamrock_version": "2025.10.0+git.da5ddecb5.patch-2026-04-19-17-05.dirty",
        "world_size": 1,
        "rate": 1359407.6273648741,
        "cnt": 33848064,
        "step_time": 24.899127619,
        "mpi_timers": {
            "MPI_Gatherv": 0.0,
            "MPI_Gather": 0.0,
            "MPI_File_close": 0.0,
            "MPI_File_read": 0.0,
            "MPI_File_write": 0.0,
            "MPI_File_write_all": 0.0,
            "MPI_File_set_view": 0.0,
            "total": 0.0004833179999650383,
            "MPI_Type_size": 0.0,
            "MPI_Probe": 0.0,
            "MPI_File_open": 0.0,
            "MPI_File_read_at": 0.0,
            "MPI_Isend": 0.0,
            "MPI_Allgatherv": 0.00016333099998178113,
            "MPI_Recv": 0.0,
            "MPI_Test": 0.0,
            "MPI_Irecv": 0.0,
            "MPI_Allreduce": 0.00016439899997067187,
            "MPI_File_write_at": 0.0,
            "MPI_Allgather": 0.00015421599999854152,
            "MPI_Send": 0.0,
            "MPI_Get_count": 0.0,
            "MPI_Wait": 0.0,
            "MPI_Exscan": 0.0,
            "MPI_Waitall": 1.372000014043806e-06,
            "MPI_Barrier": 0.0,
        },
        "system_metric_duration": 24.899207496000002,
    },
    {
        "device_properties": {
            "vendor": "Unknown",
            "backend": "OpenMP",
            "type": "CPU",
            "name": "Apple M4 Max",
            "platform": "OpenMP (platform 0)",
            "global_mem_size": 68719476736,
            "global_mem_cache_line_size": 64,
            "global_mem_cache_size": 1,
            "local_mem_size": 18446744073709551615,
            "max_compute_units": 16,
            "max_mem_alloc_size_dev": 18446744073709551615,
            "max_mem_alloc_size_host": 68719476736,
            "mem_base_addr_align": 8,
        },
        "microbench_results": {
            "p2p_bandwidth": 77053362186.41933,
            "saxpy_f64": 313013844329.9971,
            "saxpy_f32_3": 306909883616.7898,
            "saxpy_f64_3": 335090551544.7831,
            "fma_chains_f32_4": 82409410401.26924,
            "saxpy_f32_2": 285611766842.20087,
            "fma_chains_f64": 40374376403.8428,
            "fma_chains_f32_3": 58926495627.24342,
            "saxpy_f32_4": 338767409596.1325,
            "fma_chains_f32": 38950032051.72212,
            "saxpy_f64_2": 338826703692.0164,
            "fma_chains_f64_2": 39031932153.242874,
            "saxpy_f64_4": 328027033808.55396,
            "saxpy_f32": 264997728796.89413,
            "fma_chains_f32_2": 39111706159.955765,
            "vector_allgather_u64_150": 2.377408909577769e-07,
            "fma_chains_f64_3": 57515646062.16901,
            "fma_chains_f64_4": 70323173069.25829,
            "vector_allgather_u64_1024": 3.353240403864739e-07,
            "vector_allgather_u64_1": 2.3136793980737145e-07,
            "vector_allgather_u64_128": 2.3548042235332763e-07,
            "vector_allgather_u64_8": 2.3093461548251047e-07,
            "vector_allgather_u64_64": 2.342667991038497e-07,
        },
        "shamrock_version": "2025.10.0+git.588f7f53f.patch-2026-04-19-17-05",
        "world_size": 1,
        "rate": 847132.8875981101,
        "cnt": 33848064,
        "step_time": 39.956026375,
        "mpi_timers": {
            "MPI_Gather": 0.0,
            "MPI_File_open": 0.0,
            "MPI_File_close": 0.0,
            "MPI_File_write_at": 0.0,
            "MPI_File_read": 0.0,
            "MPI_Test": 0.0,
            "MPI_File_write_all": 0.0,
            "MPI_Type_size": 0.0,
            "MPI_Recv": 0.0,
            "MPI_Barrier": 0.0,
            "MPI_Waitall": 1.9999999949504854e-06,
            "MPI_Wait": 0.0,
            "MPI_File_set_view": 0.0,
            "MPI_Allreduce": 1.3956999850961438e-05,
            "MPI_Get_count": 0.0,
            "MPI_File_read_at": 0.0,
            "MPI_Allgather": 2.1001000050091534e-05,
            "MPI_File_write": 0.0,
            "MPI_Send": 0.0,
            "MPI_Irecv": 0.0,
            "MPI_Probe": 0.0,
            "MPI_Exscan": 0.0,
            "MPI_Isend": 0.0,
            "MPI_Gatherv": 0.0,
            "MPI_Allgatherv": 3.7920000295343925e-06,
            "total": 4.074999992553785e-05,
        },
        "system_metric_duration": 39.95611154199992,
    },
]


results_per_model = {}

for result in results:
    name = result["device_properties"]["name"]
    if name not in results_per_model:
        results_per_model[name] = result
    else:
        if result["rate"] > results_per_model[name]["rate"]:
            results_per_model[name] = result

for name, result in results_per_model.items():
    print(f"{name}:")
    print(
        f"  - {result['world_size']} ranks, {result['rate']} rate, {result['cnt']} cnt, {result['step_time']} step time"
    )


def _rate_bar_color(device_name: str) -> str:
    """Color for the rate bar from device name (case-insensitive)."""
    lower = device_name.lower()
    if "nvidia" in lower:
        return "#2ca02c"  # green
    if "amd" in lower or "radeon" in lower:
        return "#d62728"  # red
    if "intel" in lower:
        return "#1f77b4"  # blue
    if "apple" in lower:
        return "#7f7f7f"  # grey
    return "steelblue"


def _micro_bw_and_fma(result):
    """saxpy f64 -> GB/s; fma_chains f32/f64 -> Gflops (MicroBenchmark raw flop/s, /1e9)."""
    m = result.get("microbench_results") or {}
    bw_bs = m.get("saxpy_f64")
    f64 = m.get("fma_chains_f64")
    f32 = m.get("fma_chains_f32")
    bw_gbps = (bw_bs / 1e9) if bw_bs is not None else float("nan")
    flops_f64 = (f64) if f64 is not None else float("nan")
    flops_f32 = (f32) if f32 is not None else float("nan")
    return bw_gbps, flops_f64, flops_f32


# Stable sort by rate descending for a readable chart
items = sorted(results_per_model.items(), key=lambda kv: kv[1]["rate"], reverse=True)
names = [kv[0] for kv in items]
rates = [kv[1]["rate"] for kv in items]
bw_gbps = []
flops_f64 = []
flops_f32 = []
for _, r in items:
    bw, f64, f32 = _micro_bw_and_fma(r)
    bw_gbps.append(bw)
    flops_f64.append(f64)
    flops_f32.append(f32)

h_in = max(3.0, 0.45 * len(names) + 5)
y = np.arange(len(names))

fig, (ax_rate, ax_micro) = plt.subplots(
    1,
    2,
    sharey=True,
    figsize=(15, h_in),
    gridspec_kw={"width_ratios": [75, 25], "wspace": 0.025},
)

# Wrap long device names so they stay inside the figure margin
_name_labels = ["\n".join(textwrap.wrap(n, 34)) for n in names]

_rate_colors = [_rate_bar_color(n) for n in names]
bars = ax_rate.barh(y, rates, color=_rate_colors, edgecolor="white", linewidth=0.5)
ax_rate.set_yticks(y)
ax_rate.set_yticklabels(_name_labels)
ax_rate.set_xlabel("rate (solver objects / s)")
ax_rate.set_xscale("log")
ax_rate.set_title("SPH homogeneous - rate by device")
ax_rate.bar_label(bars, fmt="%.3g", padding=3)
ax_rate.grid(axis="x", linestyle=":", alpha=0.6)
ax_rate.invert_yaxis()

# Extra room for bar-end labels; drop rightmost x tick (avoids clash with right panel)
_xmin, _xmax = ax_rate.get_xlim()
ax_rate.set_xlim(_xmin, _xmax + 0.5 * (_xmax - _xmin))
# ax_rate.xaxis.set_major_locator(MaxNLocator(prune="upper"))

# Three equal-height rows per device, evenly spaced around the tick (name at y)
_bar_h = 0.22
_spacing = 0.26  # distance between bar centers; middle bar (f32) on the tick
_y_saxpy = y - _spacing
_y_f32 = y
_y_f64 = y + _spacing

ax_micro.barh(
    _y_saxpy,
    bw_gbps,
    height=_bar_h,
    color="coral",
    label="saxpy f64 (GB/s)",
    edgecolor="white",
    linewidth=0.5,
)
ax_micro.set_xlabel("Memory bandwidth saxpy f64 (GB/s)")
ax_micro.grid(axis="x", linestyle=":", alpha=0.6)
ax_micro.tick_params(axis="y", labelleft=False)

# f32 / f64 FMA can differ a lot in scale -> log-scaled Gflops axis (same y layout as saxpy)
ax_micro_top = ax_micro.twiny()
ax_micro_top.barh(
    _y_f32,
    flops_f32,
    height=_bar_h,
    color="mediumpurple",
    label="fma_chains f32 (flops)",
    edgecolor="white",
    linewidth=0.5,
)
ax_micro_top.barh(
    _y_f64,
    flops_f64,
    height=_bar_h,
    color="seagreen",
    label="fma_chains f64 (flops)",
    edgecolor="white",
    linewidth=0.5,
)
ax_micro_top.set_xlabel("Peak FMA f32 / f64 (flops, log scale)")
ax_micro_top.set_xscale("log")
ax_micro.set_xscale("log")

h0, l0 = ax_micro.get_legend_handles_labels()
h1, l1 = ax_micro_top.get_legend_handles_labels()
ax_micro.legend(h0 + h1, l0 + l1, loc="lower right", fontsize=8)

# Flush panels: constrained_layout always leaves a gap; manual wspace=0 truly abuts axes
ax_rate.spines["right"].set_visible(True)
ax_micro.spines["left"].set_visible(False)
fig.subplots_adjust(left=0.22, right=0.99, top=0.90, bottom=0.12, wspace=0)

plt.show()
