"""
Sph homogeneous benchmarks results
==================================

Show the results on various devices
"""

# sphinx_gallery_multi_image = "single"

import json
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

try:
    base_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_path = os.getcwd()

json_file = os.path.join(base_path, "sph_homogeneous_bench_result.json")
results = json.load(open(json_file))

json.dump(results,open(json_file,'w'),indent=4)

print(f"results from {json_file}")

results_per_model = {}

def key_name(name, world_size):
    if world_size == 1:
        return name
    else:
        return f"{world_size} x {name}"

for result in results:
    name = key_name(result["device_properties"]["name"], result["world_size"])
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
