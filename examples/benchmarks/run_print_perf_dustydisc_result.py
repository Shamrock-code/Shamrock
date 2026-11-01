"""
Dusty disc performance results
==============================

Solver rate of the dusty disc perf test (``dustydisc_perftest.py``) on an
H200, as a function of the number of dust species.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.facecolor": "white",
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
    }
)

RESULTS_H200_BASELINE = [
    {"kernel": "M4", "ndust": 0, "rate": np.float64(20743174.163988266)},
    {"kernel": "M6", "ndust": 0, "rate": np.float64(13799390.088761112)},
    {"kernel": "M6", "ndust": 1, "rate": np.float64(12633702.042326497)},
    {"kernel": "M6", "ndust": 2, "rate": np.float64(12087424.370965026)},
    {"kernel": "M6", "ndust": 3, "rate": np.float64(11881595.212850157)},
    {"kernel": "M6", "ndust": 4, "rate": np.float64(11300411.830739705)},
    {"kernel": "M6", "ndust": 5, "rate": np.float64(11222855.348508552)},
    {"kernel": "M6", "ndust": 6, "rate": np.float64(10755208.603137823)},
    {"kernel": "M6", "ndust": 7, "rate": np.float64(10688006.42580047)},
    {"kernel": "M6", "ndust": 8, "rate": np.float64(10142698.857492624)},
    {"kernel": "M6", "ndust": 9, "rate": np.float64(10214823.874985265)},
    {"kernel": "M6", "ndust": 10, "rate": np.float64(9761393.55028728)},
    {"kernel": "M6", "ndust": 11, "rate": np.float64(9792483.14187795)},
    {"kernel": "M6", "ndust": 12, "rate": np.float64(9219263.745632991)},
    {"kernel": "M6", "ndust": 13, "rate": np.float64(9388890.664223792)},
    {"kernel": "M6", "ndust": 14, "rate": np.float64(8948305.414125301)},
    {"kernel": "M6", "ndust": 15, "rate": np.float64(9006443.219386235)},
    {"kernel": "M6", "ndust": 16, "rate": np.float64(8591518.90067488)},
    {"kernel": "M6", "ndust": 17, "rate": np.float64(8663145.798459316)},
    {"kernel": "M6", "ndust": 18, "rate": np.float64(8252145.146574976)},
    {"kernel": "M6", "ndust": 19, "rate": np.float64(8367198.978522308)},
    {"kernel": "M6", "ndust": 20, "rate": np.float64(7823688.987683165)},
    {"kernel": "M6", "ndust": 21, "rate": np.float64(8078987.197870458)},
    {"kernel": "M6", "ndust": 22, "rate": np.float64(7692061.528587866)},
    {"kernel": "M6", "ndust": 23, "rate": np.float64(7820297.588177133)},
    {"kernel": "M6", "ndust": 24, "rate": np.float64(7334507.940110926)},
    {"kernel": "M6", "ndust": 25, "rate": np.float64(7578401.6513876775)},
    {"kernel": "M6", "ndust": 26, "rate": np.float64(7200301.321665853)},
    {"kernel": "M6", "ndust": 27, "rate": np.float64(7341230.143859156)},
    {"kernel": "M6", "ndust": 28, "rate": np.float64(6858490.479193717)},
    {"kernel": "M6", "ndust": 29, "rate": np.float64(7151851.505432058)},
    {"kernel": "M6", "ndust": 30, "rate": np.float64(6780589.608352743)},
    {"kernel": "M6", "ndust": 31, "rate": np.float64(6957826.151235339)},
    {"kernel": "M6", "ndust": 32, "rate": np.float64(6871322.974704008)},
    {"kernel": "M6", "ndust": 33, "rate": np.float64(6766758.272378567)},
    {"kernel": "M6", "ndust": 34, "rate": np.float64(6416124.696613544)},
    {"kernel": "M6", "ndust": 35, "rate": np.float64(6595107.764492851)},
    {"kernel": "M6", "ndust": 36, "rate": np.float64(6124688.97834106)},
    {"kernel": "M6", "ndust": 37, "rate": np.float64(6428013.769450867)},
    {"kernel": "M6", "ndust": 38, "rate": np.float64(6106585.969761499)},
    {"kernel": "M6", "ndust": 39, "rate": np.float64(6261067.583511858)},
    {"kernel": "M6", "ndust": 40, "rate": np.float64(5846567.91142668)},
    {"kernel": "M6", "ndust": 41, "rate": np.float64(6130846.680557474)},
    {"kernel": "M6", "ndust": 42, "rate": np.float64(5781410.068906898)},
    {"kernel": "M6", "ndust": 43, "rate": np.float64(5978457.374960809)},
    {"kernel": "M6", "ndust": 44, "rate": np.float64(5510755.835376551)},
    {"kernel": "M6", "ndust": 45, "rate": np.float64(5847274.351241548)},
    {"kernel": "M6", "ndust": 46, "rate": np.float64(5533625.961877414)},
    {"kernel": "M6", "ndust": 47, "rate": np.float64(5712080.273564)},
    {"kernel": "M6", "ndust": 48, "rate": np.float64(5405527.541108767)},
    {"kernel": "M6", "ndust": 49, "rate": np.float64(5590447.5575348595)},
    {"kernel": "M6", "ndust": 50, "rate": np.float64(5307054.447021384)},
    {"kernel": "M6", "ndust": 51, "rate": np.float64(5474498.7673062915)},
    {"kernel": "M6", "ndust": 52, "rate": np.float64(5065365.2100912845)},
    {"kernel": "M6", "ndust": 53, "rate": np.float64(5353810.653996148)},
    {"kernel": "M6", "ndust": 54, "rate": np.float64(5066047.936317459)},
    {"kernel": "M6", "ndust": 55, "rate": np.float64(5242961.026497445)},
    {"kernel": "M6", "ndust": 56, "rate": np.float64(4883427.61396701)},
    {"kernel": "M6", "ndust": 57, "rate": np.float64(5132179.980544963)},
    {"kernel": "M6", "ndust": 58, "rate": np.float64(4853087.799315191)},
    {"kernel": "M6", "ndust": 59, "rate": np.float64(5020866.021284079)},
    {"kernel": "M6", "ndust": 60, "rate": np.float64(4618314.988822764)},
    {"kernel": "M6", "ndust": 61, "rate": np.float64(4921603.58455902)},
    {"kernel": "M6", "ndust": 62, "rate": np.float64(4648092.049223436)},
]


def _lookup_rate(results, kernel, ndust):
    for entry in results:
        if entry["kernel"] == kernel and entry["ndust"] == ndust:
            return float(entry["rate"])
    return None


def plot_perf_results(results, title="Dusty disc performance"):
    visible_ndust = {1, 5, 10, 20, 30, 40, 50, 60, 70}
    labels = ["gas only M4", "gas only M6"]
    rates_off = [
        _lookup_rate(results, "M4", 0),
        _lookup_rate(results, "M6", 0),
    ]

    ndust_values = sorted({entry["ndust"] for entry in results if entry["ndust"] > 0})
    for ndust in ndust_values:
        labels.append(str(ndust))
        rates_off.append(_lookup_rate(results, "M6", ndust))

    fig, ax = plt.subplots(figsize=(6.4, 6))
    gas_count = 2
    gas_spread = 2.0
    ndust_gap = 4.0
    x = np.arange(len(labels), dtype=float)
    x[1:] += gas_spread
    x[gas_count:] += ndust_gap
    width = 2

    gas_x = x[:gas_count]
    gas_y = rates_off[:gas_count]
    dust_x = x[gas_count:]
    dust_y = rates_off[gas_count:]

    ax.bar(gas_x, gas_y, width, label="coala off", color="C0")
    ax.plot(dust_x, dust_y, color="C0", marker="o", markersize=3)

    separator_x = (x[gas_count - 1] + x[gas_count]) / 2
    ax.axvline(separator_x, color="0.4", linewidth=1.2, linestyle="-", zorder=0)

    # ax.set_yscale("log")
    # ax.set_ylim(1e6,3e7)
    ax.set_ylabel("rate (particles/s)")
    ax.set_xlabel("dust species", labelpad=-16)
    ax.set_title(title)

    baseline_rate = _lookup_rate(results, "M6", 0)
    if baseline_rate:
        ax.tick_params(axis="y", right=False, which="both")
        secax = ax.secondary_yaxis(
            "right", functions=(lambda r: r / baseline_rate, lambda s: s * baseline_rate)
        )
        secax.set_ylabel("speedup relative to M6 (gas only)")
        for frac in (1.00, 0.75, 0.5):
            y = frac * baseline_rate
            ax.axhline(y, color="red", linewidth=1.0, linestyle="--", zorder=0)
            ax.text(
                x[-1],
                y,
                f"{frac:.0%}",
                color="red",
                fontsize=11,
                va="bottom",
                ha="right",
            )
    ax.set_xticks(x)
    visible_indices = {0, 1}
    for ndust in visible_ndust:
        visible_indices.add(1 + ndust)
    tick_labels = [labels[i] if i in visible_indices else "" for i in range(len(labels))]
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=12)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if i not in visible_indices:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
    # ax.legend()
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig, ax


# %%
# Plot the results
fig, ax = plot_perf_results(RESULTS_H200_BASELINE, title="Dusty disc performance (H200)")
plt.show()
