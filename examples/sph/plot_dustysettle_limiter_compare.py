"""
Compare dusty-settle rho slices across limiter runs
===================================================

Load ``snapshot_data_0050.npy`` from the limiter comparison simulations and
plot the vertical density slice (``ax_rho``) in a 1x4 panel, with dust mass
conservation histories on a second row.

Example (from the repository root or the build directory):

```bash
python ../examples/sph/plot_dustysettle_limiter_compare.py
```
"""

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D

try:
    import shamrock

    shamrock.matplotlib.set_shamrock_mpl_style()
except ImportError:
    pass

cmap = "plasma"
dpi = 250
sz = 1
H = 0.05  # H_r_0 * R0 in run_dustysettle_tva.py
range_plot = 2.5 * H
snapshot_index = 50

LZ = 256

simulations = [
    ("none", f"dusty_settle_none_{LZ}"),
    ("Ballabio", f"dusty_settle_ballabio_{LZ}"),
    ("smooth", f"dusty_settle_smooth_{LZ}"),
    ("hard", f"dusty_settle_hard_{LZ}"),
]


def find_sim_root():
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "_to_trash",
        Path.cwd() / "build" / "_to_trash",
        here.parents[2] / "build" / "_to_trash",
    ]
    for candidate in candidates:
        if (candidate / f"dusty_settle_hard_{LZ}").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find build/_to_trash with dusty_settle_*_{LZ} simulations. "
        "Run this script from the repository root or the build directory."
    )


def load_snapshot(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = data.item()
    return data


def load_dust_mass(dump_dir):
    filepath = dump_dir / "dust_mass.json"
    with open(filepath) as fp:
        data = json.load(fp)["dust_mass"]
    t = np.array([d["t"] for d in data])
    values = np.array([d["dust_mass"] for d in data])
    return t, values


def plot_rho(ax, snapshot_data, dust_colors):
    z = snapshot_data["z"]
    s_j = snapshot_data["s_j"]
    rho = snapshot_data["rho"]
    to_dens = snapshot_data["to_dens"]
    rho_dust_all = snapshot_data["rho_dust_all"]
    reference_plots = snapshot_data["reference_plots"]
    ndust = s_j.shape[1]

    for i in range(ndust):
        c = dust_colors[i]
        ax.scatter(z, s_j[:, i] ** 2 * to_dens, s=sz, color=c, edgecolors="none", rasterized=True)

        if reference_plots is not None:
            ax.plot(
                reference_plots[i]["zbar"],
                reference_plots[i]["rho"],
                "--",
                linewidth=1,
                color="0.0",
                alpha=0.7,
            )

    ax.scatter(
        z, rho * to_dens - rho_dust_all, s=sz, color="0.0", edgecolors="none", rasterized=True
    )
    ax.scatter(z, rho_dust_all, s=sz, color="0.5", edgecolors="none", rasterized=True)

    ax.set_xlabel(r"$z$")
    ax.set_yscale("log")
    ax.set_ylim(1e-20, 1e-8)
    ax.set_xlim(-range_plot, range_plot)


def plot_dust_mass_conservation(ax, t, dust_mass, dust_colors):
    iinject = np.argmax(~np.isnan(dust_mass)[:, 0])
    t = np.array(t) - t[iinject]
    ndust = dust_mass.shape[1]

    for k in range(ndust):
        mh = dust_mass[:, k]
        deviation = (mh / mh[iinject]) - 1
        ax.plot(t, deviation, color=dust_colors[k], linewidth=1)

    total_dust_mass = np.sum(dust_mass, axis=1)
    ax.plot(
        t,
        (total_dust_mass / total_dust_mass[iinject]) - 1,
        color="grey",
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(0, color="k", linestyle=":", linewidth=0.8)

    ax.set_xlabel("t")
    ax.set_yscale("symlog", linthresh=1e-8)


def main():
    sim_root = find_sim_root()

    snapshots = []
    mass_histories = []
    for label, folder in simulations:
        dump_dir = sim_root / folder / "dump"
        snapshots.append(
            (label, load_snapshot(dump_dir / f"snapshot_data_{snapshot_index:04d}.npy"))
        )
        mass_histories.append(load_dust_mass(dump_dir))

    grain_size_si = snapshots[0][1]["grain_size_si"]
    time = snapshots[0][1]["time"]

    dust_cmap = plt.colormaps[cmap]
    dust_norm = mcolors.LogNorm(vmin=grain_size_si.min(), vmax=grain_size_si.max() * 10)
    dust_colors = dust_cmap(dust_norm(grain_size_si))

    # Two-column journal width (~180 mm / 7.1 in for A&A, ApJ, MNRAS)
    fig = plt.figure(figsize=(9 * 1.2, 5 * 1.2), dpi=dpi)
    gs = fig.add_gridspec(2, 4, wspace=0.12, hspace=0.25, height_ratios=[1.8, 1])

    axes_rho = [fig.add_subplot(gs[0, 0])]
    for i in range(1, 4):
        axes_rho.append(fig.add_subplot(gs[0, i], sharey=axes_rho[0]))

    axes_mass = [fig.add_subplot(gs[1, 0])]
    for i in range(1, 4):
        axes_mass.append(fig.add_subplot(gs[1, i], sharey=axes_mass[0]))

    fig.suptitle(f"t = {time:.2f} [yr]")
    fig.subplots_adjust(left=0.12, right=1.05, top=0.90, bottom=0.10)

    for ax, (label, snapshot_data) in zip(axes_rho, snapshots):
        plot_rho(ax, snapshot_data, dust_colors)
        ax.set_title(label)

    axes_rho[0].set_ylabel(r"$\rho$ [kg/m$^3$]")
    for ax in axes_rho[1:]:
        ax.tick_params(labelleft=False)

    for ax, (t, dust_mass) in zip(axes_mass, mass_histories):
        plot_dust_mass_conservation(ax, t, dust_mass, dust_colors)

    axes_mass[0].set_ylabel(r"$\delta M_{\mathrm{dust}} / M_{\mathrm{dust},0}$")
    for ax in axes_mass[1:]:
        ax.tick_params(labelleft=False)

    gas_handle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="0.",
        markeredgecolor="none",
        label="gas",
    )
    dust_handle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="0.5",
        markeredgecolor="none",
        label="dust",
    )
    analytic_handle = Line2D(
        [0],
        [0],
        linestyle="--",
        color="0.0",
        alpha=0.7,
        label="reference",
    )
    axes_rho[-1].legend(
        handles=[gas_handle, dust_handle, analytic_handle], loc="upper right", fontsize=6
    )

    total_handle = Line2D(
        [0],
        [0],
        linestyle="--",
        color="grey",
        label="total dust mass",
    )
    axes_mass[-1].legend(handles=[total_handle], loc="upper right", fontsize=6)

    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
    dust_sm.set_array([])
    cbar = fig.colorbar(dust_sm, ax=axes_rho + axes_mass, pad=0.01, shrink=0.95, aspect=40)
    cbar.set_label(r"grain size $s$ [m]")

    out_path = sim_root / f"dustysettle_rho_limiters_{snapshot_index:04d}.pdf"
    fig.savefig(out_path)
    print(f"saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
