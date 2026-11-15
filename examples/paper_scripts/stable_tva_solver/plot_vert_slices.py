import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np
import json
import os

cmap = plt.get_cmap("magma").copy()
cmap.set_bad(color="black")

def load_data(prefix, fallback_prefix):
    print(prefix)
    try:
        data = np.load(f"{prefix}.npy")
        with open(f"{prefix}.json") as f:
            meta = json.load(f)
        return data, meta
    except FileNotFoundError:
        data = np.load(f"{fallback_prefix}.npy")
        with open(f"{fallback_prefix}.json") as f:
            meta = json.load(f)
        data[:, :] = np.nan
        return data, meta


def crop_x_slice(data, meta, xmin, xmax):
    x0, x1, z0, z1 = meta["extent"]
    nx = data.shape[1]
    i0 = int(np.round((xmin - x0) / (x1 - x0) * nx))
    i1 = int(np.round((xmax - x0) / (x1 - x0) * nx))
    print(f"cropping x slice: x0 = {x0}, x1 = {x1}, i0 = {i0}, i1 = {i1}")
    return data[:, i0:i1]


def add_dnf_text(ax, x, y, *, rotation=90):
    ax.text(
        x,
        0.5,
        "DNF",
        transform=ax.get_xaxis_transform(),
        rotation=rotation,
        ha="center",
        va="center",
        fontsize=11,
        color="red",
        zorder=12,
    )

Z_TICKS = [-40, -20, 0, 20, 40]
Z_SPAN = Z_TICKS[-1] - Z_TICKS[0]
Z_BOTTOM = (Z_TICKS[0], Z_TICKS[-1])
Z_TOP = (Z_TICKS[-1], Z_TICKS[-1] + Z_SPAN)

codeu_to_kg_m3 = 0.0005939403178403637

def plot_dust_panels(ax, time_idx, xlim=(40, 60)):
    xmin, xmax = xlim
    section_w = xmax - xmin

    x_min_ax = np.inf
    x_max_ax = -np.inf

    ims = []
    for j in range(len(sims)):
        x_left = xmin + j * section_w
        x_right = x_left + section_w
        x_min_ax = min(x_min_ax, x_left)
        x_max_ax = max(x_max_ax, x_right)

        print(f"positioning: x_left = {x_left}, x_right = {x_right}")

        for row, dust_idx, z_bottom, z_top in (
            (0, 1, *Z_TOP),   # dust species 3, top half
            (1, 0, *Z_BOTTOM),  # dust species 4, bottom half
        ):
            data = all_data[dust_idx][time_idx][j]
            meta = all_meta[dust_idx][time_idx][j]
            crop = crop_x_slice(data, meta, xmin, xmax)
            im = ax.imshow(
                crop*codeu_to_kg_m3,
                origin="lower",
                extent=[x_left, x_right, z_bottom, z_top],
                aspect="auto",
                cmap=cmap,
                norm=LogNorm(vmin=1e-18, vmax=1e-11),
            )
            ims.append(im)
            print(np.isnan(crop).all(), np.isnan(crop))
            # INSERT_YOUR_CODE
            # Print indices of crop that are not NaN
            not_nan_indices = np.argwhere(~np.isnan(crop))
            #print(f"Not-NaN indices (row,col): {not_nan_indices}")
   
            if np.isnan(crop).all():
                print("adding DNF text")
                add_dnf_text(
                    ax,
                    0.5 * (x_left + x_right),
                    0.5 * (z_bottom + z_top),
                )

    ax.set_xlim(x_min_ax, x_max_ax)
    ax.set_ylim(Z_BOTTOM[0], Z_TOP[1])
    tick_positions = Z_TICKS + [t + Z_SPAN for t in Z_TICKS[1:]]
    tick_labels = [str(t) for t in Z_TICKS[:-1]] + ["-40\n40"] + [str(t) for t in Z_TICKS[1:]]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    x_tick_positions = [xmin + j * section_w for j in range(2)]
    x_tick_labels = [str(xmin)] + [str(xmax)] 
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)

    for j in range(1, len(sims)):
        ax.axvline(
            xmin + j * section_w,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            zorder=10,
        )
    return ims

# --- simulations ---
sims = [
    ("none",      "circular_dustydisc_5_10000000_M6_none"),
    ("Ballabio",  "circular_dustydisc_5_10000000_M6_ballabio"),
    ("EoM",    "circular_dustydisc_5_10000000_M6_smooth"),
    ("hard",      "circular_dustydisc_5_10000000_M6_hard"),
]
fallback_sim =     "circular_dustydisc_5_10000000_M6_hard"

# --- dump IDs (times) ---
dump_ids = [0, 7, 10, 40]

dust_ids = [4, 3]
grain_label = ["$10^{-2}$ m", "$10^{-3}$ m"]

# --- base path ---
base = "_to_trash"

# --- storage ---
all_data = []
all_meta = []

for dust_id in dust_ids:
    dust_specie_data = []
    dust_specie_meta = []

    for dump in dump_ids:
        row_data = []
        row_meta = []
        for _, sim in sims:
            prefix = os.path.join(
                base,
                sim,
                "analysis/plots",
                f"rho_slice_dust_{dust_id}_{dump:07d}"
            )
            fallback_prefix = os.path.join(
                base,
                fallback_sim,
                "analysis/plots",
                f"rho_slice_dust_{dust_id}_{dump:07d}"
            )
            data, meta = load_data(prefix, fallback_prefix)
            row_data.append(data)
            row_meta.append(meta)
        dust_specie_data.append(row_data)
        dust_specie_meta.append(row_meta)
    all_data.append(dust_specie_data)
    all_meta.append(dust_specie_meta)

# --- plotting ---
nrows = len(dump_ids)
ncols = len(sims)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(4 * ncols, 4 * nrows),
    sharex=True,
    sharey=True
)

for i, dump in enumerate(dump_ids):
    for j, (label, _) in enumerate(sims):
        ax = axes[i, j] if nrows > 1 else axes[j]

        data = all_data[0][i][j]
        meta = all_meta[0][i][j]

        im = ax.imshow(
            data,
            origin="lower",
            extent=meta["extent"],
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(vmin=1e-13, vmax=1e-7),
        )
        if np.isnan(data).all():
            add_dnf_text(ax, 0.5, 0.5)

        # Column titles (simulation names)
        if i == 0:
            ax.set_title(label)

        # Row labels (time)
        if j == 0:
            ax.set_ylabel(f"t = {meta['time']:.2f}")

# --- colorbar ---
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
cbar.set_label("rho")

# --- labels ---
for ax in axes[-1]:
    ax.set_xlabel("x [au]")

plt.tight_layout()
plt.savefig("prout.png", dpi=200)
plt.close()


# --- fetch times ---
times = []

for i, dump in enumerate(dump_ids):
    for j, (label, _) in enumerate(sims):
        if j == 0:
            times.append(all_meta[0][i][j]["time"])

print(f"times = {times}")

# --- make figure ---

fig = plt.figure(dpi=250,figsize=(11, 5))

gs = GridSpec(
    1, len(times) + 3,
    width_ratios=[0.35, 0.08] + [1] * len(times) + [0.05],
    wspace=0.15,
)


# -----------------------
# left colored strip
# -----------------------
ax_strip = fig.add_subplot(gs[0, 0])
fig.add_subplot(gs[0, 1]).axis("off")
ax_strip.axis('off')
ax_strip.set_xlim(0, 1)
ax_strip.set_ylim(0, 1)

# colored regions
ax_strip.fill_between([0, 0.8], 0.5, 1, color="#e8b6b6")
ax_strip.fill_between([0, 0.8], 0, 0.5, color="#d6cfe3")

# labels (shifted right to avoid overlap)
for y, dust_idx, color in ((0.75, 1, "r"), (0.25, 0, "purple")):
    dust_id = dust_ids[dust_idx]
    ax_strip.text(
        0.45,
        y,
        rf"$j={dust_id}$" + "\n $s=$" + grain_label[dust_idx],
        ha="center",
        va="center",
        rotation=90,
        color=color,
        fontsize=12,
    )

# --- axes ---
axs = []
for i, t in enumerate(times):
    if i == 0:
        ax = fig.add_subplot(gs[0, i + 2])
    else:
        ax = fig.add_subplot(gs[0, i + 2], sharex=axs[0], sharey=axs[0])
    ax.set_title(f"t = {t:.2f}")
    axs.append(ax)


axs[0].set_ylabel(r"$z$ [au]")
for i, ax in enumerate(axs):
    if i != 0:
        ax.tick_params(left=False, labelleft=False)
    ax.set_xlabel(r"$x$ [au]")

# --- images ---
ims = []
for i, ax in enumerate(axs):
    ims.extend(plot_dust_panels(ax, i))

xmin, xmax = 40, 60
section_w = xmax - xmin
for j, (label, _) in enumerate(sims):
    axs[0].text(
        xmin + (j + 0.5) * section_w,
        0.5,
        label,
        transform=axs[0].get_xaxis_transform(),
        rotation=90,
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        zorder=11,
    )

ax_cbar = fig.add_subplot(gs[0, len(times) + 2])
cbar = fig.colorbar(ims[-1], cax=ax_cbar)
cbar.set_label(r"$rho_{\mathrm{d},j}$ [kg m$^{-3}$]")

plt.tight_layout()
fig.subplots_adjust(left=0.1, right=0.9)
plt.savefig("thefig.pdf", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.show()
