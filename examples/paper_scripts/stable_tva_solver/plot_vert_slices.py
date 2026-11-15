import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np
import json
import os
from matplotlib.lines import Line2D

from settling_reference_1d import GasAtRestSettle, MultifluidSettle

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

# --- Stokes number contours (parameters of run_dustydisc.py) ---
# St = t_s Omega_K with the Epstein stopping time used by the solver
# t_s = rho_grain s / (rho cs) sqrt(pi gamma / 8), rho = total SPH density (gas slice)
# Code units: au, yr, Msol
G_code = 39.47813         # au^3 Msol^-1 yr^-2
au_m = 149597870700.0     # au -> m
center_mass = 1.0         # Msol
disc_r0 = 10.0            # au
disc_H_r_0 = 0.1
disc_q = 0.5
gamma_ad = 1.4
rho_grain_si = 2300.0     # kg m^-3
cs0_code = disc_H_r_0 * np.sqrt(G_code * center_mass / disc_r0)  # au/yr, cs at r0
ST_LEVELS = [1e-2, 0.5, 1.0]
ST_LABELS = {1e-2: r"$10^{-2}$", 0.5: r"$0.5$", 1.0: r"$1$"}


def grain_size_au(dust_id):
    # bin centers of logspace(-6.5, -1.5, ndust + 1) m: s_j = 10^(j - 6) m
    return 10.0 ** (dust_id - 6) / au_m


def stokes_field(rho_gas_code, meta, xmin, xmax, dust_id):
    """St on the cropped slice (same pixels as crop_x_slice)."""
    x0, x1, z0, z1 = meta["extent"]
    ny, nx = rho_gas_code.shape
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    zs = z0 + (np.arange(ny) + 0.5) * (z1 - z0) / ny
    i0 = int(np.round((xmin - x0) / (x1 - x0) * nx))
    i1 = int(np.round((xmax - x0) / (x1 - x0) * nx))
    X, Z = np.meshgrid(xs[i0:i1], zs)

    omega_k = np.sqrt(G_code * center_mass / np.abs(X) ** 3)  # cylindrical radius (y = 0)
    cs = cs0_code * (np.sqrt(X**2 + Z**2) / disc_r0) ** (-disc_q)  # LP07, spherical radius
    rho_grain_code = rho_grain_si / codeu_to_kg_m3
    rho = rho_gas_code[:, i0:i1]

    ts = rho_grain_code * grain_size_au(dust_id) / (rho * cs) * np.sqrt(np.pi * gamma_ad / 8)
    return ts * omega_k


def add_stokes_contours(ax, st, extent):
    if np.isnan(st).all():
        return
    cs_ = ax.contour(
        st,
        levels=ST_LEVELS,
        origin="lower",
        extent=extent,
        colors="cyan",
        linewidths=0.6,
        zorder=9,
    )
    ax.clabel(cs_, fmt=ST_LABELS, fontsize=5, inline=True)

# --- 1D settling references at R = REF_R (as in dustysettle_limiters.py) ---
yr_s = 31557600.0  # Julian year [s]
disc_ndust = 5
disc_eps_base = 0.01
disc_mrn_pow = 3.5
REF_R = 50.0  # au
REF_ZRANGE = 5.0  # half height of the 1D domain, in units of H
REF_NZ = 4000
REF_TAU = 0.025  # time step of the gas-at-rest reference, in units of 1/OmegaK
MF_NZ = 4000  # cells of the multifluid reference (even)
MF_ETA = 0.0  # no velocity dissipation in the disc run
REF_TVA_COLOR = "0.0"
REF_MF_COLOR = "lightskyblue"
REF_ST_COLOR = "tab:green"
ST_XLIM = (1e-2, 10)


def disc_mrn_weights():
    # same as DustMRNDistribution with grain_size_si_edges = logspace(-6.5, -1.5, ndust + 1)
    s_edges = np.logspace(-6.5, -1.5, disc_ndust + 1)
    alpha = 3 - disc_mrn_pow
    prim = s_edges ** (alpha + 1) / (alpha + 1)
    w = prim[1:] - prim[:-1]
    return w / w.sum()


def midplane_density(data, meta, r, dx=1.0, dz=0.25):
    """Mean of a density slice over |x - r| <= dx, |z| <= dz."""
    x0, x1, z0, z1 = meta["extent"]
    ny, nx = data.shape
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    zs = z0 + (np.arange(ny) + 0.5) * (z1 - z0) / ny
    return np.nanmean(data[np.ix_(np.abs(zs) <= dz, np.abs(xs - r) <= dx)])


def compute_settling_references(rho_mid, t_yr):
    """
    Gas-at-rest (TVA reference) and multifluid (non-TVA) settling at R = REF_R after
    t_yr years, for every dust species of the disc run. The gas is a Gaussian of
    scale height H = cs / OmegaK at REF_R with midplane gas density rho_mid [kg m^-3];
    the dust starts with the uniform dust fraction of run_dustydisc.py (no mask).
    """
    omega_code = np.sqrt(G_code * center_mass / REF_R**3)  # 1/yr
    cs_code = cs0_code * (REF_R / disc_r0) ** (-disc_q)  # au/yr
    H = cs_code / omega_code  # au
    omega_si = omega_code / yr_s
    cs_si = cs_code * au_m / yr_s
    tbar = t_yr * omega_code

    rhoeff = rho_grain_si * np.sqrt(np.pi * gamma_ad / 8)
    s_grain = [10.0 ** (j - 6) for j in range(disc_ndust)]  # m
    dust_to_gas = disc_eps_base * disc_mrn_weights() / (1 - disc_eps_base)
    print(f"1D references: R = {REF_R} au, H = {H:.4f} au, rho_mid = {rho_mid:.4e} kg/m^3")

    refs = {}
    for j in range(disc_ndust):
        ref = GasAtRestSettle(
            REF_NZ, H, REF_R, rho_mid, dust_to_gas[j], rhoeff, s_grain[j], cs_si, omega_si,
            REF_TAU, REF_ZRANGE, mask=False,
        )
        ref.advance_until(tbar)
        refs[j] = {"z": ref.z_arr(), "rho_tva": ref.rho_dust(), "St": ref.Stj}
        print(f"  j = {j}: midplane St = {ref.Stj[REF_NZ // 2]:.4e}")

    mf = MultifluidSettle(
        MF_NZ,
        cs_bar=H / REF_R,
        eta_bar=MF_ETA,
        inv_ts_coef=[cs_si * rho_mid / (rhoeff * s * omega_si) for s in s_grain],
        dust_to_gas=dust_to_gas,
        h_bar=H / REF_R,
        zrange=REF_ZRANGE,
        mask=False,
    )
    mf.advance_until(tbar)
    for j in range(disc_ndust):
        refs[j]["z_mf"] = mf.zc * REF_R
        refs[j]["rho_mf"] = mf.rho_d[j] * rho_mid
    return refs


def z_to_axis(z, extent, z_bottom, z_top):
    """Same z mapping as the images: the slice z range goes to [z_bottom, z_top]."""
    z0, z1 = extent[2], extent[3]
    return z_bottom + (np.asarray(z) - z0) / (z1 - z0) * (z_top - z_bottom)


def plot_reference_panel(ax, refs, title):
    ax_st = ax.twiny()
    for dust_idx, z_bottom, z_top in ((1, *Z_TOP), (0, *Z_BOTTOM)):
        ref = refs[dust_ids[dust_idx]]
        extent = all_meta[dust_idx][-1][0]["extent"]
        ax.plot(
            ref["rho_tva"], z_to_axis(ref["z"], extent, z_bottom, z_top),
            "--", color=REF_TVA_COLOR, alpha=0.7, linewidth=1,
        )
        ax.plot(
            ref["rho_mf"], z_to_axis(ref["z_mf"], extent, z_bottom, z_top),
            "--", color=REF_MF_COLOR, linewidth=1,
        )
        ax_st.plot(
            ref["St"], z_to_axis(ref["z"], extent, z_bottom, z_top),
            "-", color=REF_ST_COLOR, linewidth=1,
        )

    ax.axhline(Z_TOP[0], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-18, 1e-11)
    ax.set_xlabel(r"$\rho_{\mathrm{d},j}$ [kg m$^{-3}$]")

    ax_st.set_xscale("log")
    ax_st.set_xlim(*ST_XLIM)
    ax_st.set_xlabel("St", color=REF_ST_COLOR)
    ax_st.tick_params(axis="x", colors=REF_ST_COLOR, which="both")
    ax.set_title(title)

    ax.legend(
        handles=[
            Line2D([0], [0], linestyle="--", color=REF_TVA_COLOR, alpha=0.7, label="reference"),
            Line2D([0], [0], linestyle="--", color=REF_MF_COLOR, label="multifluid (non-TVA)"),
            Line2D([0], [0], linestyle="-", color=REF_ST_COLOR, label="St"),
        ],
        loc="upper left",
        fontsize=5,
    )


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

            st = stokes_field(
                all_gas_data[time_idx][j], all_gas_meta[time_idx][j], xmin, xmax, dust_ids[dust_idx]
            )
            #add_stokes_contours(ax, st, [x_left, x_right, z_bottom, z_top])

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

# gas (total) density slices, used for the Stokes number contours
all_gas_data = []
all_gas_meta = []
for dump in dump_ids:
    row_data = []
    row_meta = []
    for _, sim in sims:
        data, meta = load_data(
            os.path.join(base, sim, "analysis/plots", f"rho_slice_gas_{dump:07d}"),
            os.path.join(base, fallback_sim, "analysis/plots", f"rho_slice_gas_{dump:07d}"),
        )
        row_data.append(data)
        row_meta.append(meta)
    all_gas_data.append(row_data)
    all_gas_meta.append(row_meta)

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

# --- 1D settling references at R = REF_R, at the last time ---
# midplane gas density of the disc at REF_R and t = 0: total density of the slice
# (first simulation that has it) times (1 - eps) to remove the dust
rho_mid_tot = next(
    midplane_density(all_gas_data[0][j], all_gas_meta[0][j], REF_R) * codeu_to_kg_m3
    for j in range(len(sims))
    if not np.isnan(all_gas_data[0][j]).all()
)
ref_time = times[-1]
refs = compute_settling_references(rho_mid_tot * (1 - disc_eps_base), ref_time)

# --- make figure ---

fig = plt.figure(dpi=250,figsize=(13.5, 5))

gs = GridSpec(
    1, len(times) + 4,
    width_ratios=[0.35, 0.08] + [1] * len(times) + [1, 0.05],
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

# --- 1D references panel (z vertical, same z layout as the slices) ---
ax_ref = fig.add_subplot(gs[0, len(times) + 2], sharey=axs[0])
ax_ref.tick_params(left=False, labelleft=False)
plot_reference_panel(ax_ref, refs, rf"R={REF_R:.0f}au t={ref_time:.0f}yr")

ax_cbar = fig.add_subplot(gs[0, len(times) + 3])
cbar = fig.colorbar(ims[-1], cax=ax_cbar)
cbar.set_label(r"$rho_{\mathrm{d},j}$ [kg m$^{-3}$]")

plt.tight_layout()
fig.subplots_adjust(left=0.1, right=0.9)
plt.savefig("thefig.pdf", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.show()
