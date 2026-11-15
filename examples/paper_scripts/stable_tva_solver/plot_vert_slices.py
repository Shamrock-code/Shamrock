import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np
import json
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator, NullFormatter

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

# z layout of the reference figure: each half shows |z| <= REF_Z_HALF in true z (au)
REF_Z_HALF = 30
REF_Z_STEP = 10
REF_Z_BOTTOM = (-REF_Z_HALF, REF_Z_HALF)
REF_Z_TOP = (REF_Z_HALF, 3 * REF_Z_HALF)

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
# SPH vertical profiles at REF_R shown in the reference panel (simulation label -> color),
# averaged over |x - REF_R| <= PROFILE_DX to smooth the particle noise of the slice
# EoM and hard are nearly identical above 1e-15 kg/m^3, so EoM is drawn wide underneath
# and hard thin and dashed on top to keep both visible
SIM_PROFILE_STYLES = {
    "Ballabio": dict(color="tab:orange", linestyle="-", linewidth=0.8, zorder=3),
    "EoM": dict(color="tab:red", linestyle="-", linewidth=2.0, alpha=0.6, zorder=3),
    "hard": dict(color="tab:purple", linestyle=(0, (3, 1.5)), linewidth=0.8, zorder=4),
}
PROFILE_DX = 1.0  # au
# inset zoom on the j = 4 peak in the reference panel
SHOW_INSET = False
INSET_XLIM = (1e-13, 1e-11)  # kg m^-3
INSET_Z_MARGIN = 1.5  # z range: +- this factor times the edge of the reference distribution
INSET_OFF = 0.12
INSET_BOUNDS = [0.35, 0.325 + INSET_OFF, 0.62, 0.155]  # in reference panel axes fraction (x0, y0, w, h)


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


def z_to_axis(z, z_bottom, z_top):
    """
    True z (au) to the axis coordinate of the half [z_bottom, z_top] (z = 0 at its centre);
    nan outside the half, so that the curves stop at its edges.
    """
    z = np.asarray(z, dtype=float)
    half = 0.5 * (z_top - z_bottom)
    return np.where(np.abs(z) <= half, z + 0.5 * (z_bottom + z_top), np.nan)


def slice_column_profile(data, meta, r, dx):
    """Vertical profile of a density slice, averaged over |x - r| <= dx: (z, rho)."""
    x0, x1, z0, z1 = meta["extent"]
    ny, nx = data.shape
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    zs = z0 + (np.arange(ny) + 0.5) * (z1 - z0) / ny
    cols = data[:, np.abs(xs - r) <= dx]
    with np.errstate(all="ignore"):
        rho = np.where(np.isnan(cols).all(axis=1), np.nan, np.nanmean(cols, axis=1))
    return zs, rho


def draw_density_profiles(ax, refs, dust_idx, z_bottom, z_top):
    """SPH profiles at REF_R (last time) + the two 1D references, for one dust species."""
    ref = refs[dust_ids[dust_idx]]

    for j, (label, _) in enumerate(sims):
        if label not in SIM_PROFILE_STYLES:
            continue
        data = all_data[dust_idx][-1][j]
        if np.isnan(data).all():  # did not reach this time
            continue
        meta = all_meta[dust_idx][-1][j]
        z, rho = slice_column_profile(data, meta, REF_R, PROFILE_DX)
        ax.plot(
            rho * codeu_to_kg_m3, z_to_axis(z, z_bottom, z_top),
            **SIM_PROFILE_STYLES[label],
        )

    ax.plot(
        ref["rho_tva"], z_to_axis(ref["z"], z_bottom, z_top),
        "--", color=REF_TVA_COLOR, alpha=0.7, linewidth=1,
    )
    ax.plot(
        ref["rho_mf"], z_to_axis(ref["z_mf"], z_bottom, z_top),
        "--", color=REF_MF_COLOR, linewidth=1,
    )


def add_peak_inset(ax, refs, dust_idx, z_bottom, z_top):
    """
    Zoom on the peak of one dust species: INSET_XLIM in density and, in z, both sides of
    the midplane up to just beyond the edges of the (gas-at-rest) reference distribution.
    """
    ref = refs[dust_ids[dust_idx]]
    z_front = np.abs(ref["z"][ref["rho_tva"] > INSET_XLIM[0]]).max()
    z_max = INSET_Z_MARGIN * z_front
    ylim = z_to_axis(np.array([-z_max, z_max]), z_bottom, z_top)

    axins = ax.inset_axes(INSET_BOUNDS, xlim=INSET_XLIM, ylim=tuple(ylim))
    draw_density_profiles(axins, refs, dust_idx, z_bottom, z_top)
    axins.set_xscale("log")
    axins.set_xlim(*INSET_XLIM)
    axins.set_ylim(*ylim)
    axins.tick_params(labelsize=5, length=2, pad=1)
    axins.tick_params(which="minor", length=1)
    # z ticks on round values in au, placed with the panel z mapping
    z_ticks = MaxNLocator(nbins=4, integer=True, symmetric=True).tick_values(-z_max, z_max)
    z_ticks = z_ticks[np.abs(z_ticks) <= z_max]
    axins.set_yticks(z_to_axis(z_ticks, z_bottom, z_top))
    axins.set_yticklabels([f"{z:g}" for z in z_ticks])
    ax.indicate_inset_zoom(axins, edgecolor="0.3", linewidth=0.6)
    return axins


def plot_reference_panel(ax, refs, title):
    ax_st = ax.twiny()
    for dust_idx, z_bottom, z_top in ((1, *REF_Z_TOP), (0, *REF_Z_BOTTOM)):
        ref = refs[dust_ids[dust_idx]]

        draw_density_profiles(ax, refs, dust_idx, z_bottom, z_top)
        ax_st.plot(
            ref["St"], z_to_axis(ref["z"], z_bottom, z_top),
            "-", color=REF_ST_COLOR, linewidth=1,
        )

    ax.axhline(REF_Z_TOP[0], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-20, 1e-10)
    ax.set_xlabel(r"$\rho_{\mathrm{d},j}$ [kg m$^{-3}$]")

    ax_st.set_xscale("log")
    ax_st.set_xlim(*ST_XLIM)
    ax_st.set_xlabel("St", color=REF_ST_COLOR)
    ax_st.tick_params(axis="x", colors=REF_ST_COLOR, which="both")

    # minor ticks on both log axes: every 1..9 x 10^k (the unlabeled decades included)
    for axis in (ax.xaxis, ax_st.xaxis):
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10), numticks=100))
        axis.set_minor_formatter(NullFormatter())

    # zoom on the peak of the j = 4 distribution (bottom half)
    if SHOW_INSET:
        add_peak_inset(ax, refs, 0, *REF_Z_BOTTOM)

    ax.text(
        0.04,
        0.03,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        zorder=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.3", linewidth=0.6),
    )

    # species of each half (same colors as the left strip of the main figure): top right of
    # the top half, bottom right of the bottom half
    species_boxes = []
    for dust_idx, y, va, color in ((1, 0.97, "top", "r"), (0, 0.03, "bottom", "purple")):
        species_boxes.append(ax.text(
            0.96,
            y,
            rf"$j={dust_ids[dust_idx]}$, $s=$" + grain_label[dust_idx],
            transform=ax.transAxes,
            ha="right",
            va=va,
            fontsize=7,
            color=color,
            zorder=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.3", linewidth=0.6
            ),
        ))

    # density curves (bottom axis)
    leg_rho = ax.legend(
        handles=[
            Line2D([0], [0], linestyle="--", color=REF_TVA_COLOR, alpha=0.7, label="reference"),
            Line2D([0], [0], linestyle="--", color=REF_MF_COLOR, label="multifluid (non-TVA)"),
        ]
        + [
            Line2D([0], [0], label=label, **style)
            for label, style in SIM_PROFILE_STYLES.items()
        ],
        loc="upper left",
        fontsize=5,
    )
    ax.add_artist(leg_rho)
    # Stokes number (top axis) in its own box, right-aligned under the j = 3 species box
    ax.figure.canvas.draw()
    box = species_boxes[0].get_bbox_patch().get_window_extent()
    box_ax = ax.transAxes.inverted().transform(box)
    leg_st = ax.legend(
        handles=[Line2D([0], [0], linestyle="-", color=REF_ST_COLOR, label="St (top axis)")],
        loc="upper right",
        bbox_to_anchor=(box_ax[1, 0], box_ax[0, 1] - 0.01),
        borderaxespad=0.0,
        fontsize=5,
    )
    for text in leg_st.get_texts():
        text.set_color(REF_ST_COLOR)


def set_z_axis(ax, z_half=Z_TICKS[-1], step=Z_TICKS[1] - Z_TICKS[0]):
    """
    z limits and ticks of the stacked layout (top half j = 3, bottom half j = 4), each half
    spanning [-z_half, z_half] with ticks every step.
    """
    ticks = list(range(-z_half, z_half + 1, step))
    ax.set_ylim(-z_half, 3 * z_half)
    tick_positions = ticks + [t + 2 * z_half for t in ticks[1:]]
    tick_labels = (
        [str(t) for t in ticks[:-1]] + [f"{-z_half}\n{z_half}"] + [str(t) for t in ticks[1:]]
    )
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)


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
    set_z_axis(ax)

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

# Horizontal layout in inches, with gaps only as wide as the labels that overhang them
# (no white space between the columns). Columns: strip | z label & ticks | time panels
# | colorbar. The reference panel is a separate figure (see below).
PANEL_W = 1.745  # time panel
REF_W = 0.96 * PANEL_W  # minimum width showing the 4 St labels (10^-2 ... 10^1)
STRIP_W = 0.49
YLABEL_W = 0.56  # z label + z tick labels of the first panel
GAP_PANELS = 0.11  # the "40" x tick label of the next panel overhangs by ~0.09 in
GAP_CBAR = 0.08
CBAR_W = 0.087
LEFT_PAD = 0.01
RIGHT_PAD = 0.73  # colorbar tick labels + label
FIG_H = 5

col_widths = [STRIP_W, YLABEL_W]
for i in range(len(times)):
    col_widths += [PANEL_W] if i == 0 else [GAP_PANELS, PANEL_W]
col_widths += [GAP_CBAR, CBAR_W]
col_panel = [2 + 2 * i for i in range(len(times))]
col_cbar = col_panel[-1] + 2

fig_w = LEFT_PAD + sum(col_widths) + RIGHT_PAD
fig = plt.figure(dpi=250, figsize=(fig_w, FIG_H))

gs = GridSpec(1, len(col_widths), width_ratios=col_widths, wspace=0)


# -----------------------
# left colored strip
# -----------------------
ax_strip = fig.add_subplot(gs[0, 0])
ax_strip.axis('off')
ax_strip.set_xlim(0, 1)
ax_strip.set_ylim(0, 1)

# colored regions
ax_strip.fill_between([0, 1], 0.5, 1, color="#e8b6b6")
ax_strip.fill_between([0, 1], 0, 0.5, color="#d6cfe3")

for y, dust_idx, color in ((0.75, 1, "r"), (0.25, 0, "purple")):
    dust_id = dust_ids[dust_idx]
    ax_strip.text(
        0.5,
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
        ax = fig.add_subplot(gs[0, col_panel[i]])
    else:
        ax = fig.add_subplot(gs[0, col_panel[i]], sharex=axs[0], sharey=axs[0])
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

ax_cbar = fig.add_subplot(gs[0, col_cbar])
cbar = fig.colorbar(ims[-1], cax=ax_cbar)
cbar.set_label(r"$rho_{\mathrm{d},j}$ [kg m$^{-3}$]")

plt.tight_layout()  # vertical margins
fig.subplots_adjust(left=LEFT_PAD / fig_w, right=1 - RIGHT_PAD / fig_w, wspace=0)
fig.savefig("thefig.pdf", dpi=200, bbox_inches="tight", pad_inches=0.01)

# --- 1D references panel (z vertical, same z layout as the slices), separate figure ---
def limiters_column_box():
    """
    Box (x0, y0, width, height, fig_height, in inches) of one column of the figure of
    dustysettle_limiters.py: from the bottom of the dust mass panel to the top of the
    density panel. Rebuilds the same (empty) layout as ``plot_limiter_comparison`` there.
    """
    fig = plt.figure(figsize=(9 * 1.2, 5 * 1.2))
    gs = fig.add_gridspec(2, 4, wspace=0.12, hspace=0.25, height_ratios=[1.8, 1])
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    fig.subplots_adjust(left=0.12, right=1.05, top=0.90, bottom=0.10)
    sm = plt.cm.ScalarMappable()
    sm.set_array([])
    fig.colorbar(sm, ax=axes, pad=0.01, shrink=0.95, aspect=40)
    top, bottom = axes[0].get_position(), axes[4].get_position()
    fig_w, fig_h = fig.get_size_inches()
    plt.close(fig)
    return (
        top.x0 * fig_w,
        bottom.y0 * fig_h,
        top.width * fig_w,
        (top.y1 - bottom.y0) * fig_h,
        fig_h,
    )


# Panel with the height of one (density + dust mass) column of the dustysettle_limiters.py
# figure and REF_WIDEN times its width. The figure is fitted to the panel and its labels
# (no white margin, nothing cut).
# The figure goes in a single column of the paper: the default font size is reduced by
# REF_FONT_SCALE (ticks, axis labels); the legend, inset ticks and title box keep their
# explicit sizes.
REF_FONT_SCALE = 0.75
REF_WIDEN = 1.2 * 1.2  # panel width relative to a limiters-figure column (height unchanged)
REF_PAD = 0.02  # in, white margin around the labels
_, _, panel_w, panel_h, _ = limiters_column_box()
panel_w *= REF_WIDEN

# first layout with 1 in around the panel, enough for any label
with plt.rc_context({"font.size": plt.rcParams["font.size"] * REF_FONT_SCALE}):
    fig_ref = plt.figure(dpi=250, figsize=(panel_w + 2, panel_h + 2))
    ax_ref = fig_ref.add_axes(
        [1 / (panel_w + 2), 1 / (panel_h + 2), panel_w / (panel_w + 2), panel_h / (panel_h + 2)]
    )
    set_z_axis(ax_ref, REF_Z_HALF, REF_Z_STEP)
    ax_ref.set_ylabel(r"$z$ [au]")
    plot_reference_panel(ax_ref, refs, rf"R={REF_R:.0f}au t={ref_time:.0f}yr")

# measure how far the labels extend around the panel (inches), then fit the figure to it
fig_ref.canvas.draw()
bb = fig_ref.get_tightbbox(fig_ref.canvas.get_renderer())  # inches
pad_l = 1 - bb.x0 + REF_PAD
pad_r = bb.x1 - (1 + panel_w) + REF_PAD
pad_b = 1 - bb.y0 + REF_PAD
pad_t = bb.y1 - (1 + panel_h) + REF_PAD
fig_ref_w = pad_l + panel_w + pad_r
fig_ref_h = pad_b + panel_h + pad_t
fig_ref.set_size_inches(fig_ref_w, fig_ref_h)
ax_ref.set_position(
    [pad_l / fig_ref_w, pad_b / fig_ref_h, panel_w / fig_ref_w, panel_h / fig_ref_h]
)
fig_ref.savefig("thefig_ref.pdf", dpi=200)


# ==========================================================================================
# ==========================================================================================
# Fused 4-panel time history figures (single column of the paper)
# ==========================================================================================
# ==========================================================================================
# One panel per run, the 4 panels fused without gaps (shared axes) like the time panels of
# the vertical slices figure, log time axis, run names above the panels and the legend above
# them. Used by the dust mass conservation and the CFL timestep figures below.

FUSED_T_MIN = 0.1  # yr, lower limit of the log time axis (first record after t = 0)
FUSED_T_MAX = 100.0  # yr
FUSED_XTICKS = [0.1, 1.0, 10.0, 100.0]  # labelled on the first panel only
FUSED_LINEAR_XTICKS = [0, 50, 100]  # same, for a linear time axis
FUSED_FIGSIZE = (3.4, 2.4)  # in, same width as thefig_ref.pdf
# box of the 4 panels in the figure (in): left, bottom, width, height
FUSED_PANELS_BOX = (0.603, 0.393, 2.694, 1.601)
FUSED_LW = 0.8


def load_records(sim, prefix):
    """All the analysis records <prefix>_*.npy of a run, in order."""
    folder = os.path.join(base, sim, "analysis")
    files = sorted(
        f for f in os.listdir(folder) if f.startswith(prefix + "_") and f.endswith(".npy")
    )
    return [np.load(os.path.join(folder, f), allow_pickle=True).item() for f in files]


def make_fused_axes(fig_h=FUSED_FIGSIZE[1], fig_w=FUSED_FIGSIZE[0], y_offset=0.0):
    """
    Figure with the 4 fused panels (shared x and y), returns (fig, axs, x range of the box).
    fig_h > FUSED_FIGSIZE[1] (fig_w > FUSED_FIGSIZE[0]) adds room on top (on the right): the
    panels keep their size and position from the bottom left, in inches. y_offset (in) moves
    the panels up, to make room below them.
    """
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=250)
    bx, by, bw, bh = FUSED_PANELS_BOX
    by += y_offset
    box_x = (bx / fig_w, (bx + bw) / fig_w)
    gs = fig.add_gridspec(
        1,
        len(sims),
        left=box_x[0],
        right=box_x[1],
        bottom=by / fig_h,
        top=(by + bh) / fig_h,
        wspace=0,
    )
    axs = [fig.add_subplot(gs[0, 0])]
    for i in range(1, len(sims)):
        axs.append(fig.add_subplot(gs[0, i], sharex=axs[0], sharey=axs[0]))
    return fig, axs, box_x


def fused_time_window(t):
    """(plotted, shown): records up to FUSED_T_MAX, and those inside the log time axis."""
    sel = (t >= 0) & (t <= FUSED_T_MAX)
    return sel, sel & (t >= FUSED_T_MIN)


def add_fused_dnf(ax, t_end, log_time=True):
    """Mark a run that stopped before FUSED_T_MAX (middle of the missing range)."""
    if t_end < FUSED_T_MAX * (1 - 1e-3):
        ax.text(
            np.sqrt(t_end * FUSED_T_MAX) if log_time else 0.5 * (t_end + FUSED_T_MAX),
            0.5,
            "DNF",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=plt.rcParams["font.size"],
            color="red",
        )


def decade_limits(values, linthresh=0.0):
    """Data range extended to the next decades (symmetric for negative values)."""
    lo, hi = np.nanmin(values), np.nanmax(values)
    if lo < -linthresh:
        lo = -(10.0 ** np.ceil(np.log10(-lo)))
    elif lo > linthresh:
        lo = 10.0 ** np.floor(np.log10(lo))
    if hi > linthresh:
        hi = 10.0 ** np.ceil(np.log10(hi))
    return lo, hi


def finish_fused_figure(fig, axs, box_x, ylabel, legend_handles, ncol=3, log_time=True):
    """
    Time axis (log from FUSED_T_MIN, or linear from 0 with FUSED_LINEAR_XTICKS), run names,
    y label, x label under the panels and legend above them.
    """
    ax0 = axs[0]
    if log_time:
        ax0.set_xscale("log")
        ax0.set_xlim(FUSED_T_MIN, FUSED_T_MAX)
        ax0.set_xticks(FUSED_XTICKS)
        ax0.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10), numticks=100))
        ax0.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax0.set_xlim(0, FUSED_T_MAX)
        ax0.set_xticks(FUSED_LINEAR_XTICKS)
        ax0.xaxis.set_minor_locator(AutoMinorLocator())
    ax0.set_ylabel(ylabel)
    for ax, (label, _) in zip(axs, sims):
        ax.set_title(label, fontsize=plt.rcParams["font.size"], pad=2)
    for ax in axs[1:]:
        ax.tick_params(labelleft=False, labelbottom=False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    # one x label centred under the 4 panels, below the tick labels
    ticks_bottom = min(to_fig.transform((0, ax.get_tightbbox(renderer).y0))[1] for ax in axs)
    fig.text(
        0.5 * (box_x[0] + box_x[1]), ticks_bottom, r"$t$ [yr]", ha="center", va="top",
        fontsize=plt.rcParams["font.size"],
    )
    if legend_handles is None:
        return
    # legend above the run names
    titles_top = max(to_fig.transform((0, ax.title.get_window_extent(renderer).y1))[1] for ax in axs)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5 * (box_x[0] + box_x[1]), titles_top),
        ncol=ncol,
        fontsize=plt.rcParams["font.size"],  # same as the axis labels
        frameon=False,
        labelspacing=0.25,
        handlelength=2.2,
        columnspacing=1.0,
        borderaxespad=0.2,
    )


# ==========================================================================================
# ==========================================================================================
# Dust mass conservation of the 4 simulations (disc + dust accreted by the sinks)
# ==========================================================================================
# ==========================================================================================
# Same quantity as the "dust_mass_history.png" plot of run_dustydisc.py (DustMassAnalysis):
# relative variation, since the injection, of the dust mass of each species in the disc plus
# the dust mass of that species accreted by the sinks (one colour per grain size), and of the
# total (dashed grey).

MASS_LINTHRESH = 1e-8  # linear range of the symlog y axis
MASS_TICK_DECADES = range(-8, 1, 2)  # labelled y ticks at +- 10^k
# grain size of each species (geometric centre of the bins of run_dustydisc.py)
mass_grain_size_si = 10.0 ** (np.arange(disc_ndust) - 6.0)  # 1 um ... 1 cm
mass_cmap = plt.colormaps["plasma"]
mass_norm = LogNorm(vmin=mass_grain_size_si.min(), vmax=mass_grain_size_si.max() * 10)
mass_colors = mass_cmap(mass_norm(mass_grain_size_si))
mass_ticks = sorted([0.0] + [sgn * 10.0**k for k in MASS_TICK_DECADES for sgn in (-1, 1)])


def load_dust_mass_history(sim):
    """time, dust mass in the disc and dust mass accreted by the sinks (per species)."""
    recs = load_records(sim, "dust_mass")
    return (
        np.array([r["time"] for r in recs]),
        np.array([r["dust_mass"] for r in recs], dtype=float),
        # records written before the sink quantities were added have none
        np.array([r.get("sink_dust_mass", [np.nan] * disc_ndust) for r in recs], dtype=float),
    )


with plt.rc_context({"font.size": plt.rcParams["font.size"] * REF_FONT_SCALE}):
    fig_mf, axs_mf, box_x = make_fused_axes()

    d_all = []  # all plotted values, for the y limits
    for ax, (label, sim) in zip(axs_mf, sims):
        t, dust_mass, sink_dust_mass = load_dust_mass_history(sim)
        m = dust_mass + np.nan_to_num(sink_dust_mass, nan=0.0)
        i0 = np.argmax(~np.isnan(dust_mass)[:, 0])
        t = t - t[i0]
        sel, shown = fused_time_window(t)

        for k in range(disc_ndust):
            d_k = m[:, k] / m[i0, k] - 1
            ax.plot(t[sel], d_k[sel], color=mass_colors[k], linewidth=FUSED_LW)
            d_all.append(d_k[shown])
        m_tot = m.sum(axis=1)
        d_tot = m_tot / m_tot[i0] - 1
        ax.plot(t[sel], d_tot[sel], color="grey", linestyle="--", linewidth=FUSED_LW)
        d_all.append(d_tot[shown])
        ax.axhline(0, color="0.5", linewidth=0.5, zorder=0)
        add_fused_dnf(ax, t[sel][-1])

    ax0 = axs_mf[0]
    ax0.set_yscale("symlog", linthresh=MASS_LINTHRESH)
    ylo, yhi = decade_limits(np.concatenate(d_all), MASS_LINTHRESH)
    ax0.set_yticks([y for y in mass_ticks if ylo <= y <= yhi])
    ax0.set_ylim(ylo, yhi)
    ax0.yaxis.set_minor_locator(plt.NullLocator())

    finish_fused_figure(
        fig_mf,
        axs_mf,
        box_x,
        r"$\delta M_{\mathrm{dust}} / M_{\mathrm{dust},0}$",
        [
            Line2D(
                [0], [0], color=mass_colors[k], linewidth=1,
                label=rf"$s=10^{{{np.log10(s):.0f}}}$ m",
            )
            for k, s in enumerate(mass_grain_size_si)
        ]
        + [Line2D([0], [0], color="grey", linestyle="--", linewidth=1, label="total")],
    )
    fig_mf.savefig("thefig_mass_fused.pdf", dpi=200, bbox_inches="tight", pad_inches=0.01)


# ==========================================================================================
# ==========================================================================================
# CFL timesteps of the 4 simulations
# ==========================================================================================
# ==========================================================================================
# Same quantities as the "cfl_history.png" plot of run_dustydisc.py (CFLHistoryAnalysis):
# the timestep of each CFL condition (without the CFL multiplier) and the global timestep
# (min of the conditions times the multiplier, dashed black). Conditions that never constrain
# the step (f64_max, e.g. sink-sink without sink pairs) are not shown.

# condition -> (label, colour), in the legend order
CFL_STYLES = {
    "cfl_dt": (r"$\Delta t_{\rm all}$", "black"),
    "cfl_dt_courant": (r"$\Delta t_{\rm cour}$", "tab:blue"),
    "cfl_dt_force": (r"$\Delta t_{\rm force}$", "tab:green"),
    "cfl_dt_dust1_fluid": (r"$\Delta t_{\rm 1-fluid}$", "tab:orange"),
    "cfl_dt_dust_drift": (r"$\Delta t_{\rm drift}$", "tab:red"),
    "cfl_dt_sink_sink": (r"$\Delta t_{\rm sink-sink}$", "tab:brown"),
}


def load_cfl_history(sim):
    """time and {condition: timesteps}, nan where unset (<= 0) or not constraining (f64_max)."""
    recs = load_records(sim, "cfl_history")
    t = np.array([r["time"] for r in recs])
    names = sorted({name for r in recs for name in r["cfl_dts"]})
    dts = {}
    for name in names:
        v = np.array([r["cfl_dts"].get(name, np.nan) for r in recs], dtype=float)
        v[(v <= 0) | (v > 1e300) | ~np.isfinite(v)] = np.nan
        dts[name] = v
    return t, dts


with plt.rc_context({"font.size": plt.rcParams["font.size"] * REF_FONT_SCALE}):
    fig_cf, axs_cf, box_x = make_fused_axes()

    dt_all = []  # all plotted values, for the y limits
    shown_names = set()
    for ax, (label, sim) in zip(axs_cf, sims):
        t, dts = load_cfl_history(sim)
        sel, shown = fused_time_window(t)
        # global timestep last, on top
        for name in sorted(dts, key=lambda n: n == "cfl_dt"):
            v = dts[name]
            if np.all(np.isnan(v[shown])):
                continue
            lab, color = CFL_STYLES.get(name, (name, None))
            is_all = name == "cfl_dt"
            ax.plot(
                t[sel], v[sel], color=color, linewidth=FUSED_LW,
                linestyle="--" if is_all else "-", zorder=3 if is_all else 2,
            )
            dt_all.append(v[shown])
            shown_names.add(name)
        add_fused_dnf(ax, t[sel][-1])

    ax0 = axs_cf[0]
    ax0.set_yscale("log")
    ax0.set_ylim(*decade_limits(np.concatenate(dt_all)))

    ordered = [n for n in CFL_STYLES if n in shown_names] + sorted(shown_names - set(CFL_STYLES))
    finish_fused_figure(
        fig_cf,
        axs_cf,
        box_x,
        r"$\Delta t$ [yr]",
        [
            Line2D(
                [0], [0], color=CFL_STYLES.get(n, (n, None))[1], linewidth=1,
                linestyle="--" if n == "cfl_dt" else "-", label=CFL_STYLES.get(n, (n,))[0],
            )
            for n in ordered
        ],
    )
    fig_cf.savefig("thefig_cfl_fused.pdf", dpi=200, bbox_inches="tight", pad_inches=0.01)


# ==========================================================================================
# ==========================================================================================
# Dust surface density at R = REF_R per Stokes number bin, for the 4 simulations
# ==========================================================================================
# ==========================================================================================
# From the vertical density slices (every 2.5 yr, hence the linear time axis): in the
# columns |x - REF_R| <= PROFILE_DX, each pixel of each dust species is put in the bin of its
# local Stokes number (same St as the contours: Epstein stopping time with the total density
# of the gas slice, LP07 sound speed), and Sigma_bin = sum over the column of rho_d dz, summed
# over the species and averaged over the columns. The dashed grey line is the total.

ST_BIN_EDGES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1.0, np.inf]
ST_BINS_YMIN = 1e-4  # kg m^-2, lower y limit
# horizontal discrete colour bar below the panels (in): gap below the x label, thickness
ST_CBAR_GAP = 0.08
ST_CBAR_H = 0.08
# right hand axis on the last panel: the same Sigma converted to the dust momentum per unit
# area Sigma v_K, assuming Keplerian dust (the slices hold no velocities). At fixed R it is a
# rescaling of Sigma, so the curves read both ways.
ST_BINS_MOMENTUM_AXIS = False
st_bin_colors = plt.colormaps["viridis"](np.linspace(0, 0.95, len(ST_BIN_EDGES) - 1))
# colour bar tick labels at the finite bin edges; the last bin (St > 1) is the end triangle
st_edge_labels = [
    rf"$10^{{{np.log10(e):.0f}}}$" if np.log10(e) == np.round(np.log10(e)) and e < 1 else f"{e:g}"
    for e in ST_BIN_EDGES[:-1]
]


def sigma_per_st_bin(sim):
    """time [yr] and Sigma [kg m^-2] of the dust in each St bin, at R = REF_R."""
    times, sigmas = [], []
    for dump in range(10000):
        gas_prefix = os.path.join(base, sim, "analysis/plots", f"rho_slice_gas_{dump:07d}")
        if not os.path.exists(gas_prefix + ".npy"):
            break
        rho = np.load(gas_prefix + ".npy")
        with open(gas_prefix + ".json") as f:
            meta = json.load(f)
        if meta["time"] > FUSED_T_MAX * (1 + 1e-6):
            break
        x0, x1, z0, z1 = meta["extent"]
        ny, nx = rho.shape
        xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        zs = z0 + (np.arange(ny) + 0.5) * (z1 - z0) / ny
        cols = np.abs(xs - REF_R) <= PROFILE_DX
        X, Z = np.meshgrid(xs[cols], zs)
        rho = rho[:, cols]
        omega_k = np.sqrt(G_code * center_mass / X**3)
        cs = cs0_code * (np.sqrt(X**2 + Z**2) / disc_r0) ** (-disc_q)
        dz_m = (z1 - z0) / ny * au_m

        sigma = np.zeros(len(ST_BIN_EDGES) - 1)
        for j in range(disc_ndust):
            rho_d = np.load(
                os.path.join(base, sim, "analysis/plots", f"rho_slice_dust_{j}_{dump:07d}.npy")
            )[:, cols]
            rho_d = np.nan_to_num(rho_d, nan=0.0) * codeu_to_kg_m3
            with np.errstate(all="ignore"):
                st = (
                    rho_grain_si / codeu_to_kg_m3 * grain_size_au(j) / (rho * cs)
                    * np.sqrt(np.pi * gamma_ad / 8) * omega_k
                )
            st = np.where(np.isfinite(st), st, np.inf)  # no gas: St -> infinity
            if np.any((st < ST_BIN_EDGES[0]) & (rho_d > 0)):
                print(f"warning: {sim} t={meta['time']} dust {j} has St < {ST_BIN_EDGES[0]}")
            ibin = np.clip(np.searchsorted(ST_BIN_EDGES, st, side="right") - 1, 0, len(sigma) - 1)
            for b in range(len(sigma)):
                # column integral, averaged over the columns
                sigma[b] += (rho_d * (ibin == b)).sum(axis=0).mean() * dz_m
        times.append(meta["time"])
        sigmas.append(sigma)
    return np.array(times), np.array(sigmas)


st_bin_data = [sigma_per_st_bin(sim) for _, sim in sims]


def make_st_bins_figure(fig_h, fig_w, y_offset):
    fig_sb, axs_sb, box_x = make_fused_axes(fig_h, fig_w, y_offset)

    sig_all = []
    for ax, (t, sig) in zip(axs_sb, st_bin_data):
        for b in range(sig.shape[1]):
            ax.plot(t, sig[:, b], color=st_bin_colors[b], linewidth=FUSED_LW)
        ax.plot(t, sig.sum(axis=1), color="grey", linestyle="--", linewidth=FUSED_LW)
        sig_all += [sig[sig > 0], sig.sum(axis=1)]
        add_fused_dnf(ax, t[-1], log_time=False)

    ax0 = axs_sb[0]
    ax0.set_yscale("log")
    ax0.set_ylim(ST_BINS_YMIN, decade_limits(np.concatenate(sig_all))[1])
    if ST_BINS_MOMENTUM_AXIS:
        v_k = np.sqrt(G_code * center_mass / REF_R) * au_m / yr_s  # m s^-1
        sec = axs_sb[-1].secondary_yaxis(
            "right", functions=(lambda y: y * v_k, lambda y: y / v_k)
        )
        sec.set_ylabel(r"$\Sigma_{\mathrm{d}} v_{\mathrm{K}}$ [kg m$^{-1}$ s$^{-1}$]")

    finish_fused_figure(
        fig_sb,
        axs_sb,
        box_x,
        rf"$\Sigma_{{\mathrm{{d}}}}(R={REF_R:.0f}\,\mathrm{{au}})$ [kg m$^{{-2}}$]",
        None,
        log_time=False,
    )
    # "total" in a box inside the first panel, in the empty area below the curves
    ax0.legend(
        handles=[Line2D([0], [0], color="grey", linestyle="--", linewidth=1, label="total")],
        loc="lower left",
        fontsize=plt.rcParams["font.size"],
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.4,
    )

    # horizontal discrete colour bar below the x label, one block per St bin (as in
    # dustysettle_limiters.py), the open last bin (St > 1) drawn as the triangular end
    renderer = fig_sb.canvas.get_renderer()
    labels_bottom = min(t.get_window_extent(renderer).y0 for t in fig_sb.texts) / fig_sb.dpi
    bx, by, bw, bh = FUSED_PANELS_BOX
    cax = fig_sb.add_axes(
        [
            bx / fig_w,
            (labels_bottom - ST_CBAR_GAP - ST_CBAR_H) / fig_h,
            bw / fig_w,
            ST_CBAR_H / fig_h,
        ]
    )
    cmap = ListedColormap(st_bin_colors[:-1])
    cmap.set_over(st_bin_colors[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=BoundaryNorm(ST_BIN_EDGES[:-1], cmap.N))
    cbar = fig_sb.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        extend="max",
        spacing="uniform",
        drawedges=True,
        ticks=ST_BIN_EDGES[:-1],
    )
    cbar.ax.set_xticklabels(st_edge_labels)
    cbar.ax.minorticks_off()
    cbar.set_label("St")
    return fig_sb


with plt.rc_context({"font.size": plt.rcParams["font.size"] * REF_FONT_SCALE}):
    # build once with a guess of the room below the panels, then redo the figure so that
    # what overflows (colour bar below, labels on the right) fits and the margins below and on
    # top are only `pad`, also in the window
    y_off, fig_h, fig_w = 0.6, FUSED_FIGSIZE[1] + 0.6, FUSED_FIGSIZE[0]
    for _ in range(3):
        fig_sb = make_st_bins_figure(fig_h, fig_w, y_off)
        fig_sb.canvas.draw()
        bb = fig_sb.get_tightbbox(fig_sb.canvas.get_renderer())  # in
        pad = 0.02
        d_bottom, d_top, d_right = pad - bb.y0, bb.y1 + pad - fig_h, bb.x1 + pad - fig_w
        if max(abs(d_bottom), abs(d_top), max(d_right, 0)) < 1e-3:
            break
        plt.close(fig_sb)
        y_off += d_bottom
        fig_h += d_bottom + d_top  # also removes empty room on top
        fig_w += max(d_right, 0)
    fig_sb.savefig("thefig_st_bins_fused.pdf", dpi=200, bbox_inches="tight", pad_inches=0.01)

plt.show()
