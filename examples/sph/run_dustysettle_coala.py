"""
Dusty settling SPH test with COALA coagulation
===============================================

Perform a dust settling test in a local stratified box, with the dust size
distribution evolved by the COALA coagulation source term.

If you want to change the resolution, you can set the LZ environment variable.

Example (from build directory):
```bash
LZ=96 ./shamrock --sycl-cfg 0:0 --smi --loglevel 1 --rscript ../examples/sph/run_dustysettle_coala.py
```
"""

# sphinx_gallery_multi_image = "single"

# %%
# Imports
# ------------------------------------------

import json
import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import PillowWriter
from matplotlib.lines import Line2D
from scipy.special import erfinv
from shamrock.external import coala
from shamrock.utils.DustMRNDistribution import DustMRNDistribution
from shamrock.utils.plot import show_image_sequence

import shamrock
from shamrock import NeighCacheStrategy

# %%
# Shamrock initialization
# ------------------------------------------

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
# ------------------------------------------

shamrock.matplotlib.set_shamrock_mpl_style()

# Same sizes as the other figures of the paper (A&A single column), from fontsize = 8:
# labels fontsize + 1, ticks fontsize - 0.5, legends and small text fontsize - 1
mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "figure.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "axes.labelpad": 2,
        "axes.titlepad": 4,
        "axes.linewidth": 0.6,
        "legend.framealpha": 1.0,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
    }
)
if "legend.linewidth" in mpl.rcParams:  # only in matplotlib >= 3.11
    mpl.rcParams["legend.linewidth"] = 0.6

# %%
# Define simulation parameters and unit system
# ------------------------------------------

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

# Disc / box
rho_i_si = 4e-11  # Initial density [kg/m^3]
rho_i = rho_i_si * codeu.get("kg") * codeu.get("m") ** -3
rho_i *= 0.2  # the input density does not match exactly the stable one
central_mass = 1  # Central mass
R0 = 20  # Where are we in radius
H_r_0 = 0.05  # Disc aspect ratio
gamma = 1.4  # Adiabatic index
box_H_count = 8  # box size in number of H
random_vel_pert = True  # add a small vecolity pertubation to fasten the setup

# Dust
ndust = 5  # not below 20 in prod otherwise the rate is too low
mrn_pow = 3.5
mrn_cutoff_si = 250e-9  # like standard MRN
epsilon_base = 0.01
rho_grains_si_edges = np.array([2.3 * 1000 for i in range(ndust + 1)])
grain_size_si_edges = np.logspace(-9, -2, ndust + 1)

# COALA coagulation
use_coala = True
dv_max = 1000000 * codeu.get("m") / codeu.get("s")
Q = 5  # number of points for the Gauss-Legendre quadrature
rhodust_eps = 1e-17  # dust density floor
tau_frac = 10.0  # the source term step moves at least tau_frac * rhodust_eps
K0_multiplier = 1

# Resolution (lattice size)
lx = 12
ly = 12
lz = int(os.environ.get("LZ", "64"))

# Time, in units of the orbital period at R0
t_orb = 2 * np.pi * np.sqrt(R0**3 / (ucte.G() * central_mass))
tinject = 2.0 * t_orb  # time at which the dust is injected
t_end = tinject + 3.0 * t_orb  # use np.inf instead for long runs
dt_analysis = 0.1 * t_orb  # time between two analysis plots

# Scheduler
scheduler_split_val = int(2e7)
scheduler_merge_val = 1

# Artificial viscosity
av_alpha_min = 0.0
av_alpha_max = 1
av_sigma_decay = 0.1
av_alpha_u = 1
av_beta_AV = 2
vel_dissipation_eta = 5 / t_orb

# CFL (before dust injection)
cfl_cour_setup = 0.25
cfl_force_setup = 0.25
# CFL (after dust injection)
cfl_cour_inject = 0.1
cfl_force_inject = 0.1

# Injection check
max_v_inject_threshold = 3.0  # in units of cs

# Paths
sim_folder = f"_to_trash/dusty_settle_{lz}_coala_{use_coala}/"
dump_folder = sim_folder + "dump/"

# Plotting
cmap = "plasma"
St_levels = [1e-3, 1e-2, 1e-1, 1.0]  # Stokes number contours on the rho_d(z, s) map
# the density plot ranges were set for rho_i = 1e-6 (code units), scale them with rho_i
rho_plot_scale = rho_i / 1e-6
dpi = 250
SINGLE_COLUMN_WIDTH = 88 / 25.4  # in, A&A single column

# %%
# Derived physical scales and lattice geometry
# ------------------------------------------

print("codeu.get('m') / codeu.get('s') =", codeu.get("m") / codeu.get("s"))
print("codeu.to('m') / codeu.to('s') =", codeu.to("m") / codeu.to("s"))

G = ucte.G()


def kep_profile(r):
    return (G * central_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


H = H_r_0 * R0
cs = H * omega_k(R0)
box = box_H_count * H

print(f"cs = {cs}")
print(f"H = {H}")

mrn_distribution = DustMRNDistribution(
    codeu, mrn_pow, mrn_cutoff_si, grain_size_si_edges, rho_grains_si_edges
)

massgrid_edges = mrn_distribution.massgrid_edges

if use_coala:
    print(f"massgrid = {mrn_distribution.massgrid} [code units]")
    print(f"massgrid = {mrn_distribution.massgrid * codeu.to('kg')} [kg]")

    K0 = np.pi * ((4.0 / 3.0) * np.pi * mrn_distribution.rho_grains[0]) ** (-2.0 / 3.0)
    K0 *= K0_multiplier
    print(f"K0 = {K0}")

    tabflux_coag = coala.coala_precalc_tabflux_coag(K0, ndust, Q, massgrid_edges)

lmin = (-(lx // 2), -(ly // 2), -(lz // 2))
lmax = (lx // 2, ly // 2, lz // 2)

# Call with dr = 1 as we will rescale on next call
(xm, ym, zm), (xM, yM, zM) = shamrock.math.get_periodic_hcp_box(1.0, lmin, lmax)
print(f"base lattice : xM = {xM}, yM = {yM}, zM = {zM}")
dr = 2 * box / (zM - zm)
print(f"dr = {dr}")
bmin, bmax = shamrock.math.get_periodic_hcp_box(dr, lmin, lmax)
print(f"new lattice : bmin = {bmin}, bmax = {bmax}")
xm, ym, zm = bmin
xM, yM, zM = bmax

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho_i * vol_b
print("Total mass :", totmass)

pmass = -1

# %%
# Initial conditions and dust injection functions
# ------------------------------------------

cs_g = cs


def scaling_rho(r):
    x, y, z = r

    loc_h = H / (2**0.5)
    gaussian = np.exp(-(y**2) / (2 * loc_h * loc_h)) / (loc_h * np.sqrt(2 * np.pi))
    return gaussian


def func_rho_t(r):
    return rho_i * scaling_rho(r)


def func_rho_d_j(r, idust):
    return (0.1 / ndust) * rho_i * scaling_rho(r)


def func_rho_g(r):
    return rho_i * scaling_rho(r) - sum([func_rho_d_j(r, i) for i in range(ndust)])


def uint_g(r):
    rho_g = func_rho_g(r)
    P = rho_g * cs_g * cs_g / gamma
    return P / ((gamma - 1) * rho_g)


def remap_positions_z(r):
    x, y, z = r

    rn = max(abs(zM), abs(zm))
    z = H * erfinv(z / rn)

    z = min(z, zM)
    z = max(z, zm)
    return (x, y, z)


def compute_sj_new_j(patchdata, j):
    pmass = model.get_particle_mass()

    hpart = patchdata["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    z = patchdata["xyz"][:, 2]
    # mask to only modify particles with |z| < H
    mask = 1 / (1 + np.exp((np.abs(z) - 1.75 * H) / (H / 16)))

    epsilon_target = epsilon_base * mrn_distribution.mrn_weight[j] * mask
    print(f"epsilon_target = {epsilon_target} {j}")
    s = np.sqrt(rho * epsilon_target)

    print(
        f"s = {s} {np.isnan(s).any()} epsilon_target = {epsilon_target} mrn_weight = {mrn_distribution.mrn_weight[j]} mask = {mask}, rho = {rho}"
    )

    return s


# %%
# Analysis helpers
# ------------------------------------------


def save_analysis_data(filename, key, value, ianalysis):
    """Helper to save analysis data to a JSON file."""
    if shamrock.sys.world_rank() == 0:
        filepath = os.path.join(dump_folder, filename)
        try:
            with open(filepath, "r") as fp:
                data = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {key: []}
        data[key] = data[key][:ianalysis]
        data[key].append({"t": model.get_time(), key: value})
        with open(filepath, "w") as fp:
            json.dump(data, fp, indent=4)


def load_data_from_json(filename, key):
    """Helper to load analysis data from a JSON file."""
    filepath = os.path.join(dump_folder, filename)
    with open(filepath, "r") as fp:
        data = json.load(fp)[key]
    t = [d["t"] for d in data]
    values = [d[key] for d in data]
    return t, values


def get_max_v():
    dic = ctx.collect_data()
    v = dic["vxyz"]
    vnorms = np.linalg.norm(v, axis=1)
    return vnorms.max()


def compute_delta_v_eff(s_j, delta_v):
    """Effective collision speed between the dust bins (same as in run_dustydisc.py)

    delta_v_eff = sum_i sum_j rho_i rho_j |dv_i - dv_j| / (sum_i rho_i)^2
    """
    rho_d = s_j**2

    # sum_i sum_j rho_i * rho_j == (sum_i rho_i)**2
    denom = np.sum(rho_d, axis=1) ** 2

    # dv_ii == 0 and dv_ij == dv_ji, so only the i<j pairs are needed, each counted twice
    numer = np.zeros(rho_d.shape[0])
    for i in range(ndust):
        for j in range(i + 1, ndust):
            dv_ij = np.linalg.norm(delta_v[:, i, :] - delta_v[:, j, :], axis=-1)
            numer += 2 * rho_d[:, i] * rho_d[:, j] * dv_ij

    # nan before the dust injection (no dust at all)
    with np.errstate(invalid="ignore", divide="ignore"):
        return numer / denom


# %%
# Plotting
# ------------------------------------------


def analyse_and_plot(j):

    pmass = model.get_particle_mass()
    hfact = model.get_hfact()

    dic = ctx.collect_data()
    print(dic["s_j"])

    print(dic["xyz"].shape)

    x = dic["xyz"][:, 0]
    y = dic["xyz"][:, 1]
    z = dic["xyz"][:, 2]
    s_j = dic["s_j"].reshape(-1, ndust)
    ds_j_dt = dic["ds_j_dt"].reshape(-1, ndust)
    cs = dic["soundspeed"]
    delta_v = dic["delta_v"].reshape(-1, ndust, 3)

    print(s_j)

    hpart = dic["hpart"]
    rho = pmass * (hfact / np.array(hpart)) ** 3

    print("compute original rho")
    estimated_rho = [func_rho_t(dic["xyz"][kk]) for kk in range(len(dic["xyz"]))]

    sz = 1

    to_dens = codeu.to("kg") * codeu.to("m") ** -3
    to_speed = codeu.to("m") / codeu.to("s")

    dust_cmap = plt.colormaps[cmap]
    # same range as the grain size axis of the rho_d(z, s) map
    dust_norm = mcolors.LogNorm(vmin=grain_size_si_edges[0], vmax=grain_size_si_edges[-1])
    dust_colors = dust_cmap(dust_norm(mrn_distribution.grain_size_si))

    dust_mass = analysis_dust_mass.get_dust_mass()

    # if all dust mass is zero replace by nans
    if np.max(dust_mass) == 0:
        print("all dust mass is zero, replacing by nans")
        dust_mass = [np.nan for _ in range(ndust)]

    save_analysis_data("dust_mass.json", "dust_mass", dust_mass, j)

    range_plot = 2.5 * H
    # rounded outwards to powers of 10
    rho_plot_min = 10 ** np.floor(np.log10(1e-20 * rho_plot_scale))
    rho_plot_max = 10 ** np.ceil(np.log10(1e-8 * rho_plot_scale))

    time = model.get_time() - tinject
    os.makedirs(f"{dump_folder}/plots", exist_ok=True)

    rho_d_j = s_j**2
    rho_d = np.sum(rho_d_j, axis=1)

    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
    dust_sm.set_array([])

    # <rho_d>(z, s): dust density of each bin averaged in z slices (as the vert_slices
    # plots of run_dustydisc.py), nan in the empty slices
    z_edges = np.linspace(-range_plot, range_plot, lz + 1)
    count_z, _ = np.histogram(z, bins=z_edges)
    rho_d_j_z = np.zeros((ndust, lz))
    for i in range(ndust):
        sum_z, _ = np.histogram(z, bins=z_edges, weights=rho_d_j[:, i] * to_dens)
        with np.errstate(invalid="ignore"):
            rho_d_j_z[i] = sum_z / count_z

    # St(z, s) = t_s * Omega_K from the gas density and sound speed averaged in the same z slices
    with np.errstate(invalid="ignore"):
        rho_g_z = np.histogram(z, bins=z_edges, weights=rho - rho_d)[0] / count_z
        cs_z = np.histogram(z, bins=z_edges, weights=cs)[0] / count_z
    z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])
    s_St_si = np.logspace(np.log10(grain_size_si_edges[0]), np.log10(grain_size_si_edges[-1]), 200)
    St_z_s = np.full((len(s_St_si), lz), np.nan)
    for iz in np.flatnonzero(count_z > 0):
        for i_s, s_si in enumerate(s_St_si):
            ts = shamrock.phys.epstein_stopping_time(
                rho_grain=mrn_distribution.rho_grains[0],  # same grain density in all bins
                s_grain=s_si * codeu.get("m"),
                rho=rho_g_z[iz],
                cs=cs_z[iz],
                gamma=gamma,
            )
            St_z_s[i_s, iz] = ts * omega_k(R0)

    # rho(z) of each particle over <rho_d>(z, s), stacked
    fig = plt.figure(figsize=(SINGLE_COLUMN_WIDTH, 5), layout="constrained", dpi=dpi)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0, hspace=0)
    ax_rho, ax_rho_s = fig.subplots(2, 1, sharex=True)
    fig.suptitle(f"t = {time:.2f} [yr]")

    for i in range(ndust):
        ax_rho.scatter(z, rho_d_j[:, i] * to_dens, s=sz, color=dust_colors[i], edgecolors="none")
    ax_rho.scatter(z, (rho - rho_d) * to_dens, s=sz, color="0.0", edgecolors="none")
    ax_rho.scatter(z, rho_d * to_dens, s=sz, color="0.5", edgecolors="none")

    ax_rho.set_ylabel(r"$\rho$ [kg/m$^3$]")
    ax_rho.set_yscale("log")
    ax_rho.set_ylim(1.1 * rho_plot_min, rho_plot_max)  # 1.1 to avoid the tick at the junction

    handles = [
        Line2D([0], [0], linestyle="none", marker="o", color=c, markeredgecolor="none")
        for c in ["0.0", "0.5"]
    ]
    shamrock.matplotlib.add_cmap_legend_entry(
        ax_rho,
        dust_cmap,
        label="dust species",
        extra_handles=handles,
        extra_labels=["gas", "dust sum"],
        loc="upper right",
        fontsize=mpl.rcParams["legend.fontsize"],
    )

    cbar = fig.colorbar(dust_sm, ax=ax_rho)
    cbar.set_label(r"grain size $s$ [m]")

    rho_norm = mcolors.LogNorm(vmin=rho_plot_min, vmax=rho_plot_max)
    im = ax_rho_s.pcolormesh(
        z_edges,
        grain_size_si_edges,
        np.where(rho_d_j_z <= 0, 1e-30, rho_d_j_z),
        cmap=dust_cmap,
        norm=rho_norm,
        shading="auto",
        rasterized=True,
    )
    St_contours = ax_rho_s.contour(
        z_centers, s_St_si, St_z_s, levels=St_levels, colors="white", linewidths=0.6
    )
    # labels all at the same z, slightly left of the midplane, on the closest slice
    # (St is linear in s there)
    z_label = -0.25 * range_plot
    iz_label = np.flatnonzero(count_z > 0)[np.argmin(np.abs(z_centers[count_z > 0] - z_label))]
    log_St_label = np.log10(St_z_s[:, iz_label])
    St_label_pos = [
        (z_centers[iz_label], 10 ** np.interp(np.log10(St), log_St_label, np.log10(s_St_si)))
        for St in St_levels
        if log_St_label[0] <= np.log10(St) <= log_St_label[-1]
    ]
    St_labels = ax_rho_s.clabel(
        St_contours,
        fmt=lambda St: "St$=1$" if St == 1 else f"St$=10^{{{np.log10(St):.0f}}}$",
        fontsize=mpl.rcParams["legend.fontsize"] - 1,
        manual=St_label_pos,
    )
    for label in St_labels:
        label.set_rotation(0)
    ax_rho_s.grid(False, which="both")
    ax_rho_s.set_ylabel(r"grain size $s$ [m]")
    ax_rho_s.set_xlabel(r"$z$")
    ax_rho_s.set_xlim(-range_plot, range_plot)
    ax_rho_s.set_yscale("log")
    ax_rho_s.set_ylim(grain_size_si_edges[0], grain_size_si_edges[-1])

    cbar = fig.colorbar(im, ax=ax_rho_s)
    cbar.set_label(r"$\langle \rho_{\rm d}(z, s) \rangle$ [kg/m$^3$]")

    fig.align_ylabels()

    plt.savefig(f"{dump_folder}/plots/vert_slice_dens_{j:04d}.png")
    plt.close()

    # <s>(z) and its growth rate (as compute_s_mean_field and
    # compute_dlog_s_mean_dt_field in run_dustydisc.py), nan without dust
    grain_size_si = mrn_distribution.grain_size_si
    drho_d_j_dt = 2 * s_j * ds_j_dt
    with np.errstate(invalid="ignore", divide="ignore"):
        s_mean = np.sum(rho_d_j * grain_size_si, axis=1) / rho_d
        ds_mean_dt = (
            np.sum(drho_d_j_dt * grain_size_si, axis=1) - s_mean * np.sum(drho_d_j_dt, axis=1)
        ) / rho_d
        dlog_s_mean_dt = ds_mean_dt / s_mean

    # delta v(z), <s>(z) and its growth rate, stacked
    fig = plt.figure(figsize=(SINGLE_COLUMN_WIDTH, 6.5), layout="constrained", dpi=dpi)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0, hspace=0)
    ax_dv, ax_s_mean, ax_s_evol = fig.subplots(3, 1, sharex=True)
    fig.suptitle(f"t = {time:.2f} [yr]")

    for i in range(ndust):
        c = dust_colors[i]
        # null where the bin is below the dust density floor
        resolved = rho_d_j[:, i] > rhodust_eps
        ax_dv.scatter(z, delta_v[:, i, 2] * resolved * to_speed, s=sz, color=c, edgecolors="none")

    delta_v_eff = compute_delta_v_eff(s_j, delta_v)
    isort = np.argsort(z)
    ax_dv.plot(
        z[isort],
        delta_v_eff[isort] * to_speed,
        color="0.0",
        linewidth=1,
        label=r"$\Delta v_{\rm eff}$",
    )
    ax_dv.legend(loc="upper right")

    ax_dv.set_ylabel(r"$\Delta v_{z,j} \, \mathbb{1}_{\rho_{{\rm d},j} > \rho_\epsilon}$ [m/s]")
    ax_dv.set_yscale("symlog", linthresh=1e-1)
    ax_dv.set_ylim(-1e2, 1e2)

    cbar = fig.colorbar(dust_sm, ax=ax_dv)
    cbar.set_label(r"grain size $s$ [m]")

    ax_s_mean.scatter(z, s_mean, s=sz, color="0.0", edgecolors="none")
    ax_s_mean.set_ylabel(r"$\langle s \rangle$ [m]")
    ax_s_mean.set_yscale("log")
    # 1.1 to avoid the ticks at the junctions
    ax_s_mean.set_ylim(1.1 * grain_size_si_edges[0], grain_size_si_edges[-1] / 1.1)

    ax_s_evol.scatter(z, dlog_s_mean_dt, s=sz, color="0.0", edgecolors="none")
    ax_s_evol.set_ylabel(r"$\partial_t \langle s \rangle / \langle s \rangle$ [yr$^{-1}$]")
    ax_s_evol.set_xlabel(r"$z$")
    ax_s_evol.set_xlim(-range_plot, range_plot)
    ax_s_evol.set_yscale("symlog", linthresh=1e-5)
    ax_s_evol.set_ylim(-1e-2, 1e-2)

    fig.align_ylabels()

    plt.savefig(f"{dump_folder}/plots/vert_slice_dv_{j:04d}.png")
    plt.close()

    fig = plt.figure(figsize=(SINGLE_COLUMN_WIDTH, 5), layout="constrained", dpi=dpi)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0, hspace=0)
    axs = fig.subplots(2, 1, sharex=True)
    fig.suptitle(f"t = {time:.2f} [yr]")

    for i in range(ndust):
        c = dust_colors[i]
        axs[0].scatter(z, s_j[:, i], s=sz, color=c, edgecolors="none")
        axs[1].scatter(z, ds_j_dt[:, i], s=sz, color=c, edgecolors="none")

    axs[0].set_ylabel(r"$s_j$")
    axs[0].set_xlim(-4 * H, 4 * H)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-20 * rho_plot_scale**0.5, 1e-1 * rho_plot_scale**0.5)  # s_j = sqrt(rho_d)

    axs[1].set_ylabel(r"$\dot{s}_j$")
    axs[1].set_xlabel(r"$z$")
    axs[1].set_yscale("symlog", linthresh=1e-10 * rho_plot_scale**0.5)

    fig.align_ylabels()

    cbar = fig.colorbar(dust_sm, ax=axs)
    cbar.set_label(r"grain size $s$ [m]")

    plt.savefig(f"{dump_folder}/plots/vert_slice_s_{j:04d}.png")
    plt.close()


# %%
# Simulation setup and restore
# ------------------------------------------

if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)
    os.makedirs(dump_folder, exist_ok=True)

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

dump_helper = shamrock.utils.dump.ShamrockDumpHandleHelper(model, dump_folder)


def setup_model():
    global bmin, bmax

    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=av_alpha_min,
        alpha_max=av_alpha_max,
        sigma_decay=av_sigma_decay,
        alpha_u=av_alpha_u,
        beta_AV=av_beta_AV,
    )

    cfg.set_dust_mode_monofluid_tva(
        nvar=ndust, C_1_fluid=0.1, C_drift=1.0, cfl_density_threshold=1e-50
    )
    cfg.set_dust_drag_epstein(gamma, mrn_distribution.grain_size, mrn_distribution.rho_grains)
    if use_coala:
        cfg.set_dust_evol_coala_coag(
            rhodust_eps, dv_max, massgrid_edges, tabflux_coag, tau_frac=tau_frac
        )
    cfg.add_ext_force_vertical_disc_potential(central_mass=central_mass, R0=R0)
    cfg.add_ext_force_velocity_dissipation(eta=vel_dissipation_eta)
    cfg.set_neigh_cache_strategy(NeighCacheStrategy.SingleStage)
    cfg.set_show_cfl_detail(True)
    cfg.set_boundary_periodic()
    cfg.set_units(codeu)
    cfg.set_eos_isothermal(cs)
    cfg.print_status()
    model.set_solver_config(cfg)

    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen, insert_step=scheduler_split_val)

    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)
    print("Current part mass :", pmass)

    # Correct the barycenter
    analysis_barycenter = shamrock.model_sph.analysisBarycenter(model=model)
    barycenter, disc_mass = analysis_barycenter.get_barycenter()

    if shamrock.sys.world_rank() == 0:
        print(f"disc barycenter = {barycenter}")

    model.apply_position_offset((-barycenter[0], -barycenter[1], -barycenter[2]))

    model.remap_positions(remap_positions_z)
    model.set_field_value_lambda_f64("uint", uint_g)

    if random_vel_pert:
        rng = np.random.default_rng(42)

        ampl = dr / t_orb
        print(f"random velocity perturbation amplitude = {ampl}")

        def pert_func(r):
            return tuple(rng.uniform(-ampl, ampl, size=3))

        model.set_field_value_lambda_f64_3("vxyz", pert_func)

    model.set_cfl_cour(cfl_cour_setup)
    model.set_cfl_force(cfl_force_setup)

    model.timestep()


analysis_dust_mass = shamrock.model_sph.analysisDustMass(model=model)
pmass = model.get_particle_mass()


# %%
# Run simulation
# ------------------------------------------

from shamrock.utils.SimulationRunner import SimulationRunner, callback, simulation_setup


class Simulation(SimulationRunner):
    # Use the global vars defined at the top of the file
    t_end = t_end
    dump_prefix = dump_folder + "dump_"

    @callback(at_tsim=[tinject])
    def inject_dust(self, _):
        max_v = get_max_v()
        print(f"max_v = {max_v}")

        if max_v > max_v_inject_threshold * cs:
            raise ValueError("max_v is too high, please increase the injection time")

        for k in range(ndust):

            def compute_sj_new(patchdata):
                return compute_sj_new_j(patchdata, k)

            self.model.overwrite_field_value_f64("s_j", compute_sj_new, k)

            self.model.set_cfl_cour(cfl_cour_inject)
            self.model.set_cfl_force(cfl_force_inject)

        self.model.set_dt(0.0)  # to help the corrector on next step after adding dust

    @callback(tsim_interval=dt_analysis)
    def analysis_plots(self, j):
        analyse_and_plot(j)

    @callback(walltime_interval=30.0)  # Checkpoint the simulation every 30 seconds
    def checkpoint(self, icheckpoint):
        self.do_checkpoint(icheckpoint, purge_old_dumps=True, keep_first=1, keep_last=3)

    @simulation_setup
    def setup(self):
        setup_model()


Simulation(model).run()

# %%
# Build animations from plot sequences
# ------------------------------------------

for plot_name in ["dens", "dv", "s"]:
    glob_str = f"{dump_folder}/plots/vert_slice_{plot_name}_*.png"
    ani = show_image_sequence(glob_str)

    writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(f"_to_trash/dustysettle_vert_slice_{plot_name}_coala.gif", writer=writer)

    if shamrock.sys.world_rank() == 0:
        plt.show()

# %%
# Plot dust mass conservation history
# ------------------------------------------

t, dust_mass = load_data_from_json("dust_mass.json", "dust_mass")
dust_mass = np.array(dust_mass)

# first analysis with dust (the dust mass is nan before the injection)
iinject = np.argmax(~np.isnan(dust_mass)[:, 0])

t = np.array(t) - tinject

St = np.zeros(ndust)

for k in range(ndust):
    t_dyn = 1
    ts = shamrock.phys.epstein_stopping_time(
        rho_grain=mrn_distribution.rho_grains[k],
        s_grain=mrn_distribution.grain_size[k],
        rho=rho_i,
        cs=cs,
        gamma=gamma,
    )
    St[k] = ts / t_dyn

for k in range(ndust):
    print(f"dust {k}, s = {mrn_distribution.grain_size_si[k]:.1e} [m], St = {St[k]:.1e}")

dust_cmap = plt.colormaps[cmap]
# same range as the grain size axis of the rho_d(z, s) map
dust_norm = mcolors.LogNorm(vmin=grain_size_si_edges[0], vmax=grain_size_si_edges[-1])

fig = plt.figure(figsize=(SINGLE_COLUMN_WIDTH, 3), layout="constrained", dpi=dpi)
fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
ax = fig.subplots()
for k in range(ndust):
    mh = dust_mass[:, k]
    deviation = (mh / mh[iinject]) - 1

    ax.plot(t, deviation, color=dust_cmap(dust_norm(mrn_distribution.grain_size_si[k])))

total_dust_mass = np.sum(dust_mass, axis=1)
ax.plot(
    t,
    (total_dust_mass / total_dust_mass[iinject]) - 1,
    color="grey",
    label="total dust mass",
    linestyle="--",
)

ax.set_xlabel("t")
ax.set_ylabel("$\\delta M_{dust} / M_{dust,0}$")
ax.set_yscale("symlog", linthresh=1e-8)
# coagulation moves mass between bins, so only the total dust mass is conserved
ax.set_title("Dust mass evolution")
ax.legend()

dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
dust_sm.set_array([])
cbar = fig.colorbar(dust_sm, ax=ax)
cbar.set_label(r"grain size $s$ [m]")

plt.savefig(f"{dump_folder}/plots/dust_mass_history.png")
plt.show()

# %%

print("##### Test result #####")
result = {
    "t": float(t[-1]),
    "dust_mass": (dust_mass[-1] / dust_mass[iinject] - 1).tolist(),
    "St": St.tolist(),
    "lz": lz,
}
print(result)

json.dump(result, open(f"{dump_folder}/test_result.json", "w"), indent=4)
