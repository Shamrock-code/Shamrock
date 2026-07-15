"""
Dusty settling SPH test
========================

Perform a dust settling test in a local stratified box.

If you want to change the resolution, you can set the LZ environment variable.

Example (from build directory):
```bash
LZ=96 ./shamrock --sycl-cfg 0:0 --smi --loglevel 1 --rscript ../examples/sph/run_dustysettle_tvi.py
```
"""

# sphinx_gallery_multi_image = "single"

# %%
# Imports
# ------------------------------------------

import json
import os

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.lines import Line2D
from scipy.linalg import solve_banded
from scipy.special import erfinv
from shamrock.utils.DustMRNDistribution import DustMRNDistribution
from shamrock.utils.numba_helper import maybe_njit
from shamrock.utils.plot import show_image_sequence

import shamrock

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
rho_i = 1e-6  # Initial density
central_mass = 1  # Central mass
R0 = 1  # Where are we in radius
H_r_0 = 0.05  # Disc aspect ratio
gamma = 1.4  # Adiabatic index
box_H_count = 8  # box size in number of H
random_vel_pert = True  # add a small vecolity pertubation to fasten the setup

# Dust
ndust = 5
mrn_pow = 3.5
mrn_cutoff_si = np.inf  # would be 250e-9 normally
epsilon_base = 0.01
rho_grains_si_edges = np.array([2.3 * 1000 for i in range(ndust + 1)])
grain_size_si_edges = np.logspace(-5, -3, ndust + 1)

# Resolution (lattice size)
lx = 12
ly = 12
lz = int(os.environ.get("LZ", 64))

# Time
tlist = [0.1 * i for i in range(1000)]
iinject = 20
tinject = tlist[iinject]
t_end = tinject + 3.0

# Scheduler
scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)

# Artificial viscosity
av_alpha_min = 0.0
av_alpha_max = 1
av_sigma_decay = 0.1
av_alpha_u = 1
av_beta_AV = 2
vel_dissipation_eta = 5

# CFL (before dust injection)
cfl_cour_setup = 0.25
cfl_force_setup = 0.25
# CFL (after dust injection)
cfl_cour_inject = 0.1
cfl_force_inject = 0.1

# Injection check
max_v_inject_threshold = 1.0

# Analytical reference
reference_tau = 0.025
reference_Nz = 4000
reference_zrange = 3.5  # in units of H

# Paths
sim_folder = f"_to_trash/dusty_settle_{lz}/"
dump_folder = sim_folder + "dump/"

# Plotting
cmap = "plasma"
dpi = 250

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
# Analytical reference solution
# ------------------------------------------


def S_rho(rho, v, vp, hz, tau, Nz):
    """
    Step rho forward using Crank-Nicolson on continuity equation.
    """

    npts = Nz + 2

    ab_A = np.zeros((3, npts))
    bd_main = np.zeros(npts)
    bd_lower = np.zeros(npts)
    bd_upper = np.zeros(npts)

    # Interior points
    for j in range(1, Nz + 1):
        C1 = -tau / (8 * hz) * (v[j + 1] + vp[j + 1])
        C2 = -tau / (8 * hz) * (v[j - 1] + vp[j - 1])

        ab_A[2, j - 1] = C2
        ab_A[1, j] = 1.0
        ab_A[0, j + 1] = -C1

        bd_lower[j] = -C2
        bd_main[j] = 1.0
        bd_upper[j] = C1

    # Left boundary
    C = -tau / (8 * hz) * (vp[0] + v[0] - vp[1] - v[1])

    ab_A[1, 0] = 0.5 + C
    ab_A[0, 1] = 0.5 + C

    bd_main[0] = 0.5 - C
    bd_upper[0] = 0.5 - C

    # Right boundary
    C = -tau / (8 * hz) * (vp[-1] + v[-1] - vp[-2] - v[-2])

    ab_A[2, npts - 2] = 0.5 - C
    ab_A[1, npts - 1] = 0.5 - C

    bd_lower[npts - 1] = 0.5 + C
    bd_main[npts - 1] = 0.5 + C

    r = bd_main * rho
    r[1:] += bd_lower[1:] * rho[:-1]
    r[:-1] += bd_upper[:-1] * rho[1:]

    ret = solve_banded((1, 1), ab_A, r)
    ret = np.maximum(ret, 0)

    ###################
    # If there are zero values, propagate the zero value to the edges
    ###################
    n = len(ret)
    center = n // 2

    # Right side (center -> end)
    right = ret[center:]
    zero_mask_right = right == 0
    propagate_right = np.cumsum(zero_mask_right) > 0
    ret[center:] = right * (~propagate_right)

    # Left side (center -> start)
    left = ret[: center + 1][::-1]  # reverse
    zero_mask_left = left == 0
    propagate_left = np.cumsum(zero_mask_left) > 0
    ret[: center + 1] = (left * (~propagate_left))[::-1]

    return ret


def S_v(v, vbar, vbarp, rho, rhop, hz, zbar, tau, Nz, Stj):
    """
    Step velocity forward using Crank-Nicolson.
    """

    npts = Nz + 2

    ab_A = np.zeros((3, npts))
    bd_main = np.zeros(npts)
    bd_lower = np.zeros(npts)
    bd_upper = np.zeros(npts)
    E0 = np.zeros_like(v)

    for j in range(1, Nz + 1):
        E0[j] = -zbar[j] / (1 + zbar[j] ** 2) ** 1.5

        E1 = (vbar[j] + vbarp[j]) / (8 * hz)
        E2 = 1.0 / (2.0 * Stj[j])

        ab_A[2, j - 1] = -E1
        ab_A[1, j] = 1.0 / tau + E2
        ab_A[0, j + 1] = E1

        bd_lower[j] = E1
        bd_main[j] = 1.0 / tau - E2
        bd_upper[j] = -E1

    # Boundary conditions
    ab_A[1, 0] = 1
    ab_A[0, 1] = 1

    ab_A[2, npts - 2] = 1
    ab_A[1, npts - 1] = 1

    r = bd_main * v
    r[1:] += bd_lower[1:] * v[:-1]
    r[:-1] += bd_upper[:-1] * v[1:]
    r += E0

    return solve_banded((1, 1), ab_A, r)


S_rho = maybe_njit(S_rho)
S_v = maybe_njit(S_v)


class ReferenceDustySettle:
    def __init__(self, Nz, H, rho_mid, R0, dust_to_gas, dt):
        self.H = H
        self.rho_mid = rho_mid
        self.rho_0 = rho_mid * np.sqrt(2 * np.pi)
        self.R0 = R0
        self.dust_to_gas = dust_to_gas
        self.dt = dt

        self.Nz = Nz

        zout = reference_zrange * H / R0
        zmin = -zout
        zmax = zout

        self.hz = (zmax - zmin) / self.Nz

        # exactly Nz+2 points
        self.zbar = np.linspace(zmin - 0.5 * self.hz, zmax + 0.5 * self.hz, Nz + 2)

    def z_arr(self):
        return self.zbar * self.R0

    def rho_g_func(self, z):
        loc_h = self.H
        ampl = self.rho_0 / (np.sqrt(2 * np.pi))
        return ampl * np.exp(-(z**2) / (2 * loc_h * loc_h))

    def rho_g_bar_func(self, z):
        return self.rho_g_func(z) / self.rho_mid

    def rho_d_bar_func(self, z):

        mask = 1 / (1 + np.exp((np.abs(z) - 1.75 * H) / (H / 16)))

        return mask * self.dust_to_gas * self.rho_g_func(z) / self.rho_mid

    def rho_g(self):
        return self.rho_g_func(self.z_arr())

    def rho_d_bar(self):
        return self.rho_d_bar_func(self.z_arr())

    def gen_IC(self, rhoeff, sj, cs, OmegaK):

        self.z = self.z_arr()
        self.rhog = self.rho_g()

        self.rhodin = self.rho_d_bar()
        self.rho = self.rhodin.copy()

        self.vdin = np.zeros_like(self.rhodin)
        self.v = self.vdin.copy()

        self.ts = rhoeff * sj / (cs * self.rhog)
        self.Stj = self.ts * OmegaK

        self.tbar = 0.0

    def advance(self, tau):

        rhop = S_rho(self.rho, self.v, self.v, self.hz, tau, self.Nz)

        vp = S_v(self.v, self.v, self.v, self.rho, rhop, self.hz, self.zbar, tau, self.Nz, self.Stj)

        rhop = S_rho(self.rho, self.v, vp, self.hz, tau, self.Nz)

        self.v = S_v(self.v, self.v, vp, self.rho, rhop, self.hz, self.zbar, tau, self.Nz, self.Stj)

        self.rho = rhop

        self.tbar += tau

    def advance_until(self, tfinal):
        print(f"tfinal = {tfinal}")
        while self.tbar < tfinal:
            print(self.tbar)
            self.advance(self.dt)

    def setup(self):
        return


class ReferenceDustySettleAll:
    def __init__(self):
        self.rhomid = get_max_rho()
        self.tau = reference_tau

        self.vK = kep_profile(R0) * codeu.to("m") / codeu.to("s")
        self.OmegaK = omega_k(R0) * codeu.to("s") ** -1
        self.tK = 1 / self.OmegaK

        self.rscale = R0
        self.rhoscale = self.rhomid
        self.vscale = self.vK
        self.tscale = self.tK

        self.cs = cs * codeu.to("m") / codeu.to("s")

        print(f"vK = {self.vK}")
        print(f"OmegaK = {self.OmegaK}")
        print(f"tK = {self.tK}")

        self.soluces = []
        for j in range(ndust):
            mrn_weight_j = mrn_distribution.mrn_weight[j]
            eps_j = mrn_weight_j * epsilon_base
            dtg_j = eps_j / (1 - epsilon_base)
            self.soluces.append(
                ReferenceDustySettle(reference_Nz, H, self.rhomid, R0, dtg_j, self.tau)
            )

            rhoeff = mrn_distribution.rho_grains_si_edges[j] * np.sqrt(np.pi * gamma / 8)

            self.soluces[j].setup()
            self.soluces[j].gen_IC(rhoeff, mrn_distribution.grain_size_si[j], self.cs, self.OmegaK)

    def evolve_until(self, t):
        for k in range(ndust):
            self.soluces[k].advance_until(t * codeu.to("s") / self.tscale)


reference_dusty_settle = None


def compute_L2_error(z, field, z_ref, field_ref):
    """
    Compute the L2 error between two fields on a given grid.
    """

    z_field_sort_args = np.argsort(z)
    z_field_sorted = z[z_field_sort_args]
    field_field_sorted = field[z_field_sort_args]

    # Interpolate field to the same grid as the reference field
    field_interp = np.interp(z_ref, z_field_sorted, field_field_sorted)

    # Compute delta
    delta = field_interp - field_ref

    # Compute L2 func
    L2_func = delta**2

    trap_func = None
    if hasattr(np, "trapezoid"):
        trap_func = getattr(np, "trapezoid")
    else:
        trap_func = getattr(np, "trapz")

    # Compute L2 integral
    L2_integral = trap_func(L2_func, z_ref)

    return np.sqrt(L2_integral)


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


def get_max_rho():
    pmass = model.get_particle_mass()
    hfact = model.get_hfact()
    dic = ctx.collect_data()
    hpart = dic["hpart"]
    rho = pmass * (hfact / np.array(hpart)) ** 3
    to_dens = codeu.to("kg") * codeu.to("m") ** -3
    return rho.max() * to_dens


def get_max_v():
    dic = ctx.collect_data()
    v = dic["vxyz"]
    vnorms = np.linalg.norm(v, axis=1)
    return vnorms.max()


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

    fig = plt.figure(figsize=(12, 7), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], wspace=0.2)

    ax_rho = fig.add_subplot(gs[0, 0])  # Top left
    ax_epsilon = fig.add_subplot(gs[0, 1])  # Top right
    ax_delta_v = fig.add_subplot(gs[1, :])  # Bottom spans both columns

    time = model.get_time() - tinject
    fig.suptitle(f"t = {time:.2f} [yr]")

    fig.subplots_adjust(left=0.08, right=1.05, wspace=0.35)

    to_dens = codeu.to("kg") * codeu.to("m") ** -3
    to_speed = codeu.to("m") / codeu.to("s")

    dust_cmap = plt.colormaps[cmap]
    dust_norm = mcolors.LogNorm(
        vmin=mrn_distribution.grain_size_si.min(), vmax=mrn_distribution.grain_size_si.max() * 10
    )
    dust_colors = dust_cmap(dust_norm(mrn_distribution.grain_size_si))

    rho_dust_all = np.zeros(len(z))
    epsilon_dust_all = np.zeros(len(z))

    l2_error_all = [np.nan for _ in range(ndust)]

    for i in range(ndust):
        c = dust_colors[i]
        ax_rho.scatter(z, s_j[:, i] ** 2 * to_dens, s=sz, color=c, edgecolors="none")
        ax_epsilon.scatter(z, s_j[:, i] ** 2 / rho, s=sz, color=c, edgecolors="none")

        rho_dust_all += s_j[:, i] ** 2 * to_dens
        epsilon_dust_all += s_j[:, i] ** 2 / rho

        if reference_dusty_settle is not None:
            ax_rho.plot(
                reference_dusty_settle.soluces[i].zbar,
                reference_dusty_settle.soluces[i].rho * reference_dusty_settle.rhoscale,
                "--",
                color="0.0",
            )

            ana_epsilon = (
                reference_dusty_settle.soluces[i].rho
                * reference_dusty_settle.rhoscale
                / reference_dusty_settle.soluces[i].rhog
            )
            print(ana_epsilon.max(), ana_epsilon.min())
            ax_epsilon.plot(reference_dusty_settle.soluces[i].zbar, ana_epsilon, "--", color="0.0")

            L2_error = compute_L2_error(
                z, s_j[:, i] ** 2 / rho, reference_dusty_settle.soluces[i].zbar, ana_epsilon
            )

            l2_error_all[i] = L2_error

    save_analysis_data("l2_error.json", "l2_error", l2_error_all, j)

    dust_mass = analysis_dust_mass.get_dust_mass()

    # if all dust mass is zero replace by nans
    if np.max(dust_mass) == 0:
        print("all dust mass is zero, replacing by nans")
        dust_mass = [np.nan for _ in range(ndust)]

    save_analysis_data("dust_mass.json", "dust_mass", dust_mass, j)

    ax_rho.scatter(z, rho * to_dens, s=sz, color="0.0", edgecolors="none")
    ax_rho.scatter(z, rho_dust_all, s=sz, color="0.5", edgecolors="none")
    ax_epsilon.scatter(z, 1 - epsilon_dust_all, s=sz, color="0.0", edgecolors="none")
    ax_epsilon.scatter(z, epsilon_dust_all, s=sz, color="0.5", edgecolors="none")

    range_plot = 2.5 * H

    ax_rho.set_ylabel(r"$\rho$ [kg/m$^3$]")
    ax_rho.set_xlabel(r"$z$")
    ax_rho.set_yscale("log")
    ax_rho.set_ylim(1e-20, 1e-8)
    ax_rho.set_xlim(-range_plot, range_plot)

    ax_epsilon.set_ylabel(r"$\epsilon_j$")
    ax_epsilon.set_xlabel(r"$z$")
    ax_epsilon.set_yscale("log")
    ax_epsilon.set_ylim(1e-4, 1e-1)  # if you want to see the dust only
    ax_epsilon.set_xlim(-range_plot, range_plot)

    ax_delta_v.set_ylabel(r"$\Delta v_z$ [m/s]")
    ax_delta_v.set_xlabel(r"$z$")
    ax_delta_v.set_xlim(-range_plot, range_plot)
    ax_delta_v.set_yscale("symlog", linthresh=1.0)
    ax_delta_v.set_ylim(-4e3, 4e3)

    for i in range(ndust):
        c = dust_colors[i]
        ax_delta_v.scatter(z, delta_v[:, i, 2] * to_speed, s=sz, color=c, edgecolors="none")

        if reference_dusty_settle is not None:
            ax_delta_v.plot(
                reference_dusty_settle.soluces[i].zbar,
                reference_dusty_settle.soluces[i].v * reference_dusty_settle.vscale,
                "--",
                color="0.0",
            )

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
        label="reference",
    )

    ax_rho.legend(handles=[gas_handle, dust_handle, analytic_handle], loc="upper right", fontsize=8)

    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
    dust_sm.set_array([])
    cbar = fig.colorbar(
        dust_sm, ax=[ax_rho, ax_epsilon, ax_delta_v], pad=0.03, shrink=0.95, aspect=40
    )
    cbar.set_label(r"grain size $s$ [m]")

    os.makedirs(f"{dump_folder}/plots", exist_ok=True)
    plt.savefig(f"{dump_folder}/plots/vert_slice_dens_{j:04d}.png")
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=dpi)
    time = model.get_time() - tinject
    fig.suptitle(f"t = {time:.2f} [yr]")

    fig.subplots_adjust(left=0.07, right=1.05, wspace=0.35)

    for i in range(ndust):
        c = dust_colors[i]
        axs[0].scatter(z, s_j[:, i], s=sz, color=c, edgecolors="none")
        axs[1].scatter(z, ds_j_dt[:, i], s=sz, color=c, edgecolors="none")

    axs[0].set_ylabel(r"$s_j$")
    axs[0].set_xlabel(r"$z$")
    axs[0].set_xlim(-4 * H, 4 * H)
    axs[0].set_yscale("symlog", linthresh=1e-10)

    axs[1].set_ylabel(r"$\dot{s}_j$")
    axs[1].set_xlabel(r"$z$")
    axs[1].set_xlim(-4 * H, 4 * H)
    axs[1].set_yscale("symlog", linthresh=1e-10)

    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
    dust_sm.set_array([])
    cbar = fig.colorbar(dust_sm, ax=axs, pad=0.02, shrink=0.85)
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

    cfg.set_dust_mode_monofluid_tvi(
        nvar=ndust, C_1_fluid=0.1, C_delta_v=1.0, cfl_density_threshold=1e-50
    )
    cfg.set_dust_drag_epstein(gamma, mrn_distribution.grain_size, mrn_distribution.rho_grains)
    cfg.add_ext_force_vertical_disc_potential(central_mass=1, R0=1)
    cfg.add_ext_force_velocity_dissipation(eta=vel_dissipation_eta)
    cfg.set_two_stage_search(False)
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

        ampl = dr / 1.0
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

        if max_v > max_v_inject_threshold:
            raise ValueError("max_v is too high, please increase the injection time")

        for k in range(ndust):

            def compute_sj_new(patchdata):
                return compute_sj_new_j(patchdata, k)

            self.model.overwrite_field_value_f64("s_j", compute_sj_new, k)

            self.model.set_cfl_cour(cfl_cour_inject)
            self.model.set_cfl_force(cfl_force_inject)

        self.model.set_dt(0.0)  # to help the corrector on next step after adding dust

    @callback(tsim_interval=0.1)  # Do the analysis every dt_stop
    def analysis_plots(self, j):
        global reference_dusty_settle

        model_time = self.model.get_time()

        if model_time > tinject and reference_dusty_settle is None:
            reference_dusty_settle = ReferenceDustySettleAll()

        if reference_dusty_settle is not None:
            reference_dusty_settle.evolve_until(model_time - tinject)

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

glob_str = f"{dump_folder}/plots/vert_slice_dens_*.png"
ani = show_image_sequence(glob_str)

writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani.save("_to_trash/dustysettle_vert_slice_tvi.gif", writer=writer)

if shamrock.sys.world_rank() == 0:
    plt.show()

# %%

glob_str = f"{dump_folder}/plots/vert_slice_s_*.png"
ani = show_image_sequence(glob_str)

writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani.save("_to_trash/dustysettle_vert_slice_s_tvi.gif", writer=writer)

if shamrock.sys.world_rank() == 0:
    plt.show()

# %%
# Plot dust mass conservation history
# ------------------------------------------

t, dust_mass = load_data_from_json("dust_mass.json", "dust_mass")
dust_mass = np.array(dust_mass)

# tinject = first non nan
iinject = np.argmax(~np.isnan(dust_mass)[:, 0])
tinject = np.array(t)[iinject]

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

plt.figure()
for k in range(ndust):
    mh = dust_mass[:, k]
    deviation = (mh / mh[iinject]) - 1

    plt.plot(
        t,
        deviation,
        label=f"dust {k}, s = {mrn_distribution.grain_size_si[k]:.1e} [m], St = {St[k]:.1e}",
    )

total_dust_mass = np.sum(dust_mass, axis=1)
plt.plot(
    t,
    (total_dust_mass / total_dust_mass[iinject]) - 1,
    color="grey",
    label="total dust mass",
    linestyle="--",
)

plt.xlabel("t")
plt.ylabel("$\\delta M_{dust} / M_{dust,0}$")
plt.yscale("symlog", linthresh=1e-8)
plt.title("Dust mass conservation")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dump_folder}/plots/dust_mass_history.png")
plt.show()


# %%
# Plot L2 error history
# ------------------------------------------

t, l2_error = load_data_from_json("l2_error.json", "l2_error")
l2_error = np.array(l2_error)

t = np.array(t) - tinject

print(t)
print(l2_error)

plt.figure()
for k in range(ndust):
    plt.plot(
        t,
        l2_error[:, k],
        label=f"dust {k}, s = {mrn_distribution.grain_size_si[k]:.1e} [m], St = {St[k]:.1e}",
    )
plt.legend()
plt.xlabel("t")
plt.ylabel("L2 error")
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"{dump_folder}/plots/l2_error.png")
plt.show()

# %%

print("##### Test result #####")
result = {
    "t": float(t[-1]),
    "l2_error": l2_error[-1].tolist(),
    "dust_mass": (dust_mass[-1] / dust_mass[iinject] - 1).tolist(),
    "St": St.tolist(),
    "lz": lz,
}
print(result)

json.dump(result, open(f"{dump_folder}/test_result.json", "w"), indent=4)
