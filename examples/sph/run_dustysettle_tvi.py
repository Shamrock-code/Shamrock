"""
Dusty settling SPH test
========================

Perform a dust settling test in a local stratified box.
"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import erfinv
from shamrock.utils.DustMRNDistribution import DustMRNDistribution
from shamrock.utils.numba_helper import maybe_njit

import shamrock

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()


# %%
# Sim parameters
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

rho_i = 1e-6
central_mass = 1
R0 = 1
H_r_0 = 0.05

box_H_count = 8

ndust = 5
mrn_pow = 3.5
mrn_cutoff_si = np.inf  # would be 250e-9 normally
gamma = 1.4

epsilon_base = 0.01

tlist = [0.0025 * i for i in range(500)]
tinject = 0
iinject = 0
t_end = 5.0


sim_folder = "_to_trash/dusty_settle/"
dump_folder = sim_folder + "dump/"

# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)
    os.makedirs(dump_folder, exist_ok=True)


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


cs_g = cs


def uint_g(r):
    rho_g = func_rho_g(r)
    P = rho_g * cs_g * cs_g / gamma
    return P / ((gamma - 1) * rho_g)


rho_grains_si_edges = np.array([2.3 * 1000 for i in range(ndust + 1)])
grain_size_si_edges = np.logspace(-5, -2, ndust + 1)

mrn_distribution = DustMRNDistribution(
    codeu, mrn_pow, mrn_cutoff_si, grain_size_si_edges, rho_grains_si_edges
)

bmin = (-box / 8, -box / 8, -box)
bmax = (box / 8, box / 8, box)

scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)

lx = 12
ly = 12
lz = 512
lmin = (-lx // 2, -ly // 2, -lz // 2)
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

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

dump_helper = shamrock.utils.dump.ShamrockDumpHandleHelper(model, dump_folder)


def setup_model():
    global bmin, bmax

    cfg = model.gen_default_config()
    # cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
    # cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )

    cfg.set_dust_mode_monofluid_tvi(nvar=ndust, C_delta_v=10.0)
    cfg.set_dust_drag_epstein(gamma, mrn_distribution.grain_size, mrn_distribution.rho_grains)
    cfg.add_ext_force_vertical_disc_potential(central_mass=1, R0=1)
    cfg.add_ext_force_velocity_dissipation(eta=5)
    cfg.set_boundary_periodic()
    cfg.set_units(codeu)
    cfg.set_eos_isothermal(cs)
    cfg.set_show_cfl_detail(True)
    cfg.print_status()
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e8), 1)

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

    def f_remap(r):
        x, y, z = r

        rn = max(abs(zM), abs(zm))
        # print(y, H, H * erfinv(y / rn))
        z = H * erfinv(z / rn) * (2**0.5)

        z = min(z, zM)
        z = max(z, zm)
        return (x, y, z)

    model.remap_positions(f_remap)
    model.set_field_value_lambda_f64("uint", uint_g)

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)

    model.timestep()


dump_helper.load_last_dump_or(setup_model)


def get_max_rho():
    pmass = model.get_particle_mass()
    hfact = model.get_hfact()
    dic = ctx.collect_data()
    hpart = dic["hpart"]
    rho = pmass * (hfact / np.array(hpart)) ** 3
    to_dens = codeu.to("kg") * codeu.to("m") ** -3
    return rho.max() * to_dens


pmass = model.get_particle_mass()


def compute_sj_new_j(patchdata, j):
    global pmass

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


# TODO: add function to modify fields e.g. get rho and do stuff according to it

cmap = "plasma"
dpi = 250


def save_analysis_data(filename, key, value, ianalysis):
    """Helper to save analysis data to a JSON file."""
    import json

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
    import json

    filepath = os.path.join(dump_folder, filename)
    with open(filepath, "r") as fp:
        data = json.load(fp)[key]
    t = [d["t"] for d in data]
    values = [d[key] for d in data]
    return t, values


"""
Analytical soluce for dusty settling
"""

from scipy.linalg import solve


def S_rho(rho, v, vp, hz, tau, Nz):
    """
    Step rho forward using Crank-Nicolson on continuity equation.
    """

    npts = Nz + 2

    A = np.zeros((npts, npts))
    B = np.zeros((npts, npts))

    # Interior points
    for j in range(1, Nz + 1):
        C1 = -tau / (8 * hz) * (v[j + 1] + vp[j + 1])
        C2 = -tau / (8 * hz) * (v[j - 1] + vp[j - 1])

        A[j, j - 1] = C2
        A[j, j] = 1.0
        A[j, j + 1] = -C1

        B[j, j - 1] = -C2
        B[j, j] = 1.0
        B[j, j + 1] = C1

    # Left boundary
    C = -tau / (8 * hz) * (vp[0] + v[0] - vp[1] - v[1])

    A[0, 0] = 0.5 + C
    A[0, 1] = 0.5 + C

    B[0, 0] = 0.5 - C
    B[0, 1] = 0.5 - C

    # Right boundary
    C = -tau / (8 * hz) * (vp[-1] + v[-1] - vp[-2] - v[-2])

    A[-1, -2] = 0.5 - C
    A[-1, -1] = 0.5 - C

    B[-1, -2] = 0.5 + C
    B[-1, -1] = 0.5 + C

    r = B @ rho

    ret = solve(A, r)
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

    A = np.zeros((npts, npts))
    B = np.zeros((npts, npts))
    E0 = np.zeros_like(v)

    for j in range(1, Nz + 1):
        E0[j] = -zbar[j] / (1 + zbar[j] ** 2) ** 1.5

        E1 = (vbar[j] + vbarp[j]) / (8 * hz)
        E2 = 1.0 / (2.0 * Stj[j])

        A[j, j - 1] = -E1
        A[j, j] = 1.0 / tau + E2
        A[j, j + 1] = E1

        B[j, j - 1] = E1
        B[j, j] = 1.0 / tau - E2
        B[j, j + 1] = -E1

    # Boundary conditions
    A[0, 0] = 1
    A[0, 1] = 1

    A[-1, -2] = 1
    A[-1, -1] = 1

    r = B @ v + E0

    return solve(A, r)


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

        zout = 3 * H / R0
        zmin = -zout
        zmax = zout

        self.hz = (zmax - zmin) / self.Nz

        # exactly Nz+2 points
        self.zbar = np.linspace(zmin - 0.5 * self.hz, zmax + 0.5 * self.hz, Nz + 2)

    def z_arr(self):
        return self.zbar * self.R0

    def rho_g_func(self, z):
        loc_h = self.H
        print(loc_h)
        ampl = self.rho_0 / (np.sqrt(2 * np.pi))
        print(ampl)
        return ampl * np.exp(-(z**2) / (2 * loc_h * loc_h))

    def rho_g_bar_func(self, z):
        return self.rho_g_func(z) / self.rho_mid

    def rho_d_bar_func(self, z):

        mask = 1 / (1 + np.exp((np.abs(z) - 1.75 * H) / (H / 16)))
        # mask = 1

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

        return

    def setup(self):
        return


class ReferenceDustySettleAll:
    def __init__(self):
        self.rhomid = get_max_rho()
        self.tau = 0.05

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
            self.soluces.append(ReferenceDustySettle(2000, H, self.rhomid, R0, dtg_j, self.tau))

            rhoeff = mrn_distribution.rho_grains_si_edges[j] * np.sqrt(np.pi * gamma / 8)

            self.soluces[j].setup()
            self.soluces[j].gen_IC(rhoeff, mrn_distribution.grain_size_si[j], self.cs, self.OmegaK)

    def evolve_until(self, t):
        for k in range(ndust):
            self.soluces[k].advance_until(t * codeu.to("s") / self.tscale)


reference_dusty_settle = None


def compute_L2_error(z, field, z_ref, field_ref):

    z_field_sort_args = np.argsort(z)
    z_field_sorted = z[z_field_sort_args]
    field_field_sorted = field[z_field_sort_args]

    # Interpolate field to the same grid as the reference field
    field_interp = np.interp(z_ref, z_field_sorted, field_field_sorted)

    # Compute delta
    delta = field_interp - field_ref

    # Compute L2 func
    L2_func = delta**2

    # Compute L2 integral    # Use the appropriate trapezoidal integration function
    if hasattr(np, "trapezoid"):
        L2_integral = np.trapezoid(L2_func, z_ref)  # NumPy >= 2.0
    else:
        L2_integral = np.trapz(L2_func, z_ref)  # NumPy < 2.0

    return np.sqrt(L2_integral)


t_l2 = []
l2_errors = [[] for _ in range(ndust)]


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

    print(s_j)

    hpart = dic["hpart"]
    rho = pmass * (hfact / np.array(hpart)) ** 3

    print("compute original rho")
    estimated_rho = [func_rho_t(dic["xyz"][kk]) for kk in range(len(dic["xyz"]))]

    sz = 1

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=dpi)
    time = model.get_time() - tinject
    fig.suptitle(f"t = {time:.2f} [yr]")

    fig.subplots_adjust(left=0.07, right=1.05, wspace=0.35)

    to_dens = codeu.to("kg") * codeu.to("m") ** -3

    dust_cmap = plt.colormaps[cmap]
    dust_norm = mcolors.LogNorm(
        vmin=mrn_distribution.grain_size_si.min(), vmax=mrn_distribution.grain_size_si.max() * 10
    )
    dust_colors = dust_cmap(dust_norm(mrn_distribution.grain_size_si))

    rho_dust_all = np.zeros(len(z))
    epsilon_dust_all = np.zeros(len(z))

    if reference_dusty_settle is not None:
        t_l2.append(time)

    for i in range(ndust):
        c = dust_colors[i]
        axs[0].scatter(z, s_j[:, i] ** 2 * to_dens, s=sz, color=c, edgecolors="none")
        axs[1].scatter(z, s_j[:, i] ** 2 / rho, s=sz, color=c, edgecolors="none")

        rho_dust_all += s_j[:, i] ** 2 * to_dens
        epsilon_dust_all += s_j[:, i] ** 2 / rho

        if reference_dusty_settle is not None:
            axs[0].plot(
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
            axs[1].plot(reference_dusty_settle.soluces[i].zbar, ana_epsilon, "--", color="0.0")

            L2_error = compute_L2_error(
                z, s_j[:, i] ** 2 / rho, reference_dusty_settle.soluces[i].zbar, ana_epsilon
            )
            print(f"L2 error for dust {i} = {L2_error}")
            l2_errors[i].append(L2_error)

    axs[0].scatter(z, rho * to_dens, s=sz, color="0.0", edgecolors="none")
    axs[0].scatter(z, rho_dust_all, s=sz, color="0.5", edgecolors="none")
    axs[1].scatter(z, 1 - epsilon_dust_all, s=sz, color="0.0", edgecolors="none")
    axs[1].scatter(z, epsilon_dust_all, s=sz, color="0.5", edgecolors="none")

    # axs[0].scatter(y,estimated_rho)
    axs[0].set_ylabel(r"$\rho$")
    axs[0].set_xlabel(r"$z$")
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-20, 1e-8)
    axs[0].set_xlim(-box, box)
    # axs[0].set_ylim(1e-12, 10**2)

    axs[1].set_ylabel(r"$\epsilon_j$")
    axs[1].set_xlabel(r"$z$")
    axs[1].set_yscale("log")
    # axs[1].set_ylim(1e-12, 2) # if you want the full range
    axs[1].set_ylim(1e-4, 1e-1)  # if you want to see the dust only
    axs[1].set_xlim(-box, box)

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
        label="analytic",
    )

    axs[0].legend(handles=[gas_handle, dust_handle, analytic_handle], loc="upper right", fontsize=8)

    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
    dust_sm.set_array([])
    cbar = fig.colorbar(dust_sm, ax=axs, pad=0.02, shrink=0.85)
    cbar.set_label(r"grain size $s$ [m]")

    os.makedirs(f"{dump_folder}/plots", exist_ok=True)
    plt.savefig(f"{dump_folder}/plots/vert_slice_dens_{j:04d}.png")
    # model.do_vtk_dump(f"dump_stratif_{j}.vtk", True)
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=dpi)
    time = model.get_time() - tinject
    fig.suptitle(f"t = {time:.2f} [yr]")

    fig.subplots_adjust(left=0.07, right=1.05, wspace=0.35)

    to_dens = codeu.to("kg") * codeu.to("m") ** -3

    rho_dust_all = np.zeros(len(z))
    epsilon_dust_all = np.zeros(len(z))

    for i in range(ndust):
        c = dust_colors[i]
        axs[0].scatter(z, s_j[:, i], s=sz, color=c, edgecolors="none")
        axs[1].scatter(z, ds_j_dt[:, i], s=sz, color=c, edgecolors="none")

    axs[0].set_ylabel(r"$s_j$")
    axs[0].set_xlabel(r"$z$")
    axs[0].set_xlim(-box, box)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-20, 1e-1)

    axs[1].set_ylabel(r"$\dot{s}_j$")
    axs[1].set_xlabel(r"$z$")
    axs[1].set_xlim(-box, box)
    axs[1].set_yscale("symlog", linthresh=1e-10)

    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
    dust_sm.set_array([])
    cbar = fig.colorbar(dust_sm, ax=axs, pad=0.02, shrink=0.85)
    cbar.set_label(r"grain size $s$ [m]")

    plt.savefig(f"{dump_folder}/plots/vert_slice_s_{j:04d}.png")
    plt.close()


# %%
# Timestep loop

analysis_dust_mass = shamrock.model_sph.analysisDustMass(model=model)

t_start = model.get_time()

dust_injected = False

idust_analysis = 0


def dust_mass_analysis():
    global idust_analysis
    dust_mass = analysis_dust_mass.get_dust_mass()
    save_analysis_data("dust_mass.json", "dust_mass", dust_mass, idust_analysis)
    idust_analysis += 1


ivtk = 0

tnext = 0
for j in range(1000):
    if j == iinject:
        reference_dusty_settle = ReferenceDustySettleAll()
        reference_dusty_settle = None

    if tlist[j] >= t_start:
        if j > 0:
            while 1:
                res = model.evolve_until(tlist[j], niter_max=10000)
                # model.timestep()

                if model.get_time() >= 0.0:
                    model.do_vtk_dump(f"{dump_folder}/vtk_{ivtk:04d}.vtk", True)
                    ivtk += 1

                    # if ivtk > 100:
                    #    exit()

                if res.reach_target_time:
                    break

            if dust_injected:
                dust_mass_analysis()

        if j == iinject:
            for k in range(ndust):

                def compute_sj_new(patchdata):
                    return compute_sj_new_j(patchdata, k)

                model.overwrite_field_value_f64("s_j", compute_sj_new, k)

                model.set_cfl_cour(0.1)
                model.set_cfl_force(0.1)

            dust_injected = True
            dust_mass_analysis()

            model.set_dt(0.0)  # to help the corrector on next step after adding dust

        if reference_dusty_settle is not None:
            reference_dusty_settle.evolve_until(tlist[j] - tinject)

        analyse_and_plot(j)

        dump_helper.write_dump(j, purge_old_dumps=True, keep_first=1, keep_last=5)

    if tlist[j] >= t_end:
        break

####################################################
# Convert PNG sequence to Image sequence in mpl
####################################################

from shamrock.utils.plot import show_image_sequence

# %%
glob_str = f"{dump_folder}/plots/vert_slice_dens_*.png"
ani = show_image_sequence(glob_str)

from matplotlib.animation import PillowWriter

writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani.save("_to_trash/dustysettle_vert_slice_tvi.gif", writer=writer)

if shamrock.sys.world_rank() == 0:
    # Show the animation
    plt.show()

# %%
glob_str = f"{dump_folder}/plots/vert_slice_s_*.png"
ani = show_image_sequence(glob_str)

from matplotlib.animation import PillowWriter

writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani.save("_to_trash/dustysettle_vert_slice_s_tvi.gif", writer=writer)

if shamrock.sys.world_rank() == 0:
    # Show the animation
    plt.show()

# %%
# Plot the mass history

t, dust_mass = load_data_from_json("dust_mass.json", "dust_mass")
dust_mass = np.array(dust_mass)

plt.figure()
for k in range(ndust):
    mh = dust_mass[:, k]
    deviation = (mh / mh[0]) - 1

    t_dyn = 1
    ts = shamrock.phys.epstein_stopping_time(
        rho_grain=mrn_distribution.rho_grains[k],
        s_grain=mrn_distribution.grain_size[k],
        rho=rho_i,
        cs=cs,
        gamma=gamma,
    )
    St = ts / t_dyn

    plt.plot(
        t,
        deviation,
        label=f"dust {k}, s = {mrn_distribution.grain_size_si[k]:.1e} [m], St = {St:.1e}",
    )

total_dust_mass = np.sum(dust_mass, axis=1)
plt.plot(
    t,
    (total_dust_mass / total_dust_mass[0]) - 1,
    color="grey",
    label="total dust mass",
    linestyle="--",
)

plt.xlabel("t")
plt.ylabel("$\|delta M_{dust} / M_{dust,0}$")
plt.yscale("symlog", linthresh=1e-8)
plt.title("Dust mass conservation")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dump_folder}/plots/dust_mass_history.png")
plt.show()

# %%
# Plot the L2 errors

plt.figure()
for k in range(ndust):
    plt.plot(t_l2, l2_errors[k], label=f"dust {k}")
    print(f"L2 error for dust {k} = {l2_errors[k]}")
plt.xlabel("t")
plt.ylabel("L2 error")
plt.yscale("log")
plt.title("L2 errors")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dump_folder}/plots/l2_errors.png")
plt.show()
