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
grain_size_si_edges = np.logspace(-5, -3, ndust + 1)

print(f"grains sizes = {grain_size_si_edges} [m]")
print(f"grains dens  = {rho_grains_si_edges} [kg.m^-3]")

grain_size_edges = grain_size_si_edges * codeu.get("m")
rho_grains_edges = codeu.get("kg") * codeu.get("m", power=-3) * np.array(rho_grains_si_edges)

print(f"grains sizes = {grain_size_edges} [code u]")
print(f"grains dens  = {rho_grains_edges} [code u]")

grain_size = np.sqrt(grain_size_edges[:-1] * grain_size_edges[1:])
rho_grains = np.sqrt(rho_grains_edges[:-1] * rho_grains_edges[1:])

grain_size_si = np.sqrt(grain_size_si_edges[:-1] * grain_size_si_edges[1:])
rho_grains_si = np.sqrt(rho_grains_si_edges[:-1] * rho_grains_si_edges[1:])

print(f"grains sizes = {grain_size_si} [m]")
print(f"grains dens  = {rho_grains_si} [kg.m^-3]")

print(f"grains sizes = {grain_size} [code units]")
print(f"grains dens  = {rho_grains} [code units]")

bmin = (-box / 8, -box / 8, -box)
bmax = (box / 8, box / 8, box)

N_target = 1e4
scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)

xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

part_vol = vol_b / N_target

# lattice volume
HCP_PACKING_DENSITY = 0.74
part_vol_lattice = HCP_PACKING_DENSITY * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)
bmin, bmax = shamrock.math.get_ideal_hcp_box(dr, bmin, bmax)
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

    cfg.set_dust_mode_monofluid_tvi(nvar=ndust)
    cfg.set_dust_drag_epstein(gamma, grain_size, rho_grains)
    cfg.add_ext_force_vertical_disc_potential(central_mass=1, R0=1)
    cfg.add_ext_force_velocity_dissipation(eta=5)
    cfg.set_boundary_periodic()
    cfg.set_units(codeu)
    cfg.set_eos_isothermal(cs)
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
        z = H * erfinv(z / rn)

        z = min(z, zM)
        z = max(z, zm)
        return (x, y, z)

    model.remap_positions(f_remap)
    model.set_field_value_lambda_f64("uint", uint_g)

    model.set_cfl_cour(0.25)
    model.set_cfl_force(0.25)

    model.timestep()


dump_helper.load_last_dump_or(setup_model)

mrn_weight = grain_size ** (4 - mrn_pow)
mrn_weight *= grain_size_si < mrn_cutoff_si
mrn_weight = mrn_weight / np.sum(mrn_weight)

print(f"mrn_weight = {mrn_weight}")

pmass = model.get_particle_mass()


def compute_sj_new_j(patchdata, j):
    global pmass

    hpart = patchdata["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    z = patchdata["xyz"][:, 2]
    # mask to only modify particles with |z| < H
    mask = 1 / (1 + np.exp((np.abs(z) - 1.75 * H) / (H / 16)))

    epsilon_target = epsilon_base * mrn_weight[j] * mask
    print(f"epsilon_target = {epsilon_target} {j}")
    s = np.sqrt(rho * epsilon_target)

    print(
        f"s = {s} {np.isnan(s).any()} epsilon_target = {epsilon_target} mrn_weight = {mrn_weight[j]} mask = {mask}, rho = {rho}"
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
    time = model.get_time()
    fig.suptitle(f"t = {time:.2f}")

    fig.subplots_adjust(left=0.07, right=1.05, wspace=0.35)

    to_dens = codeu.to("kg") * codeu.to("m") ** -3

    dust_cmap = plt.colormaps[cmap]
    dust_norm = mcolors.LogNorm(vmin=grain_size_si.min(), vmax=grain_size_si.max() * 10)
    dust_colors = dust_cmap(dust_norm(grain_size_si))

    rho_dust_all = np.zeros(len(z))
    epsilon_dust_all = np.zeros(len(z))

    for i in range(ndust):
        c = dust_colors[i]
        axs[0].scatter(z, s_j[:, i] ** 2 * to_dens, s=sz, color=c, edgecolors="none")
        axs[1].scatter(z, s_j[:, i] ** 2 / rho, s=sz, color=c, edgecolors="none")

        rho_dust_all += s_j[:, i] ** 2 * to_dens
        epsilon_dust_all += s_j[:, i] ** 2 / rho

    axs[0].scatter(z, rho * to_dens, s=sz, color="0.0", edgecolors="none")
    axs[0].scatter(z, rho_dust_all, s=sz, color="0.5", edgecolors="none")
    axs[1].scatter(z, 1 - epsilon_dust_all, s=sz, color="0.0", edgecolors="none")
    axs[1].scatter(z, epsilon_dust_all, s=sz, color="0.5", edgecolors="none")

    # axs[0].scatter(y,estimated_rho)
    axs[0].set_ylabel(r"$\rho$")
    axs[0].set_xlabel(r"$z$")
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-20, 1e-8)
    axs[0].set_xlim(-4 * H, 4 * H)
    # axs[0].set_ylim(1e-12, 10**2)

    axs[1].set_ylabel(r"$\epsilon_j$")
    axs[1].set_xlabel(r"$z$")
    axs[1].set_yscale("log")
    axs[1].set_ylim(1e-12, 2)
    axs[1].set_xlim(-4 * H, 4 * H)

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
    axs[0].legend(handles=[gas_handle, dust_handle], loc="upper right", fontsize=8)

    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
    dust_sm.set_array([])
    cbar = fig.colorbar(dust_sm, ax=axs, pad=0.02, shrink=0.85)
    cbar.set_label(r"grain size $s$ [m]")

    os.makedirs(f"{dump_folder}/plots", exist_ok=True)
    plt.savefig(f"{dump_folder}/plots/vert_slice_dens_{j:04d}.png")
    # model.do_vtk_dump(f"dump_stratif_{j}.vtk", True)
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=dpi)
    time = model.get_time()
    fig.suptitle(f"t = {time:.2f}")

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
# Timestep loop

analysis_dust_mass = shamrock.model_sph.analysisDustMass(model=model)

t_start = model.get_time()

tlist = [0.1 * i for i in range(20)] + [i * 0.1 + 2 for i in range(3000)]

dust_injected = False

idust_analysis = 0


def dust_mass_analysis():
    global idust_analysis
    dust_mass = analysis_dust_mass.get_dust_mass()
    save_analysis_data("dust_mass.json", "dust_mass", dust_mass, idust_analysis)
    idust_analysis += 1


tnext = 0
for j in range(1000):
    if tlist[j] >= t_start:
        if j > 0:
            model.evolve_until(tlist[j])
            # model.timestep()

            if dust_injected:
                dust_mass_analysis()

        if j == 20:
            for k in range(ndust):

                def compute_sj_new(patchdata):
                    return compute_sj_new_j(patchdata, k)

                model.overwrite_field_value_f64("s_j", compute_sj_new, k)

                model.set_cfl_cour(0.1)
                model.set_cfl_force(0.1)

            dust_injected = True
            dust_mass_analysis()

            model.set_dt(0.0)  # to help the corrector on next step after adding dust

        analyse_and_plot(j)

        dump_helper.write_dump(j, purge_old_dumps=True, keep_first=1, keep_last=3)

    if tlist[j] >= 5.0:
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
        rho_grain=rho_grains[k], s_grain=grain_size[k], rho=rho_i, cs=cs, gamma=gamma
    )
    St = ts / t_dyn

    plt.plot(t, deviation, label=f"dust {k}, s = {grain_size_si[k]:.1e} [m], St = {St:.1e}")

total_dust_mass = np.sum(dust_mass, axis=1)
plt.plot(
    t,
    (total_dust_mass / total_dust_mass[0]) - 1,
    color="grey",
    label="total dust mass",
    linestyle="--",
)

plt.xlabel("t")
plt.ylabel("$\delta M_{dust} / M_{dust,0}$")
plt.yscale("log")
plt.title("Dust mass conservation")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dump_folder}/plots/dust_mass_history.png")
plt.show()
