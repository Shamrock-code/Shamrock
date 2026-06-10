"""
Testing Sod tube with SPH
=========================

CI test for Sod tube with SPH
"""

import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shamrock.external.coala as coala
from matplotlib.lines import Line2D
from scipy.special import erfinv

import shamrock

print(f"coala path : {coala.__file__}")

shamrock.enable_experimental_features()

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)


gamma = 1.4
rho_i = 1e-7
central_mass = 1
R0 = 1
H_r_0 = 0.05

box_H_count = 8

ndust = 40
mrn_pow = 3.5
mrn_cutoff = 250e-9

epsilon_base = 0.01


do_coag = True

print("codeu.get('m') / codeu.get('s') =", codeu.get("m") / codeu.get("s"))
print("codeu.to('m') / codeu.to('s') =", codeu.to("m") / codeu.to("s"))

dv_max = 1000000 * codeu.get("m") / codeu.get("s")
Q = 5
rhodust_eps = 1e-17
K0_multiplier = 100


sim_folder = "dusty_settle_coag/"
dump_folder = sim_folder + "dump/"

# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)
    os.makedirs(dump_folder, exist_ok=True)

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


print(f"dv_max_si = {dv_max * codeu.to('m') / codeu.to('s')} [m/s]")
print(f"dv_max = {dv_max} [code units]")

print(f"dv_max/cs = {dv_max / cs}")


rho_grains_si_edges = np.array([2.3 * 1000 for i in range(ndust + 1)])
grain_size_si_edges = np.logspace(-9, -2, ndust + 1)

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

massgrid_edges = (4 * np.pi / 3) * rho_grains_edges * grain_size_edges**3
massgrid = np.sqrt(massgrid_edges[:-1] * massgrid_edges[1:])

print(f"massgrid = {massgrid} [code units]")
print(f"massgrid = {massgrid * codeu.to('kg')} [kg]")


K0 = np.pi * ((4.0 / 3.0) * np.pi * rho_grains[0]) ** (-2.0 / 3.0)
K0 *= K0_multiplier
print(f"K0 = {K0}")

tabflux_coag = coala.coala_precalc_tabflux_coag(K0, ndust, Q, massgrid_edges)


bmin = (-box / 8, -box / 8, -box)
bmax = (box / 8, box / 8, box)

N_target = 2e4
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

    cfg.set_dust_mode_monofluid_tvi(ndust)
    cfg.set_dust_drag_epstein(grain_size, rho_grains)
    if do_coag:
        cfg.set_dust_evol_coala_coag(rhodust_eps, dv_max, massgrid_edges, tabflux_coag)

    cfg.add_ext_force_vertical_disc_potential(central_mass=1, R0=1)
    cfg.add_ext_force_velocity_dissipation(eta=10)
    cfg.set_boundary_periodic()
    cfg.set_units(codeu)
    cfg.set_eos_isothermal(cs)
    cfg.print_status()
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e8), 1)

    bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
    xm, ym, zm = bmin
    xM, yM, zM = bmax

    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen, insert_step=scheduler_split_val)

    vol_b = (xM - xm) * (yM - ym) * (zM - zm)
    totmass = rho_i * vol_b
    print("Total mass :", totmass)

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

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)

    model.timestep()


dump_helper.load_last_dump_or(setup_model)

mrn_weight = grain_size ** (4 - mrn_pow)
mrn_weight *= grain_size_si < mrn_cutoff
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

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), dpi=dpi)
    time = model.get_time()
    fig.suptitle(f"t = {time:.2f}")

    fig.subplots_adjust(left=0.07, right=1.05, wspace=0.35)

    to_dens = codeu.to("kg") * codeu.to("m") ** -3

    dust_cmap = plt.colormaps[cmap]
    dust_norm = mcolors.LogNorm(vmin=grain_size_si.min(), vmax=grain_size_si.max())
    dust_colors = dust_cmap(dust_norm(grain_size_si))

    rho_dust_all = np.zeros(len(z))
    epsilon_dust_all = np.zeros(len(z))

    for i in range(ndust):
        c = dust_colors[i]
        axs[0].scatter(z, s_j[:, i] ** 2 * to_dens, s=sz, color=c, edgecolors="none")
        axs[1].scatter(z, s_j[:, i] ** 2 / rho, s=sz, color=c, edgecolors="none")
        axs[2].scatter(z, 2 * s_j[:, i] * ds_j_dt[:, i], s=sz, color=c, edgecolors="none")

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
    axs[0].set_ylim(1e-20, 1e-9)
    # axs[0].set_ylim(1e-12, 10**2)

    axs[1].set_ylabel(r"$\epsilon_j$")
    axs[1].set_xlabel(r"$z$")
    axs[1].set_yscale("log")
    axs[1].set_ylim(1e-12, 2)

    axs[2].set_ylabel(r"$\dot{\rho}_{d,j} = 2 s_j \frac{d s_j}{dt}$")
    axs[2].set_xlabel(r"$z$")

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

    os.makedirs(f"mono_{'coag' if do_coag else 'mono'}", exist_ok=True)
    plt.savefig(f"mono_{'coag' if do_coag else 'mono'}/vert_slice_{j}.png")
    # model.do_vtk_dump(f"dump_stratif_{j}.vtk", True)
    plt.close()

    if False:
        fig_coala, ax_coala = plt.subplots(dpi=dpi)
        fig_coala.suptitle(f"t = {time:.2f}")
        z_cmap = plt.colormaps[cmap]
        z_norm = mcolors.Normalize(vmin=0, vmax=2 * H)
        for i in range(ndust):
            rho_dust = s_j[:, i] ** 2 * to_dens
            x_scat = grain_size_si[i] * np.ones(len(rho_dust))

            cmap_val = np.abs(z)
            sorted_indices = np.argsort(cmap_val)

            cmap_val = cmap_val[sorted_indices]
            x_scat = x_scat[sorted_indices]
            rho_dust = rho_dust[sorted_indices]

            ax_coala.scatter(x_scat, rho_dust, c=cmap_val, cmap=z_cmap, norm=z_norm)
        ax_coala.set_xscale("log")
        ax_coala.set_yscale("log")
        ax_coala.set_xlabel("grain size [m]")
        ax_coala.set_ylabel("density [kg/m^3]")
        ax_coala.set_ylim(1e-21, 1e-11)
        fig_coala.colorbar(cm.ScalarMappable(cmap=z_cmap, norm=z_norm), ax=ax_coala, label="z [au]")
        fig_coala.savefig(f"mono_{'coag' if do_coag else 'mono'}/coala_plot_{j}.png")
        plt.close(fig_coala)

    fig_coala, (ax_coala, ax_zsum) = plt.subplots(1, 2, dpi=dpi, figsize=(12, 5))
    fig_coala.suptitle(f"t = {time:.2f}")
    z_cmap = plt.colormaps[cmap]
    z_norm = mcolors.Normalize(vmin=0, vmax=2.5 * H)

    # The goal of this plot is to do multiple rho(sgrain) averaged for multiple bins of z values

    zbins = np.linspace(0, 3 * H, 25).tolist()

    zbins_centers = (np.array(zbins[1:]) + np.array(zbins[:-1])) / 2

    mdust_zsums = np.zeros((len(zbins_centers), ndust))
    print(f"mdust_zsums = {mdust_zsums}")

    mgas_zsums = np.zeros(len(zbins_centers))

    for ibin in range(len(zbins) - 1):
        zmin = zbins[ibin]
        zmax = zbins[ibin + 1]
        cmap_val = (zmin + zmax) / 2

        mask = (np.abs(z) > zmin) & (np.abs(z) < zmax)
        mgas_zsums[ibin] = np.sum(mask) * pmass

        rho_dust_avg = np.zeros(ndust)
        for idust in range(ndust):
            rho_dust = s_j[:, idust] ** 2 * to_dens
            rho_dust_avg[idust] = np.sum(rho_dust[mask]) / np.sum(mask)

            mdust_zsums[ibin, idust] += np.sum(pmass * (s_j[:, idust] ** 2 / rho) * mask)

        color = z_cmap(z_norm(cmap_val))
        print(f"rendering zbin {ibin} zmin={zmin} zmax={zmax} with color {color}")

        ax_coala.plot(grain_size_si, rho_dust_avg, color=color, linewidth=1)

    mdust_zsums_all = np.sum(mdust_zsums, axis=1)

    ax_coala.set_xscale("log")
    ax_coala.set_yscale("log")
    ax_coala.set_xlabel("grain size [m]")
    ax_coala.set_ylabel("density [kg/m^3]")
    ax_coala.set_ylim(1e-21, 1e-11)
    fig_coala.colorbar(cm.ScalarMappable(cmap=z_cmap, norm=z_norm), ax=ax_coala, label="z [au]")

    for idust in range(ndust):
        ax_zsum.plot(
            zbins_centers, mdust_zsums[:, idust], marker="o", markersize=3, color=dust_colors[idust]
        )

    ax_zsum.plot(zbins_centers, mdust_zsums_all, marker="o", markersize=3, color="0.5")
    ax_zsum.plot(zbins_centers, mgas_zsums, marker="o", markersize=3, color="0.0")
    ax_zsum.legend(handles=[gas_handle, dust_handle], loc="upper right", fontsize=8)

    fig_coala.colorbar(dust_sm, ax=ax_zsum, label="grain size [m]")

    print(f"mdust_zsums = {mdust_zsums}")
    print(f"zbins_centers = {zbins_centers}")
    ax_zsum.set_ylabel(r"$\sum m_{\rm dust}$ in $z$ bin")
    ax_zsum.set_xlabel("z [au]")
    ax_zsum.set_yscale("log")
    ax_zsum.set_ylim(1e-17, 1e-9)

    fig_coala.tight_layout()
    fig_coala.savefig(f"mono_{'coag' if do_coag else 'mono'}/coala_plot_avg_{j}.png")
    # plt.show()
    plt.close(fig_coala)


t_start = model.get_time()

tlist = [i * 0.1 for i in range(3000)]

tnext = 0
for j in range(1000):
    if tlist[j] >= t_start:
        if j > 0:
            model.evolve_until(tlist[j])
            # model.timestep()

        if j == 30:
            for k in range(ndust):

                def compute_sj_new(patchdata):
                    return compute_sj_new_j(patchdata, k)

                model.overwrite_field_value_f64("s_j", compute_sj_new, k)
            model.set_dt(0.0)  # to help the corrector on next step after adding dust

        analyse_and_plot(j)

        dump_helper.write_dump(j, purge_old_dumps=True, keep_first=1, keep_last=3)
