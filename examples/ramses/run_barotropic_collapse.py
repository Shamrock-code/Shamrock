"""
Barotropic collapse of a uniform sphere in RAMSES solver
========================================================

A uniform, cold (:math:`T_0 = 10` K) sphere of molecular gas of about one solar mass, at rest in
a 100 times more diffuse medium, collapses under its own gravity. The self-gravity is solved
with the conjugate gradient Poisson solver and applied as an external force with the trapezoidal
coupling (``cfg.set_ext_force_coupling(ExtForceCoupling.Trapezoidal)``).

The thermodynamics is the RAMSES-like barotropic one (``cfg.set_eos_reset_barotropic``): the hydro
step is adiabatic, and at the end of every timestep the internal energy of each cell is reset
onto the barotrope

.. math::
    P(\\rho) = c_{s,0}^2 \\rho \\left[1 + \\left(\\frac{\\rho}{\\rho_c}\\right)^{\\gamma - 1}\\right],

isothermal at low density and adiabatic above the critical density :math:`\\rho_c`.

The box is periodic (the Poisson solver is periodic), its size is four times the sphere radius.
The grid is refined so that the local Jeans length :math:`\\lambda_J = \\sqrt{\\pi c_s^2 / (G
\\rho)}` is always resolved by at least :math:`N_J` cells
(``cfg.set_amr_mode_jeans_length_based``), up to a maximum number of refinement levels. The
resolution parameters are kept low so that the example runs in a few minutes.

Adapted from the collapse setup of the PhD work of Léodasce Sewanou.
"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from shamrock.model_ramses import InstantBarotropic

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

from shamrock.model_ramses import HLLC_PVRS, ExtForceCoupling

shamrock.enable_experimental_features()

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Units
# ^^^^^
# The simulation runs in code units (au, yr, solar mass), the plots are in SI units, except for
# the radius which is in au.

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()
kb = ucte.kb()
m_H = ucte.proton_mass()

# SI to code units
u_m = codeu.get("m")
u_kg = codeu.get("kg")
u_s = codeu.get("s")
u_dens = u_kg / u_m**3

# code units to SI
to_m = 1 / u_m
to_dens = 1 / u_dens  # [kg/m^3]
to_pressure = u_m * u_s**2 / u_kg  # [Pa]
to_vel = u_s / u_m  # [m/s]

# %%
# Setup parameters
# ^^^^^^^^^^^^^^^^

# Physical parameters
T0 = 10.0  # [K]
mu = 2.3  # mean molecular weight of the molecular gas
R0 = 7.07e14 * u_m  # sphere radius (7.07e16 cm)
rho0 = 1.38e-15 * u_dens  # sphere density (1.38e-18 g/cm^3)
rho_ext = rho0 / 100  # density of the surrounding medium
rho_crit = 3.7e-10 * u_dens  # critical density of the barotrope (3.7e-13 g/cm^3)
gamma = 5.0 / 3.0

L0 = 4 * R0  # box size

# Resolution: base grid of base^3 blocks of 2^3 cells, refined up to max_amr_lev levels such that
# the Jeans length is resolved by at least N_J cells
base = 16
max_amr_lev = 5
N_J = 8

# int coords size of a base block, such that a cell of the finest level is one int coords unit
sz = 2 << max_amr_lev

# output times in units of the free-fall time: most of the collapse happens at the end, from the
# isothermal phase to the formation of the adiabatic core (rho > rho_crit), around 1.1 t_ff
frames_t_tff = np.concatenate([np.linspace(0, 1.0, 21), np.linspace(1.0, 1.125, 26)[1:]])
n_frames = len(frames_t_tff)
t_end_tff = frames_t_tff[-1]

sim_folder = "_to_trash/ramses_barotropic_collapse/"

# %%
# Deduced quantities

cs0 = np.sqrt(kb * T0 / (mu * m_H))  # isothermal sound speed
M0 = rho0 * 4.0 / 3.0 * np.pi * R0**3  # sphere mass
t_ff = np.sqrt(3.0 * np.pi / (32.0 * G * rho0))  # free-fall time
lambda_J = np.sqrt(np.pi * cs0**2 / (G * rho0))  # Jeans length

# thermal to gravitational energy ratio of the sphere
alpha = (1.5 * M0 * cs0**2) / (3.0 / 5.0 * G * M0**2 / R0)

scale_fact = L0 / (sz * base)  # grid int coords to [0, L0]^3
dx_base = L0 / (2 * base)  # cell size of the base grid
dx_min = scale_fact  # cell size of the finest level

# refinement levels as in RAMSES: a cell of level l is L0 / 2^l wide. Where the Jeans length is
# well resolved (the diffuse medium), the blocks are merged below the base level.
level_base = int(np.log2(2 * base))
level_max = level_base + max_amr_lev
level_min = 1

if shamrock.sys.world_rank() == 0:
    print(f"M0 = {M0:.3f} Msun, alpha = {alpha:.3f}")
    print(f"cs0 = {cs0 * to_vel:.2f} m/s, t_ff = {t_ff:.4e} yr")
    print(f"Jeans length = {lambda_J:.1f} au = {lambda_J / dx_base:.1f} base cells")
    print(f"base cell size = {dx_base:.1f} au, finest cell size = {dx_min:.1f} au")

# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)

# %%
# Setup the solver
# ^^^^^^^^^^^^^^^^

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

cfg = model.gen_default_config()
cfg.set_scale_factor(scale_fact)
cfg.set_Csafe(0.3)
cfg.set_eos_adiabatic(gamma)
cfg.set_cooling_mode(InstantBarotropic(rho0=rho_crit, gamma0=gamma, cs0=cs0))
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)
cfg.set_riemann_solver(HLLC_PVRS)
cfg.set_units(codeu)
cfg.set_gravity_mode_cg()
cfg.set_self_gravity_tol(1e-6)
cfg.set_self_gravity_Niter_max(3000)
cfg.set_ext_force_coupling(ExtForceCoupling.Trapezoidal)
cfg.set_amr_mode_old(False)
cfg.set_amr_mode_jeans_length_based(N_jeans=N_J)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base, base, base))

# %%
# Initial conditions
# ^^^^^^^^^^^^^^^^^^
# Uniform sphere at rest centered in the box, with the pressure of the barotrope everywhere (the
# sphere is therefore not in pressure equilibrium with the diffuse medium).


def barotrope_pressure(rho):
    return cs0**2 * rho * (1.0 + (rho / rho_crit) ** (gamma - 1.0))


def radius(rmin, rmax):
    center = 0.5 * (np.array(rmin) + np.array(rmax))
    return np.linalg.norm(center - 0.5 * L0)


def rho_map(rmin, rmax):
    if radius(rmin, rmax) < R0:
        return rho0
    else:
        return rho_ext


def rhovel_map(rmin, rmax):
    return (0.0, 0.0, 0.0)


def rhoetot_map(rmin, rmax):
    return barotrope_pressure(rho_map(rmin, rmax)) / (gamma - 1.0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)
model.set_field_value_lambda_f64("phi", lambda rmin, rmax: 0.0)
model.set_field_value_lambda_f64("phi_old", lambda rmin, rmax: 0.0)

# %%
# Data extraction utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# At each output we render density slices in the :math:`z = L_0 / 2` plane (the whole box and a
# zoom on its center), and compute for every cell its distance to the center :math:`r`, the
# density, pressure, temperature :math:`T = \mu m_H P / (k_B \rho)` and radial velocity
# :math:`v_r = \mathbf{v} \cdot \mathbf{r} / r`.

n_slice = 256
zoom = 8  # the zoomed slice spans L0 / zoom


def slice_positions(width):
    x = 0.5 * L0 + width * ((np.arange(n_slice) + 0.5) / n_slice - 0.5)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return [(x, y, 0.5 * L0) for x, y in zip(xx.ravel().tolist(), yy.ravel().tolist())]


slice_positions_box = slice_positions(L0)
slice_positions_zoom = slice_positions(L0 / zoom)


def render_rho_slices():
    return [
        np.array(model.render_slice("rho", "f64", pos)).reshape(n_slice, n_slice)
        for pos in (slice_positions_box, slice_positions_zoom)
    ]


def get_profiles():
    dic = ctx.collect_data()

    # cell centers, from the block bounds (blocks of 2^3 cells, ordered block by block, x fastest)
    nside = 2
    block_min = np.array(dic["cell_min"], dtype=float)
    block_max = np.array(dic["cell_max"], dtype=float)
    lid = np.arange(nside**3)
    local_coord = np.stack([lid % nside, (lid // nside) % nside, lid // (nside * nside)], axis=1)
    cell_size = (block_max - block_min) / nside
    pos = block_min[:, None, :] + (local_coord[None, :, :] + 0.5) * cell_size[:, None, :]
    pos = pos.reshape(-1, 3) * scale_fact - 0.5 * L0

    rho = np.array(dic["rho"])
    rhovel = np.array(dic["rhovel"]).reshape(-1, 3)
    rhoetot = np.array(dic["rhoetot"])

    # refinement level of the cells
    level = np.log2(L0 / (cell_size[:, 0] * scale_fact))
    level = np.repeat(np.rint(level), nside**3)

    r = np.linalg.norm(pos, axis=1)
    ekin = 0.5 * np.sum(rhovel**2, axis=1) / rho
    P = (gamma - 1.0) * (rhoetot - ekin)
    T = mu * m_H * P / (kb * rho)
    vr = np.sum(rhovel * pos, axis=1) / (rho * r)

    # in the units of the plots (the radius in au, i.e. code units, the rest in SI)
    return {
        "r": r.astype(np.float32),
        "rho": (rho * to_dens).astype(np.float32),
        "P": (P * to_pressure).astype(np.float32),
        "T": T.astype(np.float32),
        "vr": (vr * to_vel).astype(np.float32),
        "level": level.astype(np.int8),
    }


# %%
# Plotting utilities
# ^^^^^^^^^^^^^^^^^^
# The frames are saved on the fly during the run, so the plot limits are fixed beforehand (shared
# by every frame of the animations): the density ranges from the diffuse medium up to ten times
# the critical density of the barotrope, and the infall velocity up to 0.4 times the free-fall
# velocity of the sphere mass at the scale of a cell of the finest level.


def barotrope_temperature(rho):
    return T0 * (1.0 + (rho / rho_crit) ** (gamma - 1.0))


rho_plot_min = rho_ext / 3
rho_plot_max = 10 * rho_crit
v_ff_cell = np.sqrt(2 * G * M0 / dx_min)

# plot limits, in SI units (au for the radius)
lim_r = (0.1, np.sqrt(3) * 0.5 * L0 * 1.1)  # [au]
lim_rho = (rho_plot_min * to_dens, rho_plot_max * to_dens)  # [kg/m^3]
lim_P = (
    barotrope_pressure(rho_plot_min) * to_pressure,
    barotrope_pressure(rho_plot_max) * to_pressure,
)  # [Pa]
lim_T = (T0 / 2, 2 * barotrope_temperature(rho_plot_max))  # [K]
lim_vr = (-0.4 * v_ff_cell * to_vel, 0.1 * v_ff_cell * to_vel)  # [m/s]
log_rho_slice_lim = np.log10(lim_rho)

# barotrope (SI) for the overlay
rho_baro = np.logspace(np.log10(lim_rho[0]), np.log10(lim_rho[1]), 200)  # [kg/m^3]
P_baro = barotrope_pressure(rho_baro / to_dens) * to_pressure  # [Pa]


def plot_slices(arr_rho_box, arr_rho_zoom, t, iplot, dpi=150):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), dpi=dpi, layout="constrained")

    for ax, arr_rho, width in zip(axs, (arr_rho_box, arr_rho_zoom), (L0, L0 / zoom)):
        extent = np.array([-0.5, 0.5, -0.5, 0.5]) * width * to_m
        res = ax.imshow(
            np.log10(arr_rho.T * to_dens),
            cmap=matplotlib.colormaps["magma"],
            origin="lower",
            extent=extent,
            vmin=log_rho_slice_lim[0],
            vmax=log_rho_slice_lim[1],
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    axs[0].set_title("whole box")
    axs[1].set_title(f"zoom x{zoom}")
    fig.suptitle(f"t = {t / t_ff:0.3f} $t_{{\\rm ff}}$")
    cbar = fig.colorbar(res, ax=axs)
    cbar.set_label(r"$\log_{10} \rho$ [kg/m$^3$]")
    fig.savefig(os.path.join(sim_folder, f"rho_slice_{iplot:04d}.png"))
    plt.close(fig)


# points colored by refinement level, with a colormap that stays dark enough to be read on a white
# background (plasma without its light yellow end)
level_cmap = matplotlib.colors.ListedColormap(
    matplotlib.colormaps["plasma"](np.linspace(0.0, 0.8, level_max - level_min + 1))
)
scatter_kwargs = dict(
    s=1,
    alpha=0.5,
    linewidths=0,
    rasterized=True,
    cmap=level_cmap,
    vmin=level_min - 0.5,
    vmax=level_max + 0.5,
    zorder=2,
)


def plot_profiles(prof, t, iplot, dpi=250):
    fig, axs = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    ax_rho, ax_P, ax_T, ax_vr = axs.ravel()

    sc = ax_rho.scatter(prof["r"], prof["rho"], c=prof["level"], **scatter_kwargs)
    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_rho.set_xlim(lim_r)
    ax_rho.set_ylim(lim_rho)
    ax_rho.set_xlabel("r [au]")
    ax_rho.set_ylabel(r"$\rho$ [kg/m$^3$]")

    ax_P.scatter(prof["rho"], prof["P"], c=prof["level"], **scatter_kwargs)
    # the barotrope is drawn below the points (zorder) so that it does not hide them
    ax_P.plot(rho_baro, P_baro, color="red", linewidth=1.5, label="barotrope", zorder=1)
    ax_P.axvline(rho_crit * to_dens, color="grey", linestyle=":", label=r"$\rho_c$", zorder=1)
    ax_P.set_xscale("log")
    ax_P.set_yscale("log")
    ax_P.set_xlim(lim_rho)
    ax_P.set_ylim(lim_P)
    ax_P.set_xlabel(r"$\rho$ [kg/m$^3$]")
    ax_P.set_ylabel(r"$P$ [Pa]")
    ax_P.legend(loc="upper left")

    ax_T.scatter(prof["r"], prof["T"], c=prof["level"], **scatter_kwargs)
    ax_T.set_xscale("log")
    ax_T.set_yscale("log")
    ax_T.set_xlim(lim_r)
    ax_T.set_ylim(lim_T)
    ax_T.set_xlabel("r [au]")
    ax_T.set_ylabel("T [K]")

    ax_vr.scatter(prof["r"], prof["vr"], c=prof["level"], **scatter_kwargs)
    ax_vr.axhline(0, color="grey", linewidth=0.5)
    ax_vr.set_xscale("log")
    ax_vr.set_xlim(lim_r)
    ax_vr.set_ylim(lim_vr)
    ax_vr.set_xlabel("r [au]")
    ax_vr.set_ylabel(r"$v_r$ [m/s]")

    for ax in (ax_rho, ax_T, ax_vr):
        ax.axvline(R0, color="grey", linestyle=":", linewidth=0.8, zorder=1)

    fig.suptitle(f"t = {t / t_ff:0.3f} $t_{{\\rm ff}}$")
    cbar = fig.colorbar(sc, ax=axs, ticks=np.arange(level_min, level_max + 1), shrink=0.6)
    cbar.set_label("AMR level")
    cbar.solids.set(alpha=1)
    fig.savefig(os.path.join(sim_folder, f"profiles_{iplot:04d}.png"), dpi=dpi)
    plt.close(fig)


# %%
# Run the simulation
# ^^^^^^^^^^^^^^^^^^
# The frames are rendered and saved at each output.

frames_t = frames_t_tff * t_ff
hist_rho_max = []

for i, t in enumerate(frames_t):
    model.evolve_until(t)

    arr_rho_box, arr_rho_zoom = render_rho_slices()
    prof = get_profiles()
    hist_rho_max.append(np.max(prof["rho"]))

    if shamrock.sys.world_rank() == 0:
        plot_slices(arr_rho_box, arr_rho_zoom, t, i)
        plot_profiles(prof, t, i)
        print(
            f"frame {i:3d} / {n_frames}, t = {t / t_ff:.3f} t_ff, "
            f"max rho = {hist_rho_max[-1]:.3e} kg/m^3, "
            f"{len(prof['rho'])} cells, max level = {np.max(prof['level'])}"
        )

# %%
# Density slice
# ^^^^^^^^^^^^^

from shamrock.utils.plot import show_image_sequence

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(os.path.join(sim_folder, "rho_slice_*.png"), render_gif=True)

if shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(os.path.join(sim_folder, "rho_slice_collapse.gif"), writer=writer)

plt.show()

# %%
# Radial profiles
# ^^^^^^^^^^^^^^^
# One point per cell, colored by its refinement level: density and temperature versus radius,
# pressure versus density with the barotrope (below the points) (every cell lies on it, as the internal energy is reset at the end of each
# timestep), and radial velocity versus radius. The dotted vertical lines mark the initial radius
# of the sphere (and the critical density on the pressure plot).

ani = show_image_sequence(os.path.join(sim_folder, "profiles_*.png"), render_gif=True, dpi=250)

if shamrock.sys.world_rank() == 0:
    writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(os.path.join(sim_folder, "profiles_collapse.gif"), writer=writer)

plt.show()

# %%
# Central density
# ^^^^^^^^^^^^^^^
# Maximum density versus time, compared with the pressureless free-fall of a uniform sphere
# (:math:`\rho(t) = \rho_0 / \cos^6 \xi` with :math:`t = t_{\rm ff} \, 2 (\xi + \sin \xi \cos \xi)
# / \pi`), which diverges at :math:`t = t_{\rm ff}`. The thermal pressure slows the collapse down,
# and the finite resolution caps the density that can be reached.

xi = np.linspace(0, 0.5 * np.pi * 0.999, 1000)
t_freefall = 2.0 * (xi + np.sin(xi) * np.cos(xi)) / np.pi
rho_freefall = rho0 * to_dens / np.cos(xi) ** 6

plt.figure()
plt.plot(frames_t / t_ff, hist_rho_max, "o-", label="simulation")
plt.plot(t_freefall, rho_freefall, color="red", linestyle="--", label="pressureless free-fall")
plt.yscale("log")
plt.xlim(0, t_end_tff)
plt.ylim(rho0 * to_dens / 2, 2 * max(hist_rho_max))
plt.xlabel(r"$t / t_{\rm ff}$")
plt.ylabel(r"$\max \rho$ [kg/m$^3$]")
plt.legend()
plt.tight_layout()
plt.show()
