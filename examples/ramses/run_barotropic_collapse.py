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
The grid is uniform: this low resolution is chosen so that the example runs in a few minutes
(adaptive refinement on the Jeans length would be needed to follow the collapse further).

Adapted from the collapse setup of the PhD work of Léodasce Sewanou.
"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

from shamrock.model_ramses import ExtForceCoupling

shamrock.enable_experimental_features()

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Setup parameters
# ^^^^^^^^^^^^^^^^
# The simulation runs in SI units.

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
G = sicte.G()
kb = sicte.kb()
m_H = sicte.proton_mass()

au = si.get("au")
year = si.get("yr")

# Physical parameters
T0 = 10.0  # [K]
mu = 2.3  # mean molecular weight of the molecular gas
R0 = 7.07e16 * 1e-2  # sphere radius [cm -> m]
rho0 = 1.38e-18 * 1e3  # sphere density [g/cm^3 -> kg/m^3]
rho_ext = rho0 / 100  # density of the surrounding medium
rho_crit = 3.7e-13 * 1e3  # critical density of the barotrope [g/cm^3 -> kg/m^3]
gamma = 5.0 / 3.0

L0 = 4 * R0  # box size

# Resolution: uniform grid of (sz * base)^3 cells
sz = 1 << 1
base = 32

t_end_tff = 1.2  # end time in units of the free-fall time
n_frames = 49

sim_folder = "_to_trash/ramses_barotropic_collapse/"

# %%
# Deduced quantities

cs0 = np.sqrt(kb * T0 / (mu * m_H))  # isothermal sound speed
M0 = rho0 * 4.0 / 3.0 * np.pi * R0**3  # sphere mass
t_ff = np.sqrt(3.0 * np.pi / (32.0 * G * rho0))  # free-fall time
lambda_J = np.sqrt(np.pi * cs0**2 / (G * rho0))  # Jeans length

# thermal to gravitational energy ratio of the sphere
alpha = (1.5 * M0 * cs0**2) / (3.0 / 5.0 * G * M0**2 / R0)

N = sz * base  # cells per side
scale_fact = L0 / N  # grid int coords to [0, L0]^3 (a cell spans one unit of int coords)
dx = scale_fact  # cell size

if shamrock.sys.world_rank() == 0:
    print(f"M0 = {M0 / sicte.sol_mass():.3f} Msun, alpha = {alpha:.3f}")
    print(f"cs0 = {cs0:.2f} m/s, t_ff = {t_ff / year:.4e} yr")
    print(f"Jeans length = {lambda_J / au:.1f} au = {lambda_J / dx:.1f} cells")

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
cfg.set_eos_reset_barotropic(rho_crit=rho_crit, cs0=cs0, gamma_baro=gamma, gamma=gamma)
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)
cfg.set_riemann_solver_hllc()
cfg.set_units(si)
cfg.set_gravity_mode_cg()
cfg.set_self_gravity_tol(1e-6)
cfg.set_self_gravity_Niter_max(3000)
cfg.set_ext_force_coupling(ExtForceCoupling.Trapezoidal)
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
# At each output we render a density slice in the :math:`z = L_0 / 2` plane, and compute for every
# cell its distance to the center :math:`r`, the density, pressure, temperature
# :math:`T = \mu m_H P / (k_B \rho)` and radial velocity :math:`v_r = \mathbf{v} \cdot
# \mathbf{r} / r`.

n_slice = 256
slice_x = np.linspace(0, L0, n_slice + 1)[:-1] + 0.5 * L0 / n_slice
slice_xx, slice_yy = np.meshgrid(slice_x, slice_x, indexing="ij")
slice_positions = [
    (x, y, 0.5 * L0) for x, y in zip(slice_xx.ravel().tolist(), slice_yy.ravel().tolist())
]


def render_rho_slice():
    return np.array(model.render_slice("rho", "f64", slice_positions)).reshape(n_slice, n_slice)


def get_profiles():
    dic = ctx.collect_data()

    # cell centers, from the block bounds (cells are ordered block by block, x fastest)
    block_min = np.array(dic["cell_min"], dtype=float)
    block_max = np.array(dic["cell_max"], dtype=float)
    lid = np.arange(sz**3)
    local_coord = np.stack([lid % sz, (lid // sz) % sz, lid // (sz * sz)], axis=1)
    cell_size = (block_max - block_min) / sz
    pos = block_min[:, None, :] + (local_coord[None, :, :] + 0.5) * cell_size[:, None, :]
    pos = pos.reshape(-1, 3) * scale_fact - 0.5 * L0

    rho = np.array(dic["rho"])
    rhovel = np.array(dic["rhovel"]).reshape(-1, 3)
    rhoetot = np.array(dic["rhoetot"])

    r = np.linalg.norm(pos, axis=1)
    ekin = 0.5 * np.sum(rhovel**2, axis=1) / rho
    P = (gamma - 1.0) * (rhoetot - ekin)
    T = mu * m_H * P / (kb * rho)
    vr = np.sum(rhovel * pos, axis=1) / (rho * r)

    # in the units of the plots (au, g/cm^3, dyn/cm^2, K, km/s)
    return {
        "r": (r / au).astype(np.float32),
        "rho": (rho * 1e-3).astype(np.float32),
        "P": (P * 10).astype(np.float32),
        "T": T.astype(np.float32),
        "vr": (vr * 1e-3).astype(np.float32),
    }


# %%
# Plotting utilities
# ^^^^^^^^^^^^^^^^^^
# The frames are saved on the fly during the run, so the plot limits are fixed beforehand (shared
# by every frame of the animations): the density ranges from the diffuse medium up to ten times
# the critical density of the barotrope, and the infall velocity up to 1.5 times the free-fall
# velocity of the sphere mass at the scale of a cell.


def barotrope_temperature(rho):
    return T0 * (1.0 + (rho / rho_crit) ** (gamma - 1.0))


rho_plot_min = rho_ext / 3  # [kg/m^3]
rho_plot_max = 10 * rho_crit  # [kg/m^3]
v_ff_cell = np.sqrt(2 * G * M0 / dx)  # [m/s]

lim_r = (0.4 * dx / au, np.sqrt(3) * 0.5 * L0 / au * 1.1)  # [au]
lim_rho = (rho_plot_min * 1e-3, rho_plot_max * 1e-3)  # [g/cm^3]
lim_P = (
    barotrope_pressure(rho_plot_min) * 10,
    barotrope_pressure(rho_plot_max) * 10,
)  # [dyn/cm^2]
lim_T = (T0 / 2, 2 * barotrope_temperature(rho_plot_max))  # [K]
lim_vr = (-1.5 * v_ff_cell * 1e-3, 0.2 * v_ff_cell * 1e-3)  # [km/s]
log_rho_slice_lim = np.log10(lim_rho)

# barotrope (cgs) for the overlay
rho_baro = np.logspace(np.log10(lim_rho[0]), np.log10(lim_rho[1]), 200)
P_baro = barotrope_pressure(rho_baro * 1e3) * 10


def plot_slice(arr_rho, t, iplot, dpi=150):
    plt.figure(dpi=dpi)
    extent = np.array([-0.5 * L0, 0.5 * L0, -0.5 * L0, 0.5 * L0]) / au
    res = plt.imshow(
        np.log10(arr_rho.T * 1e-3),
        cmap=matplotlib.colormaps["magma"],
        origin="lower",
        extent=extent,
        vmin=log_rho_slice_lim[0],
        vmax=log_rho_slice_lim[1],
        interpolation="nearest",
        aspect="equal",
    )
    plt.xlabel("x [au]")
    plt.ylabel("y [au]")
    plt.title(f"t = {t / t_ff:0.3f} $t_{{\\rm ff}}$")
    cbar = plt.colorbar(res)
    cbar.set_label(r"$\log_{10} \rho$ [g/cm$^3$]")
    plt.tight_layout()
    plt.savefig(os.path.join(sim_folder, f"rho_slice_{iplot:04d}.png"))
    plt.close()


scatter_kwargs = dict(s=1, alpha=0.3, linewidths=0, rasterized=True)


def plot_profiles(prof, t, iplot, dpi=250):
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    ax_rho, ax_P, ax_T, ax_vr = axs.ravel()

    ax_rho.scatter(prof["r"], prof["rho"], **scatter_kwargs)
    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_rho.set_xlim(lim_r)
    ax_rho.set_ylim(lim_rho)
    ax_rho.set_xlabel("r [au]")
    ax_rho.set_ylabel(r"$\rho$ [g/cm$^3$]")

    ax_P.scatter(prof["rho"], prof["P"], **scatter_kwargs)
    ax_P.plot(rho_baro, P_baro, color="red", linestyle="--", label="barotrope")
    ax_P.axvline(rho_crit * 1e-3, color="grey", linestyle=":", label=r"$\rho_c$")
    ax_P.set_xscale("log")
    ax_P.set_yscale("log")
    ax_P.set_xlim(lim_rho)
    ax_P.set_ylim(lim_P)
    ax_P.set_xlabel(r"$\rho$ [g/cm$^3$]")
    ax_P.set_ylabel(r"$P$ [dyn/cm$^2$]")
    ax_P.legend(loc="upper left")

    ax_T.scatter(prof["r"], prof["T"], **scatter_kwargs)
    ax_T.set_xscale("log")
    ax_T.set_yscale("log")
    ax_T.set_xlim(lim_r)
    ax_T.set_ylim(lim_T)
    ax_T.set_xlabel("r [au]")
    ax_T.set_ylabel("T [K]")

    ax_vr.scatter(prof["r"], prof["vr"], **scatter_kwargs)
    ax_vr.axhline(0, color="grey", linewidth=0.5)
    ax_vr.set_xscale("log")
    ax_vr.set_xlim(lim_r)
    ax_vr.set_ylim(lim_vr)
    ax_vr.set_xlabel("r [au]")
    ax_vr.set_ylabel(r"$v_r$ [km/s]")

    for ax in (ax_rho, ax_T, ax_vr):
        ax.axvline(R0 / au, color="grey", linestyle=":", linewidth=0.8)

    fig.suptitle(f"t = {t / t_ff:0.3f} $t_{{\\rm ff}}$")
    fig.tight_layout()
    fig.savefig(os.path.join(sim_folder, f"profiles_{iplot:04d}.png"), dpi=dpi)
    plt.close(fig)


# %%
# Run the simulation
# ^^^^^^^^^^^^^^^^^^
# The frames are rendered and saved at each output.

frames_t = np.linspace(0, t_end_tff, n_frames) * t_ff
hist_rho_max = []

for i, t in enumerate(frames_t):
    model.evolve_until(t)

    arr_rho = render_rho_slice()
    prof = get_profiles()
    hist_rho_max.append(np.max(prof["rho"]))

    if shamrock.sys.world_rank() == 0:
        plot_slice(arr_rho, t, i)
        plot_profiles(prof, t, i)
        print(
            f"frame {i:3d} / {n_frames}, t = {t / t_ff:.3f} t_ff, "
            f"max rho = {hist_rho_max[-1]:.3e} g/cm^3"
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
# One point per cell: density and temperature versus radius, pressure versus density with the
# barotrope overlaid (every cell lies on it, as the internal energy is reset at the end of each
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
rho_freefall = rho0 * 1e-3 / np.cos(xi) ** 6

plt.figure()
plt.plot(frames_t / t_ff, hist_rho_max, "o-", label="simulation")
plt.plot(t_freefall, rho_freefall, color="red", linestyle="--", label="pressureless free-fall")
plt.yscale("log")
plt.xlim(0, t_end_tff)
plt.ylim(rho0 * 1e-3 / 2, 2 * max(hist_rho_max))
plt.xlabel(r"$t / t_{\rm ff}$")
plt.ylabel(r"$\max \rho$ [g/cm$^3$]")
plt.legend()
plt.tight_layout()
plt.show()
