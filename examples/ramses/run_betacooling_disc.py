"""
Beta-cooled self-gravitating disc
==================================================

A self-gravitating disc around a star, cooled with a beta cooling
(:math:`t_{\\rm cool} = \\beta / \\Omega`, Gammie 2001, ``BetaCooling``).

"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from shamrock.model_ramses import BetaCooling, ExtForceCoupling, RiemannSolverMode

# maybe_njit applies numba.njit when numba is installed, and does nothing otherwise
from shamrock.utils.numba_helper import maybe_njit

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

# Grid
boxlen = 12.0
levelmax = 6
resolution = 2**levelmax
center = boxlen / 2.0

# Units
# G = 1 in the code units (unit_mass = 1 / G_SI kg)
G_SI = shamrock.Constants(shamrock.UnitSystem()).G()
units = shamrock.UnitSystem(unit_time=1.0, unit_length=1.0, unit_mass=1.0 / G_SI)
G_code = shamrock.Constants(units).G()
assert abs(G_code - 1) < 1e-12

# Gravity
star_mass = 1.0
sink_soft = 8
softening = sink_soft * boxlen / 2**levelmax  # = 0.09375

enable_self_gravity = True

# Hydro
gamma = 5.0 / 3.0
courant_factor = 0.8

# Cooling
enable_beta_cooling = True
disk_beta_cool = 10.0


# IC
contrast_factor = 1000.0
contrast_radius = 3.5
temper_iso = 0.01  # cs^2 at disk_radius
temper_expo = 1.0  # T propto R^-temper_expo
disk_density = 0.09070519419177281
disk_radius = 1.0
radius_min_factor = 0.1
radius_max_factor = 4.0
inner_iso_z_flaring = 1.0
smallr = 1e-5

# Outputs

# orbital period at disk_radius (delta_tout = 1 orbit, tend = 20 orbits)
orbit_period = 2 * np.pi * np.sqrt(disk_radius**3 / (G_code * star_mass))
tend = 20 * orbit_period  # 20 orbits at disk_radius
delta_tout = orbit_period  # 1 orbit at disk_radius
n_dumps = round(tend / delta_tout) + 1

# Plots
frame_width = 6.0


sim_folder = "_to_trash/godunov_betacooling_disc/"

# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)


# %%
# Initial conditions (``disk_condinit`` of RAMSES, Baehr et al. setup)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


@maybe_njit
def disk_ic(x, y, z):
    """Density, velocity and sound speed of the disk at (x, y, z) relative to the star"""

    rc_soft = np.sqrt(x * x + y * y + softening**2)
    rs_soft = np.sqrt(x * x + y * y + z * z + softening**2)

    # inner isothermal zone
    rin = np.sqrt(
        disk_radius
        * radius_min_factor
        * (disk_radius * radius_min_factor + inner_iso_z_flaring * np.abs(z))
    )

    # locally isothermal sound speed
    cs0 = np.sqrt(temper_iso)
    if rc_soft > rin:
        cs = cs0 * (rc_soft / disk_radius) ** (-temper_expo / 2.0)
    else:
        cs = cs0 * (rin / disk_radius) ** (-temper_expo / 2.0)

    # density
    H = 0.1 * rc_soft
    dens = disk_density * (disk_radius / rc_soft) ** 3 * np.exp(-0.5 * z * z / (H * H))

    if rc_soft > disk_radius * contrast_radius or np.abs(z) > 0.5 * disk_radius * contrast_radius:
        dens = dens / contrast_factor
    dens = max(dens, smallr)

    # angular velocity
    GM = G_code * star_mass
    if temper_expo == 1.0 or rc_soft > rin:
        omega2 = GM / (rc_soft * rc_soft * rs_soft) - (4.0 - temper_expo / 2.0) * (cs * cs) / (
            rc_soft * rc_soft
        )
    else:
        omega2 = GM / (rs_soft**3) - (3.0 - temper_expo / 2.0) * (cs * cs) / (rc_soft * rc_soft)
    omega = np.sqrt(max(omega2, 0.0))

    # velocity
    vx = -omega * y
    vy = omega * x
    vz = 0.0

    return dens, vx, vy, vz, cs


@maybe_njit
def cell_center(rmin, rmax):
    rmin = np.array(rmin) - center
    rmax = np.array(rmax) - center
    return (rmin + rmax) / 2


@maybe_njit
def rho_map(rmin, rmax):
    x, y, z = cell_center(rmin, rmax)
    dens, vx, vy, vz, cs = disk_ic(x, y, z)
    return dens


@maybe_njit
def rhovel_map(rmin, rmax):
    x, y, z = cell_center(rmin, rmax)
    dens, vx, vy, vz, cs = disk_ic(x, y, z)
    return (dens * vx, dens * vy, dens * vz)


@maybe_njit
def rhoe_map(rmin, rmax):
    x, y, z = cell_center(rmin, rmax)
    dens, vx, vy, vz, cs = disk_ic(x, y, z)

    # P = rho cs^2
    eint = dens * cs * cs / (gamma - 1.0)
    ekin = 0.5 * dens * (vx * vx + vy * vy + vz * vz)
    return eint + ekin


# %%
# Plotting helpers (frames of width ``deltax_frame`` centred on the star)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

npix = 256
frame_min = center - frame_width / 2
frame_max = center + frame_width / 2
frame_extent = [-frame_width / 2, frame_width / 2, -frame_width / 2, frame_width / 2]

_u = np.linspace(frame_min, frame_max, npix)
_a, _b = np.meshgrid(_u, _u)
_a, _b = _a.ravel(), _b.ravel()
_c = np.full_like(_a, center)

slices = {
    # name: (positions, xlabel, ylabel)
    "face": ([tuple(p) for p in np.column_stack([_a, _b, _c])], "x", "y"),
    "edge": ([tuple(p) for p in np.column_stack([_a, _c, _b])], "x", "z"),
}


def render(field, positions):
    if field == "temp":
        # T = P / rho (code units), as the RAMSES 'temp' movie variable
        rho = np.array(model.render_slice("rho", "f64", positions))
        rhov = np.array(model.render_slice("rhovel", "f64_3", positions))
        rhoe = np.array(model.render_slice("rhoetot", "f64", positions))
        eint = rhoe - 0.5 * np.sum(rhov**2, axis=1) / rho
        return (gamma - 1.0) * eint / rho
    return np.array(model.render_slice(field, "f64", positions))


plot_settings = {
    "rho": (r"$\rho$ [code unit]", "gist_heat", matplotlib.colors.LogNorm(vmin=1e-4, vmax=1e2)),
    "temp": (
        r"$T = P / \rho$ [code unit]",
        "magma",
        matplotlib.colors.LogNorm(vmin=1e-4, vmax=1e-1),
    ),
}


def plot(t, iplot):
    for field, (label, cmap_name, norm) in plot_settings.items():
        for slice_name, (positions, xlabel, ylabel) in slices.items():
            arr = render(field, positions).reshape(npix, npix)

            cmap = matplotlib.colormaps[cmap_name].copy()
            cmap.set_bad(color="yellow")

            plt.figure(dpi=150)
            res = plt.imshow(arr, cmap=cmap, origin="lower", extent=frame_extent, norm=norm)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"t = {t:0.3f} ({t / orbit_period:0.2f} orbits at R = {disk_radius})")
            cbar = plt.colorbar(res, extend="both")
            cbar.set_label(label)
            plt.savefig(os.path.join(sim_folder, f"{field}_{slice_name}_{iplot:04d}.png"))
            plt.close()


# %%
# Initialisation
# ^^^^^^^^^^^^^^

print("initialising model...")

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

cell_size = 2
base = resolution // cell_size

cfg = model.gen_default_config()

cfg.set_scale_factor(boxlen / (cell_size * base))
cfg.set_units(units)
cfg.set_eos_gamma(gamma)
cfg.set_Csafe(courant_factor)

cfg.set_riemann_solver(RiemannSolverMode.HLLC_Davis)
cfg.set_slope_lim_minmod()  # slope_type=1
cfg.set_face_time_interpolation(True)  # scheme='muscl'


for axis in ("x", "y", "z"):
    cfg.set_boundary_condition(axis, "periodic" if enable_self_gravity else "outflow")

cfg.add_ext_force_point_mass_gravity(
    mass=star_mass, pos=(center, center, center), epsilon_soft=softening
)
cfg.set_ext_force_coupling(ExtForceCoupling.Trapezoidal)

if enable_self_gravity:
    shamrock.enable_experimental_features()
    cfg.set_gravity_mode_cg()
    cfg.set_self_gravity_Niter_max(500)
    cfg.set_self_gravity_tol(1e-4)

if enable_beta_cooling:
    cfg.set_cooling_mode(BetaCooling(beta=disk_beta_cool, center=(center, center, center)))

model.set_solver_config(cfg)

model.init_scheduler(int(1e9), 1)  # (crit_split - patches, crit_merge - patches)
model.make_base_grid((0, 0, 0), (cell_size, cell_size, cell_size), (base, base, base))

print("initialising fields...", flush=True)
model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)
model.set_field_value_lambda_f64("rhoetot", rhoe_map)
if enable_self_gravity:
    model.set_field_value_lambda_f64("phi", lambda rmin, rmax: 0.0)
print("initialisation done", flush=True)


# %%
# Run the simulation
# ^^^^^^^^^^^^^^^^^^

for i in range(n_dumps):
    t = delta_tout * i
    model.evolve_until(t)
    d = ctx.collect_data()
    print(
        f"Output {i}, t = {t:0.3f}, rho in [{d['rho'].min():0.3e}, {d['rho'].max():0.3e}], "
        f"rhoe in [{d['rhoetot'].min():0.3e}, {d['rhoetot'].max():0.3e}]",
        flush=True,
    )
    plot(t, i)


# %%
# Slices
# ^^^^^^

from shamrock.utils.plot import show_image_sequence


def show_slice_animation(name):
    # If the animation is not returned only a static image will be shown in the doc
    ani = show_image_sequence(os.path.join(sim_folder, f"{name}_*.png"), render_gif=True)

    if shamrock.sys.world_rank() == 0:
        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(os.path.join(sim_folder, f"{name}.gif"), writer=writer)

    return ani


# %%
# Midplane density
ani = show_slice_animation("rho_face")
plt.show()

# %%
# Vertical density slice
ani = show_slice_animation("rho_edge")
plt.show()

# %%
# Midplane temperature
ani = show_slice_animation("temp_face")
plt.show()

# %%
# Vertical temperature slice
ani = show_slice_animation("temp_edge")
plt.show()
