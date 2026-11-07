"""
Protoplanetary disc around a point mass in RAMSES solver
========================================================

A locally isothermal disc in rotational and vertical hydrostatic equilibrium around a softened
point mass (``cfg.add_ext_force_point_mass_gravity``), optionally with self-gravity.
"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from shamrock.model_ramses import ExtForceCoupling

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
# ^^^^^^^^^^^^^^^^^^


## Units

# G = 1 in the code units (unit_mass = 1 / G_SI kg), self-gravity takes G from the unit system
G_SI = shamrock.Constants(shamrock.UnitSystem()).G()
units = shamrock.UnitSystem(unit_time=1.0, unit_length=1.0, unit_mass=1.0 / G_SI)
G_code = shamrock.Constants(units).G()
assert abs(G_code - 1) < 1e-12

## Grid setup
resolution = 64
boxlen = 1

## Initial conditions
contrast_factor = 10
ref_density = 1
ref_sound_speed = 0.2
cs_expo = 0.5
point_mass = 1
gravity_softening = 0.05
outer_radius = 0.25
inner_radius_factor = 0.1
inner_iso_z_flaring = 2
gamma0 = 1.00001
smallr = 1e-4
center = boxlen / 2.0

## Physics
enable_sg = False

## Outputs
dt_dump = 0.01
n_dumps = 21

sim_folder = "_to_trash/ramses_disc/"

# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)


# %%
# Initial conditions
# ^^^^^^^^^^^^^^^^^^


@maybe_njit
def get_cs(rc_soft, z):
    inner_radius = np.sqrt(
        outer_radius
        * inner_radius_factor
        * (outer_radius * inner_radius_factor + inner_iso_z_flaring * np.abs(z))
    )

    if rc_soft > inner_radius:
        cs = ref_sound_speed * (rc_soft / outer_radius) ** (-cs_expo)
    else:
        cs = ref_sound_speed * (inner_radius / outer_radius) ** (-cs_expo)

    return cs, inner_radius


@maybe_njit
def get_omega(rs_soft, rc_soft, cs, inner_radius):

    if cs_expo == 0.5 or rc_soft > inner_radius:
        omega2 = point_mass * G_code / (rc_soft * rc_soft * rs_soft) - (4.0 - cs_expo) * (
            cs * cs
        ) / (rc_soft * rc_soft)
        omega = np.sqrt(np.maximum(omega2, 0.0))
    else:
        omega2 = point_mass * G_code / (rs_soft**3) - (3.0 - cs_expo) * (cs * cs) / (
            rc_soft * rc_soft
        )
        omega = np.sqrt(np.maximum(omega2, 0.0))

    return omega


@maybe_njit
def rho_map(rmin, rmax):

    rmin = np.array(rmin) - center
    rmax = np.array(rmax) - center
    x, y, z = (rmin + rmax) / 2

    rc_soft = np.sqrt(x * x + y * y + gravity_softening**2)
    rs_soft = np.sqrt(x * x + y * y + z * z + gravity_softening**2)

    cs, _ = get_cs(rc_soft, z)

    # Radial power law
    dens = ref_density * (outer_radius / rc_soft) ** ((5.0 - 2.0 * cs_expo) / 2.0)

    # Vertical hydrostatic equilibrium
    dens *= np.exp((point_mass * G_code / (cs * cs)) * (1.0 / rs_soft - 1.0 / rc_soft))

    # Outer taper
    if rc_soft > outer_radius or np.abs(z) > 0.5 * outer_radius:
        dens /= contrast_factor

    return np.maximum(dens, smallr)


@maybe_njit
def rhovel_map(rmin, rmax):

    rho = rho_map(rmin, rmax)

    rmin = np.array(rmin) - center
    rmax = np.array(rmax) - center
    x, y, z = (rmin + rmax) / 2

    rc_soft = np.sqrt(x * x + y * y + gravity_softening**2)
    rs_soft = np.sqrt(x * x + y * y + z * z + gravity_softening**2)

    cs, inner_radius = get_cs(rc_soft, z)
    omega = get_omega(rs_soft, rc_soft, cs, inner_radius)

    vx = -omega * y
    vy = omega * x
    vz = 0.0

    return (rho * vx, rho * vy, rho * vz)


@maybe_njit
def rhoe_map(rmin, rmax):

    rho = rho_map(rmin, rmax)
    rho_vx, rho_vy, rho_vz = rhovel_map(rmin, rmax)

    rmin = np.array(rmin) - center
    rmax = np.array(rmax) - center
    x, y, z = (rmin + rmax) / 2

    rc_soft = np.sqrt(x * x + y * y + gravity_softening**2)
    cs, _ = get_cs(rc_soft, z)

    eint = rho * cs * cs / (gamma0 - 1.0)
    ekin = 0.5 * (rho_vx * rho_vx + rho_vy * rho_vy + rho_vz * rho_vz) / np.maximum(rho, smallr)

    return eint + ekin


def make_cartesian_coords(nx, ny, z_val, min_x, max_x, min_y, max_y):
    # Create the cylindrical coordinate grid
    x_vals = np.linspace(min_x, max_x, nx)
    y_vals = np.linspace(min_y, max_y, ny)

    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Convert to Cartesian coordinates (z = 0 for a disc in the xy-plane)
    z_grid = z_val * np.ones_like(x_grid)

    # Flatten and stack to create list of positions
    positions = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    return [tuple(pos) for pos in positions]


nx, ny = resolution, resolution


positions = make_cartesian_coords(nx, ny, center, 0, boxlen - 1e-6, 0, boxlen - 1e-6)


def plot_rho_slice_cartesian(
    metadata, arr_rho_pos, iplot, case_name, dpi=200, xlabel="x", ylabel="y"
):
    ext = metadata["extent"]

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="yellow")

    arr_rho_pos = np.array(arr_rho_pos).reshape(nx, ny)

    plt.figure(dpi=dpi)
    res = plt.imshow(
        arr_rho_pos,
        cmap=my_cmap,
        origin="lower",
        extent=ext,
        aspect="auto",
        norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=6),
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"t = {metadata['time']:0.5f} [code units]")
    cbar = plt.colorbar(res, extend="both")
    plt.gca().set_aspect("equal")
    cbar.set_label(r"$\rho$ [code unit]")
    plt.savefig(os.path.join(sim_folder, f"rho_{case_name}_{iplot:04d}.png"))
    plt.close()


def plot(t, iplot):
    metadata = {"extent": [0, boxlen, 0, boxlen], "time": t}
    arr_rho_pos = model.render_slice("rho", "f64", positions)
    plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, "disk_face")
    pos_XZ = [(x, z, y) for (x, y, z) in positions]
    arr_rho_pos = model.render_slice("rho", "f64", pos_XZ)
    plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, "disk_xz", ylabel="z")
    pos_YZ = [(z, x, y) for (x, y, z) in positions]
    arr_rho_pos = model.render_slice("rho", "f64", pos_YZ)
    plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, "disk_yz", xlabel="y", ylabel="z")


# %%
# Initialisation
# ^^^^^^^^^^^^^^^^^^


print("initialising model...")

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")


cell_size = 2
base = resolution // cell_size

cfg = model.gen_default_config()

scale_fact = boxlen / (cell_size * base)
cfg.set_scale_factor(scale_fact)
cfg.set_eos_gamma(gamma0)
cfg.set_Csafe(0.8)

cfg.set_riemann_solver_hllc_pvrs()
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)
cfg.set_units(units)

if enable_sg:
    shamrock.enable_experimental_features()
    cfg.set_gravity_mode_cg()
    cfg.set_self_gravity_Niter_max(500)
    cfg.set_self_gravity_tol(1e-4)
cfg.add_ext_force_point_mass_gravity(
    mass=point_mass, pos=(center, center, center), epsilon_soft=gravity_softening
)
cfg.set_ext_force_coupling(ExtForceCoupling.Trapezoidal)


model.set_solver_config(cfg)

model.init_scheduler(int(1e9), 1)  # (crit_split - patches, crit_merge - patches)
model.make_base_grid((0, 0, 0), (cell_size, cell_size, cell_size), (base, base, base))
print("initialising fields...", flush=True)

print("initial density...", flush=True)
model.set_field_value_lambda_f64("rho", rho_map)
print("initial momentum...", flush=True)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)
print("initial total energy...", flush=True)
model.set_field_value_lambda_f64("rhoetot", rhoe_map)
print("initialisation done, plotting initial conditions...", flush=True)


# %%
# Run the simulation
# ^^^^^^^^^^^^^^^^^^

for i in range(n_dumps):
    t = dt_dump * i
    model.evolve_until(t)
    d = ctx.collect_data()
    print(
        f"Step {i}, t = {t:0.5f} [code unit], min rho = {d['rho'].min():0.5e}, max rho = {d['rho'].max():0.5e}"
    )
    print(
        f"Step {i}, t = {t:0.5f} [code unit], min rhoe = {d['rhoetot'].min():0.5e}, max rhoe = {d['rhoetot'].max():0.5e}"
    )
    print(
        f"Step {i}, t = {t:0.5f} [code unit], min rhov = {d['rhovel'].min():0.5e}, max rhov = {d['rhovel'].max():0.5e}"
    )
    plot(t, i)


# %%
# Density slices
# ^^^^^^^^^^^^^^

from shamrock.utils.plot import show_image_sequence


def show_slice_animation(case_name):
    # If the animation is not returned only a static image will be shown in the doc
    ani = show_image_sequence(os.path.join(sim_folder, f"rho_{case_name}_*.png"), render_gif=True)

    if shamrock.sys.world_rank() == 0:
        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(os.path.join(sim_folder, f"rho_{case_name}.gif"), writer=writer)

    return ani


# %%
# Midplane slice (:math:`z = z_{\rm center}`)

ani = show_slice_animation("disk_face")
plt.show()

# %%
# Vertical slice (:math:`y = y_{\rm center}`)

ani = show_slice_animation("disk_xz")
plt.show()

# %%
# Vertical slice (:math:`x = x_{\rm center}`)

ani = show_slice_animation("disk_yz")
plt.show()
