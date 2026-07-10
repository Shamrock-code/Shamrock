import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

import shamrock

### Parameters

## Grid setup
resolution = 128

## Initial conditions
contrast_factor = 1000
ref_density = 1
ref_sound_speed = 0.2
cs_expo = 0.5
point_mass_Gmass = 1
gravity_softening = 0.05
outer_radius = 0.25
inner_radius_factor = 0.1
inner_iso_z_flaring = 2
gamma0 = 1.6
smallr = 1e-4
center = 0.5
h_over_r = 1


## Initial conditions


@njit
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


@njit
def get_omega(rs_soft, rc_soft, cs, inner_radius):

    if cs_expo == 0.5 or rc_soft > inner_radius:
        omega2 = point_mass_Gmass / (rc_soft * rc_soft * rs_soft) - (4.0 - cs_expo) * (cs * cs) / (
            rc_soft * rc_soft
        )
        omega = np.sqrt(np.maximum(omega2, 0.0))
    else:
        omega2 = point_mass_Gmass / (rs_soft**3) - (3.0 - cs_expo) * (cs * cs) / (rc_soft * rc_soft)
        omega = np.sqrt(np.maximum(omega2, 0.0))

    return omega


@njit
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
    dens *= np.exp((point_mass_Gmass / (cs * cs)) * (1.0 / rs_soft - 1.0 / rc_soft))

    # Outer taper
    if rc_soft > outer_radius or np.abs(z) > 0.5 * outer_radius:
        dens /= contrast_factor

    return np.maximum(dens, smallr)


@njit
def rhovel_map(rmin, rmax):

    rho = rho_map(rmin, rmax)

    rmin = np.array(rmin) - center
    rmax = np.array(rmax) - center
    x, y, z = (rmin + rmax) / 2

    rc_soft = np.sqrt(x * x + y * y + gravity_softening**2)
    rs = np.sqrt(x * x + y * y + z * z)
    rs_soft = np.sqrt(x * x + y * y + z * z + gravity_softening**2)

    cs, inner_radius = get_cs(rc_soft, z)
    omega = get_omega(rs_soft, rc_soft, cs, inner_radius)

    x_soft = x * (rs_soft / rs)
    y_soft = y * (rs_soft / rs)

    vx = -omega * y_soft
    vy = omega * x_soft
    vz = 0.0

    return (rho * vx, rho * vy, rho * vz)


@njit
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


nx, ny = 128, 128


positions = make_cartesian_coords(nx, ny, 0.5, 0, 1 - 1e-6, 0, 1 - 1e-6)


def plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, case_name, dpi=200):
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
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.5f} [seconds]")
    cbar = plt.colorbar(res, extend="both")
    plt.gca().set_aspect("equal")
    cbar.set_label(r"$\rho$ [code unit]")
    plt.savefig(os.path.join(".", f"rho_{case_name}_{iplot:04d}.png"))
    plt.close()


def plot(t, iplot):
    metadata = {"extent": [0, 1, 0, 1], "time": t}
    arr_rho_pos = model.render_slice("rho", "f64", positions)
    plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, "disk")
    pos_XZ = [(x, z, y) for (x, y, z) in positions]
    arr_rho_pos = model.render_slice("rho", "f64", pos_XZ)
    plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, "disk_xz")
    pos_YZ = [(z, y, x) for (x, y, z) in positions]
    arr_rho_pos = model.render_slice("rho", "f64", pos_YZ)
    plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, "disk_yz")


def plot_force(t, iplot):
    metadata = {"extent": [0, 1, 0, 1], "time": t}
    force = model.render_slice("gravitational_force", "f64_3", positions)
    force = np.array(force).reshape(nx, ny, 3)
    force = np.linalg.norm(force, axis=2)
    plot_rho_slice_cartesian(metadata, force, iplot, "force")


### Initialisation

print("initialising model...")

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")


cell_size = 2
base = resolution

cfg = model.gen_default_config()

scale_fact = 1 / (cell_size * base)
cfg.set_scale_factor(scale_fact)
cfg.set_eos_gamma(1.00001)

cfg.set_riemann_solver_hll()
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)

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

dt = 0.01
for i in range(0, 30):
    t = dt * i
    model.evolve_until(dt * i)
    d = ctx.collect_data()
    print(
        f"Step {i}, t = {t:0.5f} seconds, min rho = {d['rho'].min():0.5e}, max rho = {d['rho'].max():0.5e}"
    )
    print(
        f"Step {i}, t = {t:0.5f} seconds, min rhoe = {d['rhoetot'].min():0.5e}, max rhov = {d['rhoetot'].max():0.5e}"
    )
    print(
        f"Step {i}, t = {t:0.5f} seconds, min rhov = {d['rhovel'].min():0.5e}, max rhov = {d['rhovel'].max():0.5e}"
    )
    plot(t, i)


plot(dt * (i + 1), i)
