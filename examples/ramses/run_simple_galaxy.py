import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numba import njit
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import i0, i1, kn
from shamrock.model_ramses import ExtForceCoupling

import shamrock

shamrock.enable_experimental_features()

# %%
# Parameters
# ^^^^^^^^^^^^^^^^^^


## Units
si = shamrock.UnitSystem()
SI_units = shamrock.Constants(si)
G_SI = SI_units.G()

code_t = 1e6 * SI_units.year()
code_l = SI_units.parsec()
code_m = code_l**3 / (code_t**2 * G_SI)
units = shamrock.UnitSystem(unit_time=code_t, unit_length=code_l, unit_mass=code_m)

kms_code = 1 / (1e3 * units.get("m") / units.get("s"))
pc_code = units.get("pc")

G_code = shamrock.Constants(units).G()
Msun_code = shamrock.Constants(units).sol_mass()
assert G_code == 1


## Grid setup
resolution = 128
boxlen = 40e3  # pc

## Initial conditions
cs = (
    np.sqrt(8.314 / (1.008e-3) * 1e4) / 1000 * kms_code
)  # km/s: isothermal sound speed for hydrogen only
h = 5.26e3 * pc_code  # pc
Rd = 16e3 * pc_code  # pc
r_0 = 4e3 * pc_code  # pc
rho_0 = 1.0 * Msun_code / pc_code  # solar mass/pc3
rho_c = 3 * Msun_code / pc_code  # solar mass/pc3
M_halo = 1.03e11 * Msun_code  # solar mass

center = boxlen / 2  # Central position
gamma0 = 1.6

smallr = 1e-4

# %%
# Initial conditions
# ^^^^^^^^^^^^^^^^^^


def rho_halo(r, rho_0=rho_0, r_0=r_0):
    return rho_0 / ((1 + r / r_0) * (1 + (r / r_0) ** 2))


def rho_gas(r_c, z, cs=cs, h=h, rho_c=rho_c):
    if z == 0:
        return rho_c * np.exp(-2 * r_c / h)
    else:
        z_0 = cs / np.sqrt(2 * np.pi * G_code * rho_gas(r_c, 0, cs=cs, h=h))
        return rho_c * np.exp(-2 * r_c / h) * 1 / np.cosh(z / z_0) ** 2


def rho_map(rmin, rmax):

    rmin = np.array(rmin) - center
    rmax = np.array(rmax) - center
    x, y, z = (rmin + rmax) / 2

    r_c = np.sqrt(x * x + y * y)
    rho = rho_gas(r_c, z, cs, h, rho_c)

    return np.maximum(rho, smallr)


def rhovel_map(rmin, rmax):

    rho = rho_map(rmin, rmax)

    rmin = np.array(rmin) - center
    rmax = np.array(rmax) - center
    x, y, z = (rmin + rmax) / 2

    r_c = np.sqrt(x * x + y * y)
    rho = rho_gas(r_c, z, cs, h, rho_c)

    ## circular velocity DM
    r_norm = r_c / r_0
    vc_DM = (
        (np.pi * G_code * rho * r_0**2)
        / r_norm
        * (np.log(1 + r_norm) ** 2 * (1 + r_norm**2) - 2 * np.arctan(r_norm))
    )

    ## circular velocity gas: exponential galaxy Freeman (1970)
    Sigma_0 = rho_c * 2 * (cs / np.sqrt(2 * np.pi * G_code * rho))
    vc_gas = np.sqrt(
        4
        * np.pi
        * G_code
        * Sigma_0
        * h
        * r_c**2
        / (2 * h) ** 2
        * (i0(r_c / (2 * h)) * kn(0, r_c / (2 * h)) - i1(r_c / (2 * h)) * kn(1, r_c / (2 * h)))
    )

    ## circular velocity pressure gradient
    vc_grad_p = -np.sqrt(r_c * cs**2 * rho_c * 2 * np.exp(-2 * r_c / h) / h / rho)

    vc = np.sqrt(vc_DM**2 + vc_gas**2 - vc_grad_p**2)

    vx = -vc * y / r_c
    vy = vc * x / r_c
    vz = 0.0

    if np.any(np.isnan(vx + vy)):
        raise RuntimeError(f"nan at r={rmin, rmax}")

    return (rho * vx, rho * vy, rho * vz)


def rhoe_map(rmin, rmax):

    rho = rho_map(rmin, rmax)
    rho_vx, rho_vy, rho_vz = rhovel_map(rmin, rmax)

    rmin = np.array(rmin) - center
    rmax = np.array(rmax) - center

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
    plt.savefig(os.path.join(".", f"galaxy_rho_{case_name}_{iplot:04d}.png"))
    plt.close()


def plot(t, iplot):
    metadata = {"extent": [0, 1, 0, 1], "time": t}
    arr_rho_pos = model.render_slice("rho", "f64", positions)
    plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, "galaxy_xy")
    pos_XZ = [(x, z, y) for (x, y, z) in positions]
    arr_rho_pos = model.render_slice("rho", "f64", pos_XZ)
    plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, "galaxy_xz")
    pos_YZ = [(z, y, x) for (x, y, z) in positions]
    arr_rho_pos = model.render_slice("rho", "f64", pos_YZ)
    plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, "galaxy_yz")


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
cfg.set_eos_reset_barotropic(rho_crit=1.0, cs0=cs, gamma_baro=1.0, gamma=gamma0)
cfg.set_Csafe(0.5)

cfg.set_riemann_solver_hllc()
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)
cfg.set_units(units)
cfg.set_gravity_mode_cg()
cfg.set_self_gravity_Niter_max(500)
cfg.set_self_gravity_tol(1e-4)
cfg.add_ext_force_burkert_profile(rho_0=rho_0, r_0=r_0, pos=(center, center, center))
cfg.set_ext_force_coupling(ExtForceCoupling.Trapezoidal)
cfg.to_json()

model.set_solver_config(cfg)

model.init_scheduler(int(1e9), 1)  # (crit_split - patches, crit_merge - patches)
model.make_base_grid((0, 0, 0), (cell_size, cell_size, cell_size), (base, base, base))


def launch_sim():
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

    dt = 0.01
    for i in range(30):
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
