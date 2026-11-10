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
boxlen = 48.0
levelmax = 12
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


sim_folder = "betacooling_disc/"

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
    rs = max(np.sqrt(x * x + y * y + z * z), 1e-30)
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

    # velocity, with the softened coordinates of RAMSES (x * rs_soft / rs)
    vx = -omega * y * (rs_soft / rs)
    vy = omega * x * (rs_soft / rs)
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
# Plotting helpers (frames of width ``deltax_frame`` centred on the star, as the RAMSES movies)
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
# Vertical density slice (:math:`y = y_{\rm star}`)
ani = show_slice_animation("rho_edge")
plt.show()

# %%
# Midplane temperature
ani = show_slice_animation("temp_face")
plt.show()

# %%
# Vertical temperature slice (:math:`y = y_{\rm star}`)
ani = show_slice_animation("temp_edge")
plt.show()


# %%
# Analysis of the last output (20 orbits)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# * radial profiles of the surface density :math:`\Sigma`, of the Toomre parameter
#   :math:`Q = c_s \Omega_K / (\pi G \Sigma)` and of :math:`c_s / v_\phi`, averaged on annuli;
# * the Reynolds and gravitational stresses,
#   :math:`\alpha_R = \frac{2}{3 \gamma P} \int \rho v_R \delta v_\phi dz` and
#   :math:`\alpha_G = \frac{2}{3 \gamma P} \int \frac{g_R g_\phi}{4 \pi G} dz`, compared with the
#   gravito-turbulent value :math:`\alpha = 4 / (9 \gamma (\gamma - 1) \beta)` (Gammie 2001);
# * a map of the surface density.
#
# The analysis is done in the box of width ``frame_width`` centred on the star, as in the RAMSES
# script (the yt ``ds.box`` region). The gravitational acceleration :math:`\mathbf{g}` is the one of
# the gas self-gravity (:math:`-\nabla \phi`), without the star.

from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

G = G_code
M = star_mass
iout = n_dumps - 1
t_last = model.get_time()
print(f"analysis of output {iout}, t = {t_last:0.3f} ({t_last / orbit_period:0.2f} orbits)")


def get_cube():
    """Fields of the uniform grid as (resolution, resolution, resolution) arrays, indexed [ix, iy, iz]"""
    dic = ctx.collect_data()

    # cell centers, from the block bounds (cells are ordered block by block, x fastest)
    block_min = np.array(dic["cell_min"], dtype=float)
    block_max = np.array(dic["cell_max"], dtype=float)
    lid = np.arange(cell_size**3)
    local_coord = np.stack(
        [lid % cell_size, (lid // cell_size) % cell_size, lid // (cell_size * cell_size)], axis=1
    )
    csize = (block_max - block_min) / cell_size
    pos = block_min[:, None, :] + (local_coord[None, :, :] + 0.5) * csize[:, None, :]
    pos = pos.reshape(-1, 3) * (boxlen / (cell_size * base))

    dx_cell = boxlen / resolution
    idx = np.floor(pos / dx_cell).astype(int)

    def to_cube(arr):
        cube = np.zeros((resolution, resolution, resolution) + arr.shape[1:])
        cube[idx[:, 0], idx[:, 1], idx[:, 2]] = arr
        return cube

    cube = {
        "rho": to_cube(np.array(dic["rho"])),
        "rhovel": to_cube(np.array(dic["rhovel"]).reshape(-1, 3)),
        "rhoetot": to_cube(np.array(dic["rhoetot"])),
    }
    if enable_self_gravity:
        cube["phi"] = to_cube(np.array(dic["phi"]))
    return cube, dx_cell


cube, dx_cell = get_cube()

# coordinates relative to the star
coords_1d = (np.arange(resolution) + 0.5) * dx_cell - center
X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")

# self-gravity acceleration g = -grad(phi), centred differences (periodic, as the Poisson solver)
if enable_self_gravity:
    phi = cube["phi"]
    g_x = -(np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx_cell)
    g_y = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx_cell)
else:
    g_x = np.zeros_like(X)
    g_y = np.zeros_like(X)

# region of width frame_width centred on the star (the yt box of the RAMSES script)
in_region = (
    (np.abs(X) <= frame_width / 2) & (np.abs(Y) <= frame_width / 2) & (np.abs(Z) <= frame_width / 2)
)

x = X[in_region]
y = Y[in_region]
z = Z[in_region]
dv = np.full_like(x, dx_cell**3)
rho = cube["rho"][in_region]
vx = cube["rhovel"][..., 0][in_region] / rho
vy = cube["rhovel"][..., 1][in_region] / rho
vz = cube["rhovel"][..., 2][in_region] / rho
press = (gamma - 1) * (cube["rhoetot"][in_region] - 0.5 * rho * (vx**2 + vy**2 + vz**2))
gx = g_x[in_region]
gy = g_y[in_region]
R = np.sqrt(x**2 + y**2)

Rbins = 64


def vertical_integrate(val, R=R, dv=dv, return_R_grid=False):
    # extent of vertical integrate
    box_height = np.amax(z) - np.amin(z)
    # just average around given R
    R_grid = np.geomspace(0.3, 3, Rbins)
    sig = 1 / 64
    res = np.zeros_like(R_grid)
    for i in range(len(R_grid)):
        I = (R > R_grid[i] * (1 - 3 * sig)) & (R <= R_grid[i] * (1 + 3 * sig))
        if not np.any(I):
            # annulus thinner than a cell (inner radii at low resolution): no data
            res[i] = np.nan
            continue
        weight = np.exp(-((R[I] / R_grid[i] - 1) ** 2) / (2 * sig**2))
        res[i] = np.sum(val[I] * dv[I] * weight) / np.sum(dv[I] * weight) * box_height
    if return_R_grid:
        return R_grid, res
    return res


R_1d, Sigma = vertical_integrate(rho, return_R_grid=True)
P = vertical_integrate(press)
cs_iso = np.sqrt(P / Sigma)
cs = cs_iso * np.sqrt(gamma)
Q = cs * np.sqrt(G * M / R_1d**3) / (np.pi * G * Sigma)

vphi = (-vx * y + vy * x) / R
vR = (vx * x + vy * y) / R
vphi_1d = vertical_integrate(vphi * rho) / Sigma

f_vphi = interp1d(R_1d, vphi_1d, fill_value="extrapolate")
dvphi = vphi - f_vphi(R)
gphi = (-gx * y + gy * x) / R
gR = (gx * x + gy * y) / R
alpha_R = vertical_integrate(rho * vR * dvphi) * 2 / 3 / gamma / P
alpha_G = vertical_integrate(gR * gphi) / (4 * np.pi * G) * 2 / 3 / gamma / P

# gravito-turbulent stress expected for a beta cooling (Gammie 2001)
alpha_analytic = 4 / (9 * gamma * (gamma - 1) * disk_beta_cool)

# Sigma snapshot, at the grid resolution (the RAMSES level 9 cell size)
in_frame = np.abs(coords_1d) <= frame_width / 2
xf, yf = np.meshgrid(coords_1d[in_frame], coords_1d[in_frame], indexing="ij")
rf = np.sqrt(xf**2 + yf**2)
rho_frame = cube["rho"][np.ix_(in_frame, in_frame, in_frame)]
box_height = np.amax(z) - np.amin(z)
Sigma_2d = np.mean(rho_frame, axis=2) * box_height
Sigma_2d[(rf < 0.3) | (rf > 3)] = 0

if shamrock.sys.world_rank() == 0:
    np.savez(
        os.path.join(sim_folder, f"cart_t3_{iout}.npz"),
        R=R_1d,
        Q=Q,
        Sigma=Sigma,
        cs_vphi=cs / vphi_1d,
        alpha=alpha_R + alpha_G,
        alpha_G=alpha_G,
        alpha_R=alpha_R,
        xf=xf,
        yf=yf,
        Sigma_2d=Sigma_2d,
    )

# %%
# Column density projections (width 8, as the yt ``ProjectionPlot``)

proj_width = 8.0
in_proj = np.abs(coords_1d) <= proj_width / 2
proj_extent = [
    coords_1d[in_proj][0] - dx_cell / 2,
    coords_1d[in_proj][-1] + dx_cell / 2,
    coords_1d[in_proj][0] - dx_cell / 2,
    coords_1d[in_proj][-1] + dx_cell / 2,
]

for axis_name, axis, (xlabel, ylabel) in (("z", 2, ("x", "y")), ("x", 0, ("y", "z"))):
    column = np.sum(cube["rho"], axis=axis) * dx_cell  # integrated through the whole box
    column = column[np.ix_(in_proj, in_proj)]

    plt.figure()
    res = plt.imshow(
        column.T,
        origin="lower",
        extent=proj_extent,
        cmap="inferno",
        norm=LogNorm(),
        rasterized=True,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(
        f"projection along {axis_name}, t = {t_last:0.2f} ({t_last / orbit_period:0.1f} orbits)"
    )
    plt.colorbar(res, label=r"$\int \rho \, d" + axis_name + "$")
    plt.gca().set_aspect("equal", "box")
    plt.savefig(os.path.join(sim_folder, f"snap_{axis_name}_{iout:05}.png"))
plt.show()

# %%
# Plots from the saved data

fname = os.path.join(sim_folder, f"cart_t3_{iout}")
data = np.load(fname + ".npz")

plt.figure()
plt.plot(data["R"], data["Q"], label="Q")
plt.plot(data["R"], data["Sigma"], label="Sigma")
plt.plot(data["R"], data["cs_vphi"], label="cs/vphi")
plt.yscale("log")
plt.ylim(1e-3, 1e1)
plt.xlim(0.3, 3)
plt.legend()
plt.xlabel("R")
plt.tight_layout()
plt.savefig(f"{fname}_radial.pdf")

plt.figure()
plt.plot(data["R"], data["alpha_R"], label="alpha_R")
plt.plot(data["R"], data["alpha_G"], label="alpha_G")
plt.plot(data["R"], data["alpha"], label="alpha")
plt.yscale("log")
plt.plot([0.3, 6], [alpha_analytic, alpha_analytic], "k:", label="analytic")
plt.ylim(1e-3, 1e0)
plt.xlim(0.3, 3)
plt.legend()
plt.xlabel("R")
plt.ylabel("alpha")
plt.tight_layout()
plt.savefig(f"{fname}_alpha.pdf")

plt.figure()
plt.pcolormesh(
    data["xf"],
    data["yf"],
    data["Sigma_2d"],
    norm=LogNorm(vmin=1e-4, vmax=1),
    cmap="Blues",
    rasterized=True,
)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.gca().set_aspect("equal", "box")
plt.colorbar(label="Sigma")
plt.savefig(f"{fname}_Sigma.pdf")
plt.show()
