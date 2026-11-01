"""
Rayleigh-Taylor instability in RAMSES solver
============================================

A heavy fluid (:math:`\\rho = 2`) sits on top of a light one (:math:`\\rho = 1`) in a constant
gravitational field :math:`\\mathbf{g} = -g \\, \\mathbf{e}_y`, applied as an external force
(``cfg.add_ext_force_constant_gravity``). The initial state is in hydrostatic equilibrium, and a
small velocity perturbation seeds the instability at the interface.

Adapted from the Rayleigh-Taylor setup of the PhD work of Léodasce Sewanou.
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

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Setup parameters

# Physical parameters
g = 0.1  # gravitational acceleration (along -y)
gamma = 1.4

rho_light = 1.0  # bottom fluid
rho_heavy = 2.0  # top fluid

# resolution (the z direction has as many blocks as the x one, to keep the patches compact)
sz = 1 << 1
base = 16

# dimension of the box in base units
multx = 1
multy = 1  # use 3 if you want well developed instability
multz = 1

Lx, Ly = 0.5 * multx, 0.5 * multy  # box size
y_interface = Ly / 2
P0 = 2.5  # pressure at the interface

v_pert = 1e-2  # amplitude of the velocity perturbation

t_end = 10.0
n_frames = 51

sim_folder = "_to_trash/ramses_rayleigh_taylor/"

# %%
# Deduced quantities

# grid int coords to [0, Lx] x [0, Ly] x [0, Lz]
scale_fact = Lx / (sz * base * multx)

nx = sz * base * multx
ny = sz * base * multy
Lz = sz * base * multz * scale_fact

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
cfg.set_eos_gamma(gamma)
cfg.set_riemann_solver_hllc()
cfg.set_Csafe(0.8)
cfg.set_slope_lim_vanleer_sym()
cfg.set_face_time_interpolation(True)
cfg.add_ext_force_constant_gravity(g=(0.0, -g, 0.0))
cfg.set_boundary_condition("x", "periodic")
cfg.set_boundary_condition("y", "reflective")
cfg.set_boundary_condition("z", "reflective")
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

# %%
# Initial conditions
# ^^^^^^^^^^^^^^^^^^
# Hydrostatic equilibrium :math:`P = P_0 - \rho g (y - y_{\rm interface})` (continuous at the
# interface), with a velocity perturbation along y localized around the interface.
# The initial conditions are evaluated at the cell centers so that they are mirror symmetric with
# respect to :math:`x = L_x / 2`.


def cell_center(rmin, rmax):
    return (np.array(rmin) + np.array(rmax)) / 2


def rho_map(rmin, rmax):
    x, y, z = cell_center(rmin, rmax)
    if y <= y_interface:
        return rho_light
    else:
        return rho_heavy


def vy_map(rmin, rmax):
    x, y, z = cell_center(rmin, rmax)
    xloc = x - Lx / 2
    yloc = y - y_interface
    return (
        v_pert
        * (1.0 / 4.0)
        * (1.0 + np.cos(4.0 * np.pi * xloc))
        * (1.0 + np.cos(3.0 * np.pi * yloc))
    )


def rhovel_map(rmin, rmax):
    rho = rho_map(rmin, rmax)
    return (0.0, rho * vy_map(rmin, rmax), 0.0)


def rhoetot_map(rmin, rmax):
    rho = rho_map(rmin, rmax)
    x, y, z = cell_center(rmin, rmax)
    P = P0 - g * rho * (y - y_interface)
    vy = vy_map(rmin, rmax)
    return 0.5 * rho * vy**2 + P / (gamma - 1.0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

# %%
# Plotting utilities
# ^^^^^^^^^^^^^^^^^^


def make_cartesian_coords(nx, ny, z_val, min_x, max_x, min_y, max_y):
    x_vals = np.linspace(min_x, max_x, nx)
    y_vals = np.linspace(min_y, max_y, ny)

    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")
    z_grid = z_val * np.ones_like(x_grid)

    positions = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    return [tuple(pos) for pos in positions]


# the x sampling is symmetric with respect to x = Lx / 2 (used for the mirror symmetry diagnostic)
positions = make_cartesian_coords(nx, ny, Lz / 2, 1e-6, Lx - 1e-6, 0, Ly - 1e-6)


def render_rho():
    return np.array(model.render_slice("rho", "f64", positions)).reshape(nx, ny)


def plot_rho(arr_rho, t, iplot, dpi=150):
    plt.figure(dpi=dpi)
    res = plt.imshow(
        arr_rho.T,
        cmap=matplotlib.colormaps["rainbow"],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        vmin=rho_light,
        vmax=rho_heavy,
        interpolation="bicubic",
        aspect="equal",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {t:0.3f}")
    cbar = plt.colorbar(res)
    cbar.set_label(r"$\rho$ [code unit]")
    plt.tight_layout()
    plt.savefig(os.path.join(sim_folder, f"rho_{iplot:04d}.png"))
    plt.close()


# %%
# Diagnostics
# ^^^^^^^^^^^
# At each output we track:
#
# * the mirror symmetry with respect to :math:`x = L_x / 2` (the setup is symmetric), through the
#   relative deviation :math:`2 |\rho_{\rm left} - \rho_{\rm right}| / (|\rho_{\rm left}| +
#   |\rho_{\rm right}|)` between each point of the rendered density slice and its mirror,
# * the energy budget: kinetic, internal and gravitational potential (:math:`\rho g y`) energies,
#   whose sum should be conserved.


def get_cell_data():
    dic = ctx.collect_data()

    # cell centers, from the block bounds (cells are ordered block by block, x fastest)
    block_min = np.array(dic["cell_min"], dtype=float)
    block_max = np.array(dic["cell_max"], dtype=float)
    lid = np.arange(sz**3)
    local_coord = np.stack([lid % sz, (lid // sz) % sz, lid // (sz * sz)], axis=1)
    cell_size = (block_max - block_min) / sz
    pos = block_min[:, None, :] + (local_coord[None, :, :] + 0.5) * cell_size[:, None, :]
    pos = pos.reshape(-1, 3) * scale_fact
    vol = np.repeat(np.prod(cell_size * scale_fact, axis=1), sz**3)

    return (
        pos,
        vol,
        np.array(dic["rho"]),
        np.array(dic["rhovel"]).reshape(-1, 3),
        np.array(dic["rhoetot"]),
    )


hist_t = []
hist_sym_max = []
hist_ekin = []
hist_eint = []
hist_epot = []


def analysis(arr_rho, t):
    # mirror symmetry, on the rendered density slice
    rho_mirror = arr_rho[::-1, :]
    rel_dev = 2 * np.abs(arr_rho - rho_mirror) / (np.abs(arr_rho) + np.abs(rho_mirror))

    # energy budget
    pos, vol, rho, rhovel, rhoetot = get_cell_data()

    ekin = 0.5 * np.sum(rhovel**2, axis=1) / rho
    eint = rhoetot - ekin
    epot = rho * g * pos[:, 1]

    hist_t.append(t)
    hist_sym_max.append(np.max(rel_dev))
    hist_ekin.append(np.sum(ekin * vol))
    hist_eint.append(np.sum(eint * vol))
    hist_epot.append(np.sum(epot * vol))


# %%
# Run the simulation
# ^^^^^^^^^^^^^^^^^^

for i, t in enumerate(np.linspace(0, t_end, n_frames)):
    model.evolve_until(t)
    arr_rho = render_rho()
    plot_rho(arr_rho, t, i)
    analysis(arr_rho, t)

# %%
# Density evolution
# ^^^^^^^^^^^^^^^^^

from shamrock.utils.plot import show_image_sequence

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(os.path.join(sim_folder, "rho_*.png"), render_gif=True)

if shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(os.path.join(sim_folder, "rho_rayleigh_taylor.gif"), writer=writer)

plt.show()

# %%
# Mirror symmetry
# ^^^^^^^^^^^^^^^
# Maximum over the rendered density slice of the relative deviation between each point and its
# mirror with respect to :math:`x = L_x / 2`.

plt.figure()
plt.plot(hist_t, hist_sym_max)
plt.xlabel("t")
plt.ylabel(
    r"$\max \, 2 |\rho_{\rm left} - \rho_{\rm right}| / (|\rho_{\rm left}| + |\rho_{\rm right}|)$"
)
plt.title("Mirror symmetry")
plt.tight_layout()
plt.show()

# %%
# Energy budget
# ^^^^^^^^^^^^^

hist_ekin = np.array(hist_ekin)
hist_eint = np.array(hist_eint)
hist_epot = np.array(hist_epot)
hist_etot = hist_ekin + hist_eint + hist_epot

energies = {
    r"$E_{\rm kin}$": (hist_ekin, {}),
    r"$E_{\rm int}$": (hist_eint, {}),
    r"$E_{\rm pot}$": (hist_epot, {}),
    r"$E_{\rm tot}$": (hist_etot, {"color": "red", "linestyle": "--"}),
}

plt.figure()
for label, (e, style) in energies.items():
    plt.plot(hist_t, e, label=label, **style)
plt.yscale("log")
plt.xlabel("t")
plt.ylabel("E [code unit]")
plt.title("Energy budget")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Energy variation with respect to the initial state

plt.figure()
for label, (e, style) in energies.items():
    plt.plot(hist_t, e - e[0], label=label, **style)
plt.yscale("symlog", linthresh=1e-6)
plt.xlabel("t")
plt.ylabel(r"$E(t) - E(0)$ [code unit]")
plt.title("Energy variation")
plt.legend()
plt.tight_layout()
plt.show()
