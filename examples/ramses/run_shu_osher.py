"""
Shu-Osher shock tube in RAMSES solver
=====================================

A Mach 3 shock moving into a sinusoidal density field (Shu & Osher 1989). The
interaction produces a train of high-frequency waves behind the shock, which is
a classic test of how well a scheme resolves both shocks and smooth structures.

The problem is set on :math:`x \\in [0, 2]` with outflow boundaries in :math:`x`
(the transverse directions are periodic):

+---------------------+------------------------------------+-------------+----------+
| Region              | :math:`\\rho`                       | :math:`v_x` | :math:`P`|
+=====================+====================================+=============+==========+
| :math:`x < 0.2`     | 3.857143                           | 2.629369    | 10.33333 |
+---------------------+------------------------------------+-------------+----------+
| :math:`x \\geq 0.2`  | :math:`1 + 0.2 \\sin(5 \\pi x)`       | 0           | 1        |
+---------------------+------------------------------------+-------------+----------+
"""

import os

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
# ^^^^^^^^^^^^^^^^
#
# The resolution is kept low so that the example runs in a reasonable time in the documentation
# build. Increase ``blocks_x`` (e.g. to 512 for 1024 cells along x) for a sharper post-shock
# wave train.

gamma = 1.4

x_s = 0.2  # initial shock position
A0 = 0.2  # amplitude of the density perturbation
fp = 5.0  # frequency of the density perturbation

timestep_list = [0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.44, 0.45, 0.47, 0.50]

# A RAMSES block has 2 cells per side, so each block is ``sz`` grid units wide.
sz = 1 << 1
blocks_x = 160  # number of blocks along x
blocks_yz = 8  # number of blocks along y and z (periodic, the problem is 1D)

nx = 2 * blocks_x  # number of cells along x
scale_fact = 2 / (sz * blocks_x)  # box length along x is 2

rez_plot = nx
positions_plot = [(x, 0, 0) for x in np.linspace(0, 2, rez_plot).tolist()[:-1]]

output_folder = "_to_trash/shu_osher/"
if shamrock.sys.world_rank() == 0:
    os.makedirs(output_folder, exist_ok=True)

# %%
# Setup the solver
# ^^^^^^^^^^^^^^^^

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

cfg = model.gen_default_config()
cfg.set_scale_factor(scale_fact)

cfg.set_eos_gamma(gamma)
cfg.set_Csafe(0.8)
cfg.set_riemann_solver_hllc()
cfg.set_slope_lim_vanleer_sym()
cfg.set_face_time_interpolation(True)
cfg.set_boundary_condition("x", "outflow")
cfg.set_boundary_condition("y", "periodic")
cfg.set_boundary_condition("z", "periodic")

model.set_solver_config(cfg)
model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (blocks_x, blocks_yz, blocks_yz))

# %%
# Initial conditions
# ^^^^^^^^^^^^^^^^^^


def rho_map(rmin, rmax):
    x, y, z = rmin
    if x < x_s:
        return 3.857143
    return 1 + A0 * np.sin(fp * np.pi * x)


def rhovel_map(rmin, rmax):
    x, y, z = rmin
    if x < x_s:
        return (rho_map(rmin, rmax) * 2.629369, 0, 0)
    return (0, 0, 0)


def rhoetot_map(rmin, rmax):
    x, y, z = rmin
    rho = rho_map(rmin, rmax)
    rhov = rhovel_map(rmin, rmax)
    rhoe_kin = 0.5 * (rhov[0] ** 2) / rho
    P = 10.33333 if x < x_s else 1.0
    return P / (gamma - 1.0) + rhoe_kin


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

# %%
# Run the simulation
# ^^^^^^^^^^^^^^^^^^


def analysis():
    rho_vals = np.array(model.render_slice("rho", "f64", positions_plot))
    rhov_vals = np.array(model.render_slice("rhovel", "f64_3", positions_plot))
    rhoetot_vals = np.array(model.render_slice("rhoetot", "f64", positions_plot))

    vx = rhov_vals[:, 0] / rho_vals
    P = (rhoetot_vals - 0.5 * rho_vals * vx**2) * (gamma - 1)
    e_int = P / ((gamma - 1.0) * rho_vals)
    return {"rho": rho_vals, "vx": vx, "P": P, "e_int": e_int}


results = []
for t in timestep_list:
    model.evolve_until(t)
    results.append(analysis())

# %%
# Plot the results
# ^^^^^^^^^^^^^^^^
# One frame per output time, with the same axis limits for every frame so that they can be
# assembled into an animation.

arr_x = [x[0] for x in positions_plot]

fields = [
    ("rho", "$\\rho$"),
    ("vx", "$v_\\mathrm{x}$"),
    ("e_int", "$\\mathrm{e}_\\mathrm{int}$"),
    ("P", "$\\mathrm{P}$"),
]


def get_ylim(key):
    vmin = min(np.min(frame[key]) for frame in results)
    vmax = max(np.max(frame[key]) for frame in results)
    margin = 0.05 * (vmax - vmin)
    return vmin - margin, vmax + margin


ylims = {key: get_ylim(key) for key, _ in fields}


def plot_frame(frame, t, iplot):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), dpi=100, constrained_layout=True)
    fig.suptitle(f"Shu-Osher test, $t = {t:.2f}$, $n_x = {nx}$")
    for ax, (key, label) in zip(axs.flat, fields):
        ax.set_xlabel("$x$")
        ax.set_ylabel(label)
        ax.set_ylim(*ylims[key])
        ax.grid(True, alpha=0.3)
        ax.scatter(arr_x, frame[key], s=4)
    plt.savefig(os.path.join(output_folder, f"frame_{iplot:04d}.png"))
    plt.close(fig)


if shamrock.sys.world_rank() == 0:
    for iplot, (t, frame) in enumerate(zip(timestep_list, results)):
        plot_frame(frame, t, iplot)

# %%
# Animation
# ^^^^^^^^^

from shamrock.utils.plot import show_image_sequence

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(os.path.join(output_folder, "frame_*.png"), interval=400)

if shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=3, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(os.path.join(output_folder, "shu_osher.gif"), writer=writer)

plt.show()

# %%
# Density evolution
# ^^^^^^^^^^^^^^^^^

cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(figsize=(8, 5), dpi=100, constrained_layout=True)
for i, (t, frame) in enumerate(zip(timestep_list, results)):
    ax.plot(arr_x, frame["rho"], color=cmap(i / (len(timestep_list) - 1)), label=f"t={t:.2f}")
ax.set_xlabel("$x$")
ax.set_ylabel("$\\rho$")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=8)
plt.savefig(os.path.join(output_folder, f"shu_osher_rho_evolution_nx_{nx}.png"))

plt.show()
