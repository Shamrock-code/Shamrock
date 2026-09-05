"""3D Sedov blast test
=======================

"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from shamrock.utils.plot import show_image_sequence

import shamrock

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# Setup parameters
multx = 1
multy = 1
multz = 1
max_amr_lev = 1
cell_size = 2 << max_amr_lev  # refinement is limited to cell_size = 2
base = 64
gamma = 5.0 / 3.0
scale_fact = 1 / (cell_size * base * multx)
Rstart = 1.0 / (2 * base) + 1e-4


# plot
nx, ny = 512, 512

sim_folder = "_to_trash/ramses_sedov_amr/"


dx = scale_fact
Vcell = dx**3.0

E0 = 10
P0 = 1e-3
L = 1
Ncells = 0


Nx = cell_size * base * multx
Ny = cell_size * base * multy
Nz = cell_size * base * multz

for k in range(Nz):
    z = (k + 0.5) * dx - L / 2.0
    for j in range(Ny):
        y = (j + 0.5) * dx - L / 2.0
        for i in range(Nx):
            x = (i + 0.5) * dx - L / 2.0

            r = np.sqrt(x * x + y * y + z * z)

            if r < Rstart:
                Ncells += 1

print("Number of injection cells =", Ncells)

Vinj = Ncells * Vcell

rhoe_in = E0 / Vinj

Pin = (gamma - 1.0) * rhoe_in

print(f"Injected volume = {Vinj:.6e}")
print(f"Injected pressure = {Pin:.6e}")
print(f"Injected energy density = {rhoe_in:.6e}")

# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)

# %%
# Simulation related function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Utility for plotting, animations, and the simulation itself


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


positions = make_cartesian_coords(nx, ny, L * 0.5, 0, L - 1e-6, 0, L - 1e-6)


def plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, dpi=200):
    ext = metadata["extent"]

    my_cmap = matplotlib.colormaps["rainbow"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    arr_rho_pos = np.array(arr_rho_pos).reshape(nx, ny)

    ampl = 1e-5

    plt.figure(dpi=dpi)
    res = plt.imshow(
        arr_rho_pos,
        cmap=my_cmap,
        origin="lower",
        extent=ext,
        aspect="auto",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [code unit]")
    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\rho$ [code unit]")
    plt.savefig(os.path.join(sim_folder, f"rho_{iplot:04d}.png"))
    plt.close()


def run_simulation(t_final, with_2to_1=True):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(scale_fact)

    cfg.set_eos_gamma(gamma)
    cfg.set_Csafe(0.3)
    cfg.set_boundary_condition("x", "periodic")
    cfg.set_boundary_condition("y", "periodic")
    cfg.set_boundary_condition("z", "periodic")
    cfg.set_riemann_solver_hllc()
    cfg.set_slope_lim_minmod()

    cfg.set_enable_2to1(with_2to_1)

    cfg.set_amr_mode_old(False)
    cfg.set_second_order_interpolation_mode()
    cfg.set_face_time_interpolation(True)

    err_min = 0.30
    err_max = 0.20

    cfg.set_amr_mode_pseudo_gradient_based(error_min=err_min, error_max=err_max)
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e9), 1)
    model.make_base_grid(
        (0, 0, 0), (cell_size, cell_size, cell_size), (base * multx, base * multy, base * multz)
    )

    def rho_map(rmin, rmax):
        return 1.0

    def rhoe_map(rmin, rmax):
        x_min, y_min, z_min = rmin
        x_max, y_max, z_max = rmax

        x = (x_min + x_max) * 0.5 - L / 2.0
        y = (y_min + y_max) * 0.5 - L / 2.0
        z = (z_min + z_max) * 0.5 - L / 2.0
        ## radius from box center
        r = np.sqrt(x * x + y * y + z * z)

        if r < Rstart:
            return rhoe_in
        else:
            return P0 / (gamma - 1.0)

    def rhovel_map(rmin, rmax):
        return (0.0, 0.0, 0.0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    # tmax = 0.03
    tmax = t_final
    fact = 15
    all_t = np.linspace(0, tmax, fact)

    def plot(t, iplot):
        metadata = {"extent": [0, L, 0, L], "time": t}
        arr_rho_pos = model.render_slice("rho", "f64", positions)
        plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot)

    current_time = 0.0
    for i, t in enumerate(all_t):
        model.dump_vtk(os.path.join(sim_folder, f"sedov_blast_{i:04d}.vtk"))
        model.evolve_until(t)
        current_time = t
        plot(current_time, i)

    plot(current_time, len(all_t))

    # If the animation is not returned only a static image will be shown in the doc
    ani = show_image_sequence(os.path.join(sim_folder, "rho_sedov_blast_*.png"), render_gif=True)

    if shamrock.sys.world_rank() == 0:
        # To save the animation using Pillow as a gif
        writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(os.path.join(sim_folder, "rho_sedov_blast.gif"), writer=writer)

        return ani
    else:
        return None


run_simulation(t_final=0.01)
plt.show()


# dt = 0
# t = 0
# freq = 1
# dX0 = []
# for i in range(100000):
#     next_dt = model.evolve_once_override_time(t, dt)

#     t += dt
#     dt = next_dt

#     if i % freq == 0:
#         # model.dump_vtk(f"test{t:.5f}.vtk")
#         model.dump_vtk(f"test{i:05d}.vtk")

#     if tmax < t + next_dt:
#         dt = tmax - t
#     if t == tmax:
#         # model.dump_vtk(f"test{t:.5f}.vtk")
#         model.dump_vtk(f"test{i:05d}.vtk")
#         break
