import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


multx = 1
multy = 1
multz = 1
max_amr_lev = 1
cell_size = 2 << max_amr_lev  # refinement is limited to cell_size = 2
base = 64
scale_fact = 0.3 / (cell_size * base * multx)
gamma = 1.4
err_min = 0.30
err_max = 0.10
nx = base * 2
ny = base * 2
sim_folder = "_to_trash/ramses_liska_wendroff_128/bis/"
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)


# Utility for plotting
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


positions = make_cartesian_coords(nx, ny, 0.2, 0, 0.3 - 1e-6, 0, 0.3 - 1e-6)


def plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, case_name, dpi=200):
    ext = metadata["extent"]

    my_cmap = matplotlib.colormaps["viridis"].copy()  # copy the default cmap
    my_cmap.set_bad(color="white")

    arr_rho_pos = np.array(arr_rho_pos).reshape(nx, ny)

    ampl = 1e-5

    X = np.linspace(ext[0], ext[1], nx)
    Y = np.linspace(ext[2], ext[3], ny)
    X, Y = np.meshgrid(X, Y)

    plt.figure(dpi=dpi)
    vmin = 0
    vmax = 1.15
    levels = np.arange(vmin, vmax + 0.025, 0.025)
    res = plt.contourf(X, Y, arr_rho_pos, levels=levels, cmap=my_cmap)
    cs = plt.contour(X, Y, arr_rho_pos, levels=levels, colors="black", linewidths=0.5)

    # plt.figure(dpi=dpi)
    # res = plt.imshow(
    #     arr_rho_pos,
    #     cmap=my_cmap,
    #     origin="lower",
    #     extent=ext,
    #     aspect="auto",
    #     vmin=0.,
    #     vmax=1.1
    # )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [seconds]")
    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\rho$ [code unit]")
    plt.savefig(os.path.join(sim_folder, f"rho_{case_name}_{iplot:04d}.png"))
    plt.close()


from shamrock.utils.plot import show_image_sequence


def run_case(set_bc_func, case_name):

    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(scale_fact)
    set_bc_func(cfg)
    cfg.set_eos_gamma(gamma)
    cfg.set_Csafe(0.5)
    cfg.set_riemann_solver_hllc()
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(True)
    # cfg.set_amr_mode_pseudo_gradient_based(error_min=err_min, error_max=err_max)

    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid(
        (0, 0, 0), (cell_size, cell_size, cell_size), (base * multx, base * multy, base * multz)
    )

    def rho_map(rmin, rmax):

        x, y, z = rmin
        if (x + y) < 0.15:
            return 0.125
        else:
            return 1.0

    etot_L = 1.0 / (gamma - 1)
    etot_R = 0.14 / (gamma - 1)

    def rhoetot_map(rmin, rmax):

        rho = rho_map(rmin, rmax)

        x, y, z = rmin
        if (x + y) < 0.15:
            return etot_R
        else:
            return etot_L

    def rhovel_map(rmin, rmax):
        return (0, 0, 0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    # model.evolve_once(0,0.1)
    fact = 200
    tmax = 0.01 * fact
    all_t = np.linspace(0, tmax, fact)

    def plot(t, iplot):
        metadata = {"extent": [0, 0.3, 0, 0.3], "time": t}
        arr_rho_pos = model.render_slice("rho", "f64", positions)
        plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, case_name)

    current_time = 0.0
    for i, t in enumerate(all_t):
        # model.dump_vtk(os.path.join(sim_folder, f"{case_name}_{i:04d}.vtk"))
        model.evolve_until(t)
        current_time = t
        plot(current_time, i)

    plot(current_time, len(all_t))

    # If the animation is not returned only a static image will be shown in the doc
    ani = show_image_sequence(os.path.join(sim_folder, f"rho_{case_name}_*.png"), render_gif=True)

    if shamrock.sys.world_rank() == 0:
        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(os.path.join(sim_folder, f"rho_{case_name}.gif"), writer=writer)

        return ani
    else:
        return None


def run_case_reflective():
    def set_bc_func(cfg):
        cfg.set_boundary_condition("x", "reflective")
        cfg.set_boundary_condition("y", "reflective")
        cfg.set_boundary_condition("z", "reflective")

    return run_case(set_bc_func, "reflective")


ani_reflective = run_case_reflective()
# plt.show()
