"""
Boundary conditions for linear wave propagation
=======================================================

"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Setup parameters
nx, ny = 512, 512

sim_folder = "_to_trash/ramses_linear_wave_with_bc/"

# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)


# %%
# Simulation related function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Utility for plotting, animations, and the simulation itself


def make_cartesian_coords(nx, ny, z_val, min_x, max_x, min_y, max_y):
    """
    Generate a list of positions in cylindrical coordinates (r, theta)
    spanning [0, ext*2] x [-pi, pi] for use with the rendering module.

    Returns:
        list: List of [x, y, z] coordinate lists
    """

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


positions = make_cartesian_coords(nx, ny, 0.5, 0, 1 - 1e-6, 0, 1 - 1e-6)


def plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, case_name, dpi=200):
    ext = metadata["extent"]

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    arr_rho_pos = np.array(arr_rho_pos).reshape(nx, ny)

    ampl = 1e-5

    plt.figure(dpi=dpi)
    res = plt.imshow(
        arr_rho_pos,
        cmap=my_cmap,
        origin="lower",
        extent=ext,
        vmin=1 - ampl,
        vmax=1 + ampl,
        aspect="auto",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [seconds]")
    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\rho$ [code unit]")
    plt.savefig(os.path.join(sim_folder, f"rho_{case_name}_{iplot:04d}.png"))
    plt.close()


def show_image_sequence(glob_str, render_gif):

    if render_gif and shamrock.sys.world_rank() == 0:

        import glob

        files = sorted(glob.glob(glob_str))

        from PIL import Image

        image_array = []
        for my_file in files:
            image = Image.open(my_file)
            image_array.append(image)

        if not image_array:
            raise RuntimeError(f"Warning: No images found for glob pattern: {glob_str}")

        pixel_x, pixel_y = image_array[0].size

        # Create the figure and axes objects
        # Remove axes, ticks, and frame & set aspect ratio
        dpi = 200
        fig = plt.figure(dpi=dpi)
        plt.gca().set_position((0, 0, 1, 1))
        plt.gcf().set_size_inches(pixel_x / dpi, pixel_y / dpi)
        plt.axis("off")

        # Set the initial image with correct aspect ratio
        im = plt.imshow(image_array[0], animated=True, aspect="auto")

        def update(i):
            im.set_array(image_array[i])
            return (im,)

        # Create the animation object
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(image_array),
            interval=50,
            blit=True,
            repeat_delay=10,
        )

        return ani


def run_case(set_bc_func, case_name):

    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 32

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    set_bc_func(cfg)
    cfg.set_eos_gamma(5.0 / 3.0)
    model.set_solver_config(cfg)

    name = f"test_{case_name}"

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    gamma = 5.0 / 3.0

    u_cs1 = 1 / (gamma * (gamma - 1))

    kx, ky, kz = 4 * np.pi, 0, 0
    delta_rho = 0
    delta_v = 1e-4

    def rho_map(rmin, rmax):

        x, y, z = rmin

        return 1.0
        # return 1.0 + delta_rho * np.cos(kx * x + ky * y + kz * z)

    def rhoetot_map(rmin, rmax):

        rho = rho_map(rmin, rmax)

        x, y, z = rmin
        rsq = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2
        # return x
        # return (u_cs1 + u_cs1 * delta_rho * np.cos(kx * x + ky * y + kz * z)) * rho
        # return u_cs1 + 0.001 * np.exp(-rsq / 0.01)
        return u_cs1

    def rhovel_map(rmin, rmax):

        rho = rho_map(rmin, rmax)

        x, y, z = rmin
        rsq = (x - 0.2) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2
        return (0 + delta_v * np.exp(-rsq / 0.01) * rho, 0, 0)
        # return (0, 0, 0)  # eturn (0 + delta_v * np.cos(kx * x + ky * y + kz * z) * rho, 0, 0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    # model.evolve_once(0,0.1)
    tmax = 0.127 * 5
    all_t = np.linspace(0, tmax, 20)

    def plot(t, iplot):
        metadata = {"extent": [0, 1, 0, 1], "time": t}
        arr_rho_pos = model.render_slice("rho", "f64", positions)
        plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, case_name)

    current_time = 0.0
    for i, t in enumerate(all_t):
        plot(current_time, i)
        model.evolve_until(t)
        current_time = t

    plot(current_time, len(all_t))

    # If the animation is not returned only a static image will be shown in the doc
    ani = show_image_sequence(os.path.join(sim_folder, f"rho_{case_name}_*.png"), True)

    if shamrock.sys.world_rank() == 0:
        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(os.path.join(sim_folder, f"rho_{case_name}.gif"), writer=writer)

        return ani
    else:
        return None


# %%
# Periodic boundary conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def run_case_periodic():
    def set_bc_func(cfg):
        cfg.set_boundary_condition("x", "periodic")
        cfg.set_boundary_condition("y", "periodic")
        cfg.set_boundary_condition("z", "periodic")

    return run_case(set_bc_func, "periodic")


ani_periodic = run_case_periodic()
plt.show()


# %%
# Reflective boundary conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def run_case_reflective():
    def set_bc_func(cfg):
        cfg.set_boundary_condition("x", "reflective")
        cfg.set_boundary_condition("y", "reflective")
        cfg.set_boundary_condition("z", "reflective")

    return run_case(set_bc_func, "reflective")


ani_reflective = run_case_reflective()
plt.show()

# %%
# Outflow boundary conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^


def run_case_outflow():
    def set_bc_func(cfg):
        cfg.set_boundary_condition("x", "outflow")
        cfg.set_boundary_condition("y", "outflow")
        cfg.set_boundary_condition("z", "outflow")

    return run_case(set_bc_func, "outflow")


ani_outflow = run_case_outflow()
plt.show()


# %%
# Outflow on y/z, reflective on x boundary conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def run_case_outflow_reflective_x():
    def set_bc_func(cfg):
        cfg.set_boundary_condition("x", "reflective")
        cfg.set_boundary_condition("y", "outflow")
        cfg.set_boundary_condition("z", "outflow")

    return run_case(set_bc_func, "outflow_reflective_x")


ani_outflow_reflective_x = run_case_outflow_reflective_x()
plt.show()
