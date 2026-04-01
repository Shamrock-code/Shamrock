import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

for c_num in list(range(1)):
    multx = 1
    multy = 1
    multz = 1
    max_amr_lev = 1
    cell_size = 2 << max_amr_lev  # refinement is limited to cell_size = 2
    base = 64
    scale_fact = 1 / (cell_size * base * multx)
    gamma = 1.4
    err_min = 0.30
    err_max = 0.10
    nx = base * 2
    ny = base * 2

    rho_array = np.array(
        [
            [0.1072, 0.2579, 0.5197, 1.0000],
            [1.0000, 0.5197, 0.5197, 1.0000],
            [0.1380, 0.5323, 0.5323, 1.5000],
            [1.1000, 0.5065, 0.5065, 1.1000],
            [1.0000, 3.0000, 2.0000, 1.0000],
            [1.0000, 3.0000, 2.0000, 1.0000],
            [0.8000, 0.5197, 0.5197, 1.0000],
            [0.8000, 1.0000, 1.0000, 0.5197],
            [1.0390, 0.5197, 2.0000, 1.0000],
            [0.2281, 0.4562, 0.5000, 1.0000],
            [0.8000, 0.5313, 0.5313, 1.0000],
            [0.8000, 1.0000, 1.0000, 0.5313],
            [1.0625, 0.5313, 2.0000, 1.0000],
            [0.4736, 0.9474, 1.0000, 2.0000],
            [0.8000, 0.5313, 0.5197, 1.0000],
            [0.8000, 1.0000, 1.0222, 0.5313],
            [1.0625, 0.5197, 2.0000, 1.0000],
            [1.0625, 0.5197, 2.0000, 1.0000],
            [1.0625, 0.5197, 2.0000, 1.0000],
        ]
    )

    pressure_array = np.array(
        [
            [0.0439, 0.1500, 0.4000, 1.0000],
            [1.0000, 0.4000, 0.4000, 1.0000],
            [0.0290, 0.3000, 0.3000, 1.5000],
            [1.1000, 0.3500, 0.3500, 1.1000],
            [1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000],
            [0.4000, 0.4000, 0.4000, 1.0000],
            [1.0000, 1.0000, 1.0000, 0.4000],
            [0.4000, 0.4000, 1.0000, 1.0000],
            [0.3333, 0.3333, 1.0000, 1.0000],
            [0.4000, 0.4000, 0.4000, 1.0000],
            [1.0000, 1.0000, 1.0000, 0.4000],
            [0.4000, 0.4000, 1.0000, 1.0000],
            [2.6667, 2.6667, 8.0000, 8.0000],
            [0.4000, 0.4000, 0.4000, 1.0000],
            [1.0000, 1.0000, 1.0000, 0.4000],
            [0.4000, 0.4000, 1.0000, 1.0000],
            [0.4000, 0.4000, 1.0000, 1.0000],
            [0.4000, 0.4000, 1.0000, 1.0000],
        ]
    )

    ux_array = np.array(
        [
            [-0.7259, 0.0000, -0.7259, 0.0000],
            [-0.7259, 0.0000, -0.7259, 0.0000],
            [1.2060, 0.0000, 1.2060, 0.0000],
            [0.8939, 0.0000, 0.8939, 0.0000],
            [0.7500, 0.7500, -0.7500, -0.7500],
            [-0.7500, -0.7500, 0.7500, 0.7500],
            [0.1000, 0.1000, -0.6259, 0.1000],
            [0.1000, 0.1000, -0.6259, 0.1000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.1000, 0.1000, 0.8276, 0.1000],
            [0.0000, 0.0000, 0.7276, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.1000, 0.1000, -0.6259, 0.1000],
            [0.1000, 0.1000, -0.6179, 0.1000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
        ]
    )

    uy_array = np.array(
        [
            [-1.4045, -1.4045, 0.0000, 0.0000],
            [-0.7259, -0.7259, 0.0000, 0.0000],
            [1.2060, 1.2060, 0.0000, 0.0000],
            [0.8939, 0.8939, 0.0000, 0.0000],
            [0.5000, -0.5000, 0.5000, -0.5000],
            [0.5000, -0.5000, 0.5000, -0.5000],
            [0.1000, -0.6259, 0.1000, 0.1000],
            [0.1000, -0.6259, 0.1000, 0.1000],
            [-0.8133, -0.4259, -0.3000, 0.3000],
            [-0.6076, -0.4297, 0.6076, 0.4297],
            [0.0000, 0.7276, 0.0000, 0.0000],
            [0.0000, 0.7276, 0.0000, 0.0000],
            [0.8145, 0.4276, 0.3000, -0.3000],
            [1.2172, 1.1606, -1.2172, -0.5606],
            [-0.3000, 0.4276, -0.3000, -0.3000],
            [0.1000, 0.8276, 0.1000, 0.1000],
            [0.2145, -1.1259, -0.3000, -0.4000],
            [0.2145, 0.2741, -0.3000, 1.0000],
            [0.2145, -0.4259, -0.3000, 0.3000],
        ]
    )

    x_0 = 0.5
    y_0 = 0.5

    sim_folder = (
        "_to_trash/ramses_riemann_2d_van_leer_"
        + "grid_reso_"
        + str(base * 2)
        + "_config_"
        + str(c_num)
        + "/"
    )
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

    positions = make_cartesian_coords(nx, ny, 0.2, 0, 1.0 - 1e-6, 0, 1.0 - 1e-6)

    def plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, case_name, dpi=200):
        ext = metadata["extent"]

        my_cmap = matplotlib.colormaps["gist_yarg"].copy()  # copy the default cmap
        my_cmap.set_bad(color="black")

        arr_rho_pos = np.array(arr_rho_pos).reshape(nx, ny)

        ampl = 1e-5

        X = np.linspace(ext[0], ext[1], nx)
        Y = np.linspace(ext[2], ext[3], ny)
        X, Y = np.meshgrid(X, Y)

        plt.figure(dpi=dpi)

        res = plt.contourf(X, Y, arr_rho_pos, levels=50, cmap=my_cmap)
        cs = plt.contour(X, Y, arr_rho_pos, levels=50, colors="black", linewidths=0.5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"config_{c_num} , t = {metadata['time']:0.3f} [seconds]")
        cbar = plt.colorbar(res, extend="both")
        cbar.set_label(r"$\rho$ [code unit]")
        plt.savefig(os.path.join(sim_folder, f"rho_{case_name}_{iplot:04d}.png"))
        plt.close()

    from shamrock.utils.plot import show_image_sequence

    def run_case(set_bc_func, case_name, case_number=0):

        ctx = shamrock.Context()
        ctx.pdata_layout_new()

        model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

        cfg = model.gen_default_config()
        cfg.set_scale_factor(scale_fact)
        set_bc_func(cfg)
        cfg.set_eos_gamma(gamma)
        cfg.set_Csafe(0.3)
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
            quadrant = 0
            x, y, z = rmin
            if y < y_0:
                if x < x_0:
                    quadrant = 0
                else:
                    quadrant = 1
            else:
                if x < x_0:
                    quadrant = 2
                else:
                    quadrant = 3

            return rho_array[case_number][quadrant]

        def rhovel_map(rmin, rmax):
            rho = rho_map(rmin, rmax)
            quadrant = 0
            x, y, z = rmin
            if y < y_0:
                if x < x_0:
                    quadrant = 0
                else:
                    quadrant = 1
            else:
                if x < x_0:
                    quadrant = 2
                else:
                    quadrant = 3

            vel_x = ux_array[case_number][quadrant]
            vel_y = uy_array[case_number][quadrant]

            return (rho * vel_x, rho * vel_y, 0)

        def rhoetot_map(rmin, rmax):
            rho = rho_map(rmin, rmax)
            quadrant = 0
            x, y, z = rmin
            if y < y_0:
                if x < x_0:
                    quadrant = 0
                else:
                    quadrant = 1
            else:
                if x < x_0:
                    quadrant = 2
                else:
                    quadrant = 3

            vel_x = ux_array[case_number][quadrant]
            vel_y = uy_array[case_number][quadrant]
            press = pressure_array[case_number][quadrant]
            Ek = 0.5 * rho * (vel_x * vel_x + vel_y * vel_y)
            Ein = press * (1.0 / (gamma - 1.0))
            return Ek + Ein

        model.set_field_value_lambda_f64("rho", rho_map)
        model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
        model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

        # model.evolve_once(0,0.1)
        fact = 25
        tmax = 0.0127 * fact
        all_t = np.linspace(0, tmax, fact)

        def plot(t, iplot):
            metadata = {"extent": [0, 1.0, 0, 1.0], "time": t}
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
        ani = show_image_sequence(
            os.path.join(sim_folder, f"rho_{case_name}_*.png"), render_gif=True
        )

        if shamrock.sys.world_rank() == 0:
            # To save the animation using Pillow as a gif
            writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
            ani.save(os.path.join(sim_folder, f"rho_{case_name}.gif"), writer=writer)

            return ani
        else:
            return None

    def run_case_outflow():

        def set_bc_func(
            cfg,
        ):
            cfg.set_boundary_condition("x", "outflow")
            cfg.set_boundary_condition("y", "outflow")
            cfg.set_boundary_condition("z", "outflow")

        return run_case(set_bc_func, "outflow", c_num)

    ani_outflow = run_case_outflow()
    # plt.show()
