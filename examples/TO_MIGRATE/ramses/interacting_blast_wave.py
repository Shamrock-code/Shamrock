import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


def run_1d_interacting_blast_wave_test(
    nblocks,
    nxbox,
    nybox,
):
    timestamps = 100
    gamma = 1.4

    output_folder = "_to_trash/interacting_blast_wave/"
    os.makedirs(output_folder, exist_ok=True)

    multx = nxbox
    multy = nybox
    multz = nybox
    sz = 1 << 1
    base = nblocks
    scale_fact = 1 / (sz * base * multx)

    rez_plot = 256
    positions_plot = [(x, 0, 0) for x in np.linspace(0, 1, rez_plot).tolist()[:-1]]

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
    cfg.set_boundary_condition("x", "reflective")
    cfg.set_boundary_condition("y", "periodic")
    cfg.set_boundary_condition("z", "periodic")

    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    x0_l = 0.1
    x0_r = 0.9
    p0_l = 1000
    p0_m = 0.01
    p0_r = 100
    tmax = 0.02

    def rho_map(rmin, rmax):
        return 1

    def rhovel_map(rmin, rmax):
        return (0, 0, 0)

    def rhoetot_map(rmin, rmax):
        x, y, z = rmin
        if x < x0_l:
            return p0_l / (gamma - 1.0)
        elif x < x0_r:
            return p0_m / (gamma - 1.0)
        else:
            return p0_r / (gamma - 1.0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    dt_evolve = tmax / timestamps

    def analysis():
        results = []
        for i in range(timestamps + 1):
            model.evolve_until(dt_evolve * i)
            rho_vals = model.render_slice("rho", "f64", positions_plot)
            results.append(rho_vals)
        return results

    def plot_results(data, tmax):
        arr_x = [x[0] for x in positions_plot]
        for i, frame in enumerate(data):
            fig, axs = plt.subplots(1, 1, figsize=(8, 8))
            fig.suptitle(f"Interacting blast wave test at t = {dt_evolve * i}", fontsize=14)
            axs.set_xlabel("$x$")
            axs.set_ylabel("$\\rho$")
            axs.grid(True, alpha=0.3)
            axs.plot(arr_x, frame, label=f"{base * 2}", linewidth=1)
            axs.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"Interacting_blast_wave_test_{i:04d}.png"))
            plt.close(fig)

    def gif_results(data, tmax, case_anim="inter-blast"):

        arr_x = [x[0] for x in positions_plot]

        import matplotlib.animation as animation

        fig2, axs = plt.subplots(1, 1, figsize=(8, 8))
        fig2.suptitle(f"{case_anim} - t = {0.0:.3f} s", fontsize=14)

        # Calculate global min/max across all frames for fixed y-axis limits
        rho_min = min(np.min(frame) for frame in data)
        rho_max = max(np.max(frame) for frame in data)
        # Add 5% margin to y-axis limits
        rho_margin = (rho_max - rho_min) * 0.05
        # Configure each axis
        axs.set_xlabel("$x$")
        axs.set_ylabel("$\\rho$")
        axs.set_ylim(rho_min - rho_margin, rho_max + rho_margin)
        axs.grid(True, alpha=0.3)
        (line_rho,) = axs.plot(arr_x, data[0], label="$\\rho$", linewidth=2, color="C0")
        axs.legend()

        def animate(frame):
            t = tmax * frame / timestamps
            line_rho.set_ydata(data[frame])

            fig2.suptitle(f"{case_anim} - t = {t:.3f} s", fontsize=14)
            return (line_rho, 0)

        anim = animation.FuncAnimation(
            fig2, animate, frames=timestamps + 1, interval=50, blit=False, repeat=True
        )
        plt.tight_layout()
        writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
        anim.save(os.path.join(output_folder, "Interacting_blast_wave_test.gif"), writer=writer)
        return anim

    def run_and_plot():

        data = analysis()

        return plot_results(data, tmax), gif_results(data, tmax)

    plot, anim = run_and_plot()


run_1d_interacting_blast_wave_test(16, 2, 1)
