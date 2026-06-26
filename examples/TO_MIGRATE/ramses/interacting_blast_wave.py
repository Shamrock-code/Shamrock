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
    timestamps = 20
    timestep_lists = [0.010, 0.016, 0.026, 0.028, 0.030, 0.032, 0.034, 0.036, 0.038, 0.040, 0.042]
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
    cfg.set_boundary_condition("y", "reflective")
    cfg.set_boundary_condition("z", "reflective")

    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    x0_l = 0.1
    x0_r = 0.9
    p0_l = 1000
    p0_m = 0.01
    p0_r = 100


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



    def analysis():
        results = []
        for dt in timestep_lists:
            model.evolve_until(dt)
            rho_vals  = model.render_slice("rho", "f64", positions_plot)
            rhov_vals = model.render_slice("rhovel", "f64_3", positions_plot)
            rhoetot_vals = model.render_slice("rhoetot", "f64", positions_plot)
            vx = np.array(rhov_vals)[:, 0] / np.array(rho_vals)
            P = (np.array(rhoetot_vals) - 0.5 * np.array(rho_vals) * vx**2) * (gamma - 1)
            e_int = P/((gamma - 1.) * np.array(rho_vals))
            results_dic = {
            "rho": np.array(rho_vals),
            "vx": np.array(vx),
            "P": np.array(P),
            "e_int":np.array(e_int)
             }
            results.append(results_dic)
       
            output = np.column_stack((np.array(rho_vals), np.array(vx), np.array(P), np.array(e_int)))
            filename= f"data_interacting_blast_wave_reso_{2*base*multx}_at_{dt}.txt"
            np.savetxt(os.path.join(output_folder,filename),
                       output,
                       fmt=["%.10f", "%.10f", "%.10f", "%.10f"],
                       header="rho vx P e_int",
                       )
        return results

    def plot_results(data):
        arr_x = [x[0] for x in positions_plot]
        for i, frame in enumerate(data):
            fig, axs = plt.subplots(2, 2, figsize=(8, 8), dpi=100, constrained_layout=True)
         
            ## density         
            axs[0,0].set_xlabel("$x$")
            axs[0,0].set_ylabel("$\\rho$")
            axs[0,0].grid(True, alpha=0.3)
            axs[0,0].scatter(arr_x, frame["rho"], label=f"{2*base*multx}", s=10, marker = "*")
            axs[0,0].legend(loc=0)

            ## velocity
            axs[0,1].set_xlabel("$x$")
            axs[0,1].set_ylabel("$v_\\mathrm{x}$")
            axs[0,1].grid(True, alpha=0.3)
            axs[0,1].scatter(arr_x, frame["vx"], label=f"{2*base*multx}", s=10, marker ="*" )
            axs[0,1].legend(loc=0)
            
            
            ## pressure
            axs[1,1].set_xlabel("$x$")
            axs[1,1].set_ylabel("$\\mathrm{P}$")
            axs[1,1].grid(True, alpha=0.3)
            axs[1,1].scatter(arr_x, frame["P"], label=f"{2*base*multx}", s=10, marker ="*" )
            axs[1,1].legend(loc=0)

            ## internal energy
            axs[1,0].set_xlabel("$x$")
            axs[1,0].set_ylabel("$\\mathrm{e}_\\mathrm{int}$")
            axs[1,0].grid(True, alpha=0.3)
            axs[1,0].scatter(arr_x, frame["e_int"], label=f"{2*base*multx}", s=10, marker ="*" )
            axs[1,0].legend(loc=0)
        
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"Interacting_blast_wave_test_at_{timestep_lists[i]}_nx_reso_{base*2*multx}.png"))
            plt.close(fig)

    # def gif_results(data, tmax, case_anim="inter-blast"):

    #     arr_x = [x[0] for x in positions_plot]

    #     import matplotlib.animation as animation

    #     fig2, axs = plt.subplots(2, 2, figsize=(8, 8))
    #     fig2.suptitle(f"{case_anim} - t = {0.0:.3f} s", fontsize=14)
    #     ax_rho, ax_vx, ax_P = axs

    #     # Calculate global min/max across all frames for fixed y-axis limits
    #     rho_min = min(np.min(frame["rho"]) for frame in data)
    #     rho_max = max(np.max(frame["rho"]) for frame in data)
    #     vx_min = min(np.min(frame["vx"]) for frame in data)
    #     vx_max = max(np.max(frame["vx"]) for frame in data)
    #     P_min = min(np.min(frame["P"]) for frame in data)
    #     P_max = max(np.max(frame["P"]) for frame in data)
    #     eint_min = min(np.min(frame["e_int"]) for frame in data)
    #     eint_max = max(np.max(frame["e_int"]) for frame in data)

    #     # Add 5% margin to y-axis limits
    #     rho_margin = (rho_max - rho_min) * 0.05
    #     vx_margin = (vx_max - vx_min) * 0.05
    #     P_margin = (P_max -P_min) * 0.05
    #     eint_margin = (eint_max - eint_min) * 0.05

    #     # Configure each axis
    #     axs[0,0].set_xlabel("$x$")
    #     axs[0,0].set_ylabel("$\\rho$")
    #     axs[0,0].set_ylim(rho_min - rho_margin, rho_max + rho_margin)
    #     axs[0,0].grid(True, alpha=0.3)

    #     axs[0,1].set_xlabel("$x$")
    #     axs[0,1].set_ylabel("$v_\\mathrm{x}$")
    #     axs[0,1].set_ylim(vx_min - vx_margin, vx_max + vx_margin)
    #     axs[0,1].grid(True, alpha=0.3)


    #     (line_rho,) = axs[0,0].plot(arr_x, data[0], label="$\\rho$", linewidth=2, color="C0")
    #     axs[0,0].legend()

    #     def animate(frame):
    #         t = tmax * frame / timestamps
    #         line_rho.set_ydata(data[frame])

    #         fig2.suptitle(f"{case_anim} - t = {t:.3f} s", fontsize=14)
    #         return (line_rho, 0)

    #     anim = animation.FuncAnimation(
    #         fig2, animate, frames=timestamps + 1, interval=50, blit=False, repeat=True
    #     )
    #     plt.tight_layout()
    #     writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    #     anim.save(os.path.join(output_folder, "Interacting_blast_wave_test.gif"), writer=writer)
    #     return anim

    def run_and_plot():

        data = analysis()

        return plot_results(data)

    plot = run_and_plot()


run_1d_interacting_blast_wave_test(8, 2<<2, 1)
