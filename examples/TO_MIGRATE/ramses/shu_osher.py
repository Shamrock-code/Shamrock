

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")
import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")



def run_1d_Shu_Osher_test(
    nblocks,
    nxbox,
    nybox,
):
    timestamps = 20
    tmax = 1.8
    gamma = 1.4
    dt_evolve = tmax / timestamps

    timestep_list = [0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.44, 0.45, 0.47, 0.50]
    output_folder = "_to_trash/shu_osher_explicit_tsteps_ref_1024_new/"
    os.makedirs(output_folder, exist_ok=True)

    multx = nxbox
    multy = nybox
    multz = nybox
    sz = 1 << 1
    base = nblocks
    scale_fact = 2 / (sz * base * multx)

    # rez_plot = 256
    rez_plot = base * 2 * multx
    positions_plot = [(x, 0, 0) for x in np.linspace(0, 2, rez_plot).tolist()[:-1]]

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
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))


    x_s = 0.2
    A0 = 0.2
    fp = 5.
    


    def rho_map(rmin, rmax):
        x,y,z = rmin
        rho = rho=1 + A0 * np.sin(fp * np.pi* x)
        if x < x_s:
            rho=3.857143
        return rho


    def rhovel_map(rmin, rmax):
        rho = rho_map(rmin,rmax)
        x,y,z = rmin
        rhovec=(0,0,0)
        if x < x_s:
            rhovec = (rho * 2.629369, 0, 0)
        return rhovec

    def rhoetot_map(rmin, rmax):
        rho = rho_map(rmin,rmax)
        rhov = rhovel_map(rmin, rmax)
        rhoe_kin = 0.5 * (rhov[0]**2)/rho
        rhoe_int = 1.0/(gamma - 1.0)
        x, y, z = rmin
        if x < x_s:
            rhoe_int = 10.33333/(gamma - 1.0)
        return rhoe_int + rhoe_kin

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    def analysis():
  
        results = []
        # for i  in range(timestamps+1):
        for dt  in timestep_list:
            # model.evolve_until(dt_evolve * i)
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
            filename= f"data_Shu_Osher_reso_{2*base*multx}_at_{dt}.txt"
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
            plt.savefig(os.path.join(output_folder, f"Shu_Osher_test_at_{timestep_list[i]}_nx_reso_{base*2*multx}.png"))
            plt.close(fig)
    def run_and_plot():

        data = analysis()

        return plot_results(data)

    plot = run_and_plot()


run_1d_Shu_Osher_test(8, 2<<5, 1)