import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#=====================
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

mpl.rcParams.update({
    "text.usetex": True,              # Use LaTeX
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],  # Match lmodern
    
    # LaTeX preamble to match your class
    "text.latex.preamble": r"""
        \usepackage{lmodern}
        \usepackage{amsmath}
        \usepackage{amssymb}
    """,

    # Optional but recommended
    "axes.unicode_minus": False
})
mpl.rcParams["pgf.texsystem"] = "pdflatex"

#============
import shamrock

#=========

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")



def run_linear_wave_cvg_test(nxbox_list):
    base = 8
    nybox = 1
    nzbox = 1


    gamma = 5./3.
    output_folder = "_to_trash/linear_wave_cvg/"
    os.makedirs(output_folder, exist_ok=True)

    all_run_num_datas = []
    all_run_ana_datas = []

    for i, nxbox in enumerate (nxbox_list):
        multx = nxbox
        multy = nybox
        multz = nzbox
        sz = 1 << 1
        ncells =  sz * base * multx
        scale_fact = 1 / (ncells)

   

        ctx = shamrock.Context()
        ctx.pdata_layout_new()

        model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

        cfg = model.gen_default_config()
        cfg.set_scale_factor(scale_fact)

        cfg.set_eos_gamma(gamma)
        cfg.set_Csafe(0.8)
        cfg.set_riemann_solver_hllc()
        cfg.set_slope_lim_minmod()
        cfg.set_face_time_interpolation(True)
        cfg.set_boundary_condition("x", "periodic")
        cfg.set_boundary_condition("y", "periodic")
        cfg.set_boundary_condition("z", "periodic")

        model.set_solver_config(cfg)
        model.init_scheduler(int(1e7), 1)
        model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

        A = 1e-6

        rho0 = 1.0
        P0 = 1.0 / gamma
        u0 = 0.0

        cs0 = np.sqrt(gamma * P0 / rho0)

        E0 = P0/(gamma - 1.0)
        H0 = (E0 + P0)/rho0

        L = 1.
        tf = 1./cs0

        def convert_to_cell_coords(dic):

            cmin = dic["cell_min"]
            cmax = dic["cell_max"]

            xmin = []
            ymin = []
            zmin = []
            xmax = []
            ymax = []
            zmax = []

            for i in range(len(cmin)):
                m, M = cmin[i], cmax[i]

                mx, my, mz = m
                Mx, My, Mz = M

                for j in range(8):
                    a, b = model.get_cell_coords(((mx, my, mz), (Mx, My, Mz)), j)

                    x, y, z = a
                    xmin.append(x)
                    ymin.append(y)
                    zmin.append(z)

                    x, y, z = b
                    xmax.append(x)
                    ymax.append(y)
                    zmax.append(z)

            dic["xmin"] = np.array(xmin)
            dic["ymin"] = np.array(ymin)
            dic["zmin"] = np.array(zmin)
            dic["xmax"] = np.array(xmax)
            dic["ymax"] = np.array(ymax)
            dic["zmax"] = np.array(zmax)

            return dic


        def analytical_sol(x_pos, time):
            right_lambda = u0 + cs0
            rho = rho0 + A * np.sin(2 * np.pi * (np.array(x_pos) - right_lambda * time))
            rhovx = A * cs0* np.sin(2* np.pi * ( np.array(x_pos) - right_lambda * time ))
            vx = rhovx / rho
            E =  E0 + H0 * A * np.sin(2 * np.pi * (np.array(x_pos) - right_lambda * time))

            results_dic = {
            "rho": np.array(rho),
            "vx": np.array(vx),
            "E": np.array(E)
                }
            return results_dic


        def analysis():
            results = []


            model.evolve_until(tf)

            dic = convert_to_cell_coords(ctx.collect_data())
            rho_vals = np.array(dic["rho"])
            rhov_vals = np.array(dic["rhovel"])
            rhoetot_vals = np.array(dic["rhoetot"])
            vx = np.array(rhov_vals)[:,0] / np.array(rho_vals)


            XV = (np.array(dic["xmin"]) + np.array(dic["xmax"]) ) * 0.5

            idx = np.argsort(XV)

            num_results_dic = {
            "dx": XV[idx],
            "rho": np.array(rho_vals[idx]),
            "vx": np.array(vx[idx]),
            "E": np.array(rhoetot_vals[idx])
                }
            results.append(num_results_dic)

            ana_results_dic = analytical_sol(XV[idx],tf)

            all_run_num_datas.append(num_results_dic)
            all_run_ana_datas.append(ana_results_dic)


        
            output = np.column_stack((XV, np.array(rho_vals), ana_results_dic["rho"], np.array(vx), ana_results_dic["vx"] , np.array(rhoetot_vals), ana_results_dic["E"]))
            filename= f"data_linear_wave_reso_{ncells}_at_{tf}.txt"
            np.savetxt(os.path.join(output_folder,filename),
                        output,
                        fmt=["%.10f", "%.10f",  "%.10f", "%.10f",  "%.10f", "%.10f", "%.10f"],
                        header="dX rho_num rho_ana  vx_num vx_ana E_num E_ana",
                        )


        def rho_map(rmin, rmax):
            x_m,_,_ = rmin
            x_M,_,_ = rmax
            x = 0.5 * (x_m + x_M)
            return rho0 + A * np.sin(2 * np.pi * x)

        def rhovel_map(rmin, rmax):
            x_m,_,_ = rmin
            x_M,_,_ = rmax
            x = 0.5 * (x_m + x_M)
            rhovx = A * cs0* np.sin(2* np.pi * x)
            return (rhovx, 0., 0.)

        def rhoetot_map(rmin, rmax):
            x_m,_,_ = rmin
            x_M,_,_ = rmax
            x = 0.5 * (x_m + x_M)
            return E0 + H0 * A * np.sin(2 * np.pi * x)

        model.set_field_value_lambda_f64("rho", rho_map)
        model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
        model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    

    
        analysis()

    return all_run_ana_datas, all_run_num_datas

nxbox_list = [2<<0, 2<<1, 2<<2, 2<<3, 2<<4, 2<<5, 2<<6]
analytical_datas,numerical_datas = run_linear_wave_cvg_test(nxbox_list)



def get_error(analytical_datas, numerical_datas, Ncells):
   
    error_list = []
    for i,N in enumerate(Ncells):
      
        err_rho = (np.abs(analytical_datas[i]["rho"] - numerical_datas[i]["rho"]).sum())/N
        err_vx  = (np.abs(analytical_datas[i]["vx"]  - numerical_datas[i]["vx"]).sum())/N
        err_E   = (np.abs(analytical_datas[i]["E"] - numerical_datas[i]["E"]).sum())/N

        Err = np.sqrt(err_rho**2 + err_vx**2 + err_E**2)
        error_list.append(Err)
    
    return np.array(error_list)



def plot_profiles_(analytical_datas, numerical_datas,Ncells):
    plt.figure(
    figsize=(6,5)
    )

    Error = get_error(analytical_datas,numerical_datas,Ncells)

    plt.loglog(
        Ncells,
        Error,
        "o-",
        label="Measured"
    )
    plt.xlabel(r"$N_{x}$")
    plt.ylabel(r"$\\langle E \\rangle$")

    ref = Error[0] * (Ncells[0]/np.array(Ncells))**2

    Nx = [32,64,128,256,512,1024,2048]
    plt.loglog(
        Nx ,
        ref,
        "--",
        label=r"$N_{x}^{-2}$"
    )

    plt.legend()

    plt.tight_layout()
    plt.savefig("Linear_wave_test_cvg.pdf")
    plt.close()

Ncells = np.array([32,64,128,256,512,1024,2048])*(16*16)
plot_profiles_(analytical_datas,numerical_datas,Ncells)


        




    




