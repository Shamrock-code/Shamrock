import os

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

import matplotlib as mpl


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


import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


sim_folder = f"_to_trash/amr_athena_kelvin_helmholtz/"
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)



# Utility for plotting
def make_cartesian_coords(nx, ny, z_val, min_x, max_x, min_y, max_y):
    # Create the cylindrical coordinate grid
    x_vals = np.linspace(min_x, max_x, nx)
    y_vals = np.linspace(min_y, max_y, ny)

    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")

    # Convert to Cartesian coordinates (z = 0 for a disc in the xy-plane)
    z_grid = z_val * np.ones_like(x_grid)

    # Flatten and stack to create list of positions
    positions = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    return [tuple(pos) for pos in positions]


#################################################################
#  Plot utility
################################################################

def plot_kh_density(ext, time, rho, nx, ny, dpi=200):
    rho = np.asarray(rho).reshape(nx, ny)
    plt.figure(figsize=(6, 6), dpi=dpi)

    im = plt.imshow(
        rho.T,
        origin="lower",
        extent=ext,
        cmap="jet",
        vmin=1.0,
        vmax=2.,
        interpolation="nearest",
        aspect="equal",
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        rf"$t = {time:.3f}$"
    )

    cbar = plt.colorbar(im)
    cbar.set_label(r"$\rho$")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            sim_folder,
            f"amr_kh_density_resx_{nx}_resy_{ny}_at_{time:.3f}.pdf"
        )
    )

    plt.close()




##########################################################
#  Main routine for simulation
##########################################################
    


def run_simulation(output_freq, t_final, extent, base, multx, multy, multz, dens_jump):

    gamma = 1.4
    amr_lev = 2
    sz = 2 << amr_lev
    # sz = 1<<1
    base = base
    scale_fact = 1. / (sz * base * multx)
    nx = base * sz * multx
    ny = base * sz * multy
    positions = make_cartesian_coords(nx, ny, 0.2, 0, 1. - 1e-6, 0, 1. - 1e-6)


    ####---------------
    shamrock.enable_experimental_features()
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(scale_fact)
    cfg.set_eos_gamma(gamma)
    cfg.set_riemann_solver_hllc()
    cfg.set_Csafe(0.5)
    # cfg.set_riemann_solver_hll()
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(True)
    cfg.set_boundary_condition("x", "periodic")
    cfg.set_boundary_condition("y", "periodic")
    cfg.set_boundary_condition("z", "periodic")



    thre_s = 0.01
    cfg.set_amr_mode_shear_based(Threshold=thre_s)


    # err_min = 0.05
    # err_max = 0.10
    # cfg.set_amr_mode_pseudo_gradient_based(error_min=err_min, error_max=err_max)
    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    ###------------


    # P0 = 2.5
    # y0_ref= 0.25
    # y0 = 0.5 + y0_ref
    # A = 0.01
    # sig = 0.2
    # L = 0.005


    # P0 = 10
    # y1= 0.5
    # y2 = 1.5
    # A = 0.01
    # sig = 0.2
    # L = 0.05
    u_flow=1





    P0 = 2.5

    dens_jump = 2.0

    L = 0.01
    A = 0.01
    sig = 0.2

    vflow = 1.0


    def rho_map(rmin, rmax):
        _, y, _ = rmin

        yloc = abs(y - 0.5)

        w = 0.5*(np.tanh((yloc - 0.25)/L) + 1.0)

        return w + (1.0 - w)*dens_jump


    # def rho_map(rmin, rmax):
    #     _, y, _ = rmin
    #     # tmp_rho = dens_jump*0.5*(np.tanh((y-y1)/L) - np.tanh((y-y2)/L))
    #     return 1.5 - 0.5 * np.tanh(np.abs(y - y0) / L)
    #     # return 1.0 + tmp_rho



    def rhovel_map(rmin, rmax):

        x, y, _ = rmin

        yloc = abs(y-0.5)

        w = 0.5*(np.tanh((yloc-0.25)/L)+1.0)

        rho = w + (1-w)*dens_jump

        vx = (w - (1-w)*dens_jump)/rho

        vy = (
            A
            * np.cos(4*np.pi*x)
            * np.exp(-(yloc-0.25)**2/(sig*sig))
        )

        return (rho*vx, rho*vy, 0)


    # def rhovel_map(rmin, rmax):
    #     # vx = 0.5 * np.tanh(np.abs(y - y0)/L)
    #     # vy = A * np.cos(4*np.pi * x)*np.exp(-(y-y0)**2/sig**2)
    #     rho = rho_map(rmin, rmax)
    #     x,y,_ =rmin
    #     tmp_vx = (np.tanh((y-y1)/L) - np.tanh((y-y2)/L) -1)
    #     tmp_vy = np.exp(-(y-y1)**2/sig**2) + np.exp(-(y-y2)**2/sig**2)
    #     vx = u_flow * tmp_vx
    #     vy = A*np.sin(2*np.pi*x) * tmp_vy

     
    #     return (rho*vx, vy*rho, 0)


    # def rhoetot_map(rmin, rmax):
    #     # vx = 0.5 * np.tanh(np.abs(y - y0)/L)
    #     # vy = A * np.cos(4*np.pi * x)*np.exp(-(y-y0)**2/sig**2)
    #     rho = rho_map(rmin, rmax)
    #     x,y,_ =rmin
    #     tmp_vx = (np.tanh((y-y1)/L) - np.tanh((y-y2)/L) -1)
    #     tmp_vy = np.exp(-(y-y1)**2/sig**2) + np.exp(-(y-y2)**2/sig**2)
    #     vx = u_flow * tmp_vx
    #     vy = A*np.sin(2*np.pi*x) * tmp_vy
    #     P = P0 
    #     Eint = P/(gamma - 1.0)
    #     Ekin = 0.5*rho*(vx**2 + vy**2)
    #     return Ekin + Eint
    


    def rhoetot_map(rmin, rmax):

        rho = rho_map(rmin,rmax)

        mx,my,mz = rhovel_map(rmin,rmax)

        kinetic = 0.5*(mx*mx+my*my+mz*mz)/rho

        return kinetic + P0/(gamma-1)


    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)


    #####------------------------
    freq = output_freq
    dt = 0.0000
    t = 0

    for i in range(100):
        if i % freq == 0:
            model.dump_vtk("amr_athena_kelvin_helmhotz" + str(i // freq) + ".vtk")

        next_dt = model.evolve_once_override_time(t, dt)
       
        t += dt
        dt = next_dt
        current_time = t
        

        if i % freq == 0:
            arr_rho_pos    = model.render_slice("rho", "f64", positions)
            rhov_vals      = model.render_slice("rhovel", "f64_3", positions)
            rhoetot_vals   = model.render_slice("rhoetot", "f64", positions)
            vx             = np.array(rhov_vals)[:,0] / np.array(arr_rho_pos)
            vy             = np.array(rhov_vals)[:,1] / np.array(arr_rho_pos)
            P              =  (np.array(rhoetot_vals) - 0.5 * np.array(arr_rho_pos) * (vx**2 + vy**2))*(gamma - 1.)

            output = np.column_stack((np.array(arr_rho_pos), np.array(vx), np.array(vy), np.array(P)))
            filename = f"amr_kh_datas_base_{base}_lev_max_{amr_lev}_nx_{nx}_ny_{ny}_at_{current_time}.txt"
            np.savetxt(os.path.join(sim_folder,filename),
                       output,
                       fmt=["%.10f",  "%.10f", "%.10f",  "%.10f"],
                       header="rho    vx      vy    P",
                       )
            plot_kh_density(extent,current_time,arr_rho_pos,nx,ny)

        if t_final < t + next_dt:
            dt = t_final - t
        if t == t_final:
            break



#####-----------------

multz = 1
#

# multx = 1
# multy = 3
# base = 16

#
base = 32
multx = 1
multy = 1


extent =  [0, 1., 0, 1.]
out_freq = 10
tend = 5
dens_jump =1
run_simulation(out_freq, tend, extent, base, multx, multy, multz, dens_jump)