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


sim_folder = f"_to_trash/rayleigh_taylor_vl/amr/"
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

def plot_rt_density(ext, time, rho, nx, ny, dpi=200):
    rho = np.asarray(rho).reshape(nx, ny)
    plt.figure(figsize=(6, 12), dpi=dpi)

    # im = plt.imshow(
    #     rho.T,
    #     origin="lower",
    #     extent=ext,
    #     cmap="viridis",
    #     vmin=1.0,
    #     vmax=2.0,
    #     interpolation="nearest",
    #     aspect="equal",
    # )

    im = plt.imshow(
    rho.T,
    origin="lower",
    extent=ext,
    cmap="rainbow",
    interpolation="bicubic",
    vmin=1.0,
    vmax=2.0,
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
            f"rt_density_resx_{nx}_resy_{ny}_at_{time:.3f}.pdf"
        )
    )

    plt.close()




##########################################################
#  Main routine for simulation
##########################################################
    


def run_simulation(output_freq, t_final, extent, base, multx, multy, multz):

    gamma = 1.4
    amr_lev = 2
    sz = 2 << amr_lev
    base = base
    scale_fact = 0.5 / (sz * base * multy)
    nx = base * sz * multx
    ny = base * sz * multy
    positions = make_cartesian_coords(nx, ny, 0.2, 0, 0.5 - 1e-6, 0, 1.5 - 1e-6)


    ####---------------
    shamrock.enable_experimental_features()
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(scale_fact)
    cfg.set_eos_gamma(gamma)
    cfg.set_riemann_solver_hllc()
    cfg.set_Csafe(0.8)
    # cfg.set_riemann_solver_hll()
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(True)
    cfg.set_boundary_condition("x", "periodic")
    cfg.set_boundary_condition("y", "reflective")
    cfg.set_boundary_condition("z", "reflective")
    err_min = 0.05
    err_max = 0.10
    cfg.set_amr_mode_pseudo_gradient_based(error_min=err_min, error_max=err_max)
    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    ###------------
    g = 0.1
    y_half = 1.5 / 2.
    P0 = 2.5


    def rho_map(rmin, rmax):
        _, y, _ = rmin
        if y <= y_half:
            return 1.
        else:
            return 2.


    def rhovel_map(rmin, rmax):
        rho = rho_map(rmin, rmax)
        x,y,_ =rmin
        xloc = x - 0.25
        yloc = y - 0.75
        vy = 1e-2 * (1./4.)*(1. + np.cos(4. * np.pi * xloc))*(1. + np.cos(3. * np.pi * yloc))
        return (0, vy*rho, 0)


    def rhoetot_map(rmin, rmax):
        rho = rho_map(rmin, rmax)
        x, y, _ = rmin
        P = P0 -0.1*rho*(y-y_half)
        xloc = x - 0.25
        yloc = y - 0.75
        vy = 1e-2 * (1./4.)*(1. + np.cos(4. * np.pi * xloc))*(1. + np.cos(3. * np.pi * yloc))
        Eint = P/(gamma - 1.0)
        Ekin = 0.5*rho*(vy**2)
        return Ekin + Eint


    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)


    #####------------------------
    freq = output_freq
    dt = 0.0000
    t = 0

    for i in range(15):
        if i % freq == 0:
            model.dump_vtk("Rayleigh_Taylor" + str(i // freq) + ".vtk")

        next_dt = model.evolve_once_override_time(t, dt)
       
        t += dt
        dt = next_dt
        current_time = t + dt
        arr_rho_pos = model.render_slice("rho", "f64", positions)

        if i % freq == 0:
            plot_rt_density(extent,current_time,arr_rho_pos,nx,ny)


        if t_final < t + next_dt:
            dt = t_final - t
            current_time = t + dt
            arr_rho_pos = model.render_slice("rho", "f64", positions)
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
multy = 3


extent =  [0, 0.5, 0, 1.5]
out_freq = 1
tend = 5
run_simulation(out_freq, tend, extent, base, multx, multy, multz)