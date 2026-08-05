"""
Kelvin-Helmholtz instability in RAMSES solver
=============================================

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

# plot
nx, ny = 512, 512

# Physical parameters
vslip = 1  # slip speed between the two layers

rho_1 = 1
fact = 2 / 3
rho_2 = rho_1 / (fact**3)

y_interface = 0.5
xs = 1

P_1 = 3.5
P_2 = 3.5

gamma = 5.0 / 3.0



sim_folder = "_to_trash/ramses_Tim_kh/amr/"

# %%
# Deduced quantities


u_1 = P_1 / ((gamma - 1) * rho_1)
u_2 = P_2 / ((gamma - 1) * rho_2)

print("Mach number 1 :", vslip / np.sqrt(gamma * P_1 / rho_1))
print("Mach number 2 :", vslip / np.sqrt(gamma * P_2 / rho_2))


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

positions = make_cartesian_coords(nx, ny, 0.5, 0, 1 - 1e-6, 0, 1 - 1e-6)



def plot_kh_density(ext, time, rho, nx, ny, iplot, dpi=200):

    
    my_cmap = matplotlib.colormaps["rainbow"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    arr_rho_pos = np.array(rho).reshape(nx, ny)

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
    plt.title(f"t = {time:0.3f}")
    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\rho$")
    plt.savefig(os.path.join(sim_folder, f"kh_density_resx_{nx}_resy_{ny}_at_{time:.3f}_{iplot:04d}.png"))
    plt.close()


# from shamrock.utils.plot import show_image_sequence


def run_simulation(output_freq, t_final, extent, base, multx, multy, multz):

    sz = 2 << 1
    base = base
    scale_fact = 1. / (sz * base * multx)





    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(scale_fact)
    cfg.set_boundary_condition("x", "periodic")
    cfg.set_boundary_condition("y", "periodic")
    cfg.set_boundary_condition("z", "periodic")

    # cfg.set_riemann_solver_rusanov()
    # cfg.set_riemann_solver_hll()
    cfg.set_riemann_solver_hllc()

    # cfg.set_slope_lim_none()
    # cfg.set_slope_lim_vanleer_f()
    # cfg.set_slope_lim_vanleer_std()
    cfg.set_slope_lim_vanleer_sym()

    # mass_crit = 0.00010299682617187501 / 2
    # cfg.set_amr_mode_density_based(crit_mass=mass_crit)
    # cfg.set_slope_lim_minmod()


    thre_s = 0.01
    cfg.set_amr_mode_shear_based(Threshold=thre_s)



    cfg.set_face_time_interpolation(True)
    cfg.set_eos_gamma(gamma)
    model.set_solver_config(cfg)


    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    u_cs1 = rho_1 / (gamma * (gamma - 1))
    u_cs2 = rho_2 / (gamma * (gamma - 1))

    def rho_map(rmin, rmax):
        x, y, z = rmin

        if y > y_interface:
            return rho_2
        else:
            return rho_1

    def rhovel_map(rmin, rmax):
        rho = rho_map(rmin, rmax)

        x, y, z = rmin

        ampl = 0.01
        n = 2
        pert = np.sin(2 * np.pi * n * x / (xs))

        sigma = 0.05 / (2**0.5)
        gauss1 = np.exp(-((y - y_interface) ** 2) / (2 * sigma * sigma))
        gauss2 = np.exp(-((y + y_interface) ** 2) / (2 * sigma * sigma))
        pert *= gauss1 + gauss2

        # Alternative formula (See T. Tricco paper)
        # interf_sz = ys/32
        # vx = np.arctan(y/interf_sz)/np.pi

        vx = 0
        if np.abs(y) > y_interface:
            vx = vslip / 2
        else:
            vx = -vslip / 2

        return (vx * rho, ampl * pert * rho, 0)

    def rhoetot_map(rmin, rmax):
        rho = rho_map(rmin, rmax)
        rhovel = rhovel_map(rmin, rmax)

        rhovel2 = rhovel[0] * rhovel[0] + rhovel[1] * rhovel[1] + rhovel[2] * rhovel[2]
        rhoekin = 0.5 * rhovel2 / rho

        x, y, z = rmin

        if y > y_interface:
            P = P_2
        else:
            P = P_1

        return rhoekin + P / (gamma - 1)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    # model.evolve_once(0,0.1)
    # fact = 25
    # tmax = 0.127 * fact
    # all_t = np.linspace(0, tmax, fact)


    #####------------------------
    freq = output_freq
    dt = 0.0000
    t = 0

    for i in range(100000):
        if i % freq == 0:
            model.dump_vtk(os.path.join(sim_folder, f"AMR_kelvin_helmhotz" + str(i // freq) + ".vtk"))

        next_dt = model.evolve_once_override_time(t, dt)
       
        t += dt
        dt = next_dt
        current_time = t
       

        if i % freq == 0:
            arr_rho_pos = model.render_slice("rho", "f64", positions)
            plot_kh_density(extent,current_time,arr_rho_pos,nx,ny, i)


        if t_final < t + next_dt:
            dt = t_final - t
        if t == t_final:
            break







multz = 1
#

# multx = 1
# multy = 3
# base = 16

#
base = 32
multx = 1
multy = 1


extent =  [0., 1, 0., 1]
out_freq = 5
tend = 5
run_simulation(out_freq, tend, extent, base, multx, multy, multz)
