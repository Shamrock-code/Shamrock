
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


import shamrock




######################################################
## Analytical solution object Adapted form
#####################################################



def run_numerical_simulation(max_amr_lev, multx, multy, multz, base, L, gamma, rho0, E0, alpha0, P0_out = 1e-3, with_amr=0 ):

    #------------------ Context -----------------------------
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")
    #--------------------------------------------------------




    # multx = 1
    # multy = 1
    # multz = 1
    max_amr_lev = 2
    cell_size = 2 << max_amr_lev  # refinement is limited to cell_size = 2
    scale_fact = L / (cell_size * base * multx)
    # base = 16

    #----------------------------------- Model configuration ------------------
    cfg = model.gen_default_config()
    cfg.set_scale_factor(scale_fact)
    cfg.set_eos_gamma(gamma)
    cfg.set_Csafe(0.5)
    cfg.set_boundary_condition("x", "periodic")
    cfg.set_boundary_condition("y", "periodic")
    cfg.set_boundary_condition("z", "periodic")

    cfg.set_riemann_solver_hllc()
    cfg.set_slope_lim_minmod()
    cfg.set_face_time_interpolation(True)


    ### Radius of 1 cell (want set the explosion in a single point )
    Rstart = scale_fact
    # gamma = 5.0 / 3.0

    ### amr configuration
    if(with_amr):
        err_min = 0.30
        err_max = 0.10
        cfg.set_amr_mode_pseudo_gradient_based(error_min=err_min, error_max=err_max)

    model.set_solver_config(cfg)

    ### split factors
    model.init_scheduler(int(1e7), 1)
    ### set grid 
    model.make_base_grid(
        (0, 0, 0), (cell_size, cell_size, cell_size), (base * multx, base * multy, base * multz)
    )


    def rho_map(rmin, rmax):
        return 1.0
        # return rho0



    ####--------pre-check for the Energy setup --------------

    dx = scale_fact
    Vcell = dx**3

    dx = scale_fact
    Vcell = dx**3

    Ncells = 0

    Nx = cell_size * base * multx
    Ny = cell_size * base * multy
    Nz = cell_size * base * multz

    for k in range(Nz):
        z = (k + 0.5) * dx - L/2.
        for j in range(Ny):
            y = (j + 0.5) * dx - L/2.
            for i in range(Nx):
                x = (i + 0.5) * dx - L/2.

                r = np.sqrt(x*x + y*y + z*z)

                if r < Rstart:
                    Ncells += 1

    print("Number of injection cells =", Ncells)

    Vinj = Ncells * Vcell

    rhoe_in = E0 / Vinj

    Pin = (gamma - 1.0) * rhoe_in

    print(f"Injected volume = {Vinj:.6e}")
    print(f"Injected pressure = {Pin:.6e}")
    print(f"Injected energy density = {rhoe_in:.6e}")


    # def rhoe_map(rmin, rmax):
    #     x_min, y_min, z_min = rmin
    #     x_max, y_max, z_max = rmax

    #     x = (x_min + x_max) * 0.5 - L/2.
    #     y = (y_min + y_max) * 0.5 - L/2.
    #     z = (z_min + z_max) * 0.5 - L/2.
    #     ## radius from box center
    #     r = np.sqrt(x * x + y * y + z * z)

    #     if r < Rstart:
    #         # Vinj = 4./3.*np.pi*Rstart**3
    #         # Pin = (gamma-1.)*E0/Vinj
    #         # rhoe_in = Pin/(gamma-1.)
    #         return rhoe_in
    #     else:
    #         return P0_out/(gamma -1.)
        
    
    def rhoe_map(rmin, rmax):
        x_min, y_min, z_min = rmin
        x_max, y_max, z_max = rmax
        x = (x_min + x_max) * 0.5 - 0.5
        y = (y_min + y_max) * 0.5 - 0.5
        z = (z_min + z_max) * 0.5 - 0.5
        r = np.sqrt(x * x + y * y + z * z)
        if r < Rstart:
            return 10.0 / (gamma - 1.0)
        else:
            return 1e-5 / (gamma - 1.0)


    def rhovel_map(rmin, rmax):
        return (0.0, 0.0, 0.0)


    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    tmax = 0.2


    dt = 0
    t = 0
    freq = 100
    dX0 = []
    for i in range(10000):
        next_dt = model.evolve_once_override_time(t, dt)

        t += dt
        dt = next_dt

        if i % freq == 0:
            model.dump_vtk(f"Sedov_blast_3d_{i:04d}.vtk")

        if tmax < t + next_dt:
            dt = tmax - t
        if t == tmax:
            model.dump_vtk(f"Sedov_blast_3d_{i:04d}.vtk")
            break




alpha0 = 0.49
gamma = 5./3.
E0 = 1e5
rho0 = 1.
P0 = 1e-3
L = 1
max_amr_lev=3
multx = 1
multy = 1
multz = 1
base = 16

run_numerical_simulation(max_amr_lev,multx,multy,multz,base,L,gamma,rho0,E0,alpha0,P0,0)


