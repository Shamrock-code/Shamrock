import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

shamrock.enable_experimental_features()
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")



def run_isentropic_vortex( rho_a, u_a,v_a,T_a, vortex_strength=5.0, base=32, L = 10, gamma=1.4, rp_solver="hllc", slope_lim = "vanLeer", t_final = 15, out_freq=100, to_ana_sol=False):


    multx = 1
    multy = 1
    multz = 1

    # sz = 2<< 1
    sz = 1 << 1
    base = base

    cfg = model.gen_default_config()
    scale_fact = L / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)

    thre_s = 0.03
    cfg.set_amr_mode_shear_based(Threshold=thre_s)


    cfg.set_eos_gamma(gamma)
    if(rp_solver == "hllc"):
        cfg.set_riemann_solver_hllc()
    elif(rp_solver == "hll"):
        cfg.set_riemann_solver_hll()

    if(slope_lim == "minmod"):
        cfg.set_slope_lim_minmod()
    elif(slope_lim == "vanLeer"):
        cfg.set_slope_lim_vanleer_sym()
   
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))


    x_c = L/2.
    y_c = L/2.
    z_c = L/2.
    gm1 = gamma - 1.0



    #-----------------------------------------------
    #         Density map 
    #-----------------------------------------------
    
    def rho_map(rmin, rmax):

        xm, ym, zm = rmin
        xM, yM, zM = rmax

        x = 0.5*(xm+xM)
        y = 0.5*(ym+yM)

        dx = x - x_c
        dy = y - y_c

        dx -= L*np.round(dx/L)
        dy -= L*np.round(dy/L)

        r2 = dx*dx + dy*dy

        delta_T = -(gamma-1)*vortex_strength**2 / (
            8*gamma*np.pi**2
        ) * np.exp(1-r2)

        T = T_a + delta_T

        return rho_a * T**(1/(gamma-1))
        
    #-----------------------------------------------
    #         Momentum map 
    #-----------------------------------------------
    def rhovel_map(rmin, rmax):
        xm, ym, zm = rmin
        xM, yM, zM = rmax
        x = 0.5*(xm+xM)
        y = 0.5*(ym+yM)


        dx = x - x_c
        dy = y - y_c

        # periodic coordinates
        dx -= L*np.round(dx/L)
        dy -= L*np.round(dy/L)

        r2 = dx*dx + dy*dy
        rho = rho_map(rmin, rmax)
        delta_u_scal = vortex_strength/(2*np.pi) * np.exp((1. - r2)/2.)
        mom_u = rho * (u_a - dy*delta_u_scal)
        mom_v = rho * (v_a + dx*delta_u_scal)
        return (mom_u, mom_v, 0)
    

    #-----------------------------------------------
    #         Energy map 
    #-----------------------------------------------
    def rhoetot_map(rmin, rmax):
        rho = rho_map(rmin, rmax)
        p = rho**(gamma)
        rhovel = rhovel_map(rmin,rmax)
        u = rhovel[0]/rho
        v = rhovel[1]/rho
        Ekin = 0.5 * rho *( u**2 + v**2)
        Eint =  p/gm1
        return Ekin + Eint

   
    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)


    tmax = t_final
    dt = 0
    t = 0
    target_time = L
    next_output = target_time
    eps = 1e-10

    freq = 2
    for i in range(100000):

        if i==0:
            model.dump_vtk(f"amr_isentropic_vortex_period_{i:04d}.vtk")



        if (to_ana_sol):
            if t < next_output < t + dt:
                dt = next_output - t

        next_dt = model.evolve_once_override_time(t, dt)

        t += dt

        if to_ana_sol:
            if t >= next_output - eps:
                model.dump_vtk(f"amr_isentropic_vortex_period_{i:04d}.vtk")
                next_output +=target_time

        else:
            if t - dt < next_output <= t:
                model.dump_vtk(f"amr_isentropic_vortex_period_{i:04d}.vtk")
                next_output +=target_time

        dt = next_dt

        if tmax < t + next_dt:
            dt = tmax - t

        if t == tmax:
            model.dump_vtk(f"amr_isentropic_vortex_period_{i:04d}.vtk")
            break

run_isentropic_vortex(rho_a=1.,u_a=1.,v_a=1.,T_a=1.,vortex_strength=5.,base=32,L=10.,gamma=1.4,rp_solver="hllc",slope_lim="minmod",t_final=150,out_freq=20, to_ana_sol=False)









# target_time = L

# for i in range(100000):

#     if t < target_time < t + dt:
#         dt = target_time - t

#     next_dt = model.evolve_once_override_time(t, dt)

#     t += dt

#     if abs(t - target_time) < 1e-12:
#         model.dump_vtk("isentropic_vortex_period.vtk")

#     dt = next_dt

#     if tmax < t + next_dt:
#         dt = tmax - t

#     if t == tmax:
#         model.dump_vtk(f"isentropic_vort{i:04d}.vtk")
#         break