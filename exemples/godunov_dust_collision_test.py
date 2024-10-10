import shamrock
import numpy as np
import matplotlib.pyplot as plt
import os

def run_sim():
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_AMRGodunov(
        context = ctx,
        vector_type = "f64_3",
        grid_repr   = "i64_3"
    )

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 32

    cfg = model.gen_default_config()
    scale_fact = 1/(sz*base*multx)
    cfg.set_scale_factor(scale_fact)
    cfg.set_Csafe(0.5)                  #  TODO : remember to add possibility of different CFL for fluids(e.g Csafe_gas and Csafe_dust ...)
    cfg.set_eos_gamma(1.4)              # set adiabatic index gamma , here adiabatic EOS
    cfg.set_dust_mode_dhll(2)           # enable dust config 
    cfg.set_drag_mode_irk1(True)        # enable drag config

    #======= set drag coefficients for test B ========
    cfg.set_alpha_values(100)          # ts := 0.01
    cfg.set_alpha_values(500)          # ts := 0.002

    """
    #======= set drag coefficients for test C ========
    cfg.set_alpha_values(0.5)          # ts  := 2
    cfg.set_alpha_values(1)            # ts  := 1

    """

    model.set_config(cfg)
    model.init_scheduler(int(1e7),1)
    model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

    #============= Fileds maps for gas ==============

    def rho_map(rmin,rmax):
        return 1  # 1 is the initial density
    
    def rhovel_map(rmin,rmax):
        return (1, 0, 0)  # vg_x:=1, vg_y:=0, vg_z:=0
    
    def rhoe_map(rmin,rmax):
        cs_0 = 1.4
        gamma = 1.4
        rho_0 = 1
        press = (cs_0*rho_0)/gamma
        rhoeint = press / (gamma - 1.0)
        rhoekin = 0.5 * rho_0    # vg_x:=1, vg_y:=0, vg_z:=0
        return (rhoeint + rhoekin)

    #=========== Fields maps for dust in test B ============
    def b_rho_d_1_map(rmin, rmax):
        return 1     # rho_d_1 := 1
    
    def b_rho_d_2_map(rmin, rmax):
        return 1   # rho_d_2 := 1
    
    def b_rhovel_d_1_map(rmin,rmax):
        return (2,0,0)  # vd_1_x:=2, vd_1_y:=0, vd_1_z:=0
    
    def b_rhovel_d_2_map(rmin,rmax):
        return (0.5,0,0)  # vd_2_x:=0.5, vd_2_y:=0, vd_2_z:=0
    

    """

    #=========== Fields maps for dust in test C ============
    def c_rho_d_1_map(rmin, rmax):
        return 10     # rho_d_1 := 10
    
    def c_rho_d_2_map(rmin, rmax):
        return 100   # rho_d_2 := 100
    
    def c_rhov_d_1_map(rmin,rmax):
        return (2,0,0)  # vd_1_x:=2, vd_1_y:=0, vd_1_z:=0
    
    def c_rhov_d_2_map(rmin,rmax):
        return (0.5,0,0)  # vd_2_x:=0.5, vd_2_y:=0, vd_2_z:=0
    
    """

    #============ set init fields values for gas =============

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    #============ set init fields values for dusts in test B ==========
    model.set_field_value_lambda_f64("rho_dust", b_rho_d_1_map,0)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_1_map,0)
    model.set_field_value_lambda_f64("rho_dust", b_rho_d_2_map,1)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_2_map,1)


    """
    #============ set init fields values for dusts in test C ==========
    model.set_field_value_lambda_f64("rho_dust", c_rho_d_1_map,0)
    model.set_field_value_lambda_f64_3("rhovel_dust", c_rhovel_d_1_map,0)
    model.set_field_value_lambda_f64("rho_dust", c_rho_d_2_map,1)
    model.set_field_value_lambda_f64_3("rhovel_dust", c_rhovel_d_2_map,1)
    """

    dt = 0.005  # b_dt := 0.005 and c_dt := 0.05
    t = 0
    tend = 0.05  # b_tend := 0.05 and c_tend := 0.3
    freq = 1

    for i in range(10000):

        if i % freq == 0:
            model.dump_vtk("colid_test" + str(i//freq)+".vtk")
        
        model.evolve_once_override_time(dt*float(i),dt)
        t= dt*i

        if t >= tend:
            break

    
    def convert_to_cell_coords(dic):

        cmin = dic['cell_min']
        cmax = dic['cell_max']

        xmin = []
        ymin = []
        zmin = []
        xmax = []
        ymax = []
        zmax = []

        for i in range(len(cmin)):

            m,M = cmin[i],cmax[i]

            mx,my,mz = m
            Mx,My,Mz = M

            for j in range(8):
                a,b = model.get_cell_coords(((mx,my,mz), (Mx,My,Mz)),j)

                x,y,z = a
                xmin.append(x)
                ymin.append(y)
                zmin.append(z)

                x,y,z = b
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
    
    dic = convert_to_cell_coords(ctx.collect_data())

    run_sim()





