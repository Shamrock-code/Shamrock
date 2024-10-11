import shamrock
import numpy as np
import matplotlib.pyplot as plt
import os

tmax = 1.0
def run_sim(vanleer = True, label = "none"):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_AMRGodunov(
        context = ctx,
        vector_type = "f64_3",
        grid_repr = "i64_3")


    # model.init_scheduler(int(1e7),1)

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    # base = 32
    base = 2
    # model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

    cfg = model.gen_default_config()
    scale_fact = 1/(sz*base*multx)
    cfg.set_scale_factor(scale_fact)
    cfg.set_eos_gamma(1.66667)
    # cfg.set_slope_lim_none()
    cfg.set_face_time_interpolation(True)
    cfg.set_dust_mode_dhll(2)
    cfg.set_drag_mode_no_drag()
    # cfg.set_alpha_values(0.2)
    # cfg.set_alpha_values(0.5)
    # cfg.set_alpha_values(0.4)
    # cfg.set_alpha_values(0.3)
    
    
    model.set_config(cfg)
    model.init_scheduler(int(1e7),1)
    model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))


    def rho_map(rmin,rmax):

        x,y,z = rmin

        if x > 0.25 and x < 0.75:
            return 2

        return 1.

    def rhoe_map(rmin,rmax):

        rho = rho_map(rmin,rmax)
        return 1.*rho

    def rhovel_map(rmin,rmax):

        rho = rho_map(rmin,rmax)
        return (1 *rho,0 *rho,0 *rho)
    
    def rho_d_map(rmin,rmax):

        x,y,z = rmin

        if x > 0.25 and x < 0.75:
            return 2

        return 1.

    def rhovel_d_map(rmin,rmax):

        x,y,z = rmin
        rho = rho_d_map(rmin,rmax)
        return (1 *rho,0 *rho,0 *rho)

    def rho_d_map_1(rmin,rmax):

        x,y,z = rmin

        if x > 0.25 and x < 0.75:
            return 2

        return 1.

    def rhovel_d_map_1(rmin,rmax):

        x,y,z = rmin
        rho = rho_d_map_1(rmin,rmax)
        return (1.25 *rho,0 *rho,0 *rho)


    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)
    model.set_field_value_lambda_f64("rho_dust", rho_d_map,0)
    model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_d_map,0)
    model.set_field_value_lambda_f64("rho_dust", rho_d_map_1,1)
    model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_d_map_1,1)
  

    
    freq = 50
    dt = 0.0000
    t = 0
    tend = 0.245

    for i in range(2):

        # if i % freq == 0:
        model.dump_vtk("test"+str(i)+".vtk")

        next_dt = model.evolve_once_override_time(t,dt)

        t += dt
        dt = next_dt

        if tend < t + next_dt:
            dt = tend - t
        if t == tend:
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

    print(dic)

    X = []
    rho = []
    velx = []
    rho_d = []
    velx_d = []
    eint = []

    for i in range(len(dic["xmin"])):

        X.append(dic["xmin"][i])
        rho.append(dic["rho"][i])
        velx.append(dic["rhovel"][i][0])
        print(f"rho shape = {dic["rho"].shape}")
        print(f"rho_vel shape = {dic["rhovel"].shape}")
        print(f"rho_d shape = {dic["rho_dust"].shape}")
        # rho_d.append(dic["rho_dust"][i])
        # velx_d.append(dic["rhovel_dust"][i][0])

    # plt.plot(X,rho,'.',label='rho')
    # plt.plot(X,rho_d,'.',label='rho_d')


run_sim(vanleer = True, label = "van leer")


# plt.legend()
# plt.grid()


#plt.ylim(0.9,2.5)
#plt.xlim(0,1)


# plt.title("t="+str(tmax))
# plt.show()
