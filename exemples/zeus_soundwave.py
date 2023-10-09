import shamrock
import numpy as np
import matplotlib.pyplot as plt 
import os


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_AMRZeus(
    context = ctx, 
    vector_type = "f64_3",
    grid_repr = "i64_3")

model.init_scheduler(int(1e7),1)

multx = 1
multy = 1
multz = 1

sz = 1 << 1
base = 32 
model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

cfg = model.gen_default_config()
scale_fact = 1/(sz*base*multx)
cfg.set_scale_factor(scale_fact)

cfg.set_eos_gamma(5./3.)
model.set_config(cfg)


gamma = 5./3. 

u_cs1 = 1/( gamma*(gamma-1))

kx,ky,kz = 2*np.pi,0,0
delta_rho = 1e-5
delta_v = 1e-5

def rho_map(rmin,rmax):

    x,y,z = rmin

    return 1. + delta_rho*np.cos(kx*x + ky*y + kz*z)

def eint_map(rmin,rmax):

    return u_cs1*10

def vel_map(rmin,rmax):

    x,y,z = rmin
    return (0+ delta_v*np.cos(kx*x + ky*y + kz*z) ,0,0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("eint", eint_map)
model.set_field_value_lambda_f64_3("vel", vel_map)

#model.evolve_once(0,0.1)
freq = 20
for i in range(2000):
    
    if i % freq == 0:
        model.dump_vtk("test"+str(i//freq)+".vtk")

    model.evolve_once(float(i),0.0001)
