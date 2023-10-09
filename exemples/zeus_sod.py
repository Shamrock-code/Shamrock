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

multx = 2
multy = 1
multz = 1

sz = 1 << 1
base = 32 
model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

cfg = model.gen_default_config()
scale_fact = 1/(sz*base*multx)
cfg.set_scale_factor(scale_fact)
model.set_config(cfg)


kx,ky,kz = 2*np.pi,0,0
delta_rho = 1e-2

def rho_map(rmin,rmax):

    x,y,z = rmin
    if x < 0.5:
        return 1
    else:
        return 0.125


def eint_map(rmin,rmax):

    x,y,z = rmin
    if x < 0.5:
        return 1
    else:
        return 0.1

def vel_map(rmin,rmax):

    return (0,0,0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("eint", eint_map)
model.set_field_value_lambda_f64_3("vel", vel_map)

#model.evolve_once(0,0.1)
freq = 50
for i in range(10000):
    
    if i % freq == 0:
        model.dump_vtk("test"+str(i//freq)+".vtk")

    model.evolve_once(0,0.0005)
