

import shamrock
import matplotlib.pyplot as plt
from math import exp

gamma = 5./3.
rho_g = 1
target_tot_u = 1


dr = 0.005
bmin = (-0.6,-0.6,-0.6)
bmax = ( 0.6, 0.6, 0.6)
pmass = -1


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)


bmin,bmax = model.get_ideal_fcc_box(dr,bmin,bmax)
xm,ym,zm = bmin
xM,yM,zM = bmax


model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho_g*vol_b)


pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)


model.set_value_in_a_box("uint","f64", 1 , bmin,bmax)



def vel_func(r):
    x,y,z = r

    vel = 0

    q = abs(z)*4

    if (q < 1) :
        vel = 1 + q * q * ((3. / 4.) * q - (3. / 2.))
    elif (q < 2) :
        vel = (1. / 4.) * (2 - q) * (2 - q) * (2 - q)
    else :
        vel = 0

    return (0,vel,0)

model.set_field_value_lambda_f64_3("vxyz", vel_func)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)



t_sum = 0
t_target = 0.5
current_dt = 1e-7
i = 0
i_dump = 0
while t_sum < t_target:

    #print("step : t=",t_sum)
    
    next_dt = model.evolve(t_sum,current_dt, True, "dump_"+str(i_dump)+".vtk", True)

    if i % 1 == 0:
        i_dump += 1

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum

    i+= 1



import numpy as np
dic = ctx.collect_data()

y = dic['xyz'][:,2]
vz = np.abs(dic['vxyz'][:,2] )


hpart = dic["hpart"]
uint = dic["uint"]

gamma = 5./3.

rho = pmass*(1.2/hpart)**3
P = (gamma-1) * rho *uint

print("result : ", dr, np.max(vz))

plt.style.use('custom_style.mplstyle')
plt.scatter(y,vz)
plt.title("$dr = {:.2f} ~ t={}$".format(dr,t_sum))
plt.xlabel("$z$")
plt.ylabel(r"$\vert v_y/v_z\vert$")
plt.yscale('log')
plt.tight_layout()
plt.show()