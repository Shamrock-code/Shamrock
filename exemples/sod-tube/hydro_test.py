import shamrock
import matplotlib.pyplot as plt

gamma = 1.4

rho_g = 1
rho_d = 0.125

fact = (rho_g/rho_d)**(1./3.)

P_g = 1
P_d = 0.1

u_g = P_g/((gamma - 1)*rho_g)
u_d = P_d/((gamma - 1)*rho_d)


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

model.init_scheduler(int(1e6),1)


(xs,ys,zs) = model.get_box_dim_fcc_3d(1,256,24,24)
dr = 1/xs
(xs,ys,zs) = model.get_box_dim_fcc_3d(dr,256,24,24)

model.resize_simulation_box((-xs,-ys/2,-zs/2),(xs,ys/2,zs/2))


model.add_cube_fcc_3d(dr, (-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
model.add_cube_fcc_3d(dr*fact, (0,-ys/2,-zs/2),(xs,ys/2,zs/2))

model.set_value_in_a_box("uint", "f64", u_g ,(-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
model.set_value_in_a_box("uint", "f64", u_d ,(0,-ys/2,-zs/2),(xs,ys/2,zs/2))



vol_b = xs*ys*zs

totmass = (rho_d*vol_b) + (rho_g*vol_b)

print("Total mass :", totmass)


pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)



model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)



t_sum = 0
t_target = 0.245
#t_target = 0.01
current_dt = 1e-7
while t_sum < t_target:

    #print("step : t=",t_sum)
    
    next_dt = model.evolve(t_sum,current_dt, False, "dump_"+str(0)+".vtk", False)

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum


import numpy as np
dic = ctx.collect_data()

x =np.array(dic['xyz'][:,0]) + 0.5
vx = dic['vxyz'][:,0]
uint = dic['uint'][:]

hpart = dic["hpart"]

rho = pmass*(model.get_hfact()/hpart)**3
P = (gamma-1) * rho *uint



plt.style.use('custom_style.mplstyle')
fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(9,6),dpi=125)

axs[0,0].scatter(x, vx,c = 'black',s=1,label = "v")
axs[1,0].scatter(x, uint,c = 'black',s=1,label = "u")
axs[0,1].scatter(x, rho,c = 'black',s=1,label = "rho")
axs[1,1].scatter(x, P,c = 'black',s=1,label = "P")


axs[0,0].set_ylabel(r"$v$")
axs[1,0].set_ylabel(r"$u$")
axs[0,1].set_ylabel(r"$\rho$")
axs[1,1].set_ylabel(r"$P$")

axs[0,0].set_xlabel("$x$")
axs[1,0].set_xlabel("$x$")
axs[0,1].set_xlabel("$x$")
axs[1,1].set_xlabel("$x$")

axs[0,0].set_xlim(0,0.55)
axs[1,0].set_xlim(0,0.55)
axs[0,1].set_xlim(0,0.55)
axs[1,1].set_xlim(0,0.55)

plt.tight_layout()
plt.show()