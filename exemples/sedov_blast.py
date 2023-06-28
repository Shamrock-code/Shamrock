import shamrock
import matplotlib.pyplot as plt

gamma = 5./3.
rho_g = 1
target_tot_u = 1


pmass = -1

#Nx = 200
#Ny = 230
#Nz = 245

Nx = int(100)
Ny = int(130)
Nz = int(145)



ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")
model.init_scheduler(int(1e7),1)

#start the scheduler
ctx.init_sched(int(1e7),1)

setup = shamrock.SetupSPH(kernel = "M4", precision = "double")
setup.init(ctx)

(xs,ys,zs) = setup.get_box_dim(1,Nx,Ny,Nz)
dr = 1/xs
(xs,ys,zs) = setup.get_box_dim(dr,Nx,Ny,Nz)

ctx.set_coord_domain_bound((-xs/2,-ys/2,-zs/2),(xs/2,ys/2,zs/2))

setup.set_boundaries("periodic")

setup.add_particules_fcc(ctx,dr, (-xs/2,-ys/2,-zs/2),(xs/2,ys/2,zs/2))

xc,yc,zc = setup.get_closest_part_to(ctx,(0,0,0))

del model
del setup
del ctx



ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)

(xs,ys,zs) = model.get_box_dim_fcc_3d(1,Nx,Ny,Nz)
dr = 1/xs
(xs,ys,zs) = model.get_box_dim_fcc_3d(dr,Nx,Ny,Nz)

bmin = (-xs/2-xc,-ys/2-yc,-zs/2-zc)
bmax = (xs/2-xc,ys/2-yc,zs/2-zc)

model.resize_simulation_box(bmin,bmax)

model.add_cube_fcc_3d(dr, bmin,bmax)

vol_b = xs*ys*zs

totmass = (rho_g*vol_b)
print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

#model.set_value_in_a_box("uint","f64", 0.005 , bmin,bmax)

print(">>> iterating toward tot u = 1")
rinj = 0.01

u_inj = 1000
while 1:
    
    model.set_value_in_sphere("uint","f64", u_inj,(0,0,0),rinj)

    tot_u = pmass*model.get_sum("uint","f64")

    u_inj *= target_tot_u/tot_u
    print("total u :",tot_u)

    if abs(tot_u - target_tot_u) < 1e-5:
        break



print("Current part mass :", pmass)

#for it in range(5):
#    setup.update_smoothing_lenght(ctx)






model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)
model.set_eos_gamma(5/3)





print("Current part mass :", pmass)


model.set_particle_mass(pmass)

#for i in range(9):
#    model.evolve(5e-4, False, False, "", False)


t_sum = 0
t_target = 0.1
current_dt = 1e-7
i = 0
i_dump = 0
while t_sum < t_target:

    print("step : t=",t_sum)
    
    next_dt = model.evolve(current_dt, True, "dump_"+str(i_dump)+".vtk", True)

    if i % 1 == 0:
        i_dump += 1

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum

    i+= 1