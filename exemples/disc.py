import shamrock
import matplotlib.pyplot as plt


gamma = 5./3.
rho_g = 1
target_tot_u = 1


pmass = -1




ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)


bmin = (-10,-10,-10)
bmax = (10,10,10)
model.resize_simulation_box(bmin,bmax)

model.add_cube_disc_3d((0,0,0),100000,0.5,1,1,0.1,3,0.05)
model.set_value_in_a_box("uint", "f64", 1, bmin,bmax)



pmass = model.total_mass_to_part_mass(1)

print("Current part mass :", pmass)

#for it in range(5):
#    setup.update_smoothing_lenght(ctx)


model.add_sink(10,(0,0,0),(0,0,0))
model.add_sink(10,(0,1,0),(0,0,1))



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
    
    next_dt = model.evolve(t_sum,current_dt, True, "dump_"+str(i_dump)+".vtk", True)

    if i % 1 == 0:
        i_dump += 1

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum

    i+= 1