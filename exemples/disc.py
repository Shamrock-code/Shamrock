import shamrock
import matplotlib.pyplot as plt



si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 3600*24*365,unit_lenght = sicte.au(), unit_mass = sicte.sol_mass(), )
ucte = shamrock.Constants(codeu)

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
cfg.set_units(codeu)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)


bmin = (-10,-10,-10)
bmax = (10,10,10)
model.resize_simulation_box(bmin,bmax)

model.set_eos_gamma(5/3)

model.add_disc_3d_keplerian((0,0,0),200000,0.5,1,1,0.2,3,0.05,1)



pmass = model.total_mass_to_part_mass(0.001)

print("Current part mass :", pmass)

#for it in range(5):
#    setup.update_smoothing_lenght(ctx)


model.add_sink(1,(0,0,0),(0,0,0),0.05)
model.add_sink(3*ucte.jupiter_mass(),(1,0,0),(0,0,6.5),0.01)
#model.add_sink(100,(0,2,0),(0,0,1))



model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)





print("Current part mass :", pmass)


model.set_particle_mass(pmass)

#for i in range(9):
#    model.evolve(5e-4, False, False, "", False)


t_sum = 0
t_target = 10
current_dt = 1e-7
i = 0
i_dump = 0
while t_sum < t_target:

    print("step : t=",t_sum)

    do_dump = (i % 10 == 0)  
    next_dt = model.evolve(t_sum,current_dt, do_dump, "dump_"+str(i_dump)+".vtk", do_dump)

    if i % 10 == 0:
        i_dump += 1

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum

    i+= 1