import shamrock
import matplotlib.pyplot as plt
import numpy as np

outputdir = "/home/ylapeyre/discs/test_lense_thirring/"
central_mass = 1e8

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 3600*24*365,unit_length = sicte.au(), unit_mass = sicte.sol_mass(), )
ucte = shamrock.Constants(codeu)


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0,alpha_max = 1,sigma_decay = 0, alpha_u = 0, beta_AV = 0)
cfg.set_artif_viscosity_ConstantDisc(alpha_AV = 0.001, alpha_u = 0, beta_AV = 0)
#cfg.add_ext_force_lense_thrirring(
#    central_mass,
#    0.1,
#    0.2,
#    (np.sin(3 * np.pi / 18),np.sin(3 * np.pi / 18),0)
#)

cfg.set_eos_locally_isothermal()
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)


bmin = (-10,-10,-10)
bmax = (10,10,10)
model.resize_simulation_box(bmin,bmax)

#cfg.set_eos_adiabatic(1) #to have cs 0

disc_mass = 0.001

pmass = model.add_disc_3d(
    (0,0,0),
    1,
    1000000,
    0.2,3,
    disc_mass,
    1.,
    0.000001, # H/r = cs = epsilon
    1./4.)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

print("Current part mass :", pmass)

model.set_particle_mass(pmass)

print("Small timestep")
model.evolve(0,1e-7, False, "", False)

print("Plot timestep")

print("Run")

t_sum = 0
t_target = 100
current_dt = 1e-7
i = 0
i_dump = 0
while t_sum < t_target:

    print("step : t=",t_sum)

    do_dump = (i % 50 == 0)  
    next_dt = model.evolve(t_sum,current_dt, do_dump, outputdir + "dump_"+str(i_dump)+".vtk", do_dump)

    if i % 50 == 0:
        i_dump += 1

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum

    i+= 1
