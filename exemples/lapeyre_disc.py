import shamrock
import matplotlib.pyplot as plt
import numpy as np

outputdir = '/Users/ylapeyre/Documents/Shamwork/08_04/test1/'
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
#cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_ConstantDisc(alpha_AV = 0, alpha_u = 0, beta_AV = 0)

#cfg.set_eos_locally_isothermalLP07(0.005,-2., 10.) #cs0 = 0.005, q = -2, r0 = 10
cfg.set_eos_locally_isothermal()
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)


bmin = (-10,-10,-10)
bmax = (10,10,10)
model.resize_simulation_box(bmin,bmax)

disc_mass = 0.001


#pmass = model.add_big_disc_3d(
#    (0,0,0),
#    1,
#    100000,
#    1, 
#    10,
#    disc_mass,
#    1.,
#    0.05,
#    1./4., 
#    200)

pmass = model.add_big_disc_3d(
    center      =(0,0,0),
    central_mass=1,
    Npart       =200000,
    r_in        =1, 
    r_out       =10,
    disc_mass   =disc_mass,
    p           =1.,
    H_r_in      =0.05,
    q           =1./4., 
    seed        =273, 
    do_warp     =True, 
    incl        =30., 
    posangle    =0., 
    Rwarp       =5, 
    Hwarp       =1)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

print("Current part mass :", pmass)

model.set_particle_mass(pmass)

model.add_sink(1,(0,0,0),(0,0,0),0.1)

dump = model.do_vtk_dump(outputdir+"initdump.vtk", False)

print("Run")

model.change_htolerance(1.3)
model.evolve_once_override_time(0,0)
model.change_htolerance(1.1)

print("Current part mass :", pmass)

#t_sum = 0
#t_target = 4e1
#i_dump = 0
#dt_dump = 50e-2
t_sum = 0
t_target = 100000

i_dump = 0
dt_dump = 0.5

next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target:

    fname = outputdir + "dump_{:04}".format(i_dump)
    fname_vtk = outputdir + "dump_{:04}.vtk".format(i_dump)

    model.evolve_until(next_dt_target)
    dump2 = model.do_vtk_dump(fname_vtk, False)

    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    i_dump += 1

    next_dt_target += dt_dump

