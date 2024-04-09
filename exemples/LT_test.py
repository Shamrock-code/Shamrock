import shamrock
import matplotlib.pyplot as plt
import numpy as np

outputdir = '/Users/ylapeyre/Documents/Shamwork/09_04/test1/'


si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 3600*24*365,unit_length = sicte.au(), unit_mass = sicte.sol_mass(), )
ucte = shamrock.Constants(codeu)


#central_mass = 1e6

#code u
central_mass = 1 
Rw = 1. #/ sicte.au() 
orbital_period = np.sqrt(4 * np.pi**2 * Rw**3 / central_mass) #physical units
print("################## orbital period = {} ##################".format(orbital_period))

ctx = shamrock.Context()
ctx.pdata_layout_new()


model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_ConstantDisc(alpha_AV = 0, alpha_u = 0, beta_AV = 0)

cfg.set_eos_locally_isothermal()
cfg.add_ext_force_lense_thirring(
    central_mass = central_mass,
    Racc = 0.1,
    a_spin = 0.9,
    dir_spin = (0,1,0) #align with BH spin
)
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)


bmin = (-10,-10,-10)
bmax = (10,10,10)
model.resize_simulation_box(bmin,bmax)

disc_mass = 0.001

pmass = model.add_big_disc_3d(
    center      =(0,0,0),
    central_mass=central_mass,
    Npart       =100000,
    r_in        =0.2, 
    r_out       =2,
    disc_mass   =disc_mass,
    p           =1.,
    H_r_in      =0.05,
    q           =1./4., 
    seed        =273, 
    do_warp     =True, 
    incl        =30., 
    posangle    =0., 
    Rwarp       =1., 
    Hwarp       =0.2)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

print("Current part mass :", pmass)

model.set_particle_mass(pmass)

#model.add_sink(1,(0,0,0),(0,0,0),0.1)
#model.add_sink(3*ucte.jupiter_mass(),(1,0,0),(0,0,vk_p),0.01)
#model.add_sink(100,(0,2,0),(0,0,1))

dump = model.make_phantom_dump()
dump.save_dump(outputdir + "initdump")
print("Run")

#model.change_htolerance(1.3)
#model.evolve_once_override_time(0,0)
model.change_htolerance(1.1)

print("Current part mass :", pmass)

t_sum = 0
t_target = 100000

i_dump = 0
dt_dump = 1
next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target:

    fname = outputdir + "dump_{:04}".format(i_dump)

    model.evolve_until(next_dt_target)
    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    i_dump += 1

    next_dt_target += dt_dump

