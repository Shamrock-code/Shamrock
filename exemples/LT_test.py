import shamrock
import matplotlib.pyplot as plt
import numpy as np

outputdir = '/Users/ylapeyre/Documents/Shamwork/01_05/test3/'
central_mass = 1

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = sicte.G()*sicte.sol_mass()/(sicte.c()**2) /sicte.c(),unit_length = sicte.G()*sicte.sol_mass()/(sicte.c()**2), unit_mass = sicte.sol_mass(), )
ucte = shamrock.Constants(codeu)


#central_mass = 1e6

#code u 
G = ucte.G()
c_lum = ucte.c()
gm_c2 = G * central_mass / (c_lum**2)

print("################## GM/c2 = {} ##################".format(gm_c2))
print("################## G = {} ##################".format(G))
print("################## c = {} ##################".format(c_lum))
Rin = 4 * gm_c2 #/ sicte.au() 
Rout = 10 * Rin

orbital_period = 2 * np.pi * np.sqrt((Rin**3) / (central_mass * G)) #physical units
print("################## orbital period = {} ##################".format(orbital_period))



ctx = shamrock.Context()
ctx.pdata_layout_new()


model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_ConstantDisc(alpha_AV = 0, alpha_u = 0, beta_AV = 0)

cfg.set_eos_locally_isothermal()
cfg.add_ext_force_lense_thirring(
    central_mass = central_mass,
    Racc = gm_c2 *2,
    a_spin = 0.9,
    dir_spin = (0,0,1) #align with BH spin
)
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)


bmin = (-100,-100,-100)
bmax = (100,100,100)
model.resize_simulation_box(bmin,bmax)

disc_mass = 0.001

pmass = model.add_big_disc_3d(
    center      =(0,0,0),
    central_mass=central_mass,
    Npart       =50000,
    r_in        =Rin, 
    r_out       =Rout,
    disc_mass   =disc_mass,
    p           =1.,
    H_r_in      =0.05,
    q           =1./4., 
    seed        =273, 
    do_warp     =True, 
    incl        =30., 
    posangle    =0., 
    Rwarp       =Rin, 
    Hwarp       =Rout *4)

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
t_target = 100 * orbital_period

i_dump = 0
dt_dump = orbital_period / 10.
next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target:

    fname = outputdir + "dump_{:04}".format(i_dump)

    model.evolve_until(next_dt_target)
    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    i_dump += 1

    next_dt_target += dt_dump

