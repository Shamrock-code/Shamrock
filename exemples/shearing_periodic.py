import shamrock
import matplotlib.pyplot as plt

gamma = 5./3.
rho_g = 1
target_tot_u = 1


dr = 0.01
bmin = (-0.6,-0.1,-0.6)
bmax = ( 0.6, 0.1, 0.6)
pmass = -1


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_boundary_shearing_periodic((1,0,0),(0,0,1),10.)
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
#print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint","f64", 1 , bmin,bmax)
#model.set_value_in_a_box("vxyz","f64_3", (-10,0,0) , bmin,bmax)

pen_sz = 0.1

model.set_value_in_a_box("uint","f64", 3 , (xm,-pen_sz,-pen_sz),(-0.2,pen_sz,pen_sz))
model.set_value_in_a_box("uint","f64", 3 , (0.2,-pen_sz,-pen_sz),(xM,pen_sz,pen_sz))

#print("Current part mass :", pmass)

#for it in range(5):
#    setup.update_smoothing_length(ctx)



#print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass*model.get_sum("uint","f64")
#print("total u :",tot_u)

a = input("continue ?")



model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)





#for i in range(9):
#    model.evolve(5e-4, False, False, "", False)


t_sum = 0
t_target = 10

i_dump = 0
dt_dump = 1e-2
next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target:

    fname = "dump_{:04}.phfile".format(i_dump)

    model.evolve_until(next_dt_target)
    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    i_dump += 1

    next_dt_target += dt_dump
