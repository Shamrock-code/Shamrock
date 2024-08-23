import shamrock


gamma = 5./3.
rho_g = 1
target_tot_u = 1

bmin = (-0.6,-0.6,-0.6)
bmax = ( 0.6, 0.6, 0.6)

N_target_base = 0.5e6
compute_multiplier = 2
scheduler_split_val = int(1e6)
scheduler_merge_val = int(1)




N_target = N_target_base*compute_multiplier
xm,ym,zm = bmin
xM,yM,zM = bmax
vol_b = (xM - xm)*(yM - ym)*(zM - zm)

part_vol = vol_b/N_target

#lattice volume
part_vol_lattice = 0.74*part_vol

dr = (part_vol_lattice / ((4./3.)*3.1416))**(1./3.)

pmass = -1




ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")
model.init_scheduler(scheduler_split_val,scheduler_merge_val)
bmin,bmax = model.get_ideal_fcc_box(dr,bmin,bmax)
xm,ym,zm = bmin
xM,yM,zM = bmax
model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)
xc,yc,zc = model.get_closest_part_to((0,0,0))
ctx.close_sched()
del model
del ctx


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

model.init_scheduler(scheduler_split_val,scheduler_merge_val)


bmin = (xm - xc,ym - yc, zm - zc)
bmax = (xM - xc,yM - yc, zM - zc)
xm,ym,zm = bmin
xM,yM,zM = bmax

model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho_g*vol_b)
#print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint","f64", 0 , bmin,bmax)

rinj = 0.008909042924642563*2/2
#rinj = 0.008909042924642563*2*2
#rinj = 0.01718181
u_inj = 1
model.add_kernel_value("uint","f64", u_inj,(0,0,0),rinj)



#print("Current part mass :", pmass)

#for it in range(5):
#    setup.update_smoothing_length(ctx)



#print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass*model.get_sum("uint","f64")
#print("total u :",tot_u)

#a = input("continue ?")



model.set_cfl_cour(0.01)
model.set_cfl_force(0.01)





#for i in range(9):
#    model.evolve(5e-4, False, False, "", False)


for i in range(5):
    model.evolve_once()


print("result rate :",model.solver_logs_last_rate())
print("result cnt :",model.solver_logs_last_obj_count())
