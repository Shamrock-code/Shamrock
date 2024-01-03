import shamrock
import matplotlib.pyplot as plt


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")



gamma = 5./3.
rho = 1
uint = 1

dr = 0.01
bmin = (-0.6,-0.6,-0.1)
bmax = ( 0.6, 0.6, 0.1)
pmass = -1

bmin,bmax = model.get_ideal_fcc_box(dr,bmin,bmax)
xm,ym,zm = bmin
xM,yM,zM = bmax

Omega_0 = 1
eta = 0.00
q = 3./2.

shear_speed = -q*Omega_0*(xM - xm)


cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_boundary_shearing_periodic((1,0,0),(0,1,0),shear_speed)
cfg.set_eos_adiabatic(gamma)
cfg.add_ext_force_shearing_box(
    Omega_0  = Omega_0,
    eta      = eta,
    q        = q
)
cfg.set_units(shamrock.UnitSystem())
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)



model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho*vol_b)
#print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint","f64", 1 , bmin,bmax)
#model.set_value_in_a_box("vxyz","f64_3", (-10,0,0) , bmin,bmax)

pen_sz = 0.1

mm = 1
MM = 0

def vel_func(r):
    global mm, MM
    x,y,z = r

    s = (x - (xM + xm)/2)/(xM - xm)
    vel = (shear_speed)*s

    mm = min(mm,vel)
    MM = max(MM,vel)

    return (0,vel,0.)
    #return (1,0,0)

model.set_field_value_lambda_f64_3("vxyz", vel_func)
#print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass*model.get_sum("uint","f64")
#print("total u :",tot_u)

print(f"v_shear = {shear_speed} | dv = {MM-mm}")

a = input("continue ?")



model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)





#for i in range(9):
#    model.evolve(5e-4, False, False, "", False)

current_dt = model.evolve_once_override_time(0,0)



dump = model.make_phantom_dump()
fname = "dump_phinit"
dump.save_dump(fname)


t_sum = 0
t_target = 10

i_dump = 1
dt_dump = 1./100

do_dump = False
next_dt_target = t_sum + dt_dump


while t_sum <= next_dt_target:

    fname = "dump_{:04}.phfile".format(i_dump)

    model.evolve_until(next_dt_target)
    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    i_dump += 1

    next_dt_target += dt_dump