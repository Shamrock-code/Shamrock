import shamrock
import matplotlib.pyplot as plt

gamma = 1.4

rho_g = 1
rho_d = 0.125

fact = (rho_g/rho_d)**(1./3.)

P_g = 1
P_d = 0.1

u_g = P_g/((gamma - 1)*rho_g)
u_d = P_d/((gamma - 1)*rho_d)

resol = 128

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

model.init_scheduler(int(1e8),1)


(xs,ys,zs) = model.get_box_dim_fcc_3d(1,resol,24,24)
dr = 1/xs
(xs,ys,zs) = model.get_box_dim_fcc_3d(dr,resol,24,24)

model.resize_simulation_box((-xs,-ys/2,-zs/2),(xs,ys/2,zs/2))


model.add_cube_fcc_3d(dr, (-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
model.add_cube_fcc_3d(dr*fact, (0,-ys/2,-zs/2),(xs,ys/2,zs/2))

model.set_value_in_a_box("uint", "f64", u_g ,(-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
model.set_value_in_a_box("uint", "f64", u_d ,(0,-ys/2,-zs/2),(xs,ys/2,zs/2))



vol_b = xs*ys*zs

totmass = (rho_d*vol_b) + (rho_g*vol_b)

print("Total mass :", totmass)


pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)



model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

t_target = 0.245

model.evolve_until(t_target)

#model.evolve_once()

sod = shamrock.phys.SodTube(gamma = gamma, rho_1 = 1,P_1 = 1,rho_5 = 0.125,P_5 = 0.1)
sodanalysis = model.make_analysis_sodtube(sod, (1,0,0), t_target, 0.0, -0.5,0.5)

#################
### Test CD
#################
rho, v, P = sodanalysis.compute_L2_dist()
vx,vy,vz = v

# normally : 
# rho 0.0001615491818848632
# v (0.0011627047434807855, 2.9881306160215856e-05, 1.7413547093275864e-07)
# P0.0001248364612976704

test_pass = True
pass_rho = 0.0001615491818848697
pass_vx = 0.0011627047434809158
pass_vy = 2.9881306160215856e-05
pass_vz = 1.7413547093275864e-07
pass_P = 0.0001248364612976704

err_log = ""

if rho > pass_rho:
    err_log += ("error on rho is too high "+str(rho) +">"+str(pass_rho) ) + "\n"
    test_pass = False
if vx > pass_vx:
    err_log += ("error on vx is too high "+str(vx) +">"+str(pass_vx) )+ "\n"
    test_pass = False
if vy > pass_vy:
    err_log += ("error on vy is too high "+str(vy) +">"+str(pass_vy) )+ "\n"
    test_pass = False
if vz > pass_vz:
    err_log += ("error on vz is too high "+str(vz) +">"+str(pass_vz) )+ "\n"
    test_pass = False
if P > pass_P:
    err_log += ("error on P is too high "+str(P) +">"+str(pass_P) )+ "\n"
    test_pass = False

if test_pass == False:
    exit("Test did not pass L2 margins : \n"+err_log)