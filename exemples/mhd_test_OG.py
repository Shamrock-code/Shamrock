import shamrock
import matplotlib.pyplot as plt
import os
import numpy as np

directory = "/Users/ylapeyre/Documents/Shamwork/"
outputdir = "Alfven2/"

os.chdir(directory)

if not os.path.exists(outputdir):
    os.mkdir(outputdir)
    print(f"Directory '{directory}' created.")
    os.chdir(directory + outputdir)
    os.mkdir("shamrockdump")
    os.mkdir("phantomdump") 



gamma = 5./3.
rho_g = 1
target_tot_u = 1

C_cour = 0.3
C_force = 0.25

lambda_vel = 1
gamma_vel = 5/3
sina = 2./3.
sinb = 2. / np.sqrt(5)
cosa = np.sqrt(5) / 3.
cosb = 1 / np.sqrt(5)

L = 3.
bmin = (0, 0, 0)
bmax = (L, L/2, L/2)
xc,yc,zc = 0.,0.,0.
pmass = -1
wavelength = 1.

dr = 0.02

################################################
################# unit system ##################
################################################
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 1.,unit_length = 1., unit_mass = 1., )
ucte = shamrock.Constants(codeu)

mu_0 = ucte.mu_0()
ctx = shamrock.Context()
ctx.pdata_layout_new()

################################################
#################### config ####################
################################################

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
cfg.set_units(codeu)
cfg.set_artif_viscosity_None()
cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
#cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e6),1)

################################################
############### size of the box ################
################################################
bmin,bmax = model.get_ideal_fcc_box(dr,bmin,bmax)
xm,ym,zm = bmin
xM,yM,zM = bmax

model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho_g*vol_b)
#print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

################################################
############## initial conditions ##############
################################################
def rotated_basis_to_regular(xvec):
    
    reg_xyz = np.array([0., 0., 0.])
    reg_xyz[0] = xvec[0]*cosa*cosb - xvec[1]*sinb - xvec[2]*sina*cosb
    reg_xyz[1] = xvec[0]*cosa*sinb + xvec[1]*cosb - xvec[2]*sina*sinb
    reg_xyz[2] = xvec[0]*sina +      xvec[2]*cosa

    return reg_xyz

def regular_basis_to_rotated(xvec):
    
    rot_xyz = np.array([0., 0., 0.])
    rot_xyz[0] = (      xvec[0] + 2 * xvec[1] + 2 * xvec[2]) / 3.
    rot_xyz[1] = (- 2 * xvec[0] +     xvec[1]) / np.sqrt(5)
    rot_xyz[2] = (- 2 * xvec[0] - 4 * xvec[1] * 5 * xvec[2]) / (3 * np.sqrt(5))

    return rot_xyz
def B_func(r):

    rot_vec = regular_basis_to_rotated(r)
    x1, x2, x3 = rot_vec

    B1 = 1 *mu_0
    B2 = 0.1 * np.sin(2 * np.pi * x1 / lambda_vel) *mu_0
    B3 = 0.1 * np.cos(2 * np.pi * x1 / lambda_vel) *mu_0
    bvec = [B1, B2, B3]

    reg_bvec = rotated_basis_to_regular(bvec)
    Bx, By, Bz = reg_bvec
    return (Bx, By, Bz)

model.set_field_value_lambda_f64_3("B/rho", B_func)

def vel_func(r):

    rot_vec = regular_basis_to_rotated(r)
    x1, x2, x3 = rot_vec

    v1 = 0
    v2 = 0.1 * np.sin(2 * np.pi * x1 / lambda_vel)
    v3 = 0.1 * np.cos(2 * np.pi * x1 / lambda_vel)
    vvec = [v1, v2, v3]

    reg_vvec = rotated_basis_to_regular(vvec)
    vx, vy, vz = reg_vvec
    return (vx, vy, vz)

model.set_field_value_lambda_f64_3("vxyz", vel_func)

model.set_particle_mass(pmass)


tot_u = pmass*model.get_sum("uint","f64")

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

t_sum = 0
t_target = 50

i_dump = 0
dt_dump = 0.5
next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target:

    fname = directory + outputdir + "phantomdump/" + "dump_{:04}.phfile".format(i_dump)
    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    fnamesh= directory + outputdir + "shamrockdump/" + "dump_" + f"{i_dump:04}" + ".sham"
    model.dump(fnamesh)

    model.evolve_until(next_dt_target)
    

    i_dump += 1

    next_dt_target += dt_dump


