import shamrock
import numpy as np
import matplotlib.pyplot as plt
import os


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_AMRGodunov(
    context = ctx,
    vector_type = "f64_3",
    grid_repr = "i64_3")

model.init_scheduler(int(1e7),1)

multx = 4
multy = 1
multz = 1

sz = 1 << 1
base = 32
model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

cfg = model.gen_default_config()
scale_fact = 2/(sz*base*multx)
cfg.set_scale_factor(scale_fact)

gamma = 1.4
cfg.set_eos_gamma(gamma)
#cfg.set_riemann_solver_rusanov()
cfg.set_riemann_solver_hll()

#cfg.set_slope_lim_none()
#cfg.set_slope_lim_vanleer_f()
#cfg.set_slope_lim_vanleer_std()
#cfg.set_slope_lim_vanleer_sym()
cfg.set_slope_lim_minmod()
model.set_config(cfg)


kx,ky,kz = 2*np.pi,0,0
delta_rho = 1e-2

def rho_map(rmin,rmax):

    x,y,z = rmin
    if x < 1:
        return 1
    else:
        return 0.125


etot_L = 1./(gamma-1)
etot_R = 0.1/(gamma-1)

def rhoetot_map(rmin,rmax):

    rho = rho_map(rmin,rmax)

    x,y,z = rmin
    if x < 1:
        return etot_L
    else:
        return etot_R

def rhovel_map(rmin,rmax):
    rho = rho_map(rmin,rmax)

    return (0,0,0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

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
