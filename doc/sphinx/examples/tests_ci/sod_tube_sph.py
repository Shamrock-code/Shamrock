"""
Testing Sod tube with SPH
=========================

CI test for Sod tube with SPH
"""

import matplotlib.pyplot as plt

import shamrock

gamma = 1.4

rho_g = 1
rho_d = 0.125

fact = (rho_g / rho_d) ** (1.0 / 3.0)

P_g = 1
P_d = 0.1

u_g = P_g / ((gamma - 1) * rho_g)
u_d = P_d / ((gamma - 1) * rho_d)

resol = 128

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8), 1)


(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)

model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))


setup = model.get_setup()
gen1 = setup.make_generator_lattice_hcp(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
gen2 = setup.make_generator_lattice_hcp(dr * fact, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
comb = setup.make_combiner_add(gen1, gen2)
# print(comb.get_dot())
setup.apply_setup(comb)

# model.add_cube_fcc_3d(dr, (-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
# model.add_cube_fcc_3d(dr*fact, (0,-ys/2,-zs/2),(xs,ys/2,zs/2))

model.set_value_in_a_box("uint", "f64", u_g, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_value_in_a_box("uint", "f64", u_d, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))


vol_b = xs * ys * zs

totmass = (rho_d * vol_b) + (rho_g * vol_b)

print("Total mass :", totmass)


pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)


model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

t_target = 0.245

model.evolve_until(t_target)

# model.evolve_once()

sod = shamrock.phys.SodTube(gamma=gamma, rho_1=1, P_1=1, rho_5=0.125, P_5=0.1)
sodanalysis = model.make_analysis_sodtube(sod, (1, 0, 0), t_target, 0.0, -0.5, 0.5)

#################
### Test CD
#################
rho, v, P = sodanalysis.compute_L2_dist()
vx, vy, vz = v

print("current errors :")
print(f"err_rho = {rho}")
print(f"err_vx = {vx}")
print(f"err_vy = {vy}")
print(f"err_vz = {vz}")
print(f"err_P = {P}")

# normally :
# rho 0.0001615491818848632
# v (0.0011627047434807855, 2.9881306160215856e-05, 1.7413547093275864e-07)
# P0.0001248364612976704

test_pass = True

expect_rho = 0.00016154918188486815
expect_vx = 0.001162704743480841
expect_vy = 2.988130616021184e-05
expect_vz = 1.7413547093230376e-07
expect_P = 0.00012483646129766217

tol = 1e-11


def float_equal(val1, val2, prec):
    return abs(val1 - val2) < prec


err_log = ""

if not float_equal(rho, expect_rho, tol * expect_rho):
    err_log += "error on rho is outside of tolerances:\n"
    err_log += f"  expected error = {expect_rho} +- {tol*expect_rho}\n"
    err_log += f"  obtained error = {rho} (relative error = {(rho - expect_rho)/expect_rho})\n"
    test_pass = False

if not float_equal(vx, expect_vx, tol * expect_vx):
    err_log += "error on vx is outside of tolerances:\n"
    err_log += f"  expected error = {expect_vx} +- {tol*expect_vx}\n"
    err_log += f"  obtained error = {vx} (relative error = {(vx - expect_vx)/expect_vx})\n"
    test_pass = False

if not float_equal(vy, expect_vy, tol * expect_vy):
    err_log += "error on vy is outside of tolerances:\n"
    err_log += f"  expected error = {expect_vy} +- {tol*expect_vy}\n"
    err_log += f"  obtained error = {vy} (relative error = {(vy - expect_vy)/expect_vy})\n"
    test_pass = False

if not float_equal(vz, expect_vz, tol * expect_vz):
    err_log += "error on vz is outside of tolerances:\n"
    err_log += f"  expected error = {expect_vz} +- {tol*expect_vz}\n"
    err_log += f"  obtained error = {vz} (relative error = {(vz - expect_vz)/expect_vz})\n"
    test_pass = False

if not float_equal(P, expect_P, tol * expect_P):
    err_log += "error on P is outside of tolerances:\n"
    err_log += f"  expected error = {expect_P} +- {tol*expect_P}\n"
    err_log += f"  obtained error = {P} (relative error = {(P - expect_P)/expect_P})\n"
    test_pass = False

if test_pass == False:
    exit("Test did not pass L2 margins : \n" + err_log)
