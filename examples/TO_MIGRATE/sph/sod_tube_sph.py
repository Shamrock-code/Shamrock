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


model.add_cube_fcc_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_fcc_3d(dr * fact, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

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

model.dump("outfile")

t_target = 0.245

model.evolve_until(t_target)

# model.evolve_once()

sod = shamrock.phys.SodTube(gamma=gamma, rho_1=1, P_1=1, rho_5=0.125, P_5=0.1)
sodanalysis = model.make_analysis_sodtube(sod, (1, 0, 0), t_target, 0.0, -0.5, 0.5)
print(sodanalysis.compute_L2_dist())


model.do_vtk_dump("end.vtk", True)
dump = model.make_phantom_dump()
dump.save_dump("end.phdump")

import numpy as np

dic = ctx.collect_data()

x = np.array(dic["xyz"][:, 0]) + 0.5
vx = dic["vxyz"][:, 0]
uint = dic["uint"][:]

hpart = dic["hpart"]
alpha = dic["alpha_AV"]

rho = pmass * (model.get_hfact() / hpart) ** 3
P = (gamma - 1) * rho * uint


plt.plot(x, rho, ".", label="rho")
plt.plot(x, vx, ".", label="v")
plt.plot(x, P, ".", label="P")
plt.plot(x, alpha, ".", label="alpha")
# plt.plot(x,hpart,'.',label="hpart")
# plt.plot(x,uint,'.',label="uint")


#### add analytical soluce
x = np.linspace(-0.5, 0.5, 1000)

rho = []
P = []
vx = []

for i in range(len(x)):
    x_ = x[i]

    _rho, _vx, _P = sod.get_value(t_target, x_)
    rho.append(_rho)
    vx.append(_vx)
    P.append(_P)

x += 0.5
plt.plot(x, rho, color="black", label="analytic")
plt.plot(x, vx, color="black")
plt.plot(x, P, color="black")
#######


plt.legend()
plt.grid()
plt.ylim(0, 1.1)
plt.xlim(0, 1)
plt.title("t=" + str(t_target))
plt.show()
