import os

import matplotlib.pyplot as plt

import shamrock

gamma = 5.0 / 3.0
rho_g = 1
target_tot_u = 1


dr = 0.01

bmin = (-0.6, -0.6, -0.6)
bmax = (0.6, 0.6, 0.6)
pmass = -1


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

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

model.init_scheduler(int(1e5), 1)


bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

model.resize_simulation_box(bmin, bmax)

# model.add_cube_hcp_3d_v2(dr, bmin,bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
setup.apply_setup(gen)

xc, yc, zc = model.get_closest_part_to((0, 0, 0))
print("closest part to (0,0,0) is in :", xc, yc, zc)

vol_b = (xM - xm) * (yM - ym) * (zM - zm)

totmass = rho_g * vol_b
# print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint", "f64", 0, bmin, bmax)

rinj = 0.008909042924642563 * 2
# rinj = 0.008909042924642563*2*2
# rinj = 0.01718181
u_inj = 1
model.add_kernel_value("uint", "f64", u_inj, (0, 0, 0), rinj)


tot_u = pmass * model.get_sum("uint", "f64")
print("total u :", tot_u)


# print("Current part mass :", pmass)

# for it in range(5):
#    setup.update_smoothing_length(ctx)


# print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass * model.get_sum("uint", "f64")

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

model.timestep()
model.do_vtk_dump("init.vtk", True)
model.dump("outfile")

t_target = 0.1
model.evolve_until(t_target)


model.do_vtk_dump("end.vtk", True)


import numpy as np

dic = ctx.collect_data()


if shamrock.sys.world_rank() == 0:
    r = np.sqrt(dic["xyz"][:, 0] ** 2 + dic["xyz"][:, 1] ** 2 + dic["xyz"][:, 2] ** 2)
    vr = np.sqrt(dic["vxyz"][:, 0] ** 2 + dic["vxyz"][:, 1] ** 2 + dic["vxyz"][:, 2] ** 2)

    hpart = dic["hpart"]
    uint = dic["uint"]

    gamma = 5.0 / 3.0

    rho = pmass * (model.get_hfact() / hpart) ** 3
    P = (gamma - 1) * rho * uint

    sedov_sol = shamrock.phys.SedovTaylor()

    r_theo = np.linspace(0, 1, 300)
    p_theo = []
    vr_theo = []
    rho_theo = []
    for i in range(len(r_theo)):
        _rho_theo, _vr_theo, _p_theo = sedov_sol.get_value(r_theo[i])
        p_theo.append(_p_theo)
        vr_theo.append(_vr_theo)
        rho_theo.append(_rho_theo)

    r_theo = np.array(r_theo)
    p_theo = np.array(p_theo)
    vr_theo = np.array(vr_theo)
    rho_theo = np.array(rho_theo)

    plt.style.use("custom_style.mplstyle")
    if True:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 6), dpi=125)

        axs[0, 0].scatter(r, vr, c="black", s=1, label="v", rasterized=True)
        axs[0, 0].plot(r_theo, vr_theo, c="red", label="v (theory)")
        axs[1, 0].scatter(r, uint, c="black", s=1, label="u", rasterized=True)
        axs[0, 1].scatter(r, rho, c="black", s=1, label="rho", rasterized=True)
        axs[0, 1].plot(r_theo, rho_theo, c="red", label="rho (theory)")
        axs[1, 1].scatter(r, P, c="black", s=1, label="P", rasterized=True)
        axs[1, 1].plot(r_theo, p_theo, c="red", label="P (theory)")

        axs[0, 0].set_ylabel(r"$v$")
        axs[1, 0].set_ylabel(r"$u$")
        axs[0, 1].set_ylabel(r"$\rho$")
        axs[1, 1].set_ylabel(r"$P$")

        axs[0, 0].set_xlabel("$r$")
        axs[1, 0].set_xlabel("$r$")
        axs[0, 1].set_xlabel("$r$")
        axs[1, 1].set_xlabel("$r$")

        axs[0, 0].set_xlim(0, 0.55)
        axs[1, 0].set_xlim(0, 0.55)
        axs[0, 1].set_xlim(0, 0.55)
        axs[1, 1].set_xlim(0, 0.55)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 3), dpi=125)

        axs.scatter(r, rho, c="black", s=1, label="rho", rasterized=True)
        axs.plot(r_theo, rho_theo, c="red", label="rho (theory)")

        axs.set_ylabel(r"$\rho$")

        axs.set_xlabel("$r$")

        axs.set_xlim(0, 0.55)

    plt.tight_layout()
    plt.show()
