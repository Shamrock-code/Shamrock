import matplotlib.pyplot as plt

import shamrock

gamma = 1.4

rho_gas_g = 1
rho_gas_d = 0.125

epsilon_start = 0.3
deltavx_start = 0.0

fact = (rho_gas_g / rho_gas_d) ** (1.0 / 3.0)

P_g = 1
P_d = 0.1

u_g = P_g / ((gamma - 1) * rho_gas_g)
u_d = P_d / ((gamma - 1) * rho_gas_d)

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
# cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 0.0,sigma_decay = 0.1, alpha_u = 0, beta_AV = 0)
cfg.set_dust_mode_monofluid_complete(ndust=1)
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

model.set_value_in_a_box(
    "epsilon", "f64", epsilon_start, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2)
)
model.set_value_in_a_box(
    "deltav", "f64_3", (deltavx_start, 0, 0), (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2)
)


vol_b = xs * ys * zs

totmass = (rho_gas_d * vol_b) + (rho_gas_g * vol_b)

print("Total mass :", totmass)


pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)


model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

model.dump("outfile")

t_target = 0.02
print("Target time :", t_target)

# model.evolve_until(t_target)

cnt = 0


def analyse():
    global cnt

    sod = shamrock.phys.SodTube(
        gamma=gamma,
        rho_1=1 * (1 - epsilon_start),
        P_1=1 * (1 - epsilon_start),
        rho_5=0.125 * (1 - epsilon_start),
        P_5=0.1 * (1 - epsilon_start),
    )

    sodanalysis = model.make_analysis_sodtube(sod, (1, 0, 0), model.get_time(), 0.0, -0.5, 0.5)
    print(sodanalysis.compute_L2_dist())

    model.do_vtk_dump("end.vtk", True)
    dump = model.make_phantom_dump()
    dump.save_dump("end.phdump")

    import numpy as np

    dic = ctx.collect_data()

    x = np.array(dic["xyz"][:, 0]) + 0.5
    vx = dic["vxyz"][:, 0]
    uint = dic["uint"][:]

    deltavx = dic["deltav"][:, 0]
    epsilon = dic["epsilon"][:]

    dtdeltavx = dic["dtdeltav"][:, 0]
    dtepsilon = dic["dtepsilon"][:]

    hpart = dic["hpart"]
    alpha = dic["alpha_AV"]

    rho = pmass * (model.get_hfact() / hpart) ** 3
    P = (gamma - 1) * rho * uint

    rhog = (1 - epsilon) * rho
    rhod = epsilon * rho
    Pg = (gamma - 1) * rhog * uint
    vg = vx - epsilon * deltavx
    vd = vx + (1 - epsilon) * deltavx

    vg_init = 0 - epsilon_start * deltavx_start

    fig, axs = plt.subplots(3, 3, figsize=(12, 8), dpi=125)

    axs[0, 0].plot(x, rhog, ".", label="rhog")
    axs[0, 0].plot(x, rhod, ".", label="rhod")

    axs[0, 1].plot(x, vg, ".", label="vg")
    axs[0, 1].plot(x, vd, ".", label="vd")

    axs[0, 2].plot(x, Pg, ".", label="P")

    axs[1, 0].plot(x, rho, ".", label="rho")
    axs[1, 1].plot(x, deltavx, ".", label="deltavx")
    axs[1, 2].plot(x, epsilon, ".", label="epsilon")

    axs[2, 0].plot(x, vx, ".", label="v")

    axs[2, 1].plot(x, uint, ".", label="uint")

    axs[2, 2].plot(x, alpha, ".", label="alpha")
    # plt.plot(x,hpart,'.',label="hpart")
    # plt.plot(x,uint,'.',label="uint")

    #### add analytical soluce
    x = np.linspace(-0.5, 0.5, 1000)

    rho_g = []
    rho_d = []
    P = []
    vx_g = []
    vx_d = []

    for i in range(len(x)):
        x_ = x[i]

        # offset = deltavx_g_start

        _rho_d = 0
        _vx_d = 0
        if x_ < 0:
            _rho_d = rho_gas_g * (epsilon_start)
        else:
            _rho_d = rho_gas_d * (epsilon_start)

        _rho_g, _vx_g, _P = sod.get_value(model.get_time(), x_ - vg_init * model.get_time())
        rho_g.append(_rho_g)
        rho_d.append(_rho_d)
        vx_g.append(_vx_g + vg_init)
        vx_d.append(_vx_d)
        P.append(_P)

    x += 0.5
    axs[0, 0].plot(x, rho_g, color="black", label="analytic")
    axs[0, 0].plot(x, rho_d, color="black", label="analytic")
    axs[0, 1].plot(x, vx_g, color="black", label="analytic")
    axs[0, 2].plot(x, P, color="black", label="analytic")

    axs[1, 0].plot(x, np.array(rho_g) + np.array(rho_d), color="black", label="analytic")
    axs[1, 1].plot(x, np.array(vx_d) - np.array(vx_g), color="black", label="analytic")
    axs[1, 2].plot(
        x, np.array(rho_d) / (np.array(rho_g) + np.array(rho_d)), color="black", label="analytic"
    )
    axs[2, 1].plot(
        x, np.array(P) / (np.array(rho_g) * (gamma - 1)), color="black", label="analytic"
    )
    axs[2, 0].plot(
        x,
        (np.array(vx_g) * np.array(rho_g) + np.array(vx_d) * np.array(rho_d))
        / (np.array(rho_g) + np.array(rho_d)),
        color="black",
        label="analytic",
    )

    #######

    # enable grid on all axis
    for ax1 in axs:
        for ax2 in ax1:
            ax2.grid()
            ax2.legend()
            ax2.set_xlim(0, 1)

    axs[0, 0].set_ylim(-0.2, 1.1)
    axs[0, 1].set_ylim(-0.2, 1.1)
    axs[0, 2].set_ylim(-0.2, 1.1)
    axs[1, 0].set_ylim(-0.2, 1.1)
    axs[1, 1].set_ylim(-1.1, 1.1)
    axs[1, 2].set_ylim(0, 0.5)
    axs[2, 0].set_ylim(-0.2, 1.1)
    axs[2, 1].set_ylim(1, 3)
    axs[2, 2].set_ylim(-0.2, 1.1)

    fig.suptitle("t=" + str(model.get_time()))
    plt.tight_layout()
    plt.savefig(f"dusty_sod_{cnt}.png")
    cnt += 1
    # plt.show()
    # clear completely the plot
    plt.close(fig)


model.evolve_once()

for i in range(400):
    analyse()
    for i in range(10):
        model.evolve_once()

model.evolve_until(t_target)
analyse()
