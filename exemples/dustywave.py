import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

###########################################
# PARAMS
###########################################
gamma = 1.4

rho_g = 1
rho_d = 1
epsilon = rho_d / (rho_g + rho_d)

delta_vx_g = 1e-3
delta_vx_d = 0.0

P_g = 1
u_g = P_g / ((gamma - 1) * rho_g)
cs_g = (gamma * P_g / rho_g) ** 0.5

k = 2 * np.pi * 2
omega = cs_g * k
T = 2 * np.pi / omega

kx = k
ky = 0
kz = 0


def func_vg(r):
    x, y, z = r
    return (0 + delta_vx_g * np.cos(kx * x + ky * y + kz * z), 0, 0)


def func_vd(r):
    x, y, z = r
    return (0 + delta_vx_d * np.cos(kx * x + ky * y + kz * z), 0, 0)


def vxyz(r):
    x, y, z = r

    vgx, vgy, vgz = func_vg(r)
    vdx, vdy, vdz = func_vd(r)

    vx = (rho_g * vgx + rho_d * vdx) / (rho_g + rho_d)
    vy = (rho_g * vgy + rho_d * vdy) / (rho_g + rho_d)
    vz = (rho_g * vgz + rho_d * vdz) / (rho_g + rho_d)

    return (vx, vy, vz)


def deltavxyz(r):
    x, y, z = r

    vgx, vgy, vgz = func_vg(r)
    vdx, vdy, vdz = func_vd(r)

    dvx = vdx - vgx
    dvy = vdy - vgy
    dvz = vdz - vgz

    return (dvx, dvy, dvz)


resol = 128
###########################################
# Sim
###########################################

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=0.0, sigma_decay=0.1, alpha_u=0, beta_AV=0
)
cfg.set_dust_mode_monofluid_complete(ndust=1)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8), 1)


(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)

model.resize_simulation_box((-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))


model.add_cube_fcc_3d(dr, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))

model.set_value_in_a_box("uint", "f64", u_g, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))
model.set_value_in_a_box(
    "epsilon", "f64", epsilon, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2)
)

model.set_field_value_lambda_f64_3("vxyz", vxyz)
model.set_field_value_lambda_f64_3("deltav", deltavxyz)


vol_b = xs * ys * zs

totmass = (rho_d * vol_b) + (rho_g * vol_b)

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

epsilon_init = epsilon
rhog_init = rho_g
rhod_init = rho_d

cnt = 0


def analyse():
    global cnt

    model.do_vtk_dump("end.vtk", True)
    dump = model.make_phantom_dump()
    dump.save_dump("end.phdump")

    import numpy as np

    dic = ctx.collect_data()

    x = np.array(dic["xyz"][:, 0])
    vx = dic["vxyz"][:, 0]
    ax = dic["axyz"][:, 0]
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

    fig, axs = plt.subplots(3, 3, figsize=(12, 8), dpi=125)

    axs[0, 0].plot(x, rhog, ".", label="rhog")
    axs[0, 1].plot(x, rhod, ".", label="rhod")
    axs[0, 2].plot(x, epsilon, ".", label="epsilon")
    axs[1, 0].plot(x, vg, ".", label="vg")
    axs[1, 1].plot(x, vd, ".", label="vd")
    axs[1, 2].plot(x, deltavx, ".", label="deltavx")
    axs[2, 0].plot(x, dtepsilon, ".", label="dtepsilon")
    axs[2, 1].plot(x, dtdeltavx, ".", label="dtdeltavx")
    axs[2, 2].plot(x, ax, ".", label="dtv")

    # axs[0,1].plot(x,vx,'.',label="v")
    # axs[0,1].plot(x,uint,'.',label="uint")
    # axs[0,1].plot(x,epsilon,'.',label="epsilon")

    # axs[1,0].plot(x,epsilon,'.',label="epsilon")
    # axs[1,1].plot(x,deltavx,'.',label="deltavx")
    # plt.plot(x,hpart,'.',label="hpart")
    # plt.plot(x,uint,'.',label="uint")

    #### add analytical soluce
    x = np.linspace(-0.5, 0.5, 1000)

    anal_rhog = []
    anal_rhod = []
    anal_vxg = []
    anal_vxd = []
    anal_epsilon = []
    anal_delta_vx = []

    anal_dtepsilon = []
    anal_dtdeltavx = []
    anal_ax = []

    for i in range(len(x)):
        x_ = x[i]

        # offset = deltavx_start
        t_ = model.get_time()

        _rho_g = rho_g + delta_vx_g * np.sin(kx * x_) * np.sin(omega * t_)
        _dtrho_g = delta_vx_g * omega * np.sin(kx * x_) * np.cos(omega * t_)

        _vx_g = delta_vx_g * np.cos(kx * x_) * np.cos(omega * t_)
        _ax_g = -delta_vx_g * omega * np.cos(kx * x_) * np.sin(omega * t_)

        _P = rho_g * np.cos(kx * x_) * np.sin(omega * t_)

        _rho_d = rho_d
        _dtrho_d = 0

        _vx_d = 0
        _ax_d = 0

        _epsilon = _rho_d / (_rho_g + _rho_d)
        _deltav = _vx_d - _vx_g

        _dtepsilon = (_rho_g * _dtrho_d - _rho_d * _dtrho_g) / ((_rho_g + _rho_d) ** 2)
        _dtdeltavx = (
            -_dtepsilon * _vx_g + (1 - _epsilon) * _ax_g + _dtepsilon * _vx_d + _epsilon * _ax_d
        )

        anal_rhog.append(_rho_g)
        anal_rhod.append(_rho_d)
        anal_vxg.append(_vx_g)
        anal_vxd.append(_vx_d)
        anal_epsilon.append(_epsilon)
        anal_delta_vx.append(_deltav)

        anal_dtepsilon.append(_dtepsilon)
        anal_dtdeltavx.append(_ax_d - _ax_g)
        anal_ax.append(_dtdeltavx)

    axs[0, 0].plot(x, anal_rhog, color="black", label="analytic")
    axs[0, 1].plot(x, anal_rhod, color="black", label="analytic")
    axs[0, 2].plot(x, anal_epsilon, color="black", label="analytic")

    axs[1, 0].plot(x, anal_vxg, color="black", label="analytic")
    axs[1, 1].plot(x, anal_vxd, color="black", label="analytic")
    axs[1, 2].plot(x, anal_delta_vx, color="black", label="analytic")

    axs[2, 0].plot(x, anal_dtepsilon, color="black", label="analytic")
    axs[2, 1].plot(x, anal_dtdeltavx, color="black", label="analytic")
    axs[2, 2].plot(x, anal_ax, color="black", label="analytic")

    #######

    # enable grid on all axis
    for ax1 in axs:
        for ax2 in ax1:
            ax2.grid()
            ax2.legend()
            ax2.set_xlim(-0.5, 0.5)

    print(rhod_init, rhod)
    axs[0, 0].set_ylim(-1.2e-3 * rhog_init + rhog_init, 1.2e-3 * rhog_init + rhog_init)
    axs[0, 1].set_ylim(-1.2e-3 * rhod_init + rhod_init, 1.2e-3 * rhod_init + rhod_init)
    axs[0, 2].set_ylim(-1.2e-3 * epsilon_init + epsilon_init, 1.2e-3 * epsilon_init + epsilon_init)

    axs[1, 0].set_ylim(-1.2e-3, 1.2e-3)
    axs[1, 1].set_ylim(-1.2e-3, 1.2e-3)
    axs[1, 2].set_ylim(-1.2e-3, 1.2e-3)

    axs[2, 0].set_ylim(-1.2e-3 * omega * epsilon_init, 1.2e-3 * omega * epsilon_init)
    axs[2, 1].set_ylim(-1.2e-3 * omega, 1.2e-3 * omega)
    axs[2, 2].set_ylim(-1.2e-3 * omega, 1.2e-3 * omega)

    # axs[2,2].set_ylim(-1.2e-3+P_g,1.2e-3+P_g)

    fig.suptitle("t/T=" + str(model.get_time() / T))
    plt.tight_layout()
    plt.savefig(f"dustywave_{cnt}.png")
    print(f"writing : dustywave_{cnt}.png (t/T={model.get_time()/T})")
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
