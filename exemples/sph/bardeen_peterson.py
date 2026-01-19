import matplotlib.pyplot as plt
import numpy as np

import shamrock

central_mass = 1e6


si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=1,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_eos_locally_isothermal()
cfg.add_ext_force_lense_thirring(
    central_mass=central_mass, Racc=0.1, a_spin=0.9, dir_spin=(3**0.5 / 2, 1 / 2, 0)
)
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)


bmin = (-10, -10, -10)
bmax = (10, 10, 10)
model.resize_simulation_box(bmin, bmax)

disc_mass = 0.001

pmass = model.add_disc_3d((0, 0, 0), central_mass, 200000, 0.2, 2, disc_mass, 1.0, 0.01, 1.0 / 4.0)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

print("Current part mass :", pmass)

model.set_particle_mass(pmass)


# model.add_sink(1,(0,0,0),(0,0,0),0.1)

vk_p = (ucte.G() * 1 / 1) ** 0.5
# model.add_sink(3*ucte.jupiter_mass(),(1,0,0),(0,0,vk_p),0.01)
# model.add_sink(100,(0,2,0),(0,0,1))


def compute_rho(h):
    return np.array([model.rho_h(h[i]) for i in range(len(h))])


def plot_vertical_profile(r, rrange, label=""):
    data = ctx.collect_data()

    rhosel = []
    ysel = []

    for i in range(len(data["hpart"][:])):
        rcy = data["xyz"][i, 0] ** 2 + data["xyz"][i, 2] ** 2

        if rcy > r - rrange and rcy < r + rrange:
            rhosel.append(model.rho_h(data["hpart"][i]))
            ysel.append(data["xyz"][i, 1])

    rhosel = np.array(rhosel)
    ysel = np.array(ysel)

    rhobar = np.mean(rhosel)

    plt.scatter(ysel, rhosel / rhobar, s=1, label=label)


print("Run")


print("Current part mass :", pmass)

t_sum = 0
t_target = 100000

i_dump = 0
dt_dump = 100
next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target:
    fname = "dump_{:04}.phfile".format(i_dump)

    model.evolve_until(next_dt_target)
    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    i_dump += 1

    next_dt_target += dt_dump
