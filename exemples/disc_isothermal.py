import shamrock
import matplotlib.pyplot as plt
import numpy as np

outputdir = "/local/ylapeyre/Shamrock_tests/30_01/"

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 3600*24*365,unit_length = sicte.au(), unit_mass = sicte.sol_mass(), )
ucte = shamrock.Constants(codeu)


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
#cfg.set_eos_locally_isothermalLP07(0.005,3/4, 1)
cfg.set_eos_locally_isothermal()
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)


bmin = (-10,-10,-10)
bmax = (10,10,10)
model.resize_simulation_box(bmin,bmax)

disc_mass = 0.001

pmass = model.add_disc_3d(
    center=(0,0,0),
    center_mass=1.,
    Npart=100000,
    r_in=0.5, r_out=2.,
    disc_mass=disc_mass,
    p=1.,
    H_r_in=0.05,
    q=1./4.,
    do_warp=True,
    posangle=0.,
    incl = 30.,
    Rwarp = 1.,
    Hwarp = 0.05)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

print("Current part mass :", pmass)

model.set_particle_mass(pmass)


model.add_sink(1,(0,0,0),(0,0,0),0.1)

vk_p = (ucte.G() * 1 / 1)**0.5
#model.add_sink(3*ucte.jupiter_mass(),(1,0,0),(0,0,vk_p),0.01)
#model.add_sink(100,(0,2,0),(0,0,1))

def compute_rho(h):
    return np.array([ model.rho_h(h[i]) for i in range(len(h))])


def plot_vertical_profile(r, rrange, label = ""):

    data = ctx.collect_data()

    rhosel = []
    ysel = []

    for i in range(len(data["hpart"][:])):
        rcy = data["xyz"][i,0]**2 + data["xyz"][i,2]**2

        if rcy > r - rrange and rcy < r + rrange:
            rhosel.append(model.rho_h(data["hpart"][i]))
            ysel.append(data["xyz"][i,1])

    rhosel = np.array(rhosel)
    ysel = np.array(ysel)

    rhobar = np.mean(rhosel)
    
    plt.scatter(ysel, rhosel/rhobar, s=1, label = label)



print("Run")

model.evolve_once_override_time(0,0)

print("Current part mass :", pmass)

plot_vertical_profile(1,0.5, label = "init")

t_sum = 0
t_target = 1000000

i_dump = 0
dt_dump = 1e-2
next_dt_target = t_sum + dt_dump 

while next_dt_target <= t_target:


    fname = outputdir + "kk_001.phfile"
    fname2 = outputdir + "kk_001.vtk"
    

    model.evolve_until(next_dt_target)
    
    do_dump = (i_dump % 50 == 0) 
    if do_dump:
        dump = model.make_phantom_dump()
        dump2 = model.do_vtk_dump(fname2, False)
        dump.save_dump(fname)

    i_dump += 1

    next_dt_target += dt_dump


plot_vertical_profile(1,0.5, label = "end")

plt.legend()
plt.show()
