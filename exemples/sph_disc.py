import shamrock
import matplotlib.pyplot as plt
import numpy as np

####################################################
# Setup parameters
####################################################
Npart = 10000000
disc_mass = 0.01 #sol mass
center_mass = 1
center_racc = 0.1

rout = 10
rin = 1

alpha_u = 1
alpha_AV = 1
beta_AV = 2

q = 0.5
p = 0.5
r0 = 1

C_cour = 0.3
C_force = 0.25

H_r_in = 0.05

do_plots = False

####################################################
####################################################
####################################################

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time = 3600*24*365,
    unit_length = sicte.au(),
    unit_mass = sicte.sol_mass(), )
ucte = shamrock.Constants(codeu)

# Deduced quantities
pmass = disc_mass/Npart
bmin = (-rout*2,-rout*2,-rout*2)
bmax = (rout*2,rout*2,rout*2)
G = ucte.G()


def sigma_profile(r):
    sigma_0 = 1
    return sigma_0 * (r / rin)**(-p)

def kep_profile(r):
    return (G * center_mass / r)**0.5

def omega_k(r):
    return kep_profile(r) / r

def cs_profile(r):
    cs_in = (H_r_in * rin) * omega_k(rin)
    return ((r / rin)**(-q))*cs_in


cs0 = cs_profile(rin)
    
def rot_profile(r):
    return kep_profile(r)

def H_profile(r):
    return (cs_profile(r) / omega_k(r))

def plot_curve_in():
    x = np.linspace(rin,rout)
    sigma = []
    kep = []
    cs = []
    rot = []
    H = []
    H_r = []
    for i in range(len(x)):
        _x = x[i]

        sigma.append(sigma_profile(_x))
        kep.append(kep_profile(_x))
        cs.append(cs_profile(_x))
        rot.append(rot_profile(_x))
        H.append(H_profile(_x))
        H_r.append(H_profile(_x)/_x)

    plt.plot(x,sigma, label = "sigma")
    plt.plot(x,kep, label = "keplerian speed")
    plt.plot(x,cs, label = "cs")
    plt.plot(x,rot, label = "rot speed")
    plt.plot(x,H, label = "H")
    plt.plot(x,H_r, label = "H_r")

if do_plots:
    plot_curve_in()
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.show()



ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_ConstantDisc(alpha_u = alpha_u, alpha_AV = alpha_AV, beta_AV = beta_AV)
cfg.set_eos_locally_isothermalLP07(cs0 = cs0, q = q, r0 = r0)
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

model.init_scheduler(int(1e8),1)

model.resize_simulation_box(bmin,bmax)





setup = model.get_setup()
gen_disc = setup.make_generator_disc_mc(
        part_mass = pmass,
        disc_mass = disc_mass,
        r_in = rin,
        r_out = rout,
        sigma_profile = sigma_profile,
        H_profile = H_profile,
        rot_profile = rot_profile,
        cs_profile = cs_profile,
        random_seed = 666
    )
#print(comb.get_dot())
setup.apply_setup(gen_disc)



model.set_particle_mass(pmass)
model.add_sink(center_mass,(0,0,0),(0,0,0),center_racc)
model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)


model.do_vtk_dump("init_disc.vtk", True)
model.dump("init_disc")


model.change_htolerance(1.3)
model.timestep()
model.change_htolerance(1.1)


t_target = 1
ndump = 100
t_dumps = [i*t_target/ndump for i in range(ndump+1)]

idump = 0
for ttarg in t_dumps:
    model.evolve_until(ttarg)

    dump_name = f"disc_{idump:04}"
    model.do_vtk_dump(dump_name+".vtk", True)
    model.dump(dump_name+".sham")
    idump += 1