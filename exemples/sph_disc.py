import shamrock
import matplotlib.pyplot as plt
import numpy as np

####################################################
# Setup parameters
####################################################
Npart = 1000000
disc_mass = 0.01 #sol mass
center_mass = 1
center_racc = 0.1

rout = 10
rin = 1

# alpha_ss ~ alpha_AV * 0.08
alpha_AV = 1e-3 / 0.08
alpha_u = 1
beta_AV = 2

q = 0.5
p = 3./2.
r0 = 1

C_cour = 0.3
C_force = 0.25

H_r_in = 0.05

do_plots = False

dump_prefix = "disc_"

R_planet_base = 2
planet_list = [
    #{"R": R_planet_base*1, "mass": 1e-3, "racc": 0.05},

    # reso 3:2
    #{"R": R_planet_base*(3/2)**(2/3), "mass": 1e-3, "racc": 0.05},
]

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
    # return kep_profile(r)

    # subkeplerian correction
    return ((kep_profile(r)**2) - (2*p+q)*cs_profile(r)**2)**0.5

def H_profile(r):
    H = (cs_profile(r) / omega_k(r))
    #fact = (2.**0.5) * 3.
    fact = 1
    return fact * H # factor taken from phantom, to fasten thermalizing

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



####################################################
# Dump handling
####################################################
def get_dump_name(idump):
    return dump_prefix + f"{idump:04}" + ".sham"

def get_vtk_dump_name(idump):
    return dump_prefix + f"{idump:04}" + ".vtk"

def get_last_dump():
    import glob

    res = glob.glob(dump_prefix + "*.sham")

    f_max = ""
    num_max = -1

    for f in res:
        try:
            dump_num = int(f[len(dump_prefix):-5])
            if dump_num > num_max:
                f_max = f
                num_max = dump_num
        except:
            pass

    if num_max == -1:
        return None
    else:
        return num_max

idump_last_dump = get_last_dump()

####################################################
####################################################
####################################################

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

if idump_last_dump is not None:
    model.load_from_dump(get_dump_name(idump_last_dump))
else:
    cfg = model.gen_default_config()
    #cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
    cfg.set_artif_viscosity_ConstantDisc(alpha_u = alpha_u, alpha_AV = alpha_AV, beta_AV = beta_AV)
    cfg.set_eos_locally_isothermalLP07(cs0 = cs0, q = q, r0 = r0)
    cfg.print_status()
    cfg.set_units(codeu)
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e8),1)

    model.resize_simulation_box(bmin,bmax)

    sink_list = [
        {"mass": center_mass, "racc": center_racc, "pos" : (0,0,0), "vel" : (0,0,0)},
    ]
    for _p in planet_list:
        mass = _p["mass"]
        R = _p["R"]
        racc = _p["racc"]

        vphi = kep_profile(R)
        pos = (R,0,0)
        vel = (0,vphi,0)

        sink_list.append({"mass": mass, "racc": racc, "pos" : pos, "vel" : vel})

    sum_mass = sum(s["mass"] for s in sink_list)
    vel_bary = (
        sum(s["mass"]*s["vel"][0] for s in sink_list) / sum_mass,
        sum(s["mass"]*s["vel"][1] for s in sink_list) / sum_mass,
        sum(s["mass"]*s["vel"][2] for s in sink_list) / sum_mass
    )
    pos_bary = (
        sum(s["mass"]*s["pos"][0] for s in sink_list) / sum_mass,
        sum(s["mass"]*s["pos"][1] for s in sink_list) / sum_mass,
        sum(s["mass"]*s["pos"][2] for s in sink_list) / sum_mass
    )

    print("sinks baryenceter : velocity {} position {}".format(vel_bary,pos_bary))

    model.set_particle_mass(pmass)
    for s in sink_list:
        mass = s["mass"]
        x,y,z = s["pos"]
        vx,vy,vz = s["vel"]
        racc = s["racc"]

        x -= pos_bary[0]
        y -= pos_bary[1]
        z -= pos_bary[2]

        vx -= vel_bary[0]
        vy -= vel_bary[1]
        vz -= vel_bary[2]

        print("add sink : mass {} pos {} vel {} racc {}".format(mass,(x,y,z),(vx,vy,vz),racc))
        model.add_sink(mass,(x,y,z),(vx,vy,vz),racc)

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

    model.set_cfl_cour(C_cour)
    model.set_cfl_force(C_force)

    model.do_vtk_dump("init_disc.vtk", True)
    model.dump("init_disc")

    model.change_htolerance(1.3)
    model.timestep()
    model.change_htolerance(1.1)

t_start = model.get_time()

dt_dump = 1e-1
ndump = 1000
t_dumps = [i*dt_dump for i in range(ndump+1)]

idump = 0
for ttarg in t_dumps:
    if ttarg >= t_start:
        model.evolve_until(ttarg)

        model.do_vtk_dump(get_vtk_dump_name(idump), True)
        model.dump(get_dump_name(idump))
    idump += 1
