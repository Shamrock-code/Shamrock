import matplotlib.pyplot as plt
import numpy as np

import shamrock

####################################################
# Setup parameters
####################################################
Npart = 1000000
disc_mass = 0.01  # sol mass
center_mass = 1
center_racc = 0.1

rout = 10
rin = 1

# alpha_ss ~ alpha_AV * 0.08
alpha_AV = 1e-3 / 0.08
alpha_u = 1
beta_AV = 2

q = 0.5
p = 3.0 / 2.0
r0 = 1

C_cour = 0.3
C_force = 0.25

H_r_in = 0.05

do_plots = False

dump_prefix = "disc_"

R_planet_base = 2  # multiplier for the planet orbit radius
planet_list = [
    # {"R": R_planet_base*1, "mass": 1e-3},
    # reso 3:2
    # {"R": R_planet_base*(3/2)**(2/3), "mass": 1e-3},
    # reso 2:1
    # {"R": R_planet_base*(2/1)**(2/3), "mass": 1e-3},
]

racc_overhill = 0.1
for i in range(len(planet_list)):
    planet_list[i]["racc"] = racc_overhill * shamrock.phys.hill_radius(
        R=planet_list[i]["R"], m=planet_list[i]["mass"], M=center_mass
    )

center_object_is_binary = False
m1_over_centermass = 0.8
m2_over_centermass = 0.2
binary_a = 0.5
binary_e = 0.8

####################################################
####################################################
####################################################

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

# Deduced quantities
pmass = disc_mass / Npart
bmin = (-rout * 2, -rout * 2, -rout * 2)
bmax = (rout * 2, rout * 2, rout * 2)
G = ucte.G()


def sigma_profile(r):
    sigma_0 = 1
    return sigma_0 * (r / rin) ** (-p)


def kep_profile(r):
    return (G * center_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    cs_in = (H_r_in * rin) * omega_k(rin)
    return ((r / rin) ** (-q)) * cs_in


cs0 = cs_profile(rin)


def rot_profile(r):
    # return kep_profile(r)

    # subkeplerian correction
    return ((kep_profile(r) ** 2) - (2 * p + q) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    # fact = (2.**0.5) * 3.
    fact = 1
    return fact * H  # factor taken from phantom, to fasten thermalizing


def plot_curve_in():
    x = np.linspace(rin, rout)
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
        H_r.append(H_profile(_x) / _x)

    plt.plot(x, sigma, label="sigma")
    plt.plot(x, kep, label="keplerian speed")
    plt.plot(x, cs, label="cs")
    plt.plot(x, rot, label="rot speed")
    plt.plot(x, H, label="H")
    plt.plot(x, H_r, label="H_r")


if do_plots:
    plot_curve_in()
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


####################################################
# Binary coordinate mapping
####################################################
def kepler_to_cartesian_no_rotation(m1, m2, a, e):
    nu = np.radians(90)
    # Total mass and reduced mass
    M = m1 + m2
    mu = m1 * m2 / M

    # Distance between the two stars
    r = a * (1 - e**2) / (1 + e * np.cos(nu))

    # Orbital positions in the orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Orbital velocities in the orbital plane
    h = np.sqrt(G * M * a * (1 - e**2))
    vx_orb = -G * M / h * np.sin(nu)
    vy_orb = G * M / h * (e + np.cos(nu))

    # Position in 2D orbital plane
    r_orb = np.array([x_orb, y_orb])

    # Velocity in 2D orbital plane
    v_orb = np.array([vx_orb, vy_orb])

    # Center of mass positions
    r1 = -m2 / M * r_orb
    r2 = m1 / M * r_orb

    # Center of mass velocities
    v1 = -m2 / M * v_orb
    v2 = m1 / M * v_orb

    return r1, r2, v1, v2


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
            dump_num = int(f[len(dump_prefix) : -5])
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

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

if idump_last_dump is not None:
    model.load_from_dump(get_dump_name(idump_last_dump))
else:
    cfg = model.gen_default_config()
    # cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
    cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)
    cfg.print_status()
    cfg.set_units(codeu)
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e8), 1)

    model.resize_simulation_box(bmin, bmax)

    sink_list = []

    if center_object_is_binary:
        m1 = m1_over_centermass * center_mass
        m2 = m2_over_centermass * center_mass

        r1, r2, v1, v2 = kepler_to_cartesian_no_rotation(m1, m2, binary_a, binary_e)

        p1 = (r1[0], r1[1], 0)
        v1 = (v1[0], v1[1], 0)

        p2 = (r2[0], r2[1], 0)
        v2 = (v2[0], v2[1], 0)
        sink_list = [
            {"mass": m1, "racc": center_racc, "pos": p1, "vel": v1},
            {"mass": m2, "racc": center_racc, "pos": p2, "vel": v2},
        ]
    else:
        sink_list = [
            {"mass": center_mass, "racc": center_racc, "pos": (0, 0, 0), "vel": (0, 0, 0)},
        ]

    for _p in planet_list:
        mass = _p["mass"]
        R = _p["R"]
        racc = _p["racc"]

        vphi = kep_profile(R)
        pos = (R, 0, 0)
        vel = (0, vphi, 0)

        sink_list.append({"mass": mass, "racc": racc, "pos": pos, "vel": vel})

    sum_mass = sum(s["mass"] for s in sink_list)
    vel_bary = (
        sum(s["mass"] * s["vel"][0] for s in sink_list) / sum_mass,
        sum(s["mass"] * s["vel"][1] for s in sink_list) / sum_mass,
        sum(s["mass"] * s["vel"][2] for s in sink_list) / sum_mass,
    )
    pos_bary = (
        sum(s["mass"] * s["pos"][0] for s in sink_list) / sum_mass,
        sum(s["mass"] * s["pos"][1] for s in sink_list) / sum_mass,
        sum(s["mass"] * s["pos"][2] for s in sink_list) / sum_mass,
    )

    print("sinks baryenceter : velocity {} position {}".format(vel_bary, pos_bary))

    model.set_particle_mass(pmass)
    for s in sink_list:
        mass = s["mass"]
        x, y, z = s["pos"]
        vx, vy, vz = s["vel"]
        racc = s["racc"]

        x -= pos_bary[0]
        y -= pos_bary[1]
        z -= pos_bary[2]

        vx -= vel_bary[0]
        vy -= vel_bary[1]
        vz -= vel_bary[2]

        print(
            "add sink : mass {} pos {} vel {} racc {}".format(mass, (x, y, z), (vx, vy, vz), racc)
        )
        model.add_sink(mass, (x, y, z), (vx, vy, vz), racc)

    setup = model.get_setup()
    gen_disc = setup.make_generator_disc_mc(
        part_mass=pmass,
        disc_mass=disc_mass,
        r_in=rin,
        r_out=rout,
        sigma_profile=sigma_profile,
        H_profile=H_profile,
        rot_profile=rot_profile,
        cs_profile=cs_profile,
        random_seed=666,
    )
    # print(comb.get_dot())
    setup.apply_setup(gen_disc)

    model.set_cfl_cour(C_cour)
    model.set_cfl_force(C_force)

    model.do_vtk_dump("init_disc.vtk", True)
    model.dump("init_disc")

    model.change_htolerance(1.3)
    model.timestep()
    model.change_htolerance(1.1)

sink_history = []

sink_history.append(model.get_sinks())

t_start = model.get_time()

dt_dump = 1e-2
ndump = 100
t_dumps = [i * dt_dump for i in range(ndump + 1)]

idump = 0
for ttarg in t_dumps:
    if ttarg >= t_start:
        model.evolve_until(ttarg)

        model.do_vtk_dump(get_vtk_dump_name(idump), True)
        model.dump(get_dump_name(idump))
        sink_history.append(model.get_sinks())
    idump += 1

sink_pos_X = [[] for i in range(len(sink_history[0]))]
sink_pos_Y = [[] for i in range(len(sink_history[0]))]
sink_pos_Z = [[] for i in range(len(sink_history[0]))]

for h in sink_history:
    for i in range(len(h)):
        x, y, z = h[i]["pos"]
        print(x, y, z)

        sink_pos_X[i].append(x)
        sink_pos_Y[i].append(y)
        sink_pos_Z[i].append(z)

for i in range(len(sink_history[0])):
    plt.plot(sink_pos_X[i], sink_pos_Y[i])

plt.axis("equal")
plt.show()
