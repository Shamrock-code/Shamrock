"""
Disc with sink in GSPH
====================================================================

"""

import os

import numpy as np

import shamrock

outputdir = "_to_trash/gsph_onesink/"
os.system("mkdir -p " + outputdir)
####################################################
# Setup parameters
####################################################
Npart = 10000
disc_mass = 0.001  # sol mass

rout = 350
rin = 90

# alpha_ss ~ alpha_AV * 0.08
alpha_AV = 0.3
alpha_u = 1
beta_AV = 2

q = 0.15
p = 1.0
r0 = 1
Rcav = 2.5
delta_0 = 1e-5

C_cour = 0.3
C_force = 0.25

H_r_in = 0.05

dump_prefix = "GSPH_"

# central star params
center_mass = 2.5
center_racc = 1

# hierarichle split
split_list = [
    {
        "index": 0,
        "mass_ratio": 0.5 / 2,
        "a": 40,
        "e": 0.5,
        "euler_angle": (np.radians(90), 0.0, 0.0),
    },
    # {"index" : 0, "mass_ratio" : 0.5, "a": 0.33333333, "e":0., "euler_angle" :(0,0,0)}
]

do_plots = True

####################################################
####################################################
####################################################

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au() * (39.42410494106729 ** (1 / 3)),  # GM=1
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

# Deduced quantities
pmass = disc_mass / Npart
bmin = (-rout * 2, -rout * 2, -rout * 2)
bmax = (rout * 2, rout * 2, rout * 2)
G = ucte.G()

print("GM =", G * center_mass)


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


####################################################
# Dump handling
####################################################
def get_dump_name(idump):
    return outputdir + dump_prefix + f"{idump:04}" + ".sham"


def get_vtk_dump_name(idump):
    return outputdir + dump_prefix + f"{idump:04}" + ".vtk"

####################################################
####################################################
####################################################

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
cfg.set_eos_locally_isothermalFA2014(h_over_r=H_r_in)
cfg.set_riemann_exact()
cfg.set_force_inutsuka_v2()
cfg.set_reconstruct_piecewise_constant()
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

model.init_scheduler(int(8e5), 1)

model.resize_simulation_box(bmin, bmax)

sink_list = [
    {"mass": center_mass, "racc": center_racc, "pos": (0, 0, 0), "vel": (0, 0, 0)},
]
print(f"sink_list = {sink_list}")

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
print(f"sinks baryenceter : velocity {vel_bary} position {pos_bary}")

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
    print(f"add sink : mass {mass} pos {(x, y, z)} vel {(vx, vy, vz)} racc {racc}")
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

setup.apply_setup(gen_disc)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

sink_history = []

t_start = model.get_time()

freq_stop = 100
norbit = 0.5

dt_stop = (1.0 / freq_stop) * 2 * np.pi * 300
nstop = int(norbit * freq_stop)

print(f"dt_stop = {dt_stop}")

t_stop = [i * dt_stop for i in range(nstop + 1)]
print(f"t_stop = {t_stop}")

idump = 0
istop = 0
c = 0
for ttarg in t_stop:
    if ttarg >= t_start and c < 10:
        # model.evolve_until(ttarg)
        model.evolve_once()

        model.do_vtk_dump(get_vtk_dump_name(idump), True)
        model.dump(get_dump_name(idump))

        idump += 1
        c += 1
