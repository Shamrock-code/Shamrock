"""
Regression test : SPH disc
==========================

This test is used to check that the SPH disc setup is able to reproduce the
same results as the reference file.
"""

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


# %%
# Setup parameters
import numpy as np

kernel = "M4"  # SPH kernel to use
Npart = 100000
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

dump_folder = "_to_trash"
sim_name = "disc_sph"


import os

# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)

# %%
# Deduced quantities


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


# %%
# Configure the solver
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)

cfg = model.gen_default_config()
cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

# Set scheduler criteria to effectively disable patch splitting and merging.
crit_split = int(1e9)
crit_merge = 1
model.init_scheduler(crit_split, crit_merge)
model.resize_simulation_box(bmin, bmax)

# %%
# Setup the sink particles

sink_list = [
    {"mass": center_mass, "racc": center_racc, "pos": (0, 0, 0), "vel": (0, 0, 0)},
]

model.set_particle_mass(pmass)
for s in sink_list:
    mass = s["mass"]
    x, y, z = s["pos"]
    vx, vy, vz = s["vel"]
    racc = s["racc"]

    print("add sink : mass {} pos {} vel {} racc {}".format(mass, (x, y, z), (vx, vy, vz), racc))

    model.add_sink(mass, (x, y, z), (vx, vy, vz), racc)

# %%
# Setup the simulation

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

model.change_htolerance(1.3)
model.timestep()
model.change_htolerance(1.1)


# %%
# Save state function
# adapted from https://gist.github.com/gilliss/7e1d451b42441e77ae33a8b790eeeb73


def save_collected_data(data_dict, fpath):

    print(f"Saving data to {fpath}")

    import h5py

    # Open HDF5 file and write in the data_dict structure and info
    f = h5py.File(fpath, "w")
    for dset_name in data_dict:
        dset = f.create_dataset(dset_name, data=data_dict[dset_name])
    f.close()


def load_collected_data(fpath):

    print(f"Loading data from {fpath}")

    if not os.path.exists(fpath):
        print(f"File {fpath} does not exist")
        raise FileNotFoundError(f"File {fpath} does not exist")

    import h5py

    # Re-open HDF5 file and read out the data_dict structure and info
    f = h5py.File(fpath, "r")

    data_dict = {}
    for dset_name in f.keys():
        data_dict[dset_name] = f[dset_name][:]

    f.close()

    return data_dict


def check_regression(data_dict1, data_dict2, tolerances):

    # Compare if keys sets match
    if set(data_dict1.keys()) != set(data_dict2.keys()):
        print("Data keys sets do not match")
        raise ValueError(
            f"Data keys sets do not match: {set(data_dict1.keys())} != {set(data_dict2.keys())}"
        )

    # Compare if tolerances are defined for all keys
    if set(tolerances.keys()) != set(data_dict1.keys()):
        print("Tolerances keys sets do not match")
        raise ValueError(
            f"Tolerances keys sets do not match: {set(tolerances.keys())} != {set(data_dict1.keys())}"
        )

    # Compare if values are equal
    for dset_name in data_dict1:

        # Compare same size
        if data_dict1[dset_name].shape != data_dict2[dset_name].shape:
            print(f"Data {dset_name} has different shape")
            print(f"shape: {data_dict1[dset_name].shape} != {data_dict2[dset_name].shape}")
            raise ValueError(f"Data {dset_name} has different shape")

        # Compare values
        delta = np.isclose(
            data_dict1[dset_name],
            data_dict2[dset_name],
            rtol=tolerances[dset_name][0],
            atol=tolerances[dset_name][1],
        )

        offenses = 0

        for i in range(len(data_dict1[dset_name])):
            if not np.all(delta[i]):
                if False:
                    print(
                        f"Data {dset_name} is not equal at index {i}, rtol={tolerances[dset_name][0]}, atol={tolerances[dset_name][1]}"
                    )
                    print(f"    value 1: {data_dict1[dset_name][i]}")
                    print(f"    value 2: {data_dict2[dset_name][i]}")
                    print(
                        f"    absolute diff: {np.abs(data_dict1[dset_name][i] - data_dict2[dset_name][i])}"
                    )
                    print(
                        f"    relative diff: {np.abs(data_dict1[dset_name][i] - data_dict2[dset_name][i]) / data_dict1[dset_name][i]}"
                    )
                offenses += 1

        if offenses > 0:
            print(
                f"Data {dset_name} has {offenses} offenses, absolute diff: {np.abs(data_dict1[dset_name] - data_dict2[dset_name]).max()}"
            )
            raise ValueError(f"Data {dset_name} is not equal")

    print(" -> Regression test passed successfully")


def save_state(iplot):
    data_dict = ctx.collect_data()
    save_collected_data(data_dict, os.path.join(dump_folder, f"{sim_name}_data_{iplot:04}.h5"))


# %%
# Running the simulation

t_sum = 0
t_target = 0.1

save_state(0)

i_dump = 1
dt_dump = 0.05

while t_sum < t_target:
    model.evolve_until(t_sum + dt_dump)

    save_state(i_dump)

    t_sum += dt_dump
    i_dump += 1

# %%
# Check regression

reference_folder = "reference-files/regression_sph_disc"

tolerances = [
    {
        "vxyz": (1e-15, 1e-15),
        "hpart": (1e-15, 1e-15),
        "duint": (1e-14, 1e-14),
        "axyz": (1e-13, 1e-13),
        "xyz": (1e-15, 1e-15),
        "axyz_ext": (1e-14, 1e-14),
        "uint": (1e-20, 1e-20),
    },
    {
        "vxyz": (1e-14, 1e-14),
        "hpart": (1e-15, 1e-15),
        "duint": (1e-14, 1e-14),
        "axyz": (1e-13, 1e-13),
        "xyz": (1e-15, 1e-15),
        "axyz_ext": (1e-14, 1e-14),
        "uint": (1e-15, 1e-15),
    },
    {
        "vxyz": (1e-14, 1e-14),
        "hpart": (1e-14, 1e-14),
        "duint": (1e-13, 1e-13),
        "axyz": (1e-13, 1e-13),
        "xyz": (1e-15, 1e-15),
        "axyz_ext": (1e-14, 1e-14),
        "uint": (1e-15, 1e-15),
    },
]

for iplot in range(i_dump):

    fpath_cur = os.path.join(dump_folder, f"{sim_name}_data_{iplot:04}.h5")
    fpath_ref = os.path.join(reference_folder, f"{sim_name}_data_{iplot:04}.h5")

    data_dict_cur = load_collected_data(fpath_cur)
    data_dict_ref = load_collected_data(fpath_ref)

    check_regression(data_dict_ref, data_dict_cur, tolerances[iplot])
