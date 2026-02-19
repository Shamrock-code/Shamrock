"""
Regression test : MHD soundwave
==========================

This test is used to check that the MHD solver is able to reproduce the same results as the reference file.
The test is a simple soundwave setup.
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

kernel = "M6"  # SPH kernel to use
Npart = 100000

C_cour = 0.3
C_force = 0.25

gamma = 5.0 / 3.0
P0 = 3.0 / 5.0
rho_g = 1.0
target_tot_u = 1

dr = 0.02

bmin = (-1.5, -0.75, -0.75)  # (0, 0, 0)
bmax = (1.5, 0.75, 0.75)  # (L, L/2, L/2)
pmass = -1

cs = 1.0


dump_folder = "_to_trash"
sim_name = "mhd_soundwave"


import os

# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)

# %%

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem()
ucte = shamrock.Constants(codeu)

mu_0 = ucte.mu_0()

A = 1e-4
B0 = mu_0
rho0 = 1.0
wavelength = 1.0
k = 2 * np.pi / wavelength  # (xM - xm)
va = np.sqrt((1 + 1.5 * 1.5) / (mu_0 * rho0))  # remove mu0

print("va :", va)
print("mu_0 :", mu_0)


def B_func(r):
    x, y, z = r
    Bx = 1.0 + A * 0.0 * np.cos(k * x)
    By = 1.5 + A * 1.0 * np.cos(k * x)
    Bz = 0.0 + A * 0.0 * np.cos(k * x)
    return (Bx, By, Bz)


def vel_func(r):
    x, y, z = r
    vx = 0.0 + A * 0.0 * np.cos(k * x)
    vy = 0.0 + A * 1.0 * np.cos(k * x)
    vz = 0.0 + A * 0.0 * np.cos(k * x)
    return (vx, vy, vz)


def u_func(r):

    u = P0 / ((gamma - 1) * rho0)
    return u


# %%
# Configure the solver
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)

cfg = model.gen_default_config()
cfg.set_units(codeu)

cfg.set_artif_viscosity_None()
cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)

cfg.print_status()
model.set_solver_config(cfg)

# Set scheduler criteria to effectively disable patch splitting and merging.
crit_split = int(1e9)
crit_merge = 1
model.init_scheduler(crit_split, crit_merge)
bmin, bmax = model.get_ideal_fcc_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

model.resize_simulation_box(bmin, bmax)
model.add_cube_fcc_3d(dr, bmin, bmax)

vol_b = (xM - xm) * (yM - ym) * (zM - zm)

totmass = rho_g * vol_b
print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)

# %%
# Setup the simulation
model.set_field_value_lambda_f64_3("B/rho", B_func)
model.set_field_value_lambda_f64_3("vxyz", vel_func)
model.set_field_value_lambda_f64("uint", u_func)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

# %%
# Save state function
# adapted from https://gist.github.com/gilliss/7e1d451b42441e77ae33a8b790eeeb73


def save_collected_data(data_dict, fpath):

    print(f"Saving data to {fpath}")

    import h5py

    # Open HDF5 file and write in the data_dict structure and info
    f = h5py.File(fpath, "w")
    for dset_name in data_dict:
        if dset_name == "B/rho":  #### because hdf5 cannot handle a backslash in a string ......
            dset = f.create_dataset("B", data=data_dict[dset_name])
        elif dset_name == "dB/rho":
            dset = f.create_dataset("dB", data=data_dict[dset_name])
        elif dset_name == "psi/ch":
            dset = f.create_dataset("psi", data=data_dict[dset_name])
        elif dset_name == "dpsi/ch":
            dset = f.create_dataset("dpsi", data=data_dict[dset_name])
        else:
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
                if True:
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
t_target = 0.5 * 1.0 / va

save_state(0)

i_dump = 0
dt_dump = 0.5 * t_target
next_dt_target = t_sum + dt_dump

while t_sum <= t_target:
    model.evolve_until(t_sum + dt_dump)

    save_state(i_dump)

    t_sum += dt_dump
    i_dump += 1

# %%
# Check regression

reference_folder = "reference-files/regression_mhd_soundwave"

tolerances = [
    {
        "vxyz": (1e-15, 1e-15),
        "hpart": (1e-15, 1e-15),
        "duint": (1e-14, 1e-14),
        "axyz": (1e-13, 1e-13),
        "xyz": (1e-15, 1e-15),
        "axyz_ext": (1e-14, 1e-14),
        "uint": (1e-20, 1e-20),
        "B": (1e-15, 1e-15),
        "dB": (1e-14, 1e-14),
        "psi": (1e-15, 1e-15),
        "dpsi": (1e-15, 1e-15),
        "divB": (1e-14, 1e-14),
    },
    {
        "vxyz": (1e-14, 1e-14),
        "hpart": (1e-15, 1e-15),
        "duint": (1e-14, 1e-14),
        "axyz": (1e-13, 1e-13),
        "xyz": (1e-15, 1e-15),
        "axyz_ext": (1e-14, 1e-14),
        "uint": (1e-15, 1e-15),
        "B": (1e-15, 1e-15),
        "dB": (1e-15, 1e-15),
        "psi": (1e-15, 1e-15),
        "dpsi": (1e-15, 1e-15),
        "divB": (1e-15, 1e-15),
    },
    {
        "vxyz": (1e-14, 1e-14),
        "hpart": (1e-14, 1e-14),
        "duint": (1e-13, 1e-13),
        "axyz": (1e-13, 1e-13),
        "xyz": (1e-15, 1e-15),
        "axyz_ext": (1e-14, 1e-14),
        "uint": (1e-15, 1e-15),
        "B": (1e-15, 1e-15),
        "dB": (1e-15, 1e-15),
        "psi": (1e-15, 1e-15),
        "dpsi": (1e-15, 1e-15),
        "divB": (1e-15, 1e-15),
    },
]

for iplot in range(i_dump):

    fpath_cur = os.path.join(dump_folder, f"{sim_name}_data_{iplot:04}.h5")
    fpath_ref = os.path.join(reference_folder, f"{sim_name}_data_{iplot:04}.h5")

    data_dict_cur = load_collected_data(fpath_cur)
    data_dict_ref = load_collected_data(fpath_ref)

    check_regression(data_dict_ref, data_dict_cur, tolerances[iplot])
