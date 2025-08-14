"""
Regression test : Godunov soundwave 3D
======================================

This test is used to check that the Godunov soundwave setup is able to reproduce the
same results as the reference file.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


tmax = 1.0
do_plot = True

multx = 1
multy = 1
multz = 1

sz = 1 << 1
base = 16


dump_folder = "_to_trash"
sim_name = "soundwave_3d_godunov"


# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)


# %%
# Set config
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

cfg = model.gen_default_config()
scale_fact = 1 / (sz * base * multx)
cfg.set_scale_factor(scale_fact)
cfg.set_eos_gamma(1.4)
model.set_solver_config(cfg)

# %%
# Setup
model.init_scheduler(int(1e3), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

kx, ky, kz = 2 * np.pi, 0, 0
delta_rho = 0
delta_v = 1e-5


def rho_map(rmin, rmax):

    x_min, y_min, z_min = rmin
    x_max, y_max, z_max = rmax

    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    z = (z_min + z_max) / 2

    # shift to center
    x -= 0.5
    y -= 0.5
    z -= 0.5

    x += 0.1  # just for the test to not be centered on the origin
    y += 0.2
    z -= 0.07

    if x**2 + y**2 + z**2 < 0.1:
        return 2.0

    return 1.0


def rhoe_map(rmin, rmax):
    rho = rho_map(rmin, rmax)
    return 1.0 * rho


def rhovel_map(rmin, rmax):
    rho = rho_map(rmin, rmax)
    return (0 * rho, 0 * rho, 0 * rho)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoe_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)


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

save_state(0)

t = [0.01 * i for i in range(100)]
# enumerate t
for i, t in enumerate(t):
    model.evolve_until(t)
    model.dump_vtk(os.path.join(dump_folder, sim_name + "_" + str(i) + ".vtk"))


save_state(1)


# %%
# Check regression

reference_folder = "reference-files/regression_godunov_soundwave_3d"

tolerances = [
    {
        "cell_min": (1e-20, 1e-20),
        "cell_max": (1e-20, 1e-20),
        "rho": (1e-20, 1e-20),
        "rhovel": (1e-20, 1e-20),
        "rhoetot": (1e-20, 1e-20),
    },
    {
        "cell_min": (1e-20, 1e-20),
        "cell_max": (1e-20, 1e-20),
        "rho": (1e-15, 1e-15),
        "rhovel": (1e-15, 1e-15),
        "rhoetot": (1e-15, 1e-15),
    },
]

for istate in [0, 1]:

    fpath_cur = os.path.join(dump_folder, f"{sim_name}_data_{istate:04}.h5")
    fpath_ref = os.path.join(reference_folder, f"{sim_name}_data_{istate:04}.h5")

    data_dict_cur = load_collected_data(fpath_cur)
    data_dict_ref = load_collected_data(fpath_ref)

    check_regression(data_dict_ref, data_dict_cur, tolerances[istate])
