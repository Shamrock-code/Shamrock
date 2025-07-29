"""
Regression test : SPH Kelvin-Helmholtz instability
==================================================

This test is used to check that the SPH Kelvin-Helmholtz instability setup is able to reproduce the
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

kernel = "M6"  # SPH kernel to use
resol = 64  # number of particles in the x & y direction
thick = 6  # number of particles in the z direction

# CFLs
C_cour = 0.3
C_force = 0.25

gamma = 1.4

vslip = 1  # slip speed between the two layers

rho_1 = 1

fact = 2 / 3
rho_2 = rho_1 / (fact**3)

P_1 = 3.5
P_2 = 3.5

render_gif = True

dump_folder = "_to_trash"
sim_name = "kh_sph"

u_1 = P_1 / ((gamma - 1) * rho_1)
u_2 = P_2 / ((gamma - 1) * rho_2)

print("Mach number 1 :", vslip / np.sqrt(gamma * P_1 / rho_1))
print("Mach number 2 :", vslip / np.sqrt(gamma * P_2 / rho_2))


import os

# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)


# %%
# Configure the solver
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

# Set scheduler criteria to effectively disable patch splitting and merging.
crit_split = int(1e9)
crit_merge = 1
model.init_scheduler(crit_split, crit_merge)

# %%
# Setup the simulation

# Compute box size
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, resol, thick)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, resol, thick)

model.resize_simulation_box((-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))

# rho1 domain
y_interface = ys / 4
model.add_cube_fcc_3d(dr, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, -y_interface, zs / 2))
model.add_cube_fcc_3d(dr, (-xs / 2, y_interface, -zs / 2), (xs / 2, ys / 2, zs / 2))

# rho 2 domain
model.add_cube_fcc_3d(dr * fact, (-xs / 2, -y_interface, -zs / 2), (xs / 2, y_interface, zs / 2))

model.set_value_in_a_box(
    "uint", "f64", u_1, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, -y_interface, zs / 2)
)
model.set_value_in_a_box(
    "uint", "f64", u_1, (-xs / 2, y_interface, -zs / 2), (xs / 2, ys / 2, zs / 2)
)

model.set_value_in_a_box(
    "uint", "f64", u_2, (-xs / 2, -y_interface, -zs / 2), (xs / 2, y_interface, zs / 2)
)


# the velocity function to trigger KH
def vel_func(r):
    x, y, z = r

    ampl = 0.01
    n = 2
    pert = np.sin(2 * np.pi * n * x / (xs))

    sigma = 0.05 / (2**0.5)
    gauss1 = np.exp(-((y - y_interface) ** 2) / (2 * sigma * sigma))
    gauss2 = np.exp(-((y + y_interface) ** 2) / (2 * sigma * sigma))
    pert *= gauss1 + gauss2

    # Alternative formula (See T. Tricco paper)
    # interf_sz = ys/32
    # vx = np.arctan(y/interf_sz)/np.pi

    vx = 0
    if np.abs(y) > y_interface:
        vx = vslip / 2
    else:
        vx = -vslip / 2

    return (vx, ampl * pert, 0)


model.set_field_value_lambda_f64_3("vxyz", vel_func)

vol_b = xs * ys * zs

totmass = (rho_1 * vol_b / 2) + (rho_2 * vol_b / 2)

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

print("Total mass :", totmass)
print("Current part mass :", pmass)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

model.timestep()


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

reference_folder = "reference-files/regression_sph_kh"

tolerances = [
    {
        "xyz": (1e-15, 1e-15),
        "vxyz": (1e-17, 1e-17),
        "hpart": (1e-16, 1e-16),
        "duint": (1e-13, 1e-13),
        "dtdivv": (1e-13, 1e-13),
        "curlv": (1e-12, 1e-12),
        "soundspeed": (1e-15, 1e-15),
        "uint": (1e-20, 1e-20),
        "axyz_ext": (1e-20, 1e-20),
        "alpha_AV": (1e-20, 1e-20),
        "divv": (1e-12, 1e-12),
        "axyz": (1e-11, 1e-11),
    },
    {
        "xyz": (1e-14, 1e-14),
        "vxyz": (1e-12, 1e-12),
        "hpart": (1e-15, 1e-15),
        "duint": (1e-10, 1e-10),
        "dtdivv": (1e-8, 1e-8),
        "curlv": (1e-12, 1e-12),
        "soundspeed": (1e-13, 1e-13),
        "uint": (1e-13, 1e-13),
        "axyz_ext": (1e-20, 1e-20),
        "alpha_AV": (1e-10, 1e-10),
        "divv": (1e-11, 1e-11),
        "axyz": (1e-10, 1e-10),
    },
    {
        "xyz": (1e-14, 1e-14),
        "vxyz": (1e-12, 1e-12),
        "hpart": (1e-15, 1e-15),
        "duint": (1e-10, 1e-10),
        "dtdivv": (1e-8, 1e-8),
        "curlv": (1e-11, 1e-11),
        "soundspeed": (1e-13, 1e-13),
        "uint": (1e-13, 1e-13),
        "axyz_ext": (1e-20, 1e-20),
        "alpha_AV": (1e-9, 1e-9),
        "divv": (1e-10, 1e-10),
        "axyz": (1e-10, 1e-10),
    },
]

for iplot in range(i_dump):

    fpath_cur = os.path.join(dump_folder, f"{sim_name}_data_{iplot:04}.h5")
    fpath_ref = os.path.join(reference_folder, f"{sim_name}_data_{iplot:04}.h5")

    data_dict_cur = load_collected_data(fpath_cur)
    data_dict_ref = load_collected_data(fpath_ref)

    check_regression(data_dict_ref, data_dict_cur, tolerances[iplot])
