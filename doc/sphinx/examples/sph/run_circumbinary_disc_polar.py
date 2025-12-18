"""
Production run: Circumbinary disc & polar binary orbit
======================================================

This example demonstrates how to run a smoothed particle hydrodynamics (SPH)
simulation of a circular disc orbiting around a central point mass potential.

The simulation models:

- A binary star in polar orbit
- A gaseous disc with specified mass, inner/outer radii, and vertical structure

Also this simulation feature rolling dumps (see `purge_old_dumps` function) to save disk space.

This example is the accumulation of 3 files in a single one to showcase the complete workflow.

- The actual run script (runscript.py)
- Plot generation (make_plots.py)
- Animation from the plots (plot_to_gif.py)

On a cluster or laptop, one can run the code as follows:

.. code-block:: bash

    mpirun <your parameters> ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript runscript.py


then after the run is done (or while it is running), one can run the following to generate the plots:

.. code-block:: bash

    python make_plots.py


"""

# %%
# Runscript (runscript.py)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The runscript is the actual simulation with on the fly analysis & rolling dumps


import glob
import json
import os  # for makedirs

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Setup units

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()

# %%
# List parameters

# Resolution
Npart = 100000

# Domain decomposition parameters
scheduler_split_val = int(1e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Dump and plot frequency and duration of the simulation
dump_freq_stop = 5
plot_freq_stop = 1

dt_stop = 5
nstop = 20

# The list of times at which the simulation will pause for analysis / dumping
t_stop = [i * dt_stop for i in range(nstop + 1)]


# central star params
center_mass = 2.5
center_racc = 1

# Disc parameter
disc_mass = 0.001  # sol mass
rout = 350  # au
rin = 90  # au

H_r_in = 0.05

q = 0.15
p = 1.0

# alpha_ss ~ alpha_AV * 0.08
alpha_AV = 0.01
alpha_u = 1
beta_AV = 2

# Integrator parameters
C_cour = 0.3
C_force = 0.25
C_mult_stiffness = 10


dump_folder = f"_to_trash/circumbinary_disc_polar_normalh_{Npart}/"
dump_prefix = dump_folder + "dump_"


# Disc profiles
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


# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)

# %%
# Utility functions and quantities deduced from the base one

# Deduced quantities
pmass = disc_mass / Npart

bsize = rout * 2
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)

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
# Split as binary
def split_as_binary(sink, m1, m2, a, e, euler_angles=(0.0, 0.0, 0.0)):
    roll, pitch, yaw = euler_angles

    m1 = float(m1)
    m2 = float(m2)
    a = float(a)
    e = float(e)
    roll = float(roll)
    pitch = float(pitch)
    yaw = float(yaw)

    r1, r2, v1, v2 = shamrock.phys.get_binary_rotated(
        m1=m1, m2=m2, a=a, e=e, nu=float(np.radians(0.0)), G=G, roll=roll, pitch=pitch, yaw=yaw
    )

    s1 = {
        "mass": m1,
        "racc": sink["racc"],
        "pos": (sink["pos"][0] + r1[0], sink["pos"][1] + r1[1], sink["pos"][2] + r1[2]),
        "vel": (sink["vel"][0] + v1[0], sink["vel"][1] + v1[1], sink["vel"][2] + v1[2]),
    }

    s2 = {
        "mass": m2,
        "racc": sink["racc"],
        "pos": (sink["pos"][0] + r2[0], sink["pos"][1] + r2[1], sink["pos"][2] + r2[2]),
        "vel": (sink["vel"][0] + v2[0], sink["vel"][1] + v2[1], sink["vel"][2] + v2[2]),
    }

    print("-------------")
    print(sink)
    print(s1)
    print(s2)
    print(r1, r2, v1, v2)
    print("-------------")

    return s1, s2


# add the sinks
sink_list = [
    {"mass": center_mass, "racc": center_racc, "pos": (0, 0, 0), "vel": (0, 0, 0)},
]

print(f"sink_list = {sink_list}")

for split in split_list:
    index_split = split["index"]
    mass_ratio = split["mass_ratio"]
    asplit = split["a"]
    esplit = split["e"]
    euler_angle_split = split["euler_angle"]

    print(f"splitting sink {split}")

    new_sink_list = []

    for i in range(len(sink_list)):
        if i == index_split:
            smass = sink_list[i]["mass"]

            s1, s2 = split_as_binary(
                sink_list[i],
                smass * mass_ratio,
                smass * (1 - mass_ratio),
                asplit,
                esplit,
                euler_angle_split,
            )

            new_sink_list.append(s1)
            new_sink_list.append(s2)
        else:
            new_sink_list.append(sink_list[i])

    sink_list = new_sink_list
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
print("sinks baryenceter : velocity {} position {}".format(vel_bary, pos_bary))

for i in range(len(sink_list)):
    s = sink_list[i]
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

    sink_list[i]["pos"] = (x, y, z)
    sink_list[i]["vel"] = (vx, vy, vz)

# %%
# Start the context
# The context holds the data of the code
# We then init the layout of the field (e.g. the list of fields used by the solver)

ctx = shamrock.Context()
ctx.pdata_layout_new()

# %%
# Attach a SPH model to the context

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")


# %%
# Dump handling
def get_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".sham"


def get_vtk_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".vtk"


def get_ph_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".phdump"


def get_last_dump():
    res = glob.glob(dump_prefix + "*.sham")

    num_max = -1

    for f in res:
        try:
            dump_num = int(f[len(dump_prefix) : -5])
            if dump_num > num_max:
                f_max = f
                num_max = dump_num
        except ValueError:
            pass

    if num_max == -1:
        return None
    else:
        return num_max


def purge_old_dumps():
    if shamrock.sys.world_rank() == 0:
        res = glob.glob(dump_prefix + "*.sham")
        res.sort()

        # The list of dumps to remove (keep the first and last 3 dumps)
        to_remove = res[1:-3]

        for f in to_remove:
            os.remove(f)


idump_last_dump = get_last_dump()

if shamrock.sys.world_rank() == 0:
    print("Last dump:", idump_last_dump)

# %%
# Load the last dump if it exists, setup otherwise

if idump_last_dump is not None:
    model.load_from_dump(get_dump_name(idump_last_dump))
else:
    # Generate the default config
    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    cfg.set_eos_locally_isothermalFA2014(h_over_r=H_r_in)

    # cfg.add_ext_force_point_mass(center_mass, center_racc)
    cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box

    cfg.set_units(codeu)
    cfg.set_particle_mass(pmass)
    # Set the CFL
    cfg.set_cfl_cour(C_cour)
    cfg.set_cfl_force(C_force)
    cfg.set_cfl_mult_stiffness(C_mult_stiffness)

    cfg.set_enable_particle_reordering(False)
    cfg.set_particle_reordering_step_freq(10)

    # Enable this to debug the neighbor counts
    cfg.set_show_neigh_stats(True, filename=dump_folder + "neigh_stats.json")

    # Standard way to set the smoothing length (e.g. Price et al. 2018)
    cfg.set_smoothing_length_density_based()

    # Standard density based smoothing lenght but with a neighbor count limit
    # Use it if you have large slowdowns due to giant particles
    # I recommend to use it if you have a circumbinary discs as the issue is very likely to happen
    # cfg.set_smoothing_length_density_based_neigh_lim(500)

    # Set the solver config to be the one stored in cfg
    model.set_solver_config(cfg)

    # Print the solver config
    model.get_current_config().print_status()

    # Init the scheduler & fields
    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    # Set the simulation box size
    model.resize_simulation_box(bmin, bmax)

    # Create the setup
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
        init_h_factor=0.1,
    )

    # Print the dot graph of the setup
    print(gen_disc.get_dot())

    # Apply the setup
    setup.apply_setup(gen_disc, insert_step=int(scheduler_split_val / 4))

    # correct the momentum of the disc to 0 before adding the sinks
    analysis = shamrock.model_sph.analysisTotalMomentum(model=model)
    total_momentum = analysis.get_total_momentum()

    if shamrock.sys.world_rank() == 0:
        print(f"disc momentum = {total_momentum}")

    model.apply_momentum_offset((-total_momentum[0], -total_momentum[1], -total_momentum[2]))

    # Correct the barycenter
    analysis = shamrock.model_sph.analysisBarycenter(model=model)
    barycenter, disc_mass = analysis.get_barycenter()

    if shamrock.sys.world_rank() == 0:
        print(f"disc barycenter = {barycenter}")

    model.apply_position_offset((-barycenter[0], -barycenter[1], -barycenter[2]))

    # add the sinks

    print(f"sink_list = {sink_list}")

    for s in sink_list:
        mass = s["mass"]
        x, y, z = s["pos"]
        vx, vy, vz = s["vel"]
        racc = s["racc"]
        print(
            "add sink : mass {} pos {} vel {} racc {}".format(mass, (x, y, z), (vx, vy, vz), racc)
        )
        model.add_sink(mass, (x, y, z), (vx, vy, vz), racc)

    # Run a single step to init the integrator and smoothing length of the particles
    # Here the htolerance is the maximum factor of evolution of the smoothing length in each
    # Smoothing length iterations, increasing it affect the performance negatively but increse the
    # convergence rate of the smoothing length
    # this is why we increase it temporely to 1.3 before lowering it back to 1.1 (default value)
    # Note that both ``change_htolerance`` can be removed and it will work the same but would converge
    # more slowly at the first timestep

    model.change_htolerances(coarse=1.3, fine=1.1)
    model.timestep()
    model.change_htolerances(coarse=1.1, fine=1.05)

    model.do_vtk_dump("init.vtk", True)


# %%
# On the fly analysis
def save_plot(ext, center, sinks, arr, iplot, prefix):
    x, y = center
    if shamrock.sys.world_rank() == 0:
        metadata = {
            "extent": [-ext + x, ext + x, -ext + y, ext + y],
            "sinks": sinks,
            "time": model.get_time(),
        }
        np.save(dump_folder + f"{prefix}_{iplot:07}.npy", arr)

        with open(dump_folder + f"{prefix}_{iplot:07}.json", "w") as fp:
            json.dump(metadata, fp, indent=4)


def analysis_plot(iplot):
    sinks = model.get_sinks()

    ext = rout * 1.5
    nx = 1024
    ny = 1024

    # global plots
    arr_rho_integ_z = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=nx,
        ny=ny,
    )

    arr_rho_integ_y = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, 0.0, ext * 2),
        nx=nx,
        ny=ny,
    )

    save_plot(ext, (0.0, 0.0), sinks, arr_rho_integ_z, iplot, "rho_integ_z_global")
    save_plot(ext, (0.0, 0.0), sinks, arr_rho_integ_y, iplot, "rho_integ_y_global")

    # local plots
    x_0, y_0, z_0 = sinks[0]["pos"]
    x_1, y_1, z_1 = sinks[1]["pos"]

    print(f"sinks positions: {x_0,y_0,z_0} {x_1,y_1,z_1}")

    ext_local = 50
    arr_rho_slice_z_local0 = model.render_cartesian_slice(
        "rho",
        "f64",
        center=(x_0, y_0, z_0),
        delta_x=(ext_local * 2, 0, 0.0),
        delta_y=(0.0, ext_local * 2, 0.0),
        nx=nx,
        ny=ny,
    )
    arr_rho_slice_y_local0 = model.render_cartesian_slice(
        "rho",
        "f64",
        center=(x_0, y_0, z_0),
        delta_x=(ext_local * 2, 0, 0.0),
        delta_y=(0.0, 0.0, ext_local * 2),
        nx=nx,
        ny=ny,
    )
    arr_rho_slice_z_local1 = model.render_cartesian_slice(
        "rho",
        "f64",
        center=(x_1, y_1, z_1),
        delta_x=(ext_local * 2, 0, 0.0),
        delta_y=(0.0, ext_local * 2, 0.0),
        nx=nx,
        ny=ny,
    )
    arr_rho_slice_y_local1 = model.render_cartesian_slice(
        "rho",
        "f64",
        center=(x_1, y_1, z_1),
        delta_x=(ext_local * 2, 0, 0.0),
        delta_y=(0.0, 0.0, ext_local * 2),
        nx=nx,
        ny=ny,
    )

    arr_rho_inte_z_local0 = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(x_0, y_0, z_0),
        delta_x=(ext_local * 2, 0, 0.0),
        delta_y=(0.0, ext_local * 2, 0.0),
        nx=nx,
        ny=ny,
    )

    arr_rho_inte_z_local1 = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(x_1, y_1, z_1),
        delta_x=(ext_local * 2, 0, 0.0),
        delta_y=(0.0, ext_local * 2, 0.0),
        nx=nx,
        ny=ny,
    )

    save_plot(ext_local, (x_0, y_0), sinks, arr_rho_slice_z_local0, iplot, "rho_slice_z_local0")
    save_plot(ext_local, (x_0, z_0), sinks, arr_rho_slice_y_local0, iplot, "rho_slice_y_local0")
    save_plot(ext_local, (x_1, y_1), sinks, arr_rho_slice_z_local1, iplot, "rho_slice_z_local1")
    save_plot(ext_local, (x_1, z_1), sinks, arr_rho_slice_y_local1, iplot, "rho_slice_y_local1")
    save_plot(ext_local, (x_0, y_0), sinks, arr_rho_inte_z_local0, iplot, "rho_integ_z_local0")
    save_plot(ext_local, (x_1, y_1), sinks, arr_rho_inte_z_local1, iplot, "rho_integ_z_local1")

    analysis = shamrock.model_sph.analysisEnergyKinetic(model=model)
    ekin = analysis.get_kinetic_energy()

    analysis = shamrock.model_sph.analysisBarycenter(model=model)
    barycenter, disc_mass = analysis.get_barycenter()

    analysis = shamrock.model_sph.analysisEnergyPotential(model=model)
    epot = analysis.get_potential_energy()

    analysis = shamrock.model_sph.analysisTotalMomentum(model=model)
    total_momentum = analysis.get_total_momentum()

    delta_cumulated_step_time = model.solver_logs_cumulated_step_time()
    model.solver_logs_reset_cumulated_step_time()

    delta_step_count = model.solver_logs_step_count()
    model.solver_logs_reset_step_count()

    # Update the json with array of (t,ekin) and create it if it doesn't exist
    if shamrock.sys.world_rank() == 0:
        if not os.path.exists(dump_folder + "ekin.json"):
            with open(dump_folder + "ekin.json", "w") as fp:
                json.dump({"ekin": []}, fp, indent=4)
        with open(dump_folder + "ekin.json", "r") as fp:
            data = json.load(fp)

        # resize the array to the correct size and expand it if needed
        data["ekin"] = data["ekin"][:iplot]

        data["ekin"].append({"t": model.get_time(), "ekin": ekin})
        with open(dump_folder + "ekin.json", "w") as fp:
            json.dump(data, fp, indent=4)

    # same with (t, epot)
    if shamrock.sys.world_rank() == 0:
        if not os.path.exists(dump_folder + "epot.json"):
            with open(dump_folder + "epot.json", "w") as fp:
                json.dump({"epot": []}, fp, indent=4)
        with open(dump_folder + "epot.json", "r") as fp:
            data = json.load(fp)
        data["epot"] = data["epot"][:iplot]
        data["epot"].append({"t": model.get_time(), "epot": epot})
        with open(dump_folder + "epot.json", "w") as fp:
            json.dump(data, fp, indent=4)

    # same with (t, barycenter)
    if shamrock.sys.world_rank() == 0:
        if not os.path.exists(dump_folder + "barycenter.json"):
            with open(dump_folder + "barycenter.json", "w") as fp:
                json.dump({"barycenter": []}, fp, indent=4)
        with open(dump_folder + "barycenter.json", "r") as fp:
            data = json.load(fp)
        data["barycenter"] = data["barycenter"][:iplot]
        data["barycenter"].append({"t": model.get_time(), "barycenter": barycenter})
        with open(dump_folder + "barycenter.json", "w") as fp:
            json.dump(data, fp, indent=4)

    # same with (t, disc_mass)
    if shamrock.sys.world_rank() == 0:
        if not os.path.exists(dump_folder + "disc_mass.json"):
            with open(dump_folder + "disc_mass.json", "w") as fp:
                json.dump({"disc_mass": []}, fp, indent=4)
        with open(dump_folder + "disc_mass.json", "r") as fp:
            data = json.load(fp)
        data["disc_mass"] = data["disc_mass"][:iplot]
        data["disc_mass"].append({"t": model.get_time(), "disc_mass": disc_mass})
        with open(dump_folder + "disc_mass.json", "w") as fp:
            json.dump(data, fp, indent=4)

    # same with (t, total_momentum)
    if shamrock.sys.world_rank() == 0:
        if not os.path.exists(dump_folder + "total_momentum.json"):
            with open(dump_folder + "total_momentum.json", "w") as fp:
                json.dump({"total_momentum": []}, fp, indent=4)
        with open(dump_folder + "total_momentum.json", "r") as fp:
            data = json.load(fp)
        data["total_momentum"] = data["total_momentum"][:iplot]
        data["total_momentum"].append({"t": model.get_time(), "total_momentum": total_momentum})
        with open(dump_folder + "total_momentum.json", "w") as fp:
            json.dump(data, fp, indent=4)

    # same with (t, delta_cumulated_step_time)
    if shamrock.sys.world_rank() == 0:
        if not os.path.exists(dump_folder + "delta_cumulated_step_time.json"):
            with open(dump_folder + "delta_cumulated_step_time.json", "w") as fp:
                json.dump({"delta_cumulated_step_time": []}, fp, indent=4)
        with open(dump_folder + "delta_cumulated_step_time.json", "r") as fp:
            data = json.load(fp)
        data["delta_cumulated_step_time"] = data["delta_cumulated_step_time"][:iplot]
        data["delta_cumulated_step_time"].append(
            {"t": model.get_time(), "delta_cumulated_step_time": delta_cumulated_step_time}
        )
        with open(dump_folder + "delta_cumulated_step_time.json", "w") as fp:
            json.dump(data, fp, indent=4)

    # same with (t, delta_step_count)
    if shamrock.sys.world_rank() == 0:
        if not os.path.exists(dump_folder + "delta_step_count.json"):
            with open(dump_folder + "delta_step_count.json", "w") as fp:
                json.dump({"delta_step_count": []}, fp, indent=4)
        with open(dump_folder + "delta_step_count.json", "r") as fp:
            data = json.load(fp)
        data["delta_step_count"] = data["delta_step_count"][:iplot]
        data["delta_step_count"].append(
            {"t": model.get_time(), "delta_step_count": delta_step_count}
        )
        with open(dump_folder + "delta_step_count.json", "w") as fp:
            json.dump(data, fp, indent=4)


# %%
# Evolve the simulation

model.solver_logs_reset_cumulated_step_time()
model.solver_logs_reset_step_count()

t_start = model.get_time()

idump = 0
iplot = 0
istop = 0
for ttarg in t_stop:

    if ttarg > t_start:

        model.evolve_until(ttarg)

        if istop % dump_freq_stop == 0:
            model.do_vtk_dump(get_vtk_dump_name(idump), True)
            model.dump(get_dump_name(idump))

            # dump = model.make_phantom_dump()
            # dump.save_dump(get_ph_dump_name(idump))

            purge_old_dumps()

        if istop % plot_freq_stop == 0:
            analysis_plot(iplot)

    if istop % dump_freq_stop == 0:
        idump += 1

    if istop % plot_freq_stop == 0:
        iplot += 1

    istop += 1

# %%
# Plot generation (make_plots.py)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Load the on-the-fly analysis after the run to make the plots
# (everything in this section can be in another file)

import matplotlib
import matplotlib.pyplot as plt

# Uncomment this and replace by you dump folder, here since it is just above i comment it out
# dump_folder = "my_masterpiece"
# dump_folder += "/"


def plot_rho_xy(metadata, arr_rho, iplot, prefix, xlabel, ylabel, colorbar_label, vmin, vmax):

    ext = metadata["extent"]
    sinks = metadata["sinks"]

    dpi = 200

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho, cmap=my_cmap, origin="lower", extent=ext, norm="log", vmin=vmin, vmax=vmax
    )

    ax = plt.gca()

    output_list = []
    for s in sinks:
        print(s)
        x, y, z = s["pos"]
        print(x, y, z)
        output_list.append(
            plt.Circle((x, y), s["accretion_radius"] * 5, linewidth=0.5, color="blue", fill=False)
        )
    for circle in output_list:
        ax.add_artist(circle)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"t = {metadata['time']:0.3f} [years]")

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(colorbar_label)

    plt.savefig(dump_folder + "plot_{}_xy_{:04}.png".format(prefix, iplot))
    plt.close()


def plot_rho_xz(metadata, arr_rho, iplot, prefix, xlabel, ylabel, colorbar_label, vmin, vmax):

    ext = metadata["extent"]
    sinks = metadata["sinks"]

    dpi = 200

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho, cmap=my_cmap, origin="lower", extent=ext, norm="log", vmin=vmin, vmax=vmax
    )

    ax = plt.gca()

    output_list = []
    for s in sinks:
        print(s)
        x, y, z = s["pos"]
        print(x, y, z)
        output_list.append(
            plt.Circle((x, z), s["accretion_radius"] * 5, linewidth=0.5, color="blue", fill=False)
        )
    for circle in output_list:
        ax.add_artist(circle)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"t = {metadata['time']:0.3f} [years]")

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(colorbar_label)

    plt.savefig(dump_folder + "plot_{}_xz_{:04}.png".format(prefix, iplot))
    plt.close()


def get_list_dumps_id():
    import glob

    list_files = glob.glob(dump_folder + "rho_integ_z_global_*.npy")
    list_files.sort()
    list_dumps_id = []
    for f in list_files:
        list_dumps_id.append(int(f.split("_")[-1].split(".")[0]))
    return list_dumps_id


def load_plot(prefix, iplot):
    with open(dump_folder + f"{prefix}_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(dump_folder + f"{prefix}_{iplot:07}.npy"), metadata


if shamrock.sys.world_rank() == 0:
    for iplot in get_list_dumps_id():
        print("Rendering rho integ plot for dump", iplot)
        arr_rho, metadata = load_plot("rho_integ_z_global", iplot)
        plot_rho_xy(
            metadata,
            arr_rho,
            iplot,
            "rho_integ_z_global",
            "x",
            "y",
            r"$\int \rho \, \mathrm{d}z$ [code unit]",
            1e-10,
            1e-7,
        )
        arr_rho, metadata = load_plot("rho_integ_y_global", iplot)
        plot_rho_xz(
            metadata,
            arr_rho,
            iplot,
            "rho_integ_y_global",
            "x",
            "z",
            r"$\int \rho \, \mathrm{d}y$ [code unit]",
            1e-10,
            1e-7,
        )
        arr_rho, metadata = load_plot("rho_slice_z_local0", iplot)
        plot_rho_xy(
            metadata,
            arr_rho,
            iplot,
            "rho_slice_z_local0",
            "x",
            "y",
            r"$\rho$ [code unit]",
            1e-12,
            1e-9,
        )
        arr_rho, metadata = load_plot("rho_slice_y_local0", iplot)
        plot_rho_xz(
            metadata,
            arr_rho,
            iplot,
            "rho_slice_y_local0",
            "x",
            "z",
            r"$\rho$ [code unit]",
            1e-12,
            1e-9,
        )
        arr_rho, metadata = load_plot("rho_slice_z_local1", iplot)
        plot_rho_xy(
            metadata,
            arr_rho,
            iplot,
            "rho_slice_z_local1",
            "x",
            "y",
            r"$\rho$ [code unit]",
            1e-12,
            1e-9,
        )
        arr_rho, metadata = load_plot("rho_slice_y_local1", iplot)
        plot_rho_xz(
            metadata,
            arr_rho,
            iplot,
            "rho_slice_y_local1",
            "x",
            "z",
            r"$\rho$ [code unit]",
            1e-12,
            1e-9,
        )
        arr_rho, metadata = load_plot("rho_integ_z_local0", iplot)
        plot_rho_xy(
            metadata,
            arr_rho,
            iplot,
            "rho_integ_z_local0",
            "x",
            "y",
            r"$\int \rho \, \mathrm{d}z$ [code unit]",
            1e-10,
            1e-7,
        )
        arr_rho, metadata = load_plot("rho_integ_z_local1", iplot)
        plot_rho_xy(
            metadata,
            arr_rho,
            iplot,
            "rho_integ_z_local1",
            "x",
            "y",
            r"$\int \rho \, \mathrm{d}z$ [code unit]",
            1e-10,
            1e-7,
        )


# %%
# Make gif for the doc (plot_to_gif.py)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Convert PNG sequence to Image sequence in mpl

# sphinx_gallery_multi_image = "single"

import matplotlib.animation as animation


def show_image_sequence(glob_str, render_gif):

    if render_gif and shamrock.sys.world_rank() == 0:

        import glob

        files = sorted(glob.glob(glob_str))

        from PIL import Image

        image_array = []
        for my_file in files:
            image = Image.open(my_file)
            image_array.append(image)

        if not image_array:
            raise RuntimeError(f"Warning: No images found for glob pattern: {glob_str}")

        pixel_x, pixel_y = image_array[0].size

        # Create the figure and axes objects
        # Remove axes, ticks, and frame & set aspect ratio
        dpi = 200
        fig = plt.figure(dpi=dpi)
        plt.gca().set_position((0, 0, 1, 1))
        plt.gcf().set_size_inches(pixel_x / dpi, pixel_y / dpi)
        plt.axis("off")

        # Set the initial image with correct aspect ratio
        im = plt.imshow(image_array[0], animated=True, aspect="auto")

        def update(i):
            im.set_array(image_array[i])
            return (im,)

        # Create the animation object
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(image_array),
            interval=50,
            blit=True,
            repeat_delay=10,
        )

        return ani


show_plots = False

# %%
# Do it for rho integ z
render_gif = True
glob_str = os.path.join(dump_folder, "plot_rho_integ_z_global_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(dump_folder + "rho_integ_z_global.gif", writer=writer)

    # Show the animation
    if show_plots:
        plt.show()
    else:
        plt.close()

# %%
# Do it for rho integ y
render_gif = True
glob_str = os.path.join(dump_folder, "plot_rho_integ_y_global_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(dump_folder + "rho_integ_y_global.gif", writer=writer)

    # Show the animation
    if show_plots:
        plt.show()
    else:
        plt.close()

# %%
# Do it for rho slice z local0
render_gif = True
glob_str = os.path.join(dump_folder, "plot_rho_slice_z_local0_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(dump_folder + "rho_slice_z_local0.gif", writer=writer)

    # Show the animation
    if show_plots:
        plt.show()
    else:
        plt.close()

# %%
# Do it for rho slice y local0
render_gif = True
glob_str = os.path.join(dump_folder, "plot_rho_slice_y_local0_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(dump_folder + "rho_slice_y_local0.gif", writer=writer)

    # Show the animation
    if show_plots:
        plt.show()
    else:
        plt.close()

# %%
# Do it for rho slice z local1
render_gif = True
glob_str = os.path.join(dump_folder, "plot_rho_slice_z_local1_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(dump_folder + "rho_slice_z_local1.gif", writer=writer)

    # Show the animation
    if show_plots:
        plt.show()
    else:
        plt.close()

# %%
# Do it for rho slice y local1
render_gif = True
glob_str = os.path.join(dump_folder, "plot_rho_slice_y_local1_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(dump_folder + "rho_slice_y_local1.gif", writer=writer)

    # Show the animation
    if show_plots:
        plt.show()
    else:
        plt.close()

# %%
# Do it for rho integ z local0
render_gif = True
glob_str = os.path.join(dump_folder, "plot_rho_integ_z_local0_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(dump_folder + "rho_integ_z_local0.gif", writer=writer)

    # Show the animation
    if show_plots:
        plt.show()
    else:
        plt.close()

# %%
# Do it for rho integ z local1
render_gif = True
glob_str = os.path.join(dump_folder, "plot_rho_integ_z_local1_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(dump_folder + "rho_integ_z_local1.gif", writer=writer)

    # Show the animation
    if show_plots:
        plt.show()
    else:
        plt.close()


# %%
# make plots from json files (json_to_figs.py)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# load the json file for ekin
with open(dump_folder + "ekin.json", "r") as fp:
    data = json.load(fp)
ekin = data["ekin"]

with open(dump_folder + "epot.json", "r") as fp:
    data = json.load(fp)
epot = data["epot"]

plt.figure(figsize=(8, 5), dpi=200)

t = [d["t"] for d in ekin]
ekin = [d["ekin"] for d in ekin]
plt.plot(t, ekin, label="ekin")

t = [d["t"] for d in epot]
epot = [d["epot"] for d in epot]
plt.plot(t, epot, label="epot")

# sum of ekin and epot
plt.plot(t, np.array(ekin) + np.array(epot), label="ekin + epot")

plt.xlabel("t")
plt.ylabel("energy")
plt.legend()
plt.savefig(dump_folder + "ekin.png")
if show_plots:
    plt.show()
else:
    plt.close()


# %%
# load the json file for barycenter
with open(dump_folder + "barycenter.json", "r") as fp:
    data = json.load(fp)
barycenter = data["barycenter"]
t = [d["t"] for d in barycenter]
barycenter_x = [d["barycenter"][0] for d in barycenter]
barycenter_y = [d["barycenter"][1] for d in barycenter]
barycenter_z = [d["barycenter"][2] for d in barycenter]

plt.figure(figsize=(8, 5), dpi=200)

plt.plot(t, barycenter_x)
plt.plot(t, barycenter_y)
plt.plot(t, barycenter_z)
plt.xlabel("t")
plt.ylabel("barycenter")
plt.legend(["x", "y", "z"])
plt.savefig(dump_folder + "barycenter.png")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# load the json file for disc_mass
with open(dump_folder + "disc_mass.json", "r") as fp:
    data = json.load(fp)
disc_mass = data["disc_mass"]
t = [d["t"] for d in disc_mass]
disc_mass = [d["disc_mass"] for d in disc_mass]

plt.figure(figsize=(8, 5), dpi=200)

plt.plot(t, disc_mass)
plt.xlabel("t")
plt.ylabel("disc_mass")
plt.savefig(dump_folder + "disc_mass.png")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# load the json file for total_momentum
with open(dump_folder + "total_momentum.json", "r") as fp:
    data = json.load(fp)
total_momentum = data["total_momentum"]
t = [d["t"] for d in total_momentum]
total_momentum_x = [d["total_momentum"][0] for d in total_momentum]
total_momentum_y = [d["total_momentum"][1] for d in total_momentum]
total_momentum_z = [d["total_momentum"][2] for d in total_momentum]

plt.figure(figsize=(8, 5), dpi=200)

plt.plot(t, total_momentum_x)
plt.plot(t, total_momentum_y)
plt.plot(t, total_momentum_z)
plt.xlabel("t")
plt.ylabel("total_momentum")
plt.legend(["x", "y", "z"])
plt.savefig(dump_folder + "total_momentum.png")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# load the json file for delta_cumulated_step_time
with open(dump_folder + "delta_cumulated_step_time.json", "r") as fp:
    data = json.load(fp)
delta_cumulated_step_time = data["delta_cumulated_step_time"]
t = [d["t"] for d in delta_cumulated_step_time]
delta_cumulated_step_time = [d["delta_cumulated_step_time"] for d in delta_cumulated_step_time]

# cumulated step time = exscan (delta_cumulated_step_time)
cumulated_step_time = np.cumsum(delta_cumulated_step_time)


with open(dump_folder + "delta_step_count.json", "r") as fp:
    data = json.load(fp)
delta_step_count = data["delta_step_count"]
t = [d["t"] for d in delta_step_count]
delta_step_count = [d["delta_step_count"] for d in delta_step_count]

# cumulated step count = exscan (delta_step_count)
cumulated_step_count = np.cumsum(delta_step_count)

plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, cumulated_step_time)
plt.xlabel("simulation time (s)")
plt.ylabel("cumulated step time (s)")
plt.savefig(dump_folder + "cumulated_step_time.png")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# step count

plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, cumulated_step_count)
plt.xlabel("simulation time (s)")
plt.ylabel("step count")
plt.savefig(dump_folder + "cumulated_step_count.png")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# time per step
plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, np.array(cumulated_step_time) / np.array(cumulated_step_count))
plt.xlabel("simulation time (s)")
plt.ylabel("time per step (s)")
plt.savefig(dump_folder + "time_per_step.png")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# load the json file for neigh_stats
with open(dump_folder + "neigh_stats.json", "r") as fp:
    data = json.load(fp)

# find first time where t > 0
i_start = 0
for i in range(len(data)):
    if data[i]["time"] > 0:
        i_start = i
        break

data = data[i_start:]

t = [d["time"] for d in data]
max_true = [d["max_true"] for d in data]
max_all = [d["max_all"] for d in data]

min_true = [d["min_true"] for d in data]
min_all = [d["min_all"] for d in data]
mean_true = [d["mean_true"] for d in data]
mean_all = [d["mean_all"] for d in data]
stddev_true = [d["stddev_true"] for d in data]
stddev_all = [d["stddev_all"] for d in data]


fig, axes = plt.subplots(4, 1, figsize=(10, 12), dpi=200, sharex=True)

# Max neighbors
axes[0].plot(t, max_true, label="max (true neighbors)", color="blue")
axes[0].plot(t, max_all, label="max (all neighbors)", color="red")
axes[0].set_ylabel("max neigh counts")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Min neighbors
axes[1].plot(t, min_true, label="min (true neighbors)", color="blue")
axes[1].plot(t, min_all, label="min (all neighbors)", color="red")
axes[1].set_ylabel("min neigh counts")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Mean neighbors
axes[2].plot(t, mean_true, label="mean (true neighbors)", color="blue")
axes[2].plot(t, mean_all, label="mean (all neighbors)", color="red")
axes[2].set_ylabel("mean neigh counts")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Standard deviation
axes[3].plot(t, stddev_true, label="stddev (true neighbors)", color="blue")
axes[3].plot(t, stddev_all, label="stddev (all neighbors)", color="red")
axes[3].set_ylabel("stddev neigh counts")
axes[3].set_xlabel("simulation time (s)")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(dump_folder + "neigh_stats.png")
if show_plots:
    plt.show()
else:
    plt.close()
