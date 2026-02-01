"""
Production run: Black hole disc & lense thirring effect
=======================================================

This example demonstrates how to run a smoothed particle hydrodynamics (SPH)
simulation of a circular disc orbiting around a central point mass potential.

The simulation models:

- A central star with a given mass and accretion radius
- A gaseous disc with specified mass, inner/outer radii, and vertical structure
- Artificial viscosity for angular momentum transport
- Locally isothermal equation of state

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
    unit_time=sicte.second(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()
c = ucte.c()

# %%
# List parameters

# Resolution
Npart = 100000

# Domain decomposition parameters
scheduler_split_val = int(1.0e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Disc parameter
center_mass = 1e6  # [sol mass]
disc_mass = 0.001  # [sol mass]
Rg = G * center_mass / (c * c)  # [au]
rin = 4.0 * Rg  # [au]
rout = 10 * rin  # [au]
r0 = rin  # [au]

H_r_0 = 0.01
q = 0.75
p = 3.0 / 2.0

Tin = 2 * np.pi * np.sqrt(rin * rin * rin / (G * center_mass))
if shamrock.sys.world_rank() == 0:
    print(" Orbital period : ", Tin, " [seconds]")

# Sink parameters
center_racc = rin / 2.0  # [au]
inclination = 30.0 * np.pi / 180.0


# Viscosity parameter
alpha_AV = 1.0e-3 / 0.08
alpha_u = 1.0
beta_AV = 2.0

# Integrator parameters
C_cour = 0.3
C_force = 0.25


# Dump and plot frequency and duration of the simulation
dump_freq_stop = 1
plot_freq_stop = 1

dt_stop = Tin / 10.0
nstop = 10

# The list of times at which the simulation will pause for analysis / dumping
t_stop = [i * dt_stop for i in range(nstop + 1)]


sim_folder = f"_to_trash/black_hole_disc_lense_thirring_{Npart}/"

dump_folder = sim_folder + "dump/"
analysis_folder = sim_folder + "analysis/"
plot_folder = analysis_folder + "plots/"

dump_prefix = dump_folder + "dump_"


# Disc profiles
def sigma_profile(r):
    sigma_0 = 1.0  # We do not care as it will be renormalized
    return sigma_0 * (r / r0) ** (-p)


def kep_profile(r):
    return (G * center_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    cs_in = (H_r_0 * r0) * omega_k(r0)
    return ((r / r0) ** (-q)) * cs_in


# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)
    os.makedirs(dump_folder, exist_ok=True)
    os.makedirs(analysis_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

# %%
# Utility functions and quantities deduced from the base one

# Deduced quantities
pmass = disc_mass / Npart

bsize = rout * 2
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)

cs0 = cs_profile(r0)


def rot_profile(r):
    return ((kep_profile(r) ** 2) - (2 * p + q) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    # fact = (2.**0.5) * 3. # factor taken from phantom, to fasten thermalizing
    fact = 1.0
    return fact * H


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


def get_vtk_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".vtk"


def get_ph_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".phdump"


dump_helper = shamrock.utils.dump.ShamrockDumpHandleHelper(model, dump_prefix)

# %%
# Load the last dump if it exists, setup otherwise


def setup_model():
    global disc_mass

    # Generate the default config
    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)

    # cfg.add_ext_force_point_mass(center_mass, center_racc)

    cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box
    cfg.add_ext_force_lense_thirring(
        central_mass=center_mass,
        Racc=rin,
        a_spin=0.9,
        dir_spin=(np.sin(inclination), np.cos(inclination), 0.0),
    )

    cfg.set_units(codeu)
    cfg.set_particle_mass(pmass)
    # Set the CFL
    cfg.set_cfl_cour(C_cour)
    cfg.set_cfl_force(C_force)

    # On a chaotic disc, we disable to two stage search to avoid giant leaves
    cfg.set_tree_reduction_level(6)
    cfg.set_two_stage_search(False)

    # Enable this to debug the neighbor counts
    # cfg.set_show_neigh_stats(True)

    # Standard way to set the smoothing length (e.g. Price et al. 2018)
    cfg.set_smoothing_length_density_based()

    # Standard density based smoothing lenght but with a neighbor count limit
    # Use it if you have large slowdowns due to giant particles
    # I recommend to use it if you have a circumbinary discs as the issue is very likely to happen
    # cfg.set_smoothing_length_density_based_neigh_lim(500)

    cfg.set_save_dt_to_fields(True)

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
        init_h_factor=0.03,
    )

    # Print the dot graph of the setup
    print(gen_disc.get_dot())

    # Apply the setup
    setup.apply_setup(gen_disc)

    # correct the momentum and barycenter of the disc to 0
    analysis_momentum = shamrock.model_sph.analysisTotalMomentum(model=model)
    total_momentum = analysis_momentum.get_total_momentum()

    if shamrock.sys.world_rank() == 0:
        print(f"disc momentum = {total_momentum}")

    model.apply_momentum_offset((-total_momentum[0], -total_momentum[1], -total_momentum[2]))

    # Correct the barycenter
    analysis_barycenter = shamrock.model_sph.analysisBarycenter(model=model)
    barycenter, disc_mass = analysis_barycenter.get_barycenter()

    if shamrock.sys.world_rank() == 0:
        print(f"disc barycenter = {barycenter}")

    model.apply_position_offset((-barycenter[0], -barycenter[1], -barycenter[2]))

    total_momentum = shamrock.model_sph.analysisTotalMomentum(model=model).get_total_momentum()

    if shamrock.sys.world_rank() == 0:
        print(f"disc momentum after correction = {total_momentum}")

    barycenter, disc_mass = shamrock.model_sph.analysisBarycenter(model=model).get_barycenter()

    if shamrock.sys.world_rank() == 0:
        print(f"disc barycenter after correction = {barycenter}")

    if not np.allclose(total_momentum, 0.0):
        raise RuntimeError("disc momentum is not 0")
    if not np.allclose(barycenter, 0.0):
        raise RuntimeError("disc barycenter is not 0")

    # Run a single step to init the integrator and smoothing length of the particles
    # Here the htolerance is the maximum factor of evolution of the smoothing length in each
    # Smoothing length iterations, increasing it affect the performance negatively but increse the
    # convergence rate of the smoothing length
    # this is why we increase it temporely to 1.3 before lowering it back to 1.1 (default value)
    # Note that both ``change_htolerances`` can be removed and it will work the same but would converge
    # more slowly at the first timestep

    model.change_htolerances(coarse=1.3, fine=1.1)
    model.timestep()
    model.change_htolerances(coarse=1.1, fine=1.1)


dump_helper.load_last_dump_or(setup_model)


# %%
# On the fly analysis
def save_rho_integ(ext, arr_rho, iplot):
    if shamrock.sys.world_rank() == 0:
        metadata = {"extent": [-ext, ext, -ext, ext], "time": model.get_time()}
        np.save(plot_folder + f"rho_integ_{iplot:07}.npy", arr_rho)

        with open(plot_folder + f"rho_integ_{iplot:07}.json", "w") as fp:
            json.dump(metadata, fp)


def save_vxyz_integ(ext, arr_vxyz, iplot):
    if shamrock.sys.world_rank() == 0:
        metadata = {"extent": [-ext, ext, -ext, ext], "time": model.get_time()}
        np.save(plot_folder + f"vxyz_integ_{iplot:07}.npy", arr_vxyz)

        with open(plot_folder + f"vxyz_integ_{iplot:07}.json", "w") as fp:
            json.dump(metadata, fp)


def save_analysis_data(filename, key, value, ianalysis):
    """Helper to save analysis data to a JSON file."""
    if shamrock.sys.world_rank() == 0:
        filepath = os.path.join(analysis_folder, filename)
        try:
            with open(filepath, "r") as fp:
                data = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {key: []}
        data[key] = data[key][:ianalysis]
        data[key].append({"t": model.get_time(), key: value})
        with open(filepath, "w") as fp:
            json.dump(data, fp, indent=4)


from shamrock.utils.analysis import (
    ColumnDensityPlot,
    PerfHistory,
    SliceDensityPlot,
    SliceDtPart,
)

perf_analysis = PerfHistory(model, analysis_folder, "perf_history")

column_density_plot = ColumnDensityPlot(
    model,
    ext_r=rout * 1.5,
    nx=1024,
    ny=1024,
    ex=(1, 0, 0),
    ey=(0, 1, 0),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="rho_integ_normal",
)

column_density_plot_hollywood = ColumnDensityPlot(
    model,
    ext_r=rout * 1.5,
    nx=1024,
    ny=1024,
    ex=(1, 0, 0),
    ey=(0, 1, 0),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="rho_integ_hollywood",
)

vertical_density_plot = SliceDensityPlot(
    model,
    ext_r=rout * 1.1 / (16.0 / 9.0),  # aspect ratio of 16:9
    nx=1920,
    ny=1080,
    ex=(1, 0, 0),
    ey=(0, 0, 1),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="rho_slice",
)


dt_part_slice_plot = SliceDtPart(
    model,
    ext_r=rout * 0.5 / (16.0 / 9.0),  # aspect ratio of 16:9
    nx=1920,
    ny=1080,
    ex=(1, 0, 0),
    ey=(0, 0, 1),
    center=((rin + rout) / 2, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="dt_part_slice",
)


def analysis(ianalysis):
    ext = rout * 1.5
    nx = 1024
    ny = 1024

    column_density_plot.analysis_save(ianalysis)
    column_density_plot_hollywood.analysis_save(ianalysis)
    vertical_density_plot.analysis_save(ianalysis)
    dt_part_slice_plot.analysis_save(ianalysis)

    arr_vxyz = model.render_cartesian_column_integ(
        "vxyz",
        "f64_3",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=nx,
        ny=ny,
    )

    save_vxyz_integ(ext, arr_vxyz, ianalysis)

    barycenter, disc_mass = shamrock.model_sph.analysisBarycenter(model=model).get_barycenter()

    total_momentum = shamrock.model_sph.analysisTotalMomentum(model=model).get_total_momentum()

    potential_energy = shamrock.model_sph.analysisEnergyPotential(
        model=model
    ).get_potential_energy()

    kinetic_energy = shamrock.model_sph.analysisEnergyKinetic(model=model).get_kinetic_energy()

    save_analysis_data("barycenter.json", "barycenter", barycenter, ianalysis)
    save_analysis_data("disc_mass.json", "disc_mass", disc_mass, ianalysis)
    save_analysis_data("total_momentum.json", "total_momentum", total_momentum, ianalysis)
    save_analysis_data("potential_energy.json", "potential_energy", potential_energy, ianalysis)
    save_analysis_data("kinetic_energy.json", "kinetic_energy", kinetic_energy, ianalysis)

    perf_analysis.analysis_save(ianalysis)


# %%
# Evolve the simulation
model.solver_logs_reset_cumulated_step_time()
model.solver_logs_reset_step_count()

t_start = model.get_time()

idump = 0
iplot = 0
istop = 0
for ttarg in t_stop:
    if ttarg >= t_start:
        model.evolve_until(ttarg)

        if istop % dump_freq_stop == 0:
            model.do_vtk_dump(get_vtk_dump_name(idump), True)
            dump_helper.write_dump(idump, purge_old_dumps=True, keep_first=1, keep_last=3)

            # dump = model.make_phantom_dump()
            # dump.save_dump(get_ph_dump_name(idump))

        if istop % plot_freq_stop == 0:
            analysis(iplot)

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

face_on_render_kwargs = {
    "x_unit": "au",
    "y_unit": "au",
    "time_unit": "year",
    "x_label": "x",
    "y_label": "y",
}

column_density_plot.render_all(
    **face_on_render_kwargs,
    field_unit="kg.m^-2",
    field_label="$\\int \\rho \\, \\mathrm{{d}} z$",
    vmin=1,
    vmax=1e4,
    norm="log",
)

column_density_plot_hollywood.render_all(
    **face_on_render_kwargs,
    field_unit="kg.m^-2",
    field_label="$\\int \\rho \\, \\mathrm{{d}} z$",
    vmin=1,
    vmax=1e7,
    norm="log",
    holywood_mode=True,
)

vertical_density_plot.render_all(vmin=1e-5, vmax=1e-2, norm="log", time_unit="second")
dt_part_slice_plot.render_all(
    vmin=1e-2, vmax=1e2, norm="log", time_unit="second", contour_list=[1e-2, 1e-1, 1, 1e1, 1e2]
)


def plot_vz_integ(metadata, arr_vz, iplot):
    ext = metadata["extent"]
    dpi = 200
    plt.figure(num=1, clear=True, dpi=dpi)

    # if you want an adaptive colorbar
    # v_ext = np.max(arr_vz)
    # v_ext = max(v_ext, np.abs(np.min(arr_vz)))
    v_ext = 1e-6

    res = plt.imshow(arr_vz, cmap="seismic", origin="lower", extent=ext, vmin=-v_ext, vmax=v_ext)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [seconds]")

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\int v_z \, \mathrm{d}z$ [code unit]")

    plt.savefig(plot_folder + "vz_integ_{:04}.png".format(iplot))
    plt.close()


def get_list_dumps_id():
    list_files = glob.glob(plot_folder + "vxyz_integ_*.npy")
    list_files.sort()
    list_dumps_id = []
    for f in list_files:
        list_dumps_id.append(int(f.split("_")[-1].split(".")[0]))
    return list_dumps_id


def load_rho_integ(iplot):
    with open(plot_folder + f"rho_integ_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(plot_folder + f"rho_integ_{iplot:07}.npy"), metadata


def load_vxyz_integ(iplot):
    with open(plot_folder + f"vxyz_integ_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(plot_folder + f"vxyz_integ_{iplot:07}.npy"), metadata


if shamrock.sys.world_rank() == 0:
    for iplot in get_list_dumps_id():
        print("Rendering vxyz integ plot for dump", iplot)
        arr_vxyz, metadata = load_vxyz_integ(iplot)
        plot_vz_integ(metadata, arr_vxyz[:, :, 2], iplot)


# %%
# Make gif for the doc (plot_to_gif.py)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Convert PNG sequence to Image sequence in mpl

# sphinx_gallery_multi_image = "single"

import matplotlib.animation as animation
from shamrock.utils.plot import show_image_sequence

render_gif = True

# %%
# Do it for rho integ
if render_gif:
    ani = column_density_plot.render_gif(gif_filename="rho_integ.gif", save_animation=True)
    if ani is not None:
        plt.show()


# %%
# Same but in hollywood
if render_gif:
    ani = column_density_plot_hollywood.render_gif(
        gif_filename="rho_integ_hollywood.gif", save_animation=True
    )
    if ani is not None:
        plt.show()


# %%
# For the vertical density plot
if render_gif and shamrock.sys.world_rank() == 0:
    ani = vertical_density_plot.render_gif(save_animation=True)
    if ani is not None:
        plt.show()

# %%
# Make a gif from the plots
if render_gif and shamrock.sys.world_rank() == 0:
    ani = dt_part_slice_plot.render_gif(save_animation=True)
    if ani is not None:
        plt.show()

# %%
# Do it for rho integ
render_gif = True
glob_str = os.path.join(plot_folder, "vz_integ_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(analysis_folder + "vz_integ.gif", writer=writer)

    # Show the animation
    plt.show()


# %%
# helper function to load data from JSON files
def load_data_from_json(filename, key):
    filepath = os.path.join(analysis_folder, filename)
    with open(filepath, "r") as fp:
        data = json.load(fp)[key]
    t = [d["t"] for d in data]
    values = [d[key] for d in data]
    return t, values


# %%
# load the json file for barycenter
t, barycenter = load_data_from_json("barycenter.json", "barycenter")
barycenter_x = [d[0] for d in barycenter]
barycenter_y = [d[1] for d in barycenter]
barycenter_z = [d[2] for d in barycenter]

plt.figure(figsize=(8, 5), dpi=200)

plt.plot(t, barycenter_x)
plt.plot(t, barycenter_y)
plt.plot(t, barycenter_z)
plt.xlabel("t [seconds]")
plt.ylabel("barycenter")
plt.legend(["x", "y", "z"])
plt.savefig(analysis_folder + "barycenter.png")
plt.show()

# %%
# load the json file for disc_mass
t, disc_mass = load_data_from_json("disc_mass.json", "disc_mass")

plt.figure(figsize=(8, 5), dpi=200)

plt.plot(t, disc_mass)
plt.xlabel("t [seconds]")
plt.ylabel("disc_mass")
plt.savefig(analysis_folder + "disc_mass.png")
plt.show()

# %%
# load the json file for total_momentum
t, total_momentum = load_data_from_json("total_momentum.json", "total_momentum")
total_momentum_x = [d[0] for d in total_momentum]
total_momentum_y = [d[1] for d in total_momentum]
total_momentum_z = [d[2] for d in total_momentum]

plt.figure(figsize=(8, 5), dpi=200)

plt.plot(t, total_momentum_x)
plt.plot(t, total_momentum_y)
plt.plot(t, total_momentum_z)
plt.xlabel("t [seconds]")
plt.ylabel("total_momentum")
plt.legend(["x", "y", "z"])
plt.savefig(analysis_folder + "total_momentum.png")
plt.show()

# %%
# load the json file for energies
t, potential_energy = load_data_from_json("potential_energy.json", "potential_energy")
_, kinetic_energy = load_data_from_json("kinetic_energy.json", "kinetic_energy")

total_energy = [p + k for p, k in zip(potential_energy, kinetic_energy)]

plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, potential_energy)
plt.plot(t, kinetic_energy)
plt.plot(t, total_energy)
plt.xlabel("t [seconds]")
plt.ylabel("energy")
plt.legend(["potential_energy", "kinetic_energy", "total_energy"])
plt.savefig(analysis_folder + "energies.png")
plt.show()


# %%
# Plot the performance history (Switch close_plots to True if doing a long run)
perf_analysis.plot_perf_history(close_plots=False)
plt.show()
