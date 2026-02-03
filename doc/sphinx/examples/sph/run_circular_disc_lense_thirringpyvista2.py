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

shamrock.enable_experimental_features()

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
Npart = int(1e4)

# Domain decomposition parameters
scheduler_split_val = int(1.0e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Disc parameter
center_mass = 1e6  # [sol mass]
disc_mass = 0.001  # [sol mass]
Rg = G * center_mass / (c * c)  # [au]
rin = 4.0 * Rg  # [au]
rout = 5 * rin  # [au]
r0 = rin  # [au]

H_r_0 = 0.01
q = 0.75
p = 3.0 / 2.0

Tin = 2 * np.pi * np.sqrt(rin * rin * rin / (G * center_mass))
if shamrock.sys.world_rank() == 0:
    print(" Orbital period : ", Tin, " [seconds]")

# Sink parameters
center_racc = rin / 2.0  # [au]
inclination =  np.pi / 4


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

dt_stop = Tin / 100.0
nstop = 1000

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
    cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)

    # cfg.add_ext_force_point_mass(center_mass, center_racc)

    cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box
    cfg.add_ext_force_lense_thirring(
        central_mass=center_mass,
        Racc=rin / 2,
        a_spin=0.9,
        dir_spin=(np.sin(inclination), 0.0, np.cos(inclination)),
    )

    cfg.set_particle_tracking(True)

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
    # cfg.set_smoothing_length_density_based()

    # Standard density based smoothing lenght but with a neighbor count limit
    # Use it if you have large slowdowns due to giant particles
    # I recommend to use it if you have a circumbinary discs as the issue is very likely to happen
    cfg.set_smoothing_length_density_based_neigh_lim(500)

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
        init_h_factor=0.06,
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


import gc

import pyvista as pv

pv.set_plot_theme("dark")


p, mesh_actor, text_actor = None, None, None


base_cam_pos = [rout* 1.7, 0, rout*0.4]
rotate_video = True
rotation_vector = [0, 0, 1]  # axis around which you rotate your POV

print(base_cam_pos)


def rotation_matrix(vec, angle):
    """
    Returns 3D rotation matrix of angle 'angle' around rotation vector 'vec'.
    """
    if isinstance(vec, list):
        vec = np.array(vec)
    vec = vec / np.linalg.norm(vec)
    R = (
        np.cos(angle) * np.identity(3)
        + (np.sin(angle)) * np.cross(vec, np.identity(vec.shape[0]) * -1)
        + (1 - np.cos(angle)) * (np.outer(vec, vec))
    )
    return R


def analysis(ianalysis):

    dic_sham = ctx.collect_data()

    #print(dic_sham["part_id"])

    global p, mesh_actor, text_actor, base_cam_pos, orig_r
    if p is None:
        p = pv.Plotter()

        p.camera_position = (base_cam_pos[0], base_cam_pos[1], base_cam_pos[2])  #'iso'

        point_cloud = pv.PolyData(dic_sham["xyz"])

        r = np.linalg.norm(dic_sham["xyz"], axis=1)
        orig_r = [ 0 for i in range(len(r))]

        for i in range(len(r)):
            orig_r[dic_sham["part_id"][i]] = r[i]

        orig_r = np.array(orig_r)

        point_cloud["original r [code unit]"] = orig_r[dic_sham["part_id"]]

        mesh_actor = p.add_mesh(
            point_cloud,
            cmap="magma_r",
            #opacity="geom",
            #clim=(-7.209004326372496, -6.96752862264146),
            render_points_as_spheres=True,
            point_size=15.0,
        )

        p.show_bounds(
            bounds=[-rout, rout, -rout, rout, -rout*0.2, rout*0.2],
            grid='back',
            location='outer',
            ticks='both',
            n_xlabels=2,
            n_ylabels=2,
            n_zlabels=2
        )

        p.show(auto_close=False, interactive_update=True)

    t = model.get_time() + 3000

    R = rotation_matrix(rotation_vector, -2 * np.pi * t / 45000.0)
    cam_pos = R @ base_cam_pos
    #print(cam_pos, base_cam_pos)
    #p.camera_position = [cam_pos, (0, 0, 0), (0, 0, 1)]  #'iso'

    # update *existing* mesh, don't recreate the plotter
    new_cloud = pv.PolyData(dic_sham["xyz"])
    new_cloud["original r [code unit]"] = orig_r[dic_sham["part_id"]]

    tmp = np.log10(pmass * (1.2 / dic_sham["hpart"]) ** (1 / 3))
    #print(tmp.min(), tmp.max())

    # overwrite the actor's mesh in place
    mesh_actor.mapper.SetInputData(new_cloud)

    if text_actor is not None:
        text_actor.SetVisibility(0)

    text_actor = p.add_text(
        "t = {:.03f} [code unit] dt = {:.03f} [code unit]".format(model.get_time(), model.get_dt())
    )
    gc.collect()
    # update rendering
    p.render()  # use render(), not update()
    p.update()  # optional — processes UI events

    #print(p.camera_position)

    p.show(auto_close=False, interactive_update=True)
    # p.show(
    #     screenshot=plot_folder + "pyvista_{:05d}.png".format(ianalysis),
    #     window_size=[1920, 1080],
    #     auto_close=False,
    #     interactive_update=True,
    # )


# %%
# Evolve the simulation
model.solver_logs_reset_cumulated_step_time()
model.solver_logs_reset_step_count()

iplot = 0
ttarget = 0
while True:
    model.timestep()

    analysis(iplot)
    iplot += 1
    ttarget += 0.75
    # exit()
