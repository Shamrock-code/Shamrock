"""
Production run: Circular disc & central sink particle
=====================================================

This example demonstrates how to run a smoothed particle hydrodynamics (SPH)
simulation of a circular disc orbiting around a central sink.

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

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


from shamrock.utils.analysis import (
    AnalysisHelper,
    ColumnDensityPlot,
    ColumnParticleCount,
    PerfHistory,
    SliceDensityPlot,
    SliceDiffVthetaProfile,
    SliceDtPart,
    SliceVzPlot,
    VerticalShearGradient,
)

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
Npart = 200000

# Domain decomposition parameters
scheduler_split_val = int(1.0e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Dump and plot frequency and duration of the simulation
dump_freq_stop = 2

dt_stop = 1
nstop = 300

# The list of times at which the simulation will pause for analysis / dumping
t_stop = [i * dt_stop for i in range(nstop + 1)]


# Disc parameter
disc_mass = 0.01  # sol mass
rout = 20.0  # au
rin = 1.0  # au
H_r_0 = 0.05
q = 0.5
p = 3.0 / 2.0
r0 = 1.0

# Viscosity parameter
alpha_AV = 1.0e-3 / 0.08
alpha_u = 1.0
beta_AV = 2.0

# Integrator parameters
C_cour = 0.3
C_force = 0.25


# Sink parameters
center_mass = 1.0


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


def simulate_disc(center_racc, is_torque_free, sim_folder):

    dump_folder = sim_folder + "dump/"
    analysis_folder = sim_folder + "analysis/"
    plot_folder = analysis_folder + "plots/"

    dump_prefix = dump_folder + "dump_"

    # %%
    # Create the dump directory if it does not exist
    if shamrock.sys.world_rank() == 0:
        os.makedirs(sim_folder, exist_ok=True)
        os.makedirs(dump_folder, exist_ok=True)
        os.makedirs(analysis_folder, exist_ok=True)
        os.makedirs(plot_folder, exist_ok=True)

    # %%
    # Utility functions and quantities deduced from the base one

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
    dump_helper = shamrock.utils.dump.ShamrockDumpHandleHelper(model, dump_prefix)

    # %%
    # Load the last dump if it exists, setup otherwise
    def setup_model():
        global disc_mass

        # Generate the default config
        cfg = model.gen_default_config()
        cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
        cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)

        cfg.add_kill_sphere(
            center=(0, 0, 0), radius=bsize
        )  # kill particles outside the simulation box

        cfg.set_units(codeu)
        cfg.set_particle_mass(pmass)
        # Set the CFL
        cfg.set_cfl_cour(C_cour)
        cfg.set_cfl_force(C_force)

        # Enable this to debug the neighbor counts
        # cfg.set_show_neigh_stats(True)

        # Standard way to set the smoothing length (e.g. Price et al. 2018)
        cfg.set_smoothing_length_density_based()

        cfg.set_save_dt_to_fields(True)

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

        # Correct the barycenter before adding the sink
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

        # now that the barycenter & momentum are 0, we can add the sink
        model.add_sink(
            center_mass, (0, 0, 0), (0, 0, 0), center_racc, is_torque_free=is_torque_free
        )

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

    profile_plot = AnalysisHelper(
        analysis_folder=os.path.join(analysis_folder, "plots"),
        analysis_prefix="density_profile",
    )

    def analysis(ianalysis):
        column_density_plot.analysis_save(ianalysis)

        barycenter, disc_mass = shamrock.model_sph.analysisBarycenter(model=model).get_barycenter()

        total_momentum = shamrock.model_sph.analysisTotalMomentum(model=model).get_total_momentum()
        angular_momentum = shamrock.model_sph.analysisAngularMomentum(
            model=model
        ).get_angular_momentum()

        potential_energy = shamrock.model_sph.analysisEnergyPotential(
            model=model
        ).get_potential_energy()

        kinetic_energy = shamrock.model_sph.analysisEnergyKinetic(model=model).get_kinetic_energy()

        save_analysis_data("barycenter.json", "barycenter", barycenter, ianalysis)
        save_analysis_data("disc_mass.json", "disc_mass", disc_mass, ianalysis)
        save_analysis_data("total_momentum.json", "total_momentum", total_momentum, ianalysis)
        save_analysis_data("angular_momentum.json", "angular_momentum", angular_momentum, ianalysis)
        save_analysis_data("potential_energy.json", "potential_energy", potential_energy, ianalysis)
        save_analysis_data("kinetic_energy.json", "kinetic_energy", kinetic_energy, ianalysis)

        sinks = model.get_sinks()
        save_analysis_data("sinks.json", "sinks", sinks, ianalysis)

        perf_analysis.analysis_save(ianalysis)

        #'''
        rho_field = model.compute_field("rho", "f64")
        hpart_field = model.compute_field("hpart", "f64")

        def internal(size: int, x: np.array, y: np.array, z: np.array) -> np.array:
            r = np.sqrt(x**2 + y**2 + z**2)
            return r

        if _HAS_NUMBA:
            internal = njit(internal)

        def custom_getter(size: int, dic_out: dict) -> np.array:
            return internal(
                size,
                dic_out["xyz"][:, 0],
                dic_out["xyz"][:, 1],
                dic_out["xyz"][:, 2],
            )

        r_field = model.compute_field("custom", "f64", custom_getter)

        print(rho_field, r_field)

        x_min = 0.25
        x_max = rout * 1.5
        x_min_log = np.log10(x_min)
        x_max_log = np.log10(x_max)

        bin_edges_x1d = np.logspace(x_min_log, x_max_log, 2049)

        histo = shamrock.compute_histogram(
            bin_edges=bin_edges_x1d,
            x_field=r_field,
            y_field=rho_field,
            do_average=True,
        )

        histo_convolve = shamrock.compute_histogram_convolve_x(
            bin_edges=bin_edges_x1d,
            x_field=r_field,
            y_field=rho_field,
            size_field=hpart_field,
            do_average=True,
        )

        bin_edges_x = np.logspace(x_min_log, x_max_log, 1025)
        bin_edges_y = np.logspace(-6, -3, 1025)
        histo_top = shamrock.compute_histogram_2d(
            bin_edges_x=bin_edges_x,
            bin_edges_y=bin_edges_y,
            x_field=r_field,
            y_field=rho_field,
        )
        histo_2d = np.array(histo_top).reshape(len(bin_edges_x) - 1, len(bin_edges_y) - 1)

        data = {
            "bin_edges_x1d": bin_edges_x1d,
            "bin_edges_x": bin_edges_x,
            "bin_edges_y": bin_edges_y,
            "histo": histo,
            "histo_convolve": histo_convolve,
            "histo_2d": histo_2d,
            "time": model.get_time(),
        }

        profile_plot.analysis_save(ianalysis, data)

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
                dump_helper.write_dump(idump, purge_old_dumps=True, keep_first=1, keep_last=3)

                # dump = model.make_phantom_dump()
                # dump.save_dump(get_ph_dump_name(idump))

            analysis(iplot)

        if istop % dump_freq_stop == 0:
            idump += 1

        iplot += 1
        istop += 1


center_racc = 0.8
sim_folder1 = f"_to_trash/circular_disc_sink_no_torque_free_{Npart}/"
simulate_disc(center_racc, False, sim_folder1)

center_racc = 0.8
sim_folder2 = f"_to_trash/circular_disc_sink_torque_free_{Npart}/"
simulate_disc(center_racc, True, sim_folder2)

center_racc = 0.25
sim_folder3 = f"_to_trash/circular_disc_sink_no_torque_free_0.25_{Npart}/"
simulate_disc(center_racc, False, sim_folder3)

center_racc = 0.25
sim_folder4 = f"_to_trash/circular_disc_sink_torque_free_0.25_{Npart}/"
simulate_disc(center_racc, True, sim_folder4)

import matplotlib.pyplot as plt


def render_profiles(sim_folder1, sim_folder2, sim_folder3, sim_folder4):
    profile_plot1 = AnalysisHelper(
        analysis_folder=os.path.join(sim_folder1, "analysis", "plots"),
        analysis_prefix="density_profile",
    )
    profile_plot2 = AnalysisHelper(
        analysis_folder=os.path.join(sim_folder2, "analysis", "plots"),
        analysis_prefix="density_profile",
    )
    profile_plot3 = AnalysisHelper(
        analysis_folder=os.path.join(sim_folder3, "analysis", "plots"),
        analysis_prefix="density_profile",
    )
    profile_plot4 = AnalysisHelper(
        analysis_folder=os.path.join(sim_folder4, "analysis", "plots"),
        analysis_prefix="density_profile",
    )
    profile_list_analysis_id1 = profile_plot1.get_list_analysis_id()
    profile_list_analysis_id2 = profile_plot2.get_list_analysis_id()
    profile_list_analysis_id3 = profile_plot3.get_list_analysis_id()
    profile_list_analysis_id4 = profile_plot4.get_list_analysis_id()

    for iplot in profile_list_analysis_id1:
        has_iplot_data_1 = iplot in profile_list_analysis_id1
        has_iplot_data_2 = iplot in profile_list_analysis_id2
        has_iplot_data_3 = iplot in profile_list_analysis_id3
        has_iplot_data_4 = iplot in profile_list_analysis_id4

        if (
            not has_iplot_data_1
            or not has_iplot_data_2
            or not has_iplot_data_3
            or not has_iplot_data_4
        ):
            continue

        data1 = profile_plot1.load_analysis(iplot).item()
        data2 = profile_plot2.load_analysis(iplot).item()
        data3 = profile_plot3.load_analysis(iplot).item()
        data4 = profile_plot4.load_analysis(iplot).item()

        time = data1["time"]

        histo_convolve_torque_free = data1["histo_convolve"]
        histo_convolve_not_torque_free = data2["histo_convolve"]
        histo_convolve_torque_free_025 = data3["histo_convolve"]
        histo_convolve_not_torque_free_025 = data4["histo_convolve"]

        bin_edges_x1d = data1["bin_edges_x1d"]
        bin_center = (bin_edges_x1d[:-1] + bin_edges_x1d[1:]) / 2

        plt.figure(dpi=150)
        plt.plot(bin_center, histo_convolve_torque_free, label="racc=0.8")
        plt.plot(bin_center, histo_convolve_not_torque_free, label="racc=0.8 torque free")
        plt.plot(bin_center, histo_convolve_torque_free_025, label="racc=0.25")
        plt.plot(bin_center, histo_convolve_not_torque_free_025, label="racc=0.25 torque free")

        text = f"t = {time:0.3f}"
        from matplotlib.offsetbox import AnchoredText

        anchored_text = AnchoredText(text, loc=1)
        plt.gca().add_artist(anchored_text)

        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1e-6, 1e-3)
        plt.xlim(0.1, 10)
        plt.legend(loc="upper left")
        plt.savefig(f"_to_trash/compare_torque_free_sink_{iplot:07}.png")
        plt.close()


render_profiles(sim_folder1, sim_folder2, sim_folder3, sim_folder4)
