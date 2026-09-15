"""
Dusty SPH disc
========================

A disc but with dust
"""

import os
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from shamrock.utils.analysis import StandardPlotHelper
from shamrock.utils.DustMRNDistribution import DustMRNDistribution
from shamrock.utils.SimulationRunner import SimulationRunner, callback, simulation_setup

import shamrock

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()


# %%
# Sim parameters
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),  # year
    unit_length=sicte.au(),  # astro unit
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)


# Resolution
Npart = int(os.environ.get("NPART", 100000))
print(f"Npart = {Npart} (NPART={os.environ.get('NPART', 'not set')!r})")

# Domain decomposition parameters
scheduler_split_val = int(1.0e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Dump and plot frequency and duration of the simulation
analysis_frequency = 2.5  # years
high_freq_analysis_frequency = 0.1  # years
t_end = 1000.0  # years
t_inject = 0.0  # years

# Sink parameters
center_mass = 1.0  # sol mass
center_racc = 8.0  # au

# Disc parameters
disc = shamrock.utils.disc_setup.StandardDisc(
    units=codeu,
    center_mass=center_mass,
    disc_mass=0.05,  # sol mass
    rin=10.0,  # au
    rout=150.0,  # au
    H_r_0=0.1,
    q=0.5,
    p=3.0 / 2.0,
    r0=10.0,
    rotation="subkeplerian_3d",
    inner_tapering=True,
)

# Solver parameters
kernel = "M6"

# Hydro parameters
alpha_min = 0.0
alpha_max = 1
sigma_decay = 0.1
alpha_u = 1
beta_AV = 2

# Dust parameters
ndust = int(os.environ.get("NDUST", 5))
gamma = 1.4

print(f"ndust = {ndust}")

mrn_pow = 3.5
mrn_cutoff_si = np.inf  # would be 250e-9 normally

epsilon_base = 0.01

rho_grains_si_edges = np.array([2.3 * 1000 for _ in range(ndust + 1)])  # 2.3 g.cm^-3
grain_size_si_edges = np.logspace(-6.5, -1.5, ndust + 1)  # 10um -> 1mm

mrn_distribution = DustMRNDistribution(
    codeu, mrn_pow, mrn_cutoff_si, grain_size_si_edges, rho_grains_si_edges
)


class DustLimiter(Enum):
    NONE = "none"
    SMOOTH = "smooth"
    BALLABIO = "ballabio"
    HARD = "hard"


_dust_limiter_env = os.environ.get("DUST_LIMITER", "none")
limiter = DustLimiter(_dust_limiter_env)
print(f"limiter = {limiter} (DUST_LIMITER={_dust_limiter_env!r})")

# Integrator parameters
C_cour = 0.1
C_force = 0.1


sim_folder = f"_to_trash/circular_dustydisc_{ndust}_{Npart}_{kernel}_{limiter.value}/"

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

# Deduced quantities

bsize = disc.rout * 2
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)
profiles = disc.get_profiles()

grain_size = mrn_distribution.grain_size
grain_size_si = mrn_distribution.grain_size_si
rho_grains = mrn_distribution.rho_grains
massgrid_edges = mrn_distribution.massgrid_edges
mrn_weight = mrn_distribution.mrn_weight


# %%
# Start the context
# The context holds the data of the code
# We then init the layout of the field (e.g. the list of fields used by the solver)

ctx = shamrock.Context()
ctx.pdata_layout_new()

# %%
# Attach a SPH model to the context

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)


class Simulation(SimulationRunner):
    # Use the global vars defined at the top of the file
    t_end = t_end
    dump_prefix = dump_prefix

    def compute_sj_new_j(self, patchdata, j):
        pmass = model.get_particle_mass()

        hpart = patchdata["hpart"]
        rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

        epsilon_target = epsilon_base * mrn_weight[j]
        print(f"epsilon_target = {epsilon_target} {j}")
        s = np.sqrt(rho * epsilon_target)

        print(
            f"s = {s} {np.isnan(s).any()} epsilon_target = {epsilon_target} mrn_weight = {mrn_weight[j]}, rho = {rho}"
        )

        return s

    @callback(at_tsim=t_inject)
    def inject_dust(self, _):
        # Add the dust
        for k in range(ndust):

            def compute_sj_new(patchdata):
                return self.compute_sj_new_j(patchdata, k)

            model.overwrite_field_value_f64("s_j", compute_sj_new, k)

        model.set_dt(0.0)  # to help the corrector on next step after adding dust

    @callback(
        tsim_interval=high_freq_analysis_frequency
    )  # Do the analysis every analysis_frequency
    def high_freq_analysis(self, ianalysis):
        # Run all the analysis modules (declared below)
        for a in self.high_freq_analysis_modules:
            a.analysis_save(ianalysis)

            if hasattr(a, "make_plot"):
                a.make_plot(
                    ianalysis,
                    **a.render_args,
                )
            elif hasattr(a, "plot_perf_history"):
                a.plot_perf_history(close_plots=True)
            elif hasattr(a, "render"):
                a.render()

    @callback(tsim_interval=analysis_frequency)  # Do the analysis every analysis_frequency
    def analysis(self, ianalysis):
        # Run all the analysis modules (declared below)
        for a in self.analysis_modules:
            a.analysis_save(ianalysis)

            if hasattr(a, "make_plot"):
                a.make_plot(
                    ianalysis,
                    **a.render_args,
                )
            elif hasattr(a, "plot_perf_history"):
                a.plot_perf_history(close_plots=True)
            elif hasattr(a, "render"):
                a.render()

        if ianalysis > 10:
            return

        dic = ctx.collect_data()

        s_j = dic["s_j"]
        s_j = s_j.reshape(-1, ndust)
        xyz = dic["xyz"]

        r = np.linalg.norm(xyz[:, :-1], axis=1)
        z_r = xyz[:, -1] / r

        plt.figure(dpi=250)
        for j in range(ndust):
            print(z_r[::10].shape, s_j[::10, j].shape)
            plt.scatter(z_r[::10], s_j[::10, j], s=1, label=f"dust {j}")
        plt.legend()
        plt.xlabel("z/r")
        plt.ylabel("s")
        plt.yscale("symlog", linthresh=1e-8)
        plt.title(f"t = {model.get_time():.1f} [yr]")
        plt.xlim(-0.4, 0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"z_r_vs_s_{ianalysis}.png"))
        plt.close()

    @callback(walltime_interval=60)  # Checkpoint the simulation every 10 minutes
    def checkpoint(self, icheckpoint):
        self.do_checkpoint(icheckpoint, purge_old_dumps=True, keep_first=1, keep_last=3)

    @simulation_setup
    def setup(self):
        cfg = model.gen_default_config()

        cfg.set_artif_viscosity_VaryingCD10(
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma_decay=sigma_decay,
            alpha_u=alpha_u,
            beta_AV=beta_AV,
        )

        cfg.set_eos_locally_isothermalLP07(cs0=disc.cs0(), q=disc.q, r0=disc.r0)

        if ndust > 0:
            cfg.set_dust_drag_epstein(gamma, grain_size, rho_grains)

            if limiter == DustLimiter.NONE:
                cfg.set_dust_mode_monofluid_tva(
                    nvar=ndust, ensure_s_j_positivity=False, smooth_s_positivity_limiter=False
                )
            elif limiter == DustLimiter.SMOOTH:
                cfg.set_dust_mode_monofluid_tva(
                    nvar=ndust, ensure_s_j_positivity=False, smooth_s_positivity_limiter=True
                )
            elif limiter == DustLimiter.BALLABIO:
                cfg.set_dust_mode_monofluid_tva(
                    nvar=ndust, ensure_s_j_positivity=False, smooth_s_positivity_limiter=False
                )
                cfg.set_dust_ballabio_ts_limiter(True)
            elif limiter == DustLimiter.HARD:
                cfg.set_dust_mode_monofluid_tva(
                    nvar=ndust, ensure_s_j_positivity=True, smooth_s_positivity_limiter=False
                )

        cfg.add_kill_sphere(
            center=(0, 0, 0), radius=bsize
        )  # kill particles outside the simulation box

        cfg.set_units(codeu)
        cfg.set_particle_mass(disc.part_mass(Npart))

        print(C_cour, C_force)

        # Set the CFL
        cfg.set_cfl_cour(C_cour)
        cfg.set_cfl_force(C_force)
        cfg.set_show_cfl_detail(True)

        cfg.set_smoothing_length_density_based_neigh_lim(500)

        # On a chaotic disc, we disable to two stage search to avoid giant leaves
        cfg.set_tree_reduction_level(6)
        cfg.set_two_stage_search(False)

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
        gen_disc = disc.make_generator(setup, Npart, random_seed=666)

        # Print the dot graph of the setup
        if shamrock.sys.world_rank() == 0:
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
        model.add_sink(center_mass, (0, 0, 0), (0, 0, 0), center_racc)

        # Run a single step to init the integrator and smoothing length of the particles
        # Here the htolerance is the maximum factor of evolution of the smoothing length in each
        # Smoothing length iterations, increasing it affect the performance negatively but increase the
        # convergence rate of the smoothing length
        # this is why we increase it temporely to 1.3 before lowering it back to 1.1 (default value)
        # Note that both ``change_htolerances`` can be removed and it will work the same but would converge
        # more slowly at the first timestep

        model.change_htolerances(coarse=1.3, fine=1.1)
        model.timestep()
        model.change_htolerances(coarse=1.1, fine=1.1)


# %%
# Custom Analysis modules


def get_rhod_j_getter(model, jdust, ndust):

    def int_getter(size: int, dic_out: dict) -> np.array:
        s_j = dic_out["s_j"].reshape(-1, ndust)
        return s_j[:, jdust] ** 2

    return int_getter


def ColumnDensityPlotDust(
    model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix, jdust, ndust
):
    def compute_rhod_integ(helper):
        return helper.column_integ_render(
            "custom", "f64", custom_getter=get_rhod_j_getter(model, jdust, ndust)
        )

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_rhod_integ,
    )


def SliceDensityPlotDust(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    jdust,
    ndust,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_rho_slice(helper):
        return helper.slice_render(
            "custom",
            "f64",
            do_normalization,
            min_normalization,
            custom_getter=get_rhod_j_getter(model, jdust, ndust),
        )

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_rho_slice,
    )


# %%
# Define sim analysis

max_rho_plot = 1e-9
min_rho_plot = 1e-18
max_rho_integ_plot = 1e3
min_rho_integ_plot = 1e-5


face_on_render_kwargs = {
    "x_unit": "au",
    "y_unit": "au",
    "time_unit": "year",
    "x_label": "x",
    "y_label": "y",
}

sink_params = {
    "sink_scale_factor": 1,
    "sink_color": "green",
    "sink_linewidth": 1,
    "sink_fill": False,
}


slice_params = {
    "ext_r": disc.rout * 0.6 / (16.0 / 9.0),  # aspect ratio of 16:9
    "nx": 1920,
    "ny": 1080,
    "ex": (1, 0, 0),
    "ey": (0, 0, 1),
    "center": ((disc.rin + disc.rout) / 2, 0, 0),
}


from shamrock.utils.analysis import (
    AnalysisHelper,
    ColumnDensityPlot,
    PerfHistory,
    SliceDensityPlot,
)

perf_analysis = PerfHistory(model, analysis_folder, "perf_history")


column_density_plot = ColumnDensityPlot(
    model,
    ext_r=disc.rout * 1.5,
    nx=1024,
    ny=1024,
    ex=(1, 0, 0),
    ey=(0, 1, 0),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="rho_integ_gas",
)

column_density_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": "kg.m^-2",
    "field_label": "$\\int \\rho \\, \\mathrm{{d}} z$",
    "vmin": min_rho_integ_plot,
    "vmax": max_rho_integ_plot,
    "norm": "log",
    **sink_params,
    "extra_title": "[gas + dust]",
}
if ndust > 0:
    dust_column_density_plot = []

    for jdust in range(ndust):
        dust_column_density_plot.append(
            ColumnDensityPlotDust(
                model,
                ext_r=disc.rout * 1.5,
                nx=1024,
                ny=1024,
                ex=(1, 0, 0),
                ey=(0, 1, 0),
                center=(0, 0, 0),
                ndust=ndust,
                jdust=jdust,
                analysis_folder=analysis_folder,
                analysis_prefix=f"rho_integ_dust_{jdust}",
            )
        )

        dust_column_density_plot[-1].render_args = {
            **column_density_plot.render_args,
            "field_unit": "kg.m^-2",
            "field_label": f"$\\int \\rho_{{d, {jdust} }} \\, \\mathrm{{d}} z$",
            "vmin": min_rho_integ_plot,
            "vmax": max_rho_integ_plot,
            "norm": "log",
            **sink_params,
            "extra_title": f"[$s_{{grain}}$ = {grain_size_si[jdust]:.2e} m]",
        }


vertical_density_plot = SliceDensityPlot(
    model,
    **slice_params,
    analysis_folder=analysis_folder,
    analysis_prefix="rho_slice_gas",
)

vertical_density_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": "kg.m^-3",
    "field_label": "$\\rho$",
    "vmin": min_rho_plot,
    "vmax": max_rho_plot,
    "norm": "log",
    **sink_params,
    "extra_title": "[gas + dust]",
}

if ndust > 0:
    dust_slice_density_plot = []

    for jdust in range(ndust):
        dust_slice_density_plot.append(
            SliceDensityPlotDust(
                model,
                **slice_params,
                ndust=ndust,
                jdust=jdust,
                analysis_folder=analysis_folder,
                analysis_prefix=f"rho_slice_dust_{jdust}",
            )
        )

        dust_slice_density_plot[-1].render_args = {
            **vertical_density_plot.render_args,
            "field_unit": "kg.m^-3",
            "field_label": f"$\\rho_{{d, {jdust} }}$",
            "vmin": min_rho_plot,
            "vmax": max_rho_plot,
            "norm": "log",
            **sink_params,
            "extra_title": f"[$s_{{grain}}$ = {grain_size_si[jdust]:.2e} m]",
        }


class DustMassAnalysis:
    def __init__(self, model, analysis_folder, analysis_prefix):
        self.model = model
        self.analysis_folder = analysis_folder
        self.analysis_prefix = analysis_prefix
        self.analysis_helper = AnalysisHelper(analysis_folder, analysis_prefix)
        self.analysis = shamrock.model_sph.analysisDustMass(model=model)
        self.render_args = {}

    def analysis_save(self, ianalysis):
        dust_mass = self.analysis.get_dust_mass()

        # if all dust mass is zero replace by nans
        if np.max(dust_mass) == 0:
            print("all dust mass is zero, replacing by nans")
            dust_mass = [np.nan for _ in range(ndust)]

        self.analysis_helper.analysis_save(
            ianalysis, {"time": self.model.get_time(), "dust_mass": dust_mass}
        )

    def render(self):
        ids = self.analysis_helper.get_list_analysis_id()

        t = []
        dust_mass = []
        for ianalysis in ids:
            ld = self.analysis_helper.load_analysis(ianalysis).item()
            t.append(ld["time"])
            dust_mass.append(ld["dust_mass"])

        t = np.array(t)
        dust_mass = np.array(dust_mass)

        # tinject = first non nan
        iinject = np.argmax(~np.isnan(dust_mass)[:, 0])
        tinject = np.array(t)[iinject]

        t = np.array(t) - tinject

        St = np.zeros(ndust)

        ref_r = disc.r0
        rho_r = 1e-11 * codeu.get("kg") * codeu.get("m", power=-3)
        cs_r = profiles.cs(ref_r)

        for k in range(ndust):
            t_dyn = 2 * np.pi / profiles.omega_k(ref_r)
            ts = shamrock.phys.epstein_stopping_time(
                rho_grain=mrn_distribution.rho_grains[k],
                s_grain=mrn_distribution.grain_size[k],
                rho=rho_r,
                cs=cs_r,
                gamma=gamma,
            )
            St[k] = ts / t_dyn

        plt.figure(dpi=250)
        for k in range(ndust):
            mh = dust_mass[:, k]
            deviation = (mh / mh[iinject]) - 1

            plt.plot(
                t,
                deviation,
                label=f"dust {k}, s = {mrn_distribution.grain_size_si[k]:.1e} [m]",
            )

        total_dust_mass = np.sum(dust_mass, axis=1)
        plt.plot(
            t,
            (total_dust_mass / total_dust_mass[iinject]) - 1,
            color="grey",
            label="total dust mass",
            linestyle="--",
        )

        plt.xlabel("t")
        plt.ylabel("$\\delta M_{dust} / M_{dust,0}$")
        plt.yscale("symlog", linthresh=1e-8)
        plt.title("Dust mass conservation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, "dust_mass_history.png"))
        plt.close()

        for k in range(ndust):
            plt.figure(dpi=250)
            plt.plot(t, dust_mass[:, k])
            plt.xlabel("t")
            plt.ylabel("M_{dust,k} [M_sol]")
            plt.title(f"s = {mrn_distribution.grain_size_si[k]:.1e} [m]")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_folder, f"dust_history_dust_{k}.png"))
            plt.close()


dust_mass_analysis = DustMassAnalysis(model, analysis_folder, "dust_mass")


# %%
# Run the simulation

sim = Simulation(model)

sim.analysis_modules = [
    column_density_plot,
    vertical_density_plot,
]

if ndust > 0:
    sim.analysis_modules.extend(dust_column_density_plot)
    sim.analysis_modules.extend(dust_slice_density_plot)

sim.high_freq_analysis_modules = [
    perf_analysis,
]

if ndust > 0:
    sim.high_freq_analysis_modules.append(dust_mass_analysis)

sim.run()
