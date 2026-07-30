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
configs = [
    {"kernel": "M4", "ndust": 0},
    {"kernel": "M6", "ndust": 0},
]

for n in range(1,100):
    configs.append({"kernel": "M6", "ndust": n})


for c in configs:
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
    kernel = c["kernel"]

    # Hydro parameters
    alpha_min = 0.0
    alpha_max = 1
    sigma_decay = 0.1
    alpha_u = 1
    beta_AV = 2

    # Dust parameters
    ndust = c["ndust"]
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

    def compute_sj_new_j( patchdata, j):
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

        # Add the dust
    for k in range(ndust):

        def compute_sj_new(patchdata):
            return compute_sj_new_j(patchdata, k)

        model.overwrite_field_value_f64("s_j", compute_sj_new, k)

    model.set_dt(0.0)  # to help the corrector on next step after adding dust

    # %%
    # Evolve the simulation
    model.solver_logs_reset_cumulated_step_time()
    model.solver_logs_reset_step_count()

    res_rates = []
    for i in range(10):
        model.timestep()
        model.set_dt(0.0)
        res_rates.append(model.solver_logs_last_rate())

    c2 = c.copy()
    c2["rate"] = np.max(res_rates)

    print(c2)
