"""
Dusty settling: comparison of the TVA positivity limiters
=========================================================

Run the dusty settling test (multi-grain TVA monofluid in a local stratified
box) once per positivity limiter (none, Ballabio, smooth, hard), then plot the
vertical density profiles at the final time and the dust mass conservation
history of the four runs side by side.

This merges ``examples/sph/run_dustysettle_tva.py`` (one simulation) and
``examples/sph/plot_dustysettle_limiter_compare.py`` (the figure) into a single
script. The paper figure uses ``LZ=256``; the default ``LZ=64`` runs on a
laptop.

The simulations are checkpointed: rerunning the script resumes unfinished
runs, and only replots when all of them are done.

Example (from the build directory, with a python venv providing scipy):

```bash
source .pyvenv/bin/activate
LZ=64 ./shamrock --sycl-cfg 0:0 --loglevel 1 --pypath-from-bin python --rscript ../examples/paper_scripts/stable_tva_solver/dustysettle_limiters.py
```

Output: ``_to_trash/stable_tva_solver/dusty_settle_lz<LZ>/dustysettle_rho_limiters_0050.pdf``
"""

# %%
# Imports
# ------------------------------------------

import json
import os
from enum import Enum

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.linalg import solve_banded
from scipy.special import erfinv
from shamrock.utils.DustMRNDistribution import DustMRNDistribution
from shamrock.utils.numba_helper import maybe_njit
from shamrock.utils.SimulationRunner import SimulationRunner, callback, simulation_setup

import shamrock
from shamrock import NeighCacheStrategy

# %%
# Shamrock initialization
# ------------------------------------------

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Simulation parameters and unit system
# ------------------------------------------
# Identical to ``examples/sph/run_dustysettle_tva.py``.

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

# Disc / box
rho_i = 1e-6  # Initial density
central_mass = 1  # Central mass
R0 = 1  # Where are we in radius
H_r_0 = 0.05  # Disc aspect ratio
gamma = 1.4  # Adiabatic index
box_H_count = 8  # box size in number of H
random_vel_pert = True  # add a small vecolity pertubation to fasten the setup

# Dust
ndust = 5
mrn_pow = 3.5
mrn_cutoff_si = np.inf  # would be 250e-9 normally
epsilon_base = 0.01
rho_grains_si_edges = np.array([2.3 * 1000 for i in range(ndust + 1)])
grain_size_si_edges = np.logspace(-5, -3, ndust + 1)


class DustLimiter(Enum):
    NONE = "none"
    SMOOTH = "smooth"
    BALLABIO = "ballabio"
    HARD = "hard"


# (limiter, panel title), in the order of the figure columns
limiter_panels = [
    (DustLimiter.NONE, "none"),
    (DustLimiter.BALLABIO, "Ballabio"),
    (DustLimiter.SMOOTH, "smooth"),
    (DustLimiter.HARD, "hard"),
]

# Resolution (lattice size)
lx = 12
ly = 12
lz = int(os.environ.get("LZ", "64"))

# Time, in units of the orbital period at R0
# The analysis times are listed explicitly so that the dust injection and the
# end of the simulation fall exactly on an analysis, whatever the rounding.
t_orb = 2 * np.pi * np.sqrt(R0**3 / (ucte.G() * central_mass))
dt_analysis = 0.1 * t_orb  # time between two analyses
iinject = 20  # analysis at which the dust is injected (t = 2 t_orb)
n_analysis = 50  # last analysis (t = 5 t_orb, i.e. 3 t_orb after injection)
analysis_times = [i * dt_analysis for i in range(n_analysis + 1)]
tinject = analysis_times[iinject]
t_end = analysis_times[-1]

# Scheduler
scheduler_split_val = int(2e7)
scheduler_merge_val = 1

# Artificial viscosity
av_alpha_min = 0.0
av_alpha_max = 1
av_sigma_decay = 0.1
av_alpha_u = 1
av_beta_AV = 2
vel_dissipation_eta = 5 / t_orb

# CFL (before dust injection)
cfl_cour_setup = 0.25
cfl_force_setup = 0.25
# CFL (after dust injection)
cfl_cour_inject = 0.1
cfl_force_inject = 0.1

# Injection check
max_v_inject_threshold = 3.0  # in units of cs

# Analytical reference
reference_tau = 0.025
reference_Nz = 4000
reference_zrange = 3.5  # in units of H

# Paths
base_folder = f"_to_trash/stable_tva_solver/dusty_settle_lz{lz}/"

# Plotting
cmap = "plasma"
dpi = 250
sz = 1

# %%
# Derived physical scales and lattice geometry
# ------------------------------------------

G = ucte.G()


def kep_profile(r):
    return (G * central_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


H = H_r_0 * R0
cs = H * omega_k(R0)
box = box_H_count * H

print(f"cs = {cs}")
print(f"H = {H}")
print(f"t_orb = {t_orb}")

mrn_distribution = DustMRNDistribution(
    codeu, mrn_pow, mrn_cutoff_si, grain_size_si_edges, rho_grains_si_edges
)

lmin = (-(lx // 2), -(ly // 2), -(lz // 2))
lmax = (lx // 2, ly // 2, lz // 2)

# Call with dr = 1 as we will rescale on next call
(xm, ym, zm), (xM, yM, zM) = shamrock.math.get_periodic_hcp_box(1.0, lmin, lmax)
print(f"base lattice : xM = {xM}, yM = {yM}, zM = {zM}")
dr = 2 * box / (zM - zm)
print(f"dr = {dr}")
bmin, bmax = shamrock.math.get_periodic_hcp_box(dr, lmin, lmax)
print(f"new lattice : bmin = {bmin}, bmax = {bmax}")
xm, ym, zm = bmin
xM, yM, zM = bmax

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho_i * vol_b
print("Total mass :", totmass)

to_dens = codeu.to("kg") * codeu.to("m") ** -3

# %%
# Initial conditions and dust injection functions
# ------------------------------------------

cs_g = cs


def scaling_rho(r):
    x, y, z = r

    loc_h = H / (2**0.5)
    gaussian = np.exp(-(y**2) / (2 * loc_h * loc_h)) / (loc_h * np.sqrt(2 * np.pi))
    return gaussian


def func_rho_d_j(r, idust):
    return (0.1 / ndust) * rho_i * scaling_rho(r)


def func_rho_g(r):
    return rho_i * scaling_rho(r) - sum([func_rho_d_j(r, i) for i in range(ndust)])


def uint_g(r):
    rho_g = func_rho_g(r)
    P = rho_g * cs_g * cs_g / gamma
    return P / ((gamma - 1) * rho_g)


def remap_positions_z(r):
    x, y, z = r

    rn = max(abs(zM), abs(zm))
    z = H * erfinv(z / rn)

    z = min(z, zM)
    z = max(z, zm)
    return (x, y, z)


def compute_sj_new_j(model, patchdata, j):
    pmass = model.get_particle_mass()

    hpart = patchdata["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    z = patchdata["xyz"][:, 2]
    # mask to only modify particles with |z| < H
    mask = 1 / (1 + np.exp((np.abs(z) - 1.75 * H) / (H / 16)))

    epsilon_target = epsilon_base * mrn_distribution.mrn_weight[j] * mask
    return np.sqrt(rho * epsilon_target)


# %%
# Analytical reference solution
# ------------------------------------------


# The matrix assembly loops are jitted with numba when available, but
# ``solve_banded`` is a scipy function that numba cannot compile, so the solve
# itself is done in plain Python by ``S_rho`` and ``S_v``.


@maybe_njit
def _S_rho_system(rho, v, vp, hz, tau, Nz):
    """
    Assemble the banded Crank-Nicolson system for the continuity equation.
    """

    npts = Nz + 2

    ab_A = np.zeros((3, npts))
    bd_main = np.zeros(npts)
    bd_lower = np.zeros(npts)
    bd_upper = np.zeros(npts)

    # Interior points
    for j in range(1, Nz + 1):
        C1 = -tau / (8 * hz) * (v[j + 1] + vp[j + 1])
        C2 = -tau / (8 * hz) * (v[j - 1] + vp[j - 1])

        ab_A[2, j - 1] = C2
        ab_A[1, j] = 1.0
        ab_A[0, j + 1] = -C1

        bd_lower[j] = -C2
        bd_main[j] = 1.0
        bd_upper[j] = C1

    # Left boundary
    C = -tau / (8 * hz) * (vp[0] + v[0] - vp[1] - v[1])

    ab_A[1, 0] = 0.5 + C
    ab_A[0, 1] = 0.5 + C

    bd_main[0] = 0.5 - C
    bd_upper[0] = 0.5 - C

    # Right boundary
    C = -tau / (8 * hz) * (vp[-1] + v[-1] - vp[-2] - v[-2])

    ab_A[2, npts - 2] = 0.5 - C
    ab_A[1, npts - 1] = 0.5 - C

    bd_lower[npts - 1] = 0.5 + C
    bd_main[npts - 1] = 0.5 + C

    r = bd_main * rho
    r[1:] += bd_lower[1:] * rho[:-1]
    r[:-1] += bd_upper[:-1] * rho[1:]

    return ab_A, r


def S_rho(rho, v, vp, hz, tau, Nz):
    """
    Step rho forward using Crank-Nicolson on continuity equation.
    """

    ab_A, r = _S_rho_system(rho, v, vp, hz, tau, Nz)
    ret = solve_banded((1, 1), ab_A, r)
    ret = np.maximum(ret, 0)

    ###################
    # If there are zero values, propagate the zero value to the edges
    ###################
    n = len(ret)
    center = n // 2

    # Right side (center -> end)
    right = ret[center:]
    zero_mask_right = right == 0
    propagate_right = np.cumsum(zero_mask_right) > 0
    ret[center:] = right * (~propagate_right)

    # Left side (center -> start)
    left = ret[: center + 1][::-1]  # reverse
    zero_mask_left = left == 0
    propagate_left = np.cumsum(zero_mask_left) > 0
    ret[: center + 1] = (left * (~propagate_left))[::-1]

    return ret


@maybe_njit
def _S_v_system(v, vbar, vbarp, hz, zbar, tau, Nz, Stj):
    """
    Assemble the banded Crank-Nicolson system for the velocity equation.
    """

    npts = Nz + 2

    ab_A = np.zeros((3, npts))
    bd_main = np.zeros(npts)
    bd_lower = np.zeros(npts)
    bd_upper = np.zeros(npts)
    E0 = np.zeros_like(v)

    for j in range(1, Nz + 1):
        E0[j] = -zbar[j] / (1 + zbar[j] ** 2) ** 1.5

        E1 = (vbar[j] + vbarp[j]) / (8 * hz)
        E2 = 1.0 / (2.0 * Stj[j])

        ab_A[2, j - 1] = -E1
        ab_A[1, j] = 1.0 / tau + E2
        ab_A[0, j + 1] = E1

        bd_lower[j] = E1
        bd_main[j] = 1.0 / tau - E2
        bd_upper[j] = -E1

    # Boundary conditions
    ab_A[1, 0] = 1
    ab_A[0, 1] = 1

    ab_A[2, npts - 2] = 1
    ab_A[1, npts - 1] = 1

    r = bd_main * v
    r[1:] += bd_lower[1:] * v[:-1]
    r[:-1] += bd_upper[:-1] * v[1:]
    r += E0

    return ab_A, r


def S_v(v, vbar, vbarp, rho, rhop, hz, zbar, tau, Nz, Stj):
    """
    Step velocity forward using Crank-Nicolson.
    """

    ab_A, r = _S_v_system(v, vbar, vbarp, hz, zbar, tau, Nz, Stj)
    return solve_banded((1, 1), ab_A, r)


class ReferenceDustySettle:
    def __init__(self, Nz, H, rho_mid, R0, dust_to_gas, dt):
        self.H = H
        self.rho_mid = rho_mid
        self.rho_0 = rho_mid * np.sqrt(2 * np.pi)
        self.R0 = R0
        self.dust_to_gas = dust_to_gas
        self.dt = dt

        self.Nz = Nz

        zout = reference_zrange * H / R0
        zmin = -zout
        zmax = zout

        self.hz = (zmax - zmin) / self.Nz

        # exactly Nz+2 points
        self.zbar = np.linspace(zmin - 0.5 * self.hz, zmax + 0.5 * self.hz, Nz + 2)

    def z_arr(self):
        return self.zbar * self.R0

    def rho_g_func(self, z):
        loc_h = self.H
        ampl = self.rho_0 / (np.sqrt(2 * np.pi))
        return ampl * np.exp(-(z**2) / (2 * loc_h * loc_h))

    def rho_d_bar_func(self, z):

        mask = 1 / (1 + np.exp((np.abs(z) - 1.75 * H) / (H / 16)))

        return mask * self.dust_to_gas * self.rho_g_func(z) / self.rho_mid

    def gen_IC(self, rhoeff, sj, cs, OmegaK):

        self.rhog = self.rho_g_func(self.z_arr())

        self.rho = self.rho_d_bar_func(self.z_arr())
        self.v = np.zeros_like(self.rho)

        self.ts = rhoeff * sj / (cs * self.rhog)
        self.Stj = self.ts * OmegaK

        self.tbar = 0.0

    def advance(self, tau):

        rhop = S_rho(self.rho, self.v, self.v, self.hz, tau, self.Nz)

        vp = S_v(self.v, self.v, self.v, self.rho, rhop, self.hz, self.zbar, tau, self.Nz, self.Stj)

        rhop = S_rho(self.rho, self.v, vp, self.hz, tau, self.Nz)

        self.v = S_v(self.v, self.v, vp, self.rho, rhop, self.hz, self.zbar, tau, self.Nz, self.Stj)

        self.rho = rhop

        self.tbar += tau

    def advance_until(self, tfinal):
        while self.tbar < tfinal:
            self.advance(self.dt)


class ReferenceDustySettleAll:
    def __init__(self, rhomid):
        self.rhomid = rhomid
        self.tau = reference_tau

        self.vK = kep_profile(R0) * codeu.to("m") / codeu.to("s")
        self.OmegaK = omega_k(R0) * codeu.to("s") ** -1
        self.tK = 1 / self.OmegaK

        self.rhoscale = self.rhomid
        self.tscale = self.tK

        self.cs = cs * codeu.to("m") / codeu.to("s")

        self.soluces = []
        for j in range(ndust):
            mrn_weight_j = mrn_distribution.mrn_weight[j]
            eps_j = mrn_weight_j * epsilon_base
            dtg_j = eps_j / (1 - epsilon_base)
            self.soluces.append(
                ReferenceDustySettle(reference_Nz, H, self.rhomid, R0, dtg_j, self.tau)
            )

            rhoeff = mrn_distribution.rho_grains_si_edges[j] * np.sqrt(np.pi * gamma / 8)

            self.soluces[j].gen_IC(rhoeff, mrn_distribution.grain_size_si[j], self.cs, self.OmegaK)

    def evolve_until(self, t):
        for k in range(ndust):
            self.soluces[k].advance_until(t * codeu.to("s") / self.tscale)


# %%
# Simulation
# ------------------------------------------


def write_json(filepath, data):
    if shamrock.sys.world_rank() == 0:
        with open(filepath, "w") as fp:
            json.dump(data, fp, indent=4)


class DustySettleSimulation(SimulationRunner):
    """One dusty settling run with the given positivity limiter."""

    # Use the global vars defined at the top of the file
    t_end = t_end

    def __init__(self, limiter):
        self.limiter = limiter
        self.dump_folder = base_folder + f"{limiter.value}/"
        self.dump_prefix = self.dump_folder + "dump_"

        if shamrock.sys.world_rank() == 0:
            os.makedirs(self.dump_folder, exist_ok=True)

        self.ctx = shamrock.Context()
        self.ctx.pdata_layout_new()

        model = shamrock.get_Model_SPH(context=self.ctx, vector_type="f64_3", sph_kernel="M6")
        self.analysis_dust_mass = shamrock.model_sph.analysisDustMass(model=model)

        self.reference = None

        super().__init__(model)

    def collect_rho_z(self):
        dic = self.ctx.collect_data()
        hpart = dic["hpart"]
        rho = self.model.get_particle_mass() * (self.model.get_hfact() / np.array(hpart)) ** 3
        return dic, rho

    def get_reference(self):
        """
        The reference solution is normalised by the midplane density measured on
        the first analysis after the injection. It is stored on disk so that a
        restarted run builds the exact same reference.
        """
        if self.reference is None:
            filepath = self.dump_folder + "reference_rhomid.json"
            if os.path.exists(filepath):
                with open(filepath, "r") as fp:
                    rhomid = json.load(fp)["rhomid"]
            else:
                _, rho = self.collect_rho_z()
                rhomid = float(rho.max() * to_dens)
                write_json(filepath, {"rhomid": rhomid})
            self.reference = ReferenceDustySettleAll(rhomid)
        return self.reference

    def save_dust_mass(self, ianalysis, dust_mass):
        filepath = self.dump_folder + "dust_mass.json"
        try:
            with open(filepath, "r") as fp:
                data = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"dust_mass": []}
        # drop entries written after the checkpoint we restarted from
        data["dust_mass"] = data["dust_mass"][:ianalysis]
        data["dust_mass"].append({"t": self.model.get_time(), "dust_mass": dust_mass})
        write_json(filepath, data)

    @simulation_setup
    def setup(self):
        model = self.model

        cfg = model.gen_default_config()
        cfg.set_artif_viscosity_VaryingCD10(
            alpha_min=av_alpha_min,
            alpha_max=av_alpha_max,
            sigma_decay=av_sigma_decay,
            alpha_u=av_alpha_u,
            beta_AV=av_beta_AV,
        )

        common_kwargs = {
            "nvar": ndust,
            "C_1_fluid": 0.1,
            "C_drift": 1.0,
            "cfl_density_threshold": 1e-50,
        }

        if self.limiter == DustLimiter.NONE:
            cfg.set_dust_mode_monofluid_tva(
                **common_kwargs, ensure_s_j_positivity=False, smooth_s_positivity_limiter=False
            )
        elif self.limiter == DustLimiter.SMOOTH:
            cfg.set_dust_mode_monofluid_tva(
                **common_kwargs, ensure_s_j_positivity=False, smooth_s_positivity_limiter=True
            )
        elif self.limiter == DustLimiter.BALLABIO:
            cfg.set_dust_mode_monofluid_tva(
                **common_kwargs, ensure_s_j_positivity=False, smooth_s_positivity_limiter=False
            )
            cfg.set_dust_ballabio_ts_limiter(True)
        elif self.limiter == DustLimiter.HARD:
            cfg.set_dust_mode_monofluid_tva(
                **common_kwargs, ensure_s_j_positivity=True, smooth_s_positivity_limiter=False
            )
        else:
            raise ValueError(f"Invalid dust limiter: {self.limiter}")

        cfg.set_dust_drag_epstein(gamma, mrn_distribution.grain_size, mrn_distribution.rho_grains)
        cfg.add_ext_force_vertical_disc_potential(central_mass=central_mass, R0=R0)
        cfg.add_ext_force_velocity_dissipation(eta=vel_dissipation_eta)
        cfg.set_neigh_cache_strategy(NeighCacheStrategy.SingleStage)
        cfg.set_show_cfl_detail(True)
        cfg.set_boundary_periodic()
        cfg.set_units(codeu)
        cfg.set_eos_isothermal(cs)
        cfg.print_status()
        model.set_solver_config(cfg)

        model.init_scheduler(scheduler_split_val, scheduler_merge_val)

        model.resize_simulation_box(bmin, bmax)

        setup = model.get_setup()
        gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
        setup.apply_setup(gen, insert_step=scheduler_split_val)

        pmass = model.total_mass_to_part_mass(totmass)
        model.set_particle_mass(pmass)
        print("Current part mass :", pmass)

        # Correct the barycenter
        analysis_barycenter = shamrock.model_sph.analysisBarycenter(model=model)
        barycenter, disc_mass = analysis_barycenter.get_barycenter()

        if shamrock.sys.world_rank() == 0:
            print(f"disc barycenter = {barycenter}")

        model.apply_position_offset((-barycenter[0], -barycenter[1], -barycenter[2]))

        model.remap_positions(remap_positions_z)
        model.set_field_value_lambda_f64("uint", uint_g)

        if random_vel_pert:
            rng = np.random.default_rng(42)

            ampl = dr / t_orb
            print(f"random velocity perturbation amplitude = {ampl}")

            def pert_func(r):
                return tuple(rng.uniform(-ampl, ampl, size=3))

            model.set_field_value_lambda_f64_3("vxyz", pert_func)

        model.set_cfl_cour(cfl_cour_setup)
        model.set_cfl_force(cfl_force_setup)

        model.timestep()

    # Declared before ``analysis`` so that, at t = tinject, the dust is
    # injected before the analysis runs.
    @callback(at_tsim=[tinject])
    def inject_dust(self, _):
        dic = self.ctx.collect_data()
        max_v = np.linalg.norm(dic["vxyz"], axis=1).max()
        print(f"max_v = {max_v}")

        if max_v > max_v_inject_threshold * cs:
            raise ValueError("max_v is too high, please increase the injection time")

        for k in range(ndust):

            def compute_sj_new(patchdata):
                return compute_sj_new_j(self.model, patchdata, k)

            self.model.overwrite_field_value_f64("s_j", compute_sj_new, k)

        self.model.set_cfl_cour(cfl_cour_inject)
        self.model.set_cfl_force(cfl_force_inject)

        self.model.set_dt(0.0)  # to help the corrector on next step after adding dust

    @callback(at_tsim=analysis_times)
    def analysis(self, j):
        model_time = self.model.get_time()

        reference_plots = None
        if model_time > tinject:
            reference = self.get_reference()
            reference.evolve_until(model_time - tinject)
            reference_plots = [
                {"z": sol.z_arr(), "rho": sol.rho * reference.rhoscale} for sol in reference.soluces
            ]

        dust_mass = self.analysis_dust_mass.get_dust_mass()

        # if all dust mass is zero (before the injection) replace by nans
        if np.max(dust_mass) == 0:
            dust_mass = [np.nan for _ in range(ndust)]

        self.save_dust_mass(j, dust_mass)

        dic, rho = self.collect_rho_z()
        s_j = dic["s_j"].reshape(-1, ndust)

        snapshot_data = {
            "z": dic["xyz"][:, 2],
            "s_j": s_j,
            "rho": rho,
            "time": model_time - tinject,
            "to_dens": to_dens,
            "grain_size_si": mrn_distribution.grain_size_si,
            "rho_dust_all": np.sum(s_j**2, axis=1) * to_dens,
            "reference_plots": reference_plots,
        }
        if shamrock.sys.world_rank() == 0:
            np.save(self.dump_folder + f"snapshot_data_{j:04d}.npy", snapshot_data)

    @callback(walltime_interval=60.0, at_tsim=[t_end])  # Checkpoint every minute
    def checkpoint(self, icheckpoint):
        self.do_checkpoint(icheckpoint, purge_old_dumps=True, keep_first=1, keep_last=3)


dump_folders = {}
for limiter, _ in limiter_panels:
    sim = DustySettleSimulation(limiter)
    sim.run()
    dump_folders[limiter] = sim.dump_folder
    del sim

# %%
# Figure
# ------------------------------------------
# Top row: vertical density profiles at the end of the simulation. Bottom row:
# relative variation of the mass of each dust species since the injection.


def load_snapshot(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = data.item()
    return data


def load_dust_mass(dump_folder):
    with open(dump_folder + "dust_mass.json") as fp:
        data = json.load(fp)["dust_mass"]
    t = np.array([d["t"] for d in data])
    values = np.array([d["dust_mass"] for d in data])
    return t, values


def plot_rho(ax, snapshot_data, dust_colors):
    z = snapshot_data["z"]
    s_j = snapshot_data["s_j"]
    rho = snapshot_data["rho"]
    to_dens = snapshot_data["to_dens"]
    rho_dust_all = snapshot_data["rho_dust_all"]
    reference_plots = snapshot_data["reference_plots"]

    for i in range(ndust):
        c = dust_colors[i]
        ax.scatter(z, s_j[:, i] ** 2 * to_dens, s=sz, color=c, edgecolors="none", rasterized=True)

        if reference_plots is not None:
            ax.plot(
                reference_plots[i]["z"],
                reference_plots[i]["rho"],
                "--",
                linewidth=1,
                color="0.0",
                alpha=0.7,
            )

    ax.scatter(
        z, rho * to_dens - rho_dust_all, s=sz, color="0.0", edgecolors="none", rasterized=True
    )
    ax.scatter(z, rho_dust_all, s=sz, color="0.5", edgecolors="none", rasterized=True)

    range_plot = 2.5 * H
    ax.set_xlabel(r"$z$")
    ax.set_yscale("log")
    ax.set_ylim(1e-20, 1e-8)
    ax.set_xlim(-range_plot, range_plot)


def plot_dust_mass_conservation(ax, t, dust_mass, dust_colors):
    # first analysis with dust (the dust mass is nan before the injection)
    i0 = np.argmax(~np.isnan(dust_mass)[:, 0])
    t = t - t[i0]

    for k in range(ndust):
        mh = dust_mass[:, k]
        ax.plot(t, (mh / mh[i0]) - 1, color=dust_colors[k], linewidth=1)

    total_dust_mass = np.sum(dust_mass, axis=1)
    ax.plot(
        t,
        (total_dust_mass / total_dust_mass[i0]) - 1,
        color="grey",
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(0, color="k", linestyle=":", linewidth=0.8)

    ax.set_xlabel("t")
    ax.set_yscale("symlog", linthresh=1e-8)


def plot_limiter_comparison(snapshot_index):
    snapshots = []
    mass_histories = []
    for limiter, label in limiter_panels:
        dump_folder = dump_folders[limiter]
        snapshots.append(
            (label, load_snapshot(dump_folder + f"snapshot_data_{snapshot_index:04d}.npy"))
        )
        mass_histories.append(load_dust_mass(dump_folder))

    grain_size_si = snapshots[0][1]["grain_size_si"]
    time = snapshots[0][1]["time"]

    dust_cmap = plt.colormaps[cmap]
    dust_norm = mcolors.LogNorm(vmin=grain_size_si.min(), vmax=grain_size_si.max() * 10)
    dust_colors = dust_cmap(dust_norm(grain_size_si))

    # Two-column journal width (~180 mm / 7.1 in for A&A, ApJ, MNRAS)
    fig = plt.figure(figsize=(9 * 1.2, 5 * 1.2), dpi=dpi)
    gs = fig.add_gridspec(2, 4, wspace=0.12, hspace=0.25, height_ratios=[1.8, 1])

    axes_rho = [fig.add_subplot(gs[0, 0])]
    for i in range(1, 4):
        axes_rho.append(fig.add_subplot(gs[0, i], sharey=axes_rho[0]))

    axes_mass = [fig.add_subplot(gs[1, 0])]
    for i in range(1, 4):
        axes_mass.append(fig.add_subplot(gs[1, i], sharey=axes_mass[0]))

    fig.suptitle(f"t = {time:.2f} [yr]")
    fig.subplots_adjust(left=0.12, right=1.05, top=0.90, bottom=0.10)

    for ax, (label, snapshot_data) in zip(axes_rho, snapshots):
        plot_rho(ax, snapshot_data, dust_colors)
        ax.set_title(label)

    axes_rho[0].set_ylabel(r"$\rho$ [kg/m$^3$]")
    for ax in axes_rho[1:]:
        ax.tick_params(labelleft=False)

    for ax, (t, dust_mass) in zip(axes_mass, mass_histories):
        plot_dust_mass_conservation(ax, t, dust_mass, dust_colors)

    axes_mass[0].set_ylabel(r"$\delta M_{\mathrm{dust}} / M_{\mathrm{dust},0}$")
    for ax in axes_mass[1:]:
        ax.tick_params(labelleft=False)

    gas_handle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="0.",
        markeredgecolor="none",
        label="gas",
    )
    dust_handle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="0.5",
        markeredgecolor="none",
        label="dust",
    )
    analytic_handle = Line2D(
        [0],
        [0],
        linestyle="--",
        color="0.0",
        alpha=0.7,
        label="reference",
    )
    axes_rho[-1].legend(
        handles=[gas_handle, dust_handle, analytic_handle], loc="upper right", fontsize=6
    )

    total_handle = Line2D(
        [0],
        [0],
        linestyle="--",
        color="grey",
        label="total dust mass",
    )
    axes_mass[-1].legend(handles=[total_handle], loc="upper right", fontsize=6)

    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
    dust_sm.set_array([])
    cbar = fig.colorbar(dust_sm, ax=axes_rho + axes_mass, pad=0.01, shrink=0.95, aspect=40)
    cbar.set_label(r"grain size $s$ [m]")

    out_path = base_folder + f"dustysettle_rho_limiters_{snapshot_index:04d}.pdf"
    fig.savefig(out_path)
    print(f"saved {out_path}")
    plt.close(fig)


if shamrock.sys.world_rank() == 0:
    plot_limiter_comparison(n_analysis)
