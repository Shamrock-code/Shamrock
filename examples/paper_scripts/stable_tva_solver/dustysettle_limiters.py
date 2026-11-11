"""
Dusty settling: comparison of the TVA positivity limiters
=========================================================

Run the dusty settling test (multi-grain TVA monofluid in a local stratified
box) once per positivity limiter (none, Ballabio, smooth, hard), then plot the
vertical density profiles at the final time and the dust mass conservation
history of the four runs side by side.

The SPH (TVA) profiles are compared to a reference where the dust settles on a
gas at rest (dashed black) and to the full gas + dust multifluid equations
solved with a simple 1D scheme (dashed light blue), which shows how far the TVA
results are from the non-TVA solution. The multifluid reference is computed
when plotting, from the midplane density stored by each run.

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
multifluid_Nz = 4000  # cells of the multifluid (non-TVA) reference, must be even
multifluid_order = 2  # order of the multifluid scheme (1 or 2)

# Paths
base_folder = f"_to_trash/stable_tva_solver/dusty_settle_lz{lz}/"

# Plotting
cmap = "plasma"
dpi = 250
sz = 1
multifluid_color = "lightskyblue"

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
# Multifluid (non-TVA) reference solution
# ------------------------------------------
#
# The reference above evolves each dust species on a gas held at rest. To see how far the
# TVA results are from the general dust-gas problem, we also solve the full 1D multifluid
# equations: one continuity and one momentum equation for the gas and for each dust species,
# isothermal gas pressure, vertical gravity, Epstein drag with back-reaction on the gas, and
# the same velocity dissipation as the SPH run (acting on the barycentric velocity).
#
# The scheme is kept simple: staggered grid (densities at cell centres, velocities at faces,
# closed walls), upwind continuity, upwind velocity advection (Godunov flux of v^2/2), explicit
# pressure / gravity / dissipation and a drag step solved exactly per face (conserves the
# momentum and keeps the terminal velocity exact). With ``order=1`` the reconstruction is
# piecewise constant with semi-implicit Euler + backward-Euler drag; with ``order=2`` it is
# van Leer limited (MUSCL) with a midpoint predictor-corrector and Crank-Nicolson drag. The gas
# starts in discrete hydrostatic equilibrium, so it stays at rest without dust.


def vertical_gravity(zbar):
    """Vertical gravity in units of vK * OmegaK (same as in ``S_v``)."""
    return -zbar / (1 + zbar**2) ** 1.5


def van_leer_slope(q):
    """van Leer limited slope (per cell) along the last axis, zero on the first and last cell."""
    slope = np.zeros_like(q)
    prod = (q[..., 1:-1] - q[..., :-2]) * (q[..., 2:] - q[..., 1:-1])
    sum_ = q[..., 2:] - q[..., :-2]
    slope[..., 1:-1] = np.where(prod > 0, 2 * prod / np.where(prod > 0, sum_, 1.0), 0.0)
    return slope


def continuity_rate(rho, v, hz, second_order):
    """-d(rho v)/dz for cell-centred rho, v at faces (zero at the walls), upwind face rho."""
    slope = van_leer_slope(rho) if second_order else 0 * rho
    v_in = v[..., 1:-1]
    rho_l = rho[..., :-1] + 0.5 * slope[..., :-1]
    rho_r = rho[..., 1:] - 0.5 * slope[..., 1:]
    flux = np.zeros_like(v)
    flux[..., 1:-1] = v_in * np.where(v_in > 0, rho_l, rho_r)
    return -(flux[..., 1:] - flux[..., :-1]) / hz


def advection_rate(v, hz, second_order):
    """v dv/dz = d(v^2/2)/dz on the faces (zero on the walls), Godunov Burgers flux at the
    cell centres from the velocities of the two neighbouring faces."""
    slope = van_leer_slope(v) if second_order else 0 * v
    u_l = v[..., :-1] + 0.5 * slope[..., :-1]
    u_r = v[..., 1:] - 0.5 * slope[..., 1:]
    f_shock = 0.5 * np.maximum(u_l**2, u_r**2)
    f_rarefaction = np.where((u_l < 0) & (u_r > 0), 0.0, 0.5 * np.minimum(u_l**2, u_r**2))
    f = np.where(u_l > u_r, f_shock, f_rarefaction)
    adv = np.zeros_like(v)
    adv[..., 1:-1] = (f[..., 1:] - f[..., :-1]) / hz
    return adv


def to_faces(rho):
    """Cell-centred to face densities (walls take the adjacent cell)."""
    rho_f = np.empty(rho.shape[:-1] + (rho.shape[-1] + 1,))
    rho_f[..., 1:-1] = 0.5 * (rho[..., :-1] + rho[..., 1:])
    rho_f[..., 0] = rho[..., 0]
    rho_f[..., -1] = rho[..., -1]
    return rho_f


def drag_solve(r_g, r_d, rho_g_f, rho_d_f, a):
    """
    Solve v_d = r_d - a (v_d - v_g) for each species, with the total momentum
    rho_g v_g + sum rho_d v_d = rho_g r_g + sum rho_d r_d (a = dt / t_j for backward Euler,
    dt / (2 t_j) for Crank-Nicolson).
    """
    w = 1 / (1 + a)
    mom = rho_g_f * r_g + np.sum(rho_d_f * r_d, axis=0)
    v_g = (mom - np.sum(rho_d_f * r_d * w, axis=0)) / (rho_g_f + np.sum(rho_d_f * a * w, axis=0))
    v_d = (r_d + a * v_g[None, :]) * w
    return v_g, v_d


class MultifluidDustySettle:
    def __init__(self, Nz, cs_bar, eta_bar, inv_ts_coef, dust_to_gas, order=2, cfl=0.4):
        """
        Dimensionless units: z / R0, v / vK, t * OmegaK, rho / rho_mid.

        cs_bar      : isothermal sound speed
        eta_bar     : velocity dissipation rate
        inv_ts_coef : 1/t_j = inv_ts_coef[j] * rho_g (Epstein drag, gas density)
        dust_to_gas : initial dust-to-gas ratio of each species (before the mask)
        order       : 1 or 2, order of the scheme in space and time
        """
        if Nz % 2 != 0:
            raise ValueError("Nz must be even (the midplane is a face)")
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")

        self.cs2 = cs_bar**2
        self.eta = eta_bar
        self.inv_ts_coef = np.asarray(inv_ts_coef)
        self.order = order
        self.cfl = cfl

        zout = reference_zrange * H / R0
        self.hz = 2 * zout / Nz
        self.zf = np.linspace(-zout, zout, Nz + 1)
        self.zc = 0.5 * (self.zf[:-1] + self.zf[1:])
        self.g_f = vertical_gravity(self.zf)
        self.g_f[[0, -1]] = 0

        # discrete hydrostatic equilibrium: cs^2 (ln rho_{i+1} - ln rho_i) / hz = g_{i+1/2}
        ln_rho = np.zeros(Nz)
        ln_rho[1:] = np.cumsum(self.hz * self.g_f[1:-1] / self.cs2)
        ln_rho += -(self.zc[Nz // 2] ** 2) / (2 * self.cs2) - ln_rho[Nz // 2]
        self.rho_g = np.exp(ln_rho)
        self.v_g = np.zeros(Nz + 1)

        mask = 1 / (1 + np.exp((np.abs(self.zc) - 1.75 * H / R0) / (H / R0 / 16)))
        self.rho_d = np.array([dtg * mask * self.rho_g for dtg in dust_to_gas])
        self.v_d = np.zeros((len(dust_to_gas), Nz + 1))

        self.tbar = 0.0

    def get_dt(self):
        vmax = max(self.cs2**0.5 + np.abs(self.v_g).max(), np.abs(self.v_d).max())
        return self.cfl * self.hz / vmax

    def density_rates(self, rho_g, v_g, rho_d, v_d):
        second_order = self.order == 2
        drho_g = continuity_rate(rho_g, v_g, self.hz, second_order)
        drho_d = continuity_rate(rho_d, v_d, self.hz, second_order)
        return drho_g, drho_d

    def accelerations(self, rho_g, v_g, rho_d, v_d):
        """Accelerations other than the drag."""
        hz = self.hz
        second_order = self.order == 2

        rho_g_f = to_faces(rho_g)
        rho_d_f = to_faces(rho_d)
        mom = rho_g_f * v_g + np.sum(rho_d_f * v_d, axis=0)
        v_bary = mom / (rho_g_f + np.sum(rho_d_f, axis=0))

        grad_P_over_rho = np.zeros_like(v_g)
        grad_P_over_rho[1:-1] = self.cs2 * np.diff(np.log(rho_g)) / hz

        acc_g = -advection_rate(v_g, hz, second_order) - grad_P_over_rho + self.g_f
        acc_d = -advection_rate(v_d, hz, second_order) + self.g_f
        return acc_g - self.eta * v_bary, acc_d - self.eta * v_bary

    def euler_step(self, dt):
        """
        Semi-implicit Euler step, returns the new state: densities first, then the velocities
        with the forces of the new densities (stable for sound waves) and backward-Euler drag.
        """
        drho_g, drho_d = self.density_rates(self.rho_g, self.v_g, self.rho_d, self.v_d)
        rho_g = self.rho_g + dt * drho_g
        rho_d = self.rho_d + dt * drho_d
        acc_g, acc_d = self.accelerations(rho_g, self.v_g, rho_d, self.v_d)
        rho_g_f = to_faces(rho_g)
        a = dt * self.inv_ts_coef[:, None] * rho_g_f[None, :]
        v_g, v_d = drag_solve(
            self.v_g + dt * acc_g, self.v_d + dt * acc_d, rho_g_f, to_faces(rho_d), a
        )
        return rho_g, v_g, rho_d, v_d

    def midpoint_step(self, dt):
        """
        Predictor (half semi-implicit Euler step) - corrector (rates at mid step, Crank-Nicolson
        drag). With this predictor the step conserves the energy of sound waves (no growth).
        """
        half = self.euler_step(0.5 * dt)
        drho_g, drho_d = self.density_rates(*half)
        acc_g, acc_d = self.accelerations(*half)
        rho_g_h, rho_d_h = half[0], half[2]

        rho_g_f = to_faces(rho_g_h)
        rho_d_f = to_faces(rho_d_h)
        a = 0.5 * dt * self.inv_ts_coef[:, None] * rho_g_f[None, :]

        # explicit half of the drag at t^n
        drag_d = -a * (self.v_d - self.v_g[None, :])
        drag_g = -np.sum(rho_d_f * drag_d, axis=0) / rho_g_f

        v_g, v_d = drag_solve(
            self.v_g + dt * acc_g + drag_g,
            self.v_d + dt * acc_d + drag_d,
            rho_g_f,
            rho_d_f,
            a,
        )
        return self.rho_g + dt * drho_g, v_g, self.rho_d + dt * drho_d, v_d

    def advance(self, dt):
        if self.order == 1:
            rho_g, v_g, rho_d, v_d = self.euler_step(dt)
        else:
            rho_g, v_g, rho_d, v_d = self.midpoint_step(dt)

        v_g[[0, -1]] = 0
        v_d[:, [0, -1]] = 0

        self.rho_g, self.v_g, self.rho_d, self.v_d = rho_g, v_g, rho_d, v_d
        self.tbar += dt

    def advance_until(self, tfinal):
        while tfinal - self.tbar > 1e-12:
            self.advance(min(self.get_dt(), tfinal - self.tbar))

    def epsilon(self):
        """Dust fraction of each species rho_j / rho, cell centred."""
        return self.rho_d / (self.rho_g + np.sum(self.rho_d, axis=0))


class MultifluidDustySettleAll:
    def __init__(self, rhomid):
        self.rhomid = rhomid

        self.vK = kep_profile(R0) * codeu.to("m") / codeu.to("s")
        self.OmegaK = omega_k(R0) * codeu.to("s") ** -1
        self.tK = 1 / self.OmegaK

        self.rscale = R0
        self.rhoscale = self.rhomid
        self.vscale = self.vK
        self.tscale = self.tK

        self.cs = cs * codeu.to("m") / codeu.to("s")

        inv_ts_coef = []
        dust_to_gas = []
        for j in range(ndust):
            eps_j = mrn_distribution.mrn_weight[j] * epsilon_base
            dust_to_gas.append(eps_j / (1 - epsilon_base))

            rhoeff = mrn_distribution.rho_grains_si_edges[j] * np.sqrt(np.pi * gamma / 8)
            s_j = mrn_distribution.grain_size_si[j]
            inv_ts_coef.append(self.cs * self.rhoscale / (rhoeff * s_j * self.OmegaK))

        self.sol = MultifluidDustySettle(
            multifluid_Nz,
            cs_bar=self.cs / self.vK,
            eta_bar=vel_dissipation_eta / omega_k(R0),
            inv_ts_coef=inv_ts_coef,
            dust_to_gas=dust_to_gas,
            order=multifluid_order,
        )

    def evolve_until(self, t):
        self.sol.advance_until(t * codeu.to("s") / self.tscale)


# %%
# Simulation
# ------------------------------------------


def load_rhomid(dump_folder):
    with open(dump_folder + "reference_rhomid.json", "r") as fp:
        return json.load(fp)["rhomid"]


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
                rhomid = load_rhomid(self.dump_folder)
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
                **common_kwargs,
                ensure_s_j_positivity=False,
                smooth_s_positivity_limiter=True,
                pair_gate_s_positivity_limiter=True,
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


def compute_multifluid_plots(rhomid, snapshot_index, time):
    """
    Multifluid (non-TVA) reference at ``time`` after the injection, normalised by
    the midplane density ``rhomid`` (same as the gas-at-rest reference).

    It does not depend on the limiter, so it is computed once for all the panels.
    It takes about a minute, so the result is cached in ``base_folder`` and only
    recomputed when its inputs change.
    """
    key = {"rhomid": rhomid, "time": time, "Nz": multifluid_Nz, "order": multifluid_order}

    cache_path = base_folder + f"multifluid_reference_{snapshot_index:04d}.npy"
    if os.path.exists(cache_path):
        cached = load_snapshot(cache_path)
        if cached["key"] == key:
            return cached["plots"]

    multifluid = MultifluidDustySettleAll(rhomid)
    multifluid.evolve_until(time)
    mf = multifluid.sol
    plots = [
        {"z": mf.zc * multifluid.rscale, "rho": mf.rho_d[i] * multifluid.rhoscale}
        for i in range(ndust)
    ]
    np.save(cache_path, {"key": key, "plots": plots})
    return plots


def plot_rho(ax, snapshot_data, multifluid_plots, dust_colors):
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

        ax.plot(
            multifluid_plots[i]["z"],
            multifluid_plots[i]["rho"],
            "--",
            linewidth=1,
            color=multifluid_color,
            zorder=1.9,  # below the reference lines (zorder 2), above the particles (zorder 1)
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
        snapshot_data = load_snapshot(dump_folder + f"snapshot_data_{snapshot_index:04d}.npy")
        snapshots.append((label, snapshot_data))
        mass_histories.append(load_dust_mass(dump_folder))

    grain_size_si = snapshots[0][1]["grain_size_si"]
    time = snapshots[0][1]["time"]

    # same multifluid reference for all the panels, normalised by the midplane
    # density of the first run
    print("computing the multifluid reference")
    multifluid_plots = compute_multifluid_plots(
        load_rhomid(dump_folders[limiter_panels[0][0]]), snapshot_index, time
    )

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
        plot_rho(ax, snapshot_data, multifluid_plots, dust_colors)
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
    multifluid_handle = Line2D(
        [0],
        [0],
        linestyle="--",
        color=multifluid_color,
        label="multifluid (non-TVA)",
    )
    axes_rho[-1].legend(
        handles=[gas_handle, dust_handle, analytic_handle, multifluid_handle],
        loc="upper right",
        fontsize=6,
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
