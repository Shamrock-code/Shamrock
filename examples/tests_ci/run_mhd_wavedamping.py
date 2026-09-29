"""
Wave damping test for the non-ideal MHD ambipolar diffusion
================================================================

(Choi et al. 2009, Wurster, Price & Ayliffe 2014, Wurster, Price & Bate 2016 section 4.1) at the same
Resolution nx=128 as in the Phantom paper section 5.7.1.
"""

import numpy as np

import shamrock

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Parameters of the test (matches examples/sph/run_mhd_wavedamping.py / Phantom nx=128)

L2_ERROR_THRESHOLD = 7.5e-5

Lx = 1.0  # box length
dr = 1 / 128  # particle spacing (nx=128, matching Phantom's benchmark resolution)
rho0 = 1.0  # initial density
Bx0 = 1.0  # background field in x
C_ADc = 0.01  # ambipolar diffusion coefficient (Phantom convention)
cs = 1.0  # isothermal sound speed
t_target = 5.0  # total simulation time
dt_dump = 0.01  # dump interval (matches Phantom's L2-error sampling interval)

# Unit system chosen so that mu_0 = 1 exactly in code units
codeu = shamrock.UnitSystem(
    unit_time=1.0,
    unit_length=1.0,
    unit_mass=1.2566370621219e-06,
)
ucte = shamrock.Constants(codeu)
mu_0 = ucte.mu_0()  # = 1 in these units

vA = Bx0 / np.sqrt(rho0)  # Alfven speed
etaAD = C_ADc * vA * vA  # ambipolar diffusivity

if shamrock.sys.world_rank() == 0:
    print(f"mu_0 = {mu_0}, vA = {vA:.3f}, etaAD = {etaAD:.3e}")

# %%
# Create context and SPH model

ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="C4")

# %%
# Set up simulation configuration
# All artificial dissipation terms are turned off (alpha_B, alpha_AV, beta_AV, sigma_mhd),
# matching Phantom's stated methodology for this test.

cfg = model.gen_default_config()
cfg.set_units(codeu)
cfg.set_artif_viscosity_None()
cfg.set_NonIdealMHD(
    sigma_mhd=0, sigma_u=0, etaO=0, etaH=0, etaAD=etaAD, alpha_B=0, alpha_AV=0, beta_AV=0
)
cfg.set_boundary_periodic()
cfg.set_eos_isothermal(cs)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e6), 1)

# %%
# Generate particle distribution in an FCC lattice

bmin = (-Lx / 2.0, -np.sqrt(3) / 4.0 * Lx, -np.sqrt(6) / 4.0 * Lx)
bmax = (Lx / 2.0, np.sqrt(3) / 4.0 * Lx, np.sqrt(6) / 4.0 * Lx)

bmin, bmax = model.get_ideal_fcc_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax
Lx_actual = xM - xm

model.resize_simulation_box(bmin, bmax)
model.add_cube_fcc_3d(dr, bmin, bmax)

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho0 * vol_b
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

# %%
# Set initial conditions: background field in x, sinusoidal velocity perturbation in z.

k = 2 * np.pi / Lx_actual
v0 = 0.01 * vA


def B_func(r):
    return (Bx0, 0.0, 0.0)


def vel_func(r):
    x, y, z = r
    vz = v0 * np.sin(k * (x - xm))
    return (0.0, 0.0, vz)


def u_func(r):
    return 0.0


model.set_field_value_lambda_f64_3("B/rho", B_func)
model.set_field_value_lambda_f64_3("vxyz", vel_func)
model.set_field_value_lambda_f64("uint", u_func)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

# %%
# Time loop with data collection

times = []
Brmsz = []

t_sum = 0.0
next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target + 1e-12:
    model.evolve_until(next_dt_target)
    t_now = model.get_time()

    data = ctx.collect_data()
    h_arr = data["hpart"]
    hfac = model.get_hfact()
    rho = pmass * (hfac / h_arr) * (hfac / h_arr) * (hfac / h_arr)
    Bz = data["B/rho"][:, 2] * rho

    rms_z = np.sqrt(np.mean(Bz**2))

    times.append(t_now)
    Brmsz.append(rms_z)

    if shamrock.sys.world_rank() == 0:
        print(f"t = {t_now:.3f}, rms Bz = {rms_z:.5f}")

    next_dt_target += dt_dump

times = np.array(times)
Brmsz = np.array(Brmsz)

# %%
# Analytical solution (damped Alfven wave dispersion relation, see
# examples/sph/run_mhd_wavedamping.py for the full derivation)

k = 2 * np.pi / Lx_actual
quadb = k**2 * etaAD
quadc = -((k * vA) ** 2)
omegaI = -0.5 * quadb
omegaR = 0.5 * np.sqrt(-(quadb**2) - 4 * quadc)
h0 = v0 * Bx0 / (vA * np.sqrt(2.0))

if shamrock.sys.world_rank() == 0:
    print(f"omegaR = {omegaR:.4f}, omegaI = {omegaI:.4f}, h0 = {h0:.4f}")

theory_at_times = h0 * np.abs(np.sin(omegaR * times)) * np.exp(omegaI * times)
l2_error = np.sqrt(np.mean((Brmsz - theory_at_times) ** 2))

if shamrock.sys.world_rank() == 0:
    print(f"L2 error (rms Bz vs theory) = {l2_error:.3e} (threshold = {L2_ERROR_THRESHOLD:.3e})")

# %%
# Check the L2 error against Phantom's published benchmark threshold

to_raise = []

if l2_error > L2_ERROR_THRESHOLD:
    to_raise.append(
        f"L2 error of rms Bz vs analytical solution is out of tolerance: "
        f"{l2_error:.3e} > {L2_ERROR_THRESHOLD:.3e}"
    )

for to_raise_item in to_raise:
    raise ValueError(to_raise_item)
