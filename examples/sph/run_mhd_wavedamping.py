"""
Wave damping test in SPH with non-ideal MHD
===========================================

This example runs the wave damping test, aimed at evaluating the behaviour of the ambipolar diffusion term.
The RMS of the magnetic field components is
monitored and compared to the analytical dispersion relation.
"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

shamrock.enable_experimental_features()

# Initialize shamrock (if not already done by the executable)
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# Use shamrock plotting style
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Define physical parameters and unit system
# ------------------------------------------
# We use a unit system where mu_0 = 1 in code units (defined via UnitSystem).
# The code unit system is set to have unit_length = 1, unit_mass = 1.2566370621219e-06
# so that mu_0 = 1 exactly.

Lx = 1.0  # box length
dr = 0.09  # particle spacing
rho0 = 1.0  # initial density
Bx0 = 1.0  # background field in x
C_ADc = 0.01  # ambipolar diffusion coefficient (Phantom convention)
cs = 1.0  # isothermal sound speed
t_target = 5.0  # total simulation time
dt_dump = 0.01 * t_target  # dump interval

# Unit system and constants
codeu = shamrock.UnitSystem(
    unit_time=1.0,
    unit_length=1.0,
    unit_mass=1.2566370621219e-06,
)
ucte = shamrock.Constants(codeu)
mu_0 = ucte.mu_0()  # = 1 in these units
c = ucte.c()  # speed of light (used only for unit conversion)

# Derived quantities
vA = Bx0 / np.sqrt(rho0)  # Alfven speed
etaAD_cgs = C_ADc * vA * vA  # ambipolar diffusivity (cgs-like)
etaAD_si = etaAD_cgs * (4 * np.pi / c)  # SI conversion (not used in code)

print(f"mu_0 = {mu_0}, vA = {vA:.3f}, etaAD = {etaAD_cgs:.3e}")

# %%
# Create context and SPH model
# ----------------------------
ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="C4")

# %%
# Set up simulation configuration
# -------------------------------
cfg = model.gen_default_config()
cfg.set_units(codeu)
cfg.set_artif_viscosity_None()  # no artificial viscosity
cfg.set_NonIdealMHD(sigma_mhd=1, sigma_u=0, etaO=0, etaH=0, etaAD=etaAD_cgs)
cfg.set_boundary_periodic()  # periodic boundaries in all directions
cfg.set_eos_isothermal(cs)  # isothermal equation of state
cfg.print_status()
model.set_solver_config(cfg)

# Initialize scheduler and particle container
model.init_scheduler(int(1e6), 1)

# %%
# Generate particle distribution in an FCC lattice
# ------------------------------------------------
bmin = (-Lx / 2.0, -np.sqrt(3) / 4.0 * Lx, -np.sqrt(6) / 4.0 * Lx)
bmax = (Lx / 2.0, np.sqrt(3) / 4.0 * Lx, np.sqrt(6) / 4.0 * Lx)

# Adjust box to exactly fit the lattice
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
# Set initial conditions
# ----------------------
# Background magnetic field in x, and a sinusoidal velocity perturbation in z.
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

# %%
# Set CFL parameters
# ------------------
model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

# %%
# Prepare storage for time series and output directory
# ----------------------------------------------------
times = []
Brmsx = []
Brmsy = []
Brmsz = []

dump_folder = "_wave_dump"
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)

# %%
# Time loop with data collection
# ------------------------------
t_sum = 0.0
i_dump = 0
next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target + 1e-12:
    # Evolve until next dump time
    model.evolve_until(next_dt_target)
    t_now = model.get_time()

    # Collect particle data from the context
    data = ctx.collect_data()
    h_arr = data["hpart"]
    hfac = 1.0
    rho = pmass * (hfac / h_arr) * (hfac / h_arr) * (hfac / h_arr)
    Bx = data["B/rho"][:, 0] * rho
    By = data["B/rho"][:, 1] * rho
    Bz = data["B/rho"][:, 2] * rho

    # Compute RMS values
    rms_x = np.sqrt(np.mean(Bx**2))
    rms_y = np.sqrt(np.mean(By**2))
    rms_z = np.sqrt(np.mean(Bz**2))

    times.append(t_now)
    Brmsx.append(rms_x)
    Brmsy.append(rms_y)
    Brmsz.append(rms_z)

    # (Optional) write VTK for visualisation
    model.do_vtk_dump(os.path.join(dump_folder, f"wave_{i_dump:04d}.vtk"), True)

    print(f"t = {t_now:.3f}, rms Bz = {rms_z:.5f}")

    i_dump += 1
    next_dt_target += dt_dump

# Convert to numpy arrays
times = np.array(times)
Brmsx = np.array(Brmsx)
Brmsy = np.array(Brmsy)
Brmsz = np.array(Brmsz)

# %%
# Analytical solution
# -------------------
# For the damping of a sinusoidal Alfven wave with ambipolar diffusion,
# the dispersion relation gives:
#   omega = omega_R + i omega_I
# with omega_I = - (k^2 etaAD)/2   (damping rate)
# and   omega_R = 0.5 * sqrt( - (k^2 etaAD)^2 - 4 (k vA)^2 )
# The z-component of B (the perturbed component) evolves as:
#   Bz(t) = Bz(0) * |sin(omega_R t)| * exp(omega_I t)
# where Bz(0) = (rho0 * v0 * Bx0) / (vA * sqrt(2))

Lx = 1.0
# Phantom's exact dispersion relation
Bx0 = 1.0
rho0 = 1.0
C_ADc = 0.01  # ion-neutral coupling, same as Phantom
vA = Bx0 / np.sqrt(rho0)  # no mu_0 since mu_0=1 in your code units
v0 = 0.01 * vA
k = 2 * np.pi / Lx  # = 2*pi since Lx=1

etaAD_cgs = C_ADc * vA * vA
etaAD_si = etaAD_cgs * 4 * np.pi / c
etaAD = etaAD_cgs

quadb = (k) ** 2 * etaAD
quadc = -((k * vA) ** 2)
omegaI = -0.5 * quadb  # negative = damping
omegaR = 0.5 * np.sqrt(-(quadb**2) - 4 * quadc)

h0 = (4 * np.pi) * v0 * Bx0 / (vA * np.sqrt(2.0))

print(f"omegaR = {omegaR:.4f}, omegaI = {omegaI:.4f}, h0 = {h0:.4f}")

time_th = np.linspace(0, t_target, 1000)
theory = h0 * np.abs(np.sin(omegaR * time_th)) * np.exp(omegaI * time_th)

# %%
# Plot results
# ------------
fig, axs = plt.subplots(1, 4, figsize=(12, 8))

axs[0].plot(times, Brmsz, "b-", linewidth=2)
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("rms Bz")
axs[0].grid(alpha=0.3)

axs[1].plot(times, Brmsy, "g-", linewidth=2)
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("rms By")
axs[1].grid(alpha=0.3)

axs[2].plot(times, Brmsx, "m-", linewidth=2)
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("rms Bx")
axs[2].grid(alpha=0.3)

axs[3].plot(time_th, theory, "r--", linewidth=2)
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("rms Bz theory")
axs[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(dump_folder, "wave_damping_analysis.png"), dpi=150)
plt.show()

print("Analysis completed. Results saved in", dump_folder)
