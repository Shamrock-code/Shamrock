"""
Extreme blast wave with GSPH (exact Riemann solver + Inutsuka V2)
====================================================================

Severe shock-tube problem from Inutsuka (2002, Section 4.3), with a
pressure ratio of :math:`3\\times10^{10}` and a peak Mach number around
:math:`10^5`. Both sides start at rest with equal density
(:math:`\\rho_L=\\rho_R=1`, :math:`P_L=3000`, :math:`P_R=10^{-7}`), so
this is a Sod-type (zero-velocity) discontinuity for which the analytic
solution is available via ``shamrock.phys.SodTube``. It stress-tests the
solver's stability at extreme pressure/Mach ratios rather than its
wave-pattern classification.
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Setup parameters: Inutsuka (2002) extreme blast wave

gamma = 1.4
rho_L, rho_R = 1.0, 1.0
P_L, P_R = 3000.0, 1e-7
u_L = P_L / ((gamma - 1) * rho_L)
u_R = P_R / ((gamma - 1) * rho_R)
resol = 32
# The disturbance from such an extreme pressure ratio moves fast enough that a
# unit-half-width box (as used by the other gsph examples) is under-resolved
# and lets the periodic boundary interact with the shock well before
# t_target; stretching the box in x (same particle spacing dr) fixes both.
domain_scale = 2.0

# %%
# Setup the solver: GSPH with the exact Riemann solver and the Inutsuka V2
# effective volume/face force formulation.

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
cfg = model.gen_default_config()
cfg.set_riemann_exact()
cfg.set_force_inutsuka_v2()
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)

# %%
# Setup the initial conditions: two equal-density half-boxes with an
# extreme pressure contrast (via internal energy).

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)
xs *= domain_scale
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

vol_b = xs * ys * zs
totmass = (rho_R * vol_b) + (rho_L * vol_b)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.3)
model.set_cfl_force(0.3)

# %%
# Run the simulation

t_target = 0.015
model.evolve_until(t_target)

# %%
# Collect the particle data

data = ctx.collect_data()

points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])
uint = np.array(data["uint"])

x = points[:, 0]
vx = velocities[:, 0]
rho = pmass * (hfact / hpart) ** 3
P = (gamma - 1) * rho * uint

# %%
# Analytic Sod-type solution for comparison (valid since both sides start
# at rest, only pressure/density differ)

sod = shamrock.phys.SodTube(gamma=gamma, rho_1=rho_L, P_1=P_L, rho_5=rho_R, P_5=P_R)

arr_x = np.linspace(-xs, xs, 2000)
arr_rho = np.zeros_like(arr_x)
arr_vx = np.zeros_like(arr_x)
arr_P = np.zeros_like(arr_x)
for i, xi in enumerate(arr_x):
    arr_rho[i], arr_vx[i], arr_P[i] = sod.get_value(t_target, xi)

# %%
# Plot the particle data against the analytic solution (log scale on
# pressure to accommodate the ~10^10 dynamic range)

fig, ax = plt.subplots(1, 2, figsize=(13, 6), dpi=125)

ax[0].scatter(x, rho, rasterized=True, s=6 * np.ones(x.shape), label="density")
ax[0].scatter(x, vx / 1e3, rasterized=True, s=6 * np.ones(x.shape), label="velocity / 1e3")
ax[0].plot(arr_x, arr_rho, ls="--", lw=2.0, color="black", label="analytic")
ax[0].plot(arr_x, arr_vx / 1e3, ls="--", lw=2.0, color="black")
ax[0].set_xlim(-xs, xs)
ax[0].set_xlabel("x")
ax[0].set_title("density, velocity")
ax[0].legend(loc=0)
ax[0].grid(alpha=0.3)

ax[1].scatter(x, P, rasterized=True, s=6 * np.ones(x.shape), label="pressure")
ax[1].plot(arr_x, arr_P, ls="--", lw=2.0, color="black", label="analytic")
ax[1].set_xlim(-xs, xs)
ax[1].set_yscale("log")
ax[1].set_xlabel("x")
ax[1].set_title("pressure (log scale)")
ax[1].legend(loc=0)
ax[1].grid(alpha=0.3)

fig.suptitle(f"GSPH extreme blast wave, Mach~1e5 (exact solver + Inutsuka V2), t={t_target}")

if shamrock.sys.world_rank() == 0:
    plt.show()
