"""
Sod shock tube with GSPH (exact Riemann solver + Inutsuka V2)
===============================================================

Runs a 3D Sod shock tube using the Godunov SPH (GSPH) solver with the
exact Riemann solver (Toro 2009) and the Inutsuka (2002) effective
volume/face force formulation, and compares the result against the
analytic solution.
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Setup parameters

gamma = 1.4
rho_L, rho_R = 1.0, 0.125
P_L, P_R = 1.0, 0.1
fact = (rho_L / rho_R) ** (1.0 / 3.0)
u_L = P_L / ((gamma - 1) * rho_L)
u_R = P_R / ((gamma - 1) * rho_R)
resol = 64

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
# Setup the initial conditions: two half-boxes at different density/pressure

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr * fact, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

vol_b = xs * ys * zs
totmass = (rho_R * vol_b) + (rho_L * vol_b)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

# %%
# Run the simulation

t_target = 0.245
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
# Analytic Sod solution for comparison

sod = shamrock.phys.SodTube(gamma=gamma, rho_1=rho_L, P_1=P_L, rho_5=rho_R, P_5=P_R)

arr_x = np.linspace(-0.5, 0.5, 1000)
arr_rho = np.zeros_like(arr_x)
arr_vx = np.zeros_like(arr_x)
arr_P = np.zeros_like(arr_x)
for i, xi in enumerate(arr_x):
    arr_rho[i], arr_vx[i], arr_P[i] = sod.get_value(t_target, xi)

# %%
# Plot the particle data against the analytic solution

fig, ax = plt.subplots(figsize=(9, 6), dpi=125)

ax.scatter(x, rho, rasterized=True, s=6 * np.ones(x.shape), label="density")
ax.scatter(x, vx, rasterized=True, s=6 * np.ones(x.shape), label="velocity")
ax.scatter(x, P, rasterized=True, s=6 * np.ones(x.shape), label="pressure")

ax.plot(arr_x, arr_rho, ls="--", lw=2.0, color="black", label="analytic")
ax.plot(arr_x, arr_vx, ls="--", lw=2.0, color="black")
ax.plot(arr_x, arr_P, ls="--", lw=2.0, color="black")

ax.set_xlim(-0.5, 0.5)
ax.set_xlabel("x")
ax.set_title(f"GSPH Sod shock tube (exact Riemann solver + Inutsuka V2), t={t_target}")
ax.legend(loc=0)
ax.grid(alpha=0.3)

if shamrock.sys.world_rank() == 0:
    plt.show()
