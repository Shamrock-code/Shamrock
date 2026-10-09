"""
Contact-discontinuity advection with GSPH (exact Riemann solver + Inutsuka V2)
================================================================================

Advects a density jump at uniform velocity with equal pressure on both
sides. Since :math:`p_L=p_R` and :math:`u_L=u_R`, the exact Riemann
solution at the interface is a pure contact discontinuity: no shock or
rarefaction forms, and the analytic solution is simply the initial
density jump translated rigidly at speed :math:`u_0`. This isolates how
much a scheme numerically diffuses a density (compositional) interface
carried by a uniform flow, independent of any shock-capturing behaviour.
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Setup parameters: equal pressure both sides, uniform advection velocity u0

gamma = 1.4
rho_L, rho_R = 1.0, 0.5
P0 = 1.0
u0 = 1.0
fact = (rho_L / rho_R) ** (1.0 / 3.0)
uint_L = P0 / ((gamma - 1) * rho_L)
uint_R = P0 / ((gamma - 1) * rho_R)
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
# Setup the initial conditions: a density jump at x=0, uniform pressure and
# a single uniform advection velocity u0 on both sides.

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr * fact, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", uint_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", uint_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
model.set_field_in_box(
    "vxyz", "f64_3", (u0, 0.0, 0.0), (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2)
)

vol_b = xs * ys * zs
totmass = (rho_R * vol_b) + (rho_L * vol_b)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

# %%
# Run the simulation: t_target is chosen well below the time needed for the
# interface to reach the periodic boundary (u0 * t_target << xs)

t_target = 0.2
model.evolve_until(t_target)

# %%
# Collect the particle data

data = ctx.collect_data()

points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])

x = points[:, 0]
vx = velocities[:, 0]
rho = pmass * (hfact / hpart) ** 3

# %%
# Analytic solution: the density jump translates rigidly to x = u0 * t_target,
# velocity stays uniform at u0 everywhere.

x_interface = u0 * t_target
arr_x = np.linspace(-0.5, 0.5, 1000)
arr_rho = np.where(arr_x < x_interface, rho_L, rho_R)

# %%
# Plot the particle data against the analytic solution

fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=125)

ax[0].scatter(x, rho, rasterized=True, s=6 * np.ones(x.shape), label="density")
ax[0].plot(arr_x, arr_rho, ls="--", lw=2.0, color="black", label="analytic")
ax[0].axvline(x_interface, color="grey", lw=1, ls=":")
ax[0].set_xlim(-0.5, 0.5)
ax[0].set_xlabel("x")
ax[0].set_title("density")
ax[0].legend(loc=0)
ax[0].grid(alpha=0.3)

ax[1].scatter(x, vx, rasterized=True, s=6 * np.ones(x.shape), label="velocity")
ax[1].axhline(u0, ls="--", lw=2.0, color="black", label="analytic (uniform)")
ax[1].set_xlim(-0.5, 0.5)
ax[1].set_xlabel("x")
ax[1].set_title("velocity (should stay uniform at u0)")
ax[1].legend(loc=0)
ax[1].grid(alpha=0.3)

fig.suptitle(f"GSPH contact-discontinuity advection (exact solver + Inutsuka V2), t={t_target}")

if shamrock.sys.world_rank() == 0:
    plt.show()
