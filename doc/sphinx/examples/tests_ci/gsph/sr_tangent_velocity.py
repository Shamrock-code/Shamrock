"""
Special Relativistic Tangent Velocity Test with SR-GSPH
========================================================

CI test for Kitajima et al. (2025) arXiv:2510.18251v1 Section 3.1.5
Riemann Problem 5: 1D Shock Tube With Tangential Velocity

Key Physics:
- In SR, tangent velocity v^t affects the solution via the Lorentz factor
- The K-invariant K = gamma*H*v_t is conserved across shocks
- gamma = 1/sqrt(1 - v_x^2 - v_t^2) includes both velocity components
- Tangent velocity CHANGES across shocks (v_t_L* != v_t_R*)

Initial Conditions (Kitajima Problem 5):
    Left:  (P, n, v^x, v^t) = (1000, 1.0, 0, 0.9)
    Right: (P, n, v^x, v^t) = (0.01, 1.0, 0, 0.9)
"""

import sys
from pathlib import Path

import numpy as np

import shamrock

SRRP_PATH = Path("/Users/guo/Downloads/sph-simulators/docs/papers/sg-gsph/srrp")
sys.path.insert(0, str(SRRP_PATH))

from srrp.Solver import Solver
from srrp.State import State

gamma_eos = 5.0 / 3.0
c_speed = 1.0

P_L, P_R = 1000.0, 0.01
n_L, n_R = 1.0, 1.0
vx_L, vx_R = 0.0, 0.0
vt_L, vt_R = 0.9, 0.9

N_per_side = 400
t_target = 0.12


def compute_lorentz_factor(vx, vt, c=1.0):
    v2 = vx**2 + vt**2
    if np.any(v2 >= c**2):
        raise ValueError(f"Superluminal velocity: v^2 = {v2} >= c^2 = {c**2}")
    return 1.0 / np.sqrt(1.0 - v2 / c**2)


def compute_internal_energy(P, n, gamma_eos):
    return P / ((gamma_eos - 1.0) * n)


u_L = compute_internal_energy(P_L, n_L, gamma_eos)
u_R = compute_internal_energy(P_R, n_R, gamma_eos)

solver = Solver()
stateL = State(rho=n_L, vx=vx_L, vt=vt_L, pressure=P_L)
stateR = State(rho=n_R, vx=vx_R, vt=vt_R, pressure=P_R)
wavefan = solver.solve(stateL, stateR, gamma_eos)

states = wavefan.states
print("SR-GSPH Tangent Velocity Test (Kitajima Problem 5)")
print(f"Solution type: {solver.solution_type}")
print(f"P*      = {states[1].pressure:.4f}")
print(f"vx*     = {states[1].vx:.4f}")
print(f"vt_L*   = {states[1].vt:.4f}")
print(f"vt_R*   = {states[2].vt:.4f}")
print(f"n_L'    = {states[1].rho:.4f}")
print(f"n_R'    = {states[2].rho:.4f}")

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="TGauss3")
cfg = model.gen_default_config()

cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma_eos)
cfg.set_sr(c_speed=c_speed)

model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, N_per_side, 8, 8)
dr = 1.0 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, N_per_side, 8, 8)

model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_hcp_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

model.set_field_in_box(
    "vxyz", "f64_3", (vx_L, vt_L, 0.0), (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2)
)
model.set_field_in_box(
    "vxyz", "f64_3", (vx_R, vt_R, 0.0), (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2)
)

h_init = dr
model.set_field_in_box(
    "hpart", "f64", h_init, (-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2)
)

vol_total = 2 * xs * ys * zs
gamma_L = compute_lorentz_factor(vx_L, vt_L, c_speed)
totmass = gamma_L * n_L * vol_total

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
hfact = model.get_hfact()

model.set_cfl_cour(0.15)
model.set_cfl_force(1e10)

print(f"Running simulation to t={t_target}...")
model.evolve_until(t_target)

data = ctx.collect_data()

points = np.array(data["xyz"])
velocities = np.array(data["vxyz"])
hpart = np.array(data["hpart"])
uint_data = np.array(data["uint"])

x = points[:, 0]
vx = velocities[:, 0]
vy = velocities[:, 1]  # This is v_t (tangent velocity stored in y-component)
vz = velocities[:, 2]

v_mag_sq = vx**2 + vy**2 + vz**2
gamma_lorentz = 1.0 / np.sqrt(1.0 - np.clip(v_mag_sq, 0, 0.9999))

N_lab = pmass * (hfact / hpart) ** 3
n_sim = N_lab / gamma_lorentz

P_sim = (gamma_eos - 1.0) * n_sim * uint_data

print(f"Particles: {len(x)}")

x_exact = np.linspace(-0.5, 0.5, 1000)
xi_exact = x_exact / t_target
state_exact = wavefan.getState(xi_exact)
n_exact = state_exact.rho
P_exact = state_exact.pressure
vx_exact = state_exact.vx
vt_exact = state_exact.vt
h_exact = (pmass / n_exact) ** (1.0 / 3.0) * hfact

mask = (x >= -0.5) & (x <= 0.5)
x_f = x[mask]
n_f = n_sim[mask]
P_f = P_sim[mask]
vx_f = vx[mask]
vt_f = vy[mask]
h_f = hpart[mask]

xi_f = x_f / t_target
state_ana = wavefan.getState(xi_f)
n_ana = state_ana.rho
P_ana = state_ana.pressure
vx_ana = state_ana.vx
vt_ana = state_ana.vt

err_n = np.sqrt(np.mean((n_f - n_ana) ** 2)) / np.mean(n_ana)
err_vx = np.sqrt(np.mean((vx_f - vx_ana) ** 2)) / max(np.mean(np.abs(vx_ana)), 0.1)
err_vt = np.sqrt(np.mean((vt_f - vt_ana) ** 2)) / np.mean(np.abs(vt_ana))
err_P = np.sqrt(np.mean((P_f - P_ana) ** 2)) / np.mean(P_ana)
err_vz = np.sqrt(np.mean(vz[mask] ** 2))

print(f"L2 errors: n={err_n:.6e}, vx={err_vx:.6e}, vt={err_vt:.6e}, P={err_P:.6e}")
print(f"v_z RMS: {err_vz:.6e} (should be ~0)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
fig.subplots_adjust(hspace=0.05)

ax1 = axes[0]
ax1.plot(x_exact, P_exact, "k-", linewidth=1.5, label="Exact (SRRP)")
ax1.scatter(x_f, P_f, s=2, c="#007F66", marker="v", alpha=0.6)
ax1.set_ylabel(r"$P$", fontsize=14)
ax1.tick_params(labelbottom=False)
ax1.axhline(states[1].pressure, color='green', linestyle='--', alpha=0.5,
            label=f"P*={states[1].pressure:.2f}")
ax1.legend(loc='upper right', fontsize=8)

ax2 = axes[1]
ax2.plot(x_exact, n_exact, "k-", linewidth=1.5, label="Exact")
ax2.scatter(x_f, n_f, s=2, c="#CC3311", marker="v", alpha=0.6)
ax2.set_ylabel(r"$n$", fontsize=14)
ax2.tick_params(labelbottom=False)

ax3 = axes[2]
ax3.plot(x_exact, vx_exact, "k-", linewidth=1.5, label="Exact")
ax3.scatter(x_f, vx_f, s=2, c="#003366", marker="x", alpha=0.6)
ax3.set_ylabel(r"$v^x$", fontsize=14)
ax3.tick_params(labelbottom=False)

ax4 = axes[3]
ax4.plot(x_exact, vt_exact, "k-", linewidth=1.5, label="Exact")
ax4.scatter(x_f, vt_f, s=2, c="#EE7733", marker="^", alpha=0.6)
ax4.set_ylabel(r"$v^t$", fontsize=14)
ax4.set_xlabel(r"$x$", fontsize=14)
ax4.axhline(vt_L, color='orange', linestyle=':', alpha=0.5, label=f"vt_init={vt_L}")
ax4.axhline(states[1].vt, color='green', linestyle='--', alpha=0.5,
            label=f"vt_L*={states[1].vt:.3f}")
ax4.axhline(states[2].vt, color='purple', linestyle='--', alpha=0.5,
            label=f"vt_R*={states[2].vt:.3f}")
ax4.legend(loc='lower right', fontsize=8)

for ax in axes:
    ax.set_xlim(-0.5, 0.5)
    ax.tick_params(direction="in", which="both")

fig.suptitle(f"SR Tangent Velocity Test t={t_target} (vt={vt_L})", fontsize=12, y=0.98)
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
plot_filename = f"sr_tangent_velocity_kitajima_{timestamp}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
print(f"Saved Kitajima-style plot to {plot_filename}")
plt.close(fig)

print("SR-GSPH Tangent Velocity: Done (visual verification)")
