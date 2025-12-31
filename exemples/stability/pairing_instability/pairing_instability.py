"""
Pairing Instability Test - Square vs FCC Lattice Comparison.

Environment variables: PAIRING_LATTICE (square/fcc), PAIRING_N (64),
PAIRING_END_TIME (1.0), PAIRING_N_OUTPUTS (50), PAIRING_SEED (1)
"""

import json
import os
import sys

import numpy as np

import shamrock

# Configuration
LATTICE = os.environ.get("PAIRING_LATTICE", "square").lower()
N = int(os.environ.get("PAIRING_N", "64"))
END_TIME = float(os.environ.get("PAIRING_END_TIME", "1.0"))
N_OUTPUTS = int(os.environ.get("PAIRING_N_OUTPUTS", "50"))
SEED = int(os.environ.get("PAIRING_SEED", "1"))

if LATTICE not in ["square", "fcc"]:
    print(f"Error: Invalid lattice '{LATTICE}'. Must be 'square' or 'fcc'.")
    sys.exit(1)

rng = np.random.default_rng(SEED)
gamma, rho_0, P_0 = 5.0 / 3.0, 1.0, 1.0
u_0 = P_0 / ((gamma - 1.0) * rho_0)
dx, perturb, h_part = 1.0 / N, 0.05 / N, 1.2 / N
bmin, bmax = (-0.5, -0.5, -dx / 2), (0.5, 0.5, dx / 2)

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.normpath(
    os.path.join(script_dir, "../../../../../../simulations_data/pairing_instability")
)
output_dir = os.path.join(base_dir, f"{LATTICE}_sph_m4", "vtk")
os.makedirs(output_dir, exist_ok=True)

print(f"=== PAIRING INSTABILITY ({LATTICE.upper()}) N={N}, t_final={END_TIME} ===")

# Initialize model
ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
cfg = model.gen_default_config()
cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
model.set_solver_config(cfg)
model.init_scheduler(int(1e6), 1)
model.resize_simulation_box(bmin, bmax)

# Generate particles (vectorized)
if LATTICE == "square":
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    x = -0.5 + dx * (0.5 + i.ravel()) + rng.uniform(-perturb, perturb, N * N)
    y = -0.5 + dx * (0.5 + j.ravel()) + rng.uniform(-perturb, perturb, N * N)
else:  # FCC
    n_cells = int(1.0 / dx)
    i, j = np.meshgrid(np.arange(n_cells), np.arange(n_cells))
    pts = []
    for ox, oy in [(0.0, 0.0), (0.5, 0.5)]:
        xb, yb = -0.5 + dx * (i.ravel() + ox), -0.5 + dx * (j.ravel() + oy)
        mask = (xb < 0.5) & (yb < 0.5)
        pts.append(np.column_stack([xb[mask], yb[mask]]))
    xy = np.vstack(pts)
    x = xy[:, 0] + rng.uniform(-perturb, perturb, len(xy))
    y = xy[:, 1] + rng.uniform(-perturb, perturb, len(xy))

positions = [(x[i], y[i], 0.0) for i in range(len(x))]
model.push_particle(positions, [h_part] * len(x), [u_0] * len(x))

n_particles = model.get_total_part_count()
vol = (bmax[0] - bmin[0]) * (bmax[1] - bmin[1]) * (bmax[2] - bmin[2])
model.set_particle_mass((rho_0 * vol) / n_particles)
model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

print(f"Particles: {n_particles}, dx={dx:.6f}")

# Time evolution
dt_output, times = END_TIME / N_OUTPUTS, []
for i in range(N_OUTPUTS + 1):
    t = min(i * dt_output, END_TIME)
    if i > 0:
        model.evolve_until(t)
    filename = f"{output_dir}/pairing_{i:04d}.vtk"
    model.do_vtk_dump(filename, True)
    times.append({"index": i, "time": t, "file": filename})
    if i % 10 == 0:
        print(f"  t={t:.3f}")

# Save metadata
metadata = {
    "test": "pairing_instability",
    "solver": "sph",
    "kernel": "M4",
    "lattice_type": LATTICE,
    "N": N,
    "n_particles": n_particles,
    "dx": dx,
    "perturbation": 0.05,
    "gamma": gamma,
    "rho_0": rho_0,
    "P_0": P_0,
    "u_0": u_0,
    "t_final": END_TIME,
    "n_outputs": N_OUTPUTS,
    "seed": SEED,
    "domain": {"min": list(bmin), "max": list(bmax)},
    "outputs": times,
}
with open(os.path.join(base_dir, f"{LATTICE}_sph_m4", "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Done! {len(times)} files in {output_dir}")
