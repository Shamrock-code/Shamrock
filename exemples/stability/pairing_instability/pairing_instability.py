"""
Pairing Instability Test - Square vs FCC Lattice Comparison
============================================================

Tests the pairing instability for SPH with different initial lattice configurations:
- SQUARE: True N×N square grid (quasi-2D)
- FCC: Face-centered cubic lattice (quasi-2D)

Both configurations use:
- M4 cubic spline kernel
- Standard SPH with artificial viscosity
- Random position perturbation ±5% of dx
- ρ=1, P=1, v=0, γ=5/3

Configuration via environment variables:
    PAIRING_LATTICE     : square or fcc (default: square)
    PAIRING_N           : Grid resolution N (default: 64)
    PAIRING_END_TIME    : Simulation end time (default: 1.0)
    PAIRING_N_OUTPUTS   : Number of VTK outputs (default: 50)
    PAIRING_SEED        : Random seed (default: 1)

Output directory: ../../../../../../simulations_data/pairing_instability/
"""

import json
import os
import sys

import numpy as np

import shamrock

# Configuration from environment variables
LATTICE = os.environ.get("PAIRING_LATTICE", "square").lower()
N = int(os.environ.get("PAIRING_N", "64"))
END_TIME = float(os.environ.get("PAIRING_END_TIME", "1.0"))
N_OUTPUTS = int(os.environ.get("PAIRING_N_OUTPUTS", "50"))
SEED = int(os.environ.get("PAIRING_SEED", "1"))

# Fixed parameters
SOLVER = "sph"
KERNEL = "M4"

# Validate lattice type
if LATTICE not in ["square", "fcc"]:
    print(f"Error: Invalid lattice '{LATTICE}'. Must be 'square' or 'fcc'.")
    sys.exit(1)

# Set random seed for reproducibility using modern NumPy Generator
rng = np.random.default_rng(SEED)

# Physical parameters
gamma = 5.0 / 3.0
rho_0 = 1.0
P_0 = 1.0
u_0 = P_0 / ((gamma - 1.0) * rho_0)  # Internal energy = 1.5

# Grid parameters
dx = 1.0 / N

# Perturbation amplitude (±5% of dx)
perturb = 0.05 * dx

# Domain: [-0.5, 0.5]² in xy, thin slab in z
z_thickness = dx
bmin = (-0.5, -0.5, -z_thickness / 2)
bmax = (0.5, 0.5, z_thickness / 2)

# Output directory (outside exemples, in simulations_data)
script_dir = os.path.dirname(os.path.abspath(__file__))
base_output_dir = os.path.normpath(
    os.path.join(script_dir, "../../../../../../simulations_data/pairing_instability")
)
output_dir = os.path.join(base_output_dir, f"{LATTICE}_{SOLVER}_{KERNEL.lower()}", "vtk")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print(f"PAIRING INSTABILITY TEST ({LATTICE.upper()} Lattice, Quasi-2D)")
print("=" * 70)
print(f"Solver:       {SOLVER.upper()}")
print(f"Kernel:       {KERNEL}")
print(f"Lattice:      {LATTICE.upper()}")
print(f"Resolution:   N = {N}")
print(f"Spacing:      dx = {dx:.6f}")
print(f"Perturbation: ±5% of dx = ±{perturb:.6f}")
print(f"Physics:      γ={gamma:.4f}, ρ={rho_0}, P={P_0}, u={u_0:.4f}")
print(f"End time:     {END_TIME}")
print(f"Outputs:      {N_OUTPUTS}")
print(f"Output dir:   {output_dir}")
print("=" * 70)

# Initialize context and model
ctx = shamrock.Context()
ctx.pdata_layout_new()

print(f"\nCreating SPH model with {KERNEL} kernel...")
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=KERNEL)

cfg = model.gen_default_config()
cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

# Initialize scheduler
model.init_scheduler(int(1e6), 1)

# Resize simulation box
model.resize_simulation_box(bmin, bmax)

# Generate particles based on lattice type
if LATTICE == "square":
    # Square lattice using push_particle for exact placement
    print(f"\nGenerating {N}×{N} particles on SQUARE lattice...")

    positions = []
    h_values = []
    u_values = []
    h_part = 1.2 * dx

    for j in range(N):
        for i in range(N):
            x_base = -0.5 + dx * (0.5 + i)
            y_base = -0.5 + dx * (0.5 + j)
            z_base = 0.0

            px = x_base + rng.uniform(-perturb, perturb)
            py = y_base + rng.uniform(-perturb, perturb)
            pz = z_base

            positions.append((px, py, pz))
            h_values.append(h_part)
            u_values.append(u_0)

    model.push_particle(positions, h_values, u_values)

else:
    # FCC lattice - generate positions manually for push_particle
    print("\nGenerating particles on FCC lattice...")

    positions = []
    h_values = []
    u_values = []
    h_part = 1.2 * dx

    # FCC unit cell has 4 atoms at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # For quasi-2D, we use the xy-plane variant: (0,0), (0.5,0.5) per cell
    fcc_offsets = [(0.0, 0.0), (0.5, 0.5)]

    # Number of cells
    n_cells = int(1.0 / dx)

    for j in range(n_cells):
        for i in range(n_cells):
            for ox, oy in fcc_offsets:
                x_base = -0.5 + dx * (i + ox)
                y_base = -0.5 + dx * (j + oy)
                z_base = 0.0

                # Check bounds
                if x_base >= 0.5 or y_base >= 0.5:
                    continue

                px = x_base + rng.uniform(-perturb, perturb)
                py = y_base + rng.uniform(-perturb, perturb)
                pz = z_base

                positions.append((px, py, pz))
                h_values.append(h_part)
                u_values.append(u_0)

    model.push_particle(positions, h_values, u_values)
    print(f"  Generated {len(positions)} FCC particles")

# Set particle mass
n_particles = model.get_total_part_count()
vol = (bmax[0] - bmin[0]) * (bmax[1] - bmin[1]) * (bmax[2] - bmin[2])
pmass = (rho_0 * vol) / n_particles
model.set_particle_mass(pmass)

print(f"Particles:     {n_particles}")
print(f"Particle mass: {pmass:.6e}")
print(f"Volume:        {vol:.6f}")

# Set CFL conditions
model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

# Time evolution parameters
t_final = END_TIME
n_outputs = N_OUTPUTS
dt_output = t_final / n_outputs

# Track simulation metadata
times = []
output_count = 0

# Initial output (t = 0)
filename = f"{output_dir}/pairing_{output_count:04d}.vtk"
model.do_vtk_dump(filename, True)
times.append({"index": output_count, "time": 0.0, "file": filename})
print(f"\nSaved: {filename} (t = 0.0)")
output_count += 1

# Time evolution with outputs
print(f"\nStarting time evolution to t = {t_final}...")
print("-" * 50)

t_current = 0.0
t_next_output = dt_output

while t_current < t_final:
    t_target = min(t_next_output, t_final)
    model.evolve_until(t_target)
    t_current = t_target

    filename = f"{output_dir}/pairing_{output_count:04d}.vtk"
    model.do_vtk_dump(filename, True)
    times.append({"index": output_count, "time": t_current, "file": filename})
    print(f"Saved: {filename} (t = {t_current:.4f})")
    output_count += 1

    t_next_output += dt_output

# Save metadata
metadata_dir = os.path.join(base_output_dir, f"{LATTICE}_{SOLVER}_{KERNEL.lower()}")
metadata_file = os.path.join(metadata_dir, "metadata.json")
metadata = {
    "test": "pairing_instability",
    "solver": SOLVER,
    "kernel": KERNEL,
    "lattice_type": LATTICE,
    "N": N,
    "n_particles": n_particles,
    "dx": dx,
    "perturbation": 0.05,
    "gamma": gamma,
    "rho_0": rho_0,
    "P_0": P_0,
    "u_0": u_0,
    "t_final": t_final,
    "n_outputs": n_outputs,
    "seed": SEED,
    "domain": {"min": list(bmin), "max": list(bmax)},
    "outputs": times,
}

with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

print("-" * 50)
print("\nSimulation complete!")
print(f"  {output_count} VTK files saved to {output_dir}/")
print(f"  Metadata saved to {metadata_file}")
print("=" * 70)
