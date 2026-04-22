"""
GSPH Sod Shock Tube Simulation with VTK Output
===============================================

Runs the Sod shock tube test using Godunov SPH (GSPH) with HLLC Riemann solver
and outputs VTK files for visualization.

Uses the same initial conditions as the SPH Sod test for direct comparison.

Output: VTK files in output/ directory with simulation time metadata
"""

import json
import os

import shamrock

# Physical parameters (same as SPH test)
gamma = 1.4

rho_L = 1.0  # Left density
rho_R = 0.125  # Right density

P_L = 1.0  # Left pressure
P_R = 0.1  # Right pressure

# Derived quantities
fact = (rho_L / rho_R) ** (1.0 / 3.0)
u_L = P_L / ((gamma - 1) * rho_L)  # Left internal energy
u_R = P_R / ((gamma - 1) * rho_R)  # Right internal energy

# Resolution (same as SPH test)
resol = 128

# Initialize context and model
ctx = shamrock.Context()
ctx.pdata_layout_new()

# Use GSPH model with M6 kernel (same as SPH test)
model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

# Configure solver
cfg = model.gen_default_config()

# Set HLLC Riemann solver
cfg.set_riemann_hllc()

# Set piecewise constant reconstruction (first-order, most stable)
cfg.set_reconstruct_piecewise_constant()

# Set periodic boundaries (with wall particles for shock tube)
cfg.set_boundary_periodic()

# Set adiabatic EOS
cfg.set_eos_adiabatic(gamma)

# Print configuration
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8), 1)

# Setup domain (same as SPH test)
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)

model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Setup initial conditions using HCP lattice (same as SPH test)
# Left side: high density (smaller spacing)
model.add_cube_hcp_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
# Right side: low density (larger spacing)
model.add_cube_hcp_3d(dr * fact, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Set internal energy for left and right states (discontinuity at x=0)
model.set_field_in_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_field_in_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Set particle mass (same as SPH test)
vol_b = xs * ys * zs
totmass = (rho_R * vol_b) + (rho_L * vol_b)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

print(f"Total mass: {totmass}")
print(f"Particle mass: {pmass}")

# Set CFL conditions (same as SPH test)
model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

# Simulation parameters (same as SPH test)
t_final = 0.245
n_outputs = 50
dt_output = t_final / n_outputs

# Track output times
times = []
output_count = 0

# Create output directory
output_dir = "simulations_data/gsph_sod/vtk"
os.makedirs(output_dir, exist_ok=True)

# Initial output
filename = f"{output_dir}/gsph_sod_{output_count:04d}.vtk"
model.do_vtk_dump(filename, True)
times.append({"index": output_count, "time": 0.0, "file": filename})
print(f"Saved: {filename} (t = 0.0)")
output_count += 1

# Time evolution with outputs
t_current = 0.0
t_next_output = dt_output

while t_current < t_final:
    # Evolve to next output time or final time
    t_target = min(t_next_output, t_final)
    model.evolve_until(t_target)
    t_current = t_target

    # Output VTK
    filename = f"{output_dir}/gsph_sod_{output_count:04d}.vtk"
    model.do_vtk_dump(filename, True)
    times.append({"index": output_count, "time": t_current, "file": filename})
    print(f"Saved: {filename} (t = {t_current:.6f})")
    output_count += 1

    t_next_output += dt_output

# Save times metadata
with open("simulations_data/gsph_sod/times_gsph_sod.json", "w") as f:
    json.dump(
        {
            "method": "GSPH",
            "riemann_solver": "HLLC",
            "kernel": "M6",
            "gamma": gamma,
            "rho_L": rho_L,
            "rho_R": rho_R,
            "P_L": P_L,
            "P_R": P_R,
            "t_final": t_final,
            "outputs": times,
        },
        f,
        indent=2,
    )

print(f"\nSimulation complete! {output_count} VTK files saved to {output_dir}/")
print("\nNote: L2 error analysis not available for GSPH model.")
print("Use post-processing scripts for comparison with analytical solution.")
