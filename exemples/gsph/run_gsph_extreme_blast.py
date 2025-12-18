"""
GSPH Extreme Blast Wave (Strong Shock Tube)
============================================

Benchmark: Extremely strong shock tube problem from Inutsuka (2002).
Tests GSPH capability to handle extreme pressure ratios (Mach ~10^5).

This problem demonstrates GSPH's ability to handle:
- Extreme pressure ratio: P_L/P_R = 3×10^10
- Mach number ~10^5
- No penetration problems at contact discontinuity

Initial conditions (from the paper):
- Left state (x < 0):  rho = 1.0, P = 3000,    v = 0
- Right state (x > 0): rho = 1.0, P = 10^-7,  v = 0
- Adiabatic index: gamma = 5/3

Reference:
    Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
    with Riemann Solver", J. Comput. Phys., 179, 238-267

Usage:
    ./shamrock --sycl-cfg 0:0 --rscript exemples/gsph/run_gsph_extreme_blast.py
"""

import os

import shamrock

print("=" * 70)
print("GSPH Extreme Blast Wave (Strong Shock Tube)")
print("=" * 70)
print()

# Simulation parameters
gamma = 5.0 / 3.0  # Monatomic gas

# Initial conditions (Extreme blast wave from Inutsuka 2002)
rho_L = 1.0  # Left density
rho_R = 1.0  # Right density (same as left)
P_L = 3000.0  # Left pressure (high)
P_R = 1e-7  # Right pressure (very low)

# Derived quantities
u_L = P_L / ((gamma - 1) * rho_L)  # Left internal energy
u_R = P_R / ((gamma - 1) * rho_R)  # Right internal energy

# Resolution (100 particles on each side as in the paper)
resol = 100

# Output settings
output_dir = "output"
vtk_prefix = "gsph_extreme_blast"

# Target time (shock should develop and propagate)
# Estimate: sound speed on left ~ sqrt(gamma * P_L / rho_L) ~ 70
# Domain is ~1, so characteristic time ~ 0.01
t_target = 0.005

print("Initial conditions (Inutsuka 2002 Extreme Blast Wave):")
print(f"  Left:  rho = {rho_L}, P = {P_L}, u = {u_L:.4e}")
print(f"  Right: rho = {rho_R}, P = {P_R:.1e}, u = {u_R:.4e}")
print(f"  Pressure ratio: P_L/P_R = {P_L/P_R:.1e}")
print(f"  gamma = {gamma:.4f}")
print(f"  Resolution: {resol} particles per side")
print()

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize context
ctx = shamrock.Context()
ctx.pdata_layout_new()

# Use M4 kernel for GSPH
model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

# Configure GSPH solver
cfg = model.gen_default_config()
cfg.set_riemann_hllc()  # HLLC is more robust for extreme pressure ratios
cfg.set_reconstruct_piecewise_constant()  # 1st order (stable for strong shocks)
cfg.set_boundary_periodic()  # Periodic boundaries (like working Sod test)
cfg.set_eos_adiabatic(gamma)

print("Solver configuration:")
cfg.print_status()
model.set_solver_config(cfg)

# Initialize scheduler
model.init_scheduler(int(1e8), 1)

# Setup domain - equal mass particles, so equal spacing since rho_L = rho_R
# Use larger cross-section (24x24) for better stability like Sod test
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)

# Resize simulation box (tube from -xs to xs)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Add particles - equal spacing since densities are equal
model.add_cube_fcc_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.add_cube_fcc_3d(dr, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

N_total = model.get_total_part_count()
print(f"Total particles: {N_total}")

# Set internal energy in each region
model.set_value_in_a_box(
    "uint", "f64", u_L, (-xs - dr, -ys / 2 - dr, -zs / 2 - dr), (0, ys / 2 + dr, zs / 2 + dr)
)
model.set_value_in_a_box(
    "uint", "f64", u_R, (0, -ys / 2 - dr, -zs / 2 - dr), (xs + dr, ys / 2 + dr, zs / 2 + dr)
)

# Calculate and set particle mass
vol_box = xs * ys * zs
totmass = (rho_L * vol_box) + (rho_R * vol_box)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print(f"Particle mass: {pmass:.6e}")

# Set CFL conditions (same as working Sod test)
model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

# Dump initial state
vtk_file = os.path.join(output_dir, f"{vtk_prefix}_0000.vtk")
model.do_vtk_dump(vtk_file, True)
print(f"Wrote initial state: {vtk_file}")

# Evolution loop
print()
print("Starting time evolution...")
print("-" * 70)

iteration = 0
dump_count = 1
dump_freq = 50  # Dump every N iterations
t_current = model.get_time()

while t_current < t_target:
    # Evolve one timestep
    model.evolve_once()
    iteration += 1
    t_current = model.get_time()
    dt = model.get_dt()

    # Progress output
    if iteration % 20 == 0:
        print(f"  iter={iteration:4d}, t={t_current:.6e}, dt={dt:.6e}")

    # VTK dump
    if iteration % dump_freq == 0:
        vtk_file = os.path.join(output_dir, f"{vtk_prefix}_{dump_count:04d}.vtk")
        model.do_vtk_dump(vtk_file, True)
        print(f"  -> Wrote: {vtk_file}")
        dump_count += 1

# Final dump
vtk_file = os.path.join(output_dir, f"{vtk_prefix}_{dump_count:04d}.vtk")
model.do_vtk_dump(vtk_file, True)
print(f"  -> Wrote final state: {vtk_file}")

print("-" * 70)
print("Simulation complete!")
print(f"  Final time: {t_current:.6e}")
print(f"  Total iterations: {iteration}")
print(f"  VTK files: {dump_count + 1}")

print()
print(f"VTK files saved to: {output_dir}/")
print("Use 'make gsph-extreme' for one-shot simulation + animation")
print()
print("=" * 70)
print("GSPH Extreme Blast Wave simulation complete")
print("=" * 70)
