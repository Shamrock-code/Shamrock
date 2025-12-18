"""
GSPH Sod Shock Tube with VTK Animation Output
==============================================

Benchmark: Sod shock tube test using Godunov SPH (GSPH).

The GSPH method originated from:
- Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
  with Riemann Solver"

This implementation follows:
- Cha, S.-H. & Whitworth, A.P. (2003) "Implementations and tests of
  Godunov-type particle hydrodynamics"

The Sod shock tube is a standard test problem for hydrodynamics codes.
It involves a Riemann problem with a high-pressure region on the left
and a low-pressure region on the right, separated by a membrane that
is instantaneously removed at t=0.

The GSPH method uses Riemann solvers at particle interfaces instead of
artificial viscosity, which typically gives sharper shock resolution.

Initial conditions:
- Left state:  rho = 1.0,   P = 1.0
- Right state: rho = 0.125, P = 0.1
- Adiabatic index: gamma = 1.4

Expected features at t = 0.245:
1. Rarefaction wave (left-propagating)
2. Contact discontinuity (density jump, no pressure jump)
3. Shock wave (right-propagating)
"""

import os

import shamrock

# Simulation parameters
gamma = 1.4

# Initial conditions (Sod problem)
rho_L = 1.0  # Left density
rho_R = 0.125  # Right density
P_L = 1.0  # Left pressure
P_R = 0.1  # Right pressure

# Derived quantities
u_L = P_L / ((gamma - 1) * rho_L)  # Left internal energy
u_R = P_R / ((gamma - 1) * rho_R)  # Right internal energy

# Resolution
resol = 128

# VTK output settings
output_dir = "output"
vtk_prefix = "gsph_sod"
dump_freq = 10  # Dump VTK every N iterations

# Target time
t_target = 0.245

print("=" * 70)
print("GSPH Sod Shock Tube Simulation")
print("=" * 70)
print()
print("Configuration:")
print(f"  Left:  rho = {rho_L}, P = {P_L}, u = {u_L:.4f}")
print(f"  Right: rho = {rho_R}, P = {P_R}, u = {u_R:.4f}")
print(f"  gamma = {gamma}")
print(f"  Resolution: {resol}")
print(f"  Target time: {t_target}")
print()

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize context
ctx = shamrock.Context()
ctx.pdata_layout_new()

# Create GSPH model with M4 kernel (cubic spline)
# Wendland kernels (C2, C4, C6) are also recommended for GSPH
model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

# Generate default configuration
cfg = model.gen_default_config()

# Configure Riemann solver - use HLLC for speed, or iterative for accuracy
cfg.set_riemann_hllc()  # Fast approximate solver
# cfg.set_riemann_iterative(tolerance=1e-6, max_iter=20)  # More accurate

# Configure reconstruction - piecewise constant (1st order) for stability
cfg.set_reconstruct_piecewise_constant()
# cfg.set_reconstruct_muscl(limiter="van_leer")  # 2nd order, less stable

# Configure EOS
cfg.set_eos_adiabatic(gamma)

# Configure boundaries
cfg.set_boundary_periodic()

# Print configuration
cfg.print_status()
model.set_solver_config(cfg)

# Initialize scheduler
model.init_scheduler(int(1e8), 1)

# Setup domain
# Adjust particle spacing for equal mass particles
fact = (rho_L / rho_R) ** (1.0 / 3.0)

# Get box dimensions for FCC lattice
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)

# Resize simulation box
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Add particles
# Left region: high density
model.add_cube_fcc_3d(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
# Right region: low density (larger spacing for equal mass)
model.add_cube_fcc_3d(dr * fact, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Set internal energy in each region
model.set_value_in_a_box("uint", "f64", u_L, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_value_in_a_box("uint", "f64", u_R, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# Calculate and set particle mass
vol_b = xs * ys * zs
totmass = (rho_R * vol_b) + (rho_L * vol_b)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

print(f"Total particles: {model.get_total_part_count()}")
print(f"Particle mass: {pmass:.6e}")

# Set CFL conditions
model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

# Dump initial state
vtk_file = os.path.join(output_dir, f"{vtk_prefix}_0000.vtk")
model.do_vtk_dump(vtk_file, True)
print(f"Wrote initial state: {vtk_file}")

# Evolution loop
iteration = 0
dump_count = 1
t_current = model.get_time()

print()
print("Starting time evolution...")
print("-" * 70)

while t_current < t_target:
    # Evolve one timestep
    model.evolve_once()
    iteration += 1
    t_current = model.get_time()
    dt = model.get_dt()

    # Progress output
    if iteration % 10 == 0:
        print(f"  iter={iteration:4d}, t={t_current:.6f}, dt={dt:.6e}")

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
print(f"  Final time: {t_current:.6f}")
print(f"  Total iterations: {iteration}")
print(f"  VTK files: {dump_count + 1}")
print()
print(f"VTK files saved to: {output_dir}/")
print("Use ParaView or similar to visualize the results.")
print()
print("=" * 70)
print("GSPH Sod tube simulation complete")
print("=" * 70)
