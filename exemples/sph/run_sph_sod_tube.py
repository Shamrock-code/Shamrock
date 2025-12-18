"""
SPH Sod Shock Tube
==================

Benchmark: Sod shock tube problem using standard SPH with artificial viscosity.

The Sod shock tube is a classic Riemann problem test case that produces:
- A rarefaction wave propagating left
- A contact discontinuity
- A shock wave propagating right

Initial conditions:
- Left state (x < 0):  rho = 1.0,   P = 1.0,   u = 0
- Right state (x > 0): rho = 0.125, P = 0.1,   u = 0
- Adiabatic index: gamma = 1.4

This uses standard SPH with Cullen & Dehnen (2010) varying artificial viscosity
for shock capturing.

Usage:
    ./shamrock --sycl-cfg 0:0 --rscript exemples/sph/run_sph_sod_tube.py
"""

import os
import shamrock

print("=" * 70)
print("SPH Sod Shock Tube Benchmark")
print("=" * 70)
print()

# Simulation parameters
gamma = 1.4

# Initial conditions (Sod problem)
rho_L = 1.0      # Left density
rho_R = 0.125    # Right density
P_L = 1.0        # Left pressure
P_R = 0.1        # Right pressure

# Derived quantities
u_L = P_L / ((gamma - 1) * rho_L)  # Left internal energy
u_R = P_R / ((gamma - 1) * rho_R)  # Right internal energy

# Particle spacing factor for equal-mass particles
fact = (rho_L / rho_R) ** (1.0 / 3.0)

# Resolution
resol = 128

# Output settings
output_dir = "output"
vtk_prefix = "sph_sod"

# Target time
t_target = 0.245

print("Initial conditions:")
print(f"  Left:  rho = {rho_L}, P = {P_L}, u = {u_L:.4f}")
print(f"  Right: rho = {rho_R}, P = {P_R}, u = {u_R:.4f}")
print(f"  gamma = {gamma}")
print(f"  Resolution: {resol}")
print()

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize context
ctx = shamrock.Context()
ctx.pdata_layout_new()

# Create SPH model with M6 kernel
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

# Configure solver
cfg = model.gen_default_config()
# Use Cullen & Dehnen (2010) varying artificial viscosity for shocks
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1.0, sigma_decay=0.1, alpha_u=1.0, beta_AV=2.0
)
cfg.set_eos_adiabatic(gamma)
cfg.set_boundary_periodic()

print("Solver configuration:")
cfg.print_status()
model.set_solver_config(cfg)

# Initialize scheduler
model.init_scheduler(int(1e8), 1)

# Setup domain
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)

# Resize simulation box
model.resize_simulation_box((-xs, -ys/2, -zs/2), (xs, ys/2, zs/2))

# Add particles using setup generators
setup = model.get_setup()
gen_left = setup.make_generator_lattice_hcp(dr, (-xs, -ys/2, -zs/2), (0, ys/2, zs/2))
gen_right = setup.make_generator_lattice_hcp(dr * fact, (0, -ys/2, -zs/2), (xs, ys/2, zs/2))
combined = setup.make_combiner_add(gen_left, gen_right)
setup.apply_setup(combined)

# Set internal energy
model.set_value_in_a_box("uint", "f64", u_L, (-xs, -ys/2, -zs/2), (0, ys/2, zs/2))
model.set_value_in_a_box("uint", "f64", u_R, (0, -ys/2, -zs/2), (xs, ys/2, zs/2))

# Calculate and set particle mass
vol_box = xs * ys * zs
totmass = (rho_R * vol_box) + (rho_L * vol_box)
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

# Evolution with regular VTK dumps
print()
print("Starting time evolution...")
print("-" * 70)

iteration = 0
dump_count = 1
dump_freq = 20  # Dump every N iterations
t_current = model.get_time()

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
print(f"Simulation complete!")
print(f"  Final time: {t_current:.6f}")
print(f"  Total iterations: {iteration}")
print(f"  VTK files: {dump_count + 1}")

# Compute L2 error against analytical solution
print()
print("Computing L2 error against analytical solution...")
sod = shamrock.phys.SodTube(gamma=gamma, rho_1=rho_L, P_1=P_L, rho_5=rho_R, P_5=P_R)
sodanalysis = model.make_analysis_sodtube(sod, (1, 0, 0), t_target, 0.0, -xs, xs)
L2_error = sodanalysis.compute_L2_dist()
print(f"L2 error (rho, v, P): {L2_error}")

print()
print(f"VTK files saved to: {output_dir}/")
print("Use 'make sph-sod' for one-shot simulation + animation")
print()
print("=" * 70)
print("SPH Sod shock tube simulation complete")
print("=" * 70)
