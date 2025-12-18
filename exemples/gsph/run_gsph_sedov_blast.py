"""
GSPH Sedov-Taylor Blast Wave
============================

Benchmark: Sedov-Taylor point explosion using Godunov SPH (GSPH).

The GSPH method originated from:
- Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
  with Riemann Solver"

The Sedov-Taylor blast wave is a classic test for strong shock propagation.
Energy is deposited at a point (or small region) in a uniform medium,
producing a spherically expanding shock wave with a self-similar solution.

Analytical solution (3D):
- Shock radius: R_s(t) = (E * t^2 / rho_0)^(1/5) * xi_0
- Post-shock density: rho_s = rho_0 * (gamma + 1) / (gamma - 1)
- xi_0 ~ 1.15 for gamma = 5/3

Initial conditions:
- Uniform background: rho_0 = 1.0, P_0 ~ 0 (cold)
- Point energy: E = 1.0 deposited in central region
- Adiabatic index: gamma = 5/3

Usage:
    ./shamrock --sycl-cfg 0:0 --rscript exemples/gsph/run_gsph_sedov_blast.py
"""

import os

import numpy as np

import shamrock

print("=" * 70)
print("GSPH Sedov-Taylor Blast Wave Benchmark")
print("=" * 70)
print()

# Simulation parameters
gamma = 5.0 / 3.0  # Adiabatic index for monatomic gas

# Initial conditions
rho_0 = 1.0  # Background density
P_0 = 1e-6  # Background pressure (effectively zero)
E_blast = 1.0  # Total blast energy

# Derived quantities
u_0 = P_0 / ((gamma - 1) * rho_0)  # Background internal energy

# Resolution
resol = 32  # Particles per dimension (keep moderate for quick test)

# Output settings
output_dir = "output_sedov"
vtk_prefix = "sedov"

# Target time
t_target = 0.1

# Sedov solution parameters
xi_0 = 1.15167  # Dimensionless shock position for gamma = 5/3

print("Initial conditions:")
print(f"  Background: rho_0 = {rho_0}, P_0 = {P_0:.1e}")
print(f"  Blast energy: E = {E_blast}")
print(f"  gamma = {gamma:.4f}")
print(f"  Resolution: {resol}^3 particles")
print(f"  Target time: {t_target}")
print()


def sedov_shock_radius(t, E, rho_0, gamma=5 / 3):
    """Analytical shock radius for Sedov-Taylor blast wave."""
    xi_0 = 1.15167 if abs(gamma - 5 / 3) < 0.01 else 1.0
    return xi_0 * (E * t**2 / rho_0) ** (1 / 5)


def sedov_post_shock_density(rho_0, gamma=5 / 3):
    """Post-shock density from Rankine-Hugoniot conditions."""
    return rho_0 * (gamma + 1) / (gamma - 1)


# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize context
ctx = shamrock.Context()
ctx.pdata_layout_new()

# Use M4 kernel for GSPH
model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

# Configure GSPH solver
cfg = model.gen_default_config()
cfg.set_riemann_hllc()  # HLLC for speed (or iterative for accuracy)
cfg.set_reconstruct_piecewise_constant()  # 1st order (stable for strong shocks)
cfg.set_boundary_periodic()  # Periodic boundaries
cfg.set_eos_adiabatic(gamma)

print("Solver configuration:")
cfg.print_status()
model.set_solver_config(cfg)

# Initialize scheduler
model.init_scheduler(int(1e8), 1)

# Setup cubic domain
L = 0.5  # Half box size
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, resol, resol)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, resol, resol)

# Resize simulation box (centered at origin)
model.resize_simulation_box((-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))

# Add uniform particle distribution
model.add_cube_fcc_3d(dr, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))

N_total = model.get_total_part_count()
print(f"Total particles: {N_total}")

# Calculate particle mass for target density
V_box = xs * ys * zs
pmass = rho_0 * V_box / N_total
model.set_particle_mass(pmass)
print(f"Particle mass: {pmass:.6e}")

# Set background internal energy (cold gas)
model.set_value_in_a_box(
    "uint",
    "f64",
    u_0,
    (-xs / 2 - dr, -ys / 2 - dr, -zs / 2 - dr),
    (xs / 2 + dr, ys / 2 + dr, zs / 2 + dr),
)

# Deposit blast energy in central region
# The blast region should be small (few particle spacings)
r_blast = 3 * dr  # Blast radius
print(f"Blast radius: {r_blast:.4f}")

# Estimate number of particles in blast region
V_blast = (4 / 3) * np.pi * r_blast**3
N_blast_estimate = int(V_blast * N_total / V_box)
if N_blast_estimate < 1:
    N_blast_estimate = 1

# u_blast = E / (N_blast * pmass)
u_blast = E_blast / (N_blast_estimate * pmass)
print(f"Estimated blast particles: {N_blast_estimate}")
print(f"Blast internal energy: {u_blast:.4e}")

# Set blast energy using box approximation (sphere would be better but not available)
# Use a small cube centered at origin
model.set_value_in_a_box(
    "uint", "f64", u_blast, (-r_blast, -r_blast, -r_blast), (r_blast, r_blast, r_blast)
)

# Set CFL conditions
model.set_cfl_cour(0.3)
model.set_cfl_force(0.3)

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
print("Simulation complete!")
print(f"  Final time: {t_current:.6f}")
print(f"  Total iterations: {iteration}")

# Compare with analytical solution
R_s_analytical = sedov_shock_radius(t_current, E_blast, rho_0, gamma)
rho_s_analytical = sedov_post_shock_density(rho_0, gamma)

print()
print("Analytical Sedov-Taylor solution:")
print(f"  At t = {t_current:.4f}:")
print(f"    Shock radius: R_s = {R_s_analytical:.4f}")
print(f"    Post-shock density: rho_s = {rho_s_analytical:.2f}")
print(f"    Density jump ratio: {rho_s_analytical/rho_0:.1f}")
print()
print(f"VTK files saved to: {output_dir}/")
print("Use ParaView to visualize the spherical shock wave.")
print()
print("=" * 70)
print("GSPH Sedov blast simulation complete")
print("=" * 70)
