"""
SPH Sod Shock Tube with Animated Output
=======================================

This script runs a standard SPH Sod shock tube simulation with
artificial viscosity and generates both VTK files and a matplotlib animation.

Note: GSPH model currently has a bug in compute_omega. This uses standard SPH
with artificial viscosity as a working alternative.

Initial conditions:
- Left state:  rho = 1.0,   P = 1.0
- Right state: rho = 0.125, P = 0.1
- Adiabatic index: gamma = 1.4

Usage:
    ./shamrock --sycl-cfg 0:0 --rscript exemples/sph_sod_animation.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import shamrock

# =============================================================================
# Simulation Parameters
# =============================================================================

gamma = 1.4  # Adiabatic index

# Initial conditions (Sod problem)
rho_L = 1.0      # Left density
rho_R = 0.125    # Right density
P_L = 1.0        # Left pressure
P_R = 0.1        # Right pressure

# Derived quantities
u_L = P_L / ((gamma - 1) * rho_L)  # Left internal energy
u_R = P_R / ((gamma - 1) * rho_R)  # Right internal energy

# Resolution (number of particles along x)
resol = 128

# Timing
t_target = 0.245      # Final simulation time
n_frames = 30         # Number of animation frames
dt_output = t_target / n_frames  # Time between frames

# Output settings
output_dir = "sph_sod_output"
vtk_prefix = "sph_sod"
animation_file = "sph_sod_animation.gif"

# =============================================================================
# Print Configuration
# =============================================================================

print("=" * 70)
print("SPH Sod Shock Tube Simulation with Animation")
print("=" * 70)
print()
print("Configuration:")
print(f"  Left:  rho = {rho_L}, P = {P_L}, u = {u_L:.4f}")
print(f"  Right: rho = {rho_R}, P = {P_R}, u = {u_R:.4f}")
print(f"  gamma = {gamma}")
print(f"  Resolution: {resol}")
print(f"  Target time: {t_target}")
print(f"  Animation frames: {n_frames}")
print()

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Initialize SPH Model
# =============================================================================

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

# =============================================================================
# Setup Domain and Particles
# =============================================================================

# Particle spacing factor for equal-mass particles
fact = (rho_L / rho_R) ** (1.0 / 3.0)

# Get box dimensions for FCC lattice
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)

# Resize simulation box
model.resize_simulation_box((-xs, -ys/2, -zs/2), (xs, ys/2, zs/2))

# Add particles
# Left region: high density
model.add_cube_fcc_3d(dr, (-xs, -ys/2, -zs/2), (0, ys/2, zs/2))
# Right region: low density (larger spacing for equal mass)
model.add_cube_fcc_3d(dr * fact, (0, -ys/2, -zs/2), (xs, ys/2, zs/2))

# Set internal energy in each region
model.set_value_in_a_box("uint", "f64", u_L, (-xs, -ys/2, -zs/2), (0, ys/2, zs/2))
model.set_value_in_a_box("uint", "f64", u_R, (0, -ys/2, -zs/2), (xs, ys/2, zs/2))

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

# =============================================================================
# Data Collection Function
# =============================================================================

def collect_frame_data(ctx, model, pmass, gamma, hfact):
    """Collect particle data for plotting."""
    dic = ctx.collect_data()

    x = np.array(dic["xyz"][:, 0]) + xs  # Shift to [0, 2*xs]
    vx = dic["vxyz"][:, 0]
    uint = dic["uint"][:]
    hpart = dic["hpart"]

    # Compute density and pressure
    rho = pmass * (hfact / hpart) ** 3
    P = (gamma - 1) * rho * uint

    return x, rho, vx, P, uint

# =============================================================================
# Analytical Solution
# =============================================================================

# Get analytical solution object
sod = shamrock.phys.SodTube(gamma=gamma, rho_1=rho_L, P_1=P_L, rho_5=rho_R, P_5=P_R)

def get_analytical_solution(t, x_range=(-xs, xs), npoints=500):
    """Get analytical solution at time t."""
    x_ana = np.linspace(x_range[0], x_range[1], npoints)
    rho_ana = []
    vx_ana = []
    P_ana = []

    for xi in x_ana:
        _rho, _vx, _P = sod.get_value(t, xi)
        rho_ana.append(_rho)
        vx_ana.append(_vx)
        P_ana.append(_P)

    # Shift x to match particle coordinates
    x_ana_shifted = x_ana + xs

    return x_ana_shifted, np.array(rho_ana), np.array(vx_ana), np.array(P_ana)

# =============================================================================
# Run Simulation and Collect Data
# =============================================================================

print()
print("Running simulation and collecting frame data...")
print("-" * 70)

hfact = model.get_hfact()

# Store data for animation
frames_data = []
frame_times = []

# Initial frame
x, rho, vx, P, uint = collect_frame_data(ctx, model, pmass, gamma, hfact)
frames_data.append({'x': x, 'rho': rho, 'vx': vx, 'P': P})
frame_times.append(0.0)

# Initial VTK dump
vtk_file = os.path.join(output_dir, f"{vtk_prefix}_0000.vtk")
model.do_vtk_dump(vtk_file, True)
print(f"Frame 0: t=0.000000, wrote {vtk_file}")

# Evolution loop
iteration = 0
frame_count = 1
t_current = model.get_time()
t_next_frame = dt_output

while t_current < t_target:
    # Evolve one timestep
    model.evolve_once()
    iteration += 1
    t_current = model.get_time()

    # Check if we should save a frame
    if t_current >= t_next_frame or t_current >= t_target:
        # Collect data
        x, rho, vx, P, uint = collect_frame_data(ctx, model, pmass, gamma, hfact)
        frames_data.append({'x': x, 'rho': rho, 'vx': vx, 'P': P})
        frame_times.append(t_current)

        # VTK dump
        vtk_file = os.path.join(output_dir, f"{vtk_prefix}_{frame_count:04d}.vtk")
        model.do_vtk_dump(vtk_file, True)
        print(f"Frame {frame_count}: t={t_current:.6f}, wrote {vtk_file}")

        frame_count += 1
        t_next_frame += dt_output

print("-" * 70)
print(f"Simulation complete! Total iterations: {iteration}")
print(f"Collected {len(frames_data)} frames")

# =============================================================================
# Create Animation
# =============================================================================

print()
print("Creating matplotlib animation...")

# Setup figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('SPH Sod Shock Tube', fontsize=14)

# Plot limits
x_lim = (0, 2 * xs)
rho_lim = (0, 1.2)
vx_lim = (-0.1, 1.1)
P_lim = (0, 1.2)

# Initialize empty plots
ax_rho = axes[0, 0]
ax_vx = axes[0, 1]
ax_P = axes[1, 0]
ax_uint = axes[1, 1]

# Particle scatter plots
scat_rho, = ax_rho.plot([], [], 'b.', markersize=1, alpha=0.5, label='SPH')
scat_vx, = ax_vx.plot([], [], 'b.', markersize=1, alpha=0.5, label='SPH')
scat_P, = ax_P.plot([], [], 'b.', markersize=1, alpha=0.5, label='SPH')

# Analytical solution lines
line_rho, = ax_rho.plot([], [], 'r-', linewidth=1.5, label='Analytical')
line_vx, = ax_vx.plot([], [], 'r-', linewidth=1.5, label='Analytical')
line_P, = ax_P.plot([], [], 'r-', linewidth=1.5, label='Analytical')

# Configure axes
ax_rho.set_xlim(x_lim)
ax_rho.set_ylim(rho_lim)
ax_rho.set_xlabel('x')
ax_rho.set_ylabel('Density')
ax_rho.set_title('Density')
ax_rho.legend(loc='upper right')
ax_rho.grid(True, alpha=0.3)

ax_vx.set_xlim(x_lim)
ax_vx.set_ylim(vx_lim)
ax_vx.set_xlabel('x')
ax_vx.set_ylabel('Velocity')
ax_vx.set_title('Velocity')
ax_vx.legend(loc='upper left')
ax_vx.grid(True, alpha=0.3)

ax_P.set_xlim(x_lim)
ax_P.set_ylim(P_lim)
ax_P.set_xlabel('x')
ax_P.set_ylabel('Pressure')
ax_P.set_title('Pressure')
ax_P.legend(loc='upper right')
ax_P.grid(True, alpha=0.3)

# Internal energy (no analytical for this plot)
scat_uint, = ax_uint.plot([], [], 'g.', markersize=1, alpha=0.5)
ax_uint.set_xlim(x_lim)
ax_uint.set_ylim(0, 3.5)
ax_uint.set_xlabel('x')
ax_uint.set_ylabel('Internal Energy')
ax_uint.set_title('Internal Energy')
ax_uint.grid(True, alpha=0.3)

# Time text
time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

def init():
    """Initialize animation."""
    scat_rho.set_data([], [])
    scat_vx.set_data([], [])
    scat_P.set_data([], [])
    scat_uint.set_data([], [])
    line_rho.set_data([], [])
    line_vx.set_data([], [])
    line_P.set_data([], [])
    time_text.set_text('')
    return scat_rho, scat_vx, scat_P, scat_uint, line_rho, line_vx, line_P, time_text

def animate(frame_idx):
    """Update animation frame."""
    data = frames_data[frame_idx]
    t = frame_times[frame_idx]

    # Update particle data
    scat_rho.set_data(data['x'], data['rho'])
    scat_vx.set_data(data['x'], data['vx'])
    scat_P.set_data(data['x'], data['P'])

    # Compute internal energy for display
    uint_display = data['P'] / ((gamma - 1) * data['rho'])
    scat_uint.set_data(data['x'], uint_display)

    # Update analytical solution
    if t > 0:
        x_ana, rho_ana, vx_ana, P_ana = get_analytical_solution(t)
        line_rho.set_data(x_ana, rho_ana)
        line_vx.set_data(x_ana, vx_ana)
        line_P.set_data(x_ana, P_ana)
    else:
        # Initial condition (step function)
        x_ana = np.array([0, xs, xs, 2*xs])
        line_rho.set_data(x_ana, [rho_L, rho_L, rho_R, rho_R])
        line_vx.set_data(x_ana, [0, 0, 0, 0])
        line_P.set_data(x_ana, [P_L, P_L, P_R, P_R])

    # Update time text
    time_text.set_text(f't = {t:.4f}')

    return scat_rho, scat_vx, scat_P, scat_uint, line_rho, line_vx, line_P, time_text

# Create animation
anim = FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(frames_data),
    interval=100,  # 100ms between frames
    blit=True
)

# Save animation
animation_path = os.path.join(output_dir, animation_file)
print(f"Saving animation to {animation_path}...")
anim.save(animation_path, writer=PillowWriter(fps=10))
print(f"Animation saved!")

# Also save final frame as PNG
final_frame_path = os.path.join(output_dir, "sph_sod_final.png")
animate(len(frames_data) - 1)
fig.savefig(final_frame_path, dpi=150)
print(f"Final frame saved to {final_frame_path}")

# =============================================================================
# Compute L2 Error
# =============================================================================

print()
print("Computing L2 error against analytical solution...")
sodanalysis = model.make_analysis_sodtube(sod, (1, 0, 0), t_target, 0.0, -xs, xs)
L2_error = sodanalysis.compute_L2_dist()
print(f"L2 error at t={t_target}: {L2_error}")

# =============================================================================
# Summary
# =============================================================================

print()
print("=" * 70)
print("Output Summary:")
print("=" * 70)
print(f"  VTK files:     {output_dir}/{vtk_prefix}_*.vtk ({frame_count} files)")
print(f"  Animation:     {animation_path}")
print(f"  Final frame:   {final_frame_path}")
print()
print("To visualize VTK files in ParaView:")
print(f"  paraview {output_dir}/{vtk_prefix}_*.vtk")
print()
print("=" * 70)
print("SPH Sod Shock Tube Simulation Complete!")
print("=" * 70)

plt.close()
