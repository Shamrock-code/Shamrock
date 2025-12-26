#!/usr/bin/env python3
"""
Generate GIF animation from GSPH Sod Shock Tube VTK files.

Usage (from project root):
    python shamrock/exemples/gsph/scripts/gen_gsph_gif.py

Or with make:
    make gsph-sod-animate
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import glob
import os
import sys

import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.animation import FuncAnimation, PillowWriter

# Import from shared analytical module
exemples_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, exemples_dir)
from common.analytical.riemann import SodAnalytical

# Parameters
gamma = 1.4
pmass = 3.371747880871523e-07
hfact = 1.2
t_target = 0.245
n_frames = 50
dt_dump = t_target / n_frames

vtk_dir = "simulations_data/gsph_sod/vtk"
output_dir = "simulations_data/gsph_sod"

# Create analytical solver (interface at x=0)
sod_solver = SodAnalytical(gamma=gamma, x0=0.0)


def read_vtk(filename):
    """Read VTK file using pyvista"""
    mesh = pv.read(filename)
    points = np.array(mesh.points)
    velocities = np.array(mesh["v"])
    hpart = np.array(mesh["h"])
    rho = np.array(mesh["rho"])
    P = np.array(mesh["P"])
    return points, velocities, hpart, rho, P


# Get list of VTK files
vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "gsph_*.vtk")))
print(f"Found {len(vtk_files)} VTK files")

if len(vtk_files) == 0:
    print("No VTK files found!")
    exit(1)

# Set up figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))


def update(frame):
    vtk_file = vtk_files[frame]
    t = frame * dt_dump

    # Read data
    points, velocities, h, rho, P = read_vtk(vtk_file)

    # No shift needed - simulation has interface at x=0, analytical also at x=0
    x = points[:, 0]
    vx = velocities[:, 0]

    # Sort by x
    idx = np.argsort(x)
    x_sort = x[idx]
    rho_sort = rho[idx]
    vx_sort = vx[idx]
    P_sort = P[idx]

    # Analytical solution using exact Riemann solver
    x_ana, rho_ana, vx_ana, P_ana, _ = sod_solver.solution_at_time(
        t, x_min=-1.0, x_max=1.0, n_points=500
    )

    # Clear and redraw
    for ax in axes.flat:
        ax.clear()

    axes[0, 0].plot(x_ana, rho_ana, "r-", lw=2, label="Analytical")
    axes[0, 0].scatter(x_sort, rho_sort, s=1, alpha=0.5, label="GSPH")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Density")
    axes[0, 0].legend()
    axes[0, 0].set_xlim(-1.1, 1.1)
    axes[0, 0].set_ylim(0, 1.2)

    axes[0, 1].plot(x_ana, vx_ana, "r-", lw=2, label="Analytical")
    axes[0, 1].scatter(x_sort, vx_sort, s=1, alpha=0.5, label="GSPH")
    axes[0, 1].set_ylabel("Velocity")
    axes[0, 1].set_title("Velocity")
    axes[0, 1].legend()
    axes[0, 1].set_xlim(-1.1, 1.1)
    axes[0, 1].set_ylim(-0.1, 1.1)

    axes[1, 0].plot(x_ana, P_ana, "r-", lw=2, label="Analytical")
    axes[1, 0].scatter(x_sort, P_sort, s=1, alpha=0.5, label="GSPH")
    axes[1, 0].set_ylabel("Pressure")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_title("Pressure")
    axes[1, 0].legend()
    axes[1, 0].set_xlim(-1.1, 1.1)
    axes[1, 0].set_ylim(0, 1.2)

    axes[1, 1].scatter(x_sort, h[idx], s=1, alpha=0.5)
    axes[1, 1].set_ylabel("h")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_title("Smoothing Length h")
    axes[1, 1].set_xlim(-1.1, 1.1)

    fig.suptitle(f"GSPH Sod Shock Tube - HLLC (t = {t:.3f})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    return axes.flat


# Create animation
print("Creating animation...")
anim = FuncAnimation(fig, update, frames=len(vtk_files), interval=100)

# Save as GIF
gif_path = os.path.join(output_dir, "gsph_sod_animation.gif")
print(f"Saving to {gif_path}...")
anim.save(gif_path, writer=PillowWriter(fps=10))
print(f"Animation saved to {gif_path}")

# Save final frame as PNG
print("Saving final frame...")
update(len(vtk_files) - 1)
final_path = os.path.join(output_dir, "gsph_sod_final.png")
plt.savefig(final_path, dpi=150)
print(f"Final frame saved to {final_path}")
