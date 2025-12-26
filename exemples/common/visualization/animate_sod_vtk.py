#!/usr/bin/env python3
"""
Generate GIF animation from Sod Shock Tube VTK files.

Works with both SPH and GSPH solver outputs.
Uses shamrock.phys.SodTube for analytical solution (no custom Python implementation).

Usage:
    python animate_sod_vtk.py <vtk_dir> [output_dir] [--solver SPH|GSPH]

Examples:
    python animate_sod_vtk.py simulations_data/gsph_sod/vtk --solver GSPH
    python animate_sod_vtk.py simulations_data/sph_sod/vtk --solver SPH
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.animation import FuncAnimation, PillowWriter

# Import shamrock for analytical solution
try:
    import shamrock
    HAS_SHAMROCK = True
except ImportError:
    HAS_SHAMROCK = False
    print("Warning: shamrock module not found. Analytical solution will not be shown.")


def parse_args():
    parser = argparse.ArgumentParser(description="Animate Sod shock tube VTK results")
    parser.add_argument("vtk_dir", help="Directory containing VTK files")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Output directory (defaults to parent of vtk_dir)")
    parser.add_argument("--solver", choices=["SPH", "GSPH"], default="GSPH",
                        help="Solver type (affects file naming)")
    parser.add_argument("--gamma", type=float, default=1.4,
                        help="Adiabatic index (default: 1.4)")
    parser.add_argument("--t-final", type=float, default=0.245,
                        help="Final simulation time (default: 0.245)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Animation frames per second (default: 10)")
    return parser.parse_args()


def get_analytical_solution(sod, t, x_array):
    """Get analytical solution at multiple x positions using shamrock.phys.SodTube."""
    rho = np.zeros(len(x_array))
    vel = np.zeros(len(x_array))
    pres = np.zeros(len(x_array))
    for i, x in enumerate(x_array):
        rho[i], vel[i], pres[i] = sod.get_value(t, x)
    return x_array, rho, vel, pres


def read_vtk(filename):
    """Read VTK file using pyvista."""
    mesh = pv.read(filename)
    points = np.array(mesh.points)
    velocities = np.array(mesh["v"])
    hpart = np.array(mesh["h"])
    rho = np.array(mesh["rho"])
    P = np.array(mesh["P"])
    return points, velocities, hpart, rho, P


def main():
    args = parse_args()

    vtk_dir = args.vtk_dir
    output_dir = args.output_dir or os.path.dirname(vtk_dir)
    solver_name = args.solver
    gamma = args.gamma
    t_final = args.t_final

    # Find VTK files
    vtk_pattern = os.path.join(vtk_dir, "*.vtk")
    vtk_files = sorted(glob.glob(vtk_pattern))

    print(f"{'=' * 70}")
    print(f"Sod Shock Tube Animation ({solver_name})")
    print(f"{'=' * 70}")
    print(f"VTK directory: {vtk_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(vtk_files)} VTK files")
    print()

    if len(vtk_files) == 0:
        print(f"ERROR: No VTK files found in {vtk_dir}")
        sys.exit(1)

    n_frames = len(vtk_files)
    dt_dump = t_final / n_frames

    # Create analytical solver using shamrock.phys.SodTube
    sod_solver = None
    if HAS_SHAMROCK:
        # Standard Sod problem: left state (rho=1, P=1), right state (rho=0.125, P=0.1)
        sod_solver = shamrock.phys.SodTube(
            gamma=gamma,
            rho_1=1.0,    # Left density
            P_1=1.0,      # Left pressure
            rho_5=0.125,  # Right density
            P_5=0.1       # Right pressure
        )
        print(f"Analytical solution: shamrock.phys.SodTube (gamma={gamma})")
    else:
        print("Analytical solution: not available")
    print()

    # Set up figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def update(frame):
        vtk_file = vtk_files[frame]
        t = frame * dt_dump

        # Read data
        points, velocities, h, rho, P = read_vtk(vtk_file)

        x = points[:, 0]
        vx = velocities[:, 0]

        # Sort by x
        idx = np.argsort(x)
        x_sort = x[idx]
        rho_sort = rho[idx]
        vx_sort = vx[idx]
        P_sort = P[idx]
        h_sort = h[idx]

        # Clear and redraw
        for ax in axes.flat:
            ax.clear()

        # Plot analytical solution if available
        if sod_solver is not None and t > 0:
            x_ana = np.linspace(-1.0, 1.0, 500)
            _, rho_ana, vx_ana, P_ana = get_analytical_solution(sod_solver, t, x_ana)

            axes[0, 0].plot(x_ana, rho_ana, "r-", lw=2, label="Analytical")
            axes[0, 1].plot(x_ana, vx_ana, "r-", lw=2, label="Analytical")
            axes[1, 0].plot(x_ana, P_ana, "r-", lw=2, label="Analytical")

        # Density
        axes[0, 0].scatter(x_sort, rho_sort, s=1, alpha=0.5, label=solver_name)
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Density")
        axes[0, 0].legend()
        axes[0, 0].set_xlim(-1.1, 1.1)
        axes[0, 0].set_ylim(0, 1.2)

        # Velocity
        axes[0, 1].scatter(x_sort, vx_sort, s=1, alpha=0.5, label=solver_name)
        axes[0, 1].set_ylabel("Velocity")
        axes[0, 1].set_title("Velocity")
        axes[0, 1].legend()
        axes[0, 1].set_xlim(-1.1, 1.1)
        axes[0, 1].set_ylim(-0.1, 1.1)

        # Pressure
        axes[1, 0].scatter(x_sort, P_sort, s=1, alpha=0.5, label=solver_name)
        axes[1, 0].set_ylabel("Pressure")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_title("Pressure")
        axes[1, 0].legend()
        axes[1, 0].set_xlim(-1.1, 1.1)
        axes[1, 0].set_ylim(0, 1.2)

        # Smoothing length
        axes[1, 1].scatter(x_sort, h_sort, s=1, alpha=0.5)
        axes[1, 1].set_ylabel("h")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_title("Smoothing Length h")
        axes[1, 1].set_xlim(-1.1, 1.1)

        fig.suptitle(f"{solver_name} Sod Shock Tube (t = {t:.3f})", fontsize=14, fontweight="bold")
        plt.tight_layout()

        return axes.flat

    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update, frames=len(vtk_files), interval=100)

    # Save as GIF
    solver_lower = solver_name.lower()
    gif_path = os.path.join(output_dir, f"{solver_lower}_sod_animation.gif")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {gif_path}...")
    anim.save(gif_path, writer=PillowWriter(fps=args.fps))
    print(f"Animation saved to {gif_path}")

    # Save final frame as PNG
    print("Saving final frame...")
    update(len(vtk_files) - 1)
    final_path = os.path.join(output_dir, f"{solver_lower}_sod_final.png")
    plt.savefig(final_path, dpi=150)
    print(f"Final frame saved to {final_path}")

    print()
    print(f"{'=' * 70}")
    print("Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
